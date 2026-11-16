"""Hash-bound, vectorized reconstruction of a submitted runtime actor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from deepmew.meta_kg.io import stable_deck_hash
from deepmew.submission_postmortem.models import ReplayDecision, ReplayEpisode


BINDING_SCHEMA_VERSION = "deepmew.submission_postmortem.binding.v0"
SCORE_SCHEMA_VERSION = "deepmew.submission_postmortem.actor_scores.v0"
DEFAULT_BINDINGS_PATH = Path("configs/submission_postmortem/submission_bindings.json")


class SubmissionBindingError(ValueError):
    """Raised when an exact submitted runtime cannot be proven."""


@dataclass(frozen=True)
class RuntimeBinding:
    """Verified local runtime artifacts for one external submission."""

    external_submission_id: str
    bundle_dir: Path
    submission_manifest: Mapping[str, Any]
    weights_manifest: Mapping[str, Any]
    weights_path: Path
    runtime_path: Path
    reader_path: Path
    deck_path: Path
    weights_sha256: str
    runtime_sha256: str
    reader_sha256: str
    deck_file_sha256: str
    submitted_deck_hash: str
    binding_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BINDING_SCHEMA_VERSION,
            "external_submission_id": self.external_submission_id,
            "bundle_dir": str(self.bundle_dir),
            "weights_path": str(self.weights_path),
            "runtime_path": str(self.runtime_path),
            "reader_path": str(self.reader_path),
            "deck_path": str(self.deck_path),
            "weights_sha256": self.weights_sha256,
            "runtime_sha256": self.runtime_sha256,
            "reader_sha256": self.reader_sha256,
            "deck_file_sha256": self.deck_file_sha256,
            "submitted_deck_hash": self.submitted_deck_hash,
            "binding_sha256": self.binding_sha256,
            "policy_id": self.weights_manifest.get("policy_id"),
            "feature_deck_hash": self.weights_manifest.get("feature_deck_hash"),
            "feature_mode": self.weights_manifest.get("feature_mode"),
            "input_dim": self.weights_manifest.get("input_dim"),
            "source_checkpoint": self.submission_manifest.get("source_checkpoint"),
            "source_checkpoint_sha256": self.submission_manifest.get("source_checkpoint_sha256"),
            "actor_rating_status": "exact_runtime_actor",
            "value_rating_status": "not_instrumented",
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _expected_hash(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    return value.strip().removeprefix("sha256:").lower()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionBindingError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SubmissionBindingError(f"expected JSON object: {path}")

    return payload


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    if not path.is_file():
        raise SubmissionBindingError(f"{label} missing: {path}")
    actual = sha256_file(path)
    normalized = _expected_hash(expected)
    if normalized is None:
        raise SubmissionBindingError(f"{label} has no expected SHA-256 binding")
    if actual != normalized:
        raise SubmissionBindingError(f"{label} SHA-256 mismatch: expected {normalized}, got {actual}")

    return actual


def _read_deck_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw_line.strip()
        if not value:
            continue
        try:
            card_id = int(value)
        except ValueError as exc:
            raise SubmissionBindingError(f"{path}:{line_number}: invalid card id {value!r}") from exc
        counts[card_id] += 1
    if sum(counts.values()) != 60:
        raise SubmissionBindingError(f"submitted deck must contain 60 cards, got {sum(counts.values())}")

    return counts


def load_binding_registry(path: str | Path = DEFAULT_BINDINGS_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.is_file():
        return {"schema_version": BINDING_SCHEMA_VERSION, "bindings": {}}
    payload = _read_json_object(registry_path)
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise SubmissionBindingError(f"binding registry has no object-valued bindings: {registry_path}")

    return payload


def resolve_runtime_binding(
    external_submission_id: str | int,
    *,
    bundle_dir: str | Path | None = None,
    registry_path: str | Path = DEFAULT_BINDINGS_PATH,
    workspace_root: str | Path = ".",
) -> RuntimeBinding:
    """Resolve only explicit, hash-bound submission mappings; never guess by deck."""

    external_id = str(external_submission_id).strip()
    if not external_id:
        raise SubmissionBindingError("external submission id is required")
    registry_entry: Mapping[str, Any] = {}
    if bundle_dir is None:
        registry = load_binding_registry(registry_path)
        candidate = registry.get("bindings", {}).get(external_id)
        if not isinstance(candidate, dict):
            raise SubmissionBindingError(
                f"unresolved submission binding for {external_id}; pass --bundle-dir or add an explicit hash-bound registry entry"
            )
        registry_entry = candidate
        bundle_dir = candidate.get("bundle_dir")
    if not isinstance(bundle_dir, (str, Path)) or not str(bundle_dir).strip():
        raise SubmissionBindingError(f"submission {external_id} binding has no bundle_dir")

    root = Path(workspace_root).resolve()
    bundle = Path(bundle_dir)
    if not bundle.is_absolute():
        bundle = root / bundle
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise SubmissionBindingError(f"submission bundle not found: {bundle}")

    submission_manifest_path = bundle / "submission_manifest.json"
    weights_manifest_path = bundle / "r5_policy_weights.manifest.json"
    submission_manifest = _read_json_object(submission_manifest_path)
    weights_manifest = _read_json_object(weights_manifest_path)
    weights_path = bundle / str(submission_manifest.get("weights") or "r5_policy_weights.npz")
    runtime_path = bundle / str(submission_manifest.get("policy_runtime") or "r5_distilled_policy.py")
    reader_path = bundle / "second_order_reader.py"
    deck_path = bundle / str(submission_manifest.get("deck") or "deck.csv")

    weights_sha = _verify_hash(weights_path, submission_manifest.get("weights_sha256"), label="runtime weights")
    deck_sha = _verify_hash(deck_path, submission_manifest.get("deck_file_sha256"), label="submitted deck")
    if not runtime_path.is_file() or not reader_path.is_file():
        raise SubmissionBindingError("bundle runtime implementation is incomplete")
    runtime_sha = sha256_file(runtime_path)
    reader_sha = sha256_file(reader_path)

    registry_expected = _expected_hash(registry_entry.get("weights_sha256"))
    if registry_expected is not None and registry_expected != weights_sha:
        raise SubmissionBindingError(
            f"registry weights SHA-256 mismatch: expected {registry_expected}, bundle has {weights_sha}"
        )
    manifest_feature_hash = str(weights_manifest.get("feature_deck_hash") or "")
    if manifest_feature_hash != str(submission_manifest.get("deck_normalized_sha256") or ""):
        raise SubmissionBindingError("runtime feature-deck hash does not match the submitted bundle binding")
    input_dim = weights_manifest.get("input_dim")
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise SubmissionBindingError("runtime weights manifest has no positive input_dim")
    if str(weights_manifest.get("feature_mode") or "") not in {"base", "second_order"}:
        raise SubmissionBindingError("unsupported runtime feature_mode")

    submitted_hash = stable_deck_hash(dict(_read_deck_counts(deck_path)))
    binding_payload = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "external_submission_id": external_id,
        "weights_sha256": weights_sha,
        "runtime_sha256": runtime_sha,
        "reader_sha256": reader_sha,
        "deck_file_sha256": deck_sha,
        "submitted_deck_hash": submitted_hash,
        "policy_id": weights_manifest.get("policy_id"),
        "feature_deck_hash": weights_manifest.get("feature_deck_hash"),
        "feature_mode": weights_manifest.get("feature_mode"),
        "input_dim": input_dim,
    }
    binding_sha = _sha256_bytes(json.dumps(binding_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    return RuntimeBinding(
        external_submission_id=external_id,
        bundle_dir=bundle,
        submission_manifest=submission_manifest,
        weights_manifest=weights_manifest,
        weights_path=weights_path,
        runtime_path=runtime_path,
        reader_path=reader_path,
        deck_path=deck_path,
        weights_sha256=weights_sha,
        runtime_sha256=runtime_sha,
        reader_sha256=reader_sha,
        deck_file_sha256=deck_sha,
        submitted_deck_hash=submitted_hash,
        binding_sha256=binding_sha,
    )


def _load_runtime_module(binding: RuntimeBinding) -> ModuleType:
    """Load the bundle's own runtime dependencies without polluting global imports."""

    module_name = f"_deepmew_submission_runtime_{binding.binding_sha256[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    dependency_names = [name for name in sys.modules if name == "cg" or name.startswith("cg.")]
    if "second_order_reader" in sys.modules:
        dependency_names.append("second_order_reader")
    saved_modules = {name: sys.modules[name] for name in dependency_names}
    for name in dependency_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(binding.bundle_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, binding.runtime_path)
        if spec is None or spec.loader is None:
            raise SubmissionBindingError(f"cannot load runtime module: {binding.runtime_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded_cg = sys.modules.get("cg")
        cg_path = Path(str(getattr(loaded_cg, "__file__", ""))).resolve() if loaded_cg is not None else None
        if cg_path is None or binding.bundle_dir not in cg_path.parents:
            raise SubmissionBindingError("runtime did not bind to the bundle-local cg package")
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "cg" or name.startswith("cg.") or name == "second_order_reader":
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)

    return module


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _softmax_masked(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(scores, dtype=np.float64)
    valid = mask & np.isfinite(scores) & (scores > -1.0e8)
    if not valid.any():
        return probabilities
    shifted = scores[valid].astype(np.float64) - float(np.max(scores[valid]))
    weights = np.exp(shifted)
    probabilities[valid] = weights / float(np.sum(weights))

    return probabilities


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if not len(positive):
        return 0.0

    return float(-np.sum(positive * np.log(positive)))


class ExactRuntimeActorScorer:
    """Batch-encode legal options once, then replay autoregressive heads."""

    def __init__(self, binding: RuntimeBinding, *, encoder_batch_size: int = 8192):
        if encoder_batch_size <= 0:
            raise ValueError("encoder_batch_size must be positive")
        self.binding = binding
        self.runtime = _load_runtime_module(binding)
        self.policy = self.runtime.DistilledPolicy(
            str(binding.weights_path),
            str(binding.bundle_dir / "r5_policy_weights.manifest.json"),
        )
        self.encoder_batch_size = int(encoder_batch_size)
        self.input_dim = int(binding.weights_manifest["input_dim"])

    def _build_decision_features(self, decision: ReplayDecision) -> tuple[list[dict[str, Any]], np.ndarray, int, int, list[Any]]:
        observation = dict(decision.observation)
        select = observation.get("select")
        if not isinstance(select, dict):
            raise ValueError(f"{decision.decision_id}: actor-visible observation has no select object")
        legal_table = self.runtime.legal_action_table(observation, select)
        features = self.runtime.build_live_features(
            observation,
            legal_table,
            policy_id=self.policy.policy_id,
            deck_hash=self.policy.deck_hash,
            feature_mode=str(self.policy.manifest.get("feature_mode") or "base"),
        )
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(legal_table), self.input_dim):
            raise ValueError(
                f"{decision.decision_id}: feature shape {matrix.shape} does not match {(len(legal_table), self.input_dim)}"
            )
        min_count, max_count = self.runtime.selection_bounds(select, len(legal_table))
        keys = self.runtime.canonical_keys(legal_table, len(legal_table))

        return legal_table, matrix, int(min_count), int(max_count), keys

    def _encode_flat(self, matrices: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not matrices:
            return []
        lengths = [len(matrix) for matrix in matrices]
        flat = np.concatenate(matrices, axis=0)
        encoded_chunks = []
        for start in range(0, len(flat), self.encoder_batch_size):
            encoded_chunks.append(self.policy._encoder(flat[start : start + self.encoder_batch_size]))
        encoded = np.concatenate(encoded_chunks, axis=0)
        result = []
        offset = 0
        for length in lengths:
            result.append(encoded[offset : offset + length])
            offset += length

        return result

    def _scores_from_encoded(
        self,
        encoded: np.ndarray,
        selected: Sequence[int],
        min_count: int,
        max_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        legal_count = len(encoded)
        selected_set = set(selected)
        option_meta = np.zeros((legal_count, 5), dtype=np.float32)
        for option_index in range(legal_count):
            option_meta[option_index] = np.asarray(
                [
                    1.0 if option_index in selected_set else 0.0,
                    min(len(selected), 8) / 8.0,
                    min(len(selected_set), 8) / 8.0,
                    max(0, min_count - len(selected_set)) / max(1, legal_count),
                    max(0, max_count - len(selected_set)) / max(1, legal_count),
                ],
                dtype=np.float32,
            )
        option_scores = self.policy._head(np.concatenate([encoded, option_meta], axis=1), "option_head")
        pooled = encoded.mean(axis=0, keepdims=True)
        step_meta = option_meta[:1] if len(option_meta) else np.zeros((1, 5), dtype=np.float32)
        stop_score = self.policy._head(np.concatenate([pooled, step_meta], axis=1), "stop_head")
        scores = np.concatenate([option_scores, stop_score])
        mask = np.zeros(legal_count + 1, dtype=bool)
        for option_index in range(legal_count):
            mask[option_index] = option_index not in selected_set
        if len(selected) >= min_count:
            mask[legal_count] = True
        if len(selected) >= max_count:
            mask[:legal_count] = False
            mask[legal_count] = True

        return np.where(mask, scores, -1.0e9), mask

    def _model_action(
        self,
        encoded: np.ndarray,
        min_count: int,
        max_count: int,
        keys: list[Any],
    ) -> list[int]:
        legal_count = len(encoded)
        selected: list[int] = []
        for _ in range(max_count + 1):
            scores, mask = self._scores_from_encoded(encoded, selected, min_count, max_count)
            choice = int(self.runtime._canonical_choice(scores, mask, keys))
            if choice >= legal_count:
                break
            if choice not in selected:
                selected.append(choice)
            if len(selected) >= max_count:
                break

        return selected

    def _prefix_rows(
        self,
        decision: ReplayDecision,
        legal_table: list[dict[str, Any]],
        encoded: np.ndarray,
        min_count: int,
        max_count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        legal_count = len(legal_table)
        observed_tokens = list(decision.action)
        if len(observed_tokens) < max_count:
            observed_tokens.append(legal_count)
        prefixes: list[dict[str, Any]] = []
        option_rows: list[dict[str, Any]] = []
        sequence_log_probability = 0.0
        selected: list[int] = []
        for prefix_index, chosen_token in enumerate(observed_tokens):
            scores, mask = self._scores_from_encoded(encoded, selected, min_count, max_count)
            probabilities = _softmax_masked(scores, mask)
            valid_indices = [int(index) for index in np.flatnonzero(mask)]
            ordered = sorted(valid_indices, key=lambda index: (-float(scores[index]), index))
            rank = ordered.index(chosen_token) + 1 if chosen_token in ordered else None
            chosen_probability = float(probabilities[chosen_token]) if 0 <= chosen_token < len(probabilities) else 0.0
            if chosen_probability > 0.0:
                sequence_log_probability += math.log(chosen_probability)
            else:
                sequence_log_probability = float("-inf")
            top_index = ordered[0] if ordered else None
            top_score = float(scores[top_index]) if top_index is not None else None
            chosen_score = float(scores[chosen_token]) if 0 <= chosen_token < len(scores) else None
            prefixes.append(
                {
                    "prefix_index": prefix_index,
                    "selected_prefix": list(selected),
                    "chosen_index": chosen_token,
                    "chosen_is_stop": chosen_token == legal_count,
                    "chosen_score": chosen_score,
                    "chosen_probability": chosen_probability,
                    "chosen_rank": rank,
                    "top_index": top_index,
                    "top_score": top_score,
                    "chosen_to_top_margin": None if chosen_score is None or top_score is None else chosen_score - top_score,
                    "entropy": _entropy(probabilities),
                    "valid_count": len(valid_indices),
                }
            )
            for option_index in valid_indices:
                semantic = {"semantic_key": "STOP", "option_type": "STOP", "index": legal_count}
                if option_index < legal_count:
                    semantic = dict(legal_table[option_index])
                option_rows.append(
                    {
                        "decision_id": decision.decision_id,
                        "episode_id": decision.episode_id,
                        "seat": decision.seat,
                        "prefix_index": prefix_index,
                        "option_index": option_index,
                        "is_stop": option_index == legal_count,
                        "is_chosen": option_index == chosen_token,
                        "score": float(scores[option_index]),
                        "probability": float(probabilities[option_index]),
                        "rank": ordered.index(option_index) + 1,
                        "semantic": semantic,
                        "provenance": "policy_rescore",
                    }
                )
            if chosen_token < legal_count:
                selected.append(chosen_token)

        return prefixes, option_rows, sequence_log_probability

    def score_decisions(self, decisions: Sequence[ReplayDecision]) -> list[dict[str, Any]]:
        prepared = [self._build_decision_features(decision) for decision in decisions]
        encoded = self._encode_flat([item[1] for item in prepared])
        results: list[dict[str, Any]] = []
        for decision, item, embeddings in zip(decisions, prepared, encoded):
            legal_table, _matrix, min_count, max_count, keys = item
            model_action = self._model_action(embeddings, min_count, max_count, keys)
            prefixes, option_rows, sequence_log_probability = self._prefix_rows(
                decision,
                legal_table,
                embeddings,
                min_count,
                max_count,
            )
            selected_semantics = [dict(legal_table[index]) for index in decision.action]
            agreement = model_action == list(decision.action)
            results.append(
                {
                    "score_schema_version": SCORE_SCHEMA_VERSION,
                    "decision_id": decision.decision_id,
                    "episode_id": decision.episode_id,
                    "seat": decision.seat,
                    "request_step": decision.request_step,
                    "action_step": decision.action_step,
                    "replay_action": list(decision.action),
                    "model_action": model_action,
                    "model_action_agreement": agreement,
                    "actor_rating_status": "exact_runtime_actor" if agreement else "model_action_mismatch",
                    "value_rating_status": "not_instrumented",
                    "value": None,
                    "sequence_log_probability": _json_safe_scalar(sequence_log_probability),
                    "min_count": min_count,
                    "max_count": max_count,
                    "legal_count": len(legal_table),
                    "selected_semantics": selected_semantics,
                    "prefix_ratings": prefixes,
                    "option_ratings": option_rows,
                    "binding_sha256": self.binding.binding_sha256,
                    "provenance": "policy_rescore",
                }
            )

        return results


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _score_cache_path(cache_root: Path, episode: ReplayEpisode, binding: RuntimeBinding) -> Path:
    return cache_root / "actor_scores" / episode.source_sha256[:2] / f"{episode.source_sha256}-{binding.binding_sha256}.json"


def score_episodes_cached(
    episodes: Sequence[ReplayEpisode],
    scorer: ExactRuntimeActorScorer,
    *,
    cache_root: str | Path,
    refresh: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Reuse complete per-episode score shards and batch all cache misses."""

    cache = Path(cache_root)
    results: dict[str, list[dict[str, Any]]] = {}
    misses: list[ReplayEpisode] = []
    stats = {"hit_episodes": 0, "miss_episodes": 0, "scored_decisions": 0}
    for episode in episodes:
        path = _score_cache_path(cache, episode, scorer.binding)
        if path.is_file() and not refresh:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            expected_ids = {decision.decision_id for decision in episode.decisions_for_actor()}
            if (
                isinstance(payload, dict)
                and payload.get("source_sha256") == episode.source_sha256
                and payload.get("binding_sha256") == scorer.binding.binding_sha256
                and isinstance(payload.get("scores"), list)
                and {row.get("decision_id") for row in payload["scores"]} == expected_ids
            ):
                results[episode.episode_id] = payload["scores"]
                stats["hit_episodes"] += 1
                continue
        misses.append(episode)

    miss_decisions = [decision for episode in misses for decision in episode.decisions_for_actor()]
    scored = scorer.score_decisions(miss_decisions)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_episode[str(row["episode_id"])].append(row)
    for episode in misses:
        episode_scores = by_episode.get(episode.episode_id, [])
        results[episode.episode_id] = episode_scores
        path = _score_cache_path(cache, episode, scorer.binding)
        _atomic_write_json(
            path,
            {
                "score_schema_version": SCORE_SCHEMA_VERSION,
                "source_sha256": episode.source_sha256,
                "binding_sha256": scorer.binding.binding_sha256,
                "scores": episode_scores,
            },
        )
        stats["miss_episodes"] += 1
        stats["scored_decisions"] += len(episode_scores)

    return results, stats
