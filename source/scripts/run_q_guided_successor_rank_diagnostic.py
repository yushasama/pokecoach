"""Frozen V0-vs-M4 ranking diagnostic on immediate real-engine successors.

The parent launches one isolated worker. Simulator/search observations and model
features exist only in that worker. The worker returns a fixed-schema receipt
containing action IDs, scalar endpoint scores, rank counts, and safety facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for candidate in (ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_privileged_teacher_smoke as teacher_smoke  # noqa: E402
import run_unified_r3_value as value_source  # noqa: E402
from deepmew.q_guided.branching import (  # noqa: E402
    LegalSelectionCapExceeded,
    assert_private_content_absent,
    enumerate_legal_selections,
)
from deepmew.q_guided.simulator_teacher import (  # noqa: E402
    PrivilegedSimulatorTeacher,
    SimulatorTeacherSafetyError,
    _actor,
    _result,
    _terminal_payoff,
)
from deepmew.value_lab.calibration import apply_calibrator  # noqa: E402
from deepmew.value_lab.frozen_v0_endpoint import (  # noqa: E402
    FrozenV0Endpoint,
    FrozenV0Spec,
)
from deepmew.value_lab.models import SnapshotMLP  # noqa: E402


SCHEMA_VERSION = "deepmew.q_guided.successor_rank_diagnostic.v1"
FAILURE_SCHEMA_VERSION = "deepmew.q_guided.successor_rank_diagnostic_failure.v1"
SEED_MANIFEST_SCHEMA_VERSION = "deepmew.q_guided.successor_rank_seed_manifest.v1"
M4_CHECKPOINT_SCHEMA_VERSION = "deepmew.q_guided.training_checkpoint.v1"
M4_CALIBRATOR_SCHEMA_VERSION = "deepmew.q_guided.specialist_v_calibrator.v1"
Q_CLASSIFICATION = "quarantine"
TIE_TOLERANCE = 1e-9
ENDPOINT_SCORE_COUNT = 2
DEFAULT_WORKER_TIMEOUT_SECONDS = 900

DEFAULT_SEED_MANIFEST = (
    ROOT
    / ".research/q_guided_starmie/modal_first_wave_20260712"
    / "successor_rank_diagnostic_seed_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / ".research/q_guided_starmie/modal_first_wave_20260712"
    / "successor_rank_diagnostic_receipt.json"
)
DEFAULT_V0_CHECKPOINT = Path(
    r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_runs"
    r"\phasef4_grouped_crossfit_raw_s1_s915002_v3\full_train\value.pt"
)
DEFAULT_V0_CALIBRATOR_DIR = (
    ROOT / ".research/value_q_experiment/artifacts/phasef5_grouped_calibration_raw_s1_s915002_v3"
)
DEFAULT_M4_CHECKPOINT = Path(
    r"D:\DeepMew_artifacts\checkpoints\q-guided-starmie-m4-final-training-be739f27.pt"
)
DEFAULT_M4_CALIBRATOR = Path(
    r"D:\DeepMew_artifacts\checkpoints\q-guided-starmie-m4-calibrator-0cbc9ed9.json"
)

_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "status",
    "frozen_at",
    "contract_sha256",
    "contract",
})
_WORKER_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "status",
    "seed_manifest_sha256",
    "endpoint_hashes",
    "roots",
    "aggregate_counts",
    "safety_counts",
    "boolean_checks",
})
_ROOT_FIELDS = frozenset({
    "scratch_seed",
    "action_count",
    "actions",
    "pairwise_agreement_count",
    "pairwise_reversal_count",
    "pairwise_tie_count",
    "pairwise_comparable_count",
    "pairwise_agreement_rate",
    "v0_top_action_ids",
    "m4_top_action_ids",
    "top_action_agreement",
})
_ACTION_FIELDS = frozenset({"action_id", "v0_score", "m4_score"})
_SAFETY_FIELDS = frozenset({"illegal_actions", "crashes", "timeouts"})
_FORBIDDEN_PERSISTED_KEYS = frozenset({
    "observation",
    "observations",
    "actor_visible_root_observation",
    "actor_visible_successor",
    "hidden_identity",
    "hidden_identities",
    "private_hash",
    "private_hashes",
    "teacher_label",
    "teacher_labels",
    "q_label",
    "q_labels",
    "search_begin_input",
    "search_token",
    "search_tokens",
    "search_id",
    "search_ids",
    "trajectory_digest",
    "trajectory_digests",
})


class DiagnosticSafetyError(RuntimeError):
    """Fail-closed diagnostic error whose message contains no engine payload."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise DiagnosticSafetyError(f"{name} is not a SHA-256")

    return value.lower()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(_canonical_bytes(dict(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise DiagnosticSafetyError(f"{name} is unreadable") from None
    if not isinstance(value, dict):
        raise DiagnosticSafetyError(f"{name} must contain one object")

    return value


def _validate_seed_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, "seed manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise DiagnosticSafetyError("seed manifest field inventory drifted")
    if (
        manifest["schema_version"] != SEED_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "frozen_pre_execution"
    ):
        raise DiagnosticSafetyError("seed manifest is not frozen")
    contract = manifest["contract"]
    if not isinstance(contract, dict):
        raise DiagnosticSafetyError("seed manifest contract is invalid")
    declared = _validate_sha256(manifest["contract_sha256"], "seed contract hash")
    if _sha256_bytes(_canonical_bytes(contract)) != declared:
        raise DiagnosticSafetyError("seed manifest contract hash mismatch")
    seeds = contract.get("scratch_seeds")
    panel = contract.get("frozen_eval_panel")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
        or contract.get("root_count") != len(seeds)
        or contract.get("one_root_per_seed") is not True
        or contract.get("replacement_seeds_allowed") is not False
        or contract.get("result_conditioned_retries_allowed") is not False
        or contract.get("source_games_per_seed") != 1
    ):
        raise DiagnosticSafetyError("seed contract does not freeze one unreplaced root per seed")
    if (
        not isinstance(panel, list)
        or set(seeds).intersection(panel)
        or contract.get("seed_overlap_with_frozen_eval_panel") is not False
    ):
        raise DiagnosticSafetyError("scratch seeds overlap the frozen evaluation panel")
    pairwise = contract.get("pairwise_primary")
    top = contract.get("top_action_secondary")
    if (
        not isinstance(pairwise, dict)
        or pairwise.get("absolute_tie_tolerance") != TIE_TOLERANCE
        or pairwise.get("comparison") != "every_unordered_action_pair"
        or pairwise.get("agreement") != "same_strict_order"
        or pairwise.get("reversal") != "opposite_strict_order"
        or not isinstance(top, dict)
        or top.get("agreement") != "nonempty_v0_m4_top_set_intersection"
    ):
        raise DiagnosticSafetyError("ranking contract drifted")
    if (
        contract.get("execution_mode") != "isolated_worker_persistent_reconstruct"
        or contract.get("successor_horizon") != "exactly_one_forced_legal_root_action"
        or contract.get("score_perspective") != "original_root_actor"
    ):
        raise DiagnosticSafetyError("successor execution contract drifted")

    return manifest


def _verify_bound_file(path: Path, expected_sha256: Any, name: str) -> str:
    expected = _validate_sha256(expected_sha256, f"{name} hash")
    if not Path(path).is_file() or _sha256_file(path) != expected:
        raise DiagnosticSafetyError(f"{name} hash mismatch")

    return expected


def _load_v0(contract: Mapping[str, Any]) -> FrozenV0Endpoint:
    identity = contract.get("v0")
    if not isinstance(identity, Mapping):
        raise DiagnosticSafetyError("V0 identity is missing")
    result = _read_json_object(DEFAULT_V0_CALIBRATOR_DIR / "result.json", "V0 calibrator result")
    receipt = result.get("final_calibrator")
    if not isinstance(receipt, Mapping):
        raise DiagnosticSafetyError("V0 calibrator receipt is missing")
    spec = FrozenV0Spec(
        checkpoint_sha256=_validate_sha256(identity.get("checkpoint_sha256"), "V0 checkpoint hash"),
        calibrator_bundle_sha256=_validate_sha256(
            identity.get("calibrator_bundle_sha256"),
            "V0 calibrator hash",
        ),
    )
    try:
        return FrozenV0Endpoint.load(
            DEFAULT_V0_CHECKPOINT,
            DEFAULT_V0_CALIBRATOR_DIR,
            receipt,
            spec,
        )
    except Exception:
        raise DiagnosticSafetyError("V0 endpoint identity gate failed") from None


@dataclass(frozen=True)
class FrozenM4Endpoint:
    model: SnapshotMLP
    calibrator: Mapping[str, Any]

    @classmethod
    def load(cls, contract: Mapping[str, Any]) -> "FrozenM4Endpoint":
        identity = contract.get("m4")
        if not isinstance(identity, Mapping):
            raise DiagnosticSafetyError("M4 identity is missing")
        _verify_bound_file(
            DEFAULT_M4_CHECKPOINT,
            identity.get("checkpoint_sha256"),
            "M4 checkpoint",
        )
        _verify_bound_file(
            DEFAULT_M4_CALIBRATOR,
            identity.get("calibrator_sha256"),
            "M4 calibrator",
        )
        try:
            checkpoint = torch.load(
                DEFAULT_M4_CHECKPOINT,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            raise DiagnosticSafetyError("M4 checkpoint could not be decoded") from None
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version") != M4_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("classification") != Q_CLASSIFICATION
            or not isinstance(checkpoint.get("model_state"), Mapping)
        ):
            raise DiagnosticSafetyError("M4 checkpoint schema or classification drifted")
        model = SnapshotMLP(value_source.FEATURE_DIM_V2, (256, 128), 0.0)
        try:
            model.load_state_dict(checkpoint["model_state"], strict=True)
        except Exception:
            raise DiagnosticSafetyError("M4 model state is incompatible") from None
        if any(not bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
            raise DiagnosticSafetyError("M4 model state is non-finite")
        lineage = _read_json_object(DEFAULT_M4_CALIBRATOR, "M4 calibrator")
        calibrator = lineage.get("calibrator")
        source_hashes = checkpoint.get("source_sha256s")
        if (
            lineage.get("schema_version") != M4_CALIBRATOR_SCHEMA_VERSION
            or lineage.get("classification") != Q_CLASSIFICATION
            or lineage.get("dataset_sha256") != checkpoint.get("dataset_sha256")
            or lineage.get("config_sha256") != checkpoint.get("config_sha256")
            or not isinstance(source_hashes, (list, tuple))
            or lineage.get("source_policy_sha256") not in source_hashes
            or lineage.get("source_v_sha256") not in source_hashes
            or not isinstance(calibrator, dict)
        ):
            raise DiagnosticSafetyError("M4 calibrator lineage mismatch")
        try:
            probe = apply_calibrator(
                np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                calibrator,
            )
        except Exception:
            raise DiagnosticSafetyError("M4 calibrator is invalid") from None
        if not np.isfinite(probe).all() or np.any((probe < 0.0) | (probe > 1.0)):
            raise DiagnosticSafetyError("M4 calibrator probe is invalid")

        return cls(model=model.eval(), calibrator=dict(calibrator))

    def score(
        self,
        features: np.ndarray,
        *,
        terminal_outcome: float | None = None,
    ) -> float:
        if not isinstance(features, np.ndarray) or features.dtype != np.float32:
            raise TypeError("M4 features must be float32")
        if features.shape != (value_source.FEATURE_DIM_V2,) or not np.isfinite(features).all():
            raise ValueError("M4 features must be one finite S1 row")
        if terminal_outcome is not None:
            if (
                isinstance(terminal_outcome, bool)
                or not isinstance(terminal_outcome, (int, float))
                or float(terminal_outcome) not in (0.0, 0.5, 1.0)
            ):
                raise ValueError("M4 terminal outcome is invalid")

            return float(terminal_outcome)
        tensor = torch.from_numpy(np.ascontiguousarray(features))[None, None, :]
        mask = torch.ones((1, 1), dtype=torch.bool)
        with torch.inference_mode():
            logits = self.model(tensor, mask)[:, 0]
            raw = torch.sigmoid(logits).cpu().numpy().astype(np.float64, copy=False)
        calibrated = apply_calibrator(raw, dict(self.calibrator))
        score = float(calibrated[0])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise FloatingPointError("M4 produced an invalid probability")

        return score


class SuccessorScorer:
    """Score an actor-visible successor and normalize to the root actor."""

    def __init__(self, v0: FrozenV0Endpoint, m4: FrozenM4Endpoint) -> None:
        self._v0 = v0
        self._m4 = m4

    def __call__(self, successor: dict[str, Any], root_actor: int) -> tuple[float, float]:
        successor_actor = _actor(successor)
        result = _result(successor)
        features = np.asarray(
            value_source._features(
                {"raw_observation": successor},
                value_source.FEATURE_CONFIG_V2,
            ),
            dtype=np.float32,
        )
        terminal_outcome = (
            None if result == -1 else _terminal_payoff(result, successor_actor)
        )
        v0_actor = float(
            self._v0.score(features, terminal_outcome=terminal_outcome)
        )
        m4_actor = self._m4.score(features, terminal_outcome=terminal_outcome)
        if successor_actor == root_actor:
            scores = v0_actor, m4_actor
        else:
            scores = 1.0 - v0_actor, 1.0 - m4_actor
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
            raise DiagnosticSafetyError("endpoint produced an invalid root-relative score")

        return scores


def _top_action_ids(
    actions: Sequence[Mapping[str, Any]],
    score_name: str,
    tolerance: float,
) -> list[str]:
    maximum = max(float(action[score_name]) for action in actions)

    return [
        str(action["action_id"])
        for action in actions
        if maximum - float(action[score_name]) <= tolerance
    ]


def _rank_summary(
    actions: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = TIE_TOLERANCE,
) -> dict[str, Any]:
    if len(actions) < 2:
        raise DiagnosticSafetyError("rank summary requires at least two actions")
    if len({str(action.get("action_id")) for action in actions}) != len(actions):
        raise DiagnosticSafetyError("rank summary action IDs are not unique")
    agreements = 0
    reversals = 0
    ties = 0
    for first, second in combinations(actions, 2):
        v0_delta = float(first["v0_score"]) - float(second["v0_score"])
        m4_delta = float(first["m4_score"]) - float(second["m4_score"])
        if abs(v0_delta) <= tolerance or abs(m4_delta) <= tolerance:
            ties += 1
        elif math.copysign(1.0, v0_delta) == math.copysign(1.0, m4_delta):
            agreements += 1
        else:
            reversals += 1
    comparable = agreements + reversals
    v0_top = _top_action_ids(actions, "v0_score", tolerance)
    m4_top = _top_action_ids(actions, "m4_score", tolerance)

    return {
        "pairwise_agreement_count": agreements,
        "pairwise_reversal_count": reversals,
        "pairwise_tie_count": ties,
        "pairwise_comparable_count": comparable,
        "pairwise_agreement_rate": agreements / comparable if comparable else None,
        "v0_top_action_ids": v0_top,
        "m4_top_action_ids": m4_top,
        "top_action_agreement": bool(set(v0_top).intersection(m4_top)),
    }


def _score_root(
    observation: dict[str, Any],
    selections: list[list[int]],
    scorer: SuccessorScorer,
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, int]]:
    factory = teacher_smoke._LiveRootSearchFactory(observation)
    teacher = PrivilegedSimulatorTeacher(
        observation,
        begin_search=factory.begin,
        step_search=factory.step,
        end_search=factory.end,
        verify_live_root_unchanged=factory.verify_unchanged,
    )
    actions: list[dict[str, Any]] = []
    try:
        with teacher:
            for selection in selections:
                v0_score, m4_score = teacher.measure_immediate_successor(
                    selection,
                    scalar_scorer=scorer,
                    expected_score_count=ENDPOINT_SCORE_COUNT,
                )
                actions.append({
                    "action_id": teacher_smoke._action_id(selection),
                    "v0_score": v0_score,
                    "m4_score": m4_score,
                })
        receipt = teacher.receipt()
        checks = {
            "actor_visible_input_stable": receipt.actor_visible_input_stable,
            "reconstructed_root_matches": receipt.reconstructed_roots_match,
            "live_root_unchanged": receipt.live_root_unchanged,
            "one_search_begin_end": (
                receipt.begin_search_count == receipt.search_end_count == 1
            ),
            "immediate_measurement_count_matches_actions": (
                receipt.immediate_measurement_count == len(actions)
                and receipt.rollout_count == 0
            ),
        }
        counts = {
            "search_begin_count": receipt.begin_search_count,
            "search_end_count": receipt.search_end_count,
            "immediate_measurement_count": receipt.immediate_measurement_count,
        }
        serialized = _canonical_bytes(actions)
        checks["factory_private_content_absent"] = (
            factory.serialized_private_content_absent(serialized)
        )
        assert_private_content_absent(actions)
        if not all(checks.values()):
            raise DiagnosticSafetyError("immediate successor teacher safety gate failed")

        return actions, checks, counts
    finally:
        factory.close()


def _find_and_score_root(
    seed: int,
    contract: Mapping[str, Any],
    decks: Mapping[int, list[int]],
    scorer: SuccessorScorer,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, int]]:
    agents = {
        0: teacher_smoke.SpecAgent.from_str(str(contract["policy0"])),
        1: teacher_smoke.SpecAgent.from_str(str(contract["policy1"])),
    }
    observation = None
    try:
        observation, _start = teacher_smoke.battle_start(decks[0], decks[1])
        if observation is None:
            raise DiagnosticSafetyError("source BattleStart failed")
        rng = random.Random(seed)
        for _ in range(int(contract["source_max_steps"])):
            current = observation.get("current")
            if not isinstance(current, dict) or current.get("result") != -1:
                break
            actor = current.get("yourIndex")
            if actor not in (0, 1):
                raise DiagnosticSafetyError("source game actor is invalid")
            try:
                selected = list(agents[int(actor)].select(observation, rng))
            except Exception:
                raise DiagnosticSafetyError("source policy failed") from None
            select = observation.get("select")
            selections: list[list[int]] = []
            candidate = False
            if isinstance(select, dict):
                turn = current.get("turn")
                materialized = (
                    isinstance(turn, int)
                    and not isinstance(turn, bool)
                    and turn >= int(contract["minimum_source_turn"])
                    and (
                        int(contract["maximum_source_turn"]) == 0
                        or turn <= int(contract["maximum_source_turn"])
                    )
                )
                if materialized:
                    try:
                        selections = enumerate_legal_selections(
                            select,
                            order_sensitive=teacher_smoke._order_sensitive(select),
                            root_cap=int(contract["root_action_cap"]),
                        )
                        normalized = list(selected)
                        if not teacher_smoke._order_sensitive(select):
                            normalized.sort()
                        candidate = 2 <= len(selections) <= int(contract["root_action_cap"])
                        candidate &= normalized in selections
                    except (ValueError, LegalSelectionCapExceeded):
                        candidate = False
            if candidate:
                actions, checks, counts = _score_root(observation, selections, scorer)
                root = {
                    "scratch_seed": seed,
                    "action_count": len(actions),
                    "actions": actions,
                    **_rank_summary(actions),
                }

                return root, checks, counts
            try:
                observation = teacher_smoke.battle_select(selected)
            except Exception:
                raise DiagnosticSafetyError("source game selection failed") from None
    finally:
        if observation is not None:
            teacher_smoke.battle_finish()

    raise DiagnosticSafetyError("seed produced no bounded multi-action root")


def _aggregate_roots(roots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    agreements = sum(int(root["pairwise_agreement_count"]) for root in roots)
    reversals = sum(int(root["pairwise_reversal_count"]) for root in roots)
    ties = sum(int(root["pairwise_tie_count"]) for root in roots)
    comparable = agreements + reversals
    top_agreements = sum(bool(root["top_action_agreement"]) for root in roots)

    return {
        "root_count": len(roots),
        "action_count": sum(int(root["action_count"]) for root in roots),
        "pairwise_agreement_count": agreements,
        "pairwise_reversal_count": reversals,
        "pairwise_tie_count": ties,
        "pairwise_comparable_count": comparable,
        "pairwise_agreement_rate": agreements / comparable if comparable else None,
        "top_action_agreement_count": top_agreements,
        "top_action_disagreement_count": len(roots) - top_agreements,
        "top_action_agreement_rate": top_agreements / len(roots),
    }


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_PERSISTED_KEYS:
                raise DiagnosticSafetyError("diagnostic receipt contains a forbidden field")
            _reject_forbidden_keys(item)


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != _WORKER_RECEIPT_FIELDS:
        raise DiagnosticSafetyError("diagnostic receipt field inventory drifted")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise DiagnosticSafetyError("diagnostic receipt schema drifted")
    roots = receipt.get("roots")
    if not isinstance(roots, list) or not roots:
        raise DiagnosticSafetyError("diagnostic receipt has no roots")
    for root in roots:
        if not isinstance(root, dict) or set(root) != _ROOT_FIELDS:
            raise DiagnosticSafetyError("diagnostic root field inventory drifted")
        actions = root.get("actions")
        if (
            not isinstance(actions, list)
            or len(actions) != root.get("action_count")
            or len(actions) < 2
        ):
            raise DiagnosticSafetyError("diagnostic root action inventory drifted")
        for action in actions:
            if not isinstance(action, dict) or set(action) != _ACTION_FIELDS:
                raise DiagnosticSafetyError("diagnostic action field inventory drifted")
            if not isinstance(action["action_id"], str):
                raise DiagnosticSafetyError("diagnostic action ID is invalid")
            for field in ("v0_score", "m4_score"):
                score = action[field]
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not 0.0 <= float(score) <= 1.0
                ):
                    raise DiagnosticSafetyError("diagnostic action score is invalid")
    safety = receipt.get("safety_counts")
    if (
        not isinstance(safety, dict)
        or set(safety) != _SAFETY_FIELDS
        or any(value != 0 for value in safety.values())
    ):
        raise DiagnosticSafetyError("diagnostic safety counters are not clean")
    checks = receipt.get("boolean_checks")
    if not isinstance(checks, dict) or any(value is not True for value in checks.values()):
        raise DiagnosticSafetyError("diagnostic boolean safety gate failed")
    _reject_forbidden_keys(receipt)
    assert_private_content_absent(receipt)


def _run_worker(args: argparse.Namespace) -> int:
    try:
        manifest_path = Path(args.seed_manifest).resolve()
        manifest = _validate_seed_manifest(manifest_path)
        contract = manifest["contract"]
        deck0 = Path(str(contract["deck0"]["path"]))
        deck1 = Path(str(contract["deck1"]["path"]))
        _verify_bound_file(deck0, contract["deck0"]["sha256"], "deck0")
        _verify_bound_file(deck1, contract["deck1"]["sha256"], "deck1")
        v0 = _load_v0(contract)
        m4 = FrozenM4Endpoint.load(contract)
        scorer = SuccessorScorer(v0, m4)
        decks = {0: teacher_smoke.read_deck(deck0), 1: teacher_smoke.read_deck(deck1)}
        roots = []
        all_checks = []
        total_counts = {
            "search_begin_count": 0,
            "search_end_count": 0,
            "immediate_measurement_count": 0,
        }
        for seed in contract["scratch_seeds"]:
            root, checks, counts = _find_and_score_root(seed, contract, decks, scorer)
            roots.append(root)
            all_checks.append(checks)
            for name in total_counts:
                total_counts[name] += counts[name]
        aggregate = _aggregate_roots(roots)
        aggregate.update(total_counts)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "worker_completed_safe",
            "seed_manifest_sha256": _sha256_file(manifest_path),
            "endpoint_hashes": {
                "v0_checkpoint_sha256": contract["v0"]["checkpoint_sha256"],
                "v0_calibrator_bundle_sha256": contract["v0"]["calibrator_bundle_sha256"],
                "m4_checkpoint_sha256": contract["m4"]["checkpoint_sha256"],
                "m4_calibrator_sha256": contract["m4"]["calibrator_sha256"],
            },
            "roots": roots,
            "aggregate_counts": aggregate,
            "safety_counts": {"illegal_actions": 0, "crashes": 0, "timeouts": 0},
            "boolean_checks": {
                "seed_contract_valid": True,
                "endpoint_hashes_valid": True,
                "one_root_per_seed": len(roots) == len(contract["scratch_seeds"]),
                "no_replacement_seeds": [root["scratch_seed"] for root in roots]
                == contract["scratch_seeds"],
                "immediate_successor_only": True,
                "actor_visible_inputs_stable": all(
                    checks["actor_visible_input_stable"] for checks in all_checks
                ),
                "reconstructed_roots_match": all(
                    checks["reconstructed_root_matches"] for checks in all_checks
                ),
                "live_roots_unchanged": all(
                    checks["live_root_unchanged"] for checks in all_checks
                ),
                "one_search_begin_end_per_root": all(
                    checks["one_search_begin_end"] for checks in all_checks
                ),
                "immediate_measurements_match_actions": all(
                    checks["immediate_measurement_count_matches_actions"]
                    for checks in all_checks
                ),
                "factory_private_content_absent": all(
                    checks["factory_private_content_absent"] for checks in all_checks
                ),
            },
        }
        _validate_receipt_shape(receipt)
        _atomic_json(Path(args.worker_output), receipt)
    except Exception as error:
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "worker_failed_safe",
            "failure_category": (
                str(error)
                if isinstance(error, (DiagnosticSafetyError, SimulatorTeacherSafetyError))
                else f"unexpected {type(error).__name__}"
            ),
            "exception_type": type(error).__name__,
        }
        _atomic_json(Path(args.failure_receipt), failure)

        return 2

    return 0


def _run_parent(args: argparse.Namespace) -> int:
    seed_manifest = Path(args.seed_manifest).resolve()
    output = Path(args.output).resolve()
    worker_output = output.with_name(f".{output.name}.worker")
    failure_receipt = output.with_name(f".{output.name}.failure")
    if output.exists() or worker_output.exists() or failure_receipt.exists():
        raise DiagnosticSafetyError("diagnostic output path already exists")
    manifest = _validate_seed_manifest(seed_manifest)
    manifest_sha256 = _sha256_file(seed_manifest)
    protected_before = teacher_smoke._tree_sha256(teacher_smoke.PROTECTED_PATHS)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seed-manifest",
        str(seed_manifest),
        "--worker-output",
        str(worker_output),
        "--failure-receipt",
        str(failure_receipt),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=args.worker_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _atomic_json(failure_receipt, {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "worker_failed_safe",
            "failure_category": "isolated diagnostic worker timed out",
            "exception_type": "TimeoutExpired",
        })
        raise DiagnosticSafetyError("isolated diagnostic worker timed out") from None
    if completed.returncode != 0:
        category = "failure receipt unavailable"
        if failure_receipt.is_file():
            failure = _read_json_object(failure_receipt, "failure receipt")
            category = str(failure.get("failure_category") or category)
        raise DiagnosticSafetyError(f"isolated diagnostic worker failed: {category}")
    if completed.stdout.strip() or completed.stderr.strip():
        raise DiagnosticSafetyError("isolated diagnostic worker emitted unexpected output")
    receipt = _read_json_object(worker_output, "worker receipt")
    _validate_receipt_shape(receipt)
    if receipt["seed_manifest_sha256"] != manifest_sha256:
        raise DiagnosticSafetyError("worker used a different seed manifest")
    contract = manifest["contract"]
    expected_endpoints = {
        "v0_checkpoint_sha256": contract["v0"]["checkpoint_sha256"],
        "v0_calibrator_bundle_sha256": contract["v0"]["calibrator_bundle_sha256"],
        "m4_checkpoint_sha256": contract["m4"]["checkpoint_sha256"],
        "m4_calibrator_sha256": contract["m4"]["calibrator_sha256"],
    }
    if receipt["endpoint_hashes"] != expected_endpoints:
        raise DiagnosticSafetyError("worker endpoint identities drifted")
    protected_after = teacher_smoke._tree_sha256(teacher_smoke.PROTECTED_PATHS)
    receipt["status"] = "completed_frozen_diagnostic"
    receipt["boolean_checks"].update({
        "worker_process_exited": True,
        "seed_manifest_unchanged": _sha256_file(seed_manifest) == manifest_sha256,
        "protected_runtime_submission_unchanged": protected_before == protected_after,
        "persisted_private_content_scan": True,
    })
    _validate_receipt_shape(receipt)
    _atomic_json(output, receipt)
    worker_output.unlink()
    failure_receipt.unlink(missing_ok=True)
    print(json.dumps({
        "status": receipt["status"],
        "output": str(output),
        "aggregate_counts": receipt["aggregate_counts"],
        "safety_counts": receipt["safety_counts"],
    }, sort_keys=True))

    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", default=str(DEFAULT_SEED_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    parser.add_argument("--failure-receipt", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_output:
        return _run_worker(args)

    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
