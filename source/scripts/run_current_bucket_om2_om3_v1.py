"""Train the frozen equal-budget current-population OM2/OM3 comparison.

The runner consumes lightweight behavior shards whose observations are
coordinates into immutable float32 X stores.  It never accepts player identity
as a model input.  OM3 differs from OM2 only by appending the manifest-declared
external competence-bucket vector.

Real corpus execution is intentionally explicit.  ``--validate-only`` is the
default; training requires ``--execute-frozen-corpus``.  Synthetic execution is
available only through the Python API for contract tests and is labelled in all
receipts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


CORPUS_SCHEMA = "deepmew.current_bucket_behavior_corpus.v1"
ROW_SCHEMA = "deepmew.current_bucket_behavior_row.v1"
RESULT_SCHEMA = "deepmew.current_bucket_om2_om3_result.v1"
FROZEN_CORPUS_STATUSES = {
    "COMPLETE_LIGHTWEIGHT_REFERENCES_NO_X_DUPLICATION",
    "FROZEN_FOR_OM2_OM3",
}
SPLITS = {"train", "calibration", "test"}
REQUIRED_SLICE_FIELDS = (
    "player",
    "deck_family",
    "player_deck",
    "bucket",
    "rating_trend",
    "playstyle_cluster",
    "legal_set_size",
    "game_phase",
    "cabt_holdout",
    "reference_policy_holdout",
    "population_scope",
    "internal_panel_bucket",
)
AUXILIARY_HEADS = ("resource_commitment", "next_turn_threat", "strategic_intent")
FORBIDDEN_MODEL_FIELDS = {
    "canonical_player_id",
    "player_identity",
    "exact_player_identity",
    "checkpoint_identity",
    "hidden_hand",
    "deck_order",
    "hidden_prizes",
    "future_action",
    "game_result",
    "future_rating",
    "oracle",
    "search_begin_input",
}
AUDIT_DECISIONS = {
    "core path",
    "proposal/ranker only",
    "diagnostic only",
    "quarantine",
    "delete",
}
DEFAULT_MAX_PADDED_ACTION_CELLS = 16_384


@dataclass(frozen=True)
class FrozenBudget:
    architecture: str = "shared_context_trunk_masked_legal_action_scorer_v1"
    optimizer: str = "AdamW"
    hidden_dim: int = 64
    depth: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 3
    batch_size: int = 64
    gradient_clip_norm: float = 1.0
    train_seed: int = 719_032_026
    bootstrap_seed: int = 719_033_026
    bootstrap_resamples: int = 10_000
    ece_bins: int = 15
    temperature_lower: float = 0.1
    temperature_upper: float = 10.0
    auxiliary_loss_weight: float = 0.10
    material_nll_gain_floor: float = 0.005


FROZEN_BUDGET = FrozenBudget()


class CorpusContractError(ValueError):
    """Raised when the immutable manifest/shard contract is not satisfied."""


@dataclass(frozen=True)
class Action:
    features: tuple[float, ...]
    family: str
    target: str
    resource_commitment: float | None
    next_turn_threat: float | None
    strategic_intent: int | None
    structure: Mapping[str, Any]


@dataclass(frozen=True)
class DecisionRow:
    row_id: str
    game_id: str
    trajectory_id: str
    split: str
    x: tuple[float, ...]
    deck_features: tuple[float, ...]
    playstyle_features: tuple[float, ...]
    bucket_features: tuple[float, ...]
    legal_actions: tuple[Action, ...]
    chosen_action_index: int
    evidence_weight: float
    actor_loss_weight: float
    forced: bool
    slices: Mapping[str, str]
    competence_context_kind: str


@dataclass(frozen=True)
class PaddedBatch:
    rows: tuple[DecisionRow, ...]
    contexts: torch.Tensor
    actions: torch.Tensor
    legal_mask: torch.Tensor
    targets: torch.Tensor
    row_weights: torch.Tensor
    resource_targets: torch.Tensor
    resource_mask: torch.Tensor
    threat_targets: torch.Tensor
    threat_mask: torch.Tensor
    intent_targets: torch.Tensor
    intent_mask: torch.Tensor


@dataclass(frozen=True)
class RowPrediction:
    row_id: str
    game_id: str
    slices: Mapping[str, str]
    nll: float
    brier: float
    top1: float
    top3: float
    family_accuracy: float
    target_accuracy: float
    confidence: float
    entropy: float
    ood_score: float
    resource_absolute_error: float | None
    threat_accuracy: float | None
    intent_accuracy: float | None
    evidence_weight: float


def heldout_slice_rows(rows: Sequence[RowPrediction], field: str) -> list[RowPrediction]:
    return [row for row in rows if row.slices[field].startswith("heldout:")]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CorpusContractError(f"expected JSON object: {path}")

    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)

    return path if path.is_absolute() else root / path


def _require_vector(value: Any, dimension: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise CorpusContractError(f"{field} must be a {dimension}-element list")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise CorpusContractError(f"{field} contains a non-finite value")

    return result


def _open_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"}
    with opener(path, **kwargs) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CorpusContractError(f"non-object row at {path}:{line_number}")
            yield value


def validate_manifest(manifest_path: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CORPUS_SCHEMA:
        raise CorpusContractError(f"unexpected corpus schema: {manifest.get('schema_version')!r}")
    if manifest.get("status") not in FROZEN_CORPUS_STATUSES:
        raise CorpusContractError("corpus is not frozen and complete for OM2/OM3")
    if manifest.get("population_model_type") != "CURRENT_POPULATION_MODEL":
        raise CorpusContractError("only CURRENT_POPULATION_MODEL corpora are accepted")
    if bool(manifest.get("synthetic_fixture")) and not allow_synthetic:
        raise CorpusContractError("synthetic corpus requires explicit allow_synthetic=True")
    schema = manifest.get("feature_schema")
    if not isinstance(schema, dict):
        raise CorpusContractError("feature_schema is required")
    for field in ("x_dim", "deck_dim", "playstyle_dim", "bucket_dim", "action_dim", "intent_classes"):
        if not isinstance(schema.get(field), int) or int(schema[field]) <= 0:
            raise CorpusContractError(f"feature_schema.{field} must be a positive integer")
    complete_action_contract = manifest.get("complete_action_contract")
    if complete_action_contract != {
        "unit": "complete_compound_action",
        "chosen_index_derivation": "exact_canonical_structure_match",
        "prompt_option_indices_not_accepted": True,
    }:
        raise CorpusContractError("manifest must bind complete compound-action enumeration and exact chosen derivation")
    target_contract = manifest.get("target_instrumentation")
    if not isinstance(target_contract, dict):
        raise CorpusContractError("target_instrumentation is required")
    for head in ("action_family", "structured_target"):
        spec = target_contract.get(head)
        if (
            not isinstance(spec, dict)
            or spec.get("status") != "instrumented"
            or not spec.get("source")
            or spec.get("causal_scope") != "actor_visible_pre_action_plus_current_complete_action"
        ):
            raise CorpusContractError(f"{head} requires a direct, named instrumentation source")
    for head in AUXILIARY_HEADS:
        spec = target_contract.get(head)
        if not isinstance(spec, dict) or spec.get("status") not in {"instrumented", "not_instrumented"}:
            raise CorpusContractError(f"{head} must be instrumented or explicitly not_instrumented")
        if spec["status"] == "instrumented" and (
            not spec.get("source")
            or spec.get("causal_scope") != "actor_visible_pre_action_plus_current_complete_action"
        ):
            raise CorpusContractError(f"{head} instrumentation needs an explicit direct source")
    evaluation_instrumentation = manifest.get("evaluation_instrumentation")
    if not isinstance(evaluation_instrumentation, dict):
        raise CorpusContractError("evaluation_instrumentation is required")
    for surface in ("cabt_holdout", "reference_policy_holdout", "internal_panel_adapter"):
        spec = evaluation_instrumentation.get(surface)
        if not isinstance(spec, dict) or spec.get("status") not in {"instrumented", "not_instrumented"}:
            raise CorpusContractError(f"evaluation_instrumentation.{surface} needs an explicit status")
    if manifest.get("model_input_excludes") != sorted(FORBIDDEN_MODEL_FIELDS):
        raise CorpusContractError("model_input_excludes does not exactly bind the leakage contract")
    stores = manifest.get("x_stores")
    shards = manifest.get("shards")
    if not isinstance(stores, list) or not stores or not isinstance(shards, list) or not shards:
        raise CorpusContractError("non-empty x_stores and shards are required")
    root = manifest_path.parent
    seen_store_ids: set[str] = set()
    for store in stores:
        if not isinstance(store, dict):
            raise CorpusContractError("x_store must be an object")
        store_id = str(store.get("id", ""))
        if not store_id or store_id in seen_store_ids:
            raise CorpusContractError("x_store IDs must be non-empty and unique")
        seen_store_ids.add(store_id)
        path = _resolve(root, str(store.get("path", "")))
        if not path.is_file() or file_sha256(path) != store.get("sha256"):
            raise CorpusContractError(f"X store missing or hash mismatch: {path}")
        if store.get("dtype") != "float32" or store.get("coordinate_contract") != "zero_based_row_index":
            raise CorpusContractError("X stores must use float32 zero-based immutable row coordinates")
        if store.get("format") not in {"raw_float32_c_order", "npy_float32_c_order"}:
            raise CorpusContractError("X store format must explicitly declare raw or npy float32 C-order")
    for shard in shards:
        if not isinstance(shard, dict):
            raise CorpusContractError("shard must be an object")
        path = _resolve(root, str(shard.get("path", "")))
        if not path.is_file() or file_sha256(path) != shard.get("sha256"):
            raise CorpusContractError(f"behavior shard missing or hash mismatch: {path}")
        if not isinstance(shard.get("rows"), int) or int(shard["rows"]) <= 0:
            raise CorpusContractError("every behavior shard needs a positive row count")

    return manifest


def load_corpus(manifest_path: Path, *, allow_synthetic: bool = False) -> tuple[dict[str, Any], list[DecisionRow]]:
    manifest_path = manifest_path.resolve()
    manifest = validate_manifest(manifest_path, allow_synthetic=allow_synthetic)
    root = manifest_path.parent
    schema = manifest["feature_schema"]
    stores: dict[str, np.ndarray] = {}
    for spec in manifest["x_stores"]:
        path = _resolve(root, str(spec["path"]))
        expected_shape = tuple(int(value) for value in spec["shape"])
        if spec["format"] == "raw_float32_c_order":
            expected_bytes = math.prod(expected_shape) * np.dtype(np.float32).itemsize
            if path.stat().st_size != expected_bytes:
                raise CorpusContractError(f"raw X store byte size does not match shape: {path}")
            array = np.memmap(path, dtype=np.float32, mode="r", shape=expected_shape, order="C")
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise CorpusContractError(f"X store dtype/shape mismatch: {path}")
        if array.ndim != 2 or array.shape[1] != int(schema["x_dim"]):
            raise CorpusContractError(f"X store feature width mismatch: {path}")
        array.flags.writeable = False
        stores[str(spec["id"])] = array

    rows: list[DecisionRow] = []
    seen_row_ids: set[str] = set()
    game_splits: dict[str, str] = {}
    for shard in manifest["shards"]:
        path = _resolve(root, str(shard["path"]))
        shard_rows = list(_open_jsonl(path))
        if len(shard_rows) != int(shard["rows"]):
            raise CorpusContractError(f"shard row count mismatch: {path}")
        for raw in shard_rows:
            if raw.get("schema_version") != ROW_SCHEMA:
                raise CorpusContractError("unexpected behavior row schema")
            forbidden = FORBIDDEN_MODEL_FIELDS & set(raw.get("model_inputs", {}))
            if forbidden:
                raise CorpusContractError(f"forbidden model inputs: {sorted(forbidden)}")
            row_id = str(raw.get("row_id", ""))
            game_id = str(raw.get("game_id", ""))
            trajectory_id = str(raw.get("trajectory_id", ""))
            split = str(raw.get("split", ""))
            if not row_id or row_id in seen_row_ids or not game_id or not trajectory_id or split not in SPLITS:
                raise CorpusContractError("row/game/trajectory identities and valid split are required")
            seen_row_ids.add(row_id)
            if game_id in game_splits and game_splits[game_id] != split:
                raise CorpusContractError(f"game split leakage: {game_id}")
            game_splits[game_id] = split
            coordinate = raw.get("x_coordinate")
            if not isinstance(coordinate, dict) or str(coordinate.get("store_id")) not in stores:
                raise CorpusContractError("valid x_coordinate store_id is required")
            x_index = coordinate.get("row_index")
            store = stores[str(coordinate["store_id"])]
            if not isinstance(x_index, int) or not 0 <= x_index < len(store):
                raise CorpusContractError("X row coordinate is out of bounds")
            model_inputs = raw.get("model_inputs")
            if not isinstance(model_inputs, dict):
                raise CorpusContractError("model_inputs is required")
            legal_raw = raw.get("legal_actions")
            if not isinstance(legal_raw, list) or not legal_raw:
                raise CorpusContractError("complete non-empty legal_actions are required")
            actions = []
            target_contract = manifest["target_instrumentation"]
            for index, action in enumerate(legal_raw):
                if not isinstance(action, dict) or not isinstance(action.get("structure"), dict):
                    raise CorpusContractError("every legal action needs a complete structure")
                auxiliary = action.get("auxiliary_targets", {})
                if not isinstance(auxiliary, dict):
                    raise CorpusContractError("auxiliary_targets must be an object")
                for head in AUXILIARY_HEADS:
                    is_instrumented = target_contract[head]["status"] == "instrumented"
                    if is_instrumented != (head in auxiliary and auxiliary[head] is not None):
                        raise CorpusContractError(f"{head} label presence disagrees with target_instrumentation")
                intent_value = auxiliary.get("strategic_intent")
                if intent_value is not None and not 0 <= int(intent_value) < int(schema["intent_classes"]):
                    raise CorpusContractError("strategic_intent label is out of range")
                actions.append(Action(
                    features=_require_vector(action.get("features"), int(schema["action_dim"]), f"action[{index}].features"),
                    family=str(action.get("family", "")),
                    target=str(action.get("target", "")),
                    resource_commitment=(float(auxiliary["resource_commitment"]) if auxiliary.get("resource_commitment") is not None else None),
                    next_turn_threat=(float(auxiliary["next_turn_threat"]) if auxiliary.get("next_turn_threat") is not None else None),
                    strategic_intent=(int(intent_value) if intent_value is not None else None),
                    structure=action["structure"],
                ))
            chosen = raw.get("chosen_action_index")
            if not isinstance(chosen, int) or not 0 <= chosen < len(actions):
                raise CorpusContractError("chosen_action_index is outside the complete legal set")
            if raw.get("chosen_action") != actions[chosen].structure:
                raise CorpusContractError("chosen_action must exactly match the indexed legal action structure")
            forced = bool(raw.get("forced"))
            actor_loss_weight = float(raw.get("actor_loss_weight", -1.0))
            if forced != (len(actions) == 1):
                raise CorpusContractError("forced flag must exactly match legal-set size one")
            if forced and actor_loss_weight != 0.0:
                raise CorpusContractError("forced rows must have actor_loss_weight=0.0")
            if not forced and actor_loss_weight <= 0.0:
                raise CorpusContractError("non-forced rows need positive actor_loss_weight")
            evidence_weight = float(raw.get("current_behavior_evidence_weight", 0.0))
            if not math.isfinite(evidence_weight) or evidence_weight <= 0.0:
                raise CorpusContractError("current_behavior_evidence_weight must be finite and positive")
            slices = raw.get("slices")
            if not isinstance(slices, dict) or any(field not in slices for field in REQUIRED_SLICE_FIELDS):
                raise CorpusContractError("complete player/deck/bucket evaluation slices are required")
            if any(not isinstance(slices[field], str) for field in REQUIRED_SLICE_FIELDS):
                raise CorpusContractError("all evaluation slices must be scalar strings")
            heldout_identity_fields = ("player", "deck_family", "player_deck")
            if split != "test" and any(slices[field].startswith("heldout:") for field in heldout_identity_fields):
                raise CorpusContractError("held-out player, deck, and player-deck rows must be test-only")
            if slices["legal_set_size"] != str(len(actions)):
                raise CorpusContractError("legal_set_size slice must equal the complete compound-action count")
            competence_context_kind = str(raw.get("competence_context_kind", ""))
            if competence_context_kind == "CURRENT_EXTERNAL_RATING":
                if slices["population_scope"] != "external_current_population" or slices["internal_panel_bucket"] != "not_applicable":
                    raise CorpusContractError("external rows must keep internal panel bucket separate and not applicable")
            elif competence_context_kind == "FROZEN_PANEL_COMPETENCE_BUCKET_ADAPTER":
                if split != "test" or slices["population_scope"] != "internal_panel_adapter":
                    raise CorpusContractError("internal panel adapter rows must be separately labeled and test-only")
                if slices["bucket"] != "not_applicable_external" or slices["internal_panel_bucket"] == "not_applicable":
                    raise CorpusContractError("internal adapter bucket must never occupy the external bucket slice")
            else:
                raise CorpusContractError("competence_context_kind must identify external rating or internal adapter")
            for surface, slice_field in (("cabt_holdout", "cabt_holdout"), ("reference_policy_holdout", "reference_policy_holdout")):
                status = manifest["evaluation_instrumentation"][surface]["status"]
                if (status == "not_instrumented") != (slices[slice_field] == "not_instrumented"):
                    raise CorpusContractError(f"{surface} slice disagrees with evaluation instrumentation")
            rows.append(DecisionRow(
                row_id=row_id,
                game_id=game_id,
                trajectory_id=trajectory_id,
                split=split,
                x=tuple(float(value) for value in store[x_index]),
                deck_features=_require_vector(model_inputs.get("deck_features"), int(schema["deck_dim"]), "deck_features"),
                playstyle_features=_require_vector(model_inputs.get("playstyle_features"), int(schema["playstyle_dim"]), "playstyle_features"),
                bucket_features=_require_vector(model_inputs.get("bucket_features"), int(schema["bucket_dim"]), "bucket_features"),
                legal_actions=tuple(actions),
                chosen_action_index=chosen,
                evidence_weight=evidence_weight,
                actor_loss_weight=actor_loss_weight,
                forced=forced,
                slices={field: str(slices[field]) for field in REQUIRED_SLICE_FIELDS},
                competence_context_kind=competence_context_kind,
            ))
    if len(rows) != int(manifest.get("rows", -1)):
        raise CorpusContractError("manifest total row count mismatch")
    if {row.split for row in rows} != SPLITS:
        raise CorpusContractError("train, calibration, and test splits must all be non-empty")

    return manifest, rows


class OpponentModel(nn.Module):
    def __init__(self, context_dim: int, action_dim: int, intent_classes: int, budget: FrozenBudget) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = context_dim
        for _ in range(budget.depth):
            layers.extend((nn.Linear(width, budget.hidden_dim), nn.GELU()))
            width = budget.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.action_head = nn.Sequential(
            nn.Linear(budget.hidden_dim + action_dim, budget.hidden_dim),
            nn.GELU(),
            nn.Linear(budget.hidden_dim, 1),
        )
        self.resource_head = nn.Linear(budget.hidden_dim, 1)
        self.threat_head = nn.Linear(budget.hidden_dim, 1)
        self.intent_head = nn.Linear(budget.hidden_dim, intent_classes)

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(context)
        expanded = hidden.unsqueeze(0).expand(actions.shape[0], -1)
        logits = self.action_head(torch.cat((expanded, actions), dim=1)).squeeze(1)

        return logits, self.resource_head(hidden).squeeze(0), self.threat_head(hidden).squeeze(0), self.intent_head(hidden)

    def forward_padded(self, contexts: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(contexts)
        expanded = hidden.unsqueeze(1).expand(-1, actions.shape[1], -1)
        logits = self.action_head(torch.cat((expanded, actions), dim=2)).squeeze(2)

        return (
            logits,
            self.resource_head(hidden).squeeze(1),
            self.threat_head(hidden).squeeze(1),
            self.intent_head(hidden),
        )


def context_vector(row: DecisionRow, rung: str, device: torch.device | None = None) -> torch.Tensor:
    values = row.x + row.deck_features + row.playstyle_features
    if rung == "OM3":
        values += row.bucket_features
    elif rung != "OM2":
        raise ValueError(f"unknown rung: {rung}")

    return torch.tensor(values, dtype=torch.float32, device=device)


def bounded_row_batches(
    rows: Sequence[DecisionRow],
    *,
    max_rows: int,
    max_padded_action_cells: int,
) -> Iterable[list[DecisionRow]]:
    if max_rows <= 0 or max_padded_action_cells <= 0:
        raise ValueError("batch bounds must be positive")
    batch: list[DecisionRow] = []
    widest = 0
    for row in rows:
        candidate_width = max(widest, len(row.legal_actions))
        candidate_cells = (len(batch) + 1) * candidate_width
        if batch and (len(batch) >= max_rows or candidate_cells > max_padded_action_cells):
            yield batch
            batch = []
            widest = 0
        batch.append(row)
        widest = max(widest, len(row.legal_actions))
    if batch:
        yield batch


def collate_rows(rows: Sequence[DecisionRow], rung: str, device: torch.device) -> PaddedBatch:
    if not rows:
        raise ValueError("cannot collate an empty row batch")
    batch_rows = tuple(rows)
    batch_size = len(batch_rows)
    legal_width = max(len(row.legal_actions) for row in batch_rows)
    action_dim = len(batch_rows[0].legal_actions[0].features)
    contexts = torch.stack([context_vector(row, rung, device) for row in batch_rows])
    actions = torch.zeros((batch_size, legal_width, action_dim), dtype=torch.float32, device=device)
    legal_mask = torch.zeros((batch_size, legal_width), dtype=torch.bool, device=device)
    targets = torch.empty(batch_size, dtype=torch.long, device=device)
    row_weights = torch.empty(batch_size, dtype=torch.float32, device=device)
    resource_targets = torch.zeros(batch_size, dtype=torch.float32, device=device)
    resource_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    threat_targets = torch.zeros(batch_size, dtype=torch.float32, device=device)
    threat_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    intent_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    intent_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for index, row in enumerate(batch_rows):
        legal_count = len(row.legal_actions)
        actions[index, :legal_count] = torch.tensor(
            [action.features for action in row.legal_actions],
            dtype=torch.float32,
            device=device,
        )
        legal_mask[index, :legal_count] = True
        targets[index] = row.chosen_action_index
        row_weights[index] = row.actor_loss_weight * row.evidence_weight
        chosen = row.legal_actions[row.chosen_action_index]
        if chosen.resource_commitment is not None:
            resource_targets[index] = chosen.resource_commitment
            resource_mask[index] = True
        if chosen.next_turn_threat is not None:
            threat_targets[index] = chosen.next_turn_threat
            threat_mask[index] = True
        if chosen.strategic_intent is not None:
            intent_targets[index] = chosen.strategic_intent
            intent_mask[index] = True

    return PaddedBatch(
        rows=batch_rows,
        contexts=contexts,
        actions=actions,
        legal_mask=legal_mask,
        targets=targets,
        row_weights=row_weights,
        resource_targets=resource_targets,
        resource_mask=resource_mask,
        threat_targets=threat_targets,
        threat_mask=threat_mask,
        intent_targets=intent_targets,
        intent_mask=intent_mask,
    )


def padded_outputs(model: OpponentModel, batch: PaddedBatch) -> tuple[torch.Tensor, ...]:
    logits, resource, threat, intent = model.forward_padded(batch.contexts, batch.actions)
    masked_logits = logits.masked_fill(~batch.legal_mask, -torch.inf)

    return masked_logits, resource, threat, intent


def batched_actor_losses(model: OpponentModel, batch: PaddedBatch, budget: FrozenBudget) -> torch.Tensor:
    logits, resource, threat, intent = padded_outputs(model, batch)
    primary = nn.functional.cross_entropy(logits, batch.targets, reduction="none")
    auxiliary_sum = torch.zeros_like(primary)
    auxiliary_count = torch.zeros_like(primary)
    resource_loss = (resource - batch.resource_targets).square()
    auxiliary_sum = auxiliary_sum + resource_loss * batch.resource_mask
    auxiliary_count = auxiliary_count + batch.resource_mask
    threat_loss = nn.functional.binary_cross_entropy_with_logits(
        threat,
        batch.threat_targets,
        reduction="none",
    )
    auxiliary_sum = auxiliary_sum + threat_loss * batch.threat_mask
    auxiliary_count = auxiliary_count + batch.threat_mask
    intent_loss = nn.functional.cross_entropy(intent, batch.intent_targets, reduction="none")
    auxiliary_sum = auxiliary_sum + intent_loss * batch.intent_mask
    auxiliary_count = auxiliary_count + batch.intent_mask
    auxiliary = auxiliary_sum / auxiliary_count.clamp_min(1.0)

    return (primary + budget.auxiliary_loss_weight * auxiliary) * batch.row_weights


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_model(
    schema: Mapping[str, Any],
    rung: str,
    budget: FrozenBudget = FROZEN_BUDGET,
    device: torch.device | None = None,
) -> OpponentModel:
    context_dim = int(schema["x_dim"]) + int(schema["deck_dim"]) + int(schema["playstyle_dim"])
    if rung == "OM3":
        context_dim += int(schema["bucket_dim"])
    elif rung != "OM2":
        raise ValueError(f"unknown rung: {rung}")
    _set_seed(budget.train_seed)

    return OpponentModel(context_dim, int(schema["action_dim"]), int(schema["intent_classes"]), budget).to(device)


def actor_loss(model: OpponentModel, row: DecisionRow, rung: str, budget: FrozenBudget) -> torch.Tensor:
    if row.forced:
        return sum(parameter.sum() * 0.0 for parameter in model.parameters())
    device = next(model.parameters()).device
    context = context_vector(row, rung, device)
    actions = torch.tensor([action.features for action in row.legal_actions], dtype=torch.float32, device=device)
    logits, resource, threat, intent = model(context, actions)
    chosen = row.legal_actions[row.chosen_action_index]
    primary = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([row.chosen_action_index], device=device))
    auxiliary_terms = []
    if chosen.resource_commitment is not None:
        auxiliary_terms.append(nn.functional.mse_loss(resource.reshape(()), torch.tensor(chosen.resource_commitment, device=device)))
    if chosen.next_turn_threat is not None:
        auxiliary_terms.append(nn.functional.binary_cross_entropy_with_logits(threat.reshape(()), torch.tensor(chosen.next_turn_threat, device=device)))
    if chosen.strategic_intent is not None:
        auxiliary_terms.append(nn.functional.cross_entropy(intent.unsqueeze(0), torch.tensor([chosen.strategic_intent], device=device)))
    auxiliary = torch.stack(auxiliary_terms).mean() if auxiliary_terms else primary * 0.0

    return (primary + budget.auxiliary_loss_weight * auxiliary) * row.actor_loss_weight * row.evidence_weight


def train_rung(
    rows: Sequence[DecisionRow],
    schema: Mapping[str, Any],
    rung: str,
    budget: FrozenBudget = FROZEN_BUDGET,
    device: torch.device | None = None,
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
) -> tuple[OpponentModel, dict[str, Any]]:
    train_rows = [row for row in rows if row.split == "train" and not row.forced]
    if not train_rows:
        raise CorpusContractError("training split has no actor-loss rows")
    model = build_model(schema, rung, budget, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=budget.learning_rate,
        weight_decay=budget.weight_decay,
    )
    started = time.perf_counter()
    updates = 0
    microbatches = 0
    losses: list[float] = []
    model.train()
    for epoch in range(budget.epochs):
        order = list(train_rows)
        random.Random(budget.train_seed + epoch).shuffle(order)
        for start in range(0, len(order), budget.batch_size):
            logical_batch = order[start:start + budget.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss_value = 0.0
            for rows_batch in bounded_row_batches(
                logical_batch,
                max_rows=len(logical_batch),
                max_padded_action_cells=max_padded_action_cells,
            ):
                padded = collate_rows(rows_batch, rung, next(model.parameters()).device)
                microbatch_loss = batched_actor_losses(model, padded, budget).sum() / len(logical_batch)
                microbatch_loss.backward()
                batch_loss_value += float(microbatch_loss.detach())
                microbatches += 1
            nn.utils.clip_grad_norm_(model.parameters(), budget.gradient_clip_norm)
            optimizer.step()
            updates += 1
            losses.append(batch_loss_value)

    return model, {
        "budget": asdict(budget),
        "actor_rows": len(train_rows),
        "forced_rows_excluded": sum(row.split == "train" and row.forced for row in rows),
        "optimizer_updates": updates,
        "vectorized_microbatches": microbatches,
        "max_padded_action_cells": max_padded_action_cells,
        "mean_batch_loss": float(np.mean(losses)),
        "final_batch_loss": losses[-1],
        "elapsed_seconds": time.perf_counter() - started,
    }


def calibration_objective(
    model: OpponentModel,
    rows: Sequence[DecisionRow],
    rung: str,
    temperature: float,
    budget: FrozenBudget,
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
) -> float:
    device = next(model.parameters()).device
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for rows_batch in bounded_row_batches(
            rows,
            max_rows=budget.batch_size,
            max_padded_action_cells=max_padded_action_cells,
        ):
            padded = collate_rows(rows_batch, rung, device)
            logits = padded_outputs(model, padded)[0] / temperature
            total += float(nn.functional.cross_entropy(logits, padded.targets, reduction="sum"))
            count += len(rows_batch)

    return total / count


def _fit_temperature(
    model: OpponentModel,
    rows: Sequence[DecisionRow],
    rung: str,
    budget: FrozenBudget,
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
) -> float:
    calibration = [row for row in rows if row.split == "calibration" and not row.forced]
    if not calibration:
        raise CorpusContractError("calibration split has no actor-loss rows")
    device = next(model.parameters()).device
    raw = torch.tensor(0.0, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.25, max_iter=30, line_search_fn="strong_wolfe")
    model.eval()
    logit_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for rows_batch in bounded_row_batches(
            calibration,
            max_rows=budget.batch_size,
            max_padded_action_cells=max_padded_action_cells,
        ):
            padded = collate_rows(rows_batch, rung, device)
            logits = model.forward_padded(padded.contexts, padded.actions)[0]
            logit_batches.append((logits.cpu(), padded.legal_mask.cpu(), padded.targets.cpu()))

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        total_value = 0.0
        total_rows = sum(len(targets) for _, _, targets in logit_batches)
        for logits_cpu, mask_cpu, targets_cpu in logit_batches:
            temperature = raw.exp().clamp(budget.temperature_lower, budget.temperature_upper)
            logits = (logits_cpu.to(device) / temperature).masked_fill(~mask_cpu.to(device), -torch.inf)
            loss_sum = nn.functional.cross_entropy(
                logits,
                targets_cpu.to(device),
                reduction="sum",
            )
            (loss_sum / total_rows).backward()
            total_value += float(loss_sum.detach())

        return torch.tensor(total_value / total_rows, device=device)

    optimizer.step(closure)

    return float(raw.detach().exp().clamp(budget.temperature_lower, budget.temperature_upper))


def _rank_auc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    if len(scores) != len(positives):
        raise ValueError("scores and positives must have identical lengths")
    positive_count = sum(bool(label) for label in positives)
    negative_count = len(positives) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(scores, positives, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(bool(label) for _, label in ordered[start:end])
        start = end
    wins_plus_half_ties = positive_rank_sum - positive_count * (positive_count + 1) / 2.0

    return wins_plus_half_ties / (positive_count * negative_count)


def _ece(rows: Sequence[RowPrediction], bins: int) -> float:
    total = len(rows)
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            row for row in rows
            if lower <= row.confidence < upper or (index == bins - 1 and row.confidence == 1.0)
        ]
        if members:
            accuracy = float(np.mean([row.top1 for row in members]))
            confidence = float(np.mean([row.confidence for row in members]))
            value += len(members) / total * abs(accuracy - confidence)

    return value


def _game_equal(rows: Sequence[RowPrediction], field: str) -> float:
    games: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = getattr(row, field)
        if value is not None:
            games[row.game_id].append(float(value))

    if not games:
        raise ValueError(f"metric is not instrumented: {field}")

    return float(np.mean([np.mean(values) for values in games.values()]))


def _summary(rows: Sequence[RowPrediction], bins: int) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "games": 0, "status": "NOT_MEASURED_EMPTY_SLICE"}
    metrics = {
        field: _game_equal(rows, field)
        for field in (
            "nll", "brier", "top1", "top3", "family_accuracy", "target_accuracy",
            "entropy", "ood_score",
        )
    }
    optional_metrics = {}
    for field in ("resource_absolute_error", "threat_accuracy", "intent_accuracy"):
        try:
            optional_metrics[field] = _game_equal(rows, field)
        except ValueError:
            optional_metrics[field] = "not_instrumented"

    return {
        "rows": len(rows),
        "games": len({row.game_id for row in rows}),
        "weight_effective_sample_size": (
            sum(row.evidence_weight for row in rows) ** 2
            / sum(row.evidence_weight ** 2 for row in rows)
        ),
        "game_equal_masked_complete_action_nll": metrics["nll"],
        "game_equal_brier": metrics["brier"],
        "game_equal_top1_accuracy": metrics["top1"],
        "game_equal_top3_accuracy": metrics["top3"],
        "game_equal_action_family_accuracy": metrics["family_accuracy"],
        "game_equal_target_accuracy": metrics["target_accuracy"],
        "ece": _ece(rows, bins),
        "mean_predictive_entropy": metrics["entropy"],
        "mean_ood_score": metrics["ood_score"],
        "uncertainty_error_detection_auc": _rank_auc([row.entropy for row in rows], [not bool(row.top1) for row in rows]),
        "ood_error_detection_auc": _rank_auc([row.ood_score for row in rows], [not bool(row.top1) for row in rows]),
        "resource_commitment_mae": optional_metrics["resource_absolute_error"],
        "next_turn_threat_accuracy": optional_metrics["threat_accuracy"],
        "strategic_intent_accuracy": optional_metrics["intent_accuracy"],
    }


def _row_prediction(
    row: DecisionRow,
    probabilities: torch.Tensor,
    resource: torch.Tensor,
    threat: torch.Tensor,
    intent: torch.Tensor,
    context: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> RowPrediction:
    chosen_probability = float(probabilities[row.chosen_action_index])
    prediction = int(probabilities.argmax())
    top_k = min(3, len(row.legal_actions))
    top_indices = set(torch.topk(probabilities, top_k).indices.tolist())
    one_hot = torch.zeros_like(probabilities)
    one_hot[row.chosen_action_index] = 1.0
    chosen = row.legal_actions[row.chosen_action_index]
    predicted = row.legal_actions[prediction]
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    z = (context - center) / scale

    return RowPrediction(
        row_id=row.row_id,
        game_id=row.game_id,
        slices=row.slices,
        nll=-math.log(max(chosen_probability, 1e-12)),
        brier=float(((probabilities - one_hot) ** 2).sum()),
        top1=float(prediction == row.chosen_action_index),
        top3=float(row.chosen_action_index in top_indices),
        family_accuracy=float(chosen.family == predicted.family),
        target_accuracy=float(chosen.target == predicted.target),
        confidence=float(probabilities.max()),
        entropy=entropy,
        ood_score=float(np.sqrt(np.mean(z * z))),
        resource_absolute_error=(
            abs(float(resource) - chosen.resource_commitment)
            if chosen.resource_commitment is not None else None
        ),
        threat_accuracy=(
            float((float(torch.sigmoid(threat)) >= 0.5) == bool(chosen.next_turn_threat))
            if chosen.next_turn_threat is not None else None
        ),
        intent_accuracy=(
            float(int(intent.argmax()) == chosen.strategic_intent)
            if chosen.strategic_intent is not None else None
        ),
        evidence_weight=row.evidence_weight,
    )


def predict_rows_scalar(
    model: OpponentModel,
    rows: Sequence[DecisionRow],
    rung: str,
    temperature: float,
    center: np.ndarray,
    scale: np.ndarray,
) -> list[RowPrediction]:
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            context = context_vector(row, rung, device)
            actions = torch.tensor(
                [action.features for action in row.legal_actions],
                dtype=torch.float32,
                device=device,
            )
            logits, resource, threat, intent = model(context, actions)
            probabilities = torch.softmax(logits / temperature, dim=0)
            predictions.append(_row_prediction(
                row,
                probabilities,
                resource,
                threat,
                intent,
                context.cpu().numpy(),
                center,
                scale,
            ))

    return predictions


def predict_rows_batched(
    model: OpponentModel,
    rows: Sequence[DecisionRow],
    rung: str,
    temperature: float,
    center: np.ndarray,
    scale: np.ndarray,
    budget: FrozenBudget,
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
) -> tuple[list[RowPrediction], dict[str, int]]:
    device = next(model.parameters()).device
    predictions = []
    batch_count = 0
    widest = 0
    model.eval()
    with torch.no_grad():
        for rows_batch in bounded_row_batches(
            rows,
            max_rows=budget.batch_size,
            max_padded_action_cells=max_padded_action_cells,
        ):
            padded = collate_rows(rows_batch, rung, device)
            logits, resource, threat, intent = padded_outputs(model, padded)
            probabilities = torch.softmax(logits / temperature, dim=1).cpu()
            resource = resource.cpu()
            threat = threat.cpu()
            intent = intent.cpu()
            batch_count += 1
            widest = max(widest, padded.actions.shape[1])
            contexts = padded.contexts.cpu().numpy()
            for index, row in enumerate(rows_batch):
                legal_count = len(row.legal_actions)
                predictions.append(_row_prediction(
                    row,
                    probabilities[index, :legal_count],
                    resource[index],
                    threat[index],
                    intent[index],
                    contexts[index],
                    center,
                    scale,
                ))

    return predictions, {
        "vectorized_batches": batch_count,
        "max_padded_legal_width": widest,
        "max_padded_action_cells": max_padded_action_cells,
    }


def evaluate_rung(
    model: OpponentModel,
    rows: Sequence[DecisionRow],
    rung: str,
    temperature: float,
    budget: FrozenBudget,
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
) -> tuple[dict[str, Any], list[RowPrediction]]:
    train_contexts = np.stack([context_vector(row, rung).numpy() for row in rows if row.split == "train"])
    center = train_contexts.mean(axis=0)
    scale = train_contexts.std(axis=0) + 1e-6
    test_rows = [row for row in rows if row.split == "test" and not row.forced]
    started = time.perf_counter()
    predictions, execution = predict_rows_batched(
        model,
        test_rows,
        rung,
        temperature,
        center,
        scale,
        budget,
        max_padded_action_cells,
    )
    elapsed = time.perf_counter() - started
    external_predictions = [
        row for row in predictions if row.slices["population_scope"] == "external_current_population"
    ]
    internal_predictions = [
        row for row in predictions if row.slices["population_scope"] == "internal_panel_adapter"
    ]
    slices: dict[str, dict[str, Any]] = {}
    for field in REQUIRED_SLICE_FIELDS:
        values = sorted({row.slices[field] for row in predictions})
        slices[field] = {
            value: _summary([row for row in predictions if row.slices[field] == value], budget.ece_bins)
            for value in values
        }

    return {
        "aggregate": _summary(external_predictions, budget.ece_bins),
        "slices": slices,
        "heldout": {
            "player": _summary(heldout_slice_rows(predictions, "player"), budget.ece_bins),
            "deck": _summary(heldout_slice_rows(predictions, "deck_family"), budget.ece_bins),
            "player_deck": _summary(heldout_slice_rows(predictions, "player_deck"), budget.ece_bins),
            "bucket": _summary([row for row in predictions if row.slices["bucket"].startswith("heldout:")], budget.ece_bins),
            "internal_panel_adapter": _summary(internal_predictions, budget.ece_bins),
        },
        "calibration_temperature": temperature,
        "execution": execution,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(predictions) / max(elapsed, 1e-12),
    }, predictions


def paired_game_bootstrap(
    om2: Sequence[RowPrediction],
    om3: Sequence[RowPrediction],
    budget: FrozenBudget = FROZEN_BUDGET,
) -> dict[str, Any]:
    om3_by_id = {row.row_id: row for row in om3}
    if {row.row_id for row in om2} != set(om3_by_id):
        raise ValueError("OM2 and OM3 decision identities differ")
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in om2:
        by_game[row.game_id].append(row.nll - om3_by_id[row.row_id].nll)
    games = sorted(by_game)
    deltas = np.asarray([np.mean(by_game[game]) for game in games], dtype=np.float64)
    rng = np.random.default_rng(budget.bootstrap_seed)
    samples = np.asarray([
        deltas[rng.integers(0, len(deltas), size=len(deltas))].mean()
        for _ in range(budget.bootstrap_resamples)
    ])
    low, high = np.quantile(samples, [0.025, 0.975])
    model_objective_delta = float(deltas.mean())
    statistically_positive = bool(low > 0.0)
    materially_positive = model_objective_delta >= budget.material_nll_gain_floor

    return {
        "estimand": "OM2_minus_OM3_game_equal_masked_complete_action_NLL",
        "games": len(games),
        "resamples": budget.bootstrap_resamples,
        "seed": budget.bootstrap_seed,
        "model_objective_delta": model_objective_delta,
        "material_nll_gain_floor": budget.material_nll_gain_floor,
        "confidence_interval_95": [float(low), float(high)],
        "statistically_positive": statistically_positive,
        "materially_positive": materially_positive,
        "OM3_beats_OM2": statistically_positive and materially_positive,
    }


def _paired_slice_deltas(om2: Sequence[RowPrediction], om3: Sequence[RowPrediction], field: str) -> dict[str, float]:
    om3_by_id = {row.row_id: row for row in om3}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in om2:
        grouped[row.slices[field]].append(row.nll - om3_by_id[row.row_id].nll)

    return {value: float(np.mean(deltas)) for value, deltas in sorted(grouped.items())}


def classify_models(
    manifest: Mapping[str, Any],
    om2_report: Mapping[str, Any],
    gate: Mapping[str, Any],
    slice_deltas: Mapping[str, Mapping[str, float]],
) -> dict[str, str]:
    prior = manifest.get("frozen_m1_playstyle_gate")
    heldout_player = om2_report["heldout"]["player"]
    holdouts_complete = all(
        manifest["evaluation_instrumentation"][surface]["status"] == "instrumented"
        for surface in ("cabt_holdout", "reference_policy_holdout")
    )
    if holdouts_complete and isinstance(prior, dict) and prior.get("same_corpus_equal_budget") is True and prior.get("ci_low", 0.0) > 0.0:
        om2 = "PLAYSTYLE_CORE_SUPPORTED"
    elif heldout_player.get("rows", 0) > 0 and math.isfinite(float(om2_report["aggregate"]["game_equal_masked_complete_action_nll"])):
        om2 = "PLAYSTYLE_PROPOSAL_ONLY"
    else:
        om2 = "PLAYSTYLE_NOT_SUPPORTED"
    broad_regression = any(
        delta < 0.0
        for field in ("player", "bucket")
        for delta in slice_deltas[field].values()
    )
    if holdouts_complete and gate["OM3_beats_OM2"] and not broad_regression:
        om3 = "CURRENT_COMPETENCE_CORE_SUPPORTED"
    elif gate["model_objective_delta"] > 0.0:
        om3 = "CURRENT_COMPETENCE_PROPOSAL_ONLY"
    else:
        om3 = "CURRENT_COMPETENCE_NOT_SUPPORTED"

    return {"OM2": om2, "OM3": om3}


def feature_contribution_audit(
    om2_report: Mapping[str, Any],
    om3_report: Mapping[str, Any],
    gate: Mapping[str, Any],
    slice_deltas: Mapping[str, Mapping[str, float]],
    classifications: Mapping[str, str],
    bucket_dim: int,
    synthetic_fixture: bool,
) -> dict[str, Any]:
    negative = {
        field: [value for value, delta in values.items() if delta < 0.0]
        for field, values in slice_deltas.items()
    }
    decision = (
        "diagnostic only" if synthetic_fixture
        else "core path" if classifications["OM3"] == "CURRENT_COMPETENCE_CORE_SUPPORTED"
        else "proposal/ranker only" if classifications["OM3"] == "CURRENT_COMPETENCE_PROPOSAL_ONLY"
        else "quarantine"
    )
    om2_speed = float(om2_report["rows_per_second"])
    om3_speed = float(om3_report["rows_per_second"])
    audit = {
        "Feature": "frozen current external competence-bucket vector",
        "Purpose": "measure whether current competence improves multimodal legal-action prediction beyond public state, deck, and causal playstyle",
        "Baseline": "OM2 with identical architecture, optimizer, budget, seeds, splits, heads, and calibration",
        "Benefit": gate,
        "Cost": {"additional_input_dimensions": bucket_dim, "paid_cost_usd": 0.0},
        "Damage": {"negative_slices": negative},
        "Uniqueness": "OM3 appends only the manifest-declared external bucket vector; player/checkpoint identity remains excluded",
        "Precision": {"nonregressed_slice_fraction": float(np.mean([delta >= 0.0 for values in slice_deltas.values() for delta in values.values()]))},
        "Recall": {"slice_families_measured": sorted(slice_deltas)},
        "Mean true objective delta": "not_measured_no_terminal_gameplay_evaluation",
        "Throughput impact": {"OM2_rows_per_second": om2_speed, "OM3_rows_per_second": om3_speed, "OM3_minus_OM2": om3_speed - om2_speed},
        "Marginal value/sec": "not_measured_no_terminal_gameplay_objective",
        "Failure clustering": negative,
        "Decision": decision,
        "Next action": (
            "discard synthetic verdict and run once on the frozen real corpus"
            if synthetic_fixture
            else "close the opponent-model lane and retain the earned classification without retuning"
        ),
    }
    if audit["Decision"] not in AUDIT_DECISIONS:
        raise AssertionError("invalid feature-contribution classification")

    return audit


def resolve_execution(
    device: str,
    max_padded_action_cells: int,
) -> tuple[torch.device, dict[str, Any]]:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if max_padded_action_cells <= 0:
        raise ValueError("max_padded_action_cells must be positive")
    torch_device = torch.device(resolved_device)
    torch.use_deterministic_algorithms(True)
    provenance = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": resolved_device,
        "device_name": torch.cuda.get_device_name(torch_device) if resolved_device == "cuda" else "CPU",
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "masked_batch_implementation": "deepmew.om_padded_masked_complete_action.v1",
        "max_padded_action_cells": max_padded_action_cells,
    }

    return torch_device, provenance


def _load_finite_checkpoint_model(
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    rung: str,
    budget: FrozenBudget,
    device: torch.device,
) -> tuple[OpponentModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != RESULT_SCHEMA:
        raise CorpusContractError(f"unexpected checkpoint schema: {checkpoint_path}")
    if payload.get("rung") != rung or payload.get("manifest_sha256") != manifest_sha256:
        raise CorpusContractError(f"checkpoint rung or corpus binding mismatch: {checkpoint_path}")
    if payload.get("budget") != asdict(budget) or payload.get("feature_schema") != manifest["feature_schema"]:
        raise CorpusContractError(f"checkpoint scientific configuration mismatch: {checkpoint_path}")
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state:
        raise CorpusContractError(f"checkpoint has no model state: {checkpoint_path}")
    nonfinite = {
        name: int((~torch.isfinite(tensor)).sum())
        for name, tensor in state.items()
        if not torch.isfinite(tensor).all()
    }
    if nonfinite:
        raise CorpusContractError(f"checkpoint contains non-finite model tensors: {nonfinite}")
    model = build_model(manifest["feature_schema"], rung, budget, device)
    model.load_state_dict(state, strict=True)

    stored_temperature = payload.get("temperature")
    stored_temperature_status = (
        "NONFINITE_IGNORED"
        if isinstance(stored_temperature, (int, float)) and not math.isfinite(stored_temperature)
        else "FINITE_IGNORED"
    )

    return model, {
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "stored_temperature_status": stored_temperature_status,
        "stored_temperature_ignored": True,
        "model_tensors_finite": True,
    }


def reevaluate_existing_checkpoints(
    manifest_path: Path,
    source_failed_result_path: Path,
    checkpoint_paths: Mapping[str, Path],
    *,
    budget: FrozenBudget = FROZEN_BUDGET,
    device: str = "cpu",
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    manifest, rows = load_corpus(manifest_path, allow_synthetic=allow_synthetic)
    manifest_hash = file_sha256(manifest_path.resolve())
    source = _read_json(source_failed_result_path.resolve())
    if source.get("schema_version") != RESULT_SCHEMA or source.get("manifest_sha256") != manifest_hash:
        raise CorpusContractError("failed result does not bind the requested corpus and result schema")
    torch_device, execution_provenance = resolve_execution(device, max_padded_action_cells)
    rung_results: dict[str, dict[str, Any]] = {}
    rung_predictions: dict[str, list[RowPrediction]] = {}
    for rung in ("OM2", "OM3"):
        checkpoint_path = checkpoint_paths[rung].resolve()
        model, checkpoint = _load_finite_checkpoint_model(
            checkpoint_path,
            manifest,
            manifest_hash,
            rung,
            budget,
            torch_device,
        )
        source_checkpoint = source["rungs"][rung].get("checkpoint", {})
        if source_checkpoint.get("sha256") != checkpoint["sha256"]:
            raise CorpusContractError(f"failed result checkpoint hash mismatch for {rung}")
        temperature = _fit_temperature(model, rows, rung, budget, max_padded_action_cells)
        if not math.isfinite(temperature):
            raise RuntimeError(f"posthoc calibration remained non-finite for {rung}")
        evaluation, predictions = evaluate_rung(
            model,
            rows,
            rung,
            temperature,
            budget,
            max_padded_action_cells,
        )
        rung_results[rung] = {
            "inputs": source["rungs"][rung]["inputs"],
            "train": source["rungs"][rung]["train"],
            "training_reused_not_rerun": True,
            "evaluation": evaluation,
            "model_parameter_count": source["rungs"][rung]["model_parameter_count"],
            "checkpoint": checkpoint,
        }
        rung_predictions[rung] = predictions
    external_om2 = [
        row for row in rung_predictions["OM2"]
        if row.slices["population_scope"] == "external_current_population"
    ]
    external_om3 = [
        row for row in rung_predictions["OM3"]
        if row.slices["population_scope"] == "external_current_population"
    ]
    gate = paired_game_bootstrap(external_om2, external_om3, budget)
    slice_deltas = {
        field: _paired_slice_deltas(rung_predictions["OM2"], rung_predictions["OM3"], field)
        for field in ("player", "deck_family", "player_deck", "bucket")
    }
    classifications = classify_models(manifest, rung_results["OM2"]["evaluation"], gate, slice_deltas)
    audit = feature_contribution_audit(
        rung_results["OM2"]["evaluation"],
        rung_results["OM3"]["evaluation"],
        gate,
        slice_deltas,
        classifications,
        int(manifest["feature_schema"]["bucket_dim"]),
        bool(manifest.get("synthetic_fixture")),
    )

    return {
        "schema_version": RESULT_SCHEMA,
        "status": "COMPLETE_POSTHOC_RECALIBRATION_EXISTING_CHECKPOINTS",
        "created_at_utc": utc_now(),
        "source_failed_result": {
            "path": str(source_failed_result_path.resolve()),
            "sha256": file_sha256(source_failed_result_path.resolve()),
            "preserved_immutable": True,
        },
        "recovery_scope": "calibration_and_evaluation_only_no_training_no_optimizer",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "equal_budget_contract": asdict(budget),
        "operational_device": execution_provenance["device"],
        "execution_provenance": execution_provenance,
        "only_intended_difference": source["only_intended_difference"],
        "x_access": source["x_access"],
        "rungs": rung_results,
        "evaluation_instrumentation": manifest["evaluation_instrumentation"],
        "primary_gate": gate,
        "paired_slices": slice_deltas,
        "classifications": classifications,
        "feature_contribution_audit": audit,
        "forced_rows_actor_loss_zero": True,
        "paid_jobs": 0,
        "protected_runtime_wiring": False,
    }


def run_experiment(
    manifest_path: Path,
    *,
    allow_synthetic: bool = False,
    budget: FrozenBudget = FROZEN_BUDGET,
    device: str = "cpu",
    artifact_dir: Path | None = None,
    max_padded_action_cells: int = DEFAULT_MAX_PADDED_ACTION_CELLS,
) -> dict[str, Any]:
    manifest, rows = load_corpus(manifest_path, allow_synthetic=allow_synthetic)
    synthetic_fixture = bool(manifest.get("synthetic_fixture"))
    torch_device, execution_provenance = resolve_execution(device, max_padded_action_cells)
    rung_results: dict[str, dict[str, Any]] = {}
    rung_predictions: dict[str, list[RowPrediction]] = {}
    for rung in ("OM2", "OM3"):
        model, train = train_rung(
            rows,
            manifest["feature_schema"],
            rung,
            budget,
            torch_device,
            max_padded_action_cells,
        )
        temperature = _fit_temperature(model, rows, rung, budget, max_padded_action_cells)
        evaluation, predictions = evaluate_rung(
            model,
            rows,
            rung,
            temperature,
            budget,
            max_padded_action_cells,
        )
        rung_results[rung] = {
            "inputs": "public_state+legal_actions+deck+causal_playstyle" + ("+external_bucket" if rung == "OM3" else ""),
            "train": train,
            "evaluation": evaluation,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = artifact_dir / f"{rung}_MODEL.pt"
            if checkpoint_path.exists():
                raise FileExistsError(f"refusing to overwrite immutable checkpoint: {checkpoint_path}")
            torch.save({
                "schema_version": RESULT_SCHEMA,
                "rung": rung,
                "manifest_sha256": file_sha256(manifest_path.resolve()),
                "budget": asdict(budget),
                "feature_schema": manifest["feature_schema"],
                "temperature": temperature,
                "execution_provenance": execution_provenance,
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            }, checkpoint_path)
            rung_results[rung]["checkpoint"] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": file_sha256(checkpoint_path),
            }
        rung_predictions[rung] = predictions
    external_om2 = [row for row in rung_predictions["OM2"] if row.slices["population_scope"] == "external_current_population"]
    external_om3 = [row for row in rung_predictions["OM3"] if row.slices["population_scope"] == "external_current_population"]
    gate = paired_game_bootstrap(external_om2, external_om3, budget)
    slice_deltas = {
        field: _paired_slice_deltas(rung_predictions["OM2"], rung_predictions["OM3"], field)
        for field in ("player", "deck_family", "player_deck", "bucket")
    }
    classifications = classify_models(manifest, rung_results["OM2"]["evaluation"], gate, slice_deltas)
    audit = feature_contribution_audit(
        rung_results["OM2"]["evaluation"],
        rung_results["OM3"]["evaluation"],
        gate,
        slice_deltas,
        classifications,
        int(manifest["feature_schema"]["bucket_dim"]),
        synthetic_fixture,
    )

    return {
        "schema_version": RESULT_SCHEMA,
        "status": "COMPLETE_SYNTHETIC_CONTRACT_TEST" if synthetic_fixture else "COMPLETE_FROZEN_REAL_CORPUS",
        "created_at_utc": utc_now(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path.resolve()),
        "equal_budget_contract": asdict(budget),
        "operational_device": execution_provenance["device"],
        "execution_provenance": execution_provenance,
        "only_intended_difference": "OM3 appends model_inputs.bucket_features; all other inputs and procedures are identical",
        "x_access": "read_only_coordinate_reuse_no_tensor_duplication",
        "rungs": rung_results,
        "evaluation_instrumentation": manifest["evaluation_instrumentation"],
        "primary_gate": gate,
        "paired_slices": slice_deltas,
        "classifications": classifications,
        "feature_contribution_audit": audit,
        "forced_rows_actor_loss_zero": True,
        "paid_jobs": 0,
        "protected_runtime_wiring": False,
    }


def write_result(output: Path, result: Mapping[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute-frozen-corpus", action="store_true")
    parser.add_argument("--recalibrate-existing-checkpoints", action="store_true")
    parser.add_argument("--source-failed-result", type=Path)
    parser.add_argument("--om2-checkpoint", type=Path)
    parser.add_argument("--om3-checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-padded-action-cells",
        type=int,
        default=DEFAULT_MAX_PADDED_ACTION_CELLS,
        help="operational memory bound; microbatching preserves the frozen optimizer batch",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = sum((args.validate_only, args.execute_frozen_corpus, args.recalibrate_existing_checkpoints))
    if modes > 1:
        raise SystemExit("choose only one execution mode")
    manifest = validate_manifest(args.manifest, allow_synthetic=False)
    if args.recalibrate_existing_checkpoints:
        required = (args.output, args.source_failed_result, args.om2_checkpoint, args.om3_checkpoint)
        if any(value is None for value in required):
            raise SystemExit(
                "posthoc recalibration requires --output, --source-failed-result, "
                "--om2-checkpoint, and --om3-checkpoint"
            )
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite immutable result: {args.output}")
        result = reevaluate_existing_checkpoints(
            args.manifest,
            args.source_failed_result,
            {"OM2": args.om2_checkpoint, "OM3": args.om3_checkpoint},
            device=args.device,
            max_padded_action_cells=args.max_padded_action_cells,
        )
        write_result(args.output, result)

        return 0
    if not args.execute_frozen_corpus:
        print(json.dumps({
            "status": "VALIDATED_NOT_TRAINED",
            "manifest": str(args.manifest.resolve()),
            "rows": manifest.get("rows"),
            "next_action": "rerun with --execute-frozen-corpus only after the corpus owner freezes the manifest",
        }, sort_keys=True))

        return 0
    if args.output is None:
        raise SystemExit("--output is required with --execute-frozen-corpus")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable result: {args.output}")
    result = run_experiment(
        args.manifest,
        device=args.device,
        artifact_dir=args.output.parent,
        max_padded_action_cells=args.max_padded_action_cells,
    )
    write_result(args.output, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
