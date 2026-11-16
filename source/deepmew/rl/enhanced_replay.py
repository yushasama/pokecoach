"""Bounded, hash-bound replay rows for the enhanced research trainer.

This module deliberately stops at the data boundary.  It selects accepted
training rows, resolves their immutable canonical/source evidence, and returns
actor-visible records.  Model feature encoding and trainer orchestration live
elsewhere.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from deepmew.replay.canonical_actor_dataset import (
    S1_SLICE,
    TRANSITION_SCHEMA_VERSION,
    UNIT_MANIFEST_SCHEMA_VERSION,
    canonical_actor_feature_observation,
    validate_transition_record,
)
from deepmew.replay.information_boundary import (
    InformationMode,
    assert_deployable_observation,
)
from deepmew.value_lab.data import LabData


SOURCE_OBSERVATION_FIELD_BY_SCHEMA: Mapping[str, Mapping[str, str]] = {
    # The canonical converter names the semantic reference raw_observation,
    # while this source schema persists the same actor observation as state_json.
    "deepmew.replay_action_row.v1": {"raw_observation": "state_json"},
}


class EnhancedReplayError(ValueError):
    """Raised when bounded replay evidence cannot be proven safe and exact."""


@dataclass(frozen=True)
class ReplayRecord:
    """One actor-visible replay decision with its accepted S1 prefix."""

    global_row: int
    source_id: str
    unit_id: str
    part: str
    local_row: int
    decision_id: str
    actor: int
    s1: np.ndarray
    actor_visible_observation: dict[str, Any]
    legal_set: tuple[dict[str, Any], ...]
    chosen_selection: dict[str, Any]
    policy_id: str | None
    opponent_policy_id: str | None
    deck_identity: str | None
    opponent_deck_identity: str | None
    source_observation_sha256: str


@dataclass(frozen=True)
class ReplayCoverage:
    """Selection and identity coverage for a bounded replay load."""

    requested_rows: int
    eligible_train_rows_available: int
    selected_rows: int
    unique_decision_ids: int
    exact_d2_deck_identity: str
    exact_d2_actor_rows: int
    exact_d2_opponent_rows: int
    source_rows: Mapping[str, int]


@dataclass(frozen=True)
class BoundedReplayDataset:
    """Immutable bounded records plus their auditable coverage receipt."""

    records: tuple[ReplayRecord, ...]
    coverage: ReplayCoverage
    cache_manifest_sha256: str
    split_manifest_sha256: str


class ReplayBatchMaterializer:
    """Resolve one bounded set of stream identities through accepted evidence."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split_manifest_path: str | Path,
        exact_deck_identity: str,
        admission_manifest_path: str | Path | None = None,
    ) -> None:
        if not isinstance(exact_deck_identity, str) or not exact_deck_identity:
            raise EnhancedReplayError("exact_deck_identity must be explicit")
        self.cache_path = _resolved_path(cache_dir)
        self.split_path = _resolved_path(split_manifest_path)
        try:
            self.data = LabData(
                str(self.cache_path),
                split_manifest_path=str(self.split_path),
                load_labels=False,
            )
        except (OSError, ValueError) as exc:
            raise EnhancedReplayError("replay cache is not an accepted label-blind LabData binding") from exc
        canonical_root, _, inputs = _validate_cache_inputs(self.data, self.split_path)
        _, catalog_units = _load_catalog_units(self.data, canonical_root)
        self.bindings = _build_part_bindings(self.data, canonical_root, inputs, catalog_units)
        self.exact_deck_identity = exact_deck_identity
        self.train_index = self.data.split_names.index("train")
        self.cache_manifest_sha256 = _sha256_path(self.cache_path / "cache_manifest.json")
        self.split_manifest_sha256 = _sha256_path(self.split_path)
        self.admission_index = None
        if admission_manifest_path is not None:
            from deepmew.rl.enhanced_replay_admission import AdmittedObservationIndex

            self.admission_index = AdmittedObservationIndex(admission_manifest_path)

    def load(self, global_rows: Sequence[int]) -> tuple[ReplayRecord, ...]:
        if not global_rows:
            raise EnhancedReplayError("stream replay batch must contain rows")
        selected = np.asarray(global_rows, dtype=np.int64)
        if len(np.unique(selected)) != len(selected):
            raise EnhancedReplayError("stream replay batch contains duplicate global rows")
        if bool(((selected < 0) | (selected >= len(self.data.meta["split_idx"]))).any()):
            raise EnhancedReplayError("stream replay batch row is outside the accepted cache")
        if not np.all(self.data.meta["split_idx"][selected] == self.train_index):
            raise EnhancedReplayError("stream replay batch escaped the accepted train split")
        if not np.all(self.data.meta["is_target_eligible"][selected]):
            raise EnhancedReplayError("stream replay batch contains a rejected target")
        transition_provider = None
        if self.admission_index is not None:
            transition_provider = lambda binding, rows: self.admission_index.read_transitions(
                f"{binding.source_id}:{binding.unit_id}:{binding.part}", rows
            )
        pending = _pending_records(
            self.data,
            self.bindings,
            selected,
            transition_provider=transition_provider,
        )
        if self.admission_index is None:
            source_rows = _source_observations(pending)
        else:
            requests = []
            for item in pending:
                source_sha256 = _expected_source_hash(item.binding, item.source_path, item.transition)
                requests.append((source_sha256, item.source_line, item.source_observation_sha256))
            admitted = self.admission_index.read(requests)
            source_rows = {}
            for item in pending:
                source_sha256 = _expected_source_hash(item.binding, item.source_path, item.transition)
                source_rows[(_path_key(item.source_path), item.source_line)] = admitted[
                    (source_sha256, item.source_line)
                ]
        records = []
        for item in pending:
            source_row = source_rows[(_path_key(item.source_path), item.source_line)]
            observation = _observation_from_source(item, source_row)
            records.append(ReplayRecord(
                global_row=item.global_row,
                source_id=item.binding.source_id,
                unit_id=item.binding.unit_id,
                part=item.binding.part,
                local_row=item.local_row,
                decision_id=str(item.transition["decision_id"]),
                actor=int(item.transition["actor"]),
                s1=_immutable_s1(self.data, item.global_row),
                actor_visible_observation=observation,
                legal_set=tuple(copy.deepcopy(item.transition["legal_action_set"])),
                chosen_selection=copy.deepcopy(item.transition["chosen_action"]),
                policy_id=item.policy_id,
                opponent_policy_id=item.opponent_policy_id,
                deck_identity=item.deck_identity,
                opponent_deck_identity=item.opponent_deck_identity,
                source_observation_sha256=item.source_observation_sha256,
            ))
        if len({record.decision_id for record in records}) != len(records):
            raise EnhancedReplayError("stream replay batch decision identities are duplicated")

        return tuple(records)


@dataclass(frozen=True)
class _PartBinding:
    source_id: str
    unit_id: str
    part: str
    unit_rank: int
    row_start: int
    row_end: int
    input_receipt: Mapping[str, Any]
    unit_manifest_path: Path
    unit_manifest: Mapping[str, Any]
    transition_path: Path


@dataclass(frozen=True)
class _PendingRecord:
    global_row: int
    local_row: int
    binding: _PartBinding
    transition: Mapping[str, Any]
    policy_id: str | None
    opponent_policy_id: str | None
    deck_identity: str | None
    opponent_deck_identity: str | None
    source_path: Path
    source_line: int
    source_field: str
    source_observation_sha256: str


def _sha256_path(path: str | Path, chunk_bytes: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnhancedReplayError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise EnhancedReplayError(f"expected JSON object: {path}")

    return value


def _resolved_path(path: str | Path) -> Path:
    return Path(path).resolve()


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EnhancedReplayError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise EnhancedReplayError(f"{field} must be a SHA-256 hex digest") from exc

    return value.lower()


def _vocab_value(vocab: list[str], index: Any, *, field: str) -> str | None:
    if isinstance(index, bool):
        raise EnhancedReplayError(f"{field} index is boolean")
    try:
        integer = int(index)
    except (TypeError, ValueError) as exc:
        raise EnhancedReplayError(f"{field} index is invalid") from exc
    if integer == -1:
        return None
    if integer < 0 or integer >= len(vocab):
        raise EnhancedReplayError(f"{field} index is outside the accepted vocabulary")
    value = vocab[integer]

    return None if value == "None" else str(value)


def _validate_cache_inputs(data: LabData, split_manifest_path: Path) -> tuple[Path, dict[str, Any], list[dict]]:
    manifest = data.manifest
    if data.has_labels or "z" in data.meta or "z" in data.blocks:
        raise EnhancedReplayError("enhanced replay requires label-blind LabData")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise EnhancedReplayError("cache input inventory is missing")
    if _stable_hash(inputs) != str(manifest.get("input_bindings_sha256", "")).lower():
        raise EnhancedReplayError("cache input binding digest is not reproducible")
    if len(inputs) != int(manifest.get("atomic_parts", -1)):
        raise EnhancedReplayError("cache input inventory does not cover every atomic part")
    if sum(int(receipt.get("rows", -1)) for receipt in inputs) != int(manifest.get("rows", -1)):
        raise EnhancedReplayError("cache input rows do not cover the cache")
    identities = [
        (receipt.get("source_id"), receipt.get("unit_id"), receipt.get("part"))
        for receipt in inputs
    ]
    if any(not all(isinstance(value, str) and value for value in identity) for identity in identities):
        raise EnhancedReplayError("cache input identity is invalid")
    if len(set(identities)) != len(identities):
        raise EnhancedReplayError("cache input identities are duplicated")

    split = _read_json_object(split_manifest_path)
    canonical_root = _resolved_path(manifest.get("canonical_root", ""))
    if not canonical_root.is_dir():
        raise EnhancedReplayError("cache canonical root is missing")
    if _path_key(split.get("canonical_root", "")) != _path_key(canonical_root):
        raise EnhancedReplayError("cache and split canonical roots differ")
    if str(split.get("catalog_sha256", "")).lower() != str(manifest.get("catalog_sha256", "")).lower():
        raise EnhancedReplayError("cache and split catalog bindings differ")

    return canonical_root, split, inputs


def _load_catalog_units(
    data: LabData,
    canonical_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = data.manifest
    catalog_path = _resolved_path(manifest.get("catalog_path", ""))
    expected_catalog_sha = _require_hash(manifest.get("catalog_sha256"), field="cache catalog_sha256")
    if not catalog_path.is_file() or _sha256_path(catalog_path) != expected_catalog_sha:
        raise EnhancedReplayError("cache catalog hash mismatch")
    catalog = _read_json_object(catalog_path)
    if _path_key(catalog.get("out_root", "")) != _path_key(canonical_root):
        raise EnhancedReplayError("catalog out_root differs from the accepted canonical root")
    units = catalog.get("included_units")
    if not isinstance(units, list) or not units:
        raise EnhancedReplayError("catalog included-unit allowlist is missing")
    if len(units) != int(manifest.get("included_units", -1)):
        raise EnhancedReplayError("cache/catalog included-unit count mismatch")

    return catalog, units


def _unit_manifest_for_receipt(
    receipt: Mapping[str, Any],
    *,
    canonical_root: Path,
    catalog_units: list[dict[str, Any]],
    cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    try:
        unit_rank = int(receipt["unit_rank"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EnhancedReplayError("cache input unit_rank is invalid") from exc
    if unit_rank < 0 or unit_rank >= len(catalog_units):
        raise EnhancedReplayError("cache input unit_rank is outside the catalog allowlist")
    catalog_unit = catalog_units[unit_rank]
    identity = (str(receipt["source_id"]), str(receipt["unit_id"]))
    if (catalog_unit.get("source_id"), catalog_unit.get("unit_id")) != identity:
        raise EnhancedReplayError("cache input unit identity differs from catalog rank")
    if identity in cache:
        return cache[identity]

    manifest_path = canonical_root / str(catalog_unit.get("manifest_path", ""))
    expected_sha = _require_hash(catalog_unit.get("manifest_sha256"), field="catalog unit manifest SHA")
    if str(receipt.get("unit_manifest_sha256", "")).lower() != expected_sha:
        raise EnhancedReplayError("cache/catalog unit-manifest bindings differ")
    if not manifest_path.is_file() or _sha256_path(manifest_path) != expected_sha:
        raise EnhancedReplayError("unit manifest hash mismatch")
    unit_manifest = _read_json_object(manifest_path)
    if unit_manifest.get("schema_version") != UNIT_MANIFEST_SCHEMA_VERSION:
        raise EnhancedReplayError("unit manifest schema is not current")
    if unit_manifest.get("status") != "completed" or unit_manifest.get("information_mode") != "actor_visible":
        raise EnhancedReplayError("unit manifest is not completed actor-visible evidence")
    if (unit_manifest.get("source_id"), unit_manifest.get("unit_id")) != identity:
        raise EnhancedReplayError("unit manifest identity mismatch")
    cache[identity] = (manifest_path, unit_manifest)

    return manifest_path, unit_manifest


def _build_part_bindings(
    data: LabData,
    canonical_root: Path,
    inputs: list[dict],
    catalog_units: list[dict[str, Any]],
) -> tuple[_PartBinding, ...]:
    unit_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    bindings: list[_PartBinding] = []
    row_start = 0
    for receipt in inputs:
        rows = int(receipt.get("rows", -1))
        if rows < 1:
            raise EnhancedReplayError("cache input part has no rows")
        manifest_path, unit_manifest = _unit_manifest_for_receipt(
            receipt,
            canonical_root=canonical_root,
            catalog_units=catalog_units,
            cache=unit_cache,
        )
        part_name = str(receipt["part"])
        matching = [part for part in unit_manifest.get("parts", []) if part.get("name") == part_name]
        if len(matching) != 1:
            raise EnhancedReplayError("unit manifest does not uniquely bind the cache part")
        part = matching[0]
        if int(part.get("rows", -1)) != rows:
            raise EnhancedReplayError("cache/unit part row counts differ")
        transition_name = f"{part_name}.transitions.jsonl.gz"
        expected = _require_hash(receipt.get("transitions_sha256"), field="transition receipt SHA")
        part_hash = str((part.get("files_sha256") or {}).get(transition_name, "")).lower()
        if part_hash != expected:
            raise EnhancedReplayError("cache/unit transition bindings differ")
        transition_path = manifest_path.parent / transition_name
        bindings.append(_PartBinding(
            source_id=str(receipt["source_id"]),
            unit_id=str(receipt["unit_id"]),
            part=part_name,
            unit_rank=int(receipt["unit_rank"]),
            row_start=row_start,
            row_end=row_start + rows,
            input_receipt=receipt,
            unit_manifest_path=manifest_path,
            unit_manifest=unit_manifest,
            transition_path=transition_path,
        ))
        row_start += rows
    if row_start != len(data.meta["split_idx"]):
        raise EnhancedReplayError("cache part bindings do not cover every global row")

    return tuple(bindings)


def _select_train_rows(data: LabData, *, max_rows: int, seed: int) -> np.ndarray:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise EnhancedReplayError("max_rows must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EnhancedReplayError("seed must be an integer")
    try:
        train_index = data.split_names.index("train")
    except ValueError as exc:
        raise EnhancedReplayError("accepted cache has no train fold") from exc
    eligible = np.flatnonzero(
        (data.meta["split_idx"] == train_index) & data.meta["is_target_eligible"]
    ).astype(np.int64, copy=False)
    if max_rows > len(eligible):
        raise EnhancedReplayError(
            f"requested {max_rows} replay rows but only {len(eligible)} eligible train rows exist"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    positions = generator.choice(len(eligible), size=max_rows, replace=False)
    selected = np.sort(eligible[positions])
    if len(np.unique(selected)) != max_rows:
        raise EnhancedReplayError("deterministic replay selection contains duplicate rows")
    if not np.all(data.meta["split_idx"][selected] == train_index):
        raise EnhancedReplayError("replay selection escaped the train fold")
    if not np.all(data.meta["is_target_eligible"][selected]):
        raise EnhancedReplayError("replay selection contains excluded targets")

    return selected


def _binding_indices(bindings: tuple[_PartBinding, ...], selected: np.ndarray) -> np.ndarray:
    ends = np.asarray([binding.row_end for binding in bindings], dtype=np.int64)
    indices = np.searchsorted(ends, selected, side="right")
    if np.any(indices >= len(bindings)):
        raise EnhancedReplayError("selected global row is outside the part bindings")
    for global_row, binding_index in zip(selected.tolist(), indices.tolist()):
        binding = bindings[binding_index]
        if global_row < binding.row_start or global_row >= binding.row_end:
            raise EnhancedReplayError("selected global row does not map to exactly one part")

    return indices


def _block_identity(data: LabData, global_row: int) -> tuple[int, str | None, str | None, str | None, str | None]:
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    ends = starts + data.blocks["row_len"].astype(np.int64, copy=False)
    block_id = int(np.searchsorted(ends, global_row, side="right"))
    if block_id >= len(starts) or global_row < starts[block_id]:
        raise EnhancedReplayError("selected row does not map to an accepted game block")
    policy_id = _vocab_value(data.policy_vocab, data.blocks["policy_gidx"][block_id], field="policy")
    opponent_policy_id = _vocab_value(
        data.policy_vocab,
        data.blocks["opp_policy_gidx"][block_id],
        field="opponent policy",
    )
    deck_identity = _vocab_value(data.deck_vocab, data.blocks["deck_gidx"][block_id], field="deck")
    opponent_deck_identity = _vocab_value(
        data.deck_vocab,
        data.blocks["opp_deck_gidx"][block_id],
        field="opponent deck",
    )

    return block_id, policy_id, opponent_policy_id, deck_identity, opponent_deck_identity


def _read_selected_transitions(path: Path, local_rows: set[int], expected_sha256: str) -> dict[int, dict[str, Any]]:
    if not path.is_file() or _sha256_path(path) != expected_sha256:
        raise EnhancedReplayError(f"transition sidecar hash mismatch: {path}")
    selected: dict[int, dict[str, Any]] = {}
    maximum = max(local_rows)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number > maximum and len(selected) == len(local_rows):
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EnhancedReplayError(f"invalid transition JSON: {path}:{line_number + 1}") from exc
                if line_number not in local_rows:
                    continue
                if not isinstance(record, dict) or record.get("row") != line_number:
                    raise EnhancedReplayError("transition sidecar row index is not canonical")
                selected[line_number] = record
    except OSError as exc:
        raise EnhancedReplayError(f"cannot read transition sidecar: {path}") from exc
    missing = sorted(local_rows - set(selected))
    if missing:
        raise EnhancedReplayError(f"transition sidecar is missing selected rows: {missing[:4]}")

    return selected


def _expected_source_hash(binding: _PartBinding, source_path: Path, transition: Mapping[str, Any]) -> str:
    source_hashes = binding.unit_manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise EnhancedReplayError("unit manifest lacks source-file hash bindings")
    by_path = {_path_key(path): value for path, value in source_hashes.items()}
    expected = by_path.get(_path_key(source_path))
    if expected is None:
        raise EnhancedReplayError("source observation path is absent from the unit manifest")
    expected = _require_hash(expected, field="unit source SHA")
    if str(transition.get("source_hash", "")).lower() != expected:
        raise EnhancedReplayError("transition source hash differs from the unit manifest")

    return expected


def _pending_records(
    data: LabData,
    bindings: tuple[_PartBinding, ...],
    selected: np.ndarray,
    transition_provider: Callable[[_PartBinding, set[int]], Mapping[int, Mapping[str, Any]]] | None = None,
) -> tuple[_PendingRecord, ...]:
    binding_indices = _binding_indices(bindings, selected)
    by_binding: dict[int, set[int]] = {}
    for global_row, binding_index in zip(selected.tolist(), binding_indices.tolist()):
        binding = bindings[binding_index]
        by_binding.setdefault(binding_index, set()).add(global_row - binding.row_start)
    transitions: dict[tuple[int, int], dict[str, Any]] = {}
    for binding_index, local_rows in by_binding.items():
        binding = bindings[binding_index]
        expected_sha = _require_hash(
            binding.input_receipt.get("transitions_sha256"),
            field="transition receipt SHA",
        )
        loaded = (
            _read_selected_transitions(binding.transition_path, local_rows, expected_sha)
            if transition_provider is None
            else transition_provider(binding, local_rows)
        )
        for local_row, transition in loaded.items():
            transitions[(binding_index, local_row)] = transition

    pending: list[_PendingRecord] = []
    seen_decision_ids: set[str] = set()
    for global_row, binding_index in zip(selected.tolist(), binding_indices.tolist()):
        binding = bindings[binding_index]
        local_row = global_row - binding.row_start
        transition = transitions[(binding_index, local_row)]
        try:
            validate_transition_record(dict(transition))
        except ValueError as exc:
            raise EnhancedReplayError(f"invalid canonical transition at global row {global_row}") from exc
        if transition.get("schema_version") != TRANSITION_SCHEMA_VERSION:
            raise EnhancedReplayError("transition schema is not current")
        if (transition.get("source_id"), transition.get("part"), transition.get("row")) != (
            binding.source_id,
            int(binding.part.rsplit("_", 1)[-1]),
            local_row,
        ):
            raise EnhancedReplayError("transition source/part/row identity mismatch")
        block_id, policy_id, opponent_policy_id, deck_identity, opponent_deck_identity = _block_identity(
            data,
            global_row,
        )
        canonical_game_id = data.block_keys[block_id].rsplit(":s", 1)[0]
        if transition.get("canonical_game_id") != canonical_game_id:
            raise EnhancedReplayError("transition canonical game differs from the accepted block")
        if int(transition.get("actor", -1)) != int(data.meta["seat"][global_row]):
            raise EnhancedReplayError("transition actor differs from accepted metadata")
        if int(transition.get("global_decision_index", -1)) != int(data.meta["decision_index"][global_row]):
            raise EnhancedReplayError("transition decision index differs from accepted metadata")
        if len(transition["legal_action_set"]) != int(data.meta["legal_count"][global_row]):
            raise EnhancedReplayError("transition legal count differs from accepted metadata")
        if transition.get("policy_id") != policy_id:
            raise EnhancedReplayError("transition policy differs from accepted block identity")
        decision_id = transition.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id or decision_id in seen_decision_ids:
            raise EnhancedReplayError("selected transition decision IDs are missing or duplicated")
        seen_decision_ids.add(decision_id)
        state_before = transition.get("state_before")
        reference = state_before.get("source_observation") if isinstance(state_before, dict) else None
        if not isinstance(reference, dict):
            raise EnhancedReplayError("transition lacks a source-observation reference")
        source_path_value = reference.get("source_path")
        if not isinstance(source_path_value, str) or not source_path_value or not Path(source_path_value).is_absolute():
            raise EnhancedReplayError("source-observation path must be absolute")
        source_path = Path(source_path_value)
        try:
            source_line = int(reference.get("source_line"))
        except (TypeError, ValueError) as exc:
            raise EnhancedReplayError("source-observation line is invalid") from exc
        if source_line < 1:
            raise EnhancedReplayError("source-observation line must be one-based")
        source_field = reference.get("source_field")
        if not isinstance(source_field, str) or not source_field:
            raise EnhancedReplayError("source-observation field is missing")
        observation_sha = _require_hash(
            reference.get("content_sha256"),
            field="source-observation content SHA",
        )
        _expected_source_hash(binding, source_path, transition)
        pending.append(_PendingRecord(
            global_row=global_row,
            local_row=local_row,
            binding=binding,
            transition=transition,
            policy_id=policy_id,
            opponent_policy_id=opponent_policy_id,
            deck_identity=deck_identity,
            opponent_deck_identity=opponent_deck_identity,
            source_path=source_path,
            source_line=source_line,
            source_field=source_field,
            source_observation_sha256=observation_sha,
        ))

    return tuple(pending)


def _open_source(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")

    return open(path, "rt", encoding="utf-8")


def _read_source_lines(path: Path, line_numbers: set[int], expected_sha256: str) -> dict[int, dict[str, Any]]:
    if not path.is_file() or _sha256_path(path) != expected_sha256:
        raise EnhancedReplayError(f"source file hash mismatch: {path}")
    maximum = max(line_numbers)
    rows: dict[int, dict[str, Any]] = {}
    try:
        with _open_source(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number > maximum and len(rows) == len(line_numbers):
                    break
                if line_number not in line_numbers:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EnhancedReplayError(f"invalid source JSON: {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise EnhancedReplayError(f"source row is not an object: {path}:{line_number}")
                rows[line_number] = row
    except OSError as exc:
        raise EnhancedReplayError(f"cannot read source observations: {path}") from exc
    missing = sorted(line_numbers - set(rows))
    if missing:
        raise EnhancedReplayError(f"source file is missing referenced lines: {missing[:4]}")

    return rows


def _source_observations(pending: tuple[_PendingRecord, ...]) -> dict[tuple[str, int], dict[str, Any]]:
    requests: dict[str, tuple[Path, str, set[int]]] = {}
    for item in pending:
        expected = _expected_source_hash(item.binding, item.source_path, item.transition)
        key = _path_key(item.source_path)
        if key not in requests:
            requests[key] = (item.source_path, expected, set())
        path, prior_expected, lines = requests[key]
        if prior_expected != expected:
            raise EnhancedReplayError("selected units disagree on a source-file hash")
        lines.add(item.source_line)
        requests[key] = (path, prior_expected, lines)
    observations: dict[tuple[str, int], dict[str, Any]] = {}
    for key, (path, expected, lines) in requests.items():
        for line, row in _read_source_lines(path, lines, expected).items():
            observations[(key, line)] = row

    return observations


def _observation_from_source(item: _PendingRecord, source_row: Mapping[str, Any]) -> dict[str, Any]:
    source_schema = item.transition.get("source_schema")
    if not isinstance(source_schema, str) or not source_schema:
        raise EnhancedReplayError("transition source schema is missing")
    adapters = SOURCE_OBSERVATION_FIELD_BY_SCHEMA.get(source_schema, {})
    actual_field = adapters.get(item.source_field, item.source_field)
    if actual_field not in source_row:
        raise EnhancedReplayError(
            f"source observation field {actual_field!r} is absent for schema {source_schema!r}"
        )
    observation = source_row[actual_field]
    if not isinstance(observation, dict):
        raise EnhancedReplayError("source observation is not an object")
    if _stable_hash(observation) != item.source_observation_sha256:
        raise EnhancedReplayError("source-observation content hash mismatch")
    try:
        projected = canonical_actor_feature_observation(observation)
        assert_deployable_observation(
            projected,
            expected_mode=InformationMode.ACTOR_VISIBLE.value,
        )
    except ValueError as exc:
        raise EnhancedReplayError("projected source observation is not safely actor-visible") from exc
    if "search_begin_input" in projected or "logs" in projected:
        raise EnhancedReplayError("forbidden top-level payload survived actor-visible projection")
    if "looking" in projected.get("current", {}):
        raise EnhancedReplayError("opaque current.looking survived actor-visible projection")
    if "deck" in projected.get("select", {}):
        raise EnhancedReplayError("opaque select.deck survived actor-visible projection")
    current = projected.get("current")
    try:
        observation_actor = int(current.get("yourIndex"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise EnhancedReplayError("actor-visible observation has no valid actor") from exc
    if observation_actor != int(item.transition["actor"]):
        raise EnhancedReplayError("source observation actor differs from transition actor")

    return projected


def _immutable_s1(data: LabData, global_row: int) -> np.ndarray:
    start, stop = S1_SLICE
    values = np.array(data.X[global_row, start:stop], dtype=np.float32, copy=True)
    if values.shape != (stop - start,) or not np.isfinite(values).all():
        raise EnhancedReplayError("accepted S1 feature prefix is malformed or non-finite")
    values.setflags(write=False)

    return values


def load_bounded_replay_dataset(
    cache_dir: str | Path,
    *,
    split_manifest_path: str | Path,
    max_rows: int,
    seed: int,
    exact_d2_deck_identity: str,
) -> BoundedReplayDataset:
    """Load a deterministic, unique, eligible-train replay subset.

    Every selected row is mapped through the cache's ordered input receipts to
    a current unit-manifest-bound transition sidecar and then to a hash-bound
    immutable source observation.  Any missing or ambiguous evidence fails the
    whole load; the function never substitutes another field, row, or source.
    """
    if not isinstance(exact_d2_deck_identity, str) or not exact_d2_deck_identity:
        raise EnhancedReplayError("exact_d2_deck_identity must be explicit")
    cache_path = _resolved_path(cache_dir)
    split_path = _resolved_path(split_manifest_path)
    try:
        data = LabData(str(cache_path), split_manifest_path=str(split_path), load_labels=False)
    except (OSError, ValueError) as exc:
        raise EnhancedReplayError("replay cache is not an accepted label-blind LabData binding") from exc
    canonical_root, _, inputs = _validate_cache_inputs(data, split_path)
    _, catalog_units = _load_catalog_units(data, canonical_root)
    bindings = _build_part_bindings(data, canonical_root, inputs, catalog_units)
    selected = _select_train_rows(data, max_rows=max_rows, seed=seed)
    pending = _pending_records(data, bindings, selected)
    source_rows = _source_observations(pending)
    records: list[ReplayRecord] = []
    source_counts: dict[str, int] = {}
    for item in pending:
        source_row = source_rows[(_path_key(item.source_path), item.source_line)]
        observation = _observation_from_source(item, source_row)
        legal_set = tuple(copy.deepcopy(item.transition["legal_action_set"]))
        chosen = copy.deepcopy(item.transition["chosen_action"])
        records.append(ReplayRecord(
            global_row=item.global_row,
            source_id=item.binding.source_id,
            unit_id=item.binding.unit_id,
            part=item.binding.part,
            local_row=item.local_row,
            decision_id=str(item.transition["decision_id"]),
            actor=int(item.transition["actor"]),
            s1=_immutable_s1(data, item.global_row),
            actor_visible_observation=observation,
            legal_set=legal_set,
            chosen_selection=chosen,
            policy_id=item.policy_id,
            opponent_policy_id=item.opponent_policy_id,
            deck_identity=item.deck_identity,
            opponent_deck_identity=item.opponent_deck_identity,
            source_observation_sha256=item.source_observation_sha256,
        ))
        source_counts[item.binding.source_id] = source_counts.get(item.binding.source_id, 0) + 1
    decision_ids = {record.decision_id for record in records}
    if len(decision_ids) != len(records):
        raise EnhancedReplayError("loaded replay records do not have unique decision IDs")
    train_index = data.split_names.index("train")
    eligible_available = int(
        ((data.meta["split_idx"] == train_index) & data.meta["is_target_eligible"]).sum()
    )
    coverage = ReplayCoverage(
        requested_rows=max_rows,
        eligible_train_rows_available=eligible_available,
        selected_rows=len(records),
        unique_decision_ids=len(decision_ids),
        exact_d2_deck_identity=exact_d2_deck_identity,
        exact_d2_actor_rows=sum(
            record.deck_identity == exact_d2_deck_identity for record in records
        ),
        exact_d2_opponent_rows=sum(
            record.opponent_deck_identity == exact_d2_deck_identity for record in records
        ),
        source_rows=dict(sorted(source_counts.items())),
    )

    return BoundedReplayDataset(
        records=tuple(records),
        coverage=coverage,
        cache_manifest_sha256=_sha256_path(cache_path / "cache_manifest.json"),
        split_manifest_sha256=_sha256_path(split_path),
    )


__all__ = [
    "BoundedReplayDataset",
    "EnhancedReplayError",
    "ReplayCoverage",
    "ReplayBatchMaterializer",
    "ReplayRecord",
    "SOURCE_OBSERVATION_FIELD_BY_SCHEMA",
    "load_bounded_replay_dataset",
]
