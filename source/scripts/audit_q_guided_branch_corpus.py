"""Audit manifest-bound Q-guided branch coverage without exposing private state.

The input directories must be local, read-only snapshots of the aggregate,
actor-visible miner artifacts.  The audit never reads miner scratch directories.
It validates immutable shard bindings with the training loader, reports pending
checkpoint rows separately, and reconstructs the exact frozen-tranche cursor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import fmean
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.q_guided.branching import assert_private_content_absent
from deepmew.q_guided.teacher_contract import validate_safe_shard
from deepmew.replay.contract.preflight import require_accepted_canonical_decision_receipt
from deepmew.q_guided.training import (
    ELIGIBLE_TEACHER_STATUS,
    ShardBinding,
    TEACHER_MANIFEST_SCHEMA_VERSION,
    load_manifest_bound_shards,
)


SCHEMA_VERSION = "deepmew.q_guided.branch_corpus_audit.v1"
FINAL_CORPUS_MANIFEST_SCHEMA_VERSION = "deepmew.q_guided.corpus_manifest.v1"
FINAL_CLOSEOUT_SCHEMA_VERSION = "deepmew.q_guided.branch_corpus_closeout.v1"
IMMUTABLE_ELIGIBILITY = "manifest_bound_aggregate_teacher_supervision"
PENDING_ELIGIBILITY = "checkpoint_pending_unsealed"
POLICY_CONTROL_ELIGIBILITY = "not_eligible_research_quarantine"
MIN_WORLD_SAMPLES = 2
MIN_PAIRED_SAMPLES = 2
STRATA_DISTRIBUTION_SHIFT_THRESHOLD = 0.05


@dataclass(frozen=True)
class CorpusRow:
    miner_id: str
    matchup: str
    opponent_policy: str
    shard_index: int | None
    shard_path: str | None
    row_offset: int | None
    row: Mapping[str, Any]
    receipt: Mapping[str, Any]
    supervision_eligibility: str

    @property
    def row_sha256(self) -> str:
        return str(self.receipt["row_sha256"]).lower()

    @property
    def signature_sha256(self) -> str:
        return str(self.receipt["actor_visible_signature_sha256"]).lower()

    @property
    def source_sha256(self) -> str:
        return str(self.receipt["actor_visible_source_sha256"]).lower()


@dataclass(frozen=True)
class MinerSnapshot:
    miner_id: str
    matchup: str
    config: Mapping[str, Any]
    progress: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    immutable_rows: tuple[CorpusRow, ...]
    pending_rows: tuple[CorpusRow, ...]
    shard_receipts: tuple[Mapping[str, Any], ...]
    hashes: Mapping[str, str]
    dataset_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")

    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            rows.append(value)

    return rows


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a SHA-256")

    return value.lower()


def _validate_receipt(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected_fields = {
        "status",
        "world_sample_count",
        "paired_sample_count",
        "row_sha256",
        "actor_visible_signature_sha256",
        "actor_visible_source_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("row receipt field inventory mismatch")
    if receipt["status"] != ELIGIBLE_TEACHER_STATUS:
        raise ValueError("row receipt is not aggregate-teacher eligible")
    if receipt["world_sample_count"] < MIN_WORLD_SAMPLES:
        raise ValueError("row receipt has insufficient world samples")
    if receipt["paired_sample_count"] < MIN_PAIRED_SAMPLES:
        raise ValueError("row receipt has insufficient paired samples")
    for name in ("row_sha256", "actor_visible_signature_sha256", "actor_visible_source_sha256"):
        _validate_sha256(receipt[name], name)
    if _sha256_bytes(_canonical_bytes(row)) != str(receipt["row_sha256"]).lower():
        raise ValueError("row receipt hash mismatch")
    validate_safe_shard(row)
    assert_private_content_absent(row)


def _safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("run manifest contains an invalid relative path")
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError("run manifest path escapes its snapshot root")

    return target


def _load_pending(
    run_dir: Path,
    miner_id: str,
    matchup: str,
    opponent_policy: str,
) -> tuple[CorpusRow, ...]:
    pending_path = run_dir / "pending.json"
    if not pending_path.is_file():
        return ()
    pending = _read_json(pending_path)
    if set(pending) != {"rows", "row_receipts"}:
        raise ValueError(f"{miner_id} pending artifact field inventory mismatch")
    rows = pending["rows"]
    receipts = pending["row_receipts"]
    if not isinstance(rows, list) or not isinstance(receipts, list) or len(rows) != len(receipts):
        raise ValueError(f"{miner_id} pending receipt coverage mismatch")
    result = []
    for row, receipt in zip(rows, receipts, strict=True):
        if not isinstance(row, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError(f"{miner_id} pending artifact contains a malformed row")
        _validate_receipt(row, receipt)
        result.append(
            CorpusRow(
                miner_id=miner_id,
                matchup=matchup,
                opponent_policy=opponent_policy,
                shard_index=None,
                shard_path=None,
                row_offset=None,
                row=dict(row),
                receipt=dict(receipt),
                supervision_eligibility=PENDING_ELIGIBILITY,
            )
        )

    return tuple(result)


def load_miner_snapshot(run_dir: Path) -> MinerSnapshot:
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    progress_path = run_dir / "progress.json"
    manifest_path = run_dir / "manifest.json"
    config = _read_json(config_path)
    progress = _read_json(progress_path)
    run_manifest = _read_json(manifest_path)
    miner_id = str(run_manifest.get("miner_id") or config.get("miner_id") or "").upper()
    if miner_id not in {"M1", "M2"} or str(config.get("miner_id")).upper() != miner_id:
        raise ValueError("miner identity is inconsistent across snapshot receipts")
    config_sha256 = _validate_sha256(config.get("config_sha256"), "config_sha256")
    if progress.get("config_sha256") != config_sha256 or run_manifest.get("config_sha256") != config_sha256:
        raise ValueError(f"{miner_id} config identity is inconsistent")
    policy0 = config.get("policy0")
    policy1 = config.get("policy1")
    if not isinstance(policy0, str) or not policy0 or not isinstance(policy1, str) or not policy1:
        raise ValueError(f"{miner_id} matchup identity is malformed")
    matchup = f"{policy0}_vs_{policy1}"
    shard_entries = run_manifest.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError(f"{miner_id} run manifest has no immutable shards")
    bindings = []
    shard_receipts = []
    shard_rows: list[tuple[Mapping[str, Any], Mapping[str, Any], int, str]] = []
    expected_paths = set()
    for shard_index, entry in enumerate(shard_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{miner_id} run manifest contains a malformed shard entry")
        shard_path = _safe_relative(run_dir, entry.get("shard"))
        sidecar_path = _safe_relative(run_dir, entry.get("manifest"))
        expected_paths.add(shard_path.resolve())
        expected_paths.add(sidecar_path.resolve())
        sidecar = _read_json(sidecar_path)
        if _sha256_file(shard_path) != _validate_sha256(entry.get("shard_sha256"), "shard_sha256"):
            raise ValueError(f"{miner_id} run-manifest shard hash mismatch")
        if _sha256_file(sidecar_path) != _validate_sha256(entry.get("manifest_sha256"), "manifest_sha256"):
            raise ValueError(f"{miner_id} run-manifest sidecar hash mismatch")
        if sidecar.get("shard_sha256") != entry.get("shard_sha256"):
            raise ValueError(f"{miner_id} sidecar disagrees with its run manifest")
        rows = _read_jsonl(shard_path)
        receipts = sidecar.get("row_receipts")
        if not isinstance(receipts, list) or len(receipts) != len(rows) or entry.get("row_count") != len(rows):
            raise ValueError(f"{miner_id} shard row coverage mismatch")
        for row_offset, (row, receipt) in enumerate(zip(rows, receipts, strict=True)):
            if not isinstance(receipt, Mapping):
                raise ValueError(f"{miner_id} sidecar contains a malformed receipt")
            shard_rows.append((row, receipt, row_offset, str(entry["shard"])))
        bindings.append(ShardBinding(shard_path, sidecar_path))
        shard_receipts.append(
            {
                "shard_index": shard_index,
                "shard": str(entry["shard"]),
                "manifest": str(entry["manifest"]),
                "shard_sha256": str(entry["shard_sha256"]).lower(),
                "manifest_sha256": str(entry["manifest_sha256"]).lower(),
                "row_count": int(entry["row_count"]),
            }
        )
    actual_paths = {
        path.resolve()
        for pattern in ("shard_*.jsonl", "shard_*.manifest.json")
        for path in (run_dir / "shards").glob(pattern)
    }
    if actual_paths != expected_paths:
        raise ValueError(f"{miner_id} downloaded shard inventory disagrees with the run manifest")
    dataset = load_manifest_bound_shards(bindings)
    if len(dataset.rows) != len(shard_rows):
        raise ValueError(f"{miner_id} validated dataset row coverage mismatch")
    immutable_rows = []
    for loaded, (row, receipt, row_offset, relative_path) in zip(dataset.rows, shard_rows, strict=True):
        if loaded.row_sha256 != str(receipt["row_sha256"]).lower():
            raise ValueError(f"{miner_id} loader and sidecar row identities disagree")
        shard_index = int(Path(relative_path).stem.split("_")[-1])
        immutable_rows.append(
            CorpusRow(
                miner_id=miner_id,
                matchup=matchup,
                opponent_policy=policy1,
                shard_index=shard_index,
                shard_path=relative_path,
                row_offset=row_offset,
                row=dict(row),
                receipt=dict(receipt),
                supervision_eligibility=IMMUTABLE_ELIGIBILITY,
            )
        )
    pending_rows = _load_pending(run_dir, miner_id, matchup, policy1)
    progress_safety = progress.get("safety")
    manifest_safety = run_manifest.get("safety")
    if progress_safety != manifest_safety or not isinstance(progress_safety, Mapping):
        raise ValueError(f"{miner_id} safety receipts disagree")
    if any(progress_safety.get(name) != 0 for name in ("illegal_actions", "crashes", "timeouts")):
        raise ValueError(f"{miner_id} snapshot is not safety-clean")

    return MinerSnapshot(
        miner_id=miner_id,
        matchup=matchup,
        config=config,
        progress=progress,
        run_manifest=run_manifest,
        immutable_rows=tuple(immutable_rows),
        pending_rows=pending_rows,
        shard_receipts=tuple(shard_receipts),
        hashes={
            "config_file_sha256": _sha256_file(config_path),
            "progress_file_sha256": _sha256_file(progress_path),
            "run_manifest_sha256": _sha256_file(manifest_path),
            "pending_file_sha256": _sha256_file(run_dir / "pending.json") if pending_rows else "not_present",
        },
        dataset_sha256=dataset.dataset_sha256,
    )


def _action_id(option: Mapping[str, Any], position: int) -> str:
    selection = option.get("selection")
    if selection is not None:
        if not isinstance(selection, list):
            raise ValueError("actor-visible legal action selection is not a list")

        return json.dumps(selection, separators=(",", ":"))
    index = option.get("index", position)

    return str(index)


def _action_family(option: Mapping[str, Any]) -> str:
    semantics = option.get("semantics")
    if not isinstance(semantics, list):
        raise ValueError("actor-visible legal action semantics are malformed")

    return "+".join(str(item.get("option_type") or "UNKNOWN") for item in semantics) or "EMPTY"


def _semantic_identity(option: Mapping[str, Any]) -> str:
    semantics = option.get("semantics")
    if not isinstance(semantics, list):
        raise ValueError("actor-visible legal action semantics are malformed")
    keys = []
    for item in semantics:
        if not isinstance(item, Mapping):
            raise ValueError("actor-visible legal action semantic is malformed")
        keys.append(str(item.get("semantic_key") or item.get("option_type") or "UNKNOWN"))

    return "+".join(keys) or "EMPTY"


def _row_dimensions(corpus_row: CorpusRow) -> dict[str, str]:
    row = corpus_row.row
    options = row["actor_visible_legal_options"]
    action_ids = [_action_id(option, index) for index, option in enumerate(options)]
    if set(action_ids) != set(row["aggregate_q_mean"]):
        raise ValueError("actor-visible sibling identities disagree with aggregate Q targets")
    family_by_id = {
        action_id: _action_family(option)
        for action_id, option in zip(action_ids, options, strict=True)
    }
    selected = str(row["selected_root_action"])
    if selected not in family_by_id:
        raise ValueError("selected root action is absent from actor-visible siblings")
    q_values = sorted((float(value) for value in row["aggregate_q_mean"].values()), reverse=True)
    gap = None if len(q_values) == 1 else q_values[0] - q_values[1]
    confidence = float(row["action_ranking_confidence"])
    max_standard_error = max(float(value) for value in row["q_standard_error"].values())

    return {
        "matchup": corpus_row.matchup,
        "opponent_policy": corpus_row.opponent_policy,
        "opponent_bucket": str(row["opponent_strength_bucket"]),
        "continuation_policy": str(row["continuation_policy_identity"]),
        "selected_action_family": family_by_id[selected],
        "confidence_bucket": _confidence_bucket(confidence),
        "standard_error_bucket": _standard_error_bucket(max_standard_error),
        "q_gap_bucket": _q_gap_bucket(gap),
        "supervision_eligibility": corpus_row.supervision_eligibility,
    }


def _confidence_bucket(value: float) -> str:
    if value < 0.25:
        return "very_low_[0,0.25)"
    if value < 0.5:
        return "low_[0.25,0.5)"
    if value < 0.75:
        return "moderate_[0.5,0.75)"
    if value < 1.0:
        return "high_[0.75,1)"

    return "certain_1"


def _standard_error_bucket(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value <= 0.1:
        return "low_(0,0.1]"
    if value <= 0.25:
        return "moderate_(0.1,0.25]"
    if value <= 0.5:
        return "high_(0.25,0.5]"

    return "very_high_>0.5"


def _q_gap_bucket(value: float | None) -> str:
    if value is None:
        return "single_action"
    if value == 0.0:
        return "tie_0"
    if value <= 0.05:
        return "tiny_(0,0.05]"
    if value <= 0.15:
        return "small_(0.05,0.15]"
    if value <= 0.35:
        return "medium_(0.15,0.35]"

    return "large_>0.35"


def _percentile(values: Sequence[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low

    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
        "max": max(values) if values else None,
        "mean": fmean(values) if values else None,
    }


def _counter(value: Counter[str]) -> dict[str, int]:
    return dict(sorted(value.items()))


def summarize_rows(rows: Sequence[CorpusRow]) -> dict[str, Any]:
    action_families = Counter()
    action_family_root_coverage = Counter()
    selected_families = Counter()
    matchups = Counter()
    opponent_policies = Counter()
    opponent_buckets = Counter()
    continuation_policies = Counter()
    confidence_buckets = Counter()
    standard_error_buckets = Counter()
    q_gap_buckets = Counter()
    supervision = Counter()
    sibling_counts = []
    sibling_count_rows = Counter()
    rollout_counts = []
    rollout_count_instances = Counter()
    root_rollout_totals = []
    confidence_values = []
    standard_errors = []
    q_gaps = []
    semantic_duplicate_rows = 0
    semantic_duplicate_instances = 0
    by_matchup: dict[str, Counter[str]] = defaultdict(Counter)
    for corpus_row in rows:
        row = corpus_row.row
        options = row["actor_visible_legal_options"]
        action_ids = [_action_id(option, index) for index, option in enumerate(options)]
        if set(action_ids) != set(row["aggregate_q_mean"]):
            raise ValueError("actor-visible sibling identities disagree with aggregate Q targets")
        families = [_action_family(option) for option in options]
        for family in families:
            action_families[family] += 1
        for family in set(families):
            action_family_root_coverage[family] += 1
        family_by_id = dict(zip(action_ids, families, strict=True))
        selected = str(row["selected_root_action"])
        if selected not in family_by_id:
            raise ValueError("selected root action is absent from actor-visible siblings")
        selected_families[family_by_id[selected]] += 1
        dimensions = _row_dimensions(corpus_row)
        matchups[dimensions["matchup"]] += 1
        opponent_policies[dimensions["opponent_policy"]] += 1
        opponent_buckets[dimensions["opponent_bucket"]] += 1
        continuation_policies[dimensions["continuation_policy"]] += 1
        confidence_buckets[dimensions["confidence_bucket"]] += 1
        standard_error_buckets[dimensions["standard_error_bucket"]] += 1
        q_gap_buckets[dimensions["q_gap_bucket"]] += 1
        supervision[dimensions["supervision_eligibility"]] += 1
        for name, value in dimensions.items():
            by_matchup[corpus_row.matchup][f"{name}={value}"] += 1
        q_values = sorted((float(value) for value in row["aggregate_q_mean"].values()), reverse=True)
        gap = None if len(q_values) == 1 else q_values[0] - q_values[1]
        if gap is not None:
            q_gaps.append(gap)
        confidence_values.append(float(row["action_ranking_confidence"]))
        standard_errors.append(max(float(value) for value in row["q_standard_error"].values()))
        sibling_counts.append(len(options))
        sibling_count_rows[str(len(options))] += 1
        row_rollout_counts = [int(value) for value in row["rollout_count"].values()]
        if set(row["rollout_count"]) != set(action_ids) or any(value < 1 for value in row_rollout_counts):
            raise ValueError("rollout counts disagree with actor-visible siblings")
        rollout_counts.extend(float(value) for value in row_rollout_counts)
        root_rollout_totals.append(float(sum(row_rollout_counts)))
        rollout_count_instances.update(str(value) for value in row_rollout_counts)
        semantic_ids = [_semantic_identity(option) for option in options]
        repeated = len(semantic_ids) - len(set(semantic_ids))
        if repeated:
            semantic_duplicate_rows += 1
            semantic_duplicate_instances += repeated

    return {
        "rows": len(rows),
        "eligible_rows": sum(row.supervision_eligibility == IMMUTABLE_ELIGIBILITY for row in rows),
        "root_state_count": len({row.source_sha256 for row in rows}),
        "sibling_actions": sum(sibling_counts),
        "unique_row_sha256": len({row.row_sha256 for row in rows}),
        "unique_actor_visible_signatures": len({row.signature_sha256 for row in rows}),
        "unique_actor_visible_sources": len({row.source_sha256 for row in rows}),
        "sibling_action_count": _numeric_summary([float(value) for value in sibling_counts]),
        "legal_option_count": _numeric_summary([float(value) for value in sibling_counts]),
        "legal_option_count_rows": _counter(sibling_count_rows),
        "rollout_count": {
            "per_sibling_action": _numeric_summary(rollout_counts),
            "per_sibling_action_instances": _counter(rollout_count_instances),
            "per_root_total": _numeric_summary(root_rollout_totals),
        },
        "action_family_sibling_actions": _counter(action_families),
        "action_family_root_coverage": _counter(action_family_root_coverage),
        "selected_action_family_rows": _counter(selected_families),
        "matchup_rows": _counter(matchups),
        "opponent_policy_rows": _counter(opponent_policies),
        "opponent_bucket_rows": _counter(opponent_buckets),
        "continuation_policy_rows": _counter(continuation_policies),
        "uncertainty": {
            "action_ranking_confidence": _numeric_summary(confidence_values),
            "action_ranking_confidence_bins": _counter(confidence_buckets),
            "maximum_q_standard_error": _numeric_summary(standard_errors),
            "maximum_q_standard_error_bins": _counter(standard_error_buckets),
        },
        "q_gap": {
            "top_two_gap": _numeric_summary(q_gaps),
            "bins": _counter(q_gap_buckets),
        },
        "supervision_eligibility_rows": _counter(supervision),
        "semantic_equivalent_sibling_options": {
            "rows_with_repeated_semantic_identity": semantic_duplicate_rows,
            "extra_option_instances": semantic_duplicate_instances,
            "interpretation": "diagnostic_only_source_indices_can_still_differ",
        },
        "by_matchup_dimension_rows": {
            matchup: _counter(counter)
            for matchup, counter in sorted(by_matchup.items())
        },
    }


def _duplicate_summary(rows: Sequence[CorpusRow], attribute: str) -> dict[str, Any]:
    grouped: dict[str, list[CorpusRow]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, attribute))].append(row)
    duplicates = [group for group in grouped.values() if len(group) > 1]
    dimensions: dict[str, Counter[str]] = defaultdict(Counter)
    cross_miner_groups = 0
    for group in duplicates:
        if len({row.miner_id for row in group}) > 1:
            cross_miner_groups += 1
        for row in group:
            for name, value in _row_dimensions(row).items():
                dimensions[name][value] += 1

    return {
        "duplicate_groups": len(duplicates),
        "row_instances_in_duplicate_groups": sum(len(group) for group in duplicates),
        "extra_instances": sum(len(group) - 1 for group in duplicates),
        "cross_miner_groups": cross_miner_groups,
        "duplicate_group_row_involvement_by_dimension": {
            name: _counter(counter)
            for name, counter in sorted(dimensions.items())
        },
    }


def analyze_duplication(rows: Sequence[CorpusRow]) -> dict[str, Any]:
    return {
        "exact_row_sha256": _duplicate_summary(rows, "row_sha256"),
        "actor_visible_signature_sha256": _duplicate_summary(rows, "signature_sha256"),
        "actor_visible_source_sha256": _duplicate_summary(rows, "source_sha256"),
    }


def _prefix(rows: Sequence[CorpusRow], count: int, miner_id: str) -> tuple[CorpusRow, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > len(rows):
        raise ValueError(f"freeze receipt has an invalid {miner_id} available-row cursor")

    return tuple(rows[:count])


def _compress_offsets(offsets: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(offsets))
    if not ordered:
        return []
    ranges = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])

    return ranges


def reconstruct_frozen_tranche(
    snapshots: Sequence[MinerSnapshot],
    tranche_dir: Path,
    freeze_receipt_path: Path,
) -> dict[str, Any]:
    by_id = {snapshot.miner_id.lower(): snapshot for snapshot in snapshots}
    if set(by_id) != {"m1", "m2"}:
        raise ValueError("frozen first-wave reconstruction requires M1 and M2")
    freeze_receipt = _read_json(freeze_receipt_path)
    available = freeze_receipt.get("eligible_row_counts")
    selected_counts = freeze_receipt.get("selected_row_counts")
    minimum_rows = freeze_receipt.get("minimum_rows")
    if not isinstance(available, Mapping) or not isinstance(selected_counts, Mapping):
        raise ValueError("freeze receipt lacks source cursor counts")
    if not isinstance(minimum_rows, int) or minimum_rows < 1:
        raise ValueError("freeze receipt minimum row count is invalid")
    seen_rows = set()
    seen_signatures = set()
    deduplicated: dict[str, list[CorpusRow]] = {"m1": [], "m2": []}
    for miner_id in ("m1", "m2"):
        source_rows = _prefix(by_id[miner_id].immutable_rows, int(available[miner_id]), miner_id)
        for row in source_rows:
            if row.row_sha256 in seen_rows or row.signature_sha256 in seen_signatures:
                continue
            seen_rows.add(row.row_sha256)
            seen_signatures.add(row.signature_sha256)
            deduplicated[miner_id].append(row)
    selected = []
    selected_by_miner = Counter()
    index = 0
    while len(selected) < minimum_rows:
        added = False
        for miner_id in ("m1", "m2"):
            rows = deduplicated[miner_id]
            if index < len(rows):
                selected.append(rows[index])
                selected_by_miner[miner_id] += 1
                added = True
                if len(selected) == minimum_rows:
                    break
        if not added:
            break
        index += 1
    expected_selected = {name: int(selected_counts[name]) for name in ("m1", "m2")}
    if dict(selected_by_miner) != expected_selected:
        raise ValueError("reconstructed source cursor disagrees with freeze receipt")
    tranche_manifest_path = Path(tranche_dir) / "tranche.manifest.json"
    tranche_shard_path = Path(tranche_dir) / "tranche.jsonl"
    tranche_manifest = _read_json(tranche_manifest_path)
    tranche_dataset = load_manifest_bound_shards(
        [ShardBinding(tranche_shard_path, tranche_manifest_path)]
    )
    if len(tranche_dataset.rows) != len(selected):
        raise ValueError("frozen tranche row count disagrees with its reconstructed cursor")
    tranche_receipts = tranche_manifest.get("row_receipts")
    if not isinstance(tranche_receipts, list) or len(tranche_receipts) != len(selected):
        raise ValueError("frozen tranche receipt coverage is malformed")
    for source, loaded, receipt in zip(selected, tranche_dataset.rows, tranche_receipts, strict=True):
        if source.row_sha256 != loaded.row_sha256 or dict(source.receipt) != receipt:
            raise ValueError("frozen tranche does not match the reconstructed miner prefix")
    expected_payload = b"".join(_canonical_bytes(row.row) + b"\n" for row in selected)
    if _sha256_bytes(expected_payload) != _sha256_file(tranche_shard_path):
        raise ValueError("frozen tranche payload differs from reconstructed source rows")
    shard_receipt_by_miner = {
        snapshot.miner_id: {
            int(item["shard_index"]): item
            for item in snapshot.shard_receipts
        }
        for snapshot in snapshots
    }
    cursor_groups: dict[tuple[str, int], list[tuple[int, CorpusRow]]] = defaultdict(list)
    for tranche_position, row in enumerate(selected):
        if row.shard_index is None:
            raise ValueError("frozen tranche mapped to an unsealed row")
        cursor_groups[(row.miner_id, row.shard_index)].append((tranche_position, row))
    sources: dict[str, Any] = {}
    mapping_digest_rows = []
    for miner_id in ("M1", "M2"):
        groups = []
        miner_positions = []
        for (group_miner, shard_index), positioned_rows in sorted(cursor_groups.items()):
            if group_miner != miner_id:
                continue
            shard_receipt = shard_receipt_by_miner[miner_id][shard_index]
            positions = [position for position, _row in positioned_rows]
            offsets = [int(row.row_offset) for _position, row in positioned_rows if row.row_offset is not None]
            miner_positions.extend(positions)
            groups.append(
                {
                    **shard_receipt,
                    "selected_row_count": len(positioned_rows),
                    "row_offset_ranges_inclusive": _compress_offsets(offsets),
                    "tranche_positions": positions,
                }
            )
            for position, row in positioned_rows:
                mapping_digest_rows.append(
                    {
                        "tranche_position": position,
                        "miner_id": miner_id,
                        "shard_index": shard_index,
                        "row_offset": row.row_offset,
                        "row_sha256": row.row_sha256,
                    }
                )
        sources[miner_id] = {
            "selected_row_count": len(miner_positions),
            "available_unique_rows_at_freeze": len(deduplicated[miner_id.lower()]),
            "source_shards": groups,
            "tranche_position_pattern": {
                "start": min(miner_positions),
                "step": 2 if len(miner_positions) > 1 else 0,
                "count": len(miner_positions),
            },
        }
    mapping_digest_rows.sort(key=lambda item: item["tranche_position"])

    return {
        "freeze_receipt_sha256": _sha256_file(freeze_receipt_path),
        "status": str(freeze_receipt.get("status")),
        "tranche_shard_sha256": _sha256_file(tranche_shard_path),
        "tranche_manifest_sha256": _sha256_file(tranche_manifest_path),
        "tranche_dataset_sha256": tranche_dataset.dataset_sha256,
        "row_count": len(selected),
        "ordered_source_mapping_sha256": _sha256_bytes(_canonical_bytes(mapping_digest_rows)),
        "source_cursor": sources,
        "checks": {
            "freeze_receipt_prefix_reconstructed": True,
            "ordered_rows_match": True,
            "row_receipts_match": True,
            "payload_hash_matches": True,
            "all_rows_manifest_bound": True,
        },
    }


def _miner_receipt(snapshot: MinerSnapshot) -> dict[str, Any]:
    immutable_count = len(snapshot.immutable_rows)
    pending_count = len(snapshot.pending_rows)
    reconstructed_cursor = immutable_count + pending_count
    progress_cursor = snapshot.progress.get("eligible_rows")
    manifest_cursor = snapshot.run_manifest.get("eligible_rows")

    return {
        "miner_id": snapshot.miner_id,
        "matchup": snapshot.matchup,
        "source_turn_range": {
            "minimum": snapshot.config.get("minimum_source_turn"),
            "maximum": snapshot.config.get("maximum_source_turn"),
            "maximum_zero_means_unbounded": snapshot.config.get("maximum_source_turn") == 0,
        },
        "classification": snapshot.run_manifest.get("classification"),
        "status": snapshot.progress.get("status"),
        "safety": dict(snapshot.progress.get("safety") or {}),
        "config_sha256": snapshot.config.get("config_sha256"),
        "dataset_sha256": snapshot.dataset_sha256,
        "hashes": dict(snapshot.hashes),
        "cursor": {
            "immutable_shards": len(snapshot.shard_receipts),
            "immutable_rows": immutable_count,
            "pending_unsealed_rows": pending_count,
            "reconstructed_checkpoint_rows": reconstructed_cursor,
            "progress_eligible_rows": progress_cursor,
            "run_manifest_eligible_rows": manifest_cursor,
            "progress_minus_reconstructed": progress_cursor - reconstructed_cursor,
            "run_manifest_minus_reconstructed": manifest_cursor - reconstructed_cursor,
            "progress_cursor_coherent": progress_cursor == reconstructed_cursor,
            "run_manifest_cursor_coherent": manifest_cursor == reconstructed_cursor,
        },
        "shards": list(snapshot.shard_receipts),
    }


def _ordered_snapshots(m1_run_dir: Path, m2_run_dir: Path) -> list[MinerSnapshot]:
    snapshots = [load_miner_snapshot(m1_run_dir), load_miner_snapshot(m2_run_dir)]
    snapshots.sort(key=lambda snapshot: snapshot.miner_id)
    if [snapshot.miner_id for snapshot in snapshots] != ["M1", "M2"]:
        raise ValueError("audit inputs must contain exactly M1 and M2")

    return snapshots


def _build_audit_from_snapshots(
    snapshots: Sequence[MinerSnapshot],
    tranche_dir: Path,
    freeze_receipt_path: Path,
    *,
    observed_at: str,
    snapshot_status: str,
) -> dict[str, Any]:
    if snapshot_status not in {"provisional_m2_running", "final_m2_completed"}:
        raise ValueError("snapshot status is invalid")
    if [snapshot.miner_id for snapshot in snapshots] != ["M1", "M2"]:
        raise ValueError("audit inputs must contain exactly M1 and M2")
    immutable_rows = tuple(row for snapshot in snapshots for row in snapshot.immutable_rows)
    pending_rows = tuple(row for snapshot in snapshots for row in snapshot.pending_rows)
    checkpoint_rows = immutable_rows + pending_rows
    tranche = reconstruct_frozen_tranche(snapshots, tranche_dir, freeze_receipt_path)
    immutable_identity = {
        "miners": [
            {
                "miner_id": snapshot.miner_id,
                "dataset_sha256": snapshot.dataset_sha256,
                "shards": [dict(item) for item in snapshot.shard_receipts],
            }
            for snapshot in snapshots
        ]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": snapshot_status,
        "classification": "quarantine",
        "observed_at": observed_at,
        "scope": {
            "inputs": "aggregate_actor_visible_research_artifacts_only",
            "scratch_directories_read": False,
            "private_identifiers_emitted": False,
            "labels_modified": False,
            "policy_control_eligibility": POLICY_CONTROL_ELIGIBILITY,
        },
        "immutable_corpus_sha256": _sha256_bytes(_canonical_bytes(immutable_identity)),
        "miners": {
            snapshot.miner_id: _miner_receipt(snapshot)
            for snapshot in snapshots
        },
        "frozen_m3_tranche": tranche,
        "coverage": {
            "immutable_manifest_bound": summarize_rows(immutable_rows),
            "checkpoint_pending_unsealed": summarize_rows(pending_rows),
            "checkpoint_total": summarize_rows(checkpoint_rows),
        },
        "duplication": {
            "immutable_manifest_bound": analyze_duplication(immutable_rows),
            "checkpoint_total": analyze_duplication(checkpoint_rows),
        },
        "gates": {
            "immutable_rows_teacher_supervision_eligible": True,
            "pending_rows_teacher_supervision_eligible": False,
            "policy_control_eligible": False,
            "final_m2_manifest_frozen": snapshot_status == "final_m2_completed",
            "serious_anchored_panel_authorized_by_this_audit": False,
        },
    }


def build_audit(
    m1_run_dir: Path,
    m2_run_dir: Path,
    tranche_dir: Path,
    freeze_receipt_path: Path,
    *,
    observed_at: str | None = None,
    snapshot_status: str = "provisional_m2_running",
) -> dict[str, Any]:
    snapshots = _ordered_snapshots(m1_run_dir, m2_run_dir)

    return _build_audit_from_snapshots(
        snapshots,
        tranche_dir,
        freeze_receipt_path,
        observed_at=observed_at or datetime.now(timezone.utc).isoformat(),
        snapshot_status=snapshot_status,
    )


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return float(value)


def _validate_final_miner(
    snapshot: MinerSnapshot,
    completion_receipt_path: Path,
    minimum_useful_seconds: float,
) -> dict[str, Any]:
    miner_id = snapshot.miner_id
    completion_receipt_path = Path(completion_receipt_path).resolve()
    completion = _read_json(completion_receipt_path)
    progress = snapshot.progress
    run_manifest = snapshot.run_manifest
    if snapshot.pending_rows:
        raise ValueError(f"{miner_id} final snapshot still contains unsealed rows")
    if progress.get("status") != "completed_useful_floor":
        raise ValueError(f"{miner_id} progress has not completed its useful-time floor")
    if run_manifest.get("status") != progress.get("status"):
        raise ValueError(f"{miner_id} final run-manifest status disagrees with progress")
    if run_manifest.get("classification") != "quarantine":
        raise ValueError(f"{miner_id} final artifacts escaped research quarantine")
    config_identity = dict(snapshot.config)
    config_sha256 = str(config_identity.pop("config_sha256"))
    if _sha256_bytes(_canonical_bytes(config_identity)) != config_sha256:
        raise ValueError(f"{miner_id} config payload does not match config_sha256")
    if progress.get("duplicate_rows") != 0:
        raise ValueError(f"{miner_id} recorded duplicate roots")
    if progress.get("v0_validated") is not True:
        raise ValueError(f"{miner_id} lacks authoritative V0 provenance")
    immutable_rows = len(snapshot.immutable_rows)
    immutable_shards = len(snapshot.shard_receipts)
    if progress.get("eligible_rows") != immutable_rows or run_manifest.get("eligible_rows") != immutable_rows:
        raise ValueError(f"{miner_id} final eligible-row cursor is incoherent")
    if progress.get("shard_count") != immutable_shards:
        raise ValueError(f"{miner_id} final shard count is incoherent")
    if progress.get("next_shard_index") != immutable_shards:
        raise ValueError(f"{miner_id} final shard cursor is incoherent")
    if [item["shard_index"] for item in snapshot.shard_receipts] != list(range(immutable_shards)):
        raise ValueError(f"{miner_id} immutable shard order is not contiguous")
    progress_seconds = _finite_number(
        progress.get("cumulative_useful_seconds"),
        f"{miner_id} progress cumulative useful seconds",
    )
    manifest_seconds = _finite_number(
        run_manifest.get("cumulative_useful_seconds"),
        f"{miner_id} manifest cumulative useful seconds",
    )
    if progress_seconds < minimum_useful_seconds:
        raise ValueError(f"{miner_id} has not reached the required useful-time floor")
    if not math.isclose(progress_seconds, manifest_seconds, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{miner_id} useful-time receipts disagree")
    if completion.get("schema_version") != "deepmew.modal_first_wave_job.v1":
        raise ValueError(f"{miner_id} completion receipt schema mismatch")
    if completion.get("job_id") != miner_id.lower() or completion.get("kind") != "miner":
        raise ValueError(f"{miner_id} completion receipt identity mismatch")
    if completion.get("status") != "completed_checkpointed":
        raise ValueError(f"{miner_id} completion receipt is not final")
    if completion.get("process_return_code") != 0 or completion.get("stopped_at_budget_bound") is not False:
        raise ValueError(f"{miner_id} completion process did not exit cleanly")
    if completion.get("required_outputs_complete") is not True:
        raise ValueError(f"{miner_id} completion receipt lacks required outputs")
    completion_seconds = _finite_number(
        completion.get("cumulative_useful_seconds"),
        f"{miner_id} completion cumulative useful seconds",
    )
    if not math.isclose(progress_seconds, completion_seconds, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{miner_id} completion useful-time cursor disagrees with progress")
    if completion.get("eligible_row_count") != immutable_rows:
        raise ValueError(f"{miner_id} completion eligible-row cursor disagrees with final shards")
    completion_safety = completion.get("safety_counters")
    if not isinstance(completion_safety, Mapping):
        raise ValueError(f"{miner_id} completion safety counters are malformed")
    if any(completion_safety.get(name, 0) != 0 for name in ("illegal_actions", "crashes", "timeouts")):
        raise ValueError(f"{miner_id} completion receipt records a safety failure")
    if completion.get("failure_category") not in {None, ""}:
        raise ValueError(f"{miner_id} completion receipt records a failure")

    return {
        "completion_receipt_sha256": _sha256_file(completion_receipt_path),
        "cumulative_useful_seconds": progress_seconds,
        "eligible_rows": immutable_rows,
        "immutable_shards": immutable_shards,
        "duplicate_rows": 0,
        "v0_validated": True,
        "safety": dict(progress["safety"]),
        "checks": {
            "completion_receipt_valid": True,
            "config_hash_valid": True,
            "useful_time_floor_met": True,
            "cursor_coherent": True,
            "no_unsealed_rows": True,
            "zero_duplicate_roots": True,
            "zero_safety_failures": True,
            "v0_provenance_valid": True,
        },
    }


def _mapping_at(value: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any] | None:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    if not isinstance(current, Mapping):
        return None

    return current


def _total_variation(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    before_total = sum(int(value) for value in before.values())
    after_total = sum(int(value) for value in after.values())
    if before_total == 0 or after_total == 0:
        return 0.0
    keys = set(before) | set(after)

    return 0.5 * sum(
        abs(int(before.get(key, 0)) / before_total - int(after.get(key, 0)) / after_total)
        for key in keys
    )


def _compare_distribution(
    provisional: Mapping[str, Any],
    final: Mapping[str, Any],
    path: Sequence[str],
) -> dict[str, Any]:
    before = _mapping_at(provisional, path)
    after = _mapping_at(final, path)
    if before is None or after is None:
        return {"status": "not_instrumented_in_provisional_snapshot"}
    if any(int(after.get(key, 0)) < int(value) for key, value in before.items()):
        raise ValueError(f"final distribution decreased at {'.'.join(path)}")
    keys = sorted(set(before) | set(after))

    return {
        "status": "compared",
        "delta_counts": {
            key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in keys
            if int(after.get(key, 0)) != int(before.get(key, 0))
        },
        "new_strata_values": [key for key in keys if key not in before and int(after.get(key, 0)) > 0],
        "total_variation_distance": _total_variation(before, after),
    }


def _assert_provisional_prefix(
    provisional: Mapping[str, Any],
    final_audit: Mapping[str, Any],
) -> None:
    if provisional.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("provisional audit schema mismatch")
    if provisional.get("status") != "provisional_m2_running":
        raise ValueError("comparison receipt is not a provisional M2 snapshot")
    for miner_id in ("M1", "M2"):
        provisional_miner = provisional.get("miners", {}).get(miner_id)
        final_miner = final_audit["miners"][miner_id]
        if not isinstance(provisional_miner, Mapping):
            raise ValueError(f"provisional audit lacks {miner_id}")
        if provisional_miner.get("config_sha256") != final_miner["config_sha256"]:
            raise ValueError(f"{miner_id} config changed after the provisional snapshot")
        provisional_shards = provisional_miner.get("shards")
        final_shards = final_miner["shards"]
        if not isinstance(provisional_shards, list):
            raise ValueError(f"provisional audit lacks {miner_id} shard inventory")
        if miner_id == "M1" and provisional_shards != final_shards:
            raise ValueError("M1 changed after the provisional snapshot")
        if miner_id == "M2" and final_shards[: len(provisional_shards)] != provisional_shards:
            raise ValueError("M2 final shards are not an append-only extension")


def _compare_provisional_to_final(
    provisional_receipt_path: Path,
    final_audit: Mapping[str, Any],
) -> dict[str, Any]:
    provisional_receipt_path = Path(provisional_receipt_path).resolve()
    provisional = _read_json(provisional_receipt_path)
    _assert_provisional_prefix(provisional, final_audit)
    provisional_coverage = provisional["coverage"]["immutable_manifest_bound"]
    final_coverage = final_audit["coverage"]["immutable_manifest_bound"]
    dimensions = {
        "action_family": ("action_family_root_coverage",),
        "legal_option_count": ("legal_option_count_rows",),
        "matchup": ("matchup_rows",),
        "opponent_policy": ("opponent_policy_rows",),
        "opponent_strength_bucket": ("opponent_bucket_rows",),
        "rollout_count": ("rollout_count", "per_sibling_action_instances"),
        "q_gap": ("q_gap", "bins"),
        "ranking_confidence": ("uncertainty", "action_ranking_confidence_bins"),
        "q_standard_error": ("uncertainty", "maximum_q_standard_error_bins"),
        "supervision_eligibility": ("supervision_eligibility_rows",),
    }
    comparisons = {
        name: _compare_distribution(provisional_coverage, final_coverage, path)
        for name, path in dimensions.items()
    }
    compared = [value for value in comparisons.values() if value["status"] == "compared"]
    new_strata_values = sum(len(value["new_strata_values"]) for value in compared)
    maximum_shift = max((value["total_variation_distance"] for value in compared), default=0.0)
    provisional_rows = int(provisional_coverage["rows"])
    final_rows = int(final_coverage["rows"])
    if final_rows < provisional_rows:
        raise ValueError("final immutable corpus is smaller than the provisional snapshot")
    provisional_m2 = provisional["miners"]["M2"]["cursor"]
    final_m2 = final_audit["miners"]["M2"]["cursor"]
    m2_rows_added = int(final_m2["immutable_rows"]) - int(provisional_m2["immutable_rows"])
    m2_shards_added = int(final_m2["immutable_shards"]) - int(provisional_m2["immutable_shards"])
    if m2_rows_added < 0 or m2_shards_added < 0:
        raise ValueError("final M2 cursor regressed after the provisional snapshot")
    if final_rows == provisional_rows:
        verdict = "no_final_segment_delta"
    elif new_strata_values or maximum_shift >= STRATA_DISTRIBUTION_SHIFT_THRESHOLD:
        verdict = "meaningful_strata_expansion"
    else:
        verdict = "mostly_more_of_same_strata"

    return {
        "provisional_receipt_sha256": _sha256_file(provisional_receipt_path),
        "provisional_immutable_rows": provisional_rows,
        "final_immutable_rows": final_rows,
        "immutable_rows_added": final_rows - provisional_rows,
        "unique_root_states_added": int(final_coverage["root_state_count"])
        - int(provisional_coverage.get("root_state_count", provisional_coverage["unique_actor_visible_sources"])),
        "m2_rows_added": m2_rows_added,
        "m2_shards_added": m2_shards_added,
        "dimension_comparisons": comparisons,
        "new_strata_value_count": new_strata_values,
        "maximum_total_variation_distance": maximum_shift,
        "meaningful_shift_threshold": STRATA_DISTRIBUTION_SHIFT_THRESHOLD,
        "verdict": verdict,
        "attribution": "M1_exact_and_M2_append_only_so_all_deltas_are_from_final_M2_segment",
    }


def _rows_by_shard(snapshot: MinerSnapshot) -> dict[int, tuple[CorpusRow, ...]]:
    grouped: dict[int, list[CorpusRow]] = defaultdict(list)
    for row in snapshot.immutable_rows:
        if row.shard_index is None or row.row_offset is None:
            raise ValueError(f"{snapshot.miner_id} immutable row lacks a shard cursor")
        grouped[row.shard_index].append(row)

    return {
        shard_index: tuple(sorted(rows, key=lambda row: int(row.row_offset)))
        for shard_index, rows in grouped.items()
    }


def _safe_volume_prefix(value: str, name: str) -> str:
    normalized = str(value).replace("\\", "/").strip("/")
    parts = PurePosixPath(normalized).parts
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} must be a safe volume-relative path")

    return "/".join(parts)


def build_final_corpus_closeout(
    m1_run_dir: Path,
    m2_run_dir: Path,
    tranche_dir: Path,
    freeze_receipt_path: Path,
    m1_completion_receipt_path: Path,
    m2_completion_receipt_path: Path,
    provisional_receipt_path: Path,
    *,
    minimum_useful_seconds: float,
    observed_at: str | None = None,
    source_prefixes: Mapping[str, str] | None = None,
    generation_id: str = "q_guided_first_wave_20260712",
) -> dict[str, Any]:
    minimum_seconds = _finite_number(minimum_useful_seconds, "minimum useful seconds")
    if minimum_seconds <= 0.0:
        raise ValueError("minimum useful seconds must be positive")
    snapshots = _ordered_snapshots(m1_run_dir, m2_run_dir)
    observed = observed_at or datetime.now(timezone.utc).isoformat()
    prefixes = dict(source_prefixes or {"M1": "runs/m1", "M2": "runs/m2"})
    if set(prefixes) != {"M1", "M2"}:
        raise ValueError("source prefixes must bind exactly M1 and M2")
    prefixes = {
        miner_id: _safe_volume_prefix(prefix, f"{miner_id} source prefix")
        for miner_id, prefix in prefixes.items()
    }
    if not isinstance(generation_id, str) or not generation_id.strip():
        raise ValueError("generation id must be non-empty")
    completion_paths = {
        "M1": Path(m1_completion_receipt_path),
        "M2": Path(m2_completion_receipt_path),
    }
    completion = {
        snapshot.miner_id: _validate_final_miner(
            snapshot,
            completion_paths[snapshot.miner_id],
            minimum_seconds,
        )
        for snapshot in snapshots
    }
    final_audit = _build_audit_from_snapshots(
        snapshots,
        tranche_dir,
        freeze_receipt_path,
        observed_at=observed,
        snapshot_status="final_m2_completed",
    )
    duplication = final_audit["duplication"]["immutable_manifest_bound"]
    if any(value["duplicate_groups"] for value in duplication.values()):
        raise ValueError("final corpus contains duplicate root or state identities")
    tranche_manifest_path = Path(tranche_dir) / "tranche.manifest.json"
    tranche_shard_path = Path(tranche_dir) / "tranche.jsonl"
    tranche_dataset = load_manifest_bound_shards(
        [ShardBinding(tranche_shard_path, tranche_manifest_path)]
    )
    consumed_hashes = [row.row_sha256 for row in tranche_dataset.rows]
    consumed_set = set(consumed_hashes)
    if len(consumed_set) != len(consumed_hashes):
        raise ValueError("M3 frozen tranche contains duplicate rows")
    immutable_rows = tuple(row for snapshot in snapshots for row in snapshot.immutable_rows)
    immutable_hashes = {row.row_sha256 for row in immutable_rows}
    if not consumed_set <= immutable_hashes:
        raise ValueError("M3 consumed rows are absent from the final corpus")
    shard_hashes = [item["shard_sha256"] for snapshot in snapshots for item in snapshot.shard_receipts]
    sidecar_hashes = [item["manifest_sha256"] for snapshot in snapshots for item in snapshot.shard_receipts]
    if len(shard_hashes) != len(set(shard_hashes)) or len(sidecar_hashes) != len(set(sidecar_hashes)):
        raise ValueError("final corpus contains duplicate shard or sidecar hashes")
    ordered_shards = []
    miner_manifests = {}
    global_shard_index = 0
    for snapshot in snapshots:
        rows_by_shard = _rows_by_shard(snapshot)
        miner_shard_indices = []
        for shard_receipt in snapshot.shard_receipts:
            shard_index = int(shard_receipt["shard_index"])
            shard_rows = rows_by_shard.get(shard_index, ())
            if len(shard_rows) != int(shard_receipt["row_count"]):
                raise ValueError(f"{snapshot.miner_id} shard row cursor is incomplete")
            consumed_offsets = [
                int(row.row_offset)
                for row in shard_rows
                if row.row_sha256 in consumed_set and row.row_offset is not None
            ]
            unseen_offsets = [
                int(row.row_offset)
                for row in shard_rows
                if row.row_sha256 not in consumed_set and row.row_offset is not None
            ]
            source_shard = str(shard_receipt["shard"]).replace("\\", "/")
            source_manifest = str(shard_receipt["manifest"]).replace("\\", "/")
            shard_summary = {
                "ordinal": global_shard_index,
                "miner_id": snapshot.miner_id,
                "source_shard_index": shard_index,
                "split": str(snapshot.config["split"]),
                "shard": f"{prefixes[snapshot.miner_id]}/{source_shard}",
                "manifest": f"{prefixes[snapshot.miner_id]}/{source_manifest}",
                "shard_sha256": shard_receipt["shard_sha256"],
                "manifest_sha256": shard_receipt["manifest_sha256"],
                "row_count": shard_receipt["row_count"],
                "m3_consumed_row_count": len(consumed_offsets),
                "m3_consumed_row_offsets_inclusive": _compress_offsets(consumed_offsets),
                "unseen_row_count": len(unseen_offsets),
                "unseen_row_offsets_inclusive": _compress_offsets(unseen_offsets),
                "coverage": summarize_rows(shard_rows),
            }
            ordered_shards.append(shard_summary)
            miner_shard_indices.append(global_shard_index)
            global_shard_index += 1
        miner_rows = snapshot.immutable_rows
        miner_consumed = sum(row.row_sha256 in consumed_set for row in miner_rows)
        miner_manifests[snapshot.miner_id] = {
            "matchup": snapshot.matchup,
            "opponent_policy": str(snapshot.config["policy1"]),
            "config_sha256": snapshot.config["config_sha256"],
            "dataset_sha256": snapshot.dataset_sha256,
            "hashes": dict(snapshot.hashes),
            "completion": completion[snapshot.miner_id],
            "ordered_global_shard_indices": miner_shard_indices,
            "coverage": summarize_rows(miner_rows),
            "m3_consumed_rows": miner_consumed,
            "unseen_rows": len(miner_rows) - miner_consumed,
        }
    final_rows = len(immutable_rows)
    consumed_rows = len(consumed_set)
    unseen_rows = final_rows - consumed_rows
    ordered_shard_identity = [
        {
            "ordinal": item["ordinal"],
            "miner_id": item["miner_id"],
            "source_shard_index": item["source_shard_index"],
            "split": item["split"],
            "shard": item["shard"],
            "manifest": item["manifest"],
            "shard_sha256": item["shard_sha256"],
            "manifest_sha256": item["manifest_sha256"],
            "row_count": item["row_count"],
        }
        for item in ordered_shards
    ]
    compatibility = {
        "teacher_manifest_schema_version": TEACHER_MANIFEST_SCHEMA_VERSION,
        "teacher_config_sha256s": sorted({row.row["config_sha256"] for row in immutable_rows}),
        "label_v_checkpoint_sha256s": sorted({row.row["v_checkpoint_sha256"] for row in immutable_rows}),
        "label_v_calibrator_sha256s": sorted({row.row["v_calibrator_sha256"] for row in immutable_rows}),
        "miner_config_sha256s": {
            snapshot.miner_id: snapshot.config["config_sha256"]
            for snapshot in snapshots
        },
        "split_values": sorted({str(snapshot.config["split"]) for snapshot in snapshots}),
    }
    lineage_identity = {
        "generation": generation_id,
        "compatibility": compatibility,
        "ordered_shard_identity": ordered_shard_identity,
    }
    comparison = _compare_provisional_to_final(provisional_receipt_path, final_audit)
    root_manifest = {
        "schema_version": FINAL_CORPUS_MANIFEST_SCHEMA_VERSION,
        "status": "frozen_final",
        "classification": "quarantine",
        "observed_at": observed,
        "generation": generation_id,
        "corpus_lineage_id": "qcl1-" + _sha256_bytes(_canonical_bytes(lineage_identity))[:32],
        "compatibility": compatibility,
        "scope": dict(final_audit["scope"]),
        "minimum_useful_seconds_per_miner": minimum_seconds,
        "ordered_shard_identity_sha256": _sha256_bytes(_canonical_bytes(ordered_shard_identity)),
        "ordered_shards": ordered_shards,
        "miners": miner_manifests,
        "coverage": final_audit["coverage"]["immutable_manifest_bound"],
        "duplication": duplication,
        "m3_consumption": {
            "tranche_dataset_sha256": final_audit["frozen_m3_tranche"]["tranche_dataset_sha256"],
            "ordered_consumed_row_sha256": _sha256_bytes(_canonical_bytes(consumed_hashes)),
            "consumed_unique_rows": consumed_rows,
            "unseen_unique_rows": unseen_rows,
            "final_immutable_rows": final_rows,
            "consumed_percentage": 100.0 * consumed_rows / final_rows,
        },
        "provisional_vs_final": comparison,
        "gates": {
            "m1_completed": True,
            "m2_completed": True,
            "minimum_useful_seconds_met": True,
            "completion_receipts_valid": True,
            "all_shards_and_sidecars_hash_validated": True,
            "ordered_shard_inventory_unique": True,
            "zero_duplicate_roots_or_states": True,
            "zero_safety_failures": True,
            "v0_provenance_valid": True,
            "m3_consumed_cursor_exact": True,
            "labels_modified": False,
            "policy_control_eligible": False,
        },
    }
    root_manifest_payload = _canonical_bytes(root_manifest) + b"\n"
    final_audit_payload = _canonical_bytes(final_audit) + b"\n"
    closeout_receipt = {
        "schema_version": FINAL_CLOSEOUT_SCHEMA_VERSION,
        "status": "complete",
        "classification": "quarantine",
        "observed_at": observed,
        "root_corpus_manifest": {
            "path": "final_corpus.manifest.json",
            "sha256": _sha256_bytes(root_manifest_payload),
        },
        "final_audit": {
            "path": "snapshot_receipt.json",
            "sha256": _sha256_bytes(final_audit_payload),
        },
        "miners": completion,
        "final_immutable_rows": final_rows,
        "final_immutable_shards": len(ordered_shards),
        "m3_consumed_unique_rows": consumed_rows,
        "m3_unseen_unique_rows": unseen_rows,
        "m3_consumed_percentage": 100.0 * consumed_rows / final_rows,
        "final_m2_segment_coverage_verdict": comparison["verdict"],
        "gates": dict(root_manifest["gates"]),
    }

    return {
        "final_audit": final_audit,
        "root_manifest": root_manifest,
        "closeout_receipt": closeout_receipt,
    }


def write_final_corpus_closeout(out_dir: Path, closeout: Mapping[str, Any]) -> None:
    out_dir = Path(out_dir).resolve()
    final_audit = closeout["final_audit"]
    root_manifest = closeout["root_manifest"]
    receipt = closeout["closeout_receipt"]
    audit_payload = _canonical_bytes(final_audit) + b"\n"
    manifest_payload = _canonical_bytes(root_manifest) + b"\n"
    if _sha256_bytes(audit_payload) != receipt["final_audit"]["sha256"]:
        raise ValueError("closeout final-audit binding mismatch")
    if _sha256_bytes(manifest_payload) != receipt["root_corpus_manifest"]["sha256"]:
        raise ValueError("closeout root-manifest binding mismatch")
    _atomic_write(out_dir / "snapshot_receipt.json", audit_payload)
    _atomic_write(out_dir / "REPORT.md", render_report(final_audit).encode("utf-8"))
    _atomic_write(out_dir / "final_corpus.manifest.json", manifest_payload)
    _atomic_write(out_dir / "m2_closeout_receipt.json", _canonical_bytes(receipt) + b"\n")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)

    return result


def render_report(audit: Mapping[str, Any]) -> str:
    immutable = audit["coverage"]["immutable_manifest_bound"]
    pending = audit["coverage"]["checkpoint_pending_unsealed"]
    checkpoint = audit["coverage"]["checkpoint_total"]
    tranche = audit["frozen_m3_tranche"]
    is_final = audit["status"] == "final_m2_completed"
    lines = [
        "# Q-Guided Branch Corpus Coverage and Duplication — "
        + ("Final Snapshot" if is_final else "Provisional Snapshot"),
        "",
        f"Status: `{audit['status']}`; classification: `quarantine`.",
        "",
        ("This receipt freezes the completed M1+M2 immutable corpus. " if is_final else
         "This receipt is provisional until M2 completes and its final immutable manifest is frozen. ")
        + "It reads only aggregate actor-visible shards, sidecars, checkpoint metadata, and the frozen "
        "tranche. Miner scratch directories and private simulator identities were not read or emitted. "
        "No labels, training, gameplay evaluation, policy selection, or protected runtime surfaces changed.",
        "",
        "## Snapshot",
        "",
    ]
    miner_rows = []
    for miner_id in ("M1", "M2"):
        miner = audit["miners"][miner_id]
        cursor = miner["cursor"]
        miner_rows.append(
            [
                miner_id,
                miner["matchup"],
                cursor["immutable_shards"],
                cursor["immutable_rows"],
                cursor["pending_unsealed_rows"],
                cursor["progress_eligible_rows"],
                miner["status"],
            ]
        )
    lines.extend(
        _markdown_table(
            ["miner", "matchup", "immutable shards", "immutable rows", "pending", "progress cursor", "status"],
            miner_rows,
        )
    )
    lines.extend(
        [
            "",
            f"Immutable corpus: `{immutable['rows']}` rows / `{immutable['sibling_actions']}` sibling actions; "
            f"checkpoint pending: `{pending['rows']}`; checkpoint total: `{checkpoint['rows']}`.",
        ]
    )
    for miner_id in ("M1", "M2"):
        cursor = audit["miners"][miner_id]["cursor"]
        if not cursor["run_manifest_cursor_coherent"]:
            lines.extend(
                [
                    "",
                    f"{miner_id} live-receipt skew: run-manifest `eligible_rows` is "
                    f"`{cursor['run_manifest_eligible_rows']}` while immutable shards plus pending reconstruct "
                    f"`{cursor['reconstructed_checkpoint_rows']}` and progress records "
                    f"`{cursor['progress_eligible_rows']}`. The progress cursor is coherent; the lagging "
                    "run-manifest scalar is not treated as a final freeze.",
                ]
            )
    if audit["status"] == "final_m2_completed":
        snapshot_note = (
            "The completed M1+M2 corpus is frozen by the root manifest; no further miner writes are "
            "authorized for this generation."
        )
    else:
        snapshot_note = (
            "A live M2 run may advance after this snapshot; its final manifest must be re-audited."
        )
    lines.extend(
        [
            "",
            "The JSON receipt contains every current shard path, shard SHA-256, sidecar SHA-256, and row count. "
            + snapshot_note,
            "",
            "## Exact M3 frozen data cursor",
            "",
            f"Frozen rows: `{tranche['row_count']}`; shard `{tranche['tranche_shard_sha256']}`; manifest "
            f"`{tranche['tranche_manifest_sha256']}`; dataset `{tranche['tranche_dataset_sha256']}`; ordered "
            f"source mapping `{tranche['ordered_source_mapping_sha256']}`.",
            "",
        ]
    )
    cursor_rows = []
    for miner_id in ("M1", "M2"):
        source = tranche["source_cursor"][miner_id]
        for shard in source["source_shards"]:
            cursor_rows.append(
                [
                    miner_id,
                    shard["shard"],
                    shard["row_offset_ranges_inclusive"],
                    shard["selected_row_count"],
                    shard["shard_sha256"],
                    shard["manifest_sha256"],
                ]
            )
    lines.extend(
        _markdown_table(
            ["miner", "source shard", "row offsets inclusive", "rows", "shard SHA-256", "manifest SHA-256"],
            cursor_rows,
        )
    )
    lines.extend(["", "## Coverage", ""])
    lines.extend(
        _markdown_table(
            ["action family", "sibling actions", "root rows containing family", "selected rows"],
            [
                [
                    family,
                    immutable["action_family_sibling_actions"].get(family, 0),
                    immutable["action_family_root_coverage"].get(family, 0),
                    immutable["selected_action_family_rows"].get(family, 0),
                ]
                for family in sorted(immutable["action_family_sibling_actions"])
            ],
        )
    )
    lines.extend(["", "Matchup rows: " + json.dumps(immutable["matchup_rows"], sort_keys=True) + "."])
    lines.extend(["", "Opponent-bucket rows: " + json.dumps(immutable["opponent_bucket_rows"], sort_keys=True) + "."])
    lines.extend(
        [
            "",
            "Ranking-confidence bins: "
            + json.dumps(immutable["uncertainty"]["action_ranking_confidence_bins"], sort_keys=True)
            + ".",
            "",
            "Maximum-Q-standard-error bins: "
            + json.dumps(immutable["uncertainty"]["maximum_q_standard_error_bins"], sort_keys=True)
            + ".",
            "",
            "Top-two Q-gap bins: " + json.dumps(immutable["q_gap"]["bins"], sort_keys=True) + ".",
            "",
            "Supervision eligibility: "
            + json.dumps(checkpoint["supervision_eligibility_rows"], sort_keys=True)
            + ". Pending rows remain unsealed and are excluded from immutable training supervision.",
            "",
            "## Duplication",
            "",
        ]
    )
    duplicate = audit["duplication"]["checkpoint_total"]
    lines.extend(
        _markdown_table(
            ["identity", "groups", "row instances", "extra instances", "cross-miner groups"],
            [
                [
                    name,
                    values["duplicate_groups"],
                    values["row_instances_in_duplicate_groups"],
                    values["extra_instances"],
                    values["cross_miner_groups"],
                ]
                for name, values in sorted(duplicate.items())
            ],
        )
    )
    semantic = checkpoint["semantic_equivalent_sibling_options"]
    lines.extend(
        [
            "",
            f"Semantic-equivalent sibling options occur in `{semantic['rows_with_repeated_semantic_identity']}` "
            f"rows with `{semantic['extra_option_instances']}` extra option instances. These are diagnostic "
            "equivalences only: different actor-visible source indices can still make them distinct legal actions.",
            "",
            "## Gate",
            "",
            (
                "Verdict: `FINAL_CORPUS_FROZEN`. All final rows pass aggregate-teacher manifest binding; "
                "this audit does not itself authorize policy control or candidate selection."
                if is_final
                else "Verdict: `PROVISIONAL`. Immutable rows pass aggregate-teacher manifest binding; pending "
                "M2 rows do not. The full anchored panel remains gated until M2 completes and this audit is "
                "rerun against the final frozen branch-data manifest."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_audit(out_dir: Path, audit: Mapping[str, Any]) -> None:
    out_dir = Path(out_dir).resolve()
    _atomic_write(out_dir / "snapshot_receipt.json", _canonical_bytes(audit) + b"\n")
    _atomic_write(out_dir / "REPORT.md", render_report(audit).encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--m1-run-dir", required=True)
    parser.add_argument("--m2-run-dir", required=True)
    parser.add_argument("--tranche-dir", required=True)
    parser.add_argument("--freeze-receipt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--snapshot-status",
        choices=("provisional_m2_running", "final_m2_completed"),
        default="provisional_m2_running",
    )
    parser.add_argument("--observed-at", default="")
    parser.add_argument("--m1-completion-receipt", default="")
    parser.add_argument("--m2-completion-receipt", default="")
    parser.add_argument("--provisional-receipt", default="")
    parser.add_argument("--minimum-useful-seconds", type=float, default=28_800.0)
    parser.add_argument("--m1-source-prefix", default="runs/m1")
    parser.add_argument("--m2-source-prefix", default="runs/m2")
    parser.add_argument("--generation-id", default="q_guided_first_wave_20260712")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    accepted_receipt = require_accepted_canonical_decision_receipt(
        args.canonical_decision_receipt,
        consumer=__name__,
    )
    args.canonical_decision_receipt = str(accepted_receipt.path)
    if args.snapshot_status == "final_m2_completed":
        required = {
            "--m1-completion-receipt": args.m1_completion_receipt,
            "--m2-completion-receipt": args.m2_completion_receipt,
            "--provisional-receipt": args.provisional_receipt,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("final corpus closeout requires " + ", ".join(missing))
        closeout = build_final_corpus_closeout(
            Path(args.m1_run_dir),
            Path(args.m2_run_dir),
            Path(args.tranche_dir),
            Path(args.freeze_receipt),
            Path(args.m1_completion_receipt),
            Path(args.m2_completion_receipt),
            Path(args.provisional_receipt),
            minimum_useful_seconds=args.minimum_useful_seconds,
            observed_at=args.observed_at or None,
            source_prefixes={"M1": args.m1_source_prefix, "M2": args.m2_source_prefix},
            generation_id=args.generation_id,
        )
        write_final_corpus_closeout(Path(args.out_dir), closeout)
        receipt = closeout["closeout_receipt"]
        print(
            "q-guided final corpus closeout "
            f"status={receipt['status']} "
            f"immutable_rows={receipt['final_immutable_rows']} "
            f"immutable_shards={receipt['final_immutable_shards']} "
            f"m3_consumed_rows={receipt['m3_consumed_unique_rows']}"
        )

        return 0
    audit = build_audit(
        Path(args.m1_run_dir),
        Path(args.m2_run_dir),
        Path(args.tranche_dir),
        Path(args.freeze_receipt),
        observed_at=args.observed_at or None,
        snapshot_status=args.snapshot_status,
    )
    write_audit(Path(args.out_dir), audit)
    print(
        "q-guided corpus audit "
        f"status={audit['status']} "
        f"immutable_rows={audit['coverage']['immutable_manifest_bound']['rows']} "
        f"pending_rows={audit['coverage']['checkpoint_pending_unsealed']['rows']} "
        f"m3_rows={audit['frozen_m3_tranche']['row_count']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
