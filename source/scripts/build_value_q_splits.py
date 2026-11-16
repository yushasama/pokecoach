"""Build catalog-bound, game-grouped chronological splits for the value/Q lab."""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sqlite3
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deepmew.value_lab.contracts as value_contracts
from deepmew.value_lab.contracts import (
    ROW_EXCLUSION_SCHEMA_VERSION,
    SPLIT_SCHEMA_VERSION,
    STATE_ACTION_SIGNATURE_VERSION,
    canonical_state_action_signature,
)


GOLD_RECEIPT_SCHEMA_VERSION = "deepmew.value_q_prospective_gold_receipt.v1"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
REPORT_SPLIT_NAMES = SPLIT_NAMES
RATIOS = (0.80, 0.06, 0.06, 0.08)
CATALOG_SCHEMA_VERSION = "deepmew.value_q_dataset_catalog.v1"
PROSPECTIVE_PROJECTION_VERSION = "deepmew.public_trace_projection.v2"
PROSPECTIVE_MANIFEST_VERSION = "deepmew.prospective_trace.manifest.v1"
PROSPECTIVE_ROW_VERSION = "deepmew.prospective_trace.public_decision.v2"

# These source contracts were verified in Phase 0/2 to carry one global
# game_index/episode order even though physical shards overlap in index range.
GLOBAL_ORDER_SOURCE_IDS = frozenset({"unified_r2", "r15_hard", "kaggle_ingest"})


def _sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(values)

    def at(frac: float) -> int:
        return ordered[min(len(ordered) - 1, int(frac * (len(ordered) - 1)))]

    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _constant_memory_split_mask_counts(connection: sqlite3.Connection) -> dict[int, int]:
    """Count the 15 possible four-fold masks without a 9M-row GROUP BY sort."""
    possible_masks = range(1, 1 << len(SPLIT_NAMES))
    expressions = [
        f"SUM(CASE WHEN split_mask = {mask} THEN 1 ELSE 0 END)"
        for mask in possible_masks
    ]
    counts = connection.execute(f"SELECT {','.join(expressions)} FROM signatures").fetchone()

    return {
        mask: int(count)
        for mask, count in zip(possible_masks, counts)
        if count
    }


def load_catalog(catalog_path: str) -> tuple[dict, str]:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"unexpected catalog schema {catalog.get('schema_version')}")
    if catalog.get("duplicate_game_seat_blocks_across_units") != 0:
        raise ValueError("catalog contains duplicate game-seat blocks")
    if catalog.get("duplicate_game_content") != 0:
        raise ValueError("catalog contains duplicate game content")
    included = catalog.get("included_units")
    if not isinstance(included, list) or not included:
        raise ValueError("catalog lacks explicit included_units allowlist")
    expected_units = sum(int(source["units"]) for source in catalog["sources"].values())
    if len(included) != expected_units:
        raise ValueError(f"catalog allowlist has {len(included)} units, expected {expected_units}")

    return catalog, _sha256(catalog_path)


def _load_manifest(canonical_root: str, unit: dict) -> tuple[dict, Path]:
    path = Path(canonical_root) / unit["manifest_path"]
    if not path.is_file() or _sha256(path) != unit["manifest_sha256"]:
        raise ValueError(f"unit manifest hash mismatch: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"included unit is not completed: {path}")
    if (manifest.get("source_id"), manifest.get("unit_id")) != (unit["source_id"], unit["unit_id"]):
        raise ValueError(f"unit identity mismatch: {path}")

    return manifest, path


def _source_batch_metadata(manifest: dict) -> tuple[str, int | None]:
    """Derive a run/round batch identity and captured source time from provenance."""
    fingerprints = manifest.get("source_fingerprint")
    if not isinstance(fingerprints, list) or not fingerprints:

        return manifest["unit_id"], None
    batch_paths: set[str] = set()
    mtimes: list[int] = []
    for fingerprint in fingerprints:
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("path"), str):
            raise ValueError(f"invalid source fingerprint in {manifest['source_id']}/{manifest['unit_id']}")
        source_path = PureWindowsPath(fingerprint["path"].replace("/", "\\"))
        parent = source_path.parent
        if parent.name.lower() == "shards":
            parent = parent.parent
        batch_paths.add(str(parent).lower())
        mtime = fingerprint.get("mtime")
        if isinstance(mtime, bool) or not isinstance(mtime, int):
            raise ValueError(f"source fingerprint lacks integer mtime in {manifest['source_id']}/{manifest['unit_id']}")
        mtimes.append(mtime)

    return "|".join(sorted(batch_paths)), min(mtimes)


def load_games(canonical_root: str, catalog: dict) -> tuple[dict[str, dict], list[dict]]:
    games: dict[str, dict] = {}
    unit_bindings: list[dict] = []
    content_owner: dict[str, str] = {}
    for unit_rank, unit in enumerate(catalog["included_units"]):
        manifest, manifest_path = _load_manifest(canonical_root, unit)
        source_batch_id, source_batch_mtime = _source_batch_metadata(manifest)
        unit_bindings.append({
            "source_id": manifest["source_id"],
            "unit_id": manifest["unit_id"],
            "manifest_sha256": unit["manifest_sha256"],
            "unit_rank": unit_rank,
            "chronology": manifest["chronology"],
            "era_rank": manifest["era_rank"],
            "source_batch_id": source_batch_id,
            "source_batch_mtime": source_batch_mtime,
        })
        unit_dir = manifest_path.parent
        for part in manifest.get("parts", []):
            games_name = f"{part['name']}.games.jsonl.gz"
            games_path = unit_dir / games_name
            expected_hash = part["files_sha256"].get(games_name)
            if not expected_hash or _sha256(games_path) != expected_hash:
                raise ValueError(f"games sidecar hash mismatch: {games_path}")
            with gzip.open(games_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = record["canonical_game_id"]
                    content_hash = record.get("game_content_hash")
                    if not isinstance(content_hash, str) or not content_hash:
                        raise ValueError(f"game lacks content hash: {key}")
                    prior_content = content_owner.get(content_hash)
                    if prior_content is not None and prior_content != key:
                        raise ValueError(f"duplicate game content {content_hash}: {prior_content}, {key}")
                    content_owner[content_hash] = key
                    entry = games.get(key)
                    if entry is None:
                        entry = {
                            "canonical_game_id": key,
                            "source_game_id": record["source_game_id"],
                            "source_id": record["source_id"],
                            "source_schema": record["source_schema"],
                            "unit_id": record["unit_id"],
                            "unit_rank": unit_rank,
                            "source_batch_id": source_batch_id,
                            "source_batch_mtime": source_batch_mtime,
                            "era_rank": manifest["era_rank"],
                            "chronology_contract": manifest["chronology"],
                            "order_key": record.get("order_key"),
                            "game_content_hash": content_hash,
                            "rows": 0,
                            "block_lengths": [],
                            "seats": set(),
                            "actor_outcomes": [],
                            "policy_ids": set(),
                            "opponent_policy_ids": set(),
                            "deck_hashes": set(),
                            "opponent_deck_hashes": set(),
                            "matchups": set(),
                        }
                        games[key] = entry
                    elif (
                        entry["source_id"] != record["source_id"]
                        or entry["unit_id"] != record["unit_id"]
                        or entry["game_content_hash"] != content_hash
                    ):
                        raise ValueError(f"inconsistent blocks for game {key}")
                    entry["rows"] += int(record["block_len"])
                    entry["block_lengths"].append(int(record["block_len"]))
                    entry["seats"].add(int(record["seat"]))
                    entry["actor_outcomes"].append(float(record["z"]))
                    for field, target in (
                        ("policy_id", "policy_ids"),
                        ("opponent_policy_id", "opponent_policy_ids"),
                        ("deck_hash", "deck_hashes"),
                        ("opponent_deck_hash", "opponent_deck_hashes"),
                    ):
                        if record.get(field) is not None:
                            entry[target].add(str(record[field]))
                    matchup = record.get("matchup_cell_key") or record.get("bucket")
                    if matchup is not None:
                        entry["matchups"].add(str(matchup))
    if len(games) != int(catalog["totals"]["games_kept"]):
        raise ValueError(f"loaded {len(games)} games, catalog says {catalog['totals']['games_kept']}")
    if sum(entry["rows"] for entry in games.values()) != int(catalog["totals"]["rows_kept"]):
        raise ValueError("loaded row total does not match catalog")

    return games, unit_bindings


def choose_chronology(games: dict[str, dict]) -> dict[str, dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for entry in games.values():
        by_source[entry["source_id"]].append(entry)
    contracts: dict[str, dict] = {}
    for source_id, entries in by_source.items():
        contract = entries[0]["chronology_contract"]
        order_values = [entry["order_key"] for entry in entries]
        has_order = all(isinstance(value, int) for value in order_values)
        batch_ids = {
            entry.get("source_batch_id", f"unit:{entry['unit_rank']}")
            for entry in entries
        }
        batch_mtimes: dict[str, int | None] = {}
        batch_unit_ranks: dict[str, int] = {}
        for entry in entries:
            batch_id = entry.get("source_batch_id", f"unit:{entry['unit_rank']}")
            batch_unit_ranks[batch_id] = min(batch_unit_ranks.get(batch_id, entry["unit_rank"]), entry["unit_rank"])
            mtime = entry.get("source_batch_mtime")
            prior = batch_mtimes.get(batch_id)
            if isinstance(mtime, int):
                batch_mtimes[batch_id] = mtime if prior is None else min(prior, mtime)
            elif batch_id not in batch_mtimes:
                batch_mtimes[batch_id] = None
        has_batch_time = all(isinstance(batch_mtimes[batch_id], int) for batch_id in batch_ids)
        ordered_batches = sorted(
            batch_ids,
            key=lambda batch_id: (
                batch_mtimes[batch_id] if has_batch_time else batch_unit_ranks[batch_id],
                batch_id,
            ),
        )
        batch_rank = {batch_id: rank for rank, batch_id in enumerate(ordered_batches)}
        unit_ranges: dict[int, tuple[int, int]] = {}
        batch_ranges: dict[str, tuple[int, int]] = {}
        if has_order:
            grouped: dict[int, list[int]] = defaultdict(list)
            grouped_batches: dict[str, list[int]] = defaultdict(list)
            for entry in entries:
                grouped[entry["unit_rank"]].append(entry["order_key"])
                grouped_batches[entry.get("source_batch_id", f"unit:{entry['unit_rank']}")].append(entry["order_key"])
            unit_ranges = {rank: (min(values), max(values)) for rank, values in grouped.items()}
            batch_ranges = {
                batch_id: (min(values), max(values))
                for batch_id, values in grouped_batches.items()
            }
        if source_id in GLOBAL_ORDER_SOURCE_IDS and has_order:
            strategy = "global_order_key"
            confidence = "strict_episode_order" if contract.startswith("episode_id") else "global_generation_order"
        elif contract.startswith("run_round") and has_order:
            if not has_batch_time:
                raise ValueError(f"{source_id} run/round chronology lacks captured source mtimes")
            strategy = "source_batch_then_order"
            confidence = "run_round_batch_order"
        elif has_order and len(batch_ranges) == 1:
            strategy = "global_order_key"
            confidence = "within_source_generation_order"
        elif has_order:
            sorted_ranges = [batch_ranges[batch_id] for batch_id in ordered_batches]
            disjoint = all(left[1] < right[0] for left, right in zip(sorted_ranges, sorted_ranges[1:]))
            strategy = "global_order_key" if disjoint else "source_batch_then_order"
            confidence = "cross_batch_generation_order" if disjoint else "source_batch_order"
            if strategy == "source_batch_then_order" and not has_batch_time:
                raise ValueError(f"{source_id} overlapping batches lack captured source mtimes")
        else:
            strategy = "deterministic_grouped_fallback"
            confidence = "non_chronological_fallback"
        contracts[source_id] = {
            "declared_contract": contract,
            "strategy": strategy,
            "confidence": confidence,
            "units": len(unit_ranges) if unit_ranges else len({entry['unit_rank'] for entry in entries}),
            "batches": len(batch_ids),
            "batch_order_basis": "captured_source_mtime" if has_batch_time else "catalog_unit_fallback",
            "missing_order_keys": sum(not isinstance(value, int) for value in order_values),
        }
        for entry in entries:
            order = entry["order_key"] if isinstance(entry["order_key"], int) else 0
            if strategy == "global_order_key":
                entry["chronology_key"] = (order, entry["unit_rank"], entry["canonical_game_id"])
            elif strategy == "source_batch_then_order":
                batch_id = entry.get("source_batch_id", f"unit:{entry['unit_rank']}")
                entry["chronology_key"] = (batch_rank[batch_id], order, entry["canonical_game_id"])
            else:
                entry["chronology_key"] = (entry["unit_rank"], entry["canonical_game_id"])
        prefix_counts = Counter(entry["chronology_key"][:-1] for entry in entries)
        contracts[source_id]["tied_chronology_groups"] = sum(count > 1 for count in prefix_counts.values())
        contracts[source_id]["games_in_tied_chronology_groups"] = sum(
            count for count in prefix_counts.values() if count > 1
        )
        contracts[source_id]["batch_order_sha256"] = _stable_hash([
            {"batch_id": batch_id, "captured_mtime": batch_mtimes[batch_id]}
            for batch_id in ordered_batches
        ])

    return contracts


def assign_splits(games: dict[str, dict]) -> dict[str, str]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for entry in games.values():
        by_source[entry["source_id"]].append(entry)
    assignment: dict[str, str] = {}
    for entries in by_source.values():
        entries.sort(key=lambda entry: entry["chronology_key"])
        n = len(entries)
        allowed_boundaries = [0]
        for index in range(1, n + 1):
            if index == n or entries[index - 1]["chronology_key"][:-1] != entries[index]["chronology_key"][:-1]:
                allowed_boundaries.append(index)
        boundaries: list[int] = []
        cumulative = 0.0
        for ratio in RATIOS[:-1]:
            cumulative += ratio
            target = int(round(cumulative * n))
            previous = boundaries[-1] if boundaries else 0
            candidates = [value for value in allowed_boundaries if value >= previous]
            boundaries.append(min(candidates, key=lambda value: (abs(value - target), value)))
        for index, entry in enumerate(entries):
            if index < boundaries[0]:
                split = "train"
            elif index < boundaries[1]:
                split = "validation"
            elif index < boundaries[2]:
                split = "calibration"
            else:
                split = "test"
            assignment[entry["canonical_game_id"]] = split

    return assignment


def load_prospective_gold(root: str | None) -> dict:
    if not root:
        return {"schema_version": GOLD_RECEIPT_SCHEMA_VERSION, "status": "not_provided", "games": 0, "decisions": 0}
    manifests = sorted(glob.glob(os.path.join(root, "tranches", "*", "manifest.json")))
    if not manifests:
        raise ValueError(f"no prospective tranche manifests under {root}")
    games: set[str] = set()
    decision_ids: set[str] = set()
    tranche_ids: set[str] = set()
    tranche_sequences: set[int] = set()
    decisions = 0
    tranche_receipts = []
    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PROSPECTIVE_MANIFEST_VERSION:
            raise ValueError(f"unexpected prospective manifest schema: {manifest_path}")
        tranche_id = manifest.get("tranche_id")
        tranche_sequence = manifest.get("tranche_sequence")
        if not isinstance(tranche_id, str) or not isinstance(tranche_sequence, int):
            raise ValueError(f"prospective manifest lacks tranche identity: {manifest_path}")
        if tranche_id in tranche_ids or tranche_sequence in tranche_sequences:
            raise ValueError(f"duplicate prospective tranche identity: {manifest_path}")
        tranche_ids.add(tranche_id)
        tranche_sequences.add(tranche_sequence)
        if manifest.get("public_projection_version") != PROSPECTIVE_PROJECTION_VERSION:
            raise ValueError(f"prospective manifest is not projection v2: {manifest_path}")
        if manifest.get("information_mode") != "public_belief":
            raise ValueError(f"prospective manifest is not strict public: {manifest_path}")
        safety = manifest.get("safety") or {}
        required_safety = ("completed", "crashes", "timeouts", "illegal")
        if any(field not in safety for field in required_safety):
            raise ValueError(f"prospective manifest lacks required safety counters: {manifest_path}")
        if any(int(safety[field]) for field in ("crashes", "timeouts", "illegal")):
            raise ValueError(f"prospective gold safety failure: {manifest_path}")
        shard_paths = sorted(glob.glob(os.path.join(os.path.dirname(manifest_path), "shards", "*.jsonl.gz")))
        if not shard_paths:
            raise ValueError(f"prospective tranche has no durable shards: {manifest_path}")
        tranche_decisions = 0
        tranche_games: set[str] = set()
        tranche_decision_ids: set[str] = set()
        shard_receipts = []
        for shard_path in shard_paths:
            shard_decisions = 0
            with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("schema_version") != PROSPECTIVE_ROW_VERSION:
                        raise ValueError(f"unexpected prospective row schema in {shard_path}")
                    if row.get("public_projection_version") != PROSPECTIVE_PROJECTION_VERSION:
                        raise ValueError(f"prospective row projection mismatch in {shard_path}")
                    if row.get("tranche_id") != manifest.get("tranche_id"):
                        raise ValueError(f"prospective row tranche mismatch in {shard_path}")
                    if row.get("tranche_sequence") != manifest.get("tranche_sequence"):
                        raise ValueError(f"prospective row tranche sequence mismatch in {shard_path}")
                    if any(bool(row.get(field)) for field in ("crashed", "timed_out", "illegal")):
                        raise ValueError(f"prospective row safety failure in {shard_path}")
                    decision_id = row.get("decision_id")
                    game_id = row.get("game_id")
                    if not isinstance(decision_id, str) or not decision_id or not isinstance(game_id, str) or not game_id:
                        raise ValueError(f"prospective row lacks durable IDs in {shard_path}")
                    if decision_id in decision_ids or decision_id in tranche_decision_ids:
                        raise ValueError(f"duplicate prospective decision_id {decision_id}")
                    tranche_decision_ids.add(decision_id)
                    tranche_decisions += 1
                    shard_decisions += 1
                    tranche_games.add(game_id)
            shard_receipts.append({
                "path": os.path.relpath(shard_path, root),
                "sha256": _sha256(shard_path),
                "decisions": shard_decisions,
            })
        coverage = manifest.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError(f"prospective manifest lacks coverage receipt: {manifest_path}")
        if int(coverage.get("decisions", -1)) != tranche_decisions:
            raise ValueError(f"prospective decision coverage mismatch: {manifest_path}")
        if int(coverage.get("games", -1)) != len(tranche_games):
            raise ValueError(f"prospective game coverage mismatch: {manifest_path}")
        completed_decision_ids = coverage.get("completed_decision_ids") or []
        completed_game_ids = coverage.get("completed_game_ids") or []
        if len(completed_decision_ids) != len(set(completed_decision_ids)):
            raise ValueError(f"prospective coverage repeats decision IDs: {manifest_path}")
        if len(completed_game_ids) != len(set(completed_game_ids)):
            raise ValueError(f"prospective coverage repeats game IDs: {manifest_path}")
        if set(completed_decision_ids) != tranche_decision_ids:
            raise ValueError(f"prospective completed decision IDs mismatch: {manifest_path}")
        if set(completed_game_ids) != tranche_games:
            raise ValueError(f"prospective completed game IDs mismatch: {manifest_path}")
        if int(safety["completed"]) != len(tranche_games):
            raise ValueError(f"prospective completed safety count mismatch: {manifest_path}")
        if games.intersection(tranche_games):
            raise ValueError(f"prospective game appears in multiple tranches: {manifest_path}")
        decisions += tranche_decisions
        games.update(tranche_games)
        decision_ids.update(tranche_decision_ids)
        tranche_receipts.append({
            "tranche_id": manifest["tranche_id"],
            "tranche_sequence": manifest["tranche_sequence"],
            "manifest_sha256": _sha256(manifest_path),
            "games": len(tranche_games),
            "decisions": tranche_decisions,
            "decision_ids_sha256": hashlib.sha256("\n".join(sorted(tranche_decision_ids)).encode()).hexdigest(),
            "shards": shard_receipts,
        })

    if sorted(tranche_sequences) != list(range(len(tranche_sequences))):
        raise ValueError("prospective tranche sequences are not contiguous from zero")
    receipt = {
        "schema_version": GOLD_RECEIPT_SCHEMA_VERSION,
        "status": "pristine_causality_smoke_only",
        "root": root,
        "projection_version": PROSPECTIVE_PROJECTION_VERSION,
        "information_mode": "public_belief",
        "games": len(games),
        "decisions": decisions,
        "game_ids_sha256": hashlib.sha256("\n".join(sorted(games)).encode()).hexdigest(),
        "tranches": tranche_receipts,
        "statistical_role": "safety_and_causality_smoke_not_powered_model_selection",
    }
    receipt["content_binding_sha256"] = _stable_hash({
        "game_ids": sorted(games),
        "decision_ids": sorted(decision_ids),
        "tranches": tranche_receipts,
    })

    return receipt


def build_summary(games: dict[str, dict], assignment: dict[str, str], chronology: dict) -> dict:
    game_counts = {split: Counter() for split in REPORT_SPLIT_NAMES}
    row_counts = {split: Counter() for split in REPORT_SPLIT_NAMES}
    policies = {split: Counter() for split in REPORT_SPLIT_NAMES}
    matchups = {split: Counter() for split in REPORT_SPLIT_NAMES}
    outcomes = {split: Counter() for split in REPORT_SPLIT_NAMES}
    lengths = {split: [] for split in REPORT_SPLIT_NAMES}
    boundaries: dict[str, dict] = defaultdict(dict)
    for key, split in assignment.items():
        entry = games[key]
        source_id = entry["source_id"]
        game_counts[split][source_id] += 1
        row_counts[split][source_id] += entry["rows"]
        lengths[split].extend(entry["block_lengths"])
        for policy_id in entry["policy_ids"]:
            policies[split][policy_id] += 1
        for matchup in entry["matchups"] or {"unknown"}:
            matchups[split][matchup] += 1
        for z in entry["actor_outcomes"]:
            outcomes[split]["win" if z == 1.0 else "loss" if z == 0.0 else "draw"] += 1
        if split in SPLIT_NAMES:
            source_boundary = boundaries[source_id].setdefault(split, {"min": None, "max": None, "games": 0})
            rendered = list(entry["chronology_key"][:-1])
            source_boundary["min"] = rendered if source_boundary["min"] is None else min(source_boundary["min"], rendered)
            source_boundary["max"] = rendered if source_boundary["max"] is None else max(source_boundary["max"], rendered)
            source_boundary["games"] += 1
    chronology_violations = []
    for source_id, split_bounds in boundaries.items():
        previous = None
        for split in SPLIT_NAMES:
            current = split_bounds.get(split)
            if current is None:
                continue
            if previous is not None and previous["max"] > current["min"]:
                chronology_violations.append({"source_id": source_id, "before": previous, "after": current})
            previous = current
    if chronology_violations:
        raise ValueError(f"chronology boundary violations: {chronology_violations[:3]}")

    return {
        "counts_games": {split: dict(game_counts[split]) for split in REPORT_SPLIT_NAMES},
        "counts_rows": {split: dict(row_counts[split]) for split in REPORT_SPLIT_NAMES},
        "policy_distribution": {split: dict(policies[split]) for split in REPORT_SPLIT_NAMES},
        "matchup_distribution": {split: dict(matchups[split]) for split in REPORT_SPLIT_NAMES},
        "actor_outcome_distribution": {split: dict(outcomes[split]) for split in REPORT_SPLIT_NAMES},
        "sequence_length_distribution": {split: _quantiles(lengths[split]) for split in REPORT_SPLIT_NAMES},
        "chronology_contracts": chronology,
        "chronology_boundaries": dict(boundaries),
        "chronology_boundary_violations": 0,
        "game_overlap_across_splits": 0,
        "game_content_overlap_across_splits": 0,
        "mirror_overlap": {"status": "not_instrumented", "reason": "no proven mirror-normalized identity exists"},
        "state_action_overlap": {"status": "pending_exact_stream_audit"},
        "policy_generation_order": {"status": "not_instrumented", "reason": "historical rows lack normalized checkpoint generation indices"},
        "source_policy_features_in_model_input": False,
    }


def _iter_transition_parts(canonical_root: str, catalog: dict):
    for unit in catalog["included_units"]:
        manifest, manifest_path = _load_manifest(canonical_root, unit)
        unit_dir = manifest_path.parent
        for part in manifest.get("parts", []):
            transitions_name = f"{part['name']}.transitions.jsonl.gz"
            transitions_path = unit_dir / transitions_name
            transitions_hash = part["files_sha256"].get(transitions_name)
            if not transitions_hash or _sha256(transitions_path) != transitions_hash:
                raise ValueError(f"transition sidecar hash mismatch: {transitions_path}")
            npz_name = f"{part['name']}.npz"
            npz_path = unit_dir / npz_name
            npz_hash = part["files_sha256"].get(npz_name)
            if not npz_hash or _sha256(npz_path) != npz_hash:
                raise ValueError(f"canonical feature part hash mismatch: {npz_path}")
            with np.load(npz_path) as arrays:
                features = arrays["X"].astype(np.float32, copy=False)
            if features.shape[0] != int(part["rows"]):
                raise ValueError(f"feature row count mismatch: {npz_path}")
            yield {
                "source_id": manifest["source_id"],
                "unit_id": manifest["unit_id"],
                "part": part["name"],
                "part_rows": int(part["rows"]),
                "npz_path": npz_path,
                "npz_sha256": npz_hash,
                "transitions_path": transitions_path,
                "transitions_sha256": transitions_hash,
            }, features


def _iter_part_transitions(part_info: dict, features: np.ndarray):
    rows = 0
    with gzip.open(part_info["transitions_path"], "rt", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index >= features.shape[0]:
                raise ValueError(f"too many transition rows: {part_info['transitions_path']}")
            record = json.loads(line)
            if int(record.get("row", -1)) != row_index:
                raise ValueError(f"transition row alignment mismatch: {record.get('decision_id')}")
            rows += 1
            yield row_index, features[row_index], record
    if rows != features.shape[0]:
        raise ValueError(f"transition row count mismatch: {part_info['transitions_path']}")


def audit_state_actions(
    canonical_root: str,
    catalog: dict,
    assignment: dict[str, str],
    db_path: str,
) -> dict:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("CREATE TABLE signatures (signature BLOB PRIMARY KEY, split_mask INTEGER NOT NULL)")
    split_bits = {name: 1 << index for index, name in enumerate(REPORT_SPLIT_NAMES)}
    action_families = {name: Counter() for name in REPORT_SPLIT_NAMES}
    rows_seen = 0
    batch: list[tuple[bytes, int]] = []
    sql = (
        "INSERT INTO signatures(signature, split_mask) VALUES (?, ?) "
        "ON CONFLICT(signature) DO UPDATE SET split_mask = split_mask | excluded.split_mask"
    )
    for part_info, features in _iter_transition_parts(canonical_root, catalog):
        for _, feature_row, record in _iter_part_transitions(part_info, features):
            game_id = record["canonical_game_id"]
            split = assignment.get(game_id)
            if split is None:
                raise ValueError(f"transition has no split assignment: {game_id}")
            chosen = record["chosen_action"]
            batch.append((canonical_state_action_signature(feature_row, record), split_bits[split]))
            rows_seen += 1
            semantics = chosen.get("semantics") if isinstance(chosen, dict) else None
            if isinstance(semantics, list) and semantics:
                for semantic in semantics:
                    family = semantic.get("option_type") if isinstance(semantic, dict) else None
                    action_families[split][str(family or "unknown_semantic")] += 1
            elif isinstance(chosen, dict) and chosen.get("indices") == []:
                action_families[split]["EMPTY_SELECTION"] += 1
            else:
                legal = record.get("legal_action_set") or []
                for index in chosen.get("indices", []) if isinstance(chosen, dict) else []:
                    if chosen.get("index_space") == "option_position" and 0 <= index < len(legal):
                        option = legal[index]
                        action_families[split][f"RAW_TYPE_{option.get('type', 'unknown')}"] += 1
            if len(batch) >= 20_000:
                connection.executemany(sql, batch)
                batch.clear()
                connection.commit()
    del part_info, features, feature_row, record
    if batch:
        connection.executemany(sql, batch)
        connection.commit()
    mask_counts = _constant_memory_split_mask_counts(connection)
    distinct = sum(mask_counts.values())
    cross_signatures = [
        bytes(signature).hex()
        for (signature,) in connection.execute(
            "SELECT signature FROM signatures WHERE (split_mask & (split_mask - 1)) != 0"
        )
    ]
    cross_split = sum(count for mask, count in mask_counts.items() if mask & (mask - 1))
    if len(cross_signatures) != cross_split:
        raise ValueError("state-action audit crossing-signature inventory is inconsistent")
    distinct_signature_split_pairs = sum(mask.bit_count() * count for mask, count in mask_counts.items())
    connection.close()
    if rows_seen != int(catalog["totals"]["rows_kept"]):
        raise ValueError(f"state-action audit saw {rows_seen} rows, expected {catalog['totals']['rows_kept']}")

    return {
        "status": "pass" if cross_split == 0 else "fail",
        "signature_version": STATE_ACTION_SIGNATURE_VERSION,
        "rows_scanned": rows_seen,
        "distinct_state_action_signatures": distinct,
        "distinct_signature_split_pairs": distinct_signature_split_pairs,
        "duplicate_rows_within_splits": rows_seen - distinct_signature_split_pairs,
        "cross_split_signature_count": cross_split,
        "cross_split_signatures": cross_signatures,
        "split_mask_counts": {str(mask): count for mask, count in sorted(mask_counts.items())},
        "action_family_coverage": {split: dict(counter) for split, counter in action_families.items()},
        "audit_db_path": str(path),
        "audit_db_bytes": path.stat().st_size,
        "audit_db_sha256": _sha256(path),
        "signature_implementation_sha256": _sha256(value_contracts.__file__),
    }


def materialize_row_exclusions(
    canonical_root: str,
    catalog: dict,
    signature_hexes: list[str],
    assignment: dict[str, str],
    audit_db_path: str,
) -> dict:
    targets = {bytes.fromhex(value) for value in signature_hexes}
    audit_root = Path(audit_db_path).parent
    mask_root = audit_root / "row_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    effective_db_path = audit_root / "state_action_effective.sqlite"
    if effective_db_path.exists():
        effective_db_path.unlink()
    connection = sqlite3.connect(effective_db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE signatures (signature BLOB PRIMARY KEY, split_mask INTEGER NOT NULL)")
    sql = (
        "INSERT INTO signatures(signature, split_mask) VALUES (?, ?) "
        "ON CONFLICT(signature) DO UPDATE SET split_mask = split_mask | excluded.split_mask"
    )
    split_bits = {name: 1 << index for index, name in enumerate(SPLIT_NAMES)}
    games: set[str] = set()
    rows_by_split = Counter()
    games_by_split: dict[str, set[str]] = defaultdict(set)
    eligible_by_split = Counter()
    part_receipts: list[dict] = []
    effective_batch: list[tuple[bytes, int]] = []
    rows_seen = 0
    for part_info, features in _iter_transition_parts(canonical_root, catalog):
        eligible = np.ones(features.shape[0], dtype=np.bool_)
        for row_index, feature_row, record in _iter_part_transitions(part_info, features):
            game_id = record["canonical_game_id"]
            split = assignment[game_id]
            signature = canonical_state_action_signature(feature_row, record)
            rows_seen += 1
            if signature in targets:
                eligible[row_index] = False
                games.add(game_id)
                rows_by_split[split] += 1
                games_by_split[split].add(game_id)
            else:
                eligible_by_split[split] += 1
                effective_batch.append((signature, split_bits[split]))
                if len(effective_batch) >= 20_000:
                    connection.executemany(sql, effective_batch)
                    effective_batch.clear()
                    connection.commit()
        mask_dir = mask_root / part_info["source_id"] / part_info["unit_id"]
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{part_info['part']}.target_mask.npz"
        tmp = str(mask_path) + ".tmp.npz"
        np.savez_compressed(tmp, is_target_eligible=eligible)
        os.replace(tmp, mask_path)
        part_receipts.append({
            "source_id": part_info["source_id"],
            "unit_id": part_info["unit_id"],
            "part": part_info["part"],
            "rows": int(eligible.shape[0]),
            "excluded_rows": int((~eligible).sum()),
            "mask_path": os.path.relpath(mask_path, audit_root),
            "mask_sha256": _sha256(mask_path),
            "input_npz_sha256": part_info["npz_sha256"],
            "input_transitions_sha256": part_info["transitions_sha256"],
        })
    del part_info, features, feature_row, record, eligible
    if effective_batch:
        connection.executemany(sql, effective_batch)
        connection.commit()
    mask_counts = _constant_memory_split_mask_counts(connection)
    effective_cross_split = sum(count for mask, count in mask_counts.items() if mask & (mask - 1))
    effective_distinct = sum(mask_counts.values())
    connection.close()
    effective_db_sha256 = _sha256(effective_db_path)
    if rows_seen != int(catalog["totals"]["rows_kept"]):
        raise ValueError(f"mask materialization saw {rows_seen} rows, expected {catalog['totals']['rows_kept']}")
    if effective_cross_split:
        raise ValueError(f"row exclusions left {effective_cross_split} cross-split signatures")
    manifest = {
        "schema_version": ROW_EXCLUSION_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalog_sha256": catalog["_catalog_sha256"],
        "assignment_sha256": catalog["_assignment_sha256"],
        "signature_version": STATE_ACTION_SIGNATURE_VERSION,
        "signature_implementation_sha256": _sha256(value_contracts.__file__),
        "excluded_signatures": sorted(signature_hexes),
        "excluded_signature_sha256": hashlib.sha256("\n".join(sorted(signature_hexes)).encode()).hexdigest(),
        "context_rows": rows_seen,
        "eligible_target_rows": rows_seen - sum(rows_by_split.values()),
        "excluded_target_rows": sum(rows_by_split.values()),
        "eligible_target_rows_by_split": dict(eligible_by_split),
        "excluded_target_rows_by_split": dict(rows_by_split),
        "affected_games": len(games),
        "affected_games_by_split": {split: len(values) for split, values in games_by_split.items()},
        "affected_game_ids_sha256": hashlib.sha256("\n".join(sorted(games)).encode()).hexdigest(),
        "effective_distinct_signatures": effective_distinct,
        "effective_cross_split_signature_count": effective_cross_split,
        "effective_db_path": str(effective_db_path),
        "effective_db_sha256": effective_db_sha256,
        "source_audit_db_path": str(audit_db_path),
        "source_audit_db_sha256": _sha256(audit_db_path),
        "parts": part_receipts,
    }
    manifest_path = audit_root / "ROW_EXCLUSION_MANIFEST.json"
    tmp_manifest = str(manifest_path) + ".tmp"
    Path(tmp_manifest).write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    return {
        "affected_rows": sum(rows_by_split.values()),
        "affected_rows_by_split": dict(rows_by_split),
        "affected_games": len(games),
        "affected_games_by_split": {split: len(values) for split, values in games_by_split.items()},
        "affected_game_ids_sha256": hashlib.sha256("\n".join(sorted(games)).encode()).hexdigest(),
        "eligible_target_rows": rows_seen - sum(rows_by_split.values()),
        "eligible_target_rows_by_split": dict(eligible_by_split),
        "effective_distinct_signatures": effective_distinct,
        "effective_cross_split_signature_count": effective_cross_split,
        "effective_db_sha256": effective_db_sha256,
        "row_exclusion_manifest_path": str(manifest_path),
        "row_exclusion_manifest_sha256": _sha256(manifest_path),
        "row_mask_parts": len(part_receipts),
    }


def build_split_manifest(
    canonical_root: str,
    catalog_path: str,
    prospective_gold_root: str | None,
    state_action_db: str | None = None,
) -> dict:
    catalog, catalog_sha = load_catalog(catalog_path)
    games, unit_bindings = load_games(canonical_root, catalog)
    chronology = choose_chronology(games)
    assignment = assign_splits(games)
    assignment_hash = _stable_hash({
        "catalog_sha256": catalog_sha,
        "ratios": RATIOS,
        "assignment": sorted(assignment.items()),
    })
    catalog["_catalog_sha256"] = catalog_sha
    catalog["_assignment_sha256"] = assignment_hash
    summary = build_summary(games, assignment, chronology)
    gold = load_prospective_gold(prospective_gold_root)
    if not state_action_db:
        raise ValueError("accepted split manifest requires --state-action-db exact audit")
    requested_db = Path(state_action_db)
    audit_run_id = (
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
        f"{assignment_hash[:12]}-{uuid.uuid4().hex[:8]}"
    )
    audit_root = requested_db.parent / "runs" / audit_run_id
    audit_root.mkdir(parents=True, exist_ok=False)
    audit_db_path = audit_root / requested_db.name
    state_action_audit = audit_state_actions(canonical_root, catalog, assignment, str(audit_db_path))
    signature_hexes = state_action_audit["cross_split_signatures"]
    exclusion_coverage = materialize_row_exclusions(
        canonical_root,
        catalog,
        signature_hexes,
        assignment,
        str(audit_db_path),
    )
    state_action_audit["status"] = "pass_with_materialized_row_exclusions"
    state_action_audit["cross_split_signature_count_before_exclusion"] = state_action_audit[
        "cross_split_signature_count"
    ]
    state_action_audit["cross_split_signature_count_after_exclusion"] = exclusion_coverage[
        "effective_cross_split_signature_count"
    ]
    state_action_audit["excluded_signature_sha256"] = hashlib.sha256(
        "\n".join(sorted(signature_hexes)).encode()
    ).hexdigest()
    state_action_audit["row_exclusion_policy"] = (
        "materialized target/metric mask excludes matching decisions in every split; "
        "sequence context remains contiguous"
    )
    state_action_audit["exclusion_coverage"] = exclusion_coverage
    summary["state_action_overlap"] = state_action_audit
    summary["action_family_coverage"] = state_action_audit["action_family_coverage"]
    split_hashes = {
        split: hashlib.sha256("\n".join(sorted(key for key, value in assignment.items() if value == split)).encode()).hexdigest()
        for split in REPORT_SPLIT_NAMES
    }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "status": "accepted",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_root": canonical_root,
        "catalog_path": catalog_path,
        "catalog_sha256": catalog_sha,
        "catalog_exclusions_sha256": catalog["unit_exclusions"]["sha256"],
        "splitter_sha256": _sha256(__file__),
        "signature_implementation_sha256": _sha256(value_contracts.__file__),
        "audit_run_id": audit_run_id,
        "audit_root": str(audit_root),
        "split_names": list(SPLIT_NAMES),
        "row_exclusion_signature_field": "summary.state_action_overlap.cross_split_signatures",
        "row_exclusion_manifest_path": exclusion_coverage["row_exclusion_manifest_path"],
        "row_exclusion_manifest_sha256": exclusion_coverage["row_exclusion_manifest_sha256"],
        "ratios": dict(zip(SPLIT_NAMES, RATIOS)),
        "chronology_claim": "chronology-first independently within each source; no cross-source global-order claim",
        "games": len(games),
        "rows": sum(entry["rows"] for entry in games.values()),
        "unit_bindings": unit_bindings,
        "split_hashes": split_hashes,
        "assignment_sha256": assignment_hash,
        "assignment": assignment,
        "summary": summary,
        "prospective_gold": gold,
        "acceptance": {
            "catalog_bound": True,
            "game_grouped": True,
            "chronology_boundary_violations": 0,
            "game_overlap_across_splits": 0,
            "game_content_overlap_across_splits": 0,
            "row_exclusions_materialized": True,
            "state_action_overlap_after_exclusion": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prospective-gold-root")
    parser.add_argument("--state-action-db", required=True)
    args = parser.parse_args()
    manifest = build_split_manifest(
        args.canonical,
        args.catalog,
        args.prospective_gold_root,
        args.state_action_db,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    os.replace(tmp, out_path)
    summary = {
        split: {
            "games": sum(manifest["summary"]["counts_games"][split].values()),
            "rows": sum(manifest["summary"]["counts_rows"][split].values()),
        }
        for split in REPORT_SPLIT_NAMES
    }
    print(json.dumps({"out": str(out_path), "summary": summary, "gold": manifest["prospective_gold"]}, indent=1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
