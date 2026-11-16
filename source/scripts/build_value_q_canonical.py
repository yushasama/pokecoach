"""Convert mass replay corpora into canonical actor-visible value/Q datasets.

Physical layout per conversion unit (one source shard file or one batch of
kaggle game files):

    <out>/<source_id>/<unit_id>/part_0000.npz          feature matrix + row metadata
    <out>/<source_id>/<unit_id>/part_0000.games.jsonl.gz  one record per game-seat block
    <out>/<source_id>/<unit_id>/unit_manifest.json     source hashes, counts, quarantine

Schemas: deepmew.temporal_actor_sequence.v1 / deepmew.actor_transition.v1
(see deepmew/replay/canonical_actor_dataset.py). Fail-closed: every row passes
the information-mode gate; every dropped row/game is counted by reason.

Usage:
    python scripts/build_value_q_canonical.py convert --out <dir> [--sources a,b]
        [--workers 6] [--max-units N] [--rows-per-part 250000]
    python scripts/build_value_q_canonical.py catalog --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deepmew.replay.canonical_actor_dataset import (  # noqa: E402
    ACTION_OPTION_SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
    FEATURE_BLOCKS,
    FEATURE_DIM,
    QuarantineReason,
    SEQUENCE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    UNIT_MANIFEST_SCHEMA_VERSION,
    canonical_actor_feature_observation,
    check_seat_sequences,
    deployable_row_visibility,
    kaggle_actor_label,
    quarantine_reason_for_visibility,
    refgame_actor_label,
    refgame_safety_flagged,
    validate_action_contract,
    validate_action_option_record,
    validate_sequence_record,
    validate_transition_record,
)


@dataclass(frozen=True)
class SourceDef:
    source_id: str
    kind: str  # refgame_jsonl_gz | kaggle_action_jsonl
    era_rank: int
    chronology: str
    patterns: tuple[str, ...]


ARTIFACTS = Path(r"D:\DeepMew_artifacts")

SOURCES: dict[str, SourceDef] = {
    "unified_r2": SourceDef(
        "unified_r2", "refgame_jsonl_gz", 0, "generation_order(game_index)",
        (str(ARTIFACTS / "datasets" / "unified_r2" / "refgames_dec061_regen3_clean" / "shards" / "decisions_shard_*.jsonl.gz"),),
    ),
    "kaggle_ingest": SourceDef(
        "kaggle_ingest", "kaggle_action_jsonl", 1, "episode_id(ladder_time)",
        (str(REPO_ROOT / ".research" / "kaggle_ingest" / "*" / "*" / "replay_action_table.jsonl"),),
    ),
    "r15_hard": SourceDef(
        "r15_hard", "refgame_jsonl_gz", 2, "generation_order(game_index)",
        (str(ARTIFACTS / "datasets" / "r15" / "hard_refgames_crossplay" / "shards" / "decisions_shard_*.jsonl.gz"),),
    ),
    "r18_awr": SourceDef(
        "r18_awr", "refgame_jsonl_gz", 3, "run_round_then_game_index",
        (
            str(REPO_ROOT / ".research" / "r18" / "awr_iterate" / "r*_data" / "shards" / "*.jsonl.gz"),
            str(REPO_ROOT / ".research" / "r18" / "awr_iterate2" / "r*_data" / "shards" / "*.jsonl.gz"),
            str(REPO_ROOT / ".research" / "r18" / "bd27_awr_long" / "r*_data" / "shards" / "*.jsonl.gz"),
        ),
    ),
    "r18_artifacts": SourceDef(
        "r18_artifacts", "refgame_jsonl_gz", 3, "generation_order(game_index)",
        (str(ARTIFACTS / "datasets" / "r18" / "*" / "shards" / "*.jsonl.gz"),),
    ),
    "r21_staryu": SourceDef(
        "r21_staryu", "refgame_jsonl_gz", 4, "run_round_then_game_index",
        (str(REPO_ROOT / ".research" / "r21_staryu_topdeck_awr" / "local_fast_from_awr_iter2" / "r*_data" / "shards" / "*.jsonl.gz"),),
    ),
    "league_smoke": SourceDef(
        "league_smoke", "refgame_jsonl_gz", 4, "generation_order(game_index)",
        (str(REPO_ROOT / ".research" / "runs" / "league_datagen_smoke" / "20260708" / "*" / "shards" / "*.jsonl.gz"),),
    ),
    "stage3_public": SourceDef(
        "stage3_public", "refgame_jsonl_gz", 5, "generation_order(game_index)",
        (
            str(REPO_ROOT / ".research" / "chosen_action_advantage_stage3" / "public_a4_20260709" / "shards" / "*.jsonl.gz"),
            str(REPO_ROOT / ".research" / "chosen_action_advantage_stage3" / "public_v24_20260709" / "shards" / "*.jsonl.gz"),
        ),
    ),
}

KAGGLE_FILES_PER_UNIT = 150
MAX_GROUPED_REFGAME_BYTES = 8 * 1024 * 1024
EPISODE_RE = re.compile(r"episode-(\d+)")


def register_extra_refgame_source(
    *,
    source_id: str,
    patterns: list[str],
    era_rank: int,
    chronology: str,
    workers: int,
) -> SourceDef:
    """Register one explicit research source without editing the frozen source table."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", source_id):
        raise ValueError("extra source id must be a lowercase filesystem-safe identifier")
    if source_id in SOURCES:
        raise ValueError(f"extra source id already exists: {source_id}")
    normalized_patterns = tuple(str(Path(pattern)) for pattern in patterns if str(pattern).strip())
    if not normalized_patterns:
        raise ValueError("extra refgame source requires at least one pattern")
    if workers != 1:
        raise ValueError("dynamic extra sources currently require --workers 1")
    source = SourceDef(
        source_id=source_id,
        kind="refgame_jsonl_gz",
        era_rank=int(era_rank),
        chronology=str(chronology),
        patterns=normalized_patterns,
    )
    SOURCES[source_id] = source

    return source


# ---------------------------------------------------------------------------
# Feature extraction (lazy per-process init; imports torch/cg once per worker)
# ---------------------------------------------------------------------------

_EXTRACT: dict = {}


def _extractors() -> dict:
    if _EXTRACT:
        return _EXTRACT
    import run_unified_r3_value as value_source  # noqa: WPS433
    import run_unified_r4_ablation as r4  # noqa: WPS433
    from cg.api import all_card_data  # noqa: WPS433

    cards = [card for card in all_card_data() if card is not None]
    max_id = max(card.cardId for card in cards)
    ctype = np.zeros(max_id + 2, dtype=np.int8)
    basic = np.zeros(max_id + 2, dtype=np.int8)
    evolution = np.zeros(max_id + 2, dtype=np.int8)
    ex_flag = np.zeros(max_id + 2, dtype=np.int8)
    for card in cards:
        cid = card.cardId
        ctype[cid] = min(int(card.cardType), 6)
        basic[cid] = 1 if (card.basic and int(card.cardType) == 0) else 0
        evolution[cid] = 1 if (card.stage1 or card.stage2 or card.megaEx) else 0
        ex_flag[cid] = 1 if (card.ex or card.megaEx) else 0
    _EXTRACT.update(
        features=value_source._features,
        feature_config=value_source.FEATURE_CONFIG_V2,
        encode_base=r4.encode_base,
        max_id=max_id,
        ctype=ctype,
        basic=basic,
        evolution=evolution,
        ex_flag=ex_flag,
    )

    return _EXTRACT


def _hand_ids(row: dict, obs: dict) -> list[int]:
    hand = row.get("own_hand")
    if not isinstance(hand, list):
        current = obs.get("current") or {}
        try:
            your_index = int(current.get("yourIndex"))
        except (TypeError, ValueError):
            return []
        players = current.get("players") or []
        hand = []
        if 0 <= your_index < len(players) and isinstance(players[your_index], dict):
            hand = players[your_index].get("hand") or []
        if not isinstance(hand, list):
            hand = []
    ids: list[int] = []
    for entry in hand:
        if isinstance(entry, int):
            ids.append(entry)
        elif isinstance(entry, dict):
            card_id = entry.get("cardId", entry.get("id"))
            if isinstance(card_id, int):
                ids.append(card_id)

    return ids


def _hand11(ids: list[int]) -> list[float]:
    ex = _extractors()
    counts = [0] * 7
    n_basic = n_evolution = n_ex = 0
    for card_id in ids:
        if 1 <= card_id <= ex["max_id"]:
            counts[int(ex["ctype"][card_id])] += 1
            n_basic += int(ex["basic"][card_id])
            n_evolution += int(ex["evolution"][card_id])
            n_ex += int(ex["ex_flag"][card_id])

    return (
        [min(count, 10) / 10.0 for count in counts]
        + [min(n_basic, 10) / 10.0, min(n_evolution, 10) / 10.0, min(n_ex, 10) / 10.0, min(len(ids), 12) / 12.0]
    )


def _row_features(row: dict, obs: dict) -> np.ndarray:
    ex = _extractors()
    projected = canonical_actor_feature_observation(obs)
    vector = ex["features"]({"raw_observation": projected}, ex["feature_config"])
    vector = list(vector) + list(ex["encode_base"](projected)) + _hand11(_hand_ids(row, projected))
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (FEATURE_DIM,):
        raise ValueError(f"feature dim {array.shape} != {FEATURE_DIM}")

    return array


# ---------------------------------------------------------------------------
# Part writer (atomic, chunked at game boundaries)
# ---------------------------------------------------------------------------

META_INT_FIELDS = (
    "seat", "turn", "select_context", "legal_count", "decision_index", "actor_decision_index",
    "next_same_actor_offset", "transition_horizon", "terminal", "has_state_after", "block_id",
    "policy_idx", "opp_policy_idx", "deck_idx", "opp_deck_idx",
)


class PartWriter:
    def __init__(self, unit_dir: Path, rows_per_part: int):
        self.unit_dir = unit_dir
        self.rows_per_part = rows_per_part
        self.part_index = 0
        self.parts: list[dict] = []
        self._reset()

    def _reset(self) -> None:
        self.X: list[np.ndarray] = []
        self.X_after: list[np.ndarray] = []
        self.z: list[np.ndarray] = []
        self.meta: dict[str, list[int]] = {f: [] for f in META_INT_FIELDS}
        self.games: list[dict] = []
        self.sequences: list[dict] = []
        self.transitions: list[dict] = []
        self.action_options: list[dict] = []
        self.rows = 0
        self.block_id = 0

    def add_block(
        self,
        features: np.ndarray,
        after_features: np.ndarray,
        z: float,
        meta_rows: dict[str, list[int]],
        game_record: dict,
        sequence_records: list[dict],
        transition_records: list[dict],
        action_option_records: list[dict],
    ) -> None:
        start = self.rows
        n = features.shape[0]
        if after_features.shape != features.shape:
            raise ValueError("before/after feature matrices must have identical shape")
        if len(sequence_records) != n or len(transition_records) != n:
            raise ValueError("each canonical feature row requires one sequence and transition record")
        self.X.append(features)
        self.X_after.append(after_features)
        self.z.append(np.full(n, z, dtype=np.float32))
        for f in META_INT_FIELDS:
            if f == "block_id":
                self.meta[f].extend([self.block_id] * n)
            else:
                self.meta[f].extend(meta_rows[f])
        game_record = dict(game_record)
        game_record.update({"part": self.part_index, "block_id": self.block_id, "block_start": start, "block_len": n})
        self.games.append(game_record)
        for row_offset, record in enumerate(sequence_records):
            self.sequences.append({**record, "part": self.part_index, "row": start + row_offset})
        for row_offset, record in enumerate(transition_records):
            self.transitions.append({**record, "part": self.part_index, "row": start + row_offset})
        for record in action_option_records:
            self.action_options.append({
                **record,
                "part": self.part_index,
                "row": start + int(record["actor_decision_index"]),
            })
        self.rows += n
        self.block_id += 1
        if self.rows >= self.rows_per_part:
            self.flush()

    def flush(self) -> None:
        if self.rows == 0:
            return
        name = f"part_{self.part_index:04d}"
        npz_path = self.unit_dir / f"{name}.npz"
        games_path = self.unit_dir / f"{name}.games.jsonl.gz"
        sequences_path = self.unit_dir / f"{name}.sequences.jsonl.gz"
        transitions_path = self.unit_dir / f"{name}.transitions.jsonl.gz"
        actions_path = self.unit_dir / f"{name}.actions.jsonl.gz"
        arrays = {
            "X": np.concatenate(self.X, axis=0),
            "X_after": np.concatenate(self.X_after, axis=0),
            "z": np.concatenate(self.z, axis=0),
        }
        for f in META_INT_FIELDS:
            arrays[f] = np.asarray(self.meta[f], dtype=np.int32)
        tmp = str(npz_path) + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz_path)
        _write_jsonl_gz_atomic(games_path, self.games)
        _write_jsonl_gz_atomic(sequences_path, self.sequences)
        _write_jsonl_gz_atomic(transitions_path, self.transitions)
        _write_jsonl_gz_atomic(actions_path, self.action_options)
        files = (npz_path, games_path, sequences_path, transitions_path, actions_path)
        self.parts.append({
            "name": name,
            "rows": int(self.rows),
            "blocks": len(self.games),
            "transitions": len(self.transitions),
            "action_options": len(self.action_options),
            "files_sha256": {path.name: _sha256(str(path)) for path in files},
        })
        self.part_index += 1
        self._reset()


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def _sha256(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_jsonl_gz_atomic(path: Path, records: list[dict]) -> None:
    tmp = str(path) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(tmp, path)


def _as_int(value, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _observation_actor(observation) -> int | None:
    if not isinstance(observation, dict):
        return None
    current = observation.get("current")
    if not isinstance(current, dict):
        return None

    return _as_int(current.get("yourIndex"))


def _source_reference(row: dict, field: str) -> dict | None:
    value = row.get(f"_{field}")
    if value is None:
        return None

    return {
        "source_path": row["_source_path"],
        "source_line": row["_source_line"],
        "source_field": field,
        "content_sha256": row.get(f"_{field}_hash") or _stable_hash(value),
    }


def _manifest_outputs_valid(unit_dir: Path, manifest: dict) -> bool:
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        return False
    if not parts:
        stats = manifest.get("stats") or {}

        return stats.get("rows_kept", 0) == 0 and stats.get("games_kept", 0) == 0
    for part in parts:
        hashes = part.get("files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            return False
        for name, expected in hashes.items():
            path = unit_dir / name
            if not path.is_file() or _sha256(str(path)) != expected:
                return False

    return True


class Vocab:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.index: dict[str, int] = {}

    def get(self, value) -> int:
        key = str(value)
        if key not in self.index:
            self.index[key] = len(self.items)
            self.items.append(key)

        return self.index[key]


GAME_META_KEYS = (
    "bucket", "matchup_cell_key", "opponent_shell", "opponent_bucket", "opponent_pilot",
    "training_phase", "deck_id", "opponent_deck_id",
)


def _finalize_game(
    game_id: str,
    rows_by_seat: dict[int, list[dict]],
    writer: PartWriter,
    vocabs: dict[str, Vocab],
    stats: Counter,
    quarantine_examples: list,
    source_id: str,
    unit_id: str,
) -> None:
    if not rows_by_seat:
        return
    all_rows = [row for rows in rows_by_seat.values() for row in rows]
    stats["games_seen"] += 1
    reasons: set[str] = set()
    for row in all_rows:
        if row.get("_visibility_reason"):
            reasons.add(row["_visibility_reason"])
        if row.get("z") is None:
            reasons.add(QuarantineReason.NO_TERMINAL)
        if row.get("_safety"):
            reasons.add(QuarantineReason.SAFETY)
        if not validate_action_contract(row.get("chosen_action"), row.get("legal_action_set")):
            reasons.add(QuarantineReason.ACTION_CONTRACT)
    if not reasons:
        check = check_seat_sequences(rows_by_seat)
        if not check.ok:
            reasons.update(check.reasons)
    if reasons:
        stats["games_quarantined"] += 1
        stats["rows_quarantined"] += len(all_rows)
        for reason in reasons:
            stats[f"gq:{reason}"] += 1
        if len(quarantine_examples) < 8:
            quarantine_examples.append({"game_id": game_id, "reasons": sorted(reasons)})

        return

    for rows in rows_by_seat.values():
        rows.sort(key=lambda row: row["decision_index"])
    game_content_hash = _stable_hash([
        {
            "seat": seat,
            "decision_index": row["decision_index"],
            "observation_sha256": row["_raw_observation_hash"],
            "chosen_action": row["chosen_action"],
            "terminal_actor_relative_result": row["z"],
        }
        for seat, seat_rows in sorted(rows_by_seat.items())
        for row in sorted(seat_rows, key=lambda item: item["decision_index"])
    ])
    for seat, rows in sorted(rows_by_seat.items()):
        features = np.stack([row["_features"] for row in rows])
        after_features = np.full((len(rows), FEATURE_DIM), np.nan, dtype=np.float32)
        for actor_index, row in enumerate(rows):
            if row.get("_features_after") is not None:
                after_features[actor_index] = row["_features_after"]
        actor_indices = list(range(len(rows)))
        next_offsets = [index + 1 if index + 1 < len(rows) else -1 for index in actor_indices]
        horizons = [
            rows[index + 1]["decision_index"] - row["decision_index"] if index + 1 < len(rows) else -1
            for index, row in enumerate(rows)
        ]
        terminals = [int(index == len(rows) - 1) for index in actor_indices]
        meta_rows = {
            "seat": [seat] * len(rows),
            "turn": [row["turn"] for row in rows],
            "select_context": [row["select_context"] for row in rows],
            "legal_count": [row["legal_count"] for row in rows],
            "decision_index": [row["decision_index"] for row in rows],
            "actor_decision_index": actor_indices,
            "next_same_actor_offset": next_offsets,
            "transition_horizon": horizons,
            "terminal": terminals,
            "has_state_after": [int(row.get("_features_after") is not None) for row in rows],
            "block_id": [],
            "policy_idx": [vocabs["policy"].get(row["policy_id"]) for row in rows],
            "opp_policy_idx": [vocabs["policy"].get(row["opponent_policy_id"]) for row in rows],
            "deck_idx": [vocabs["deck"].get(row["deck_hash"]) for row in rows],
            "opp_deck_idx": [vocabs["deck"].get(row["opponent_deck_hash"]) for row in rows],
        }
        first = rows[0]
        canonical_game_id = f"{source_id}:{unit_id}:{game_id}"
        record = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "canonical_game_id": canonical_game_id,
            "source_game_id": game_id,
            "source_id": source_id,
            "source_schema": first["source_schema"],
            "unit_id": unit_id,
            "seat": seat,
            "z": first["z"],
            "order_key": first["order_key"],
            "policy_id": first["policy_id"],
            "opponent_policy_id": first["opponent_policy_id"],
            "deck_hash": first["deck_hash"],
            "opponent_deck_hash": first["opponent_deck_hash"],
            "information_mode": "actor_visible",
            "game_completion_status": "completed",
            "safety_status": "clean",
            "source_hash": first["_source_hash"],
            "game_content_hash": game_content_hash,
            "source_path": first["_source_path"],
            "provenance": {
                "observation_representation": "canonical_feature_vector.v1",
                "raw_observation_copied": False,
                "raw_observation_reference_is_content_addressed": True,
            },
        }
        for key in GAME_META_KEYS:
            if first.get(key) is not None:
                record[key] = first[key]
        if first.get("source_rating") is not None:
            record["source_rating"] = first["source_rating"]
        sequence_records: list[dict] = []
        transition_records: list[dict] = []
        action_option_records: list[dict] = []
        previous_action = None
        for actor_index, row in enumerate(rows):
            decision_id = f"{canonical_game_id}:d{row['decision_index']:06d}:s{seat}"
            state_before = {
                "representation": "canonical_feature_vector.v1",
                "feature_row": actor_index,
                "feature_dim": FEATURE_DIM,
                "source_observation": _source_reference(row, "raw_observation"),
            }
            next_state = None
            horizon = None
            if actor_index + 1 < len(rows):
                next_row = rows[actor_index + 1]
                next_state = {
                    "representation": "canonical_feature_vector.v1",
                    "feature_row": actor_index + 1,
                    "decision_id": f"{canonical_game_id}:d{next_row['decision_index']:06d}:s{seat}",
                }
                horizon = next_row["decision_index"] - row["decision_index"]
            state_after = None
            if row.get("_features_after") is not None:
                state_after = {
                    "representation": "canonical_after_feature_vector.v1",
                    "feature_row": actor_index,
                    "source_observation": _source_reference(row, "post_observation"),
                }
            sequence_record = {
                "schema_version": SEQUENCE_SCHEMA_VERSION,
                "canonical_game_id": canonical_game_id,
                "decision_id": decision_id,
                "information_mode": "actor_visible",
                "source_id": source_id,
                "source_schema": row["source_schema"],
                "source_game_id": game_id,
                "source_order": row.get("order_key"),
                "actor": seat,
                "actor_decision_index": actor_index,
                "global_decision_index": row["decision_index"],
                "turn": row["turn"],
                "phase": row.get("phase"),
                "control_owner": seat,
                "control_passed": row.get("control_passed"),
                "state_before": state_before,
                "previous_actor_action": previous_action,
                "public_event_history": {
                    "status": "not_materialized_opaque_source_logs",
                    "reason": "CABT logs/looking are not certified public-safe",
                },
                "terminal_actor_relative_result": row["z"],
                "game_completion_status": "completed",
                "safety_status": "clean",
                "policy_id": row["policy_id"],
                "opponent_policy_id": row["opponent_policy_id"],
                "deck_hash": row["deck_hash"],
                "opponent_deck_hash": row["opponent_deck_hash"],
                "source_hash": row["_source_hash"],
            }
            validate_sequence_record(sequence_record)
            sequence_records.append(sequence_record)
            transition = {
                "schema_version": TRANSITION_SCHEMA_VERSION,
                "canonical_game_id": canonical_game_id,
                "decision_id": decision_id,
                "actor": seat,
                "global_decision_index": row["decision_index"],
                "actor_decision_index": actor_index,
                "state_before": state_before,
                "chosen_action": row["chosen_action"],
                "legal_action_set": row["legal_action_set"],
                "state_after": state_after,
                "state_after_status": row["state_after_status"],
                "next_same_actor_state": next_state,
                "transition_horizon_logical_decisions": horizon,
                "terminal": actor_index + 1 == len(rows),
                "terminal_actor_relative_result": row["z"],
                "immediate_reward": None,
                "immediate_reward_status": "not_available_no_reward_contract",
                "information_mode": "actor_visible",
                "policy_id": row["policy_id"],
                "source_id": source_id,
                "source_schema": row["source_schema"],
                "source_hash": row["_source_hash"],
            }
            validate_transition_record(transition)
            transition_records.append(transition)
            chosen_indices = set(row["chosen_action"]["indices"])
            uses_option_positions = row["chosen_action"].get("index_space") == "option_position"
            for option_position, option in enumerate(row["legal_action_set"]):
                source_index = option.get("index", option_position)
                action_index = option_position if uses_option_positions else source_index
                action_option_record = {
                    "schema_version": ACTION_OPTION_SCHEMA_VERSION,
                    "canonical_game_id": canonical_game_id,
                    "decision_id": decision_id,
                    "actor": seat,
                    "global_decision_index": row["decision_index"],
                    "actor_decision_index": actor_index,
                    "option_position": option_position,
                    "action_index": action_index,
                    "source_option_index": source_index,
                    "option": option,
                    "is_chosen": action_index in chosen_indices,
                    "information_mode": "actor_visible",
                    "source_hash": row["_source_hash"],
                }
                validate_action_option_record(action_option_record)
                action_option_records.append(action_option_record)
            previous_action = row["chosen_action"]
        writer.add_block(
            features,
            after_features,
            first["z"],
            meta_rows,
            record,
            sequence_records,
            transition_records,
            action_option_records,
        )
        stats["blocks_kept"] += 1
        stats["rows_kept"] += len(rows)
    stats["games_kept"] += 1


def _prepare_refgame_row(
    raw: dict,
    stats: Counter,
    *,
    source_path: str,
    source_line: int,
    source_hash: str,
) -> dict:
    accepted, classification = deployable_row_visibility(raw)
    obs = raw.get("raw_observation") or {}
    select = obs.get("select") or {}
    options = select.get("option") or []
    legal_set = raw.get("legal_set") or []
    chosen_action = dict(raw.get("chosen_action")) if isinstance(raw.get("chosen_action"), dict) else raw.get("chosen_action")
    if isinstance(chosen_action, dict) and chosen_action.get("indices") == []:
        chosen_action["allows_empty_choice"] = _as_int(select.get("minCount"), -1) == 0
    decision_index = _as_int(raw.get("decision_index"), -1)
    seat = _as_int(raw.get("seat"), -1)
    post_observation = raw.get("post_observation") if isinstance(raw.get("post_observation"), dict) else None
    row = {
        "decision_index": decision_index,
        "turn": _as_int(raw.get("turn"), 0),
        "select_context": _as_int(select.get("context"), -1),
        "legal_count": len(legal_set),
        "chosen_action": chosen_action,
        "legal_action_set": legal_set,
        "policy_id": raw.get("policy_id"),
        "opponent_policy_id": raw.get("opponent_policy_id"),
        "deck_hash": raw.get("deck_hash"),
        "opponent_deck_hash": raw.get("opponent_deck_hash"),
        "order_key": raw.get("game_index"),
        "z": refgame_actor_label(raw.get("outcome")),
        "source_schema": raw.get("schema_version"),
        "phase": raw.get("training_phase") or select.get("context"),
        "control_passed": (
            _observation_actor(post_observation) != seat if post_observation is not None and seat in (0, 1) else None
        ),
        "state_after_status": "unavailable_missing_post_observation",
        "_raw_observation": obs,
        "_raw_observation_hash": _stable_hash(obs),
        "_post_observation": post_observation,
        "_post_observation_hash": _stable_hash(post_observation) if post_observation is not None else None,
        "_source_path": source_path,
        "_source_line": source_line,
        "_source_hash": source_hash,
        "_safety": refgame_safety_flagged(raw.get("outcome")),
        "_visibility_reason": None if accepted else quarantine_reason_for_visibility(classification),
    }
    for key in GAME_META_KEYS:
        if raw.get(key) is not None:
            row[key] = raw.get(key)
    if decision_index is None or decision_index < 0 or seat not in (0, 1):
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.MISSING_FIELDS
    if _observation_actor(obs) != seat:
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.INVALID_ACTOR
    if not isinstance(row["source_schema"], str) or not row["source_schema"]:
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.MISSING_FIELDS
    if not all(isinstance(value, str) and value for value in (row["policy_id"], row["opponent_policy_id"])):
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.MISSING_FIELDS
    if not validate_action_contract(row["chosen_action"], row["legal_action_set"]):
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.ACTION_CONTRACT
    if row["_visibility_reason"] is None:
        try:
            row["_features"] = _row_features(raw, obs)
            if post_observation is not None and _observation_actor(post_observation) == seat:
                row["_features_after"] = _row_features({}, post_observation)
                row["state_after_status"] = "available_same_actor"
            elif post_observation is not None:
                row["_features_after"] = None
                row["state_after_status"] = "unavailable_post_observation_other_actor"
            else:
                row["_features_after"] = None
        except Exception:
            stats["rows_feature_error"] += 1
            row["_visibility_reason"] = QuarantineReason.PARSE_ERROR

    return row


def _prepare_kaggle_row(
    raw: dict,
    stats: Counter,
    *,
    source_path: str,
    source_line: int,
    source_hash: str,
) -> tuple[dict, int]:
    accepted, classification = deployable_row_visibility(raw)
    obs = raw.get("state_json") or {}
    select = obs.get("select") or {}
    options = select.get("option") or []
    seat = _as_int(raw.get("player_id"), -1)
    episode = -1
    match = EPISODE_RE.search(str(raw.get("source_path") or ""))
    if match:
        episode = int(match.group(1))
    legal_actions = raw.get("legal_options_json") if isinstance(raw.get("legal_options_json"), list) else []
    chosen_action = {"indices": raw.get("chosen_indices"), "index_space": "option_position"}
    decision_index = _as_int(raw.get("step"), -1)
    row = {
        "decision_index": decision_index,
        "turn": _as_int(raw.get("turn"), 0),
        "select_context": _as_int(select.get("context"), -1),
        "legal_count": len(legal_actions),
        "chosen_action": chosen_action,
        "legal_action_set": legal_actions,
        "policy_id": f"kaggle:{raw.get('source_kind')}",
        "opponent_policy_id": "kaggle:opponent",
        "deck_hash": raw.get("deck_hash"),
        "opponent_deck_hash": None,
        "order_key": episode,
        "source_rating": raw.get("source_rating"),
        "z": kaggle_actor_label(raw.get("result"), seat),
        "source_schema": raw.get("schema_version"),
        "phase": select.get("context"),
        "control_passed": None,
        "state_after_status": "unavailable_source_not_recorded",
        "_raw_observation": obs,
        "_raw_observation_hash": _stable_hash(obs),
        "_post_observation": None,
        "_post_observation_hash": None,
        "_source_path": source_path,
        "_source_line": source_line,
        "_source_hash": source_hash,
        "_features_after": None,
        "_safety": False,
        "_visibility_reason": None if accepted else quarantine_reason_for_visibility(classification),
    }
    if decision_index is None or decision_index < 0 or seat not in (0, 1):
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.MISSING_FIELDS
    if _observation_actor(obs) != seat:
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.INVALID_ACTOR
    if not isinstance(row["source_schema"], str) or not row["source_schema"]:
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.MISSING_FIELDS
    if not validate_action_contract(chosen_action, legal_actions):
        row["_visibility_reason"] = row["_visibility_reason"] or QuarantineReason.ACTION_CONTRACT
    if row["_visibility_reason"] is None:
        try:
            row["_features"] = _row_features(raw, obs)
        except Exception:
            stats["rows_feature_error"] += 1
            row["_visibility_reason"] = QuarantineReason.PARSE_ERROR

    return row, seat


def convert_unit(unit: dict, out_root: str, rows_per_part: int) -> dict:
    source = SOURCES[unit["source_id"]]
    unit_dir = Path(out_root) / source.source_id / unit["unit_id"]
    manifest_path = unit_dir / "unit_manifest.json"
    source_hashes = {path: _sha256(path) for path in unit["files"]}
    fingerprint = [
        {
            "path": path,
            "size": os.path.getsize(path),
            "mtime": int(os.path.getmtime(path)),
            "sha256": source_hashes[path],
        }
        for path in unit["files"]
    ]
    converter_hash = _stable_hash({
        "builder": _sha256(__file__),
        "schema": _sha256(str(REPO_ROOT / "deepmew" / "replay" / "canonical_actor_dataset.py")),
    })
    conversion_key = {
        "source_fingerprint": fingerprint,
        "converter_hash": converter_hash,
        "rows_per_part": rows_per_part,
        "schemas": [SEQUENCE_SCHEMA_VERSION, TRANSITION_SCHEMA_VERSION, ACTION_OPTION_SCHEMA_VERSION],
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("conversion_key") == conversion_key
                and existing.get("status") == "completed"
                and _manifest_outputs_valid(unit_dir, existing)
            ):
                existing["skipped"] = True
                return existing
        except Exception:
            pass
    unit_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats: Counter = Counter()
    quarantine_examples: list = []
    vocabs = {"policy": Vocab(), "deck": Vocab()}
    writer = PartWriter(unit_dir, rows_per_part)
    finalized_games: set[str] = set()

    def finalize(game_id: str, rows_by_seat: dict) -> None:
        if game_id in finalized_games:
            stats["games_refragmented"] += 1
            stats["unit_fatal_duplicate_games"] += 1
        finalized_games.add(game_id)
        _finalize_game(game_id, rows_by_seat, writer, vocabs, stats, quarantine_examples,
                       source.source_id, unit["unit_id"])

    if source.kind == "refgame_jsonl_gz":
        for path in unit["files"]:
            group_whole_file = os.path.getsize(path) <= MAX_GROUPED_REFGAME_BYTES
            grouped_rows: dict[tuple[str, int | None], dict[int, list[dict]]] | None = (
                defaultdict(lambda: defaultdict(list)) if group_whole_file else None
            )
            current_game: str | None = None
            rows_by_seat: dict[int, list[dict]] = defaultdict(list)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for source_line, line in enumerate(handle, start=1):
                    stats["rows_read"] += 1
                    try:
                        raw = json.loads(line)
                    except Exception:
                        stats["rows_quarantined"] += 1
                        stats[f"rq:{QuarantineReason.PARSE_ERROR}"] += 1
                        stats["unit_fatal_parse_errors"] += 1
                        continue
                    raw_game_id = raw.get("game_id")
                    game_id = str(raw_game_id) if raw_game_id is not None else f"__invalid__:{source_line}"
                    if grouped_rows is None and current_game is not None and game_id != current_game:
                        finalize(current_game, rows_by_seat)
                        rows_by_seat = defaultdict(list)
                    current_game = game_id
                    row = _prepare_refgame_row(
                        raw,
                        stats,
                        source_path=path,
                        source_line=source_line,
                        source_hash=source_hashes[path],
                    )
                    if raw_game_id is None:
                        row["_visibility_reason"] = QuarantineReason.MISSING_FIELDS
                    seat = _as_int(raw.get("seat"), -1)
                    if grouped_rows is not None:
                        grouped_rows[(game_id, _as_int(raw.get("game_index")))][seat].append(row)
                    else:
                        rows_by_seat[seat].append(row)
            if grouped_rows is not None:
                stats["source_files_grouped_for_noncontiguous_games"] += 1
                instances_per_id = Counter(game_id for game_id, _ in grouped_rows)
                for (game_id, game_index), grouped_seats in grouped_rows.items():
                    canonical_source_game_id = game_id
                    if instances_per_id[game_id] > 1:
                        canonical_source_game_id = f"{game_id}:game_index={game_index}"
                        stats["source_game_id_collisions_disambiguated"] += 1
                    finalize(canonical_source_game_id, grouped_seats)
            elif current_game is not None:
                finalize(current_game, rows_by_seat)
    elif source.kind == "kaggle_action_jsonl":
        for path in unit["files"]:
            rows_by_game: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
            with open(path, "rt", encoding="utf-8") as handle:
                for source_line, line in enumerate(handle, start=1):
                    stats["rows_read"] += 1
                    try:
                        raw = json.loads(line)
                    except Exception:
                        stats["rows_quarantined"] += 1
                        stats[f"rq:{QuarantineReason.PARSE_ERROR}"] += 1
                        stats["unit_fatal_parse_errors"] += 1
                        continue
                    raw_game_id = raw.get("game_id")
                    game_id = str(raw_game_id) if raw_game_id is not None else f"__invalid__:{source_line}"
                    row, seat = _prepare_kaggle_row(
                        raw,
                        stats,
                        source_path=path,
                        source_line=source_line,
                        source_hash=source_hashes[path],
                    )
                    if raw_game_id is None:
                        row["_visibility_reason"] = QuarantineReason.MISSING_FIELDS
                    rows_by_game[game_id][seat].append(row)
            for game_id, rows_by_seat in rows_by_game.items():
                finalize(game_id, rows_by_seat)
    else:
        raise ValueError(f"unknown source kind {source.kind}")

    writer.flush()
    manifest = {
        "schema_version": UNIT_MANIFEST_SCHEMA_VERSION,
        "sequence_schema": SEQUENCE_SCHEMA_VERSION,
        "transition_schema": TRANSITION_SCHEMA_VERSION,
        "action_option_schema": ACTION_OPTION_SCHEMA_VERSION,
        "feature_blocks": list(FEATURE_BLOCKS),
        "feature_dim": FEATURE_DIM,
        "source_id": source.source_id,
        "unit_id": unit["unit_id"],
        "era_rank": source.era_rank,
        "chronology": source.chronology,
        "information_mode": "actor_visible",
        "source_fingerprint": fingerprint,
        "source_sha256": {path: source_hashes[path] for path in unit["files"]},
        "source_set_sha256": _stable_hash(source_hashes),
        "converter_hash": converter_hash,
        "conversion_key": conversion_key,
        "stats": dict(stats),
        "quarantine_examples": quarantine_examples,
        "vocab_policy": vocabs["policy"].items,
        "vocab_deck": vocabs["deck"].items,
        "parts": writer.parts,
        "elapsed_seconds": round(time.time() - started, 1),
        "status": (
            "failed" if stats.get("unit_fatal_parse_errors") or stats.get("unit_fatal_duplicate_games") else "completed"
        ),
    }
    tmp = str(manifest_path) + ".tmp"
    Path(tmp).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Unit discovery / catalog / CLI
# ---------------------------------------------------------------------------


def discover_units(source_ids: list[str]) -> list[dict]:
    units: list[dict] = []
    for source_id in source_ids:
        source = SOURCES[source_id]
        files: list[str] = []
        for pattern in source.patterns:
            files.extend(sorted(glob.glob(pattern)))
        if not files:
            continue
        if source.kind == "kaggle_action_jsonl":
            for start in range(0, len(files), KAGGLE_FILES_PER_UNIT):
                batch = files[start:start + KAGGLE_FILES_PER_UNIT]
                units.append({"source_id": source_id, "unit_id": f"batch_{start // KAGGLE_FILES_PER_UNIT:04d}", "files": batch})
        else:
            for path in files:
                rel = os.path.relpath(path, str(REPO_ROOT)) if path.startswith(str(REPO_ROOT)) else path
                # never allow a leading dot: glob("*") skips dot-prefixed dirs
                unit_id = re.sub(r"[^A-Za-z0-9._-]+", "_", rel).strip("_.")[-120:].strip("_.")
                units.append({"source_id": source_id, "unit_id": unit_id, "files": [path]})

    return units


def build_catalog(out_root: str, exclusions_path: str | None = None) -> dict:
    manifests = sorted(glob.glob(os.path.join(out_root, "*", "*", "unit_manifest.json")))
    exclusion_payload = None
    exclusions: set[tuple[str, str]] = set()
    if exclusions_path:
        exclusion_payload = json.loads(Path(exclusions_path).read_text(encoding="utf-8"))
        exclusions = {
            (entry["source_id"], entry["unit_id"])
            for entry in exclusion_payload.get("entries", [])
        }
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_root": out_root,
        "sources": {},
        "totals": Counter(),
        "unit_exclusions": {
            "path": exclusions_path,
            "sha256": _sha256(exclusions_path) if exclusions_path else None,
            "entries": exclusion_payload.get("entries", []) if exclusion_payload else [],
        },
        "included_units": [],
    }
    game_ids: dict[str, str] = {}
    duplicate_games: list = []
    content_games: dict[str, str] = {}
    duplicate_content: list = []
    excluded_manifests: list[tuple[str, dict]] = []
    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        unit_key = (manifest.get("source_id"), manifest.get("unit_id"))
        if unit_key in exclusions:
            excluded_manifests.append((manifest_path, manifest))
            continue
        if manifest.get("status") != "completed":
            catalog["totals"]["units_excluded_not_completed"] += 1
            continue
        unit_dir = Path(manifest_path).parent
        if not _manifest_outputs_valid(unit_dir, manifest):
            raise ValueError(f"manifest output validation failed: {manifest_path}")
        catalog["included_units"].append({
            "source_id": manifest["source_id"],
            "unit_id": manifest["unit_id"],
            "manifest_path": os.path.relpath(manifest_path, out_root),
            "manifest_sha256": _sha256(manifest_path),
            "source_set_sha256": manifest["source_set_sha256"],
            "converter_hash": manifest["converter_hash"],
        })
        source_id = manifest["source_id"]
        entry = catalog["sources"].setdefault(source_id, {"era_rank": manifest["era_rank"], "chronology": manifest["chronology"], "units": 0, "stats": Counter()})
        entry["units"] += 1
        for key, value in manifest["stats"].items():
            entry["stats"][key] += value
            catalog["totals"][key] += value
        games_paths = [
            str(unit_dir / f"{part['name']}.games.jsonl.gz")
            for part in manifest.get("parts", [])
        ]
        for games_path in games_paths:
            with gzip.open(games_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = record["canonical_game_id"] + f":s{record['seat']}"
                    if key in game_ids:
                        duplicate_games.append({"key": key, "units": [game_ids[key], manifest["unit_id"]]})
                    game_ids[key] = manifest["unit_id"]
                    content_hash = record.get("game_content_hash")
                    if not isinstance(content_hash, str) or not content_hash:
                        raise ValueError(f"game record lacks content hash: {games_path}")
                    prior_game = content_games.get(content_hash)
                    if prior_game is not None and prior_game != record["canonical_game_id"]:
                        duplicate_content.append({
                            "game_content_hash": content_hash,
                            "canonical_game_ids": [prior_game, record["canonical_game_id"]],
                        })
                    content_games[content_hash] = record["canonical_game_id"]
    excluded_games = 0
    excluded_rows = 0
    for manifest_path, manifest in excluded_manifests:
        unit_dir = Path(manifest_path).parent
        if manifest.get("status") != "completed" or not _manifest_outputs_valid(unit_dir, manifest):
            raise ValueError(f"excluded unit is not a valid completed unit: {manifest_path}")
        missing_hashes: list[str] = []
        unit_games = 0
        for part in manifest.get("parts", []):
            games_path = unit_dir / f"{part['name']}.games.jsonl.gz"
            with gzip.open(games_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    unit_games += 1
                    if record["game_content_hash"] not in content_games:
                        missing_hashes.append(record["game_content_hash"])
        if missing_hashes:
            raise ValueError(
                f"excluded unit {manifest['source_id']}/{manifest['unit_id']} has "
                f"{len(missing_hashes)} game blocks not present in retained data"
            )
        excluded_games += unit_games
        excluded_rows += int(manifest.get("stats", {}).get("rows_kept", 0))
    if exclusions - {(manifest["source_id"], manifest["unit_id"]) for _, manifest in excluded_manifests}:
        raise ValueError("unit exclusion manifest references unknown conversion units")
    catalog["duplicate_game_seat_blocks_across_units"] = len(duplicate_games)
    catalog["duplicate_examples"] = duplicate_games[:10]
    catalog["duplicate_game_content"] = len(duplicate_content)
    catalog["duplicate_content_examples"] = duplicate_content[:10]
    catalog["distinct_game_seat_blocks"] = len(game_ids)
    catalog["excluded_duplicate_units"] = len(excluded_manifests)
    catalog["excluded_duplicate_game_seat_blocks"] = excluded_games
    catalog["excluded_duplicate_rows"] = excluded_rows
    for entry in catalog["sources"].values():
        entry["stats"] = dict(entry["stats"])
    catalog["totals"] = dict(catalog["totals"])
    if duplicate_games or duplicate_content:
        raise ValueError(
            f"canonical catalog rejected {len(duplicate_games)} duplicate game-seat blocks and "
            f"{len(duplicate_content)} duplicate game contents; "
            f"examples={(duplicate_games + duplicate_content)[:3]}"
        )

    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--out", required=True)
    convert.add_argument("--sources", default=",".join(SOURCES))
    convert.add_argument("--workers", type=int, default=6)
    convert.add_argument("--max-units", type=int, default=0)
    convert.add_argument("--rows-per-part", type=int, default=250_000)
    convert.add_argument("--extra-refgame-source-id")
    convert.add_argument("--extra-refgame-pattern", action="append", default=[])
    convert.add_argument("--extra-refgame-era-rank", type=int, default=6)
    convert.add_argument(
        "--extra-refgame-chronology",
        default="plan_then_tranche_then_game_index",
    )
    catalog_cmd = sub.add_parser("catalog")
    catalog_cmd.add_argument("--out", required=True)
    catalog_cmd.add_argument("--exclusions")
    args = parser.parse_args()

    if args.command == "catalog":
        catalog = build_catalog(args.out, args.exclusions)
        path = Path(args.out) / "DATASET_CATALOG.json"
        tmp = str(path) + ".tmp"
        Path(tmp).write_text(json.dumps(catalog, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        print(json.dumps({"catalog": str(path), "totals": catalog["totals"],
                          "duplicates": catalog["duplicate_game_seat_blocks_across_units"],
                          "blocks": catalog["distinct_game_seat_blocks"]}, indent=1))
        return 0

    if args.extra_refgame_source_id:
        register_extra_refgame_source(
            source_id=args.extra_refgame_source_id,
            patterns=args.extra_refgame_pattern,
            era_rank=args.extra_refgame_era_rank,
            chronology=args.extra_refgame_chronology,
            workers=args.workers,
        )
    elif args.extra_refgame_pattern:
        raise SystemExit("--extra-refgame-pattern requires --extra-refgame-source-id")
    source_ids = [source.strip() for source in args.sources.split(",") if source.strip()]
    for source_id in source_ids:
        if source_id not in SOURCES:
            raise SystemExit(f"unknown source {source_id}; known: {sorted(SOURCES)}")
    units = discover_units(source_ids)
    if args.max_units:
        units = units[: args.max_units]
    print(f"converting {len(units)} units from {len(source_ids)} sources -> {args.out}", flush=True)
    results = []
    if args.workers <= 1:
        for unit in units:
            manifest = convert_unit(unit, args.out, args.rows_per_part)
            results.append(manifest)
            print(json.dumps({"unit": manifest["unit_id"], "skipped": manifest.get("skipped", False),
                              "rows_kept": manifest["stats"].get("rows_kept", 0),
                              "games_kept": manifest["stats"].get("games_kept", 0),
                              "quarantined": manifest["stats"].get("games_quarantined", 0),
                              "elapsed": manifest.get("elapsed_seconds")}), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_unit, unit, args.out, args.rows_per_part): unit for unit in units}
            for future in as_completed(futures):
                manifest = future.result()
                results.append(manifest)
                print(json.dumps({"unit": manifest["unit_id"], "skipped": manifest.get("skipped", False),
                                  "rows_kept": manifest["stats"].get("rows_kept", 0),
                                  "games_kept": manifest["stats"].get("games_kept", 0),
                                  "quarantined": manifest["stats"].get("games_quarantined", 0),
                                  "elapsed": manifest.get("elapsed_seconds")}), flush=True)
    totals = Counter()
    for manifest in results:
        for key, value in manifest["stats"].items():
            totals[key] += value
    print(json.dumps({"units": len(results), "totals": dict(totals)}, indent=1), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
