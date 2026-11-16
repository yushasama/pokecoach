"""Build the clean-game E1 Derived V1 as references to existing observation assets.

This is intentionally narrower than Admission V0.  It reads only rows belonging
to the frozen clean-game whitelist, reconstructs only the legal/chosen identities
missing from the aggregate semantic receipt, and never scans or hashes Admission.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepmew.replay.canonical_actor_dataset import FEATURE_DIM  # noqa: E402
from deepmew.replay.contract.causal_alignment import (  # noqa: E402
    KaggleEpisodeRepository,
    align_decision,
)
from deepmew.replay.contract.identities import identity, sha256_json  # noqa: E402
from deepmew.replay.contract.source_events import source_event_from_admission  # noqa: E402
from deepmew.replay.contract.validation import (  # noqa: E402
    reconstruct_legal_actions,
    reconstruct_selected_action,
)
from deepmew.rl.enhanced_replay import (  # noqa: E402
    _build_part_bindings,
    _load_catalog_units,
    _validate_cache_inputs,
)
from deepmew.rl.enhanced_replay_derived import (  # noqa: E402
    DerivedReplayHardStop,
    DerivedReplayReferenceBuilder,
    DerivedReplayReferenceIdentity,
    REFERENCE_ROW_SCHEMA_VERSION,
)
from scripts.build_value_q_canonical import _row_features  # noqa: E402


MISSION = ROOT / ".research" / "centralized_causal_replay_contract_and_derived_v1_20260714"
E1_DIR = (
    ROOT
    / ".research"
    / "terminal_league_starmie_grimmsnarl_v0"
    / "e1_starmie_complete_enhanced_generation_20260714"
)
CACHE = Path(
    r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_cache_v2\7459a801-77afa18a"
)
CACHE_MANIFEST = CACHE / "cache_manifest.json"
ORIGINAL_SPLIT = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\SPLIT_MANIFEST.json")
ADMISSION_MANIFEST = E1_DIR / "production_admission" / "admission.manifest.json"
AUDIT_RECEIPT = MISSION / "FULL_CORPUS_SEMANTIC_AUDIT_RECEIPT.json"
AUDIT_GAME_STATS = MISSION / "FULL_AUDIT_GAME_STATS.csv.gz"
AUDIT_BINDINGS = MISSION / "FULL_AUDIT_BINDING_REGISTRY.json"
AUDIT_IMPLEMENTATIONS = MISSION / "FULL_AUDIT_IMPLEMENTATION_HASHES.json"
COMPOSITION = MISSION / "E1_CLEAN_REPLAY_COMPOSITION.json"
SPLIT_MANIFEST = MISSION / "GAME_LEVEL_SPLIT_MANIFEST.json"
SPLIT_ASSIGNMENTS = MISSION / "GAME_LEVEL_SPLIT_ASSIGNMENTS.json.gz"
MAPPING_REPORT = MISSION / "ELIGIBLE_ONLY_CACHE_MAPPING_REPORT.json"
DERIVED_MANIFEST = MISSION / "DEEPMEW_DERIVED_REPLAY_V1_MANIFEST.json"

EXPECTED_ROWS = 775_816
EXPECTED_GAMES = 21_185
EXPECTED_BLOCKS = 23_597
EXPECTED_ACTOR = 709_522
EXPECTED_FORCED = 66_294
EXPECTED_CLEAN_IDS_SHA256 = "b22b7433202afbbb7f8a0293840ab6662e1e0089deaa3e223cd497a847c5bb6d"
EXPECTED_ASSIGNMENT_SHA256 = "85c3ff01919ebb69f70e9a1d9e921755a4b9ea0927c42d3d15f789b7fa4f6884"
EXPECTED_CACHE_MANIFEST_SHA256 = "d7082677186be962eacc7b5639e0bf8f845413d485bc2b929df8cdcc17b47a1f"
EXPECTED_GAME_STATS_SHA256 = "4df2646df04cc085d9504851b4022d666f1322cfbc1adbbdbeb431e6b62fc970"
SPLIT_DOMAIN = "deepmew.derived_replay_v1.game_split.v1"
SPLIT_NAMES = ("train", "validation", "holdout")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(value) + b"\n")


def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as handle:
                handle.write(_canonical_bytes(value))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DerivedReplayHardStop(f"expected JSON object: {path}")

    return value


def _combined_code_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def _open_admission(database: Path) -> sqlite3.Connection:
    uri = f"file:{database.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=60.0)
    connection.execute("PRAGMA query_only=ON")

    return connection


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _selected_transitions(
    connection: sqlite3.Connection,
    binding_id: str,
    local_rows: Sequence[int],
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for chunk in _chunks(local_rows, 800):
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT local_row,compressed_json FROM transitions "
            f"WHERE binding_id=? AND local_row IN ({placeholders})"
        )
        for local_row, compressed in connection.execute(query, (binding_id, *chunk)):
            value = json.loads(zlib.decompress(compressed))
            if not isinstance(value, dict):
                raise DerivedReplayHardStop("selected Admission transition is not an object")
            selected[int(local_row)] = value
    if set(selected) != set(local_rows):
        raise DerivedReplayHardStop("selected Admission transition coverage is incomplete")

    return selected


def _transition_observation_key(transition: Mapping[str, Any]) -> tuple[str, int, str]:
    state_before = transition.get("state_before")
    reference = state_before.get("source_observation") if isinstance(state_before, Mapping) else None
    if not isinstance(reference, Mapping):
        raise DerivedReplayHardStop("selected transition lacks an observation reference")

    return (
        str(transition.get("source_hash") or "").lower(),
        int(reference["source_line"]),
        str(reference["content_sha256"]).lower(),
    )


def _selected_observations(
    connection: sqlite3.Connection,
    transitions: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    requested = sorted({_transition_observation_key(transition) for transition in transitions})
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for start in range(0, len(requested), 250):
        chunk = requested[start:start + 250]
        values = ",".join("(?,?,?)" for _ in chunk)
        parameters = tuple(item for key in chunk for item in key)
        query = (
            "WITH wanted(source_sha256,source_line,content_sha256) AS (VALUES "
            + values
            + ") SELECT o.source_sha256,o.source_line,o.content_sha256,o.compressed_json "
            "FROM wanted w JOIN observations o "
            "ON o.source_sha256=w.source_sha256 "
            "AND o.source_line=w.source_line "
            "AND o.content_sha256=w.content_sha256"
        )
        for source_sha, source_line, content_sha, compressed in connection.execute(query, parameters):
            value = json.loads(zlib.decompress(compressed))
            if not isinstance(value, dict):
                raise DerivedReplayHardStop("selected Admission observation is not an object")
            selected[(str(source_sha), int(source_line), str(content_sha))] = value
    if set(selected) != set(requested):
        raise DerivedReplayHardStop("selected Admission observation coverage is incomplete")

    return selected


@dataclass(frozen=True)
class CleanGame:
    canonical_game_id: str
    source_id: str
    actor_safe: bool
    decisions: int
    actor_rows: int
    forced_rows: int


@dataclass(frozen=True)
class SelectedGame:
    facts: CleanGame
    cache_coordinates: tuple[int, ...]


def _clean_games() -> dict[str, CleanGame]:
    if _sha256_file(AUDIT_GAME_STATS) != EXPECTED_GAME_STATS_SHA256:
        raise DerivedReplayHardStop("semantic game-stats hash drifted")
    selected: dict[str, CleanGame] = {}
    with gzip.open(AUDIT_GAME_STATS, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            decisions = int(row["decisions"])
            actor = int(row["valid_actor_and_critic"])
            forced = int(row["forced_critic_only"])
            clean = (
                int(row["rejected"]) == 0
                and int(row["unresolved_critic_only"]) == 0
                and decisions == actor + forced
            )
            if not clean:
                continue
            game_id = row["canonical_game_id"]
            selected[game_id] = CleanGame(
                canonical_game_id=game_id,
                source_id=row["source_id"],
                actor_safe=bool(int(row["actor_safe_candidate_span"])),
                decisions=decisions,
                actor_rows=actor,
                forced_rows=forced,
            )
    ordered_ids = sorted(selected)
    if (
        len(selected) != EXPECTED_GAMES
        or hashlib.sha256(_canonical_bytes(ordered_ids)).hexdigest() != EXPECTED_CLEAN_IDS_SHA256
    ):
        raise DerivedReplayHardStop("clean-game whitelist identity drifted")

    return selected


def _split_assignments(
    games: Mapping[str, CleanGame],
    semantic_receipt_sha256: str,
) -> dict[str, str]:
    strata: dict[tuple[str, bool], list[tuple[str, str]]] = defaultdict(list)
    for game in games.values():
        digest = hashlib.sha256(
            (
                SPLIT_DOMAIN
                + "\0"
                + semantic_receipt_sha256
                + "\0"
                + EXPECTED_CACHE_MANIFEST_SHA256
                + "\0"
                + game.canonical_game_id
            ).encode("utf-8")
        ).hexdigest()
        strata[(game.source_id, game.actor_safe)].append((digest, game.canonical_game_id))
    assignments: dict[str, str] = {}
    for values in strata.values():
        values.sort()
        size = len(values)
        raw = (0.8 * size, 0.1 * size, 0.1 * size)
        counts = [math.floor(value) for value in raw]
        remainder = size - sum(counts)
        order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
        for index in order[:remainder]:
            counts[index] += 1
        cursor = 0
        for split, count in zip(SPLIT_NAMES, counts, strict=True):
            for _, game_id in values[cursor:cursor + count]:
                assignments[game_id] = split
            cursor += count
    payload = [
        {"canonical_game_id": game_id, "split": assignments[game_id]}
        for game_id in sorted(assignments)
    ]
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if digest != EXPECTED_ASSIGNMENT_SHA256:
        raise DerivedReplayHardStop("derived game-level split identity drifted")
    _atomic_gzip_json(SPLIT_ASSIGNMENTS, payload)

    return assignments


def _lightweight_cache() -> tuple[Any, tuple[Any, ...], np.ndarray]:
    if _sha256_file(CACHE_MANIFEST) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise DerivedReplayHardStop("cache manifest hash drifted")
    manifest = _read_json(CACHE_MANIFEST)
    rows = int(manifest["rows"])
    output_files = manifest["output_files"]
    for filename, receipt in output_files.items():
        path = CACHE / filename
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
            raise DerivedReplayHardStop(f"cache artifact extent drifted: {filename}")
    with np.load(CACHE / "meta.npz", allow_pickle=False) as source:
        meta = {key: source[key] for key in source.files if key != "z"}
    with np.load(CACHE / "blocks.npz", allow_pickle=False) as source:
        block_z = source["z"].copy()
        blocks = {key: source[key] for key in source.files if key != "z"}
    with gzip.open(CACHE / "blocks_keys.json.gz", "rt", encoding="utf-8") as handle:
        keys = json.load(handle)
    data = SimpleNamespace(
        manifest=manifest,
        has_labels=False,
        meta=meta,
        blocks=blocks,
        block_keys=keys["block_keys"],
        source_ids=keys["source_ids"],
        policy_vocab=keys["policy_vocab"],
        deck_vocab=keys["deck_vocab"],
        split_names=keys["split_names"],
    )
    if any(array.shape != (rows,) for array in meta.values()):
        raise DerivedReplayHardStop("cache metadata shape drifted")
    starts = blocks["row_start"].astype(np.int64, copy=False)
    lengths = blocks["row_len"].astype(np.int64, copy=False)
    if (
        len(starts) != len(data.block_keys)
        or int(starts[0]) != 0
        or bool((lengths < 1).any())
        or bool((starts[1:] != starts[:-1] + lengths[:-1]).any())
        or int(starts[-1] + lengths[-1]) != rows
    ):
        raise DerivedReplayHardStop("cache block coverage drifted")
    canonical_root, _, inputs = _validate_cache_inputs(data, ORIGINAL_SPLIT)
    _, catalog_units = _load_catalog_units(data, canonical_root)
    bindings = _build_part_bindings(data, canonical_root, inputs, catalog_units)

    return data, bindings, block_z


def _selected_games(
    data: Any,
    clean: Mapping[str, CleanGame],
) -> tuple[list[SelectedGame], dict[int, int]]:
    train_index = data.split_names.index("train")
    accepted = (
        (data.meta["split_idx"] == train_index)
        & data.meta["is_target_eligible"]
    )
    by_game: dict[str, list[int]] = defaultdict(list)
    coordinate_to_block: dict[int, int] = {}
    selected_blocks = 0
    for block_id, block_key in enumerate(data.block_keys):
        game_id = block_key.rsplit(":s", 1)[0]
        if game_id not in clean:
            continue
        selected_blocks += 1
        start = int(data.blocks["row_start"][block_id])
        length = int(data.blocks["row_len"][block_id])
        for coordinate in np.flatnonzero(accepted[start:start + length]).tolist():
            global_coordinate = start + int(coordinate)
            by_game[game_id].append(global_coordinate)
            coordinate_to_block[global_coordinate] = block_id
    selected = []
    for game_id, facts in clean.items():
        coordinates = tuple(sorted(by_game.get(game_id, [])))
        if len(coordinates) != facts.decisions:
            raise DerivedReplayHardStop(f"clean-game cache count mismatch: {game_id}")
        selected.append(SelectedGame(facts=facts, cache_coordinates=coordinates))
    selected.sort(key=lambda game: game.cache_coordinates[0])
    flattened = [coordinate for game in selected for coordinate in game.cache_coordinates]
    if (
        selected_blocks != EXPECTED_BLOCKS
        or len(flattened) != EXPECTED_ROWS
        or len(set(flattened)) != EXPECTED_ROWS
        or flattened != sorted(flattened)
    ):
        raise DerivedReplayHardStop("selected clean cache-coordinate conservation failed")

    return selected, coordinate_to_block


def _implementation_identity() -> tuple[dict[str, Any], str]:
    frozen = _read_json(AUDIT_IMPLEMENTATIONS)
    causal = ROOT / frozen["causal_aligner"]["path"]
    legality = ROOT / frozen["legality_validator"]["path"]
    if _sha256_file(causal) != frozen["causal_aligner"]["sha256"]:
        raise DerivedReplayHardStop("frozen causal aligner implementation drifted")
    if _sha256_file(legality) != frozen["legality_validator"]["sha256"]:
        raise DerivedReplayHardStop("frozen legality implementation drifted")
    action_paths = [ROOT / path for path in frozen["action_schema"]["ordered_paths"]]
    if _combined_code_sha256(action_paths) != frozen["action_schema"]["sha256"]:
        raise DerivedReplayHardStop("frozen action-schema implementation drifted")
    regeneration_paths = (
        ROOT / "deepmew" / "replay" / "canonical_actor_dataset.py",
        ROOT / "scripts" / "build_value_q_canonical.py",
        ROOT / "scripts" / "run_unified_r3_value.py",
        ROOT / "scripts" / "run_unified_r4_ablation.py",
    )

    return frozen, _combined_code_sha256(regeneration_paths)


def _merge_counter(target: Counter[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


def _receipt_aggregates(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    action_families: Counter[str] = Counter()
    cardinalities: Counter[str] = Counter()
    continuity: Counter[str] = Counter()
    transition_proofs: Counter[str] = Counter()
    game_counts: dict[str, Counter[str]] = defaultdict(Counter)
    mapping = Counter()
    regenerated_assets = 0
    for receipt in receipts:
        _merge_counter(outcomes, receipt["validation_outcomes"])
        _merge_counter(sources, receipt["source_distribution"])
        _merge_counter(action_families, receipt["action_family_distribution"])
        _merge_counter(cardinalities, receipt["complete_legal_action_count_distribution"])
        _merge_counter(continuity, receipt["continuity_distribution"])
        _merge_counter(transition_proofs, receipt["resulting_transition_proof_distribution"])
        for game_id, counts in receipt["game_semantic_counts"].items():
            _merge_counter(game_counts[game_id], counts)
        storage = receipt["observation_storage"]
        for key in (
            "cache_reuse_rows",
            "cache_reuse_same_candidate_rows",
            "cache_reuse_alias_rows",
            "duplicate_cache_coordinate_reference_rows",
            "rows_with_multiple_exact_alias_candidates",
            "regenerated_reference_rows",
        ):
            mapping[key] += int(storage[key])
        regenerated_assets += int(storage["new_regenerated_assets"])

    return {
        "validation_outcomes": outcomes,
        "sources": sources,
        "action_families": action_families,
        "cardinalities": cardinalities,
        "continuity": continuity,
        "transition_proofs": transition_proofs,
        "game_counts": game_counts,
        "mapping": mapping,
        "regenerated_assets": regenerated_assets,
    }


def _feature_converter_identity(
    bindings: Sequence[Any],
    selected_sources: set[str],
) -> tuple[dict[str, str], str]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for binding in bindings:
        if binding.source_id not in selected_sources:
            continue
        by_source[binding.source_id].add(
            str(binding.unit_manifest.get("converter_hash", "")).lower()
        )
    if set(by_source) != selected_sources or any(len(values) != 1 for values in by_source.values()):
        raise DerivedReplayHardStop("selected source converter identities are incomplete or conflicting")
    resolved = {source: next(iter(values)) for source, values in sorted(by_source.items())}
    if any(len(value) != 64 for value in resolved.values()):
        raise DerivedReplayHardStop("one selected feature converter identity is invalid")

    return resolved, hashlib.sha256(_canonical_bytes(resolved)).hexdigest()


def _commit_buffer(
    builder: DerivedReplayReferenceBuilder,
    rows: list[dict[str, Any]],
    regenerated: list[tuple[str, np.ndarray]],
    registry: dict[str, tuple[int, int]],
) -> None:
    if not rows:
        return
    receipt = builder.commit(rows, regenerated_observations=regenerated)
    shard_index = int(receipt["shard_index"])
    for item in receipt["observation_storage"]["regenerated_assets"]:
        registry[str(item["pre_action_observation_id"])] = (
            shard_index,
            int(item["coordinate"]),
        )
    rows.clear()
    regenerated.clear()


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    receipt = _read_json(AUDIT_RECEIPT)
    if receipt.get("status") != "COMPLETE_READ_ONLY_EXACT":
        raise DerivedReplayHardStop("semantic audit receipt is not complete")
    admission = _read_json(ADMISSION_MANIFEST)
    if admission.get("database_sha256") != receipt.get("admission_database_sha256"):
        raise DerivedReplayHardStop("Admission identity differs from semantic authority")
    cache_manifest = _read_json(CACHE_MANIFEST)
    clean = _clean_games()
    assignments = _split_assignments(clean, str(receipt["receipt_sha256"]))
    data, bindings, block_z = _lightweight_cache()
    selected_games, coordinate_to_block = _selected_games(data, clean)
    frozen_implementations, regeneration_encoder_sha = _implementation_identity()
    feature_converters, feature_converter_set_sha = _feature_converter_identity(
        bindings,
        {game.source_id for game in clean.values()},
    )
    binding_registry_payload = _read_json(AUDIT_BINDINGS)
    binding_registry = {
        item["binding_id"]: item
        for item in binding_registry_payload["bindings"]
    }
    if len(binding_registry) != len(bindings):
        raise DerivedReplayHardStop("semantic binding registry coverage drifted")
    binding_ends = np.asarray([binding.row_end for binding in bindings], dtype=np.int64)
    train_index = data.split_names.index("train")
    eligibility = (
        (data.meta["split_idx"] == train_index)
        & data.meta["is_target_eligible"]
    )
    identity_value = DerivedReplayReferenceIdentity(
        semantic_audit_receipt_sha256=str(receipt["receipt_sha256"]),
        semantic_audit_identity=str(receipt["audit_identity"]),
        admission_identity=str(receipt["admission_identity"]),
        semantic_game_stats_sha256=EXPECTED_GAME_STATS_SHA256,
        cache_manifest_sha256=EXPECTED_CACHE_MANIFEST_SHA256,
        cache_x_sha256=str(cache_manifest["output_files"]["X.f32.bin"]["sha256"]),
        game_split_assignment_sha256=EXPECTED_ASSIGNMENT_SHA256,
        canonical_contract_version=str(receipt["contract_version"]),
        causal_aligner_implementation_sha256=str(
            frozen_implementations["causal_aligner"]["sha256"]
        ),
        legality_validator_implementation_sha256=str(
            frozen_implementations["legality_validator"]["sha256"]
        ),
        action_schema_implementation_sha256=str(
            frozen_implementations["action_schema"]["sha256"]
        ),
        feature_converter_set_sha256=feature_converter_set_sha,
        regeneration_source_converter_sha256=feature_converters["kaggle_ingest"],
        regeneration_encoder_code_sha256=regeneration_encoder_sha,
        feature_dim=FEATURE_DIM,
    )
    output_dir = Path(args.output_dir)
    builder = DerivedReplayReferenceBuilder(
        output_dir,
        identity_value,
        shard_rows=args.shard_rows,
    )
    committed_rows = builder.committed_rows
    skipped_games = 0
    skipped_rows = 0
    for game in selected_games:
        if skipped_rows == committed_rows:
            break
        if skipped_rows + len(game.cache_coordinates) > committed_rows:
            raise DerivedReplayHardStop("resume cursor splits a clean game")
        skipped_rows += len(game.cache_coordinates)
        skipped_games += 1
    if skipped_rows != committed_rows:
        raise DerivedReplayHardStop("resume cursor is not a clean-game boundary")
    regenerated_registry: dict[str, tuple[int, int]] = {}
    for prior in builder.receipts:
        shard_index = int(prior["shard_index"])
        for item in prior["observation_storage"]["regenerated_assets"]:
            observation_id = str(item["pre_action_observation_id"])
            if observation_id in regenerated_registry:
                raise DerivedReplayHardStop("regenerated observation asset is duplicated across shards")
            regenerated_registry[observation_id] = (shard_index, int(item["coordinate"]))

    database = Path(admission["database_path"])
    connection = _open_admission(database)
    repository = KaggleEpisodeRepository(ROOT)
    x = np.memmap(
        CACHE / "X.f32.bin",
        dtype=np.float32,
        mode="r",
        shape=(int(cache_manifest["rows"]), int(cache_manifest["feature_dim"])),
    )
    prefix_cache: dict[int, np.ndarray] = {}
    rows_buffer: list[dict[str, Any]] = []
    regenerated_buffer: list[tuple[str, np.ndarray]] = []
    pending_regenerated: dict[str, int] = {}
    selected_point_reads = 0
    conflicting_observation_identities = 0
    duplicate_alias_groups = 0

    def flush() -> None:
        nonlocal pending_regenerated
        _commit_buffer(
            builder,
            rows_buffer,
            regenerated_buffer,
            regenerated_registry,
        )
        pending_regenerated = {}

    for game_number, game in enumerate(selected_games[skipped_games:], start=skipped_games):
        coordinates = game.cache_coordinates
        binding_indices = np.searchsorted(binding_ends, np.asarray(coordinates), side="right")
        if len(set(binding_indices.tolist())) != 1:
            raise DerivedReplayHardStop("one clean game crosses cache input bindings")
        binding_index = int(binding_indices[0])
        binding = bindings[binding_index]
        binding_id = f"{binding.source_id}:{binding.unit_id}:{binding.part}"
        registered = binding_registry.get(binding_id)
        if registered is None:
            raise DerivedReplayHardStop("clean game binding is absent from semantic registry")
        local_rows = [coordinate - binding.row_start for coordinate in coordinates]
        transitions = _selected_transitions(connection, binding_id, local_rows)
        observations = _selected_observations(connection, transitions.values())
        selected_point_reads += len(local_rows)
        old_observations: dict[tuple[str, int], list[int]] = defaultdict(list)
        for coordinate, local_row in zip(coordinates, local_rows, strict=True):
            transition = transitions[local_row]
            _, _, content_sha = _transition_observation_key(transition)
            old_observations[(content_sha, int(transition["actor"]))].append(coordinate)
        prefix = prefix_cache.get(binding_index)
        if prefix is None:
            mask = eligibility[binding.row_start:binding.row_end]
            prefix = np.cumsum(mask, dtype=np.int64) - 1
            prefix_cache[binding_index] = prefix

        game_rows: list[dict[str, Any]] = []
        game_vectors: dict[str, np.ndarray] = {}
        reconstructed_actor = 0
        reconstructed_forced = 0
        for coordinate, local_row in zip(coordinates, local_rows, strict=True):
            if not bool(eligibility[coordinate]):
                raise DerivedReplayHardStop("clean-game coordinate is not an admitted target")
            admitted_ordinal = int(prefix[local_row])
            global_candidate_row = int(registered["global_candidate_row_start"]) + admitted_ordinal
            transition = transitions[local_row]
            source_record = observations[_transition_observation_key(transition)]
            event = source_event_from_admission(
                admission_identity=str(receipt["admission_identity"]),
                binding_id=binding_id,
                local_row=local_row,
                global_row=global_candidate_row,
                transition=transition,
                source_record=source_record,
                verify_source_content=True,
            )
            aligned = align_decision(
                event,
                kaggle_repository=repository,
                transition_registry=None,
            )
            legal = reconstruct_legal_actions(aligned)
            chosen = reconstruct_selected_action(aligned, legal)
            if (
                legal.failures
                or not legal.stages
                or legal.complete_action_count is None
                or legal.complete_action_count < 1
                or not chosen.complete
                or chosen.chosen_action_id is None
            ):
                raise DerivedReplayHardStop(
                    f"selected legal/chosen reconstruction failed: {game.facts.canonical_game_id}:{local_row}"
                )
            forced = legal.complete_action_count == 1
            reconstructed_forced += int(forced)
            reconstructed_actor += int(not forced)
            pre_asset = aligned.pre_action_event.observation_asset
            if pre_asset is None:
                raise DerivedReplayHardStop("selected corrected pre-action observation is absent")
            aliases = old_observations.get(
                (pre_asset.source_content_sha256, aligned.pre_action_event.actor_seat),
                [],
            )
            alias_coordinate: int | None = None
            if aliases:
                alias_coordinate = min(aliases)
                if len(aliases) > 1:
                    duplicate_alias_groups += 1
                    baseline = np.asarray(x[alias_coordinate])
                    if any(not np.array_equal(baseline, np.asarray(x[item])) for item in aliases[1:]):
                        conflicting_observation_identities += 1
                        raise DerivedReplayHardStop("one exact observation alias maps to conflicting X tensors")
            else:
                game_vectors[pre_asset.observation_id] = _row_features(
                    dict(aligned.pre_action_event.source_record),
                    dict(aligned.pre_action_event.observation or {}),
                )
            transition_sha = sha256_json(transition)
            stable_row_id = identity("auditrow1", {
                "semantic_audit_identity": receipt["audit_identity"],
                "global_candidate_row": global_candidate_row,
                "binding_id": binding_id,
                "local_row": local_row,
            })
            transition_reference_id = identity("transref1", {
                "admission_identity": receipt["admission_identity"],
                "binding_id": binding_id,
                "local_row": local_row,
                "source_transition_decision_id": transition["decision_id"],
                "source_transition_sha256": transition_sha,
            })
            block_id = coordinate_to_block[coordinate]
            terminal_outcome = transition.get("terminal_actor_relative_result")
            if terminal_outcome is None:
                terminal_outcome = float(block_z[block_id])
            validation_outcome = (
                "VALID_CRITIC_ONLY_FORCED"
                if forced
                else "VALID_ACTOR_AND_CRITIC"
            )
            game_rows.append({
                "schema_version": REFERENCE_ROW_SCHEMA_VERSION,
                "stable_row_id": stable_row_id,
                "semantic_audit_global_candidate_row": global_candidate_row,
                "source_game_id": game.facts.canonical_game_id,
                "trajectory_id": data.block_keys[block_id],
                "actor_seat": int(aligned.pre_action_event.actor_seat),
                "pre_action_observation_id": pre_asset.observation_id,
                "pre_action_observation_content_sha256": pre_asset.source_content_sha256,
                "source_candidate_cache_coordinate": coordinate,
                "observation_storage_kind": "cache" if alias_coordinate is not None else "regenerated",
                "observation_storage_coordinate": alias_coordinate if alias_coordinate is not None else -1,
                "observation_storage_shard": None,
                "observation_alias_candidate_count": len(aliases),
                "legal_action_set_id": legal.legal_action_set_id,
                "complete_chosen_action_id": chosen.chosen_action_id,
                "action_family": legal.action_family,
                "action_template": legal.action_template,
                "complete_legal_action_count": int(legal.complete_action_count),
                "source_transition_decision_id": str(transition["decision_id"]),
                "source_transition_sha256": transition_sha,
                "transition_reference_id": transition_reference_id,
                "resulting_transition_id": aligned.proof.resulting_event_id,
                "resulting_transition_proof": (
                    "terminal_exact"
                    if aligned.proof.resulting_event_id is not None
                    else "unproven_nonterminal_selected_only_no_registry"
                ),
                "terminal_outcome": float(terminal_outcome),
                "validation_outcome": validation_outcome,
                "forced_state": forced,
                "source_family": game.facts.source_id,
                "source_schema_version": aligned.pre_action_event.source_version,
                "canonical_transition_schema_version": str(transition["schema_version"]),
                "continuity": "full_trajectory" if game.facts.actor_safe else "isolated_or_segmented",
            })
        if (
            reconstructed_actor != game.facts.actor_rows
            or reconstructed_forced != game.facts.forced_rows
        ):
            raise DerivedReplayHardStop(
                f"selected semantic counts differ from audit: {game.facts.canonical_game_id}"
            )
        if rows_buffer and len(rows_buffer) + len(game_rows) > args.shard_rows:
            flush()
        current_shard = len(builder.receipts)
        for row in game_rows:
            if row["observation_storage_kind"] != "regenerated":
                continue
            observation_id = str(row["pre_action_observation_id"])
            prior = regenerated_registry.get(observation_id)
            if prior is not None:
                row["observation_storage_shard"], row["observation_storage_coordinate"] = prior
                continue
            coordinate = pending_regenerated.get(observation_id)
            if coordinate is None:
                coordinate = len(regenerated_buffer)
                pending_regenerated[observation_id] = coordinate
                regenerated_buffer.append((observation_id, game_vectors[observation_id]))
            row["observation_storage_shard"] = current_shard
            row["observation_storage_coordinate"] = coordinate
        rows_buffer.extend(game_rows)
        if args.progress_every_games and (game_number + 1) % args.progress_every_games == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            completed = builder.committed_rows + len(rows_buffer)
            print(json.dumps({
                "games_processed": game_number + 1,
                "rows_resolved": completed,
                "rows_per_second": completed / elapsed,
                "committed_shards": len(builder.receipts),
            }), flush=True)
    flush()
    connection.close()
    aggregates = _receipt_aggregates(builder.receipts)
    outcomes = aggregates["validation_outcomes"]
    if outcomes != Counter({
        "VALID_ACTOR_AND_CRITIC": EXPECTED_ACTOR,
        "VALID_CRITIC_ONLY_FORCED": EXPECTED_FORCED,
    }):
        raise DerivedReplayHardStop("final selected semantic outcome totals drifted")
    if len(aggregates["game_counts"]) != EXPECTED_GAMES:
        raise DerivedReplayHardStop("final selected game receipt coverage drifted")
    for game_id, facts in clean.items():
        if aggregates["game_counts"].get(game_id) != Counter({
            "VALID_ACTOR_AND_CRITIC": facts.actor_rows,
            "VALID_CRITIC_ONLY_FORCED": facts.forced_rows,
        }):
            raise DerivedReplayHardStop(f"final game receipt count drifted: {game_id}")

    mapping = aggregates["mapping"]
    final_resolved = int(mapping["cache_reuse_rows"] + mapping["regenerated_reference_rows"])
    if final_resolved != EXPECTED_ROWS:
        raise DerivedReplayHardStop("final observation asset coverage drifted")
    elapsed_seconds = time.perf_counter() - started
    mapping_report: dict[str, Any] = {
        "schema_version": "deepmew.eligible_only_cache_mapping_report.v1",
        "status": "COMPLETE_WITH_INDIVIDUAL_REGENERATION" if mapping["regenerated_reference_rows"] else "COMPLETE_ALL_CACHE_REUSED",
        "authority": {
            "semantic_audit_receipt_sha256": receipt["receipt_sha256"],
            "semantic_audit_identity": receipt["audit_identity"],
            "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
            "cache_x_sha256": identity_value.cache_x_sha256,
            "game_split_assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        },
        "selected_population": {
            "games": EXPECTED_GAMES,
            "seat_blocks": EXPECTED_BLOCKS,
            "rows_requested": EXPECTED_ROWS,
            "source_candidate_coordinates_found": EXPECTED_ROWS,
            "missing_source_candidate_coordinates": 0,
            "out_of_bounds_source_candidate_coordinates": 0,
            "duplicate_source_candidate_coordinates": 0,
            "source_candidate_count_delta": 0,
        },
        "corrected_pre_action_observation_resolution": {
            "exact_existing_cache_reuse_rows": int(mapping["cache_reuse_rows"]),
            "same_candidate_coordinate_rows": int(mapping["cache_reuse_same_candidate_rows"]),
            "different_selected_alias_coordinate_rows": int(mapping["cache_reuse_alias_rows"]),
            "initial_missing_cache_mapping_rows": int(mapping["regenerated_reference_rows"]),
            "rows_requiring_individual_regeneration": int(mapping["regenerated_reference_rows"]),
            "unique_regenerated_observation_assets": int(aggregates["regenerated_assets"]),
            "duplicate_cache_coordinate_alias_rows": int(mapping["duplicate_cache_coordinate_reference_rows"]),
            "rows_with_multiple_exact_alias_candidates": int(
                mapping["rows_with_multiple_exact_alias_candidates"]
            ),
            "multiple_alias_rows_verified_identical_X_this_invocation": duplicate_alias_groups,
            "conflicting_observation_identities_or_X": conflicting_observation_identities,
            "rows_excluded_after_cache_validation": 0,
            "final_resolved_rows": final_resolved,
        },
        "mapping_rule": {
            "refgame": "reuse the selected source-candidate coordinate only when corrected pre-action source content hash and actor seat match exactly",
            "kaggle": "resolve corrected raw pre-action content hash and actor seat against coordinates from the same selected clean game; choose the smallest exact alias after proving duplicate aliases have identical X",
            "missing": "regenerate only the individual actor-visible corrected pre-action observation with the frozen 144-column converter; never inspect rejected rows",
        },
        "expectation_resolution": {
            "source_candidate_coordinate_expectation_775816": "met",
            "corrected_pre_action_tensor_reuse_expectation_775816": (
                "met"
                if mapping["regenerated_reference_rows"] == 0
                else "not_met_individual_regeneration_rule_applied"
            ),
        },
        "work_scope": {
            "selected_admission_transition_point_reads_this_invocation": selected_point_reads,
            "selected_rows_receipted": EXPECTED_ROWS,
            "rejected_admission_rows_read": 0,
            "mixed_outcome_game_rows_read": 0,
            "admission_database_rehashed": False,
            "x_file_rehashed": False,
            "full_population_registry_created": False,
            "observation_tensors_duplicated_from_cache": 0,
        },
        "materialization_metrics": {
            "elapsed_seconds": elapsed_seconds,
            "rows_per_second": EXPECTED_ROWS / max(elapsed_seconds, 1e-9),
            "resume_start_committed_rows": committed_rows,
            "resume_skipped_committed_games": skipped_games,
        },
    }
    mapping_report["report_sha256"] = hashlib.sha256(_canonical_bytes(mapping_report)).hexdigest()
    _atomic_json(MAPPING_REPORT, mapping_report)

    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for game in clean.values():
        split = assignments[game.canonical_game_id]
        split_counts[split]["games"] += 1
        split_counts[split]["rows"] += game.decisions
        split_counts[split]["actor_rows"] += game.actor_rows
        split_counts[split]["forced_rows"] += game.forced_rows
    manifest_aggregates = {
        "rows": EXPECTED_ROWS,
        "games": EXPECTED_GAMES,
        "seat_blocks": EXPECTED_BLOCKS,
        "actor_rows": EXPECTED_ACTOR,
        "forced_rows": EXPECTED_FORCED,
        "source_distribution": dict(sorted(aggregates["sources"].items())),
        "continuity_distribution": dict(sorted(aggregates["continuity"].items())),
        "complete_legal_action_count_distribution": dict(
            sorted(aggregates["cardinalities"].items(), key=lambda item: int(item[0]))
        ),
        "action_family_distribution": dict(sorted(aggregates["action_families"].items())),
        "resulting_transition_proof_distribution": dict(sorted(aggregates["transition_proofs"].items())),
        "split_distribution": {
            split: dict(counts)
            for split, counts in sorted(split_counts.items())
        },
        "observation_asset_resolution": {
            "cache_reuse_rows": int(mapping["cache_reuse_rows"]),
            "regenerated_reference_rows": int(mapping["regenerated_reference_rows"]),
            "unique_regenerated_assets": int(aggregates["regenerated_assets"]),
            "excluded_rows": 0,
        },
        "materialization_metrics": mapping_report["materialization_metrics"],
        "feature_converters_by_selected_source": feature_converters,
    }
    manifest = builder.finalize(
        DERIVED_MANIFEST,
        expected_rows=EXPECTED_ROWS,
        aggregates=manifest_aggregates,
        cache_manifest_path=CACHE_MANIFEST,
        cache_mapping_report_path=MAPPING_REPORT,
        split_manifest_path=SPLIT_MANIFEST,
        split_assignments_path=SPLIT_ASSIGNMENTS,
    )
    composition = _read_json(COMPOSITION)
    composition["legal_action_cardinality"]["corrected_complete_action_count_distribution"] = (
        manifest_aggregates["complete_legal_action_count_distribution"]
    )
    composition["legal_action_cardinality"]["corrected_distribution_status"] = (
        "MATERIALIZED_SELECTED_ONLY_EXISTING_CANONICAL_IMPLEMENTATION"
    )
    composition["action_family"]["selected_only_distribution"] = (
        manifest_aggregates["action_family_distribution"]
    )
    composition["action_family"]["status"] = (
        "MATERIALIZED_SELECTED_ONLY_EXISTING_CANONICAL_IMPLEMENTATION"
    )
    composition["derived_v1_materialization"] = {
        "manifest_path": str(DERIVED_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": manifest["manifest_sha256"],
        "mapping_report_path": str(MAPPING_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "mapping_report_sha256": mapping_report["report_sha256"],
        "reference_rows": EXPECTED_ROWS,
        "observation_tensors_copied_from_cache": 0,
        "individual_regenerated_observation_assets": int(aggregates["regenerated_assets"]),
    }
    _atomic_json(COMPOSITION, composition)

    return {
        "manifest": manifest,
        "mapping_report": mapping_report,
        "composition_path": str(COMPOSITION),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(MISSION / "deepmew_derived_replay_v1"),
    )
    parser.add_argument("--shard-rows", type=int, default=8192)
    parser.add_argument("--progress-every-games", type=int, default=250)
    args = parser.parse_args()
    result = _materialize(args)
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
