"""Run the zero-paid post-E1 opponent-model M0/M1 contribution audit.

The corpus is fresh and actor-visible. D1 is the fixed collection anchor. Only
the frozen 20 reference cells plus r7, E1, and GRIMM_CONTROL contribute target
decision rows. No historical replay is read and no evaluation is rerun.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for candidate in (ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

import run_prospective_trace as prospective
import run_r17_bucket_arena as r17
import run_r9_r4f_retrain as r9
import run_unified_r4_ablation as r4
from deepmew.artifacts.writers import write_json
from deepmew.deck import read_deck
from deepmew.option_identity_bc import EncodedDecision
from deepmew.real_meta_arena.runner import SpecAgent
from deepmew.replay.information_boundary import InformationBoundary, InformationMode
from deepmew.replay.prospective_trace import (
    ACTOR_VISIBLE_ROW_SCHEMA_VERSION,
    PUBLIC_PROJECTION_VERSION,
    ActorMetadata,
    _assert_public_observation_safe,
    assert_actor_visible_trace_row_safe,
    atomic_write_gzip_jsonl,
    public_trace_observation,
    unavailable_rating_snapshot,
)
from deepmew.rl.algorithms.r4f_ppo import R4FActor, R4FActorConfig
from deepmew.rl.metadata import normalized_deck_hash
from deepmew.second_order.reader import augment_option_features


SCHEMA_VERSION = "deepmew.post_e1_request_a_m0_m1.v1"
CORPUS_SCHEMA_VERSION = "deepmew.post_e1_opponent_action_corpus.v1"
LEAKAGE_SCHEMA_VERSION = "deepmew.post_e1_opponent_action_leakage.v1"
RESULT_SCHEMA_VERSION = "deepmew.post_e1_opponent_model_result.v1"
CONFIG_NAME = "FROZEN_CONFIG.json"
REGISTRY_PATH = ROOT / ".research" / "post_e1_population_v0" / "POST_E1_POPULATION_REGISTRY.json"
DECK_FEATURE_PATH = ROOT / ".research" / "post_e1_population_v0" / "POST_E1_DECK_FEATURES.json"
BUCKET_PATH = ROOT / ".research" / "runs" / "bucket_pool" / "20260707" / "20260707T141406Z-bucket_pool-1c29c3dd" / "bucket_pools.json"
R7_PATH = Path(r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt")
EXPLOITER_PATH = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
TRANCHE_GAMES_PER_ORIENTATION = {
    "S0_CANARY": 1,
    "T0_TRAIN": 3,
    "T0_CALIBRATION": 1,
    "T1_TEST": 3,
}
TRANCHE_SEED_OFFSETS = {
    "S0_CANARY": 0,
    "T0_TRAIN": 100,
    "T0_CALIBRATION": 200,
    "T1_TEST": 300,
}
BASE_SEED = 719_010_000
BOOTSTRAP_SEED = 718_020_026
TRAIN_SEED = 718_030_026
MAX_STEPS = 5_000
FORBIDDEN_ORACLE_FIELDS = {
    "actor_private_observation",
    "chosen_indices_private",
    "exact_legal_actions",
    "search_begin_input",
    "true_private_state",
}
DECK_SCALAR_FIELDS = (
    "pokemon",
    "trainer",
    "energy",
    "basic_pokemon",
    "stage1_pokemon",
    "stage2_pokemon",
    "mega_pokemon",
    "ex_rule_cards",
    "unique_cards",
    "unknown_card_ids",
)
MODEL_CONFIG = {
    "architecture": "r7_lineage_r4f_intent_autoregressive_masked_action_head",
    "source_checkpoint": str(R7_PATH),
    "source_input_dim": 304,
    "hidden_dim": 512,
    "depth": 3,
    "embed_dim": 256,
    "dropout": 0.0,
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "weight_decay": 0.0001,
    "epochs": 3,
    "batch_size": 256,
    "gradient_clip_norm": 1.0,
    "temperature_calibration": "single_log_temperature_LBFGS_on_T0_CALIBRATION",
    "temperature_bounds": [0.1, 10.0],
    "bootstrap_resamples": 10_000,
    "primary_gate": "paired_game_bootstrap_95pct_CI_of_M0_minus_M1_complete_action_NLL_excludes_zero_above",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

    return hashlib.sha256(encoded).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_deck(path: Path, card_ids: Sequence[int]) -> None:
    if len(card_ids) != 60:
        raise ValueError(f"deck must contain 60 cards: {path}")
    payload = "\n".join(str(int(card_id)) for card_id in card_ids) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"materialized deck differs from frozen card list: {path}")

        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _deck_feature_schema(deck_features: Mapping[str, Any]) -> dict[str, Any]:
    energy_types = sorted({
        str(energy_type)
        for row in _as_list(deck_features.get("decks"))
        for energy_type in _as_dict(_as_dict(row).get("features")).get("energy_types", {})
    })
    names = [f"count_fraction:{field}" for field in DECK_SCALAR_FIELDS]
    names.append("max_hp_fraction_of_400")
    names.extend(f"energy_type_fraction:{energy_type}" for energy_type in energy_types)

    return {
        "names": names,
        "energy_types": energy_types,
        "normalization": "card counts / 60; max_hp / 400",
    }


def deck_feature_vector(features: Mapping[str, Any], schema: Mapping[str, Any]) -> list[float]:
    vector = [float(features.get(field) or 0.0) / 60.0 for field in DECK_SCALAR_FIELDS]
    vector.append(float(features.get("max_hp") or 0.0) / 400.0)
    energy_counts = _as_dict(features.get("energy_types"))
    vector.extend(float(energy_counts.get(energy_type) or 0.0) / 60.0 for energy_type in schema["energy_types"])
    if len(vector) != len(schema["names"]):
        raise ValueError("deck feature vector width drifted")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("deck feature vector contains a non-finite value")

    return vector


def _deck_feature_by_identity(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for value in _as_list(payload.get("decks")):
        row = _as_dict(value)
        identity = str(row.get("deck_identity") or "")
        features = row.get("features")
        if identity and isinstance(features, dict):
            rows[identity] = features

    return rows


def _checkpoint_receipt(path: Path, expected: str | None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256_file(resolved)
    if expected and actual != expected:
        raise ValueError(f"checkpoint identity mismatch: {resolved}: {actual} != {expected}")

    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def _target_policy(
    *,
    member: Mapping[str, Any],
    policy_spec: str,
    live_feature_policy_id: str,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    receipt = None
    if checkpoint_path is not None:
        receipt = _checkpoint_receipt(checkpoint_path, member.get("policy_checkpoint_sha256"))

    return {
        "policy_spec": policy_spec,
        "live_feature_policy_id": live_feature_policy_id,
        "checkpoint": receipt,
    }


def _freeze_targets(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    registry = _read_json(REGISTRY_PATH)
    deck_features = _read_json(DECK_FEATURE_PATH)
    pool = _read_json(BUCKET_PATH)
    pool_by_hash = {
        str(row["deck_hash"]): row
        for rows in _as_dict(pool.get("pools")).values()
        for row in _as_list(rows)
    }
    feature_by_identity = _deck_feature_by_identity(deck_features)
    feature_schema = _deck_feature_schema(deck_features)
    participant_by_alias = {str(row["alias"]): row for row in _as_list(registry.get("participants"))}
    targets: list[dict[str, Any]] = []
    deck_dir = out_dir / "corpus" / "decks"
    for member in _as_list(registry.get("reference_cells")):
        deck_identity = str(member["deck_identity_normalized"])
        pool_row = _as_dict(pool_by_hash.get(deck_identity))
        card_ids = [int(value) for value in _as_list(pool_row.get("card_ids"))]
        deck_path = deck_dir / f"{deck_identity}.deck.csv"
        _write_deck(deck_path, card_ids)
        policy_identity = str(member["policy_identity"])
        checkpoint_path = EXPLOITER_PATH if policy_identity == "exploiter" else None
        policy_spec = f"checkpoint:{checkpoint_path.resolve()}" if checkpoint_path else policy_identity
        target = {
            "target_index": len(targets),
            "member_id": str(member["member_id"]),
            "alias": str(member["alias"]),
            "slice_kind": "reference_cell",
            "reference_cell_id": str(member["cell_id"]),
            "policy_identity": policy_identity,
            "deck_identity": deck_identity,
            "deck": {
                "path": str(deck_path.resolve()),
                "file_sha256": sha256_file(deck_path),
                "normalized_hash": normalized_deck_hash(read_deck(deck_path)),
            },
            "deck_feature_vector": deck_feature_vector(feature_by_identity[deck_identity], feature_schema),
            "policy": _target_policy(
                member=member,
                policy_spec=policy_spec,
                live_feature_policy_id="bucket_exploiter" if checkpoint_path else policy_identity,
                checkpoint_path=checkpoint_path,
            ),
        }
        targets.append(target)
    participant_specs = {
        "r7": (R7_PATH, "starmie"),
        "E1": (ROOT / str(participant_by_alias["E1"]["policy_checkpoint_path"]), "starmie"),
        "GRIMM_CONTROL": (
            ROOT / str(participant_by_alias["GRIMM_CONTROL"]["policy_checkpoint_path"]),
            "grimmsnarl",
        ),
    }
    for alias in ("r7", "E1", "GRIMM_CONTROL"):
        member = participant_by_alias[alias]
        checkpoint_path, live_feature_policy_id = participant_specs[alias]
        deck_path = ROOT / str(member["deck_path"])
        deck_identity = str(member.get("deck_identity_normalized") or normalized_deck_hash(read_deck(deck_path)))
        feature_identity = str(member["deck_sha256"])
        targets.append({
            "target_index": len(targets),
            "member_id": str(member["member_id"]),
            "alias": alias,
            "slice_kind": "frozen_participant",
            "reference_cell_id": None,
            "policy_identity": alias,
            "deck_identity": deck_identity,
            "deck": {
                "path": str(deck_path.resolve()),
                "file_sha256": sha256_file(deck_path),
                "normalized_hash": normalized_deck_hash(read_deck(deck_path)),
            },
            "deck_feature_vector": deck_feature_vector(feature_by_identity[feature_identity], feature_schema),
            "policy": _target_policy(
                member=member,
                policy_spec=f"checkpoint:{checkpoint_path.resolve()}",
                live_feature_policy_id=live_feature_policy_id,
                checkpoint_path=checkpoint_path,
            ),
        })
    if len(targets) != 23 or len({row["member_id"] for row in targets}) != 23:
        raise ValueError("Request A target set must contain exactly 23 distinct frozen members")
    anchor = participant_by_alias["D1"]
    anchor_path = ROOT / str(anchor["policy_checkpoint_path"])
    anchor_deck_path = ROOT / str(anchor["deck_path"])
    anchor_payload = {
        "member_id": str(anchor["member_id"]),
        "alias": "D1",
        "policy": _target_policy(
            member=anchor,
            policy_spec=f"checkpoint:{anchor_path.resolve()}",
            live_feature_policy_id="starmie",
            checkpoint_path=anchor_path,
        ),
        "deck": {
            "path": str(anchor_deck_path.resolve()),
            "file_sha256": sha256_file(anchor_deck_path),
            "normalized_hash": normalized_deck_hash(read_deck(anchor_deck_path)),
        },
    }

    return targets, anchor_payload, feature_schema


def freeze_config(out_dir: Path) -> dict[str, Any]:
    config_path = out_dir / CONFIG_NAME
    if config_path.exists():
        return _read_json(config_path)
    if out_dir.exists():
        unexpected = [
            path
            for path in out_dir.rglob("*")
            if path.is_file() and path.parent != out_dir / "corpus" / "decks"
        ]
        if unexpected:
            raise FileExistsError(f"unfrozen Request A output contains unexpected files: {unexpected}")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets, anchor, feature_schema = _freeze_targets(out_dir)
    source_receipts = {
        "population_registry": {"path": str(REGISTRY_PATH), "sha256": sha256_file(REGISTRY_PATH)},
        "deck_features": {"path": str(DECK_FEATURE_PATH), "sha256": sha256_file(DECK_FEATURE_PATH)},
        "bucket_pool": {"path": str(BUCKET_PATH), "sha256": sha256_file(BUCKET_PATH)},
        "entrypoint": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "prospective_trace": {
            "path": str(Path(prospective.__file__).resolve()),
            "sha256": sha256_file(Path(prospective.__file__)),
        },
        "actor_visible_contract": {
            "path": str(Path(sys.modules[assert_actor_visible_trace_row_safe.__module__].__file__).resolve()),
            "sha256": sha256_file(Path(sys.modules[assert_actor_visible_trace_row_safe.__module__].__file__)),
        },
        "r4_encoder": {"path": str(Path(r4.__file__).resolve()), "sha256": sha256_file(Path(r4.__file__))},
        "r9_encoder": {"path": str(Path(r9.__file__).resolve()), "sha256": sha256_file(Path(r9.__file__))},
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "status": "AUTHORIZE_POST_E1_REQUEST_A_FROZEN_BEFORE_FIRST_GAME",
        "created_at_utc": utc_now(),
        "authorization": "AUTHORIZE_POST_E1_REQUEST_A",
        "cost": {"paid_jobs": 0, "modal_jobs": 0, "execution": "local_cpu_only"},
        "collection": {
            "anchor": anchor,
            "targets": targets,
            "orientations": ["target_seat_0", "target_seat_1"],
            "tranche_games_per_orientation": TRANCHE_GAMES_PER_ORIENTATION,
            "base_seed": BASE_SEED,
            "max_steps": MAX_STEPS,
            "information_mode": InformationMode.ACTOR_VISIBLE.value,
            "public_projection_version": PUBLIC_PROJECTION_VERSION,
            "persisted_rows": "target_actor_only",
            "source": "fresh_seeded_games_only_no_replay_reconstruction",
        },
        "features": {
            "M0": "304-wide r7-lineage public-state projection plus legal-action mask with identity channels zeroed",
            "M1": "M0 plus frozen interpretable deck feature vector concatenated to every legal option",
            "deck_feature_schema": feature_schema,
            "checkpoint_or_member_identity_input": False,
        },
        "model": MODEL_CONFIG,
        "source_receipts": source_receipts,
        "prohibited_work": [
            "paid_exploiters",
            "league_training",
            "hybrids",
            "mixture_of_experts",
            "new_branch_mining",
            "replay_reconstruction",
        ],
    }
    config["contract_sha256"] = stable_hash(config)
    write_json(config_path, config)

    return config


def verify_config(config: Mapping[str, Any]) -> None:
    expected = dict(config)
    contract_sha = str(expected.pop("contract_sha256", ""))
    if not contract_sha or stable_hash(expected) != contract_sha:
        raise ValueError("frozen Request A config contract hash mismatch")
    for receipt in _as_dict(config.get("source_receipts")).values():
        path = Path(str(receipt["path"]))
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"Request A source identity drifted: {path}")
    if config.get("authorization") != "AUTHORIZE_POST_E1_REQUEST_A":
        raise ValueError("wrong post-E1 authorization")


def _agent(policy: Mapping[str, Any], deck: Sequence[int], deck_hash: str) -> Any:
    checkpoint = policy.get("checkpoint")
    if isinstance(checkpoint, dict):
        return r17.build_exploiter(
            Path(str(checkpoint["path"])),
            list(deck),
            deck_hash,
            "reference_abomasnow",
            feature_policy_id=str(policy["live_feature_policy_id"]),
        )

    return SpecAgent.from_str(str(policy["policy_spec"]))


def _actor_metadata(member_id: str, policy: Mapping[str, Any], deck: Mapping[str, Any], *, alias: str) -> ActorMetadata:
    checkpoint = policy.get("checkpoint")
    policy_metadata = {
        "feature_policy_id": member_id,
        "policy_spec": str(policy["policy_spec"]),
        "policy_family": "checkpoint" if checkpoint else "named_policy",
        "checkpoint_path": checkpoint.get("path") if isinstance(checkpoint, dict) else None,
        "checkpoint_sha256": checkpoint.get("sha256") if isinstance(checkpoint, dict) else None,
        "live_feature_policy_id": str(policy["live_feature_policy_id"]),
        "alias": alias,
    }
    deck_metadata = {
        "deck_id": str(deck["normalized_hash"]),
        "deck_hash": str(deck["normalized_hash"]),
        "deck_path": str(deck["path"]),
        "deck_file_sha256": str(deck["file_sha256"]),
        "card_count": 60,
    }

    return ActorMetadata(seat=-1, policy=policy_metadata, deck=deck_metadata)


def _row_has_forbidden_nested_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_ORACLE_FIELDS.intersection(value):
            return True

        return any(_row_has_forbidden_nested_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_row_has_forbidden_nested_key(item) for item in value)

    return False


def _assert_public_source_projection(row: Mapping[str, Any]) -> None:
    raw = _as_dict(row.get("raw_observation"))
    public = _as_dict(row.get("public_observation_before"))
    expected = public_trace_observation(raw)
    if public != expected:
        raise ValueError(f"public projection drifted from actor-visible source: {row.get('decision_id')}")
    _assert_public_observation_safe(public, field="public_observation_before")


def _persist_target_rows(rows: list[dict[str, Any]], target: Mapping[str, Any], shard_path: Path) -> tuple[int, int]:
    target_rows = [row for row in rows if row.get("policy_id") == target["member_id"]]
    if not target_rows:
        raise ValueError(f"game produced no target decisions: {target['alias']}")
    for row in rows:
        assert_actor_visible_trace_row_safe(row)
        _assert_public_source_projection(row)
    for row in target_rows:
        raw_select = _as_dict(_as_dict(row.get("raw_observation")).get("select"))
        legal_set = _as_list(row.get("legal_set"))
        row["action_mask"] = {
            "n_options": len(legal_set),
            "min_count": _as_int(raw_select.get("minCount"), 0),
            "max_count": _as_int(raw_select.get("maxCount"), len(legal_set)),
        }
        row["target_member_id"] = str(target["member_id"])
        row["target_alias"] = str(target["alias"])
        row["target_slice_kind"] = str(target["slice_kind"])
        row["reference_cell_id"] = target.get("reference_cell_id")
        row["target_policy_identity"] = str(target["policy_identity"])
        row["target_deck_identity"] = str(target["deck_identity"])
        row["source_shard"] = str(shard_path.resolve())
        assert_actor_visible_trace_row_safe(row)
        if _row_has_forbidden_nested_key(row):
            raise ValueError(f"forbidden oracle/search field in actor-visible row: {row['decision_id']}")
    atomic_write_gzip_jsonl(shard_path, target_rows)

    return len(rows), len(target_rows)


def collect_corpus(out_dir: Path, config: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    manifest_path = out_dir / "CORPUS_MANIFEST.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("contract_sha256") != config["contract_sha256"]:
            raise ValueError("existing corpus belongs to a different frozen contract")

        return manifest
    shard_dir = out_dir / "corpus" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    anchor = _as_dict(_as_dict(config["collection"])["anchor"])
    anchor_deck = read_deck(anchor["deck"]["path"])
    anchor_hash = str(anchor["deck"]["normalized_hash"])
    target_entries: list[dict[str, Any]] = []
    safety = {"completed": 0, "crashes": 0, "timeouts": 0, "illegal": 0}
    started = time.perf_counter()
    for target in _as_list(_as_dict(config["collection"]).get("targets")):
        target_deck = read_deck(target["deck"]["path"])
        target_hash = str(target["deck"]["normalized_hash"])
        for orientation_index, orientation in enumerate(("target_seat_0", "target_seat_1")):
            target_seat = orientation_index
            anchor_seat = 1 - target_seat
            target_meta = _actor_metadata(
                str(target["member_id"]), target["policy"], target["deck"], alias=str(target["alias"]),
            )
            anchor_meta = _actor_metadata(
                str(anchor["member_id"]), anchor["policy"], anchor["deck"], alias="D1",
            )
            actors_by_seat = {
                target_seat: ActorMetadata(target_seat, target_meta.policy, target_meta.deck),
                anchor_seat: ActorMetadata(anchor_seat, anchor_meta.policy, anchor_meta.deck),
            }
            decks = {target_seat: target_deck, anchor_seat: anchor_deck}
            agents = {
                target_seat: _agent(target["policy"], target_deck, target_hash),
                anchor_seat: _agent(anchor["policy"], anchor_deck, anchor_hash),
            }
            for tranche, game_count in TRANCHE_GAMES_PER_ORIENTATION.items():
                tranche_payload = {
                    "tranche_id": tranche,
                    "tranche_sequence": list(TRANCHE_GAMES_PER_ORIENTATION).index(tranche),
                    "declared_at_utc": str(config["created_at_utc"]),
                    "frozen_snapshot": {"rating_snapshot": unavailable_rating_snapshot()},
                }
                for game_index in range(game_count):
                    seed = (
                        BASE_SEED
                        + int(target["target_index"]) * 10_000
                        + orientation_index * 1_000
                        + TRANCHE_SEED_OFFSETS[tranche]
                        + game_index
                    )
                    game_id = f"post-e1-request-a:{target['target_index']:02d}:{orientation}:{tranche}:g{game_index:03d}"
                    shard_path = shard_dir / f"target_{target['target_index']:02d}_{orientation}_{tranche}_g{game_index:03d}.jsonl.gz"
                    if shard_path.exists():
                        if not resume:
                            raise FileExistsError(f"partial corpus exists; pass --resume: {shard_path}")
                        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
                            persisted = [json.loads(line) for line in handle if line.strip()]
                        if not persisted or {row.get("game_id") for row in persisted} != {game_id}:
                            raise ValueError(f"durable corpus shard identity mismatch: {shard_path}")
                        for row in persisted:
                            assert_actor_visible_trace_row_safe(row)
                            _assert_public_source_projection(row)
                            expected_identity = (
                                row.get("target_member_id") == target["member_id"]
                                and row.get("target_alias") == target["alias"]
                                and row.get("tranche_id") == tranche
                                and _as_int(row.get("seed"), -1) == seed
                            )
                            if not expected_identity or _row_has_forbidden_nested_key(row):
                                raise ValueError(f"durable corpus shard contract mismatch: {shard_path}")
                        target_entries.append({
                            "path": str(shard_path.relative_to(out_dir)),
                            "sha256": sha256_file(shard_path),
                            "bytes": shard_path.stat().st_size,
                            "game_id": game_id,
                            "seed": seed,
                            "tranche": tranche,
                            "orientation": orientation,
                            "target_member_id": target["member_id"],
                            "target_alias": target["alias"],
                            "target_decisions": len(persisted),
                            "resumed": True,
                        })
                        safety["completed"] += 1
                        continue
                    try:
                        rows, game_safety = prospective._play_game(
                            game_id=game_id,
                            run_id="post-e1-request-a-m0-m1-v1",
                            tranche=tranche_payload,
                            seed=seed,
                            actors=[actors_by_seat[0], actors_by_seat[1]],
                            agents=agents,
                            decks=decks,
                            league_identity="post-e1-frozen-population-v0",
                            curriculum_round=None,
                            boundary=InformationBoundary(mode=InformationMode.ACTOR_VISIBLE),
                            max_steps=MAX_STEPS,
                            game_index=game_index,
                        )
                    except Exception as exc:
                        failure_safety = prospective._exception_safety(exc)
                        for key in ("crashes", "timeouts", "illegal"):
                            safety[key] += failure_safety[key]
                        write_json(out_dir / "SAFETY_FAILURE_RECEIPT.json", {
                            "schema_version": SCHEMA_VERSION,
                            "status": "HARD_STOP_SAFETY_FAILURE",
                            "created_at_utc": utc_now(),
                            "contract_sha256": config["contract_sha256"],
                            "game_id": game_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "safety": safety,
                        })
                        raise
                    for key in ("crashes", "timeouts", "illegal"):
                        safety[key] += int(game_safety[key])
                    if any(game_safety[key] for key in ("crashes", "timeouts", "illegal")):
                        write_json(out_dir / "SAFETY_FAILURE_RECEIPT.json", {
                            "schema_version": SCHEMA_VERSION,
                            "status": "HARD_STOP_SAFETY_FAILURE",
                            "created_at_utc": utc_now(),
                            "contract_sha256": config["contract_sha256"],
                            "game_id": game_id,
                            "safety": safety,
                            "game_safety": game_safety,
                        })
                        raise RuntimeError(f"Request A safety hard stop: {game_safety}")
                    _all_decisions, target_decisions = _persist_target_rows(rows, target, shard_path)
                    safety["completed"] += 1
                    target_entries.append({
                        "path": str(shard_path.relative_to(out_dir)),
                        "sha256": sha256_file(shard_path),
                        "bytes": shard_path.stat().st_size,
                        "game_id": game_id,
                        "seed": seed,
                        "tranche": tranche,
                        "orientation": orientation,
                        "target_member_id": target["member_id"],
                        "target_alias": target["alias"],
                        "target_decisions": target_decisions,
                        "resumed": False,
                    })
                    if safety["completed"] % 25 == 0:
                        print(
                            f"corpus games={safety['completed']}/{23 * 2 * sum(TRANCHE_GAMES_PER_ORIENTATION.values())} "
                            f"decisions={sum(row['target_decisions'] for row in target_entries)} "
                            f"elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
    target_entries.sort(key=lambda row: row["path"])
    expected_games = 23 * 2 * sum(TRANCHE_GAMES_PER_ORIENTATION.values())
    if safety != {"completed": expected_games, "crashes": 0, "timeouts": 0, "illegal": 0}:
        raise RuntimeError(f"Request A corpus did not close safety-clean: {safety}")
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "status": "COMPLETE_SAFETY_CLEAN",
        "created_at_utc": utc_now(),
        "contract_sha256": config["contract_sha256"],
        "information_mode": InformationMode.ACTOR_VISIBLE.value,
        "source": "fresh_seeded_games_no_replay_reconstruction",
        "anchor": "D1",
        "target_members": 23,
        "reference_cells": 20,
        "frozen_participants": ["r7", "E1", "GRIMM_CONTROL"],
        "games": expected_games,
        "decisions": sum(row["target_decisions"] for row in target_entries),
        "by_tranche": {
            tranche: {
                "games": sum(row["tranche"] == tranche for row in target_entries),
                "decisions": sum(row["target_decisions"] for row in target_entries if row["tranche"] == tranche),
            }
            for tranche in TRANCHE_GAMES_PER_ORIENTATION
        },
        "safety": safety,
        "shards": target_entries,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest["shard_identity_sha256"] = stable_hash([
        {key: row[key] for key in ("path", "sha256", "game_id", "seed", "target_member_id")}
        for row in target_entries
    ])
    write_json(manifest_path, manifest)

    return manifest


def _iter_rows(out_dir: Path, manifest: Mapping[str, Any], tranche: str) -> Iterator[dict[str, Any]]:
    for shard in _as_list(manifest.get("shards")):
        if shard.get("tranche") != tranche:
            continue
        path = out_dir / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"corpus shard drifted: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def build_leakage_receipt(out_dir: Path, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = out_dir / "LEAKAGE_RECEIPT.json"
    if receipt_path.exists():
        return _read_json(receipt_path)
    rows = 0
    action_contracts = 0
    schemas: Counter[str] = Counter()
    projections: Counter[str] = Counter()
    decision_ids: set[str] = set()
    games: set[str] = set()
    for tranche in TRANCHE_GAMES_PER_ORIENTATION:
        for row in _iter_rows(out_dir, manifest, tranche):
            assert_actor_visible_trace_row_safe(row)
            _assert_public_source_projection(row)
            if _row_has_forbidden_nested_key(row):
                raise ValueError(f"leakage receipt found a forbidden field: {row.get('decision_id')}")
            decision_id = str(row.get("decision_id") or "")
            if not decision_id or decision_id in decision_ids:
                raise ValueError(f"missing or duplicate decision identity: {decision_id}")
            decision_ids.add(decision_id)
            games.add(str(row["game_id"]))
            rows += 1
            action_contracts += 1
            schemas[str(row.get("schema_version"))] += 1
            projections[str(row.get("public_projection_version"))] += 1
    if rows != int(manifest["decisions"]) or len(games) != int(manifest["games"]):
        raise ValueError("leakage scan coverage differs from corpus manifest")
    receipt = {
        "schema_version": LEAKAGE_SCHEMA_VERSION,
        "status": "PASS_ACTOR_VISIBLE_LEGALITY_VALIDATED_ZERO_ORACLE_LEAKAGE",
        "created_at_utc": utc_now(),
        "contract_sha256": config["contract_sha256"],
        "corpus_manifest_sha256": sha256_file(out_dir / "CORPUS_MANIFEST.json"),
        "games_scanned": len(games),
        "rows_scanned": rows,
        "unique_decision_ids": len(decision_ids),
        "action_contracts_validated": action_contracts,
        "information_mode": InformationMode.ACTOR_VISIBLE.value,
        "actor_visible_schema_counts": dict(schemas),
        "public_projection_counts": dict(projections),
        "forbidden_oracle_or_search_fields": 0,
        "opponent_hidden_zone_fields": 0,
        "checkpoint_or_member_identity_model_inputs": 0,
        "m0_state_inputs": "public_projection_plus_validated_legal_action_mask",
        "m0_actor_private_hand_or_prize_identity_inputs": 0,
        "m0_identity_channels": "zeroed_before_second_order_encoding",
        "m1_additional_inputs": "registered interpretable deck feature vector only",
        "historical_replay_rows": 0,
        "replay_reconstruction": False,
        "paid_jobs": 0,
        "safety": manifest["safety"],
        "passed": True,
    }
    write_json(receipt_path, receipt)

    return receipt


def _public_model_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = copy.deepcopy(_as_dict(row.get("public_observation_before")))
    raw_select = _as_dict(_as_dict(row.get("raw_observation")).get("select"))
    legal_set = [_as_dict(value) for value in _as_list(row.get("legal_set"))]
    action_mask = _as_dict(row.get("action_mask"))
    observation["select"] = {
        "context": raw_select.get("context"),
        "type": raw_select.get("type"),
        "minCount": _as_int(action_mask.get("min_count"), 0),
        "maxCount": _as_int(action_mask.get("max_count"), len(legal_set)),
        "option": copy.deepcopy(legal_set),
    }

    return observation


def encode_row(row: Mapping[str, Any], *, deck_vector: Sequence[float] | None) -> EncodedDecision:
    assert_actor_visible_trace_row_safe(dict(row))
    observation = _public_model_observation(row)
    legal_set = [_as_dict(value) for value in _as_list(row.get("legal_set"))]
    if not observation or not legal_set:
        raise ValueError(f"opponent-model row lacks observation/legal set: {row.get('decision_id')}")
    base = r4.build_feature_matrix(observation, legal_set, policy_id="", deck_hash="")
    features = augment_option_features(observation, legal_set, base)
    if not features or len(features[0]) != MODEL_CONFIG["source_input_dim"]:
        raise ValueError("r7-lineage M0 feature width drifted")
    if deck_vector is not None:
        features = [list(option) + [float(value) for value in deck_vector] for option in features]
    chosen = [int(value) for value in _as_list(_as_dict(row.get("chosen_action")).get("indices"))]
    mask = _as_dict(row.get("action_mask"))
    first_target = chosen[0] if chosen else len(legal_set)
    metadata = {
        "chosen_indices": chosen,
        "min_count": _as_int(mask.get("min_count"), 0),
        "max_count": _as_int(mask.get("max_count"), len(legal_set)),
        "decision_id": str(row["decision_id"]),
        "game_id": str(row["game_id"]),
        "target_member_id": str(row["target_member_id"]),
        "target_alias": str(row["target_alias"]),
        "target_slice_kind": str(row["target_slice_kind"]),
        "reference_cell_id": row.get("reference_cell_id"),
        "target_policy_identity": str(row["target_policy_identity"]),
        "target_deck_identity": str(row["target_deck_identity"]),
        "select_context": str(row.get("phase") or "UNKNOWN"),
        "turn": _as_int(row.get("turn"), 0),
        "legal_count": len(legal_set),
        "legal_set": legal_set,
    }

    return EncodedDecision(
        features=features,
        target_index=first_target,
        legal_count=len(legal_set),
        metadata=metadata,
    )


def _model_for_rung(source_payload: Mapping[str, Any], input_dim: int) -> tuple[R4FActor, dict[str, Any]]:
    config = R4FActorConfig(
        input_dim=input_dim,
        hidden_dim=int(source_payload["hidden_dim"]),
        depth=int(source_payload["depth"]),
        embed_dim=int(source_payload["embed_dim"]),
        dropout=float(source_payload.get("dropout", 0.0)),
        intent_dim=len(source_payload["intent_vocab"]),
    )
    model = R4FActor(config)
    source_state = source_payload["model_state"]
    target_state = model.state_dict()
    exact = []
    partial = []
    for key, value in source_state.items():
        if key not in target_state:
            continue
        if target_state[key].shape == value.shape:
            target_state[key] = value.clone()
            exact.append(key)
        elif key == "encoder.0.weight" and target_state[key].shape[0] == value.shape[0]:
            if target_state[key].shape[1] < value.shape[1]:
                raise ValueError("rung model input is narrower than the r7 source")
            extended = torch.zeros_like(target_state[key])
            extended[:, : value.shape[1]] = value
            target_state[key] = extended
            partial.append(key)
    model.load_state_dict(target_state, strict=True)

    return model, {
        "source_checkpoint_sha256": sha256_file(R7_PATH),
        "exact_tensors_copied": len(exact),
        "partial_zero_extended_tensors": partial,
        "input_dim": input_dim,
    }


def _deck_vectors_by_member(config: Mapping[str, Any]) -> dict[str, list[float]]:
    return {
        str(row["member_id"]): [float(value) for value in row["deck_feature_vector"]]
        for row in _as_list(_as_dict(config["collection"]).get("targets"))
    }


def _encoded_rows(
    out_dir: Path,
    manifest: Mapping[str, Any],
    tranche: str,
    *,
    rung: str,
    deck_vectors: Mapping[str, Sequence[float]],
    shard_order: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[EncodedDecision]:
    shards = list(shard_order) if shard_order is not None else [
        row for row in _as_list(manifest.get("shards")) if row.get("tranche") == tranche
    ]
    for shard in shards:
        path = out_dir / str(shard["path"])
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                deck_vector = deck_vectors[str(row["target_member_id"])] if rung == "M1" else None
                yield encode_row(row, deck_vector=deck_vector)


def _train_model(
    model: R4FActor,
    out_dir: Path,
    manifest: Mapping[str, Any],
    *,
    rung: str,
    deck_vectors: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(MODEL_CONFIG["learning_rate"]),
        weight_decay=float(MODEL_CONFIG["weight_decay"]),
    )
    train_shards = [row for row in _as_list(manifest.get("shards")) if row.get("tranche") == "T0_TRAIN"]
    started = time.perf_counter()
    rows_seen = 0
    step_rows_seen = 0
    batches = 0
    losses: list[float] = []
    model.train()
    for epoch in range(int(MODEL_CONFIG["epochs"])):
        order = list(train_shards)
        random.Random(TRAIN_SEED + epoch).shuffle(order)
        buffer: list[r4.StepSample] = []
        for encoded in _encoded_rows(
            out_dir,
            manifest,
            "T0_TRAIN",
            rung=rung,
            deck_vectors=deck_vectors,
            shard_order=order,
        ):
            rows_seen += 1
            buffer.extend(r4.make_step_samples(encoded, weighted=False))
            if len(buffer) < int(MODEL_CONFIG["batch_size"]):
                continue
            features, option_meta, target, mask, _weights = r4.batch_steps(buffer)
            logits = model(features, option_meta).masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(MODEL_CONFIG["gradient_clip_norm"]))
            optimizer.step()
            step_rows_seen += len(buffer)
            batches += 1
            losses.append(float(loss.item()))
            buffer = []
        if buffer:
            features, option_meta, target, mask, _weights = r4.batch_steps(buffer)
            logits = model(features, option_meta).masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(MODEL_CONFIG["gradient_clip_norm"]))
            optimizer.step()
            step_rows_seen += len(buffer)
            batches += 1
            losses.append(float(loss.item()))
        print(
            f"{rung} epoch={epoch + 1}/{MODEL_CONFIG['epochs']} rows={rows_seen} "
            f"steps={step_rows_seen} batches={batches} loss={losses[-1]:.6f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    return {
        "epochs": int(MODEL_CONFIG["epochs"]),
        "decision_rows_seen_including_epochs": rows_seen,
        "step_rows_seen_including_epochs": step_rows_seen,
        "batches": batches,
        "mean_batch_loss": float(np.mean(losses)),
        "final_batch_loss": losses[-1],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _collect_step_logits(
    model: R4FActor,
    encoded_rows: Iterable[EncodedDecision],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    logits_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    buffer: list[r4.StepSample] = []

    def flush() -> None:
        if not buffer:
            return
        features, option_meta, target, mask, _weights = r4.batch_steps(buffer)
        with torch.no_grad():
            logits_rows.append(model(features, option_meta).cpu())
        mask_rows.append(mask.cpu())
        target_rows.append(target.cpu())

    for encoded in encoded_rows:
        buffer.extend(r4.make_step_samples(encoded, weighted=False))
        if len(buffer) >= batch_size:
            flush()
            buffer = []
    flush()
    max_width = max(row.shape[1] for row in logits_rows)
    logits = torch.cat([
        torch.nn.functional.pad(row, (0, max_width - row.shape[1]), value=-1e9)
        for row in logits_rows
    ])
    masks = torch.cat([
        torch.nn.functional.pad(row, (0, max_width - row.shape[1]), value=False)
        for row in mask_rows
    ])
    targets = torch.cat(target_rows)

    return logits, masks, targets


def _fit_temperature(logits: torch.Tensor, masks: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict[str, Any]]:
    lower, upper = [float(value) for value in MODEL_CONFIG["temperature_bounds"]]
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=50, line_search_fn="strong_wolfe")
    iterations = 0

    def closure() -> torch.Tensor:
        nonlocal iterations
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(lower, upper)
        scaled = (logits / temperature).masked_fill(~masks, -1e9)
        loss = torch.nn.functional.cross_entropy(scaled, targets)
        loss.backward()
        iterations += 1

        return loss

    before = float(torch.nn.functional.cross_entropy(logits.masked_fill(~masks, -1e9), targets).item())
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(lower, upper).item())
    after = float(
        torch.nn.functional.cross_entropy((logits / temperature).masked_fill(~masks, -1e9), targets).item()
    )

    return temperature, {
        "temperature": temperature,
        "token_nll_before": before,
        "token_nll_after": after,
        "optimizer_closure_calls": iterations,
        "samples": int(targets.numel()),
    }


def _option_family(option: Mapping[str, Any]) -> str:
    return str(option.get("option_type") or option.get("type") or "STOP")


def _structured_target(option: Mapping[str, Any]) -> str:
    fields = (
        "source_card_id",
        "source_card",
        "target_card_id",
        "target_card",
        "target_serial",
        "target_area",
        "attack_id",
        "value",
    )
    payload = {field: option.get(field) for field in fields if option.get(field) is not None}

    return stable_hash(payload) if payload else "none"


def _ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 15) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(bool(correct[position]) for position in members) / len(members)
        confidence = sum(confidences[position] for position in members) / len(members)
        result += len(members) / total * abs(accuracy - confidence)

    return result


def _phase_band(turn: int) -> str:
    if turn <= 3:
        return "opening_turn_0_3"
    if turn <= 8:
        return "midgame_turn_4_8"

    return "late_turn_9_plus"


def _legal_band(count: int) -> str:
    if count <= 2:
        return "1_2"
    if count <= 5:
        return "3_5"
    if count <= 10:
        return "6_10"

    return "11_plus"


@dataclass
class DecisionMetric:
    decision_id: str
    game_id: str
    target_alias: str
    target_member_id: str
    target_slice_kind: str
    reference_cell_id: str | None
    target_policy_identity: str
    target_deck_identity: str
    select_context: str
    phase_band: str
    legal_band: str
    action_family: str
    complete_action_nll: float
    complete_action_top1: bool
    teacher_forced_complete_action_top3: bool
    token_brier_mean: float
    action_family_correct: bool
    structured_target_correct: bool
    tokens: int


def _evaluate_model(
    model: R4FActor,
    encoded_rows: Iterable[EncodedDecision],
    *,
    temperature: float,
    batch_size: int,
) -> tuple[dict[str, Any], list[DecisionMetric]]:
    model.eval()
    step_buffer: list[r4.StepSample] = []
    decision_metadata: dict[str, dict[str, Any]] = {}
    decision_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_confidence: list[float] = []
    token_correct: list[bool] = []
    started = time.perf_counter()

    def flush() -> None:
        if not step_buffer:
            return
        features, option_meta, target, mask, _weights = r4.batch_steps(step_buffer)
        with torch.no_grad():
            logits = (model(features, option_meta) / temperature).masked_fill(~mask, -1e9)
            probabilities = torch.softmax(logits, dim=1)
        for index, step in enumerate(step_buffer):
            target_index = int(target[index])
            probability = float(probabilities[index, target_index])
            prediction = int(logits[index].argmax().item())
            top_k = min(3, int(mask[index].sum().item()))
            top3 = {int(value) for value in torch.topk(logits[index], k=max(1, top_k)).indices.tolist()}
            one_hot = torch.zeros_like(probabilities[index])
            one_hot[target_index] = 1.0
            brier = float(((probabilities[index] - one_hot) ** 2).sum().item())
            confidence = float(probabilities[index].max().item())
            is_correct = prediction == target_index
            token_confidence.append(confidence)
            token_correct.append(is_correct)
            decision_id = str(step.metadata["decision_id"])
            decision_steps[decision_id].append({
                "step_index": int(step.step_index),
                "target": int(step.target),
                "prediction": prediction if prediction < step.legal_count else step.legal_count,
                "nll": -math.log(max(probability, 1e-12)),
                "top1": is_correct,
                "top3": target_index in top3,
                "brier": brier,
            })

    for encoded in encoded_rows:
        metadata = dict(encoded.metadata)
        decision_metadata[str(metadata["decision_id"])] = metadata
        step_buffer.extend(r4.make_step_samples(encoded, weighted=False))
        if len(step_buffer) >= batch_size:
            flush()
            step_buffer = []
    flush()
    decision_metrics: list[DecisionMetric] = []
    for decision_id, steps in decision_steps.items():
        metadata = decision_metadata[decision_id]
        steps.sort(key=lambda row: row["step_index"])
        legal_set = _as_list(metadata["legal_set"])
        first_target = steps[0]["target"]
        first_prediction = steps[0]["prediction"]
        target_option = _as_dict(legal_set[first_target]) if first_target < len(legal_set) else {}
        predicted_option = _as_dict(legal_set[first_prediction]) if first_prediction < len(legal_set) else {}
        decision_metrics.append(DecisionMetric(
            decision_id=decision_id,
            game_id=str(metadata["game_id"]),
            target_alias=str(metadata["target_alias"]),
            target_member_id=str(metadata["target_member_id"]),
            target_slice_kind=str(metadata["target_slice_kind"]),
            reference_cell_id=metadata.get("reference_cell_id"),
            target_policy_identity=str(metadata["target_policy_identity"]),
            target_deck_identity=str(metadata["target_deck_identity"]),
            select_context=str(metadata["select_context"]),
            phase_band=_phase_band(int(metadata["turn"])),
            legal_band=_legal_band(int(metadata["legal_count"])),
            action_family=_option_family(target_option),
            complete_action_nll=sum(row["nll"] for row in steps),
            complete_action_top1=all(row["top1"] for row in steps),
            teacher_forced_complete_action_top3=all(row["top3"] for row in steps),
            token_brier_mean=float(np.mean([row["brier"] for row in steps])),
            action_family_correct=_option_family(target_option) == _option_family(predicted_option),
            structured_target_correct=_structured_target(target_option) == _structured_target(predicted_option),
            tokens=len(steps),
        ))
    elapsed = time.perf_counter() - started

    return {
        "decision_rows": len(decision_metrics),
        "token_rows": sum(row.tokens for row in decision_metrics),
        "masked_complete_action_log_loss": float(np.mean([row.complete_action_nll for row in decision_metrics])),
        "complete_action_top1_accuracy": float(np.mean([row.complete_action_top1 for row in decision_metrics])),
        "teacher_forced_complete_action_top3_coverage": float(
            np.mean([row.teacher_forced_complete_action_top3 for row in decision_metrics])
        ),
        "token_brier_mean": float(np.mean([row.token_brier_mean for row in decision_metrics])),
        "action_family_accuracy": float(np.mean([row.action_family_correct for row in decision_metrics])),
        "structured_target_accuracy": float(np.mean([row.structured_target_correct for row in decision_metrics])),
        "token_top_label_ece_15bin": _ece(token_confidence, token_correct),
        "elapsed_seconds": round(elapsed, 3),
        "decision_rows_per_second": len(decision_metrics) / max(elapsed, 1e-9),
    }, decision_metrics


def _slice_metrics(rows: Sequence[DecisionMetric], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[DecisionMetric]] = defaultdict(list)
    for row in rows:
        value = getattr(row, field)
        grouped[str(value if value is not None else "not_applicable")].append(row)
    result = []
    for value, members in sorted(grouped.items()):
        result.append({
            "value": value,
            "decision_rows": len(members),
            "masked_complete_action_log_loss": float(np.mean([row.complete_action_nll for row in members])),
            "complete_action_top1_accuracy": float(np.mean([row.complete_action_top1 for row in members])),
            "action_family_accuracy": float(np.mean([row.action_family_correct for row in members])),
        })

    return result


def _paired_bootstrap(
    m0_rows: Sequence[DecisionMetric],
    m1_rows: Sequence[DecisionMetric],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    m0_by_decision = {row.decision_id: row for row in m0_rows}
    m1_by_decision = {row.decision_id: row for row in m1_rows}
    if set(m0_by_decision) != set(m1_by_decision):
        raise ValueError("M0/M1 test decision identities differ")
    by_game: dict[str, list[float]] = defaultdict(list)
    for decision_id, m0 in m0_by_decision.items():
        by_game[m0.game_id].append(m0.complete_action_nll - m1_by_decision[decision_id].complete_action_nll)
    games = sorted(by_game)
    game_deltas = np.asarray([np.mean(by_game[game_id]) for game_id in games], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(games), size=len(games))
        samples[index] = float(game_deltas[chosen].mean())
    low, high = np.quantile(samples, [0.025, 0.975])

    return {
        "estimand": "M0_minus_M1_masked_complete_action_log_loss_game_equal_weighted",
        "games": len(games),
        "resamples": resamples,
        "seed": seed,
        "mean_delta": float(game_deltas.mean()),
        "confidence_interval_95": [float(low), float(high)],
        "M1_beats_M0": bool(low > 0.0),
    }


def _paired_slices(m0_rows: Sequence[DecisionMetric], m1_rows: Sequence[DecisionMetric], field: str) -> list[dict[str, Any]]:
    m1_by_id = {row.decision_id: row for row in m1_rows}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in m0_rows:
        value = getattr(row, field)
        grouped[str(value if value is not None else "not_applicable")].append(
            row.complete_action_nll - m1_by_id[row.decision_id].complete_action_nll
        )

    return [
        {
            "value": value,
            "decision_rows": len(deltas),
            "M0_minus_M1_complete_action_nll": float(np.mean(deltas)),
            "M1_nonregressed": float(np.mean(deltas)) >= 0.0,
        }
        for value, deltas in sorted(grouped.items())
    ]


def run_rung(
    out_dir: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    rung: str,
) -> tuple[dict[str, Any], list[DecisionMetric]]:
    result_path = out_dir / f"{rung}_RESULT.json"
    metric_path = out_dir / f"{rung}_TEST_DECISION_METRICS.jsonl.gz"
    if result_path.exists() or metric_path.exists():
        raise FileExistsError(f"refusing to rerun completed/partial {rung}; inspect before resume")
    deck_vectors = _deck_vectors_by_member(config)
    source_payload = torch.load(R7_PATH, map_location="cpu", weights_only=False)
    input_dim = int(source_payload["input_dim"])
    if rung == "M1":
        input_dim += len(next(iter(deck_vectors.values())))
    model, initialization = _model_for_rung(source_payload, input_dim)
    train = _train_model(model, out_dir, manifest, rung=rung, deck_vectors=deck_vectors)
    calibration_logits, calibration_masks, calibration_targets = _collect_step_logits(
        model,
        _encoded_rows(
            out_dir,
            manifest,
            "T0_CALIBRATION",
            rung=rung,
            deck_vectors=deck_vectors,
        ),
        batch_size=int(MODEL_CONFIG["batch_size"]),
    )
    temperature, calibration = _fit_temperature(calibration_logits, calibration_masks, calibration_targets)
    aggregate, decision_metrics = _evaluate_model(
        model,
        _encoded_rows(out_dir, manifest, "T1_TEST", rung=rung, deck_vectors=deck_vectors),
        temperature=temperature,
        batch_size=int(MODEL_CONFIG["batch_size"]),
    )
    checkpoint_path = out_dir / f"{rung}_MODEL.pt"
    torch.save({
        "schema_version": RESULT_SCHEMA_VERSION,
        "rung": rung,
        "contract_sha256": config["contract_sha256"],
        "input_dim": input_dim,
        "model_state": model.state_dict(),
        "temperature": temperature,
        "initialization": initialization,
        "train": train,
    }, checkpoint_path)
    with metric_path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0) as gzip_handle:
            for metric in decision_metrics:
                line = json.dumps(asdict(metric), sort_keys=True) + "\n"
                gzip_handle.write(line.encode("utf-8"))
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "COMPLETE",
        "created_at_utc": utc_now(),
        "rung": rung,
        "inputs": (
            "public-state projection plus validated legal-action mask"
            if rung == "M0"
            else "public-state projection plus validated legal-action mask and frozen deck features"
        ),
        "contract_sha256": config["contract_sha256"],
        "corpus_manifest_sha256": sha256_file(out_dir / "CORPUS_MANIFEST.json"),
        "initialization": initialization,
        "train": train,
        "calibration": calibration,
        "test": {
            "aggregate": aggregate,
            "slices": {
                field: _slice_metrics(decision_metrics, field)
                for field in (
                    "target_alias",
                    "target_policy_identity",
                    "reference_cell_id",
                    "target_deck_identity",
                    "select_context",
                    "action_family",
                    "phase_band",
                    "legal_band",
                )
            },
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "decision_metrics": {"path": str(metric_path), "sha256": sha256_file(metric_path)},
        "paid_jobs": 0,
        "protected_runtime_wiring": False,
    }
    write_json(result_path, result)

    return result, decision_metrics


def compare_rungs(
    out_dir: Path,
    config: Mapping[str, Any],
    m0: Mapping[str, Any],
    m1: Mapping[str, Any],
    m0_rows: Sequence[DecisionMetric],
    m1_rows: Sequence[DecisionMetric],
) -> dict[str, Any]:
    paired = _paired_bootstrap(
        m0_rows,
        m1_rows,
        resamples=int(MODEL_CONFIG["bootstrap_resamples"]),
        seed=BOOTSTRAP_SEED,
    )
    slice_fields = (
        "target_alias",
        "target_policy_identity",
        "reference_cell_id",
        "target_deck_identity",
        "select_context",
        "action_family",
        "phase_band",
        "legal_band",
    )
    paired_slices = {field: _paired_slices(m0_rows, m1_rows, field) for field in slice_fields}
    reference_rows = [row for row in paired_slices["reference_cell_id"] if row["value"] != "not_applicable"]
    policy_rows = paired_slices["target_policy_identity"]
    m0_aggregate = m0["test"]["aggregate"]
    m1_aggregate = m1["test"]["aggregate"]
    throughput_delta = (
        float(m1_aggregate["decision_rows_per_second"])
        - float(m0_aggregate["decision_rows_per_second"])
    )
    negative_references = [row for row in reference_rows if not row["M1_nonregressed"]]
    negative_policies = [row for row in policy_rows if not row["M1_nonregressed"]]
    decision = "proposal-ranker only" if paired["M1_beats_M0"] else "quarantine"
    feature_audit = {
        "Feature": "registered interpretable opponent deck feature vector",
        "Purpose": "condition the legal opponent-action distribution on deck composition",
        "Baseline": "M0 public-state projection plus legal-action mask with identity channels zeroed",
        "Benefit": paired,
        "Cost": {
            "additional_input_dimensions": len(_as_dict(config["features"])["deck_feature_schema"]["names"]),
            "model_parameter_delta": (
                int(m1["initialization"]["input_dim"] - m0["initialization"]["input_dim"])
                * int(MODEL_CONFIG["hidden_dim"])
            ),
            "paid_cost_usd": 0.0,
        },
        "Damage": {
            "negative_reference_slices": negative_references,
            "negative_policy_slices": negative_policies,
            "worst_reference_delta": min(
                (row["M0_minus_M1_complete_action_nll"] for row in reference_rows),
                default=None,
            ),
        },
        "Uniqueness": "M0 zeros checkpoint/policy/deck identity channels; M1 adds only frozen interpretable deck composition",
        "Precision": {
            "nonregressed_reference_slice_fraction": (
                sum(row["M1_nonregressed"] for row in reference_rows) / len(reference_rows)
                if reference_rows else None
            ),
        },
        "Recall": {
            "test_decision_fraction_in_nonregressed_reference_slices": (
                sum(row["decision_rows"] for row in reference_rows if row["M1_nonregressed"])
                / max(1, sum(row["decision_rows"] for row in reference_rows))
            ),
        },
        "Mean true objective delta": paired["mean_delta"],
        "Throughput impact": {
            "M0_decisions_per_second": m0_aggregate["decision_rows_per_second"],
            "M1_decisions_per_second": m1_aggregate["decision_rows_per_second"],
            "M1_minus_M0_decisions_per_second": throughput_delta,
        },
        "Marginal value/sec": (
            paired["mean_delta"] / max(float(m1_aggregate["elapsed_seconds"]), 1e-9)
        ),
        "Failure clustering": {
            "reference_cells": [row["value"] for row in negative_references],
            "policy_identities": [row["value"] for row in negative_policies],
        },
        "Decision": decision,
        "Next action": (
            "retain M1 for the frozen M2/M3 proposal-ranker comparison"
            if paired["M1_beats_M0"]
            else "drop M1 without retuning; base M2/M3 on the selected M0 path"
        ),
    }
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "status": "M0_M1_COMPLETE",
        "created_at_utc": utc_now(),
        "contract_sha256": config["contract_sha256"],
        "primary_gate": paired,
        "M0": m0_aggregate,
        "M1": m1_aggregate,
        "paired_slices": paired_slices,
        "deck_feature_contribution_verdict": (
            "EARNED_PROPOSAL_RANKER_ONLY" if paired["M1_beats_M0"] else "FAILED_DROP_WITHOUT_RETUNING"
        ),
        "feature_contribution_audit": feature_audit,
        "paid_jobs": 0,
    }
    write_json(out_dir / "M0_M1_COMPARISON.json", comparison)
    write_json(out_dir / "FEATURE_CONTRIBUTION_AUDIT.json", feature_audit)

    return comparison


def _load_decision_metrics(path: Path) -> list[DecisionMetric]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(DecisionMetric(**json.loads(line)))

    return rows


def run_all(out_dir: Path, *, resume: bool) -> dict[str, Any]:
    config = freeze_config(out_dir)
    verify_config(config)
    corpus = collect_corpus(out_dir, config, resume=resume)
    leakage = build_leakage_receipt(out_dir, config, corpus)
    if not leakage.get("passed"):
        raise RuntimeError("Request A leakage gate failed")
    comparison_path = out_dir / "M0_M1_COMPARISON.json"
    if comparison_path.exists():
        return _read_json(comparison_path)
    if (out_dir / "M0_RESULT.json").exists():
        m0 = _read_json(out_dir / "M0_RESULT.json")
        m0_rows = _load_decision_metrics(out_dir / "M0_TEST_DECISION_METRICS.jsonl.gz")
    else:
        m0, m0_rows = run_rung(out_dir, config, corpus, rung="M0")
    if (out_dir / "M1_RESULT.json").exists():
        m1 = _read_json(out_dir / "M1_RESULT.json")
        m1_rows = _load_decision_metrics(out_dir / "M1_TEST_DECISION_METRICS.jsonl.gz")
    else:
        m1, m1_rows = run_rung(out_dir, config, corpus, rung="M1")

    return compare_rungs(out_dir, config, m0, m1, m0_rows, m1_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / ".research" / "post_e1_request_a_m0_m1_v1"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    result = run_all(Path(args.out_dir), resume=bool(args.resume))
    print(json.dumps({
        "status": result["status"],
        "deck_feature_contribution_verdict": result["deck_feature_contribution_verdict"],
        "primary_gate": result["primary_gate"],
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
