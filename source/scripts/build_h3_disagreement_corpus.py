"""Run a zero-paid H3 collector capability canary and fail closed on gaps.

This script deliberately does not train. It reuses the proven prospective
actor-visible game collector, persists only quarantined source-probe rows, and
admits no H3 distillation row unless the collector proves complete canonical
compound-action grouping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_post_e1_request_a as request_a  # noqa: E402
import run_prospective_trace as prospective  # noqa: E402
from deepmew.artifacts.writers import write_json  # noqa: E402
from deepmew.deck import read_deck  # noqa: E402
from deepmew.hybrid_distillation.corpus import audit_collector_capability  # noqa: E402
from deepmew.replay.information_boundary import InformationBoundary, InformationMode  # noqa: E402
from deepmew.replay.prospective_trace import (  # noqa: E402
    ActorMetadata,
    assert_actor_visible_trace_row_safe,
    atomic_write_gzip_jsonl,
    unavailable_rating_snapshot,
    utc_now,
)


SCHEMA_VERSION = "deepmew.h3.disagreement_corpus_build.v1"
FROZEN_CONFIG = ROOT / ".research" / "post_e1_request_a_m0_m1_v1" / "FROZEN_CONFIG.json"
FORBIDDEN_SEEDS = frozenset({40009, 50021, 60013, 613249997, 613250019, 613250041})
DEFAULT_SEED = 720_310_000
MAX_CANARY_GAMES = 2
MAX_STEPS = 5_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parent_rows(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    collection = dict(config["collection"])
    anchor = dict(collection["anchor"])
    e1 = next(dict(row) for row in collection["targets"] if row.get("alias") == "E1")
    parents = {"D1": anchor, "E1": e1}
    for alias, parent in parents.items():
        if parent.get("alias") != alias:
            raise ValueError(f"{alias} frozen alias drifted")
        checkpoint = dict(parent["policy"]["checkpoint"])
        path = Path(checkpoint["path"])
        if not path.is_file() or _sha256_file(path) != checkpoint["sha256"]:
            raise ValueError(f"{alias} checkpoint identity drifted")
        deck = dict(parent["deck"])
        deck_path = Path(deck["path"])
        if not deck_path.is_file() or _sha256_file(deck_path) != deck["file_sha256"]:
            raise ValueError(f"{alias} deck identity drifted")
    if parents["D1"]["deck"] != parents["E1"]["deck"]:
        raise ValueError("H3 parents no longer share one frozen deck")

    return parents


def _actor(parent: Mapping[str, Any], seat: int) -> ActorMetadata:
    metadata = request_a._actor_metadata(
        str(parent["member_id"]),
        parent["policy"],
        parent["deck"],
        alias=str(parent["alias"]),
    )

    return ActorMetadata(seat=seat, policy=metadata.policy, deck=metadata.deck)


def _play_canary_game(
    *,
    parents: Mapping[str, Mapping[str, Any]],
    seed: int,
    game_index: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    aliases = ("D1", "E1") if game_index % 2 == 0 else ("E1", "D1")
    actors = [_actor(parents[alias], seat) for seat, alias in enumerate(aliases)]
    decks = {
        seat: read_deck(parents[alias]["deck"]["path"])
        for seat, alias in enumerate(aliases)
    }
    agents = {
        seat: request_a._agent(
            parents[alias]["policy"],
            decks[seat],
            str(parents[alias]["deck"]["normalized_hash"]),
        )
        for seat, alias in enumerate(aliases)
    }
    tranche = {
        "tranche_id": "S0_CANARY",
        "tranche_sequence": 0,
        "declared_at_utc": utc_now(),
        "frozen_snapshot": {"rating_snapshot": unavailable_rating_snapshot()},
    }

    return prospective._play_game(
        game_id=f"h3-disagreement:S0:g{game_index:03d}",
        run_id="h3-disagreement-corpus-preflight-v1",
        tranche=tranche,
        seed=seed,
        actors=actors,
        agents=agents,
        decks=decks,
        league_identity="h3-d1-e1-same-deck-preflight",
        curriculum_round=None,
        boundary=InformationBoundary(mode=InformationMode.ACTOR_VISIBLE),
        max_steps=MAX_STEPS,
        game_index=game_index,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.games < 1 or args.games > MAX_CANARY_GAMES:
        raise ValueError(f"games must be within [1, {MAX_CANARY_GAMES}]")
    seeds = tuple(args.seed + index for index in range(args.games))
    overlap = sorted(FORBIDDEN_SEEDS.intersection(seeds))
    if overlap:
        raise ValueError(f"H3 canary requested frozen evaluation seeds: {overlap}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source_probe"
    source_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(FROZEN_CONFIG)
    request_a.verify_config(config)
    parents = _parent_rows(config)
    all_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    safety = {"completed": 0, "crashes": 0, "timeouts": 0, "illegal": 0}
    for game_index, seed in enumerate(seeds):
        rows, game_safety = _play_canary_game(
            parents=parents,
            seed=seed,
            game_index=game_index,
        )
        for key in safety:
            safety[key] += int(game_safety[key])
        if any(game_safety[key] for key in ("crashes", "timeouts", "illegal")):
            raise RuntimeError(f"H3 collector safety hard stop: {game_safety}")
        for row in rows:
            assert_actor_visible_trace_row_safe(row)
            request_a._assert_public_source_projection(row)
            if request_a._row_has_forbidden_nested_key(row):
                raise ValueError(f"forbidden oracle/search field in source row: {row['decision_id']}")
        shard_path = source_dir / f"game_{game_index:03d}.jsonl.gz"
        atomic_write_gzip_jsonl(shard_path, rows)
        shards.append({
            "path": str(shard_path.relative_to(out_dir)).replace("\\", "/"),
            "bytes": shard_path.stat().st_size,
            "sha256": _sha256_file(shard_path),
            "game_id": rows[0]["game_id"],
            "seed": seed,
            "rows": len(rows),
        })
        all_rows.extend(rows)
    if safety != {"completed": args.games, "crashes": 0, "timeouts": 0, "illegal": 0}:
        raise RuntimeError(f"H3 canary did not close safety-clean: {safety}")
    capability = audit_collector_capability(all_rows)
    alias_counts = Counter(str(row["actor_policy"]["alias"]) for row in all_rows)
    seat_counts = Counter(str(row["actor_seat"]) for row in all_rows)
    context_counts = Counter(str(row["phase"]) for row in all_rows)
    leakage_receipt = {
        "schema_version": "deepmew.h3.source_probe_leakage.v1",
        "status": "PASS_QUARANTINED_ACTOR_VISIBLE_SOURCE",
        "created_at_utc": utc_now(),
        "information_mode": InformationMode.ACTOR_VISIBLE.value,
        "rows_scanned": len(all_rows),
        "actor_visible_rows": len(all_rows),
        "public_projections_recomputed": len(all_rows),
        "action_contracts_validated": len(all_rows),
        "opponent_hidden_zone_fields": 0,
        "forbidden_oracle_or_search_fields": 0,
        "source_provenance_identity_fields_present": True,
        "admitted_student_rows": 0,
        "identity_fields_in_admitted_student_inputs": 0,
        "branch_q_used": False,
        "safety": safety,
    }
    write_json(out_dir / "H3_SOURCE_PROBE_LEAKAGE_RECEIPT.json", leakage_receipt)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "READY_FOR_DERIVATION"
            if capability["passed"]
            else "BLOCKED_NO_DISTILLATION_ROWS_ADMITTED"
        ),
        "created_at_utc": utc_now(),
        "source": "fresh prospective actor-visible games",
        "source_disposition": "QUARANTINED_CAPABILITY_PROBE",
        "blocker_scope": (
            "existing prospective collector persistence schema only; "
            "no claim about CABT or vendor capability"
        ),
        "games_requested": args.games,
        "seeds": list(seeds),
        "forbidden_evaluation_seeds": sorted(FORBIDDEN_SEEDS),
        "parent_aliases": ["D1", "E1"],
        "parent_checkpoint_sha256": {
            alias: parents[alias]["policy"]["checkpoint"]["sha256"]
            for alias in ("D1", "E1")
        },
        "shared_deck_file_sha256": parents["D1"]["deck"]["file_sha256"],
        "information_mode": InformationMode.ACTOR_VISIBLE.value,
        "safety": safety,
        "source_rows": len(all_rows),
        "source_rows_by_actor_alias": dict(sorted(alias_counts.items())),
        "source_rows_by_actor_seat": dict(sorted(seat_counts.items())),
        "source_rows_by_select_context": dict(sorted(context_counts.items())),
        "admitted_distillation_rows": len(all_rows) if capability["passed"] else 0,
        "collection_stop_reason": (
            None
            if capability["passed"]
            else "first fresh canary proved a collector schema gap; more games cannot supply absent compound-action semantics"
        ),
        "branch_q_used": False,
        "training_started": False,
        "collector_capability": capability,
        "shards": shards,
    }
    write_json(out_dir / "H3_CORPUS_BUILD_RECEIPT.json", receipt)

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({
        "status": receipt["status"],
        "source_rows": receipt["source_rows"],
        "admitted_distillation_rows": receipt["admitted_distillation_rows"],
        "safety": receipt["safety"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
