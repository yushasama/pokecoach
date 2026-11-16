"""K2B real-meta arena builder: validate K3 decks, then evaluate DeepMew vs them.

Phase 1: validate every canonical K3 Deck Zoo decklist through the engine
(`battle_start`) and emit a deck-validation report + arena deck registry.
Phase 2/4: run fixed-policy matchups of our deck (`deck.csv`) under v2.4 and
first-legal against the K3 bucket opponent decks (top_meta /
anti_overfit_holdout / unknown_mixed), and against a ReplayPolicy opponent on
top_meta. Eval-only: root `main.py` / `deck.csv` / `cg/` / default `v2.4` are
never modified, and nothing here is imported by the submission runtime.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import main as submission_main
from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_json, write_markdown
from deepmew.deck_zoo.io import find_latest_meta_kg  # noqa: F401  (kept for symmetry/discoverability)
from deepmew.meta_kg.io import write_csv
from deepmew.real_meta_arena.decklists import (
    ZooDeck,
    load_zoo_decklists,
    validate_decklist,
    validation_to_row,
)
from deepmew.real_meta_arena.replay_policy import ReplayAgent, build_replay_policy_table
from deepmew.real_meta_arena.report import render_report
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup

REAL_META_ARENA_KIND = "real_meta_arena"
ARTIFACT_VERSION = "deepmew.real_meta_arena.v1"
OPPONENT_SETS_FILENAME = "arena_opponent_sets.csv"

OUTPUT_FILES = {
    "deck_validation": "real_meta_deck_validation.csv",
    "deck_registry": "arena_deck_registry.csv",
    "results": "real_meta_arena_results.csv",
    "summary": "real_meta_arena_summary.csv",
    "replay_table": "replay_policy_table.json",
    "report": "real_meta_arena_report.md",
}

VALIDATION_FIELDS = [
    "deck_hash", "archetype", "variant_label", "split", "card_count", "is_standard_size",
    "known_card_ids", "ace_spec_ok", "has_basic", "engine_accepted", "valid", "errors", "warnings",
]
REGISTRY_FIELDS = ["deck_hash", "archetype", "variant_label", "split", "valid", "engine_accepted", "card_count"]
RESULTS_FIELDS = [
    "bucket", "our_label", "opp_label", "opp_kind", "opp_deck_hash", "opp_archetype",
    "games", "completed", "our_wins", "opp_wins", "draws", "crashes", "timeouts", "avg_turns", "elapsed_seconds",
]
SUMMARY_FIELDS = [
    "bucket", "our_label", "opp_kind", "decks", "games", "completed", "our_wins", "opp_wins",
    "draws", "our_win_rate", "crashes", "timeouts", "avg_turns",
]
BUCKETS = ["top_meta", "anti_overfit_holdout", "unknown_mixed"]
OUR_POLICIES = ["v2.4", "first-legal"]


@dataclass
class RealMetaArenaConfig:
    games_per_matchup: int = 6
    max_decks_per_bucket: int = 6
    buckets: tuple[str, ...] = tuple(BUCKETS)
    our_policies: tuple[str, ...] = tuple(OUR_POLICIES)
    opponent_policy: str = "first-legal"
    include_replay: bool = True
    replay_bucket: str = "top_meta"
    seed: int = 20260623
    max_steps: int = 5000


@dataclass
class RealMetaArenaResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]


def _load_opponent_sets(deck_zoo_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = deck_zoo_dir / OPPONENT_SETS_FILENAME
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_set[row["set_name"]].append(row)
    for rows in by_set.values():
        rows.sort(key=lambda r: int(r.get("rank_in_set") or 0))
    return dict(by_set)


def build_real_meta_arena(
    deck_zoo_dir: str | Path,
    imitation_dir: str | Path,
    out_root: str | Path = ".research/runs",
    config: RealMetaArenaConfig | None = None,
    *,
    command: str = "",
    run_id: str | None = None,
    suffix: str | None = None,
) -> RealMetaArenaResult:
    cfg = config or RealMetaArenaConfig()
    deck_zoo_dir = Path(deck_zoo_dir)

    # Phase 1: validate every canonical decklist.
    decks = load_zoo_decklists(deck_zoo_dir)
    decks_by_hash = {d.deck_hash: d for d in decks}
    validations = [validate_decklist(d) for d in decks]
    valid_hashes = {v.deck_hash for v in validations if v.valid}
    validation_rows = [validation_to_row(v) for v in validations]
    validation_rows.sort(key=lambda r: (not r["valid"], r["archetype"], r["deck_hash"]))
    registry_rows = [
        {k: row[k] for k in REGISTRY_FIELDS} for row in validation_rows
    ]

    # ReplayPolicy table (built from K4 train examples).
    replay_table = build_replay_policy_table(imitation_dir, split="train")

    # Phase 2/4: bucket eval matrix.
    opponent_sets = _load_opponent_sets(deck_zoo_dir)
    our_deck = submission_main.read_deck_csv()
    matchups: list[Any] = []
    matchup_meta: list[tuple[str, str]] = []  # (bucket-with-kind, opp_kind)

    seed_counter = cfg.seed
    for bucket in cfg.buckets:
        bucket_decks = _select_bucket_decks(opponent_sets.get(bucket, []), valid_hashes, cfg.max_decks_per_bucket)
        for our_policy in cfg.our_policies:
            our_agent = SpecAgent.from_str(our_policy)
            for entry in bucket_decks:
                opp_deck = decks_by_hash[entry["deck_hash"]]
                opp_agent = SpecAgent.from_str(cfg.opponent_policy)
                seed_counter += 1
                summary = run_real_meta_matchup(
                    our_agent, opp_agent, our_deck, list(opp_deck.card_ids),
                    games=cfg.games_per_matchup, bucket=bucket,
                    opp_deck_hash=opp_deck.deck_hash, opp_archetype=opp_deck.archetype,
                    seed=seed_counter, max_steps=cfg.max_steps,
                )
                matchups.append(summary)
                matchup_meta.append((bucket, f"spec:{cfg.opponent_policy}"))

    # ReplayPolicy opponent on the replay bucket.
    if cfg.include_replay:
        replay_decks = _select_bucket_decks(
            opponent_sets.get(cfg.replay_bucket, []), valid_hashes, cfg.max_decks_per_bucket
        )
        for our_policy in cfg.our_policies:
            our_agent = SpecAgent.from_str(our_policy)
            for entry in replay_decks:
                opp_deck = decks_by_hash[entry["deck_hash"]]
                opp_agent = ReplayAgent(archetype=opp_deck.archetype, table=replay_table)
                seed_counter += 1
                summary = run_real_meta_matchup(
                    our_agent, opp_agent, our_deck, list(opp_deck.card_ids),
                    games=cfg.games_per_matchup, bucket=f"{cfg.replay_bucket}(replay)",
                    opp_deck_hash=opp_deck.deck_hash, opp_archetype=opp_deck.archetype,
                    seed=seed_counter, max_steps=cfg.max_steps,
                )
                matchups.append(summary)
                matchup_meta.append((f"{cfg.replay_bucket}(replay)", "replay"))

    results_rows = [_matchup_row(s, meta[1]) for s, meta in zip(matchups, matchup_meta)]
    summary_rows = _aggregate(matchups, matchup_meta)

    run_dir = create_run_dir(kind=REAL_META_ARENA_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    output_paths: dict[str, str] = {}
    output_paths["deck_validation"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["deck_validation"]), VALIDATION_FIELDS, validation_rows))
    output_paths["deck_registry"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["deck_registry"]), REGISTRY_FIELDS, registry_rows))
    output_paths["results"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["results"]), RESULTS_FIELDS, results_rows))
    output_paths["summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["summary"]), SUMMARY_FIELDS, summary_rows))
    output_paths["replay_table"] = str(write_json(run_dir.output_path(OUTPUT_FILES["replay_table"]), replay_table.to_dict()))

    counts = _build_counts(validations, valid_hashes, matchups, summary_rows)
    report_ctx = {
        "built_at_utc": _utc(),
        "artifact_version": ARTIFACT_VERSION,
        "deck_zoo_dir": str(deck_zoo_dir),
        "imitation_dir": str(imitation_dir),
        "output_dir": str(run_dir.path),
        "counts": counts,
        "config": cfg,
        "summary_rows": summary_rows,
        "validation_rows": validation_rows,
        "output_paths": output_paths,
    }
    output_paths["report"] = str(write_markdown(run_dir.output_path(OUTPUT_FILES["report"]), render_report(report_ctx)))

    manifest = build_manifest(
        run_id=run_dir.run_id, kind=REAL_META_ARENA_KIND, command=command,
        inputs={"deck_zoo_dir": str(deck_zoo_dir), "imitation_dir": str(imitation_dir),
                "games_per_matchup": cfg.games_per_matchup, "max_decks_per_bucket": cfg.max_decks_per_bucket},
        outputs=output_paths,
        summary={"artifact_version": ARTIFACT_VERSION, **counts},
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return RealMetaArenaResult(run_dir.path, manifest_path, output_paths, counts)


def _select_bucket_decks(rows: list[dict[str, Any]], valid_hashes: set[str], cap: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["deck_hash"] in valid_hashes:
            selected.append(row)
        if len(selected) >= cap:
            break
    return selected


def _matchup_row(s: Any, opp_kind: str) -> dict[str, Any]:
    return {
        "bucket": s.bucket, "our_label": s.our_label, "opp_label": s.opp_label, "opp_kind": opp_kind,
        "opp_deck_hash": s.opp_deck_hash, "opp_archetype": s.opp_archetype,
        "games": s.games, "completed": s.completed, "our_wins": s.our_wins, "opp_wins": s.opp_wins,
        "draws": s.draws, "crashes": s.crashes, "timeouts": s.timeouts,
        "avg_turns": round(s.avg_turns, 2), "elapsed_seconds": round(s.elapsed_seconds, 2),
    }


def _aggregate(matchups: list[Any], matchup_meta: list[tuple[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for s, (bucket, opp_kind) in zip(matchups, matchup_meta):
        groups[(s.bucket, s.our_label, opp_kind)].append(s)
    rows: list[dict[str, Any]] = []
    for (bucket, our_label, opp_kind), group in groups.items():
        completed = sum(s.completed for s in group)
        our_wins = sum(s.our_wins for s in group)
        opp_wins = sum(s.opp_wins for s in group)
        draws = sum(s.draws for s in group)
        decided = our_wins + opp_wins
        turns = sum(s.avg_turns * s.completed for s in group)
        rows.append(
            {
                "bucket": bucket, "our_label": our_label, "opp_kind": opp_kind,
                "decks": len(group), "games": sum(s.games for s in group), "completed": completed,
                "our_wins": our_wins, "opp_wins": opp_wins, "draws": draws,
                "our_win_rate": round(our_wins / decided, 4) if decided else 0.0,
                "crashes": sum(s.crashes for s in group), "timeouts": sum(s.timeouts for s in group),
                "avg_turns": round(turns / completed, 2) if completed else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["bucket"], r["our_label"], r["opp_kind"]))
    return rows


def _build_counts(validations, valid_hashes, matchups, summary_rows) -> dict[str, Any]:
    return {
        "decklists_total": len(validations),
        "decklists_valid": len(valid_hashes),
        "decklists_invalid": len(validations) - len(valid_hashes),
        "matchup_count": len(matchups),
        "games_total": sum(s.games for s in matchups),
        "games_completed": sum(s.completed for s in matchups),
        "crashes": sum(s.crashes for s in matchups),
        "timeouts": sum(s.timeouts for s in matchups),
        "summary_rows": len(summary_rows),
    }


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _find_latest(runs_root: str, marker: str) -> Path:
    root = Path(runs_root)
    runs = [c for c in sorted(root.glob("*/*")) if (c / marker).exists()]
    if not runs:
        raise FileNotFoundError(f"no run with {marker} under {root}")
    return runs[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the K2B real-meta arena (validate K3 decks + eval DeepMew).")
    parser.add_argument("--deck-zoo", default=None, help="K3 deck zoo run dir (default: latest)")
    parser.add_argument("--imitation", default=None, help="K4 imitation run dir (default: latest)")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--games", type=int, default=6, help="games per matchup")
    parser.add_argument("--max-decks-per-bucket", type=int, default=6)
    parser.add_argument("--no-replay", action="store_true", help="skip the ReplayPolicy opponent")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    deck_zoo_dir = args.deck_zoo or _find_latest(".research/runs/deck_zoo", "deck_zoo_decklists.jsonl")
    imitation_dir = args.imitation or _find_latest(".research/runs/imitation_dataset", "imitation_examples.jsonl")
    cfg = RealMetaArenaConfig(
        games_per_matchup=args.games, max_decks_per_bucket=args.max_decks_per_bucket,
        include_replay=not args.no_replay, seed=args.seed,
    )
    result = build_real_meta_arena(
        deck_zoo_dir=deck_zoo_dir, imitation_dir=imitation_dir, out_root=args.out_root, config=cfg,
        command="python -m deepmew.real_meta_arena.build " + " ".join(argv or []), suffix=args.suffix,
    )
    print(f"real_meta_arena build complete: {result.run_dir}")
    print(f"  deck_zoo: {deck_zoo_dir}")
    print(f"  imitation: {imitation_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
