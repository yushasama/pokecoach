"""K2B evaluation of K5-lite pilots vs v2.4 / first-legal / ReplayPolicy.

Pilots DeepMew's frozen `deck.csv` with each candidate policy (v2.4, first-legal,
filtered-BC, reward-weighted-BC) against the K3 deck-zoo bucket opponents
(first-legal piloted) plus the category-only ReplayPolicy opponent on top_meta.
Reuses the K2B runner verbatim; the K2B builder itself stays frozen. Eval-only:
root `main.py` / `deck.csv` / `cg/` / default `v2.4` are never modified.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import main as submission_main
from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_markdown
from deepmew.imitation_fine import io as fio
from deepmew.imitation_fine.action_value import ConservativeActionValueTable
from deepmew.imitation_fine.bc import FineBCTable
from deepmew.imitation_fine.scorer_agent import ConservativeValueAgent, FineBCAgent
from deepmew.meta_kg.io import write_csv
from deepmew.real_meta_arena.build import _load_opponent_sets, _select_bucket_decks
from deepmew.real_meta_arena.decklists import load_zoo_decklists, validate_decklist
from deepmew.real_meta_arena.replay_policy import ReplayAgent, build_replay_policy_table
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup

K5_ARENA_KIND = "k5_lite_arena"
ARTIFACT_VERSION = "deepmew.k5_lite_arena.v1"
BUCKETS = ("top_meta", "anti_overfit_holdout", "unknown_mixed")
OUR_ARCHETYPE_DEFAULT = "Mega Abomasnow ex"

RESULTS_FIELDS = [
    "bucket", "our_label", "opp_kind", "opp_deck_hash", "opp_archetype",
    "games", "completed", "our_wins", "opp_wins", "draws", "crashes", "timeouts", "avg_turns",
]
SUMMARY_FIELDS = [
    "bucket", "our_label", "opp_kind", "decks", "games", "completed",
    "our_wins", "opp_wins", "draws", "our_win_rate", "crashes", "timeouts", "avg_turns",
]


@dataclass
class K5ArenaConfig:
    games_per_matchup: int = 6
    max_decks_per_bucket: int = 3
    buckets: tuple[str, ...] = BUCKETS
    our_archetype: str = OUR_ARCHETYPE_DEFAULT
    include_replay: bool = True
    replay_bucket: str = "top_meta"
    seed: int = 20260624
    max_steps: int = 5000


@dataclass
class K5ArenaResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]
    summary_rows: list[dict[str, Any]] = field(default_factory=list)


def _load_fine_table(imitation_fine_dir: Path, filename: str) -> FineBCTable:
    data = json.loads((imitation_fine_dir / filename).read_text(encoding="utf-8"))
    return FineBCTable.from_dict(data)


def _build_pilots(imitation_fine_dir: Path, archetype: str) -> list[Any]:
    filtered = _load_fine_table(imitation_fine_dir, "filtered_bc_table.json")
    reward = _load_fine_table(imitation_fine_dir, "reward_weighted_bc_table.json")
    value_data = json.loads((imitation_fine_dir / "action_value_table.json").read_text(encoding="utf-8"))
    value_table = ConservativeActionValueTable.from_dict(value_data)
    return [
        SpecAgent.from_str("v2.4"),
        SpecAgent.from_str("first-legal"),
        FineBCAgent(archetype=archetype, table=filtered, label="finebc_filtered"),
        FineBCAgent(archetype=archetype, table=reward, label="finebc_reward"),
        ConservativeValueAgent(archetype=archetype, table=value_table, label="conservative_value"),
    ]


def build_k5_lite_arena(
    deck_zoo_dir: str | Path,
    imitation_fine_dir: str | Path,
    imitation_dir: str | Path,
    out_root: str | Path = ".research/runs",
    config: K5ArenaConfig | None = None,
    *,
    command: str = "",
    suffix: str | None = None,
) -> K5ArenaResult:
    cfg = config or K5ArenaConfig()
    deck_zoo_dir = Path(deck_zoo_dir)
    imitation_fine_dir = Path(imitation_fine_dir)

    decks = load_zoo_decklists(deck_zoo_dir)
    decks_by_hash = {d.deck_hash: d for d in decks}
    valid_hashes = {d.deck_hash for d in decks if validate_decklist(d).valid}
    opponent_sets = _load_opponent_sets(deck_zoo_dir)
    replay_table = build_replay_policy_table(imitation_dir, split="train")

    our_deck = submission_main.read_deck_csv()
    pilots = _build_pilots(imitation_fine_dir, cfg.our_archetype)

    matchups: list[Any] = []
    meta: list[tuple[str, str]] = []
    seed = cfg.seed
    for bucket in cfg.buckets:
        bucket_decks = _select_bucket_decks(opponent_sets.get(bucket, []), valid_hashes, cfg.max_decks_per_bucket)
        for pilot in pilots:
            for entry in bucket_decks:
                opp = decks_by_hash[entry["deck_hash"]]
                seed += 1
                summary = run_real_meta_matchup(
                    pilot, SpecAgent.from_str("first-legal"), our_deck, list(opp.card_ids),
                    games=cfg.games_per_matchup, bucket=bucket,
                    opp_deck_hash=opp.deck_hash, opp_archetype=opp.archetype,
                    seed=seed, max_steps=cfg.max_steps,
                )
                matchups.append(summary)
                meta.append((bucket, "spec:first-legal"))

    if cfg.include_replay:
        replay_decks = _select_bucket_decks(opponent_sets.get(cfg.replay_bucket, []), valid_hashes, cfg.max_decks_per_bucket)
        for pilot in pilots:
            for entry in replay_decks:
                opp = decks_by_hash[entry["deck_hash"]]
                seed += 1
                summary = run_real_meta_matchup(
                    pilot, ReplayAgent(archetype=opp.archetype, table=replay_table),
                    our_deck, list(opp.card_ids),
                    games=cfg.games_per_matchup, bucket=f"{cfg.replay_bucket}(replay)",
                    opp_deck_hash=opp.deck_hash, opp_archetype=opp.archetype,
                    seed=seed, max_steps=cfg.max_steps,
                )
                matchups.append(summary)
                meta.append((f"{cfg.replay_bucket}(replay)", "replay"))

    results_rows = [_row(s, m[1]) for s, m in zip(matchups, meta)]
    summary_rows = _aggregate(matchups, meta)

    run_dir = create_run_dir(kind=K5_ARENA_KIND, out_root=out_root, suffix=suffix)
    out: dict[str, str] = {}
    out["results"] = str(write_csv(run_dir.output_path("k5_lite_arena_results.csv"), RESULTS_FIELDS, results_rows))
    out["summary"] = str(write_csv(run_dir.output_path("k5_lite_arena_summary.csv"), SUMMARY_FIELDS, summary_rows))

    counts = {
        "matchup_count": len(matchups),
        "games_total": sum(s.games for s in matchups),
        "games_completed": sum(s.completed for s in matchups),
        "crashes": sum(s.crashes for s in matchups),
        "timeouts": sum(s.timeouts for s in matchups),
        "pilots": [p.label for p in pilots],
    }
    out["report"] = str(write_markdown(run_dir.output_path("k5_lite_arena_report.md"),
                                       _render(run_dir.path, cfg, counts, summary_rows)))
    manifest = build_manifest(
        run_id=run_dir.run_id, kind=K5_ARENA_KIND, command=command,
        inputs={"deck_zoo_dir": str(deck_zoo_dir), "imitation_fine_dir": str(imitation_fine_dir),
                "imitation_dir": str(imitation_dir), "games": cfg.games_per_matchup,
                "max_decks_per_bucket": cfg.max_decks_per_bucket},
        outputs=out, summary={"artifact_version": ARTIFACT_VERSION, **counts},
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return K5ArenaResult(run_dir.path, manifest_path, out, counts, summary_rows)


def _row(s: Any, opp_kind: str) -> dict[str, Any]:
    return {
        "bucket": s.bucket, "our_label": s.our_label, "opp_kind": opp_kind,
        "opp_deck_hash": s.opp_deck_hash, "opp_archetype": s.opp_archetype,
        "games": s.games, "completed": s.completed, "our_wins": s.our_wins, "opp_wins": s.opp_wins,
        "draws": s.draws, "crashes": s.crashes, "timeouts": s.timeouts, "avg_turns": round(s.avg_turns, 2),
    }


def _aggregate(matchups: list[Any], meta: list[tuple[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for s, (bucket, opp_kind) in zip(matchups, meta):
        groups[(s.bucket, s.our_label, opp_kind)].append(s)
    rows: list[dict[str, Any]] = []
    for (bucket, our_label, opp_kind), group in groups.items():
        our_wins = sum(s.our_wins for s in group)
        opp_wins = sum(s.opp_wins for s in group)
        decided = our_wins + opp_wins
        completed = sum(s.completed for s in group)
        turns = sum(s.avg_turns * s.completed for s in group)
        rows.append({
            "bucket": bucket, "our_label": our_label, "opp_kind": opp_kind,
            "decks": len(group), "games": sum(s.games for s in group), "completed": completed,
            "our_wins": our_wins, "opp_wins": opp_wins, "draws": sum(s.draws for s in group),
            "our_win_rate": round(our_wins / decided, 4) if decided else 0.0,
            "crashes": sum(s.crashes for s in group), "timeouts": sum(s.timeouts for s in group),
            "avg_turns": round(turns / completed, 2) if completed else 0.0,
        })
    rows.sort(key=lambda r: (r["bucket"], r["our_label"], r["opp_kind"]))
    return rows


def _render(out_dir: Path, cfg: K5ArenaConfig, counts: dict[str, Any], summary_rows: list[dict[str, Any]]) -> str:
    lines = ["# K5-lite Real-Meta Arena (eval-only)", "",
             f"- Output: `{out_dir}`",
             f"- Our deck: frozen `deck.csv` (archetype piloted as `{cfg.our_archetype}`)",
             f"- Pilots: {', '.join(counts['pilots'])}",
             f"- Games/matchup: {cfg.games_per_matchup}, decks/bucket: {cfg.max_decks_per_bucket}",
             f"- Totals: {counts['games_completed']}/{counts['games_total']} completed, "
             f"crashes={counts['crashes']}, timeouts={counts['timeouts']}", "",
             "## Our-deck win rate by pilot × bucket", "",
             "| bucket | pilot | opp_kind | decks | completed | our_win_rate |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in summary_rows:
        lines.append(f"| {r['bucket']} | {r['our_label']} | {r['opp_kind']} | {r['decks']} | "
                     f"{r['completed']} | {r['our_win_rate']} |")
    lines += ["", "Promotion gate: a pilot is promotion-worthy only if it beats v2.4 AND first-legal on "
              "widened K2B, does not regress anti_overfit_holdout, does not only farm weak ReplayPolicy, "
              "and keeps 0 crashes/timeouts -- then ladder A/B before any default change. Eval-only.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="K2B eval of K5-lite pilots vs v2.4/first-legal/ReplayPolicy.")
    parser.add_argument("--deck-zoo", default=None)
    parser.add_argument("--imitation-fine", default=None)
    parser.add_argument("--imitation", default=None)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--max-decks-per-bucket", type=int, default=3)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--archetype", default=OUR_ARCHETYPE_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    deck_zoo_dir = args.deck_zoo or fio.find_latest_deck_zoo()
    imitation_fine_dir = args.imitation_fine or fio.find_latest_run(".research/runs/imitation_fine", "filtered_bc_table.json")
    imitation_dir = args.imitation or fio.find_latest_imitation()
    cfg = K5ArenaConfig(
        games_per_matchup=args.games, max_decks_per_bucket=args.max_decks_per_bucket,
        our_archetype=args.archetype, include_replay=not args.no_replay, seed=args.seed,
    )
    result = build_k5_lite_arena(
        deck_zoo_dir=deck_zoo_dir, imitation_fine_dir=imitation_fine_dir, imitation_dir=imitation_dir,
        out_root=args.out_root, config=cfg,
        command="python -m deepmew.imitation_fine.arena_eval " + " ".join(argv or []), suffix=args.suffix,
    )
    print(f"k5_lite_arena build complete: {result.run_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    for r in result.summary_rows:
        print(f"  {r['bucket']:>22} | {r['our_label']:>18} | wr={r['our_win_rate']} (n={r['completed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
