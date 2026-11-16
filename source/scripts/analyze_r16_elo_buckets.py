"""R16: per-ELO-bucket meta / mistake / punishment / consistency analysis.

Joins K4.1 fine-grained decision examples (punishment / risk / last-stand
pseudo-labels + outcomes) to the R16 score-bucket player sample, and reports
per bucket:

* meta composition + per-archetype win rates (what each score band plays),
* decomposed mistake rates and how often mistakes go UNPUNISHED (pilot still
  won despite unjustified punished-surface choices) — exploit opportunities,
* which opponent archetypes convert mistakes into losses hardest — the
  punishers we must not be exploitable by,
* playstyle profile (MAIN-context option mix, risk/last-stand rates) and
  per-player consistency (win rate, deck stickiness).

All mistake/punishment findings are observational single-frame pseudo-labels:
possible, not proven causal. Bucket = leaderboard score at scout time, not
per-game rating. Offline/research-only; never touches the eval/runtime path.

Examples
--------
    python scripts/analyze_r16_elo_buckets.py --section r16-elo-001
    python scripts/analyze_r16_elo_buckets.py --section r16-elo-001 \
        --fine-run .research/runs/imitation_fine/20260707/<run_id>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_markdown  # noqa: E402
from deepmew.imitation_fine import io as fio  # noqa: E402
from deepmew.imitation_fine.schemas import IMITATION_FINE_KIND, OUTPUT_FILES, PUNISHMENT_VOCAB  # noqa: E402
from deepmew.kaggle_replays.elo_buckets import find_latest_bucket_scout  # noqa: E402
from deepmew.meta_kg.io import write_csv  # noqa: E402
from deepmew.rank_meta.loader import DEFAULT_OPPONENT_DECKS, iter_deck_summaries  # noqa: E402

ELO_BUCKET_ANALYSIS_KIND = "elo_bucket_analysis"

META_FIELDS = ["bucket", "archetype", "n_players", "n_game_seats", "game_share", "wins", "win_rate"]
MISTAKE_FIELDS = ["bucket", "punishment_label", "n_rows", "row_share", "unjustified_rows",
                  "unjustified_in_won_games", "unpunished_rate"]
PUNISHER_FIELDS = ["bucket", "opponent_archetype", "games_with_mistakes", "losses", "mistake_loss_rate",
                   "mistake_rows"]
PLAYSTYLE_FIELDS = ["bucket", "dimension", "value", "n_rows", "share"]
PLAYER_FIELDS = ["bucket", "team_name", "score", "games", "distinct_decks", "wins", "win_rate",
                 "mistake_rate", "unpunished_mistake_share"]
COVERAGE_FIELDS = ["bucket", "teams_sampled", "teams_with_mined_games", "game_seats", "decision_rows"]


def _normalize_team_name(name: str) -> str:
    r"""Decode Kaggle-CLI byte-repr team names (``D\xc5\xa9ng`` -> ``Dũng``).

    The Windows CLI prints non-ASCII names as a Python bytes repr; deck
    summaries store the decoded UTF-8 name, so the join needs the decoded form.
    """
    if "\\x" not in name:
        return name
    try:
        return name.encode("ascii").decode("unicode_escape").encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def _load_bucket_teams(metadata_path: Path) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, int]]:
    """team_name -> {bucket, score}; ambiguous duplicate names are dropped with a warning."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sampled_per_bucket: Counter = Counter()
    for record in metadata.get("sampled_teams") or []:
        name = record.get("team_name")
        if name:
            by_name[_normalize_team_name(str(name))].append(record)
            sampled_per_bucket[str(record.get("bucket"))] += 1
    teams: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for name, records in by_name.items():
        team_ids = {str(r.get("team_id")) for r in records}
        if len(team_ids) > 1:
            warnings.append(f"ambiguous team_name {name!r} across team_ids {sorted(team_ids)}; dropped")
            continue
        teams[name] = records[0]
    return teams, warnings, dict(sampled_per_bucket)


def _episode_seat_buckets(
    teams: dict[str, dict[str, Any]],
    opponent_decks_root: str | Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """(episode_id, seat) -> sampled-team record, via the deck-summary team_name join."""
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    for record in iter_deck_summaries(opponent_decks_root):
        team = teams.get(str(record.get("team_name") or ""))
        seat = record.get("player_index")
        if team is None or seat is None or not record.get("episode_id"):
            continue
        mapping[(record["episode_id"], seat)] = team
    return mapping


def _iter_fine_examples(fine_run: Path) -> Iterator[dict[str, Any]]:
    path = fine_run / OUTPUT_FILES["examples"]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_unjustified_mistake(row: dict[str, Any]) -> bool:
    return bool(row.get("punishment_labels")) and not row.get("risk_justification") and not row.get("last_stand_triggered")


def analyze(args: argparse.Namespace) -> int:
    metadata_path = Path(args.bucket_scout) if args.bucket_scout else find_latest_bucket_scout()
    fine_run = Path(args.fine_run) if args.fine_run else fio.find_latest_run(
        Path(args.runs_root) / IMITATION_FINE_KIND, OUTPUT_FILES["examples"])
    teams, warnings, sampled_per_bucket = _load_bucket_teams(metadata_path)
    seat_bucket = _episode_seat_buckets(teams, args.opponent_decks)

    # Aggregations. Game grain = (episode_id, seat) of a bucketed pilot.
    game_info: dict[tuple[str, int], dict[str, Any]] = {}
    decision_rows: Counter = Counter()                      # bucket -> rows
    punish_rows: dict[str, Counter] = defaultdict(Counter)  # bucket -> label -> rows
    punish_unjust: dict[str, Counter] = defaultdict(Counter)
    punish_unjust_won: dict[str, Counter] = defaultdict(Counter)
    style_main_option: dict[str, Counter] = defaultdict(Counter)
    style_flags: dict[str, Counter] = defaultdict(Counter)
    mistakes_by_game: Counter = Counter()                   # (episode, seat) -> unjustified mistake rows

    for row in _iter_fine_examples(fine_run):
        key = (str(row.get("episode_id") or ""), row.get("seat"))
        team = seat_bucket.get(key)
        if team is None:
            continue
        bucket = str(team.get("bucket"))
        decision_rows[bucket] += 1
        won = bool(row.get("won"))
        info = game_info.setdefault(key, {
            "bucket": bucket, "team_name": str(team.get("team_name")), "score": team.get("score"),
            "archetype": str(row.get("archetype") or "Unknown"),
            "opponent_archetype": str(row.get("opponent_archetype") or "Unknown"),
            "deck_hash": str(row.get("deck_hash") or ""), "won": won,
        })
        for label in row.get("punishment_labels") or []:
            punish_rows[bucket][label] += 1
        if _is_unjustified_mistake(row):
            mistakes_by_game[key] += 1
            for label in row.get("punishment_labels") or []:
                punish_unjust[bucket][label] += 1
                if info["won"]:
                    punish_unjust_won[bucket][label] += 1
        if row.get("select_context") == "MAIN":
            style_main_option[bucket][str(row.get("option_type") or "?")] += 1
        if row.get("risk_justification"):
            style_flags[bucket]["risk_justified_rows"] += 1
        if row.get("last_stand_triggered"):
            style_flags[bucket]["last_stand_rows"] += 1

    if not game_info:
        print("no bucketed decision rows found — mine bucket replays and rebuild the fine dataset first",
              file=sys.stderr)
        return 2

    # --- meta composition + archetype win rates per bucket ---
    meta_rows: list[dict[str, Any]] = []
    bucket_games: Counter = Counter(info["bucket"] for info in game_info.values())
    arch_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for info in game_info.values():
        arch_group[(info["bucket"], info["archetype"])].append(info)
    for (bucket, archetype), infos in sorted(arch_group.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        wins = sum(1 for i in infos if i["won"])
        meta_rows.append({
            "bucket": bucket, "archetype": archetype,
            "n_players": len({i["team_name"] for i in infos}),
            "n_game_seats": len(infos),
            "game_share": round(len(infos) / bucket_games[bucket], 4),
            "wins": wins, "win_rate": round(wins / len(infos), 4),
        })

    # --- mistakes + unpunished rates per bucket ---
    mistake_rows: list[dict[str, Any]] = []
    for bucket in sorted(decision_rows):
        for label in sorted(PUNISHMENT_VOCAB):
            n_rows = punish_rows[bucket].get(label, 0)
            unjust = punish_unjust[bucket].get(label, 0)
            unjust_won = punish_unjust_won[bucket].get(label, 0)
            mistake_rows.append({
                "bucket": bucket, "punishment_label": label, "n_rows": n_rows,
                "row_share": round(n_rows / decision_rows[bucket], 4) if decision_rows[bucket] else 0.0,
                "unjustified_rows": unjust, "unjustified_in_won_games": unjust_won,
                "unpunished_rate": round(unjust_won / unjust, 4) if unjust else 0.0,
            })

    # --- punishers: opponent archetypes converting mistakes into losses ---
    punisher_group: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for key, info in game_info.items():
        n_mistakes = mistakes_by_game.get(key, 0)
        if n_mistakes:
            punisher_group[(info["bucket"], info["opponent_archetype"])].append((info, n_mistakes))
    punisher_rows: list[dict[str, Any]] = []
    for (bucket, opp), games in sorted(punisher_group.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        losses = sum(1 for info, _n in games if not info["won"])
        punisher_rows.append({
            "bucket": bucket, "opponent_archetype": opp, "games_with_mistakes": len(games),
            "losses": losses, "mistake_loss_rate": round(losses / len(games), 4),
            "mistake_rows": sum(n for _info, n in games),
        })

    # --- playstyle profile per bucket ---
    playstyle_rows: list[dict[str, Any]] = []
    for bucket in sorted(decision_rows):
        main_total = sum(style_main_option[bucket].values())
        for option_type, n in style_main_option[bucket].most_common():
            playstyle_rows.append({"bucket": bucket, "dimension": "main_option_type", "value": option_type,
                                   "n_rows": n, "share": round(n / main_total, 4) if main_total else 0.0})
        for flag in ("risk_justified_rows", "last_stand_rows"):
            n = style_flags[bucket].get(flag, 0)
            playstyle_rows.append({"bucket": bucket, "dimension": "flag", "value": flag, "n_rows": n,
                                   "share": round(n / decision_rows[bucket], 4) if decision_rows[bucket] else 0.0})

    # --- per-player consistency ---
    player_group: dict[tuple[str, str], list[tuple[tuple[str, int], dict[str, Any]]]] = defaultdict(list)
    for key, info in game_info.items():
        player_group[(info["bucket"], info["team_name"])].append((key, info))
    player_rows: list[dict[str, Any]] = []
    for (bucket, team_name), games in sorted(player_group.items()):
        wins = sum(1 for _k, i in games if i["won"])
        n_mist_games = sum(1 for k, _i in games if mistakes_by_game.get(k, 0))
        n_unpunished = sum(1 for k, i in games if mistakes_by_game.get(k, 0) and i["won"])
        player_rows.append({
            "bucket": bucket, "team_name": team_name, "score": games[0][1]["score"],
            "games": len(games), "distinct_decks": len({i["deck_hash"] for _k, i in games}),
            "wins": wins, "win_rate": round(wins / len(games), 4),
            "mistake_rate": round(n_mist_games / len(games), 4),
            "unpunished_mistake_share": round(n_unpunished / n_mist_games, 4) if n_mist_games else 0.0,
        })

    # --- coverage ---
    coverage_rows: list[dict[str, Any]] = []
    teams_seen: dict[str, set] = defaultdict(set)
    for info in game_info.values():
        teams_seen[info["bucket"]].add(info["team_name"])
    for bucket in sorted(set(sampled_per_bucket) | set(bucket_games)):
        coverage_rows.append({
            "bucket": bucket, "teams_sampled": sampled_per_bucket.get(bucket, 0),
            "teams_with_mined_games": len(teams_seen.get(bucket, set())),
            "game_seats": bucket_games.get(bucket, 0), "decision_rows": decision_rows.get(bucket, 0),
        })

    run_dir = create_run_dir(kind=ELO_BUCKET_ANALYSIS_KIND, out_root=args.out_root, suffix=args.section)
    out = {
        "bucket_meta": str(write_csv(run_dir.output_path("bucket_meta.csv"), META_FIELDS, meta_rows)),
        "bucket_mistakes": str(write_csv(run_dir.output_path("bucket_mistakes.csv"), MISTAKE_FIELDS, mistake_rows)),
        "bucket_punishers": str(write_csv(run_dir.output_path("bucket_punishers.csv"), PUNISHER_FIELDS, punisher_rows)),
        "bucket_playstyle": str(write_csv(run_dir.output_path("bucket_playstyle.csv"), PLAYSTYLE_FIELDS, playstyle_rows)),
        "bucket_players": str(write_csv(run_dir.output_path("bucket_players.csv"), PLAYER_FIELDS, player_rows)),
        "coverage": str(write_csv(run_dir.output_path("coverage.csv"), COVERAGE_FIELDS, coverage_rows)),
    }
    report = _report(args, metadata_path, fine_run, coverage_rows, meta_rows, mistake_rows, punisher_rows,
                     player_rows, warnings)
    out["report"] = str(write_markdown(run_dir.output_path("report.md"), report))
    summary = {
        "buckets": {r["bucket"]: r["game_seats"] for r in coverage_rows},
        "total_game_seats": len(game_info),
        "total_decision_rows": sum(decision_rows.values()),
        "warnings": len(warnings),
    }
    write_manifest(run_dir.manifest_path, build_manifest(
        run_id=run_dir.run_id, kind=ELO_BUCKET_ANALYSIS_KIND,
        command="python scripts/analyze_r16_elo_buckets.py " + " ".join(sys.argv[1:]),
        inputs={"bucket_scout": str(metadata_path), "fine_run": str(fine_run),
                "opponent_decks": str(args.opponent_decks)},
        outputs=out, summary=summary,
    ))
    print(f"run_dir={run_dir.path}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


def _top(rows: list[dict[str, Any]], bucket: str, key: str, n: int = 5, min_games: int = 0,
         games_key: str | None = None) -> list[dict[str, Any]]:
    pool = [r for r in rows if r["bucket"] == bucket and (not games_key or r[games_key] >= min_games)]
    return sorted(pool, key=lambda r: -r[key])[:n]


def _report(args, metadata_path, fine_run, coverage_rows, meta_rows, mistake_rows, punisher_rows,
            player_rows, warnings) -> str:
    lines = [
        "# R16 ELO Bucket Analysis",
        "",
        f"bucket_scout={metadata_path}",
        f"fine_run={fine_run}",
        "",
        "## Coverage",
        "",
        "| bucket | teams sampled | teams w/ mined games | game seats | decision rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in coverage_rows:
        lines.append(f"| {r['bucket']} | {r['teams_sampled']} | {r['teams_with_mined_games']} "
                     f"| {r['game_seats']} | {r['decision_rows']} |")
    buckets = [r["bucket"] for r in coverage_rows if r["game_seats"]]
    for bucket in buckets:
        lines += ["", f"## Bucket {bucket}", "", "Top archetypes (by games):"]
        for r in _top(meta_rows, bucket, "n_game_seats"):
            lines.append(f"- {r['archetype']}: share {r['game_share']}, win_rate {r['win_rate']} "
                         f"({r['n_game_seats']} games, {r['n_players']} players)")
        lines += ["", "Most unpunished mistake surfaces (unjustified rows that still won):"]
        for r in _top([m for m in mistake_rows if m["unjustified_rows"] >= 20], bucket, "unpunished_rate"):
            lines.append(f"- {r['punishment_label']}: unpunished_rate {r['unpunished_rate']} "
                         f"({r['unjustified_in_won_games']}/{r['unjustified_rows']} unjustified rows in won games)")
        lines += ["", "Hardest punishers (opponent archetypes converting mistakes to losses, >=5 games):"]
        for r in _top(punisher_rows, bucket, "mistake_loss_rate", min_games=5, games_key="games_with_mistakes"):
            lines.append(f"- {r['opponent_archetype']}: mistake_loss_rate {r['mistake_loss_rate']} "
                         f"({r['losses']}/{r['games_with_mistakes']} games)")
        bucket_players = [p for p in player_rows if p["bucket"] == bucket and p["games"] >= 3]
        if bucket_players:
            wrs = [p["win_rate"] for p in bucket_players]
            sticky = sum(1 for p in bucket_players if p["distinct_decks"] == 1) / len(bucket_players)
            lines += ["", f"Consistency ({len(bucket_players)} players with >=3 games): "
                          f"mean win_rate {round(statistics.mean(wrs), 4)}, "
                          f"stdev {round(statistics.stdev(wrs), 4) if len(wrs) > 1 else 0.0}, "
                          f"single-deck players {round(sticky, 4)}"]
    lines += ["", "## Warnings", ""]
    lines.extend(f"- {w}" for w in warnings) if warnings else lines.append("- None.")
    lines += [
        "",
        "## Caveats",
        "",
        "- Mistake/punishment labels are observational single-frame pseudo-labels: possible, not proven causal.",
        "- Bucket = leaderboard public score at scout time, not per-game rating; episodes include any mined",
        "  historical games by the sampled teams.",
        "- 'Unpunished' means the pilot won the game despite unjustified punished-surface rows — an upper-bound",
        "  proxy for exploitable slack at that score band, not a per-decision counterfactual.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R16 per-ELO-bucket meta/mistake/punishment analysis.")
    parser.add_argument("--section", required=True, help="Section label for the analysis run id.")
    parser.add_argument("--bucket-scout", default=None, help="bucket_teams.json (default: latest).")
    parser.add_argument("--fine-run", default=None, help="imitation_fine run dir (default: latest).")
    parser.add_argument("--runs-root", default=".research/runs")
    parser.add_argument("--opponent-decks", default=DEFAULT_OPPONENT_DECKS)
    parser.add_argument("--out-root", default=".research/runs")
    return analyze(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
