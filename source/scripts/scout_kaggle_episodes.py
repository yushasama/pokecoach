from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.scout import ScoutOptions, scout_kaggle_episodes  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scout public Kaggle episodes for a stratified replay corpus.")
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--section", required=True)
    parser.add_argument("--rank-ranges", nargs="*", default=[])
    parser.add_argument("--teams-per-range", type=int, default=10)
    parser.add_argument("--all-teams-in-range", action="store_true")
    parser.add_argument("--team-ids", nargs="*", default=[])
    parser.add_argument("--team-name-contains", nargs="*", default=[])
    parser.add_argument("--neighbor-window", type=int)
    parser.add_argument("--submission-ids", nargs="*", default=[])
    parser.add_argument("--max-submissions-per-team", type=int, default=2)
    parser.add_argument("--max-episodes-per-submission", type=int, default=20)
    parser.add_argument("--download-after", action="store_true")
    parser.add_argument("--ingest-after", action="store_true")
    parser.add_argument(
        "--no-corpus-after",
        dest="corpus_after",
        action="store_false",
        help="Skip refreshing .research/kaggle_corpus after a real download/ingest (on by default).",
    )
    parser.set_defaults(corpus_after=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--kaggle-exe")
    parser.add_argument("--out-root", default=".research")
    parser.add_argument("--episode-ids-root", default=".research/kaggle_episode_ids")
    return parser.parse_args(argv)


def _options(args: argparse.Namespace) -> ScoutOptions:
    return ScoutOptions(
        competition=args.competition,
        section=args.section,
        rank_ranges=args.rank_ranges,
        teams_per_range=args.teams_per_range,
        all_teams_in_range=args.all_teams_in_range,
        team_ids=args.team_ids,
        team_name_contains=args.team_name_contains,
        neighbor_window=args.neighbor_window,
        submission_ids=args.submission_ids,
        max_submissions_per_team=args.max_submissions_per_team,
        max_episodes_per_submission=args.max_episodes_per_submission,
        download_after=args.download_after,
        ingest_after=args.ingest_after,
        corpus_after=args.corpus_after,
        dry_run=args.dry_run,
        strict=args.strict,
        sleep_seconds=args.sleep_seconds,
        kaggle_exe=args.kaggle_exe,
        out_root=Path(args.out_root),
        episode_ids_root=Path(args.episode_ids_root),
        command=" ".join(sys.argv),
        args=vars(args),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = scout_kaggle_episodes(_options(args))
    print(f"episode_file={result.episode_file}")
    print(f"unique_episode_ids={len(result.episode_ids)}")
    print(f"manifest={result.manifest_path}")
    print(f"report={result.report_path}")
    print(f"leaderboard_rows={result.summary['leaderboard_row_count']}")
    print(f"selected_teams={result.summary['selected_team_count']}")
    print(f"selected_submissions={result.summary['selected_submission_count']}")
    if result.download_after_result is not None:
        print(f"download_manifest={result.download_after_result.manifest_path}")
        print(f"download_replay_dir={result.download_after_result.replay_dir}")
        print(f"download_summary={result.download_after_result.summary}")
    if result.corpus_after_result is not None:
        print(f"corpus_manifest={result.corpus_after_result.manifest_path}")
        print(f"corpus_report={result.corpus_after_result.report_path}")
        print(f"corpus_summary={result.corpus_after_result.summary}")
    return 0 if result.summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
