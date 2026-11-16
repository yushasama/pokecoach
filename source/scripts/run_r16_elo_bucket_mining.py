"""R16: ELO score-bucket ladder mining (scout + mine phases).

Phase ``scout``: snapshot the full leaderboard, bucket teams by public score
(default 600-800 / 800-900 / 900-1000 / 1000-1100 / 1100-1200 / 1200-1300),
sample random players per bucket, collect their public episode ids, and write
per-bucket episode files + a versioned bucket-metadata artifact under
``.research/elo_bucket_scout/``.

Phase ``mine``: feed each bucket's pending episode file through the
throttle-aware resumable queue (one section per bucket). Re-run after a Kaggle
quota window to resume; already-downloaded episodes are skipped.

Offline mining infrastructure only; never touches main.py/deck.csv/cg/ or the
eval/runtime path.

Examples
--------
    python scripts/run_r16_elo_bucket_mining.py scout --section r16-elo-001
    python scripts/run_r16_elo_bucket_mining.py mine --section r16-elo-001
    python scripts/run_r16_elo_bucket_mining.py mine --section r16-elo-001 --buckets 900-1000 1000-1100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.download import (  # noqa: E402
    DEFAULT_COMPETITION,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPLAY_ROOT,
    KaggleDownloadError,
)
from deepmew.kaggle_replays.elo_buckets import (  # noqa: E402
    DEFAULT_BUCKET_SPECS,
    DEFAULT_ELO_BUCKET_ROOT,
    DEFAULT_EPISODE_IDS_ROOT,
    EloBucketScoutError,
    EloBucketScoutOptions,
    bucket_section_name,
    downloaded_episode_ids,
    find_latest_bucket_scout,
    rescout_failed_teams,
    scout_elo_buckets,
)
from deepmew.kaggle_replays.replay_queue import QueueOptions, ReplayQueueError, mine_replay_queue  # noqa: E402


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R16 ELO score-bucket ladder mining.")
    sub = parser.add_subparsers(dest="phase", required=True)

    scout = sub.add_parser("scout", help="Snapshot leaderboard, sample bucket players, collect episode ids.")
    scout.add_argument("--section", required=True)
    scout.add_argument("--competition", default=DEFAULT_COMPETITION)
    scout.add_argument("--buckets", nargs="*", default=list(DEFAULT_BUCKET_SPECS))
    scout.add_argument("--players-per-bucket", type=int, default=60)
    scout.add_argument("--seed", type=int, default=20260706)
    scout.add_argument("--max-submissions-per-team", type=int, default=1)
    scout.add_argument("--max-episodes-per-submission", type=int, default=6)
    scout.add_argument("--sleep-seconds", type=float, default=0.1)
    scout.add_argument("--kaggle-exe")
    scout.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    scout.add_argument("--scout-root", default=DEFAULT_ELO_BUCKET_ROOT)
    scout.add_argument("--episode-ids-root", default=DEFAULT_EPISODE_IDS_ROOT)
    scout.add_argument("--replay-root", default=DEFAULT_REPLAY_ROOT)
    scout.add_argument("--dry-run", action="store_true")

    rescout = sub.add_parser("rescout", help="Retry teams that failed in a previous scout (quota-wall recovery).")
    rescout.add_argument("--section", required=True)
    rescout.add_argument("--bucket-scout", default=None, help="bucket_teams.json path (default: latest).")
    rescout.add_argument("--competition", default=DEFAULT_COMPETITION)
    rescout.add_argument("--max-submissions-per-team", type=int, default=1)
    rescout.add_argument("--max-episodes-per-submission", type=int, default=6)
    rescout.add_argument("--sleep-seconds", type=float, default=0.1)
    rescout.add_argument("--kaggle-exe")
    rescout.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    rescout.add_argument("--scout-root", default=DEFAULT_ELO_BUCKET_ROOT)
    rescout.add_argument("--episode-ids-root", default=DEFAULT_EPISODE_IDS_ROOT)
    rescout.add_argument("--replay-root", default=DEFAULT_REPLAY_ROOT)

    mine = sub.add_parser("mine", help="Mine pending bucket episodes through the resumable queue.")
    mine.add_argument("--section", required=True)
    mine.add_argument("--buckets", nargs="*", default=None, help="Only these bucket labels (default: all).")
    mine.add_argument("--bucket-scout", default=None, help="bucket_teams.json path (default: latest).")
    mine.add_argument("--batch-size", type=int, default=50)
    mine.add_argument("--max-batches", type=int, default=None)
    mine.add_argument("--sleep-seconds", type=float, default=0.15)
    mine.add_argument("--kaggle-exe")
    mine.add_argument("--replay-root", default=DEFAULT_REPLAY_ROOT)
    mine.add_argument("--out-root", default=DEFAULT_OUT_ROOT)

    return parser.parse_args(argv)


def _run_scout(args: argparse.Namespace) -> int:
    result = scout_elo_buckets(
        EloBucketScoutOptions(
            section=args.section,
            competition=args.competition,
            bucket_specs=args.buckets,
            players_per_bucket=args.players_per_bucket,
            seed=args.seed,
            max_submissions_per_team=args.max_submissions_per_team,
            max_episodes_per_submission=args.max_episodes_per_submission,
            sleep_seconds=args.sleep_seconds,
            kaggle_exe=args.kaggle_exe,
            out_root=Path(args.out_root),
            scout_root=Path(args.scout_root),
            episode_ids_root=Path(args.episode_ids_root),
            replay_root=Path(args.replay_root),
            dry_run=args.dry_run,
            command="python scripts/run_r16_elo_bucket_mining.py " + " ".join(sys.argv[1:]),
            args=vars(args),
        )
    )
    print(f"run_dir={result.run_dir}")
    print(f"metadata={result.metadata_path}")
    print(f"report={result.report_path}")
    for key in ("leaderboard_row_count", "bucket_populations", "bucket_sampled", "teams_with_episodes",
                "unique_episode_count", "already_downloaded_count", "pending_by_bucket", "failure_count"):
        print(f"{key}={result.summary[key]}")
    for label, path in sorted(result.episode_files.items()):
        print(f"episode_file[{label}]={path}")
    return 0 if result.summary["failure_count"] == 0 else 2


def _run_rescout(args: argparse.Namespace) -> int:
    metadata_path = Path(args.bucket_scout) if args.bucket_scout else find_latest_bucket_scout(args.scout_root)
    result = rescout_failed_teams(
        metadata_path,
        EloBucketScoutOptions(
            section=args.section,
            competition=args.competition,
            max_submissions_per_team=args.max_submissions_per_team,
            max_episodes_per_submission=args.max_episodes_per_submission,
            sleep_seconds=args.sleep_seconds,
            kaggle_exe=args.kaggle_exe,
            out_root=Path(args.out_root),
            scout_root=Path(args.scout_root),
            episode_ids_root=Path(args.episode_ids_root),
            replay_root=Path(args.replay_root),
            command="python scripts/run_r16_elo_bucket_mining.py " + " ".join(sys.argv[1:]),
            args=vars(args),
        ),
    )
    print(f"run_dir={result.run_dir}")
    print(f"metadata={result.metadata_path}")
    for key in ("retried_teams", "recovered_teams", "still_failed_teams", "teams_with_episodes",
                "unique_episode_count", "pending_by_bucket", "failure_count"):
        print(f"{key}={result.summary[key]}")
    return 0 if result.summary["still_failed_teams"] == 0 else 2


def _run_mine(args: argparse.Namespace) -> int:
    metadata_path = Path(args.bucket_scout) if args.bucket_scout else find_latest_bucket_scout()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("section") != args.section:
        print(f"NOTE: metadata section {metadata.get('section')!r} != --section {args.section!r}; "
              f"using metadata episode lists with the requested section names.")
    pending_by_bucket: dict[str, list[str]] = metadata.get("pending_episode_ids_by_bucket") or {}
    labels = args.buckets or sorted(pending_by_bucket)
    already = downloaded_episode_ids(args.replay_root)

    throttled = False
    totals = {"requested": 0, "downloaded": 0, "ingested": 0, "remaining": 0}
    for label in labels:
        episode_ids = [eid for eid in pending_by_bucket.get(label, []) if eid not in already]
        if not episode_ids:
            print(f"bucket {label}: nothing pending")
            continue
        totals["requested"] += len(episode_ids)
        try:
            result = mine_replay_queue(
                QueueOptions(
                    episode_ids=episode_ids,
                    section=bucket_section_name(args.section, label),
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    sleep_seconds=args.sleep_seconds,
                    replay_root=Path(args.replay_root),
                    out_root=Path(args.out_root),
                    kaggle_exe=args.kaggle_exe,
                    command="python scripts/run_r16_elo_bucket_mining.py " + " ".join(sys.argv[1:]),
                    args={**vars(args), "bucket": label},
                )
            )
        except (ReplayQueueError, KaggleDownloadError) as exc:
            print(f"bucket {label}: queue failed: {exc}", file=sys.stderr)
            return 1
        s = result.summary
        totals["downloaded"] += s["downloaded"]
        totals["ingested"] += s["ingested"]
        totals["remaining"] += s["remaining"]
        print(f"bucket {label}: downloaded={s['downloaded']} ingested={s['ingested']} "
              f"remaining={s['remaining']} throttled={s['throttled_stop']}")
        if s["throttled_stop"]:
            throttled = True
            break

    print(f"TOTAL requested={totals['requested']} downloaded={totals['downloaded']} "
          f"ingested={totals['ingested']} remaining={totals['remaining']} throttled={throttled}")
    if throttled:
        print("NOTE: stopped on Kaggle throttle; re-run the same command after the quota window resets.")
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    try:
        if args.phase == "scout":
            return _run_scout(args)
        if args.phase == "rescout":
            return _run_rescout(args)
        return _run_mine(args)
    except EloBucketScoutError as exc:
        print(f"scout failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
