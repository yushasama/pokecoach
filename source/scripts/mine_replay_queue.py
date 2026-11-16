"""Resumable, throttle-aware Kaggle replay mining queue.

Downloads public replays X at a time, stops the instant Kaggle throttles the
per-account window quota, ingests what it fetched, and refreshes the corpus report.
Re-run the same command after the quota resets to continue: already-downloaded
replays are skipped, so only missing episodes are attempted.

Offline-safe infrastructure only; never touches root main.py/deck.csv/cg/ or the
eval/runtime path.

Examples
--------
    # Resume the stratified pull, 50 at a time, stop on throttle (default):
    python scripts/mine_replay_queue.py \
      --episodes-file .research/kaggle_episode_ids/stratified-001.txt \
      --section stratified-001 --batch-size 50

    # Just one batch (e.g. to probe whether the quota has reset):
    python scripts/mine_replay_queue.py --episodes 81258940 81253346 --section probe --batch-size 2
"""

from __future__ import annotations

import argparse
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
    parse_episode_file,
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
    parser = argparse.ArgumentParser(description="Resumable, throttle-aware Kaggle replay mining queue.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episodes", nargs="+", help="Episode IDs to mine.")
    source.add_argument("--episodes-file", help="Text file of episode IDs (one per line; # comments and blanks ignored).")
    parser.add_argument("--section", required=True, help="Output section under .research/kaggle_replays/<section>.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--batch-size", type=int, default=50, help="Replays to download per batch.")
    parser.add_argument("--max-batches", type=int, help="Cap the number of batches this run (default: all).")
    parser.add_argument("--no-stop-on-throttle", dest="stop_on_throttle", action="store_false",
                        help="Keep attempting after a throttle wall instead of stopping (off by default).")
    parser.add_argument("--no-ingest-after", dest="ingest_after", action="store_false", help="Skip ingesting downloaded replays.")
    parser.add_argument("--no-corpus-after", dest="corpus_after", action="store_false", help="Skip refreshing the corpus report.")
    parser.add_argument("--force", action="store_true", help="Re-download replays even if already present.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Seconds to sleep between Kaggle calls.")
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--kaggle-exe", help="Explicit Kaggle executable path (else env/.research/local config/PATH).")
    parser.set_defaults(stop_on_throttle=True, ingest_after=True, corpus_after=True)
    return parser.parse_args(argv)


def _episodes(args: argparse.Namespace) -> list[str]:
    return [str(value) for value in args.episodes] if args.episodes else parse_episode_file(args.episodes_file)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    effective_argv = argv if argv is not None else sys.argv[1:]
    try:
        episode_ids = _episodes(args)
        result = mine_replay_queue(
            QueueOptions(
                episode_ids=episode_ids,
                section=args.section,
                competition=args.competition,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                stop_on_throttle=args.stop_on_throttle,
                ingest_after=args.ingest_after,
                corpus_after=args.corpus_after,
                force=args.force,
                sleep_seconds=args.sleep_seconds,
                replay_root=Path(args.replay_root),
                out_root=Path(args.out_root),
                kaggle_exe=args.kaggle_exe,
                command="python scripts/mine_replay_queue.py " + " ".join(effective_argv),
                args=vars(args),
            )
        )
    except (KaggleDownloadError, ReplayQueueError) as exc:
        print(f"queue failed: {exc}", file=sys.stderr)
        return 1

    s = result.summary
    print(f"section={s['section']}")
    print(f"requested={s['requested_episode_count']} already_downloaded={s['already_downloaded']} pending_at_start={s['pending_at_start']}")
    print(f"batches_run={s['batches_run']} batch_size={s['batch_size']} downloaded={s['downloaded']} ingested={s['ingested']}")
    print(f"throttled_stop={s['throttled_stop']} remaining={s['remaining']} complete={s['complete']}")
    if result.corpus_result is not None:
        print(f"corpus_report={result.corpus_result.report_path}")
        print(f"corpus_summary={result.corpus_result.summary}")
    print(f"queue_report={result.report_path}")
    print(f"manifest={result.manifest_path}")
    if s["throttled_stop"]:
        print(f"NOTE: stopped on Kaggle throttle; {s['remaining']} remain. Re-run the same command after the quota resets.")
    # Non-zero only on a hard failure to start; a throttle stop is an expected, resumable state.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
