"""Download public Kaggle replay JSONs for DeepMew K1.5.

Examples
--------
    python scripts/download_kaggle_replays.py --episodes 81454940 81412357 --section smoke-manual
    python scripts/download_kaggle_replays.py --episodes-file .research/kaggle_episode_ids/top_sample.txt --section top-sample
    python scripts/download_kaggle_replays.py --episodes 81454940 --section smoke-manual --logs 0 1 --ingest-after
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
    DEFAULT_LOG_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPLAY_ROOT,
    DownloadOptions,
    KaggleDownloadError,
    download_kaggle_replays,
    parse_episode_file,
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle replay JSONs into DeepMew research artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episodes", nargs="+", help="Episode IDs to download.")
    source.add_argument("--episodes-file", help="Text file of episode IDs, one per line; blank lines and # comments ignored.")
    parser.add_argument("--section", required=True, help="Output section under .research/kaggle_replays/<section>.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION, help="Competition slug recorded in the manifest.")
    parser.add_argument("--logs", nargs="+", type=int, help="Optional agent log indices to download, e.g. --logs 0 1.")
    parser.add_argument("--force", action="store_true", help="Re-download existing replay/log files.")
    parser.add_argument("--dry-run", action="store_true", help="Record what would be downloaded without calling Kaggle.")
    parser.add_argument("--ingest-after", action="store_true", help="Run K1 ingest on the replay section after download.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Seconds to sleep after each Kaggle call.")
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY_ROOT, help="Root for replay JSON downloads.")
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT, help="Root for optional agent logs.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Root for kaggle_downloads run manifests.")
    parser.add_argument("--kaggle-exe", help="Explicit Kaggle executable path when this shell does not inherit PATH.")
    return parser.parse_args(argv)


def _episodes(args: argparse.Namespace) -> list[str]:
    return [str(value) for value in args.episodes] if args.episodes else parse_episode_file(args.episodes_file)


def _args_dict(args: argparse.Namespace, episode_ids: list[str]) -> dict[str, object]:
    return {
        "episodes": episode_ids,
        "episodes_file": args.episodes_file,
        "section": args.section,
        "competition": args.competition,
        "logs": args.logs,
        "force": args.force,
        "dry_run": args.dry_run,
        "ingest_after": args.ingest_after,
        "sleep_seconds": args.sleep_seconds,
        "replay_root": args.replay_root,
        "log_root": args.log_root,
        "out_root": args.out_root,
        "kaggle_exe": args.kaggle_exe,
    }


def _print_result(result) -> None:
    print(f"run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"download_report={result.report_path}")
    print(f"replay_dir={result.replay_dir}")
    print(f"log_dir={result.log_dir}")
    summary = result.summary
    print(
        "replays "
        f"downloaded={summary['downloaded_replays']} "
        f"skipped={summary['skipped_replays']} "
        f"failed={summary['failed_replays']} "
        f"dry_run={summary['dry_run_replays']}"
    )
    print(
        "logs "
        f"downloaded={summary['downloaded_logs']} "
        f"skipped={summary['skipped_logs']} "
        f"forbidden={summary['forbidden_logs']} "
        f"failed={summary['failed_logs']} "
        f"dry_run={summary['dry_run_logs']}"
    )
    if result.ingest_after_result is not None:
        print(
            f"ingest_after returncode={result.ingest_after_result['returncode']} "
            f"command={' '.join(result.ingest_after_result['args'])}"
        )
    for record in result.replay_records:
        print(f"replay episode={record['episode_id']} status={record['status']} path={record['path']}")
    for record in result.log_records:
        print(
            f"log episode={record['episode_id']} agent={record['agent_index']} "
            f"status={record['status']} path={record['path']}"
        )


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    effective_argv = argv if argv is not None else sys.argv[1:]
    try:
        episode_ids = _episodes(args)
        options = DownloadOptions(
            episode_ids=episode_ids,
            section=args.section,
            competition=args.competition,
            logs=args.logs,
            force=args.force,
            dry_run=args.dry_run,
            ingest_after=args.ingest_after,
            sleep_seconds=args.sleep_seconds,
            replay_root=Path(args.replay_root),
            log_root=Path(args.log_root),
            out_root=Path(args.out_root),
            kaggle_exe=args.kaggle_exe,
            command="python scripts/download_kaggle_replays.py " + " ".join(effective_argv),
            args=_args_dict(args, episode_ids),
        )
        result = download_kaggle_replays(options)
    except KaggleDownloadError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    ingest_rc = None if result.ingest_after_result is None else result.ingest_after_result.get("returncode")
    if result.summary["failed_replays"] or (ingest_rc not in (None, 0)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
