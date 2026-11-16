from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.episode_sources import build_episode_source  # noqa: E402


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Kaggle episode IDs into a replay-farming source file.")
    parser.add_argument("--input", nargs="*", default=[], help="Text files with episode IDs, comments, or Kaggle URLs.")
    parser.add_argument("--scan-replay-dir", nargs="*", default=[], help="Directories to scan for episode-*-replay.json files.")
    parser.add_argument("--out", required=True, help="Output episode-ID file.")
    parser.add_argument("--out-root", default=".research", help="Research artifact root for manifest/report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_episode_source(
        input_paths=_expand_inputs(args.input),
        scan_replay_dirs=[Path(path) for path in args.scan_replay_dir],
        out_path=args.out,
        out_root=args.out_root,
        command=" ".join(sys.argv if argv is None else [Path(__file__).name, *argv]),
        args=vars(args),
    )
    print(f"episode_file={result.episode_file}")
    print(f"unique_episode_ids={len(result.episode_ids)}")
    print(f"manifest={result.manifest_path}")
    print(f"report={result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
