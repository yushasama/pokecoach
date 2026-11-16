from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.trace.handoff_bundle import export_babymew_handoff_bundle, find_latest_timing_run  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact BabyMew PLAY identity review bundle.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--latest-timing-run", action="store_true", help="Use latest deck_attachment_timing run.")
    source.add_argument("--timing-run", help="Explicit deck_attachment_timing run directory.")
    parser.add_argument("--out-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timing_run = find_latest_timing_run() if args.latest_timing_run else Path(args.timing_run)
    command = "python scripts\\export_babymew_handoff_bundle.py " + " ".join(sys.argv[1:])
    result = export_babymew_handoff_bundle(
        timing_run=timing_run,
        out_root=Path(args.out_root),
        command=command,
    )
    print(f"handoff_bundle_dir={result.bundle_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"files={','.join(result.files)}")
    print(f"counts={result.counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
