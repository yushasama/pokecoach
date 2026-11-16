from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.submissions.bundle import DEFAULT_OUT_ROOT, SubmissionBundleError, build_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen DeepMew submission bundle.")
    parser.add_argument("--agent", required=True, help="Agent spec to freeze. Currently supports v2.4.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned files without writing a bundle.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Submission output root.")
    parser.add_argument("--no-zip", action="store_true", help="Skip zip creation.")
    parser.add_argument("--no-tar-gz", action="store_true", help="Skip Kaggle-ready tar.gz creation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_submission(
            repo_root=ROOT,
            agent=args.agent,
            out_root=args.out_root,
            dry_run=args.dry_run,
            make_zip=not args.no_zip,
            make_tar_gz=not args.no_tar_gz,
        )
    except SubmissionBundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
