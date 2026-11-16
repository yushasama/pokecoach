"""Thin CLI wrapper for the DeepMew repository scanner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_json, write_markdown
from deepmew.repo.scan import render_scan_markdown, scan_repo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan DeepMew repo organization and artifact boundaries.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--out-root", default=".research/runs", help="Generated run output root.")
    parser.add_argument("--kind", default="repo_scan", help="Run kind used in the output layout.")
    parser.add_argument("--no-write", action="store_true", help="Print Markdown summary only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    scan = scan_repo(root)
    markdown = render_scan_markdown(scan)

    if args.no_write:
        print(markdown)
        return 0

    command = "python scripts\\repo_scan.py " + " ".join(sys.argv[1:])
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    json_path = run_dir.output_path("repo_scan.json")
    md_path = run_dir.output_path("repo_scan.md")
    write_json(json_path, scan)
    write_markdown(md_path, markdown)
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=run_dir.kind,
        command=command,
        inputs={"root": str(root)},
        outputs={
            "repo_scan_json": str(json_path),
            "repo_scan_markdown": str(md_path),
        },
        summary={
            "total_files": scan["summary"]["total_files"],
            "warning_count": scan["summary"]["warning_count"],
            "unknown_files": scan["summary"]["unknown_files"],
        },
    )
    write_manifest(run_dir.manifest_path, manifest)
    print(f"Wrote repo scan run: {run_dir.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
