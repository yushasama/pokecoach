"""Thin CLI wrapper for the DeepMew code-quality audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_json, write_markdown  # noqa: E402
from deepmew.repo.code_quality import render_code_quality_markdown, scan_code_quality  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DeepMew code quality and artifact-output discipline.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--out-root", default=".research/runs", help="Generated run output root.")
    parser.add_argument("--kind", default="code_quality", help="Run kind used in the output layout.")
    parser.add_argument("--no-write", action="store_true", help="Print Markdown summary only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    audit = scan_code_quality(root)
    markdown = render_code_quality_markdown(audit)

    if args.no_write:
        print(markdown)
        return 0

    command_args = sys.argv[1:] if argv is None else argv
    command = "python scripts\\code_quality_audit.py " + " ".join(command_args)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    json_path = run_dir.output_path("code_quality.json")
    md_path = run_dir.output_path("report.md")
    write_json(json_path, audit)
    write_markdown(md_path, markdown)
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=run_dir.kind,
        command=command,
        inputs={"root": str(root)},
        outputs={
            "code_quality_json": str(json_path),
            "code_quality_report": str(md_path),
        },
        summary={
            "warning_count": audit["summary"]["warning_count"],
            "python_modules_scanned": audit["summary"]["python_modules_scanned"],
            "generated_run_directories_scanned": audit["artifact_discipline"]["generated_run_directories_scanned"],
            "run_directories_missing_manifests": audit["artifact_discipline"]["run_directories_missing_manifests"],
            "legacy_flat_outputs": audit["artifact_discipline"]["legacy_flat_outputs"],
        },
    )
    write_manifest(run_dir.manifest_path, manifest)
    print(f"Wrote code-quality audit run: {run_dir.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
