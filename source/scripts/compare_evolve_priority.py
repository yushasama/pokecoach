from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.trace.evolve_priority_review import RUN_KIND, write_review_outputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BabyMew evolve_priority review artifacts.")
    parser.add_argument("--pair-summary", required=True)
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--default-audit", required=True)
    parser.add_argument("--ablation-audit", required=True)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    parser.add_argument("--export-handoff-bundle", action="store_true")
    parser.add_argument("--handoff-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def _command() -> str:
    return "python scripts\\compare_evolve_priority.py " + " ".join(sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    result = write_review_outputs(
        run_dir=run_dir.path,
        command=_command(),
        pair_summary_path=Path(args.pair_summary),
        matrix_summary_path=Path(args.matrix_summary),
        default_audit_path=Path(args.default_audit),
        ablation_audit_path=Path(args.ablation_audit),
        export_handoff_bundle=args.export_handoff_bundle,
        handoff_root=args.handoff_root,
    )
    print(f"evolve_priority_run_dir={result['run_dir']}")
    print(f"manifest={run_dir.manifest_path}")
    print(f"comparison={run_dir.path / 'comparison.json'}")
    print(f"report={run_dir.path / 'report.md'}")
    if result["bundle_dir"] is not None:
        print(f"handoff_bundle_dir={result['bundle_dir']}")
    comparison = result["comparison"]
    print(f"play_beat_evolve_default={comparison['play_beat_evolve']['default']}")
    print(f"play_beat_evolve_ablation={comparison['play_beat_evolve']['ablation']}")
    print(f"non_target_play_beat_evolve_default={comparison['non_target_play_beat_evolve']['default']}")
    print(f"non_target_play_beat_evolve_ablation={comparison['non_target_play_beat_evolve']['ablation']}")
    print(f"win_rate_assessment={comparison['win_rate_assessment']}")
    print(f"recommendation={comparison['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
