from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_jsonl, write_markdown  # noqa: E402
from deepmew.trace.diff import diff_probe_sets  # noqa: E402
from deepmew.trace.probe import probe_frames, read_frames, resolve_frames_path  # noqa: E402
from deepmew.trace.report import build_probe_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe deterministic policies against recorded DeepMew trace frames.")
    parser.add_argument("trace", help="Path to frames.jsonl or a trace run directory.")
    parser.add_argument("--policies", nargs="+", default=["v2.4", "v2.4[semantics=off]", "first-legal"])
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default="probe")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames_path = resolve_frames_path(args.trace)
    frames = read_frames(frames_path, limit=args.limit)
    probes = probe_frames(frames, args.policies)
    base_policy = probes[0].policy_spec if probes else None
    diffs = diff_probe_sets(probes, base_policy=base_policy)

    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    probes_path = run_dir.output_path("probes.jsonl")
    diffs_path = run_dir.output_path("diffs.jsonl")
    report_path = run_dir.output_path("report.md")

    write_jsonl(probes_path, [probe.to_jsonable() for probe in probes])
    write_jsonl(diffs_path, [diff.to_jsonable() for diff in diffs])

    skipped_frames = 0
    for frame in frames:
        frame_id = frame.get("frame_id")
        frame_probes = [probe for probe in probes if probe.frame_id == frame_id]
        if frame_probes and all(probe.error for probe in frame_probes):
            skipped_frames += 1

    report = build_probe_report(
        run_id=run_dir.run_id,
        frames=frames,
        probes=probes,
        diffs=diffs,
        agents=args.policies,
        source_trace=str(frames_path),
        skipped_frames=skipped_frames,
    )
    write_markdown(report_path, report)

    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=run_dir.kind,
        command="python scripts\\trace_probe.py " + " ".join(sys.argv[1:]),
        inputs={
            "trace": str(frames_path),
            "policies": args.policies,
            "limit": args.limit,
        },
        outputs={
            "probes_jsonl": str(probes_path),
            "diffs_jsonl": str(diffs_path),
            "report_markdown": str(report_path),
        },
        summary={
            "frames": len(frames),
            "policies": len(args.policies),
            "probes": len(probes),
            "diffs": len(diffs),
            "probe_errors": sum(1 for probe in probes if probe.error),
            "skipped_frames": skipped_frames,
        },
    )
    write_manifest(run_dir.manifest_path, manifest)

    print(f"probe_run_dir={run_dir.path}")
    print(f"probes={probes_path}")
    print(f"diffs={diffs_path}")
    print(f"report={report_path}")
    print(f"skipped_frames={skipped_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
