from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown  # noqa: E402
from deepmew.trace.attach_target_quality import (  # noqa: E402
    DEFAULT_MATCHUPS,
    DEFAULT_POLICIES,
    RUN_KIND,
    SEMANTICS_CONTROL_MATCHUP,
    analyze_target_quality,
    build_probe_outputs,
    build_target_quality_from_existing,
    build_target_quality_report,
    run_trace_matchups,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current-deck ATTACH target quality from deterministic trace probes."
    )
    parser.add_argument("--trace-run", help="Existing trace run directory or frames.jsonl.")
    parser.add_argument("--probe-run", help="Existing probe run directory with probes.jsonl and diffs.jsonl.")
    parser.add_argument("--diagnosis-run", help="Existing diagnosis run directory with frames.jsonl and probes.jsonl.")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20240625)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_POLICIES), help="Policies to probe.")
    parser.add_argument("--skip-ablation-runs", action="store_true", help="Only generate v2.4 vs first-legal traces.")
    parser.add_argument("--include-semantics-control-run", action="store_true", help="Also trace v2.4 vs semantics-off.")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    return parser.parse_args(argv)


def _resolve_reuse_inputs(args: argparse.Namespace) -> tuple[str | None, str | None]:
    trace_run = args.trace_run
    probe_run = args.probe_run
    if args.diagnosis_run:
        if trace_run is None:
            trace_run = args.diagnosis_run
        if probe_run is None:
            probe_run = args.diagnosis_run
    return trace_run, probe_run


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    trace_run, probe_run = _resolve_reuse_inputs(args)

    if args.skip_ablation_runs:
        matchups = [DEFAULT_MATCHUPS[0]]
    else:
        matchups = list(DEFAULT_MATCHUPS)
    if args.include_semantics_control_run:
        matchups.append(SEMANTICS_CONTROL_MATCHUP)

    if trace_run:
        frames, probes, diffs, target_quality, events = build_target_quality_from_existing(
            run_id=run_dir.run_id,
            trace_run=trace_run,
            probe_run=probe_run,
            policies=args.agents,
            limit_frames=args.limit_frames,
        )
        recorder_summary = {"frames": len(frames), "skipped_frames": 0, "games": target_quality["games"]}
        pair_summaries = []
        trace_source = trace_run
        probe_source = probe_run
    else:
        frames, pair_summaries, recorder_summary = run_trace_matchups(
            run_id=run_dir.run_id,
            matchups=matchups,
            games=args.games,
            seed=args.seed,
            max_steps=args.max_steps,
            limit_frames=args.limit_frames,
        )
        probe_objects, diff_objects = build_probe_outputs(frames, args.agents)
        probes = [probe.to_jsonable() for probe in probe_objects]
        diffs = [diff.to_jsonable() for diff in diff_objects]
        trace_source = str(run_dir.output_path("frames.jsonl"))
        probe_source = str(run_dir.output_path("probes.jsonl"))
        target_quality, events = analyze_target_quality(
            run_id=run_dir.run_id,
            frames=frames,
            probes=probes,
            diffs=diffs,
            policies=args.agents,
            pair_summaries=pair_summaries,
            trace_source=trace_source,
            probe_source=probe_source,
        )

    frames_path = run_dir.output_path("frames.jsonl")
    probes_path = run_dir.output_path("probes.jsonl")
    diffs_path = run_dir.output_path("diffs.jsonl")
    events_path = run_dir.output_path("target_events.jsonl")
    quality_path = run_dir.output_path("target_quality.json")
    report_path = run_dir.output_path("report.md")

    write_jsonl(frames_path, frames)
    write_jsonl(probes_path, probes)
    write_jsonl(diffs_path, diffs)
    write_jsonl(events_path, events)
    write_json(quality_path, target_quality)
    write_markdown(report_path, build_target_quality_report(target_quality))

    metrics = target_quality["metrics"]
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=run_dir.kind,
        command="python scripts\\audit_deck_v3_attach_targets.py " + " ".join(sys.argv[1:]),
        inputs={
            "trace_run": args.trace_run,
            "probe_run": args.probe_run,
            "diagnosis_run": args.diagnosis_run,
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "limit_frames": args.limit_frames,
            "policies": args.agents,
            "matchups": matchups,
            "skip_ablation_runs": args.skip_ablation_runs,
            "include_semantics_control_run": args.include_semantics_control_run,
        },
        outputs={
            "frames_jsonl": str(frames_path),
            "probes_jsonl": str(probes_path),
            "diffs_jsonl": str(diffs_path),
            "target_events_jsonl": str(events_path),
            "target_quality_json": str(quality_path),
            "report_markdown": str(report_path),
        },
        summary={
            **recorder_summary,
            "probes": len(probes),
            "diffs": len(diffs),
            "probe_errors": target_quality["probe_errors"],
            "attach_legal_frames": metrics["attach_legal_frames"],
            "target_events": metrics["target_events"],
            "missed_attach_frames_by_default_v24": metrics["missed_attach_frames_by_default_v24"],
            "timing_vs_target_vs_option_order_conclusion": metrics["timing_vs_target_vs_option_order_conclusion"],
        },
    )
    write_manifest(run_dir.manifest_path, manifest)

    print(f"target_quality_run_dir={run_dir.path}")
    print(f"manifest={run_dir.manifest_path}")
    print(f"target_quality={quality_path}")
    print(f"target_events={events_path}")
    print(f"report={report_path}")
    print(f"attach_legal_frames={metrics['attach_legal_frames']}")
    print(f"missed_attach_frames_by_default_v24={metrics['missed_attach_frames_by_default_v24']}")
    print(f"conclusion={metrics['timing_vs_target_vs_option_order_conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
