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
from deepmew.trace.semantics_diagnosis import (  # noqa: E402
    DEFAULT_MATCHUPS,
    DEFAULT_POLICIES,
    FIRST_CHOICE_MATCHUP,
    analyze_diagnosis,
    build_diagnosis_from_existing,
    build_diagnosis_report,
    build_probe_outputs,
    run_trace_matchups,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose current-deck attack semantics with deterministic trace probes."
    )
    parser.add_argument("--trace-run", help="Existing trace run directory or frames.jsonl.")
    parser.add_argument("--probe-run", help="Existing probe run directory with probes.jsonl and diffs.jsonl.")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20240623)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_POLICIES), help="Policies to probe.")
    parser.add_argument("--include-first-choice", action="store_true", help="Also trace v2.4[first=yes] vs v2.4.")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default="deck_semantics_diagnosis")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    matchups = list(DEFAULT_MATCHUPS)
    if args.include_first_choice:
        matchups.append(FIRST_CHOICE_MATCHUP)

    if args.trace_run:
        frames, probes, diffs, diagnosis = build_diagnosis_from_existing(
            run_id=run_dir.run_id,
            trace_run=args.trace_run,
            probe_run=args.probe_run,
            policies=args.agents,
            limit_frames=args.limit_frames,
        )
        pair_summaries = []
        recorder_summary = {"frames": len(frames), "skipped_frames": 0, "games": diagnosis["games"]}
        trace_source = args.trace_run
        probe_source = args.probe_run
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
        diagnosis = analyze_diagnosis(
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
    diagnosis_path = run_dir.output_path("diagnosis.json")
    report_path = run_dir.output_path("report.md")

    write_jsonl(frames_path, frames)
    write_jsonl(probes_path, probes)
    write_jsonl(diffs_path, diffs)
    write_json(diagnosis_path, diagnosis)
    write_markdown(report_path, build_diagnosis_report(diagnosis))

    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=run_dir.kind,
        command="python scripts\\diagnose_deck_v3_semantics.py " + " ".join(sys.argv[1:]),
        inputs={
            "trace_run": args.trace_run,
            "probe_run": args.probe_run,
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "limit_frames": args.limit_frames,
            "policies": args.agents,
            "matchups": matchups,
        },
        outputs={
            "frames_jsonl": str(frames_path),
            "probes_jsonl": str(probes_path),
            "diffs_jsonl": str(diffs_path),
            "diagnosis_json": str(diagnosis_path),
            "report_markdown": str(report_path),
        },
        summary={
            **recorder_summary,
            "probes": len(probes),
            "diffs": len(diffs),
            "probe_errors": diagnosis["probe_errors"],
            "semantics_choice_diffs": diagnosis["semantics_on_vs_off"]["different_choice"],
            "first_legal_choice_diffs": diagnosis["first_legal_comparison"]["different_choice"],
        },
    )
    write_manifest(run_dir.manifest_path, manifest)

    print(f"diagnosis_run_dir={run_dir.path}")
    print(f"manifest={run_dir.manifest_path}")
    print(f"diagnosis={diagnosis_path}")
    print(f"report={report_path}")
    print(f"semantics_choice_diffs={diagnosis['semantics_on_vs_off']['different_choice']}")
    print(f"first_legal_choice_diffs={diagnosis['first_legal_comparison']['different_choice']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
