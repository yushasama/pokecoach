from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.trace.attachment_timing import (  # noqa: E402
    DEFAULT_MATCHUPS,
    DEFAULT_POLICIES,
    build_probe_outputs,
    run_trace_matchups,
    analyze_attachment_timing,
)
from deepmew.trace.turn_followthrough import (  # noqa: E402
    RUN_KIND,
    load_source_run,
    write_turn_followthrough_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit turn-level PLAY -> ATTACH follow-through for current-deck traces."
    )
    parser.add_argument("--source-run", help="Existing deck_attachment_timing run directory.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20240629)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_POLICIES), help="Policies to probe when tracing.")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    parser.add_argument("--export-handoff-bundle", action="store_true")
    parser.add_argument("--handoff-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def _command() -> str:
    return "python scripts\\audit_deck_v3_turn_followthrough.py " + " ".join(sys.argv[1:])


def _frames_and_events(args: argparse.Namespace, run_id: str) -> tuple[list[dict], list[dict], str | None]:
    if args.source_run:
        frames, timing_events = load_source_run(args.source_run)
        return frames, timing_events, args.source_run

    frames, pair_summaries, _recorder_summary = run_trace_matchups(
        run_id=run_id,
        matchups=list(DEFAULT_MATCHUPS),
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        limit_frames=args.limit_frames,
    )
    probe_objects, diff_objects = build_probe_outputs(frames, args.agents)
    probes = [probe.to_jsonable() for probe in probe_objects]
    diffs = [diff.to_jsonable() for diff in diff_objects]
    _audit, timing_events = analyze_attachment_timing(
        run_id=f"{run_id}-timing-source",
        frames=frames,
        probes=probes,
        diffs=diffs,
        policies=args.agents,
        pair_summaries=pair_summaries,
        trace_source="generated-inside-turn-followthrough",
        probe_source="generated-inside-turn-followthrough",
        target_quality_reference=None,
    )
    return frames, timing_events, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    frames, timing_events, source_run = _frames_and_events(args, run_dir.run_id)
    result = write_turn_followthrough_outputs(
        run_dir=run_dir.path,
        run_id=run_dir.run_id,
        command=_command(),
        source_run=source_run,
        frames=frames,
        timing_events=timing_events,
        export_handoff_bundle=args.export_handoff_bundle,
        handoff_root=args.handoff_root,
    )
    metrics = result.audit["metrics"]
    print(f"turn_followthrough_run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"turn_followthrough={result.run_dir / 'turn_followthrough.json'}")
    print(f"play_attach_sequences={result.run_dir / 'play_attach_sequences.jsonl'}")
    print(f"report={result.run_dir / 'report.md'}")
    if result.bundle_dir is not None:
        print(f"handoff_bundle_dir={result.bundle_dir}")
    print(f"total_play_before_attach_events={metrics['total_play_before_attach_events']}")
    print(f"attached_later_same_turn={metrics['attached_later_same_turn_count']}")
    print(f"ended_turn_without_attach={metrics['ended_turn_without_attach_count']}")
    print(f"already_attached_before_play={metrics['already_attached_before_play_count']}")
    print(f"conclusion={metrics['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
