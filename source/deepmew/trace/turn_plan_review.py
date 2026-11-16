"""Review artifacts for the experimental V4 turn-plan policy."""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cg.api import OptionType, SelectContext, SelectType
from deepmew.arena import run_pair
from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.options.identity import resolve_option_identity
from deepmew.policy import choose_action
from deepmew.policy.enums import enum_name
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.trace.record import TraceRecorder
from deepmew.turn_plan import plan_first_action, plan_selection_action, selection_option_summaries


_PRESSURE_RANK = {"ko": 3, "meaningful": 2, "weak": 1, "unknown": 0}
V4_1_REFERENCE = {
    "run": ".research/runs/v4_turn_plan/20260622/20260622T100312Z-v4_turn_plan-ae9b5e73/",
    "fallback_rate": 0.5795847750865052,
    "planned_count": 243,
    "fallback_count": 335,
    "note": "V4.1 saved schema lacks select context counts; V4.2 adds them.",
}


RUN_KIND = "v4_turn_plan"
BUNDLE_KIND = "babymew_v4_turn_plan_review"
REQUIRED_RUN_FILES = [
    "manifest.json",
    "report.md",
    "turn_plan_summary.json",
    "turn_plan_events.jsonl",
    "turn_plan_events.csv",
    "example_frames.md",
]
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "turn_plan_summary.json",
    "turn_plan_events.jsonl",
    "turn_plan_events.csv",
    "example_frames.md",
    "relevant_paths.txt",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_bundle_dir(out_root: Path | str = ".research/handoff_bundles") -> Path:
    now = _utc_now()
    bundle_dir = Path(out_root) / now.strftime("%Y%m%d") / make_run_id(BUNDLE_KIND, now=now)
    if bundle_dir.exists():
        raise FileExistsError(f"bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)
    return bundle_dir


def collect_turn_plan_frames(*, run_id: str, games: int, seed: int, max_steps: int = 5000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recorder = TraceRecorder(run_id=run_id)
    pair = run_pair("v4", "v2.4", games=games, seed=seed, max_steps=max_steps, trace_recorder=recorder, game_id_prefix="v4review")
    frames = [frame.to_jsonable() for frame in recorder.frames]
    return frames, {
        "pair": {
            "agent_a": pair.agent_a,
            "agent_b": pair.agent_b,
            "games": pair.games,
            "completed": pair.completed,
            "a_wins": pair.a_wins,
            "b_wins": pair.b_wins,
            "draws": pair.draws,
            "crashes": pair.crashes,
            "timeouts": pair.timeouts,
            "avg_turns": pair.avg_turns,
            "elapsed_seconds": pair.elapsed_seconds,
        },
        "recorder": recorder.summary(),
    }


def _describe_option(raw: dict[str, Any], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    options = (raw.get("select") or {}).get("option") or []
    if index < 0 or index >= len(options):
        return {"index": index, "option_type": "OUT_OF_RANGE", "card_name": None}
    option = options[index]
    identity = resolve_option_identity(raw, option)
    return {
        "index": index,
        "option_type": enum_name(OptionType, option.get("type")),
        "card_name": identity.card_name,
        "attack_id": identity.attack_id,
    }


def _v3_fallback_action(raw: dict[str, Any]) -> dict[str, Any] | None:
    try:
        chosen = choose_action(raw)
    except Exception as error:  # diagnostics must never crash the review
        return {"error": type(error).__name__}
    index = chosen[0] if chosen else None
    described = _describe_option(raw, index)
    if described is None:
        return {"selected_indices": list(chosen)}
    described["selected_indices"] = list(chosen)
    return described


def _play_card_names(facts: Any) -> list[str | None]:
    return [fact.card_name for fact in facts.legal_options if fact.option_type == "PLAY"]


def _evolve_summary(facts: Any) -> list[dict[str, Any]]:
    return [
        {"index": fact.index, "card_name": fact.card_name, "important_mega_evolve": fact.important_mega_evolve}
        for fact in facts.legal_options
        if fact.option_type == "EVOLVE"
    ]


def _attach_summary(facts: Any) -> list[dict[str, Any]]:
    return [
        {
            "index": fact.index,
            "improves_plan": fact.improves_plan,
            "enables_attack": fact.enables_attack,
            "target_is_closest": fact.target_is_closest,
        }
        for fact in facts.legal_options
        if fact.option_type == "ATTACH"
    ]


def _best_attack_pressure(facts: Any) -> dict[str, Any] | None:
    attacks = [fact for fact in facts.legal_options if fact.option_type == "ATTACK"]
    if not attacks:
        return None
    best = max(attacks, key=lambda fact: (_PRESSURE_RANK.get(fact.attack_pressure or "unknown", 0), fact.attack_damage or -1))
    return {"index": best.index, "pressure": best.attack_pressure, "damage": best.attack_damage}


def _turn_option_summaries(facts: Any) -> list[dict[str, Any]]:
    return [
        {
            "index": fact.index,
            "option_type": fact.option_type,
            "card_id": fact.card_id,
            "card_name": fact.card_name,
            "attack_id": fact.attack_id,
            "attack_pressure": fact.attack_pressure,
            "attack_damage": fact.attack_damage,
            "improves_plan": fact.improves_plan,
            "enables_attack": fact.enables_attack,
            "target_is_closest": fact.target_is_closest,
            "important_mega_evolve": fact.important_mega_evolve,
            "target_creating_play": fact.target_creating_play,
            "setup_search_play": fact.setup_search_play,
        }
        for fact in facts.legal_options
    ]


def _event_from_frame(frame: dict[str, Any]) -> dict[str, Any] | None:
    raw = frame.get("raw_observation")
    if not isinstance(raw, dict) or raw.get("select") is None:
        return None
    select = raw.get("select") or {}
    select_type = enum_name(SelectType, select.get("type"))
    context = enum_name(SelectContext, select.get("context"))
    is_main = select_type == "MAIN" and context == "MAIN"
    if is_main:
        decision = plan_first_action(raw)
        facts = decision.facts
        intent = decision.intent.value
        selection_state = None
        option_summaries = _turn_option_summaries(facts)
    else:
        decision = plan_selection_action(raw)
        facts = decision.facts
        intent = decision.state.value
        selection_state = decision.state.value
        option_summaries = selection_option_summaries(facts)
    return {
        "schema_version": "deepmew.v4_turn_plan_event.v3",
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "source_agent": frame.get("agent_label"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "planning_surface": "MAIN" if is_main else "NON_MAIN",
        "select_type": select_type,
        "context": context,
        "actual_selected_indices": frame.get("chosen_indices") or [],
        "planned_selected_indices": list(decision.selected_indices),
        "intent": intent,
        "selection_state": selection_state,
        "rule_name": decision.rule_name,
        "fallback": decision.fallback,
        "reason": decision.reason,
        "board_state": getattr(facts, "board_state", None),
        "active_card_id": facts.active_card_id,
        "active_card_name": facts.active_card_name,
        "bench_count": getattr(facts, "bench_count", None),
        "energy_attached": getattr(facts, "energy_attached", None),
        "primary_mega_in_play": getattr(facts, "primary_mega_in_play", None),
        "primary_mega_active": getattr(facts, "primary_mega_active", None),
        "primary_plan_online": getattr(facts, "primary_plan_online", None),
        "primary_plan_incomplete": getattr(facts, "primary_plan_incomplete", None),
        "weak_attack_only": getattr(facts, "weak_attack_only", None),
        "meaningful_attack_available": getattr(facts, "meaningful_attack_available", None),
        "ko_or_prize_attack_available": getattr(facts, "ko_or_prize_attack_available", None),
        "play_card_names": _play_card_names(facts) if is_main else [],
        "evolve_summary": _evolve_summary(facts) if is_main else [],
        "attach_summary": _attach_summary(facts) if is_main else [],
        "option_summaries": option_summaries,
        "best_attack_pressure": _best_attack_pressure(facts) if is_main else None,
        "v3_fallback_action": _v3_fallback_action(raw) if decision.fallback else None,
        "ko_attack_indices": list(getattr(facts, "ko_attack_indices", ())),
        "prize_pressure_attack_indices": list(getattr(facts, "prize_pressure_attack_indices", ())),
        "meaningful_attack_indices": list(getattr(facts, "meaningful_attack_indices", ())),
        "weak_attack_indices": list(getattr(facts, "weak_attack_indices", ())),
        "setup_search_play_indices": list(getattr(facts, "setup_search_play_indices", ())),
        "target_creating_play_indices": list(getattr(facts, "target_creating_play_indices", ())),
        "important_evolve_indices": list(getattr(facts, "important_evolve_indices", ())),
        "attach_to_plan_indices": list(getattr(facts, "attach_to_plan_indices", ())),
        "important_evolve_targets": list(getattr(facts, "important_evolve_targets", ())),
        "attach_plan_targets": list(getattr(facts, "attach_plan_targets", ())),
        "visible_plan_card_indices": list(getattr(facts, "visible_plan_card_indices", ())),
    }


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_count = sum(1 for event in events if event["fallback"])
    main_events = [event for event in events if event.get("planning_surface") == "MAIN"]
    non_main_events = [event for event in events if event.get("planning_surface") == "NON_MAIN"]
    planned_events = [event for event in events if not event["fallback"]]
    fallback_events = [event for event in events if event["fallback"]]
    return {
        "main_decision_events": len(events),
        "planned_count": len(events) - fallback_count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(events) if events else 0.0,
        "main_frames": len(main_events),
        "non_main_frames": len(non_main_events),
        "planned_main_frames": sum(1 for event in main_events if not event["fallback"]),
        "planned_non_main_frames": sum(1 for event in non_main_events if not event["fallback"]),
        "fallback_main_frames": sum(1 for event in main_events if event["fallback"]),
        "fallback_non_main_frames": sum(1 for event in non_main_events if event["fallback"]),
        "intent_counts": dict(Counter(event["intent"] for event in events)),
        "rule_counts": dict(Counter(event["rule_name"] for event in events)),
        "planned_intent_counts": dict(Counter(event["intent"] for event in planned_events)),
        "fallback_reason_counts": dict(Counter(event["reason"] for event in fallback_events)),
        "fallback_context_counts": dict(Counter(f"{event['select_type']}/{event['context']}" for event in fallback_events)),
        "planned_context_counts": dict(Counter(f"{event['select_type']}/{event['context']}" for event in planned_events)),
        "handler_usage_counts": dict(Counter(event["rule_name"] for event in planned_events if event.get("planning_surface") == "NON_MAIN")),
    }


def _main_only_baseline_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: list[dict[str, Any]] = []
    for event in events:
        if event.get("planning_surface") == "MAIN":
            baseline.append(event)
            continue
        old_event = dict(event)
        old_event.update(
            {
                "planned_selected_indices": [],
                "intent": "FALLBACK",
                "selection_state": None,
                "rule_name": "fallback",
                "fallback": True,
                "reason": "v4.1-no-non-main-selection-handler",
            }
        )
        baseline.append(old_event)
    return baseline


def analyze_turn_plan_frames(frames: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = [event for frame in frames if (event := _event_from_frame(frame)) is not None]
    source_agents = sorted({event["source_agent"] for event in events if event["source_agent"] is not None})
    by_source_agent = {
        agent: _summarize_events([event for event in events if event["source_agent"] == agent])
        for agent in source_agents
    }
    main_only_baseline = _summarize_events(_main_only_baseline_events(events))
    summary = {
        "schema_version": "deepmew.v4_turn_plan_summary.v3",
        "total_frames": len(frames),
        "baseline_reference": V4_1_REFERENCE,
        "main_only_baseline": main_only_baseline,
        **_summarize_events(events),
        "source_agents": source_agents,
        "by_source_agent": by_source_agent,
    }
    return summary, events


def _csv_text(events: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "game_id",
        "source_agent",
        "turn",
        "decision_index",
        "planning_surface",
        "select_type",
        "context",
        "intent",
        "selection_state",
        "rule_name",
        "fallback",
        "reason",
        "actual_selected_indices",
        "planned_selected_indices",
        "board_state",
        "active_card_id",
        "active_card_name",
        "bench_count",
        "primary_mega_active",
        "primary_plan_online",
        "weak_attack_only",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for event in events:
        writer.writerow({field: event.get(field) for field in fields})
    return stream.getvalue()


def _report(summary: dict[str, Any], run_meta: dict[str, Any]) -> str:
    fallback_contexts = Counter(summary.get("fallback_context_counts") or {}).most_common(12)
    planned_contexts = Counter(summary.get("planned_context_counts") or {}).most_common(12)
    handler_usage = Counter(summary.get("handler_usage_counts") or {}).most_common()
    baseline = summary.get("baseline_reference") or {}
    main_only = summary.get("main_only_baseline") or {}
    main_only_fallback_contexts = Counter(main_only.get("fallback_context_counts") or {}).most_common(12)
    baseline_non_main_total = main_only.get("non_main_frames") or 0
    baseline_non_main_fallback = main_only.get("fallback_non_main_frames") or 0
    after_non_main_total = summary.get("non_main_frames") or 0
    after_non_main_fallback = summary.get("fallback_non_main_frames") or 0
    baseline_non_main_rate = baseline_non_main_fallback / baseline_non_main_total if baseline_non_main_total else 0.0
    after_non_main_rate = after_non_main_fallback / after_non_main_total if after_non_main_total else 0.0
    return "\n".join(
        [
            title("DeepMew V4 Turn Plan Review"),
            section("Executive Summary"),
            bullets(
                [
                    "Experimental V4 planner only; default v2.4 is unchanged.",
                    f"Pair smoke: {run_meta.get('pair')}",
                    f"Frames={summary['total_frames']} plan_events={summary['main_decision_events']} planned={summary['planned_count']} fallback={summary['fallback_count']} fallback_rate={summary['fallback_rate']:.3f}.",
                    f"V4.1 reference fallback_rate={baseline.get('fallback_rate'):.3f} from {baseline.get('run')}" if baseline.get("fallback_rate") is not None else "No comparable V4.1 fallback reference recorded.",
                    f"MAIN planned/fallback={summary['planned_main_frames']}/{summary['fallback_main_frames']}; non-MAIN planned/fallback={summary['planned_non_main_frames']}/{summary['fallback_non_main_frames']}.",
                ]
            ),
            section("Before/After Fallback"),
            table(
                ["Metric", "V4.1 main-only baseline", "V4.2 selection-aware"],
                [
                    ("overall fallback rate", f"{main_only.get('fallback_rate', 0.0):.3f}", f"{summary['fallback_rate']:.3f}"),
                    ("overall fallback count", main_only.get("fallback_count", 0), summary["fallback_count"]),
                    ("non-MAIN fallback rate", f"{baseline_non_main_rate:.3f}", f"{after_non_main_rate:.3f}"),
                    ("non-MAIN fallback count", baseline_non_main_fallback, after_non_main_fallback),
                    ("planned non-MAIN frames", main_only.get("planned_non_main_frames", 0), summary["planned_non_main_frames"]),
                ],
            ),
            section("By Source Agent"),
            table(
                ["Source Agent", "Events", "Planned", "Fallback", "Fallback Rate", "MAIN P/F", "non-MAIN P/F"],
                [
                    (
                        agent,
                        stats["main_decision_events"],
                        stats["planned_count"],
                        stats["fallback_count"],
                        f"{stats['fallback_rate']:.3f}",
                        f"{stats['planned_main_frames']}/{stats['fallback_main_frames']}",
                        f"{stats['planned_non_main_frames']}/{stats['fallback_non_main_frames']}",
                    )
                    for agent, stats in (summary.get("by_source_agent") or {}).items()
                ],
            ),
            section("MAIN vs non-MAIN Split"),
            table(
                ["Bucket", "Count"],
                [
                    ("MAIN frames", summary["main_frames"]),
                    ("non-MAIN frames", summary["non_main_frames"]),
                    ("planned MAIN frames", summary["planned_main_frames"]),
                    ("planned non-MAIN frames", summary["planned_non_main_frames"]),
                    ("fallback MAIN frames", summary["fallback_main_frames"]),
                    ("fallback non-MAIN frames", summary["fallback_non_main_frames"]),
                ],
            ),
            section("Top Fallback Contexts"),
            table(["Context", "Count"], fallback_contexts),
            section("Top Baseline Fallback Contexts"),
            table(["Context", "Count"], main_only_fallback_contexts),
            section("Top Planned Contexts"),
            table(["Context", "Count"], planned_contexts),
            section("Handler Usage"),
            table(["Handler", "Count"], handler_usage),
            section("Intent Counts"),
            table(["Intent", "Count"], Counter(summary.get("intent_counts") or {}).most_common()),
            section("Rule Counts"),
            table(["Rule", "Count"], Counter(summary.get("rule_counts") or {}).most_common()),
            section("Fallback Reasons"),
            table(["Reason", "Count"], Counter(summary.get("fallback_reason_counts") or {}).most_common()),
            section("What Not To Change"),
            bullets(
                [
                    "Do not promote V4 to default from this review.",
                    "Do not promote attach_timing=new or evolve_priority=new.",
                    "Do not change deck.csv, cg/, root main.py, trainer semantics, broad parser behavior, ML/search/belief/replay, or default v2.4.",
                ]
            ),
        ]
    )


def _example_block(event: dict[str, Any]) -> list[str]:
    return [
        f"## {event.get('frame_id')}",
        "",
        f"- source_agent: {event.get('source_agent')}",
        f"- selection: {event.get('select_type')}/{event.get('context')} surface={event.get('planning_surface')}",
        f"- active_card: {event.get('active_card_name')} (id={event.get('active_card_id')})",
        f"- board_state: {event.get('board_state')}",
        f"- intent_or_state/rule: {event.get('intent')} / {event.get('rule_name')}",
        f"- fallback: {event.get('fallback')} reason={event.get('reason')}",
        f"- planned_selected_indices: {event.get('planned_selected_indices')} actual_selected_indices={event.get('actual_selected_indices')}",
        f"- primary_plan_online: {event.get('primary_plan_online')} "
        f"(mega_in_play={event.get('primary_mega_in_play')} mega_active={event.get('primary_mega_active')})",
        f"- weak_attack_only: {event.get('weak_attack_only')} "
        f"meaningful_attack_available={event.get('meaningful_attack_available')} "
        f"ko_or_prize_attack_available={event.get('ko_or_prize_attack_available')}",
        f"- hand_play_cards: {event.get('play_card_names')}",
        f"- legal_evolve: {event.get('evolve_summary')}",
        f"- legal_attach: {event.get('attach_summary')}",
        f"- option_summaries: {event.get('option_summaries')}",
        f"- best_attack_pressure: {event.get('best_attack_pressure')}",
        f"- v3_fallback_action: {event.get('v3_fallback_action')}",
        "",
    ]


def _examples(events: list[dict[str, Any]]) -> str:
    fallback_events = [event for event in events if event.get("fallback")]
    planned_events = [event for event in events if not event.get("fallback")]
    lines = ["# Example Frames", "", "## Fallback Examples", ""]
    for event in fallback_events[:30]:
        lines.extend(_example_block(event))
    lines.extend(["## Planned Examples", ""])
    for event in planned_events[:10]:
        lines.extend(_example_block(event))
    return "\n".join(lines)


def _bundle_readme(command: str) -> str:
    return "\n".join(
        [
            "# BabyMew V4 Turn Plan Review Bundle",
            "",
            "Contains the experimental V4 turn-level planner smoke review.",
            "",
            f"- generated by: `{command}`",
            "- default v2.4 is unchanged.",
            "- V4 is arena-only and experimental.",
            "",
        ]
    )


def _relevant_paths(run_dir: Path, bundle_dir: Path | None) -> str:
    paths = [
        run_dir / "report.md",
        run_dir / "turn_plan_summary.json",
        run_dir / "turn_plan_events.jsonl",
        bundle_dir or "",
        Path("deepmew/turn_plan/planner.py"),
        Path("deepmew/turn_plan/policy.py"),
        Path("deepmew/turn_plan/selection.py"),
        Path("deepmew/trace/turn_plan_review.py"),
        Path("scripts/audit_v4_turn_plan.py"),
        Path("tests/test_turn_plan.py"),
    ]
    return "\n".join(str(path) for path in paths if str(path)) + "\n"


def export_turn_plan_bundle(
    *,
    run_dir: Path,
    command: str,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    out_root: Path | str = ".research/handoff_bundles",
) -> Path:
    bundle_dir = _create_bundle_dir(out_root)
    write_markdown(bundle_dir / "README.md", _bundle_readme(command))
    write_markdown(bundle_dir / "executive_summary.md", _report(summary, {}))
    write_json(bundle_dir / "turn_plan_summary.json", summary)
    write_jsonl(bundle_dir / "turn_plan_events.jsonl", events)
    write_text(bundle_dir / "turn_plan_events.csv", _csv_text(events))
    write_markdown(bundle_dir / "example_frames.md", _examples(events))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(run_dir, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=BUNDLE_KIND,
        command=command,
        inputs={"source_run": str(run_dir)},
        outputs={name: str(bundle_dir / name) for name in REQUIRED_BUNDLE_FILES if name != "manifest.json"},
        summary={"planned_count": summary.get("planned_count"), "fallback_count": summary.get("fallback_count")},
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def write_turn_plan_outputs(
    *,
    run_dir: Path,
    run_id: str,
    command: str,
    frames: list[dict[str, Any]],
    run_meta: dict[str, Any],
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> tuple[dict[str, Any], Path | None]:
    summary, events = analyze_turn_plan_frames(frames)
    run_path = Path(run_dir)
    write_json(run_path / "turn_plan_summary.json", summary)
    write_jsonl(run_path / "turn_plan_events.jsonl", events)
    write_text(run_path / "turn_plan_events.csv", _csv_text(events))
    write_markdown(run_path / "example_frames.md", _examples(events))
    write_markdown(run_path / "report.md", _report(summary, run_meta))
    bundle_dir = export_turn_plan_bundle(run_dir=run_path, command=command, summary=summary, events=events, out_root=handoff_root) if export_handoff_bundle else None
    outputs = {
        "report_markdown": str(run_path / "report.md"),
        "turn_plan_summary_json": str(run_path / "turn_plan_summary.json"),
        "turn_plan_events_jsonl": str(run_path / "turn_plan_events.jsonl"),
        "turn_plan_events_csv": str(run_path / "turn_plan_events.csv"),
        "example_frames_markdown": str(run_path / "example_frames.md"),
    }
    if bundle_dir is not None:
        outputs["handoff_bundle"] = str(bundle_dir)
    manifest = build_manifest(
        run_id=run_id,
        kind=RUN_KIND,
        command=command,
        inputs=run_meta,
        outputs=outputs,
        summary={
            "planned_count": summary.get("planned_count"),
            "fallback_count": summary.get("fallback_count"),
            "planned_non_main_frames": summary.get("planned_non_main_frames"),
            "fallback_non_main_frames": summary.get("fallback_non_main_frames"),
        },
    )
    write_manifest(run_path / "manifest.json", manifest)
    return summary, bundle_dir
