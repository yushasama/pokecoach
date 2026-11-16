"""Compact external-review bundle for BabyMew PLAY identity evidence."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.deck_profiles import (
    CYRANO_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.policy_rules.attachment_timing import classify_attach_timing, extract_attach_timing_facts
from deepmew.state_machine import analyze_board, describe_plan
from deepmew.trace.schema import to_jsonable


RUN_KIND = "babymew_play_identity_review"
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "play_identity_counts.json",
    "attach_skip_play_events.jsonl",
    "attach_skip_play_events.csv",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
    "relevant_paths.txt",
]
CARD_BUCKETS = {
    MEGA_SIGNAL_CARD_ID: "Mega Signal",
    WAITRESS_CARD_ID: "Waitress",
    LILLIES_DETERMINATION_CARD_ID: "Lillie's Determination",
    CYRANO_CARD_ID: "Cyrano",
    MAXIMUM_BELT_CARD_ID: "Maximum Belt",
}


@dataclass(frozen=True)
class HandoffBundleResult:
    bundle_dir: Path
    manifest_path: Path
    files: list[str]
    counts: dict[str, Any]


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_latest_timing_run(root: Path | str = ".research/runs/deck_attachment_timing") -> Path:
    root_path = Path(root)
    candidates = [path for path in root_path.glob("*/*") if path.is_dir() and (path / "timing_audit.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"no timing runs found under {root_path}")
    return max(candidates, key=lambda path: path.name)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def create_bundle_dir(
    *,
    out_root: Path | str = ".research/handoff_bundles",
    kind: str = RUN_KIND,
    run_id: str | None = None,
    now: datetime | None = None,
) -> Path:
    created = now or _utc_now()
    day = created.strftime("%Y%m%d")
    final_run_id = run_id or make_run_id(kind, now=created)
    bundle_dir = Path(out_root) / day / final_run_id
    if bundle_dir.exists():
        raise FileExistsError(f"bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)
    return bundle_dir


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _frame_map(frames: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(frame.get("frame_id") or ""): frame for frame in frames}


def _pokemon_name(summary: dict[str, Any] | None) -> str | None:
    if not isinstance(summary, dict):
        return None
    return summary.get("name") or summary.get("card_name")


def _energy_count(summary: dict[str, Any] | None) -> int | None:
    if not isinstance(summary, dict):
        return None
    return _int_or_none(summary.get("energy_count"))


def _count_bucket(card_id: int | None, card_name: str | None, resolved: bool) -> str:
    if not resolved or card_id is None:
        return "unresolved PLAY"
    return CARD_BUCKETS.get(card_id, "other named PLAY")


def _play_identity_counts(audit: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = audit.get("metrics") or {}
    per_card: Counter[tuple[int, str]] = Counter()
    bucket_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for event in events:
        if event.get("chosen_type") != "PLAY":
            continue
        play = event.get("play_info") or {}
        card_id = _int_or_none(play.get("card_id"))
        card_name = str(play.get("card_name") or card_id or "unknown")
        resolved = bool(play.get("resolved"))
        category_counts[str(play.get("identity_category") or "unresolved_play")] += 1
        bucket_counts[_count_bucket(card_id, card_name, resolved)] += 1
        if resolved and card_id is not None:
            per_card[(card_id, card_name)] += 1
    return {
        "total_attach_legal_frames": metrics.get("attach_legal_frames"),
        "v24_skipped_attach_frames": metrics.get("default_v24_skipped_attach_frames"),
        "v24_skipped_attach_for_play_frames": metrics.get("default_skipped_for_play"),
        "named_setup_play_count": category_counts.get("named_setup_play", 0),
        "named_nonsetup_play_count": category_counts.get("named_nonsetup_play", 0),
        "unresolved_play_count": category_counts.get("unresolved_play", 0),
        "named_bucket_counts": dict(bucket_counts),
        "per_card_counts": [
            {"card_id": card_id, "card_name": card_name, "count": count}
            for (card_id, card_name), count in per_card.most_common()
        ],
    }


def _raw_frame_plan(frame: dict[str, Any]) -> tuple[Any | None, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    raw = frame.get("raw_observation")
    if not isinstance(raw, dict):
        return None, None, None, None
    try:
        plan = analyze_board(raw)
        plan_text = describe_plan(plan)
        select = raw.get("select") if isinstance(raw.get("select"), dict) else {}
        options = select.get("option") or []
        facts = extract_attach_timing_facts(raw, select, options, plan, {"use_card_semantics": True})
        decision = classify_attach_timing(facts)
        return plan, plan_text, to_jsonable(facts), to_jsonable(decision)
    except Exception as exc:
        return None, f"state-machine-error={type(exc).__name__}: {exc}", None, None


def _readiness_rows(plan: Any | None) -> list[dict[str, Any]]:
    if plan is None:
        return []
    return [to_jsonable(item) for item in getattr(plan, "readiness", ()) or ()]


def _event_row(event: dict[str, Any], frame: dict[str, Any] | None) -> dict[str, Any]:
    frame = frame or {}
    play = event.get("play_info") or {}
    board = event.get("board_timing") or {}
    plan_summary = board.get("plan") if isinstance(board.get("plan"), dict) else {}
    board_summary = frame.get("board_summary") or {}
    own_active = board_summary.get("own_active") or (board.get("own_active") if isinstance(board, dict) else {})
    opponent_active = board_summary.get("opponent_active") or (board.get("opponent_active") if isinstance(board, dict) else {})
    attach_targets = board.get("attach_targets") if isinstance(board.get("attach_targets"), dict) else {}
    default = event.get("default_choice") or {}
    first = event.get("first_legal_choice") or {}
    return {
        "frame_id": event.get("frame_id"),
        "game_id": event.get("game_id"),
        "turn": event.get("turn"),
        "decision_index": event.get("decision_index"),
        "agent_label": default.get("policy_spec") or "v2.4",
        "chosen_option_index": event.get("chosen_index"),
        "chosen_option_type": event.get("chosen_type"),
        "resolved": play.get("resolved"),
        "resolve_reason": play.get("resolve_reason"),
        "identity_category": play.get("identity_category"),
        "card_id": play.get("card_id"),
        "card_name": play.get("card_name"),
        "source_area": play.get("source_area"),
        "source_index": play.get("source_index"),
        "opponent_active_hp": _int_or_none((opponent_active or {}).get("hp")),
        "own_active_card_id": (own_active or {}).get("card_id"),
        "own_active_card_name": _pokemon_name(own_active),
        "own_active_energy_count": _energy_count(own_active),
        "bench_summary": board_summary.get("own_bench") or board.get("own_bench"),
        "board_plan_state": plan_summary.get("state"),
        "board_plan": plan_summary,
        "attach_timing_state": None,
        "attach_timing_reason": None,
        "first_legal_chosen_option_type": first.get("chosen_type"),
        "best_attach_target_summary": (attach_targets.get("best_candidates") or [])[:2],
        "final_result": frame.get("final_result"),
        "final_result_reason": frame.get("final_result_reason"),
        "classification_tags": event.get("buckets") or [],
    }


def _snapshot_row(event: dict[str, Any], frame: dict[str, Any] | None) -> dict[str, Any]:
    frame = frame or {}
    plan, plan_text, facts, decision = _raw_frame_plan(frame)
    row = {
        "frame_id": event.get("frame_id"),
        "board_plan": to_jsonable(plan) if plan is not None else None,
        "describe_plan": plan_text,
        "readiness": _readiness_rows(plan),
        "attach_timing_facts": facts,
        "attach_timing_decision": decision,
    }
    return row


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "turn",
        "card_id",
        "card_name",
        "board_plan_state",
        "attach_timing_state",
        "attach_timing_reason",
        "first_legal_chosen_option_type",
        "final_result",
        "classification_tags",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue()


def _readme(timing_run: Path, command: str) -> str:
    return "\n".join(
        [
            "# BabyMew PLAY Identity Review Bundle",
            "",
            "This bundle is compact external-review evidence for ATTACH-legal frames where default BabyMew chose PLAY instead of ATTACH.",
            "",
            f"- source timing run: `{timing_run}`",
            f"- generated by: `{command}`",
            "- question: when BabyMew skipped ATTACH for PLAY, which visible hand card did PLAY actually refer to?",
            "",
            "Inspect first:",
            "",
            "1. `executive_summary.md`",
            "2. `play_identity_counts.json`",
            "3. `attach_skip_play_events.csv`",
            "4. `example_frames.md`",
            "",
        ]
    )


def _executive_summary(counts: dict[str, Any]) -> str:
    bucket = counts.get("named_bucket_counts") or {}
    lines = [
        "# Executive Summary",
        "",
        "1. BabyMew PLAY identity is now resolved from `PLAY.option.index -> current.players[yourIndex].hand[index]` when visible.",
        "2. Counts by requested card bucket:",
    ]
    for label in ["Mega Signal", "Waitress", "Lillie's Determination", "Cyrano", "Maximum Belt", "other named PLAY", "unresolved PLAY"]:
        lines.append(f"   - {label}: {bucket.get(label, 0)}")
    lines.extend(
        [
            "3. The previous generic PLAY conclusion should be treated as instrumentation-limited evidence, not a gameplay conclusion.",
            f"4. Named setup PLAY skips observed: {counts.get('named_setup_play_count', 0)}.",
            f"5. Still unknown: {counts.get('unresolved_play_count', 0)} PLAY skips remained unresolved from visible hand metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def _examples_markdown(rows: list[dict[str, Any]]) -> str:
    named = [row for row in rows if row.get("resolved")]
    unresolved = [row for row in rows if not row.get("resolved")]
    first_attached = [row for row in rows if row.get("first_legal_chosen_option_type") == "ATTACH"]
    selected = (named[:5] + unresolved[:5] + first_attached[:3])[:20]
    lines = ["# Example Frames", ""]
    if not selected:
        lines.append("No ATTACH-skip-for-PLAY examples were available.")
        return "\n".join(lines)
    for row in selected:
        lines.extend(
            [
                f"## {row.get('frame_id')}",
                "",
                f"- turn: {row.get('turn')}",
                f"- chosen PLAY card: {row.get('card_name') or 'unresolved'} ({row.get('card_id')})",
                f"- board plan: {row.get('board_plan_state')}",
                f"- attach timing state: {row.get('attach_timing_state')}",
                f"- attach timing reason: {row.get('attach_timing_reason')}",
                f"- best attach target: {row.get('best_attach_target_summary')}",
                f"- first-legal choice: {row.get('first_legal_chosen_option_type')}",
                f"- final result: {row.get('final_result')} / {row.get('final_result_reason')}",
                f"- interpretation: tags={', '.join(row.get('classification_tags') or [])}",
                "",
            ]
        )
    return "\n".join(lines)


def _relevant_paths(timing_run: Path, bundle_dir: Path) -> str:
    paths = [
        timing_run / "report.md",
        timing_run,
        bundle_dir,
        Path("deepmew/options/identity.py"),
        Path("deepmew/trace/record.py"),
        Path("deepmew/trace/attachment_timing.py"),
        Path("deepmew/trace/handoff_bundle.py"),
        Path("scripts/export_babymew_handoff_bundle.py"),
        Path("tests/test_option_identity.py"),
        Path("tests/test_trace_record.py"),
        Path("tests/test_attachment_timing.py"),
        Path("tests/test_handoff_bundle.py"),
        Path(".research/STATE.md"),
        Path(".research/CURRENT_STATE_COMPACT.md"),
        Path("HANDOFF_SUMMARY.md"),
    ]
    return "\n".join(str(path) for path in paths) + "\n"


def export_babymew_handoff_bundle(
    *,
    timing_run: Path | str,
    out_root: Path | str = ".research/handoff_bundles",
    command: str = "python scripts\\export_babymew_handoff_bundle.py --latest-timing-run",
) -> HandoffBundleResult:
    timing_run = Path(timing_run)
    audit = json.loads((timing_run / "timing_audit.json").read_text(encoding="utf-8"))
    events = load_jsonl(timing_run / "timing_events.jsonl")
    frames = load_jsonl(timing_run / "frames.jsonl")
    frames_by_id = _frame_map(frames)
    play_events = [event for event in events if event.get("chosen_type") == "PLAY"]
    counts = _play_identity_counts(audit, play_events)

    bundle_dir = create_bundle_dir(out_root=out_root)
    event_rows = [_event_row(event, frames_by_id.get(str(event.get("frame_id") or ""))) for event in play_events]
    snapshot_rows = [_snapshot_row(event, frames_by_id.get(str(event.get("frame_id") or ""))) for event in play_events]
    for event_row, snapshot in zip(event_rows, snapshot_rows):
        decision = snapshot.get("attach_timing_decision") or {}
        event_row["attach_timing_state"] = decision.get("state")
        event_row["attach_timing_reason"] = decision.get("reason")

    outputs = {
        "readme": str(bundle_dir / "README.md"),
        "executive_summary": str(bundle_dir / "executive_summary.md"),
        "play_identity_counts": str(bundle_dir / "play_identity_counts.json"),
        "attach_skip_play_events_jsonl": str(bundle_dir / "attach_skip_play_events.jsonl"),
        "attach_skip_play_events_csv": str(bundle_dir / "attach_skip_play_events.csv"),
        "state_machine_snapshots": str(bundle_dir / "state_machine_snapshots.jsonl"),
        "example_frames": str(bundle_dir / "example_frames.md"),
        "relevant_paths": str(bundle_dir / "relevant_paths.txt"),
    }
    write_markdown(bundle_dir / "README.md", _readme(timing_run, command))
    write_markdown(bundle_dir / "executive_summary.md", _executive_summary(counts))
    write_json(bundle_dir / "play_identity_counts.json", counts)
    write_jsonl(bundle_dir / "attach_skip_play_events.jsonl", event_rows)
    write_text(bundle_dir / "attach_skip_play_events.csv", _csv_text(event_rows))
    write_jsonl(bundle_dir / "state_machine_snapshots.jsonl", snapshot_rows)
    write_markdown(bundle_dir / "example_frames.md", _examples_markdown(event_rows))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(timing_run, bundle_dir))

    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=RUN_KIND,
        command=command,
        inputs={"timing_run": str(timing_run)},
        outputs=outputs,
        summary={
            "source_run_id": audit.get("run_id"),
            "attach_legal_frames": counts.get("total_attach_legal_frames"),
            "play_skip_frames": counts.get("v24_skipped_attach_for_play_frames"),
            "named_setup_play_count": counts.get("named_setup_play_count"),
            "named_nonsetup_play_count": counts.get("named_nonsetup_play_count"),
            "unresolved_play_count": counts.get("unresolved_play_count"),
        },
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return HandoffBundleResult(
        bundle_dir=bundle_dir,
        manifest_path=bundle_dir / "manifest.json",
        files=REQUIRED_BUNDLE_FILES,
        counts=counts,
    )
