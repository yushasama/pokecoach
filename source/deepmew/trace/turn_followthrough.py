"""Turn-level PLAY -> ATTACH follow-through audit."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.deck_profiles import (
    CYRANO_CARD_ID,
    KYOGRE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.policy_rules.attachment_timing import classify_attach_timing, extract_attach_timing_facts
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board, describe_plan
from deepmew.trace.schema import to_jsonable


RUN_KIND = "deck_turn_followthrough"
HANDOFF_KIND = "babymew_turn_followthrough_review"
TURN_FOLLOWTHROUGH_SCHEMA_VERSION = "deepmew.deck_turn_followthrough.v1"
TURN_EVENT_SCHEMA_VERSION = "deepmew.turn_followthrough_event.v1"
SEQUENCE_SCHEMA_VERSION = "deepmew.play_attach_sequence.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "deepmew.turn_followthrough_snapshot.v1"

REQUIRED_RUN_FILES = [
    "manifest.json",
    "turn_followthrough.json",
    "turn_events.jsonl",
    "play_attach_sequences.jsonl",
    "play_attach_sequences.csv",
    "state_machine_snapshots.jsonl",
    "report.md",
]
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "play_attach_followthrough_counts.json",
    "play_attach_sequences.jsonl",
    "play_attach_sequences.csv",
    "state_machine_snapshots.jsonl",
    "example_turns.md",
    "relevant_paths.txt",
]

NAMED_PLAY_LABELS = {
    MEGA_SIGNAL_CARD_ID: "Mega Signal",
    WAITRESS_CARD_ID: "Waitress",
    LILLIES_DETERMINATION_CARD_ID: "Lillie's Determination",
    CYRANO_CARD_ID: "Cyrano",
    MAXIMUM_BELT_CARD_ID: "Maximum Belt",
    SNOVER_CARD_ID: "Snover",
    KYOGRE_CARD_ID: "Kyogre",
}
SETUP_OR_DEVELOPMENT_IDS = frozenset(
    {
        MEGA_SIGNAL_CARD_ID,
        WAITRESS_CARD_ID,
        LILLIES_DETERMINATION_CARD_ID,
        CYRANO_CARD_ID,
        SNOVER_CARD_ID,
        KYOGRE_CARD_ID,
    }
)


@dataclass(frozen=True)
class TurnFollowthroughResult:
    run_dir: Path
    manifest_path: Path
    audit: dict[str, Any]
    sequences: list[dict[str, Any]]
    bundle_dir: Path | None = None


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_handoff_dir(
    *,
    out_root: Path | str = ".research/handoff_bundles",
    kind: str = HANDOFF_KIND,
    now: datetime | None = None,
) -> Path:
    created = now or _utc_now()
    day = created.strftime("%Y%m%d")
    bundle_dir = Path(out_root) / day / make_run_id(kind, now=created)
    if bundle_dir.exists():
        raise FileExistsError(f"bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)
    return bundle_dir


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return bool(value)


def _current(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("raw_observation")
    if not isinstance(raw, dict):
        return {}
    current = raw.get("current")
    return current if isinstance(current, dict) else {}


def _select(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("raw_observation")
    if not isinstance(raw, dict):
        return {}
    select = raw.get("select")
    return select if isinstance(select, dict) else {}


def _raw_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    options = _select(frame).get("option") or []
    return [option for option in options if isinstance(option, dict)]


def _energy_attached(frame: dict[str, Any]) -> bool:
    return bool(_current(frame).get("energyAttached"))


def _bench_count(frame: dict[str, Any]) -> int:
    board = frame.get("board_summary") or {}
    bench = board.get("own_bench") or []
    return len(bench) if isinstance(bench, list) else 0


def _turn_key(frame: dict[str, Any]) -> tuple[str, str, int | None, int | None]:
    return (
        str(frame.get("game_id") or ""),
        str(frame.get("agent_label") or ""),
        _int_or_none(frame.get("seat")),
        _int_or_none(frame.get("turn")),
    )


def _chosen(frame: dict[str, Any]) -> dict[str, Any] | None:
    chosen = frame.get("chosen_option_summaries") or []
    return chosen[0] if chosen and isinstance(chosen[0], dict) else None


def _chosen_type(frame: dict[str, Any]) -> str | None:
    chosen = _chosen(frame)
    return str(chosen.get("option_type")) if chosen else None


def _chosen_index(frame: dict[str, Any]) -> int | None:
    indices = frame.get("chosen_indices") or []
    return _int_or_none(indices[0]) if indices else None


def _attach_legal(frame: dict[str, Any]) -> bool:
    board = frame.get("board_summary") or {}
    if board.get("attach_legal"):
        return True
    return any(str(option.get("option_type")) == "ATTACH" for option in frame.get("legal_options") or [])


def _frame_by_id(frames: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(frame.get("frame_id") or ""): frame for frame in frames}


def _events_by_frame(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(event.get("frame_id") or ""): event for event in events}


def _turn_groups(frames: list[dict[str, Any]]) -> dict[tuple[str, str, int | None, int | None], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        groups[_turn_key(frame)].append(frame)
    for group in groups.values():
        group.sort(key=lambda frame: _int_or_none(frame.get("decision_index")) or -1)
    return groups


def _choice_action(frame: dict[str, Any]) -> str:
    choice = _chosen(frame)
    if not choice:
        return "none"
    option_type = str(choice.get("option_type") or "unknown")
    card = choice.get("card_name")
    if card:
        return f"{option_type}:{card}"
    return option_type


def _first_legal_choice(event: dict[str, Any] | None) -> str | None:
    if not event:
        return None
    first = event.get("first_legal_choice") or {}
    return first.get("chosen_type")


def _best_attach_targets(event: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not event:
        return []
    board = event.get("board_timing") or {}
    targets = board.get("attach_targets") if isinstance(board, dict) else {}
    if not isinstance(targets, dict):
        return []
    return list(targets.get("best_candidates") or [])[:3]


def _visible_target_keys(frame: dict[str, Any]) -> set[tuple[str, int | None, int | None]]:
    board = frame.get("board_summary") or {}
    keys: set[tuple[str, int | None, int | None]] = set()
    active = board.get("own_active")
    if isinstance(active, dict):
        keys.add(("ACTIVE", 0, _int_or_none(active.get("card_id"))))
    for index, pokemon in enumerate(board.get("own_bench") or []):
        if isinstance(pokemon, dict):
            keys.add(("BENCH", index, _int_or_none(pokemon.get("card_id"))))
    return keys


def _attach_target_summary(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    choice = _chosen(frame)
    if not choice or choice.get("option_type") != "ATTACH":
        return None
    return {
        "frame_id": frame.get("frame_id"),
        "option_index": _chosen_index(frame),
        "target_area": choice.get("target_area") or choice.get("area"),
        "target_index": _int_or_none(choice.get("target_index")),
        "target_card_id": _int_or_none(choice.get("target_card_id")),
        "target_card_name": choice.get("target_card_name"),
        "summary": choice.get("summary"),
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


def _plan_summary(plan: Any | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "state": getattr(getattr(plan, "state", None), "value", str(getattr(plan, "state", ""))),
        "energy_attached": getattr(plan, "energy_attached", None),
        "legal_attack_available": getattr(plan, "legal_attack_available", None),
        "legal_attach_available": getattr(plan, "legal_attach_available", None),
        "legal_play_available": getattr(plan, "legal_play_available", None),
        "bench_count": getattr(plan, "bench_count", None),
        "active_shortfall": getattr(plan, "active_shortfall", None),
        "best_shortfall": getattr(plan, "best_shortfall", None),
        "closest_area": getattr(plan, "closest_area", None),
        "closest_index": getattr(plan, "closest_index", None),
    }


def _readiness_rows(plan: Any | None) -> list[dict[str, Any]]:
    if plan is None:
        return []
    return [to_jsonable(item) for item in getattr(plan, "readiness", ()) or ()]


def _snapshot_row(frame: dict[str, Any], event: dict[str, Any] | None) -> dict[str, Any]:
    plan, plan_text, facts, decision = _raw_frame_plan(frame)
    return {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "agent_label": frame.get("agent_label"),
        "board_plan": _plan_summary(plan),
        "describe_plan": plan_text,
        "readiness": _readiness_rows(plan),
        "attach_timing_facts": facts,
        "attach_timing_decision": decision,
        "best_attach_targets": _best_attach_targets(event),
    }


def _sequence_base(frame: dict[str, Any], event: dict[str, Any] | None, snapshot: dict[str, Any]) -> dict[str, Any]:
    choice = _chosen(frame) or {}
    play = (event or {}).get("play_info") or {}
    board = frame.get("board_summary") or {}
    active = board.get("own_active") if isinstance(board.get("own_active"), dict) else {}
    decision = snapshot.get("attach_timing_decision") or {}
    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "agent_label": frame.get("agent_label"),
        "seat": frame.get("seat"),
        "chosen_option_index": _chosen_index(frame),
        "chosen_option_type": _chosen_type(frame),
        "card_id": _int_or_none(choice.get("card_id") or play.get("card_id")),
        "card_name": choice.get("card_name") or play.get("card_name"),
        "source_area": choice.get("source_area") or play.get("source_area"),
        "source_index": _int_or_none(choice.get("source_index") if choice.get("source_index") is not None else play.get("source_index")),
        "board_plan_state": (snapshot.get("board_plan") or {}).get("state"),
        "describe_plan": snapshot.get("describe_plan"),
        "energy_attached_at_play": _energy_attached(frame),
        "attach_legal_at_play": _attach_legal(frame),
        "attach_option_mode": "attach_legal_after_manual_used_or_effect_context"
        if _energy_attached(frame)
        else "manual_attach_available",
        "best_attach_target_summary": _best_attach_targets(event),
        "first_legal_same_frame_choice": _first_legal_choice(event),
        "attach_timing_state": decision.get("state"),
        "attach_timing_reason": decision.get("reason"),
        "own_active_card_id": _int_or_none(active.get("card_id")) if isinstance(active, dict) else None,
        "own_active_card_name": active.get("name") if isinstance(active, dict) else None,
        "own_active_energy_count": _int_or_none(active.get("energy_count")) if isinstance(active, dict) else None,
        "bench_summary": board.get("own_bench") or [],
        "final_result": frame.get("final_result"),
        "final_result_reason": frame.get("final_result_reason"),
    }


def _classify_sequence(
    frame: dict[str, Any],
    later_frames: list[dict[str, Any]],
    base: dict[str, Any],
) -> dict[str, Any]:
    tags: list[str] = []
    later_actions: list[dict[str, Any]] = []
    energy_at_play = bool(base["energy_attached_at_play"])
    card_id = _int_or_none(base.get("card_id"))
    card_name = str(base.get("card_name") or "unresolved")
    setup_or_development = card_id in SETUP_OR_DEVELOPMENT_IDS
    visible_before = _visible_target_keys(frame)
    bench_count_at_play = _bench_count(frame)

    later_attach_frame: dict[str, Any] | None = None
    later_attack_frame: dict[str, Any] | None = None
    turn_end_frame: dict[str, Any] | None = None
    effect_energy_frame: dict[str, Any] | None = None

    for later in later_frames:
        action = _chosen_type(later) or "UNKNOWN"
        later_actions.append(
            {
                "frame_id": later.get("frame_id"),
                "decision_index": later.get("decision_index"),
                "action": _choice_action(later),
                "energy_attached": _energy_attached(later),
            }
        )
        if not energy_at_play and later_attach_frame is None and action == "ATTACH":
            later_attach_frame = later
            break
        if not energy_at_play and effect_energy_frame is None and _energy_attached(later):
            effect_energy_frame = later
            later_attach_frame = later
            break
        if later_attack_frame is None and action == "ATTACK":
            later_attack_frame = later
            break
        if turn_end_frame is None and action == "END":
            turn_end_frame = later
            break

    if energy_at_play:
        primary = "already_attached_before_play"
        tags.append(primary)
    elif later_attach_frame is not None:
        target = _attach_target_summary(later_attach_frame)
        target_key = None
        if target:
            target_key = (
                str(target.get("target_area") or ""),
                _int_or_none(target.get("target_index")),
                _int_or_none(target.get("target_card_id")),
            )
        created_target = bool(
            target
            and (
                target_key not in visible_before
                or _bench_count(later_attach_frame) > bench_count_at_play
                or card_id in {SNOVER_CARD_ID, KYOGRE_CARD_ID, MEGA_SIGNAL_CARD_ID}
            )
        )
        primary = "play_created_attach_target_then_attached" if created_target else "attached_later_same_turn"
        tags.append(primary)
        tags.append("attached_later_same_turn")
        if effect_energy_frame is not None and _chosen_type(effect_energy_frame) != "ATTACH":
            tags.append("effect_attachment_context_possible")
        if setup_or_development:
            tags.append("setup_play_then_attach")
    elif later_attack_frame is not None:
        primary = "attacked_without_attach_after_play"
        tags.append(primary)
        if setup_or_development:
            tags.append("setup_play_no_attach")
    elif turn_end_frame is not None:
        primary = "ended_turn_without_attach"
        tags.append(primary)
        if setup_or_development:
            tags.append("setup_play_no_attach")
    else:
        primary = "unknown_followthrough"
        tags.append(primary)

    if card_id == WAITRESS_CARD_ID and (
        "effect_attachment_context_possible" in tags or energy_at_play or effect_energy_frame is not None
    ):
        tags.append("waitress_effect_attach_possible")

    later_attach_target = _attach_target_summary(later_attach_frame)
    return {
        "later_same_turn_actions": later_actions,
        "later_attach_frame_id": later_attach_frame.get("frame_id") if later_attach_frame else None,
        "later_attach_target_summary": later_attach_target,
        "later_attack_frame_id": later_attack_frame.get("frame_id") if later_attack_frame else None,
        "turn_end_frame_id": turn_end_frame.get("frame_id") if turn_end_frame else None,
        "followthrough_class": primary,
        "tags": sorted(set(tags)),
        "setup_or_development_play": setup_or_development,
        "play_card_bucket": NAMED_PLAY_LABELS.get(card_id, "other named PLAY" if card_id is not None else "unresolved PLAY"),
        "card_name_for_counts": card_name,
    }


def _is_actual_play_before_attach(frame: dict[str, Any]) -> bool:
    return (
        str(frame.get("agent_label") or "") == "v2.4"
        and _attach_legal(frame)
        and _chosen_type(frame) == "PLAY"
    )


def analyze_turn_followthrough(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    timing_events: list[dict[str, Any]] | None = None,
    source_run: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events_by_frame = _events_by_frame(timing_events or [])
    groups = _turn_groups(frames)
    sequences: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []

    for frame in frames:
        if not _is_actual_play_before_attach(frame):
            continue
        event = events_by_frame.get(str(frame.get("frame_id") or ""))
        snapshot = _snapshot_row(frame, event)
        group = groups.get(_turn_key(frame), [])
        decision_index = _int_or_none(frame.get("decision_index")) or -1
        later = [
            item
            for item in group
            if (_int_or_none(item.get("decision_index")) or -1) > decision_index
        ]
        row = _sequence_base(frame, event, snapshot)
        row.update(_classify_sequence(frame, later, row))
        sequences.append(row)
        snapshots.append(snapshot)

    by_turn: dict[tuple[str, str, int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in sequences:
        by_turn[(str(row.get("game_id")), str(row.get("agent_label")), _int_or_none(row.get("seat")), _int_or_none(row.get("turn")))].append(row)
    for key, rows in by_turn.items():
        game_id, agent_label, seat, turn = key
        turn_rows.append(
            {
                "schema_version": TURN_EVENT_SCHEMA_VERSION,
                "game_id": game_id,
                "agent_label": agent_label,
                "seat": seat,
                "turn": turn,
                "play_before_attach_event_count": len(rows),
                "event_frame_ids": [row.get("frame_id") for row in rows],
                "cards": [row.get("card_name") for row in rows],
                "followthrough_classes": [row.get("followthrough_class") for row in rows],
                "actions_after_first_event": rows[0].get("later_same_turn_actions") if rows else [],
            }
        )

    audit = _build_audit(run_id=run_id, sequences=sequences, turn_rows=turn_rows, source_run=source_run)
    return audit, turn_rows, sequences, snapshots


def _build_audit(
    *,
    run_id: str,
    sequences: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    source_run: str | None,
) -> dict[str, Any]:
    class_counts = Counter(str(row.get("followthrough_class")) for row in sequences)
    tag_counts = Counter(tag for row in sequences for tag in row.get("tags") or [])
    per_card: dict[str, Counter[str]] = defaultdict(Counter)
    for row in sequences:
        card = str(row.get("card_name_for_counts") or row.get("card_name") or "unresolved")
        per_card[card]["event_count"] += 1
        if "attached_later_same_turn" in set(row.get("tags") or []):
            per_card[card]["attached_later_same_turn"] += 1
        if row.get("followthrough_class") in {"ended_turn_without_attach", "attacked_without_attach_after_play"}:
            per_card[card]["no_attach_by_turn_end"] += 1
        if row.get("followthrough_class") == "already_attached_before_play":
            per_card[card]["already_attached_before_play"] += 1
        if row.get("followthrough_class") == "attacked_without_attach_after_play":
            per_card[card]["attacked_without_attach"] += 1
        if row.get("followthrough_class") == "unknown_followthrough":
            per_card[card]["unknown"] += 1

    per_card_rows = [
        {
            "card_name": card,
            "event_count": counts.get("event_count", 0),
            "attached_later_same_turn": counts.get("attached_later_same_turn", 0),
            "no_attach_by_turn_end": counts.get("no_attach_by_turn_end", 0),
            "already_attached_before_play": counts.get("already_attached_before_play", 0),
            "attacked_without_attach": counts.get("attacked_without_attach", 0),
            "unknown": counts.get("unknown", 0),
        }
        for card, counts in sorted(per_card.items(), key=lambda item: (-item[1]["event_count"], item[0]))
    ]

    total = len(sequences)
    attached_later = tag_counts.get("attached_later_same_turn", 0)
    already = class_counts.get("already_attached_before_play", 0)
    no_attach = class_counts.get("ended_turn_without_attach", 0) + class_counts.get("attacked_without_attach_after_play", 0)
    conclusion = _conclusion(total=total, attached_later=attached_later, already=already, no_attach=no_attach)
    metrics = {
        "total_play_before_attach_events": total,
        "turns_with_play_before_attach_events": len(turn_rows),
        "count_by_play_card_name": dict(Counter(str(row.get("card_name_for_counts") or "unresolved") for row in sequences)),
        "attached_later_same_turn_count": attached_later,
        "ended_turn_without_attach_count": class_counts.get("ended_turn_without_attach", 0),
        "already_attached_before_play_count": already,
        "attacked_without_attach_after_play_count": class_counts.get("attacked_without_attach_after_play", 0),
        "unknown_followthrough_count": class_counts.get("unknown_followthrough", 0),
        "setup_play_then_attach_count": tag_counts.get("setup_play_then_attach", 0),
        "setup_play_no_attach_count": tag_counts.get("setup_play_no_attach", 0),
        "waitress_effect_attach_possible_count": tag_counts.get("waitress_effect_attach_possible", 0),
        "play_created_attach_target_then_attached_count": class_counts.get("play_created_attach_target_then_attached", 0),
        "per_card": per_card_rows,
        "class_counts": dict(class_counts),
        "tag_counts": dict(tag_counts),
        "conclusion": conclusion,
    }
    return {
        "schema_version": TURN_FOLLOWTHROUGH_SCHEMA_VERSION,
        "run_id": run_id,
        "source_run": source_run,
        "metrics": metrics,
        "report_answers": _report_answers(metrics),
        "do_not_implement_yet": [
            "Do not promote attach_timing=new from this audit.",
            "Do not implement trainer effects from turn-level labels.",
            "Do not tune scores from this observed follow-through alone.",
            "Do not treat counterfactual same-frame probes as same-turn causal proof.",
            "Do not change deck.csv, cg/, or main.py.",
        ],
    }


def _conclusion(*, total: int, attached_later: int, already: int, no_attach: int) -> str:
    if total == 0:
        return "no observed actual v2.4 PLAY-before-ATTACH events found"
    if already + attached_later >= max(1, total * 2 // 3):
        return "frame-level PLAY instead of ATTACH was often misleading; turn-level follow-through frequently preserves attachment"
    if no_attach >= max(4, attached_later + already):
        return "possible missed attachment windows remain after turn-level follow-through, not causal proof"
    return "mixed turn-level follow-through evidence; revise only with targeted evidence"


def _report_answers(metrics: dict[str, Any]) -> list[str]:
    return [
        "Manual attachment was still available only when energy_attached_at_play was false; already-attached rows are not missed manual windows.",
        f"Observed actual v2.4 PLAY-before-ATTACH events: {metrics['total_play_before_attach_events']}.",
        f"Attached later same turn: {metrics['attached_later_same_turn_count']}.",
        f"Ended without attachment: {metrics['ended_turn_without_attach_count']}; attacked without attach: {metrics['attacked_without_attach_after_play_count']}.",
        "Per-card rows below show which named PLAY cards were followed by ATTACH.",
        "This is turn-level follow-through, not causal proof.",
    ]


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "turn",
        "card_id",
        "card_name",
        "board_plan_state",
        "energy_attached_at_play",
        "attach_timing_state",
        "attach_timing_reason",
        "first_legal_same_frame_choice",
        "later_attach_frame_id",
        "later_attack_frame_id",
        "turn_end_frame_id",
        "followthrough_class",
        "final_result",
        "tags",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: ";".join(row.get(field) or []) if field == "tags" else row.get(field) for field in fields})
    return stream.getvalue()


def build_turn_followthrough_report(audit: dict[str, Any], sequences: list[dict[str, Any]]) -> str:
    metrics = audit.get("metrics") or {}
    lines = [title("Deck Turn Follow-through Audit")]
    lines.append(section("Scope"))
    lines.append(
        bullets(
            [
                "Audit type: turn-level follow-through, not causal proof.",
                "Event definition: actual default BabyMew `v2.4` frame where ATTACH was legal and the chosen option was PLAY.",
                "Manual attachment is distinguished from already-attached or possible effect attachment context.",
                "Frame-level PLAY before ATTACH is not automatically PLAY instead of ATTACH.",
            ]
        )
    )
    lines.append(section("Headline Metrics"))
    lines.append(
        table(
            ["Metric", "Value"],
            [
                ["PLAY-before-ATTACH events", metrics.get("total_play_before_attach_events", 0)],
                ["Attached later same turn", metrics.get("attached_later_same_turn_count", 0)],
                ["Ended turn without attach", metrics.get("ended_turn_without_attach_count", 0)],
                ["Already attached before PLAY", metrics.get("already_attached_before_play_count", 0)],
                ["Attacked without attach after PLAY", metrics.get("attacked_without_attach_after_play_count", 0)],
                ["Unknown follow-through", metrics.get("unknown_followthrough_count", 0)],
                ["Setup PLAY then attach", metrics.get("setup_play_then_attach_count", 0)],
                ["Setup PLAY no attach", metrics.get("setup_play_no_attach_count", 0)],
                ["Waitress effect attach possible", metrics.get("waitress_effect_attach_possible_count", 0)],
            ],
        )
    )
    lines.append(section("Per-card Follow-through"))
    lines.append(
        table(
            [
                "Card",
                "Events",
                "Attached Later",
                "No Attach By Turn End",
                "Already Attached",
                "Attacked Without Attach",
                "Unknown",
            ],
            [
                [
                    row["card_name"],
                    row["event_count"],
                    row["attached_later_same_turn"],
                    row["no_attach_by_turn_end"],
                    row["already_attached_before_play"],
                    row["attacked_without_attach"],
                    row["unknown"],
                ]
                for row in metrics.get("per_card", [])
            ],
        )
    )
    lines.append(section("Questions Answered"))
    lines.append(
        bullets(
            [
                f"When BabyMew skipped ATTACH for PLAY, manual attachment was still available on {metrics.get('total_play_before_attach_events', 0) - metrics.get('already_attached_before_play_count', 0)} observed rows and already used before PLAY on {metrics.get('already_attached_before_play_count', 0)} rows.",
                f"BabyMew attached later in the same turn on {metrics.get('attached_later_same_turn_count', 0)} rows.",
                "Named PLAY cards followed by later ATTACH are shown in the per-card table.",
                "Named PLAY cards ending without attachment are shown by the no-attach and attacked-without-attach columns.",
                "Lillie's-before-ATTACH should be treated as sequencing evidence, not proof of either good sequencing or a missed window without the row context.",
                "Mega Signal before ATTACH is labeled as target-creation follow-through only when visible state suggests a new/improved target before attachment.",
                "Waitress is not treated as a missed attachment when energy was already attached or possible effect attachment context appears.",
                "Snover/Kyogre PLAY rows are development-target evidence when a later attach targets a new or improved line.",
                f"Frame-level skipped-ATTACH evidence becomes: {metrics.get('conclusion')}.",
                "Next implementation candidate should target only repeatable no-attach rows after named non-effect setup PLAY, if a follow-up audit finds them stable.",
                "Do not implement trainer effects, score tuning, deck changes, broad parsing, or default timing promotion yet.",
            ]
        )
    )
    lines.append(section("Example Turns"))
    for row in _example_rows(sequences):
        lines.append(f"### {row.get('frame_id')}\n")
        lines.append(
            bullets(
                [
                    f"turn={row.get('turn')} decision={row.get('decision_index')} card={row.get('card_name')}",
                    f"board_plan={row.get('board_plan_state')} attach_timing={row.get('attach_timing_state')} reason={row.get('attach_timing_reason')}",
                    f"energy_attached_at_play={row.get('energy_attached_at_play')} first_legal={row.get('first_legal_same_frame_choice')}",
                    f"later_actions={row.get('later_same_turn_actions')}",
                    f"later_attach={row.get('later_attach_frame_id')} later_attack={row.get('later_attack_frame_id')} turn_end={row.get('turn_end_frame_id')}",
                    f"followthrough={row.get('followthrough_class')} tags={row.get('tags')}",
                ]
            )
        )
    return "\n".join(lines)


def _example_rows(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    wanted_names = ["Lillie's Determination", "Mega Signal", "Waitress", "Snover", "Kyogre"]
    for name in wanted_names:
        for row in [item for item in rows if item.get("card_name") == name][:3]:
            if row not in selected:
                selected.append(row)
    for klass in ["attached_later_same_turn", "play_created_attach_target_then_attached", "ended_turn_without_attach", "attacked_without_attach_after_play", "already_attached_before_play"]:
        for row in rows:
            if len(selected) >= limit:
                return selected
            if row.get("followthrough_class") == klass and row not in selected:
                selected.append(row)
    return selected[:limit]


def _example_turns_markdown(audit: dict[str, Any], sequences: list[dict[str, Any]]) -> str:
    lines = ["# Example Turns", "", "These are compact observed follow-through examples, not causal proof.", ""]
    for row in _example_rows(sequences):
        lines.extend(
            [
                f"## {row.get('frame_id')}",
                "",
                f"- turn: {row.get('turn')}",
                f"- chosen PLAY card: {row.get('card_name')} ({row.get('card_id')})",
                f"- board plan: {row.get('board_plan_state')}",
                f"- attach timing: {row.get('attach_timing_state')} / {row.get('attach_timing_reason')}",
                f"- best attach target: {row.get('best_attach_target_summary')}",
                f"- later actions: {row.get('later_same_turn_actions')}",
                f"- later attach: {row.get('later_attach_frame_id')} target={row.get('later_attach_target_summary')}",
                f"- final result: {row.get('final_result')} / {row.get('final_result_reason')}",
                f"- interpretation: {row.get('followthrough_class')} tags={row.get('tags')}",
                "",
            ]
        )
    return "\n".join(lines)


def _readme(source_run: str | None, command: str) -> str:
    return "\n".join(
        [
            "# BabyMew Turn Follow-through Review Bundle",
            "",
            "This bundle is compact external-review evidence for observed actual `v2.4` turns where BabyMew chose PLAY while ATTACH was legal.",
            "",
            f"- source run: `{source_run}`",
            f"- generated by: `{command}`",
            "- question: after PLAY before ATTACH, did BabyMew attach later in the same turn?",
            "",
            "Inspect first:",
            "",
            "1. `executive_summary.md`",
            "2. `play_attach_followthrough_counts.json`",
            "3. `play_attach_sequences.csv`",
            "4. `example_turns.md`",
            "",
        ]
    )


def _executive_summary(audit: dict[str, Any]) -> str:
    metrics = audit.get("metrics") or {}
    lines = [
        "# Executive Summary",
        "",
        "This is turn-level follow-through, not causal proof.",
        "",
        f"- PLAY-before-ATTACH events: {metrics.get('total_play_before_attach_events', 0)}",
        f"- attached later same turn: {metrics.get('attached_later_same_turn_count', 0)}",
        f"- ended without attachment: {metrics.get('ended_turn_without_attach_count', 0)}",
        f"- already attached before PLAY: {metrics.get('already_attached_before_play_count', 0)}",
        f"- attacked without attach after PLAY: {metrics.get('attacked_without_attach_after_play_count', 0)}",
        f"- unknown follow-through: {metrics.get('unknown_followthrough_count', 0)}",
        f"- conclusion: {metrics.get('conclusion')}",
        "",
        "Per-card follow-through is in `play_attach_followthrough_counts.json` and `play_attach_sequences.csv`.",
        "",
    ]
    return "\n".join(lines)


def _relevant_paths(source_run: str | None, audit_run: Path, bundle_dir: Path | None) -> str:
    paths = [
        source_run or "",
        audit_run / "report.md",
        audit_run / "turn_followthrough.json",
        audit_run / "play_attach_sequences.jsonl",
        audit_run / "state_machine_snapshots.jsonl",
        bundle_dir or "",
        Path("deepmew/trace/turn_followthrough.py"),
        Path("scripts/audit_deck_v3_turn_followthrough.py"),
        Path("tests/test_turn_followthrough.py"),
        Path(".research/TASKS.md"),
        Path(".research/STATE.md"),
        Path(".research/CURRENT_STATE_COMPACT.md"),
        Path("HANDOFF_SUMMARY.md"),
    ]
    return "\n".join(str(path) for path in paths if str(path)) + "\n"


def write_turn_followthrough_outputs(
    *,
    run_dir: Path,
    run_id: str,
    command: str,
    source_run: str | None,
    frames: list[dict[str, Any]],
    timing_events: list[dict[str, Any]],
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> TurnFollowthroughResult:
    audit, turn_rows, sequences, snapshots = analyze_turn_followthrough(
        run_id=run_id,
        frames=frames,
        timing_events=timing_events,
        source_run=source_run,
    )
    run_path = Path(run_dir)
    outputs = {
        "turn_followthrough_json": str(run_path / "turn_followthrough.json"),
        "turn_events_jsonl": str(run_path / "turn_events.jsonl"),
        "play_attach_sequences_jsonl": str(run_path / "play_attach_sequences.jsonl"),
        "play_attach_sequences_csv": str(run_path / "play_attach_sequences.csv"),
        "state_machine_snapshots_jsonl": str(run_path / "state_machine_snapshots.jsonl"),
        "report_markdown": str(run_path / "report.md"),
    }
    write_json(run_path / "turn_followthrough.json", audit)
    write_jsonl(run_path / "turn_events.jsonl", turn_rows)
    write_jsonl(run_path / "play_attach_sequences.jsonl", sequences)
    write_text(run_path / "play_attach_sequences.csv", _csv_text(sequences))
    write_jsonl(run_path / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(run_path / "report.md", build_turn_followthrough_report(audit, sequences))

    bundle_dir = None
    if export_handoff_bundle:
        bundle_dir = export_turn_followthrough_bundle(
            audit=audit,
            sequences=sequences,
            snapshots=snapshots,
            source_run=source_run,
            audit_run=run_path,
            command=command,
            out_root=handoff_root,
        )
        outputs["handoff_bundle"] = str(bundle_dir)

    manifest = build_manifest(
        run_id=run_id,
        kind=RUN_KIND,
        command=command,
        inputs={"source_run": source_run},
        outputs=outputs,
        summary=audit.get("metrics") or {},
    )
    write_manifest(run_path / "manifest.json", manifest)
    return TurnFollowthroughResult(run_dir=run_path, manifest_path=run_path / "manifest.json", audit=audit, sequences=sequences, bundle_dir=bundle_dir)


def export_turn_followthrough_bundle(
    *,
    audit: dict[str, Any],
    sequences: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    source_run: str | None,
    audit_run: Path,
    command: str,
    out_root: Path | str = ".research/handoff_bundles",
) -> Path:
    bundle_dir = _create_handoff_dir(out_root=out_root)
    counts = audit.get("metrics") or {}
    write_markdown(bundle_dir / "README.md", _readme(source_run, command))
    write_markdown(bundle_dir / "executive_summary.md", _executive_summary(audit))
    write_json(bundle_dir / "play_attach_followthrough_counts.json", counts)
    write_jsonl(bundle_dir / "play_attach_sequences.jsonl", sequences)
    write_text(bundle_dir / "play_attach_sequences.csv", _csv_text(sequences))
    write_jsonl(bundle_dir / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(bundle_dir / "example_turns.md", _example_turns_markdown(audit, sequences))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(source_run, audit_run, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=HANDOFF_KIND,
        command=command,
        inputs={"turn_followthrough_run": str(audit_run), "source_run": source_run},
        outputs={name: str(bundle_dir / name) for name in REQUIRED_BUNDLE_FILES if name != "manifest.json"},
        summary={
            "source_run": source_run,
            "turn_followthrough_run": str(audit_run),
            **counts,
        },
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def load_source_run(source_run: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = Path(source_run)
    if not source.is_dir():
        raise FileNotFoundError(f"source run must be a directory: {source}")
    frames_path = source / "frames.jsonl"
    events_path = source / "timing_events.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"missing frames.jsonl in source run: {source}")
    if not events_path.exists():
        raise FileNotFoundError(f"missing timing_events.jsonl in source run: {source}")
    return load_jsonl(frames_path), load_jsonl(events_path)
