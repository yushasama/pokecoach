"""Decoupled EVOLVE sequencing audit for current-deck traces."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from cg.api import OptionType

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.deck_profiles import (
    CURRENT_WATER_MEGA_ABOMASNOW,
    KYOGRE_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
)
from deepmew.options import resolve_option_identity
from deepmew.policy import score_candidates
from deepmew.policy.attacks import attack_damage_for_option, opponent_active_hp
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board, describe_plan
from deepmew.trace.schema import to_jsonable


RUN_KIND = "deck_evolve_sequencing"
HANDOFF_KIND = "babymew_evolve_sequencing_review"
EVOLVE_SEQUENCING_SCHEMA_VERSION = "deepmew.deck_evolve_sequencing.v1"
EVOLVE_EVENT_SCHEMA_VERSION = "deepmew.evolve_sequencing_event.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "deepmew.evolve_sequencing_snapshot.v1"

REQUIRED_RUN_FILES = [
    "manifest.json",
    "report.md",
    "evolve_sequencing.json",
    "evolve_events.jsonl",
    "evolve_events.csv",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
]
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "evolve_sequencing_counts.json",
    "evolve_events.csv",
    "evolve_events.jsonl",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
    "relevant_paths.txt",
]

CLASSIFICATION_TAGS = [
    "evolve_taken_same_frame",
    "evolve_later_same_turn",
    "evolve_later_same_game",
    "evolve_never_observed",
    "play_beat_evolve",
    "attack_beat_evolve",
    "attach_beat_evolve",
    "bench_empty",
    "lone_active",
    "weak_attack_ready",
    "meaningful_attack_ready",
    "ko_attack_available",
    "prize_pressure_attack_available",
    "primary_plan_evolve",
    "chosen_play_created_evolve_target",
    "chosen_play_generic_or_nonessential",
    "score_gap_favors_play",
    "score_gap_favors_evolve",
    "unknown",
]

_PRESSURE_RANK = {"unknown": 0, "weak": 1, "meaningful": 2, "prize_pressure": 3, "ko": 4}
_TARGET_CREATING_PLAY_IDS = frozenset({SNOVER_CARD_ID, KYOGRE_CARD_ID})


@dataclass(frozen=True)
class EvolveSequencingResult:
    run_dir: Path
    manifest_path: Path
    audit: dict[str, Any]
    events: list[dict[str, Any]]
    bundle_dir: Path | None = None


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_source_run_frames(source_run: Path | str) -> list[dict[str, Any]]:
    source = Path(source_run)
    if not source.is_dir():
        raise FileNotFoundError(f"source run must be a directory: {source}")
    frames_path = source / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"missing frames.jsonl in source run: {source}")
    return load_jsonl(frames_path)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_handoff_dir(
    *,
    out_root: Path | str = ".research/handoff_bundles",
    kind: str = HANDOFF_KIND,
    now: datetime | None = None,
) -> Path:
    created = now or _utc_now()
    bundle_dir = Path(out_root) / created.strftime("%Y%m%d") / make_run_id(kind, now=created)
    if bundle_dir.exists():
        raise FileExistsError(f"bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)
    return bundle_dir


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mean(values: list[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[int | float]) -> float | None:
    return float(median(values)) if values else None


def _enum_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _raw_observation(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("raw_observation")
    return raw if isinstance(raw, dict) else {}


def _select(frame: dict[str, Any]) -> dict[str, Any]:
    select = _raw_observation(frame).get("select")
    return select if isinstance(select, dict) else {}


def _raw_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    options = _select(frame).get("option") or []
    return [option for option in options if isinstance(option, dict)]


def _legal_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return [option for option in frame.get("legal_options") or [] if isinstance(option, dict)]


def _legal_summary_at(frame: dict[str, Any], index: int | None) -> dict[str, Any]:
    if index is None:
        return {}
    for option in _legal_options(frame):
        if option.get("index") == index:
            return option
    return {}


def _chosen(frame: dict[str, Any]) -> dict[str, Any]:
    chosen = frame.get("chosen_option_summaries") or []
    return chosen[0] if chosen and isinstance(chosen[0], dict) else {}


def _chosen_index(frame: dict[str, Any]) -> int | None:
    indices = frame.get("chosen_indices") or []
    return _int_or_none(indices[0]) if indices else None


def _chosen_type(frame: dict[str, Any]) -> str | None:
    chosen = _chosen(frame)
    if chosen:
        return str(chosen.get("option_type") or "UNKNOWN")
    index = _chosen_index(frame)
    return str(_legal_summary_at(frame, index).get("option_type") or "UNKNOWN") if index is not None else None


def _turn_key(frame: dict[str, Any]) -> tuple[str, str, int | None]:
    return str(frame.get("game_id") or ""), str(frame.get("agent_label") or ""), _int_or_none(frame.get("seat"))


def _turn_groups(frames: list[dict[str, Any]]) -> dict[tuple[str, str, int | None], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        groups[_turn_key(frame)].append(frame)
    for group in groups.values():
        group.sort(key=lambda item: _int_or_none(item.get("decision_index")) or -1)
    return groups


def _is_target_agent_frame(frame: dict[str, Any], agent_label: str) -> bool:
    return str(frame.get("agent_label") or "") == agent_label


def _score_modes(agent_label: str) -> dict[str, str]:
    return {
        "attach_timing": "new" if "attach_timing=new" in agent_label else "old",
        "evolve_priority": "new" if "evolve_priority=new" in agent_label else "old",
    }


def _score_map(frame: dict[str, Any], agent_label: str) -> tuple[dict[int, Any], str | None]:
    raw = _raw_observation(frame)
    if not raw:
        return {}, "missing-raw-observation"
    modes = _score_modes(agent_label)
    try:
        candidates = score_candidates(
            raw,
            use_card_semantics=True,
            attach_timing=modes["attach_timing"],
            evolve_priority=modes["evolve_priority"],
        )
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return {candidate.index: candidate for candidate in candidates}, None


def _score_payload(candidate: Any | None) -> dict[str, Any]:
    if candidate is None:
        return {"score": None, "reasons": []}
    return {"score": candidate.score, "reasons": list(candidate.reasons)}


def _evolve_option_summary(
    frame: dict[str, Any],
    option_index: int,
    candidate_by_index: dict[int, Any],
) -> dict[str, Any]:
    raw = _raw_observation(frame)
    option = _raw_options(frame)[option_index]
    legal = _legal_summary_at(frame, option_index)
    identity = resolve_option_identity(raw, option)
    card_id = identity.card_id if identity.card_id is not None else _int_or_none(legal.get("card_id"))
    target_card_id = (
        identity.target_card_id
        if identity.target_card_id is not None
        else _int_or_none(legal.get("target_card_id"))
    )
    target_area = identity.target_area or legal.get("target_area") or legal.get("area")
    target_index = (
        identity.target_index
        if identity.target_index is not None
        else _int_or_none(legal.get("target_index"))
    )
    source_index = (
        identity.source_index
        if identity.source_index is not None
        else _int_or_none(legal.get("source_index"))
    )
    resolved = bool(identity.resolved or legal.get("resolved")) and card_id is not None and target_card_id is not None
    important = (
        resolved
        and card_id == MEGA_ABOMASNOW_CARD_ID
        and CURRENT_WATER_MEGA_ABOMASNOW.is_mega_line(target_card_id)
    )
    candidate = candidate_by_index.get(option_index)
    score = _score_payload(candidate)
    return {
        "option_index": option_index,
        "card_id": card_id,
        "card_name": identity.card_name or legal.get("card_name"),
        "source_area": identity.source_area or legal.get("source_area"),
        "source_index": source_index,
        "target_area": target_area,
        "target_index": target_index,
        "target_card_id": target_card_id,
        "target_card_name": identity.target_card_name or legal.get("target_card_name"),
        "resolved": resolved,
        "resolve_reason": identity.reason or legal.get("resolve_reason"),
        "important_mega_line": important,
        "score": score["score"],
        "score_reasons": score["reasons"],
        "summary": legal.get("summary"),
    }


def _important_evolve_options(
    frame: dict[str, Any],
    candidate_by_index: dict[int, Any],
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for index, option in enumerate(_raw_options(frame)):
        if _enum_name(OptionType, option.get("type")) != "EVOLVE":
            continue
        summary = _evolve_option_summary(frame, index, candidate_by_index)
        if summary["important_mega_line"]:
            options.append(summary)
    return options


def _best_evolve_option(options: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not options:
        return None
    return max(options, key=lambda item: (item.get("score") is not None, item.get("score") or -999999, -item["option_index"]))


def _choice_summary(frame: dict[str, Any], candidate_by_index: dict[int, Any]) -> dict[str, Any]:
    index = _chosen_index(frame)
    chosen = _chosen(frame)
    raw_options = _raw_options(frame)
    raw_option = raw_options[index] if index is not None and 0 <= index < len(raw_options) else {}
    identity = resolve_option_identity(_raw_observation(frame), raw_option) if raw_option else None
    candidate = candidate_by_index.get(index) if index is not None else None
    score = _score_payload(candidate)
    return {
        "chosen_option_index": index,
        "chosen_option_type": _chosen_type(frame),
        "chosen_card_id": _int_or_none(chosen.get("card_id")) or (identity.card_id if identity else None),
        "chosen_card_name": chosen.get("card_name") or (identity.card_name if identity else None),
        "chosen_attack_id": _int_or_none(chosen.get("attack_id")) or (identity.attack_id if identity else None),
        "chosen_attack_name": chosen.get("attack_name"),
        "chosen_target_area": chosen.get("target_area") or (identity.target_area if identity else None),
        "chosen_target_index": _int_or_none(chosen.get("target_index")) if chosen.get("target_index") is not None else (identity.target_index if identity else None),
        "chosen_target_card_id": _int_or_none(chosen.get("target_card_id")) or (identity.target_card_id if identity else None),
        "chosen_target_card_name": chosen.get("target_card_name") or (identity.target_card_name if identity else None),
        "chosen_score": score["score"],
        "chosen_score_reasons": score["reasons"],
        "chosen_summary": chosen.get("summary"),
    }


def _active_and_bench(frame: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    board = frame.get("board_summary") or {}
    active = board.get("own_active")
    bench = board.get("own_bench") or []
    return active if isinstance(active, dict) else {}, [item for item in bench if isinstance(item, dict)]


def _plan(frame: dict[str, Any]) -> tuple[Any | None, str, list[dict[str, Any]]]:
    raw = _raw_observation(frame)
    if not raw:
        return None, "missing-raw-observation", []
    try:
        plan = analyze_board(raw)
    except Exception as exc:
        return None, f"state-machine-error={type(exc).__name__}: {exc}", []
    return plan, describe_plan(plan), [to_jsonable(item) for item in getattr(plan, "readiness", ()) or ()]


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


def _attack_summaries(frame: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _raw_observation(frame)
    opponent_hp = opponent_active_hp(raw) if raw else None
    rows: list[dict[str, Any]] = []
    for index, option in enumerate(_raw_options(frame)):
        if _enum_name(OptionType, option.get("type")) != "ATTACK":
            continue
        legal = _legal_summary_at(frame, index)
        damage = attack_damage_for_option(option, raw, use_card_semantics=True) if raw else None
        pressure = "unknown"
        if damage is not None:
            if opponent_hp is not None and opponent_hp > 0:
                if damage >= opponent_hp:
                    pressure = "ko"
                elif damage * 2 >= opponent_hp:
                    pressure = "prize_pressure"
                elif damage <= 30:
                    pressure = "weak"
                else:
                    pressure = "meaningful"
            elif damage >= 100:
                pressure = "meaningful"
            elif damage <= 30:
                pressure = "weak"
        rows.append(
            {
                "option_index": index,
                "attack_id": _int_or_none(option.get("attackId")) or _int_or_none(legal.get("attack_id")),
                "attack_name": legal.get("attack_name"),
                "effective_damage": damage,
                "opponent_active_hp": opponent_hp,
                "pressure": pressure,
            }
        )
    rows.sort(key=lambda item: _PRESSURE_RANK.get(str(item.get("pressure")), 0), reverse=True)
    return rows


def _snapshot_row(
    frame: dict[str, Any],
    important_options: list[dict[str, Any]],
    best_attack: dict[str, Any] | None,
) -> dict[str, Any]:
    plan, plan_text, readiness = _plan(frame)
    return {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "agent_label": frame.get("agent_label"),
        "seat": frame.get("seat"),
        "board_plan": _plan_summary(plan),
        "describe_plan": plan_text,
        "readiness": readiness,
        "best_attack": best_attack,
        "important_evolve_options": important_options,
    }


def _chosen_important_evolve(frame: dict[str, Any]) -> dict[str, Any] | None:
    index = _chosen_index(frame)
    if index is None or _chosen_type(frame) != "EVOLVE":
        return None
    options = _important_evolve_options(frame, {})
    for option in options:
        if option["option_index"] == index:
            return option
    return None


def _later_evolve(
    frame: dict[str, Any],
    group: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    current_decision = _int_or_none(frame.get("decision_index")) or -1
    current_turn = _int_or_none(frame.get("turn"))
    same_turn: dict[str, Any] | None = None
    same_game: dict[str, Any] | None = None
    for later in group:
        if (_int_or_none(later.get("decision_index")) or -1) <= current_decision:
            continue
        if _chosen_important_evolve(later) is None:
            continue
        if same_game is None:
            same_game = later
        if current_turn == _int_or_none(later.get("turn")) and same_turn is None:
            same_turn = later
            break
    return same_turn, same_game


def _base_event(
    frame: dict[str, Any],
    important_options: list[dict[str, Any]],
    best_evolve: dict[str, Any],
    choice: dict[str, Any],
    snapshot: dict[str, Any],
    score_error: str | None,
) -> dict[str, Any]:
    active, bench = _active_and_bench(frame)
    attack_rows = _attack_summaries(frame)
    best_attack = attack_rows[0] if attack_rows else None
    score_gap = None
    if choice.get("chosen_score") is not None and best_evolve.get("score") is not None:
        score_gap = choice["chosen_score"] - best_evolve["score"]
    return {
        "schema_version": EVOLVE_EVENT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "agent_label": frame.get("agent_label"),
        "seat": frame.get("seat"),
        "select_type": frame.get("select_type"),
        "select_context": frame.get("select_context"),
        "board_plan_state": (snapshot.get("board_plan") or {}).get("state"),
        "describe_plan": snapshot.get("describe_plan"),
        "own_active": active,
        "own_bench": bench,
        "bench_count": len(bench),
        "energy_attached": (_raw_observation(frame).get("current") or {}).get("energyAttached"),
        "attach_legal": bool((frame.get("board_summary") or {}).get("attach_legal")),
        "play_legal": bool((frame.get("board_summary") or {}).get("play_legal")),
        "attack_legal": bool((frame.get("board_summary") or {}).get("hammer_lanche_legal")) or bool(attack_rows),
        "important_evolve_legal": True,
        "important_evolve_option_count": len(important_options),
        "important_evolve_options": important_options,
        "evolve_option": best_evolve,
        "best_attack": best_attack,
        "attack_options": attack_rows,
        "score_error": score_error,
        "evolve_score": best_evolve.get("score"),
        "evolve_score_reasons": best_evolve.get("score_reasons") or [],
        "score_gap_chosen_minus_evolve": score_gap,
        "final_result": frame.get("final_result"),
        "final_result_reason": frame.get("final_result_reason"),
        **choice,
    }


def _classify_event(
    event: dict[str, Any],
    frame: dict[str, Any],
    same_turn_evolve: dict[str, Any] | None,
    same_game_evolve: dict[str, Any] | None,
) -> dict[str, Any]:
    tags: set[str] = set()
    chosen_type = str(event.get("chosen_option_type") or "UNKNOWN")
    chosen_index = _int_or_none(event.get("chosen_option_index"))
    evolve_indices = {option["option_index"] for option in event["important_evolve_options"]}
    same_frame = chosen_type == "EVOLVE" and chosen_index in evolve_indices
    score_gap = event.get("score_gap_chosen_minus_evolve")
    active, bench = _active_and_bench(frame)
    best_attack = event.get("best_attack") or {}
    best_pressure = str(best_attack.get("pressure") or "unknown")

    if same_frame:
        primary = "evolve_taken_same_frame"
        tags.add(primary)
    elif same_turn_evolve is not None:
        primary = "evolve_later_same_turn"
        tags.add(primary)
    elif same_game_evolve is not None:
        primary = "evolve_later_same_game"
        tags.add(primary)
    else:
        primary = "evolve_never_observed"
        tags.add(primary)

    if not same_frame:
        if chosen_type == "PLAY":
            tags.add("play_beat_evolve")
        elif chosen_type == "ATTACK":
            tags.add("attack_beat_evolve")
        elif chosen_type == "ATTACH":
            tags.add("attach_beat_evolve")

    if not bench:
        tags.add("bench_empty")
        if active:
            tags.add("lone_active")

    if best_pressure == "weak":
        tags.add("weak_attack_ready")
    if best_pressure in {"meaningful", "prize_pressure", "ko"}:
        tags.add("meaningful_attack_ready")
    if any(row.get("pressure") == "ko" for row in event.get("attack_options") or []):
        tags.add("ko_attack_available")
    if any(row.get("pressure") == "prize_pressure" for row in event.get("attack_options") or []):
        tags.add("prize_pressure_attack_available")

    plan = event.get("evolve_option") or {}
    if plan.get("card_id") == MEGA_ABOMASNOW_CARD_ID and CURRENT_WATER_MEGA_ABOMASNOW.is_mega_line(
        _int_or_none(plan.get("target_card_id"))
    ):
        tags.add("primary_plan_evolve")

    if chosen_type == "PLAY":
        if (
            event.get("chosen_card_id") in _TARGET_CREATING_PLAY_IDS
            and (same_turn_evolve is not None or same_game_evolve is not None)
        ):
            tags.add("chosen_play_created_evolve_target")
        else:
            tags.add("chosen_play_generic_or_nonessential")

    if score_gap is None or event.get("score_error"):
        tags.add("unknown")
    elif chosen_type == "PLAY" and score_gap > 0:
        tags.add("score_gap_favors_play")
    elif score_gap < 0:
        tags.add("score_gap_favors_evolve")

    later = same_turn_evolve or same_game_evolve
    later_decision = _int_or_none(later.get("decision_index")) if later else None
    later_turn = _int_or_none(later.get("turn")) if later else None
    decision_index = _int_or_none(frame.get("decision_index"))
    turn = _int_or_none(frame.get("turn"))
    return {
        "classification": primary,
        "tags": [tag for tag in CLASSIFICATION_TAGS if tag in tags],
        "later_evolve_frame_id": later.get("frame_id") if later else None,
        "later_evolve_turn": later_turn,
        "later_evolve_decision_index": later_decision,
        "delay_decisions": None if later_decision is None or decision_index is None else later_decision - decision_index,
        "delay_turns": None if later_turn is None or turn is None else later_turn - turn,
    }


def analyze_evolve_sequencing(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    source_run: str | None = None,
    agent_label: str = "v2.4",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = _turn_groups(frames)
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    for frame in frames:
        if not _is_target_agent_frame(frame, agent_label):
            continue
        candidate_by_index, score_error = _score_map(frame, agent_label)
        important_options = _important_evolve_options(frame, candidate_by_index)
        best_evolve = _best_evolve_option(important_options)
        if best_evolve is None:
            continue
        best_attack = (_attack_summaries(frame) or [None])[0]
        snapshot = _snapshot_row(frame, important_options, best_attack)
        choice = _choice_summary(frame, candidate_by_index)
        same_turn_evolve, same_game_evolve = _later_evolve(frame, groups.get(_turn_key(frame), []))
        event = _base_event(frame, important_options, best_evolve, choice, snapshot, score_error)
        event.update(_classify_event(event, frame, same_turn_evolve, same_game_evolve))
        events.append(event)
        snapshots.append(snapshot)

    metrics = _metrics(events)
    audit = {
        "schema_version": EVOLVE_SEQUENCING_SCHEMA_VERSION,
        "run_id": run_id,
        "source_run": source_run,
        "agent_label": agent_label,
        "frames": len(frames),
        "target_agent_frames": sum(1 for frame in frames if _is_target_agent_frame(frame, agent_label)),
        "classification_tags": CLASSIFICATION_TAGS,
        "metrics": metrics,
        "report_answers": _report_answers(metrics),
        "do_not_implement_yet": [
            "Do not change gameplay policy from this audit.",
            "Do not tune scores from this audit.",
            "Do not promote attach_timing=new from this audit.",
            "Do not change deck.csv, cg/, main.py, trainer semantics, parser, ML/search, belief, or replay code.",
            "If evidence is strong, the next step is a separate v2.4[evolve_priority=new] ablation only.",
        ],
    }
    return audit, events, snapshots


def _metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    tag_counts = Counter(tag for event in events for tag in event.get("tags") or [])
    class_counts = Counter(str(event.get("classification") or "unknown") for event in events)
    beat_counts = Counter(
        str(event.get("chosen_option_type") or "UNKNOWN")
        for event in events
        if event.get("classification") != "evolve_taken_same_frame"
    )
    gaps = [event["score_gap_chosen_minus_evolve"] for event in events if event.get("score_gap_chosen_minus_evolve") is not None]
    delayed = [
        event
        for event in events
        if event.get("classification") in {"evolve_later_same_turn", "evolve_later_same_game"}
    ]
    delay_decisions = [event["delay_decisions"] for event in delayed if event.get("delay_decisions") is not None]
    delay_turns = [event["delay_turns"] for event in delayed if event.get("delay_turns") is not None]
    by_action = _score_gap_by_action(events)
    final_split = {
        "same_frame": dict(
            Counter(str(event.get("final_result")) for event in events if event.get("classification") == "evolve_taken_same_frame")
        ),
        "delayed": dict(Counter(str(event.get("final_result")) for event in delayed)),
        "never": dict(
            Counter(str(event.get("final_result")) for event in events if event.get("classification") == "evolve_never_observed")
        ),
    }
    total = len(events)
    delayed_count = len(delayed)
    conclusion = _conclusion(total, class_counts, tag_counts, gaps)
    return {
        "total_important_evolve_legal_frames": total,
        "same_frame_chosen_count": class_counts.get("evolve_taken_same_frame", 0),
        "delayed_count": delayed_count,
        "never_observed_count": class_counts.get("evolve_never_observed", 0),
        "later_same_turn_count": class_counts.get("evolve_later_same_turn", 0),
        "later_same_game_count": class_counts.get("evolve_later_same_game", 0),
        "mean_delay_decisions": _mean(delay_decisions),
        "median_delay_decisions": _median(delay_decisions),
        "mean_delay_turns": _mean(delay_turns),
        "median_delay_turns": _median(delay_turns),
        "action_that_beat_evolve": dict(beat_counts),
        "most_frequent_action_that_beat_evolve": beat_counts.most_common(1)[0][0] if beat_counts else "none",
        "bench_empty_count": tag_counts.get("bench_empty", 0),
        "lone_active_count": tag_counts.get("lone_active", 0),
        "attack_ready_count": sum(1 for event in events if event.get("board_plan_state") == "ATTACK_READY"),
        "weak_attack_ready_count": tag_counts.get("weak_attack_ready", 0),
        "meaningful_attack_ready_count": tag_counts.get("meaningful_attack_ready", 0),
        "ko_attack_available_count": tag_counts.get("ko_attack_available", 0),
        "prize_pressure_attack_available_count": tag_counts.get("prize_pressure_attack_available", 0),
        "mean_score_gap_chosen_minus_evolve": _mean(gaps),
        "median_score_gap_chosen_minus_evolve": _median(gaps),
        "score_gap_by_chosen_action": by_action,
        "classification_counts": dict(class_counts),
        "tag_counts": dict(tag_counts),
        "final_result_split": final_split,
        "future_evolve_priority_ablation_supported": conclusion["supports_ablation"],
        "conclusion": conclusion["text"],
        "evidence_label": conclusion["label"],
    }


def _score_gap_by_action(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int | float]] = defaultdict(list)
    for event in events:
        gap = event.get("score_gap_chosen_minus_evolve")
        if gap is not None:
            grouped[str(event.get("chosen_option_type") or "UNKNOWN")].append(gap)
    return [
        {
            "chosen_action": action,
            "count": len(values),
            "mean_gap": _mean(values),
            "median_gap": _median(values),
            "min_gap": min(values),
            "max_gap": max(values),
        }
        for action, values in sorted(grouped.items())
    ]


def _conclusion(
    total: int,
    class_counts: Counter[str],
    tag_counts: Counter[str],
    gaps: list[int | float],
) -> dict[str, Any]:
    if total == 0:
        return {
            "label": "not_proven",
            "supports_ablation": False,
            "text": "not proven: no important Mega-line EVOLVE legal frames were observed",
        }
    delayed_or_never = (
        class_counts.get("evolve_later_same_turn", 0)
        + class_counts.get("evolve_later_same_game", 0)
        + class_counts.get("evolve_never_observed", 0)
    )
    play_pressure = tag_counts.get("play_beat_evolve", 0)
    median_gap = _median(gaps)
    if total >= 5 and delayed_or_never >= max(3, total // 3) and play_pressure >= 2 and (median_gap or 0) > 0:
        return {
            "label": "suggested",
            "supports_ablation": True,
            "text": "suggested: PLAY sometimes outranks legal Mega-line EVOLVE with positive score gaps; test only via future evolve_priority ablation",
        }
    if class_counts.get("evolve_taken_same_frame", 0) >= max(1, total * 2 // 3):
        return {
            "label": "not_proven",
            "supports_ablation": False,
            "text": "not proven: default v2.4 usually takes important Mega-line EVOLVE when it is legal",
        }
    return {
        "label": "mixed",
        "supports_ablation": False,
        "text": "mixed: delayed or skipped EVOLVE frames exist, but this audit alone does not prove a policy issue",
    }


def _report_answers(metrics: dict[str, Any]) -> list[str]:
    return [
        f"Important Mega-line EVOLVE legal frames: {metrics['total_important_evolve_legal_frames']}.",
        f"Same-frame EVOLVE chosen: {metrics['same_frame_chosen_count']}.",
        f"Delayed EVOLVE: {metrics['delayed_count']} total, with {metrics['later_same_turn_count']} later same turn and {metrics['later_same_game_count']} later same game.",
        f"Never observed later in the same game: {metrics['never_observed_count']}.",
        f"Most frequent action that beat EVOLVE: {metrics['most_frequent_action_that_beat_evolve']}.",
        f"Median chosen-minus-EVOLVE score gap: {metrics['median_score_gap_chosen_minus_evolve']}.",
        f"Bench-empty / lone-active concentration: {metrics['bench_empty_count']} / {metrics['lone_active_count']}.",
        f"ATTACK_READY concentration: {metrics['attack_ready_count']} frames; weak attack ready: {metrics['weak_attack_ready_count']}.",
        f"Evidence label: {metrics['evidence_label']} ({metrics['conclusion']}).",
        "Recommended next step, if any: future diagnostic ablation `v2.4[evolve_priority=new]`; no behavior patch from this audit.",
    ]


def _csv_text(events: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "game_id",
        "turn",
        "decision_index",
        "classification",
        "chosen_option_type",
        "chosen_card_name",
        "evolve_score",
        "chosen_score",
        "score_gap_chosen_minus_evolve",
        "later_evolve_frame_id",
        "delay_decisions",
        "delay_turns",
        "board_plan_state",
        "bench_count",
        "final_result",
        "tags",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for event in events:
        writer.writerow(
            {
                field: ";".join(event.get(field) or []) if field == "tags" else event.get(field)
                for field in fields
            }
        )
    return stream.getvalue()


def _metric_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    keys = [
        "total_important_evolve_legal_frames",
        "same_frame_chosen_count",
        "delayed_count",
        "never_observed_count",
        "later_same_turn_count",
        "later_same_game_count",
        "most_frequent_action_that_beat_evolve",
        "bench_empty_count",
        "lone_active_count",
        "attack_ready_count",
        "weak_attack_ready_count",
        "meaningful_attack_ready_count",
        "ko_attack_available_count",
        "prize_pressure_attack_available_count",
        "median_score_gap_chosen_minus_evolve",
        "conclusion",
    ]
    return [[key, metrics.get(key)] for key in keys]


def build_evolve_sequencing_report(audit: dict[str, Any], events: list[dict[str, Any]]) -> str:
    metrics = audit.get("metrics") or {}
    agent_label = audit.get("agent_label") or "v2.4"
    lines = [title("Deck Track V3.8 EVOLVE Sequencing Audit")]
    lines.append(section("Scope"))
    lines.append(
        bullets(
            [
                f"Diagnostic-only audit of BabyMew `{agent_label}` frames.",
                "Event definition: at least one visible/resolved current-deck Mega-line EVOLVE option is legal.",
                "Each row is one frame; duplicate legal EVOLVE choices are retained but compared through the highest-scored important EVOLVE option.",
                "Scores use the audited agent's explicit ablation modes; no default behavior policy is changed.",
                "This is sequencing evidence, not causal proof.",
            ]
        )
    )
    lines.append(section("Headline Metrics"))
    lines.append(table(["Metric", "Value"], _metric_rows(metrics)))
    lines.append(section("Score Gap By Chosen Action"))
    lines.append(
        table(
            ["Action", "Count", "Mean Gap", "Median Gap", "Min Gap", "Max Gap"],
            [
                [row["chosen_action"], row["count"], row["mean_gap"], row["median_gap"], row["min_gap"], row["max_gap"]]
                for row in metrics.get("score_gap_by_chosen_action", [])
            ]
            or [["none", 0, "none", "none", "none", "none"]],
        )
    )
    lines.append(section("Classification Counts"))
    lines.append(table(["Class", "Count"], Counter(metrics.get("classification_counts", {})).most_common()))
    lines.append(section("Tag Counts"))
    lines.append(table(["Tag", "Count"], Counter(metrics.get("tag_counts", {})).most_common()))
    lines.append(section("Questions Answered"))
    lines.append(bullets(audit.get("report_answers") or []))
    lines.append(section("Interpretation Boundary"))
    lines.append(
        bullets(
            [
                "Confirmed means a repeatable pattern is strong enough to justify an ablation, not a direct behavior patch.",
                "Suggested means the next step is a focused ablation, not score tuning.",
                "Not proven means the current trace either lacks enough rows or already evolves promptly.",
                "Old outputs and legacy evidence are not moved or deleted by this audit.",
            ]
        )
    )
    lines.append(section("Example Frames"))
    for event in _example_rows(events):
        evolve = event.get("evolve_option") or {}
        best_attack = event.get("best_attack") or {}
        lines.append(f"### {event.get('frame_id')}\n")
        lines.append(
            bullets(
                [
                    f"turn={event.get('turn')} decision={event.get('decision_index')} class={event.get('classification')}",
                    f"chosen={event.get('chosen_option_type')} {event.get('chosen_card_name') or event.get('chosen_attack_name') or ''}",
                    f"evolve={evolve.get('card_name')} -> {evolve.get('target_card_name')} target={evolve.get('target_area')}[{evolve.get('target_index')}]",
                    f"scores chosen={event.get('chosen_score')} evolve={event.get('evolve_score')} gap={event.get('score_gap_chosen_minus_evolve')}",
                    f"board={event.get('board_plan_state')} bench={event.get('bench_count')} best_attack={best_attack.get('pressure')}",
                    f"later_evolve={event.get('later_evolve_frame_id')} delay_decisions={event.get('delay_decisions')}",
                    f"tags={event.get('tags')}",
                ]
            )
        )
    return "\n".join(lines)


def _example_rows(events: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for klass in [
        "evolve_taken_same_frame",
        "evolve_later_same_turn",
        "evolve_later_same_game",
        "evolve_never_observed",
    ]:
        for event in events:
            if len(selected) >= limit:
                return selected
            if event.get("classification") == klass and event not in selected:
                selected.append(event)
    return selected[:limit]


def _example_frames_markdown(audit: dict[str, Any], events: list[dict[str, Any]]) -> str:
    lines = ["# Example Frames", "", "Compact examples for external review; not causal proof.", ""]
    for event in _example_rows(events):
        evolve = event.get("evolve_option") or {}
        lines.extend(
            [
                f"## {event.get('frame_id')}",
                "",
                f"- classification: {event.get('classification')}",
                f"- chosen: {event.get('chosen_option_type')} {event.get('chosen_card_name') or event.get('chosen_attack_name') or ''}",
                f"- evolve option: {evolve.get('card_name')} -> {evolve.get('target_card_name')} at {evolve.get('target_area')}[{evolve.get('target_index')}]",
                f"- scores: chosen={event.get('chosen_score')} evolve={event.get('evolve_score')} gap={event.get('score_gap_chosen_minus_evolve')}",
                f"- board: {event.get('board_plan_state')} bench={event.get('bench_count')} active={event.get('own_active')}",
                f"- best attack: {event.get('best_attack')}",
                f"- later evolve: {event.get('later_evolve_frame_id')} delay={event.get('delay_decisions')}",
                f"- tags: {event.get('tags')}",
                "",
            ]
        )
    return "\n".join(lines)


def _readme(source_run: str | None, command: str, agent_label: str) -> str:
    return "\n".join(
        [
            "# BabyMew EVOLVE Sequencing Review Bundle",
            "",
            f"This bundle contains diagnostic-only evidence for `{agent_label}` frames with legal current-deck Mega-line EVOLVE.",
            "",
            f"- source run: `{source_run}`",
            f"- generated by: `{command}`",
            f"- policy under review: BabyMew `{agent_label}`; no default behavior is promoted here.",
            "",
            "Inspect first:",
            "",
            "1. `executive_summary.md`",
            "2. `evolve_sequencing_counts.json`",
            "3. `evolve_events.csv`",
            "4. `example_frames.md`",
            "",
        ]
    )


def _executive_summary(audit: dict[str, Any]) -> str:
    metrics = audit.get("metrics") or {}
    agent_label = audit.get("agent_label") or "v2.4"
    return "\n".join(
        [
            "# Executive Summary",
            "",
            "This is a decoupled EVOLVE sequencing audit, not a behavior patch.",
            "",
            f"- agent: `{agent_label}`",
            f"- important EVOLVE legal frames: {metrics.get('total_important_evolve_legal_frames', 0)}",
            f"- same-frame EVOLVE chosen: {metrics.get('same_frame_chosen_count', 0)}",
            f"- delayed EVOLVE: {metrics.get('delayed_count', 0)}",
            f"- never observed: {metrics.get('never_observed_count', 0)}",
            f"- most frequent action that beat EVOLVE: {metrics.get('most_frequent_action_that_beat_evolve')}",
            f"- median chosen-minus-EVOLVE score gap: {metrics.get('median_score_gap_chosen_minus_evolve')}",
            f"- conclusion: {metrics.get('conclusion')}",
            f"- future evolve_priority ablation supported: {metrics.get('future_evolve_priority_ablation_supported')}",
            "",
        ]
    )


def _relevant_paths(source_run: str | None, audit_run: Path, bundle_dir: Path | None) -> str:
    paths = [
        source_run or "",
        audit_run / "report.md",
        audit_run / "evolve_sequencing.json",
        audit_run / "evolve_events.jsonl",
        audit_run / "state_machine_snapshots.jsonl",
        bundle_dir or "",
        Path("deepmew/trace/evolve_sequencing.py"),
        Path("scripts/audit_deck_v3_evolve_sequencing.py"),
        Path("tests/test_evolve_sequencing.py"),
        Path(".research/TASKS.md"),
        Path(".research/STATE.md"),
        Path(".research/CURRENT_STATE_COMPACT.md"),
        Path("HANDOFF_SUMMARY.md"),
    ]
    return "\n".join(str(path) for path in paths if str(path)) + "\n"


def write_evolve_sequencing_outputs(
    *,
    run_dir: Path,
    run_id: str,
    command: str,
    source_run: str | None,
    frames: list[dict[str, Any]],
    agent_label: str = "v2.4",
    inputs: dict[str, Any] | None = None,
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> EvolveSequencingResult:
    audit, events, snapshots = analyze_evolve_sequencing(
        run_id=run_id,
        frames=frames,
        source_run=source_run,
        agent_label=agent_label,
    )
    run_path = Path(run_dir)
    outputs = {
        "report_markdown": str(run_path / "report.md"),
        "evolve_sequencing_json": str(run_path / "evolve_sequencing.json"),
        "evolve_events_jsonl": str(run_path / "evolve_events.jsonl"),
        "evolve_events_csv": str(run_path / "evolve_events.csv"),
        "state_machine_snapshots_jsonl": str(run_path / "state_machine_snapshots.jsonl"),
        "example_frames_markdown": str(run_path / "example_frames.md"),
    }
    write_json(run_path / "evolve_sequencing.json", audit)
    write_jsonl(run_path / "evolve_events.jsonl", events)
    write_text(run_path / "evolve_events.csv", _csv_text(events))
    write_jsonl(run_path / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(run_path / "report.md", build_evolve_sequencing_report(audit, events))
    write_markdown(run_path / "example_frames.md", _example_frames_markdown(audit, events))

    bundle_dir = None
    if export_handoff_bundle:
        bundle_dir = export_evolve_sequencing_bundle(
            audit=audit,
            events=events,
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
        inputs=inputs or {"source_run": source_run, "agent_label": agent_label},
        outputs=outputs,
        summary=audit.get("metrics") or {},
    )
    write_manifest(run_path / "manifest.json", manifest)
    return EvolveSequencingResult(run_dir=run_path, manifest_path=run_path / "manifest.json", audit=audit, events=events, bundle_dir=bundle_dir)


def export_evolve_sequencing_bundle(
    *,
    audit: dict[str, Any],
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    source_run: str | None,
    audit_run: Path,
    command: str,
    out_root: Path | str = ".research/handoff_bundles",
) -> Path:
    bundle_dir = _create_handoff_dir(out_root=out_root)
    counts = audit.get("metrics") or {}
    agent_label = str(audit.get("agent_label") or "v2.4")
    write_markdown(bundle_dir / "README.md", _readme(source_run, command, agent_label))
    write_markdown(bundle_dir / "executive_summary.md", _executive_summary(audit))
    write_json(bundle_dir / "evolve_sequencing_counts.json", counts)
    write_text(bundle_dir / "evolve_events.csv", _csv_text(events))
    write_jsonl(bundle_dir / "evolve_events.jsonl", events)
    write_jsonl(bundle_dir / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(bundle_dir / "example_frames.md", _example_frames_markdown(audit, events))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(source_run, audit_run, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=HANDOFF_KIND,
        command=command,
        inputs={"evolve_sequencing_run": str(audit_run), "source_run": source_run},
        outputs={name: str(bundle_dir / name) for name in REQUIRED_BUNDLE_FILES if name != "manifest.json"},
        summary={
            "source_run": source_run,
            "evolve_sequencing_run": str(audit_run),
            **counts,
        },
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir
