"""Same-observation delta audit for the experimental evolve_priority ablation."""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

from deepmew.arena import run_pair
from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id, safe_slug
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown, write_text
from deepmew.deck_profiles import (
    CYRANO_CARD_ID,
    KYOGRE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.policy import score_candidates
from deepmew.policy_rules.evolve_priority import (
    classify_evolve_priority,
    extract_evolve_priority_facts,
)
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board, describe_plan
from deepmew.trace.evolve_sequencing import (
    _attack_summaries,
    _best_evolve_option,
    _important_evolve_options,
)
from deepmew.trace.probe import probe_frame
from deepmew.trace.record import TraceRecorder
from deepmew.trace.schema import to_jsonable


RUN_KIND = "deck_evolve_priority_delta"
HANDOFF_KIND = "babymew_evolve_priority_delta_review"
DELTA_SUMMARY_SCHEMA_VERSION = "deepmew.evolve_priority_delta.v1"
DELTA_EVENT_SCHEMA_VERSION = "deepmew.evolve_priority_delta_event.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "deepmew.evolve_priority_delta_snapshot.v1"

DEFAULT_POLICY = "v2.4"
ABLATION_POLICY = "v2.4[evolve_priority=new]"
DEFAULT_SOURCE_AGENTS = (DEFAULT_POLICY, ABLATION_POLICY)
TARGET_CREATING_PLAY_IDS = frozenset({SNOVER_CARD_ID, KYOGRE_CARD_ID})
NAMED_PLAY_BUCKETS = {
    LILLIES_DETERMINATION_CARD_ID: "Lillie's Determination",
    MEGA_SIGNAL_CARD_ID: "Mega Signal",
    WAITRESS_CARD_ID: "Waitress",
    CYRANO_CARD_ID: "Cyrano",
    MAXIMUM_BELT_CARD_ID: "Maximum Belt",
}

REQUIRED_RUN_FILES = [
    "manifest.json",
    "report.md",
    "delta_summary.json",
    "delta_events.jsonl",
    "delta_events.csv",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
]
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "delta_summary.json",
    "delta_events.csv",
    "delta_events.jsonl",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
    "relevant_paths.txt",
]


class DeltaClassification(str, Enum):
    NO_DIFFERENCE = "NO_DIFFERENCE"
    NOT_EVOLVE_RELATED = "NOT_EVOLVE_RELATED"
    GOOD_EVOLVE_CORRECTION = "GOOD_EVOLVE_CORRECTION"
    PREMATURE_EVOLVE_BEFORE_VALUABLE_PLAY = "PREMATURE_EVOLVE_BEFORE_VALUABLE_PLAY"
    PREMATURE_EVOLVE_BEFORE_TARGET_CREATION = "PREMATURE_EVOLVE_BEFORE_TARGET_CREATION"
    NEUTRAL_EVOLVE_SAME_TURN_ANYWAY = "NEUTRAL_EVOLVE_SAME_TURN_ANYWAY"
    DEFAULT_DELAYED_BUT_RECOVERED = "DEFAULT_DELAYED_BUT_RECOVERED"
    ABLATED_SKIPPED_USEFUL_DRAW_OR_SEARCH = "ABLATED_SKIPPED_USEFUL_DRAW_OR_SEARCH"
    ABLATED_SKIPPED_TARGET_CREATING_PLAY = "ABLATED_SKIPPED_TARGET_CREATING_PLAY"
    PRESERVED_KO_OR_PRIZE_ATTACK = "PRESERVED_KO_OR_PRIZE_ATTACK"
    ATTACH_SEQUENCE_DIFFERENCE = "ATTACH_SEQUENCE_DIFFERENCE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DeltaFacts:
    same_choice: bool
    evolve_priority_related: bool
    default_type: str | None
    ablation_type: str | None
    default_card_id: int | None
    important_evolve_legal: bool
    default_later_evolved_same_turn: bool
    default_later_evolved_same_game: bool
    ko_or_prize_attack_available: bool
    ablation_score_adjusted_top_action: bool


@dataclass(frozen=True)
class DeltaRule:
    name: str
    guard: Callable[[DeltaFacts], bool]
    classification: DeltaClassification

    def classify(self) -> dict[str, str]:
        return {"classification": self.classification.value, "classification_reason": self.name}


@dataclass(frozen=True)
class EvolvePriorityDeltaResult:
    run_dir: Path
    manifest_path: Path
    summary: dict[str, Any]
    events: list[dict[str, Any]]
    bundle_dir: Path | None = None


def same_choice(facts: DeltaFacts) -> bool:
    return facts.same_choice


def not_evolve_related(facts: DeltaFacts) -> bool:
    return not facts.evolve_priority_related


def ko_or_prize_attack_available(facts: DeltaFacts) -> bool:
    return facts.ko_or_prize_attack_available


def ablation_changes_target_play_to_evolve(facts: DeltaFacts) -> bool:
    return (
        facts.default_type == "PLAY"
        and facts.ablation_type == "EVOLVE"
        and facts.default_card_id in TARGET_CREATING_PLAY_IDS
    )


def ablation_changes_named_play_to_evolve(facts: DeltaFacts) -> bool:
    return (
        facts.default_type == "PLAY"
        and facts.ablation_type == "EVOLVE"
        and facts.default_card_id in NAMED_PLAY_BUCKETS
    )


def default_evolved_later_same_turn(facts: DeltaFacts) -> bool:
    return facts.default_type == "PLAY" and facts.ablation_type == "EVOLVE" and facts.default_later_evolved_same_turn


def default_evolved_later_same_game(facts: DeltaFacts) -> bool:
    return facts.default_type == "PLAY" and facts.ablation_type == "EVOLVE" and facts.default_later_evolved_same_game


def default_play_to_evolve(facts: DeltaFacts) -> bool:
    return facts.default_type == "PLAY" and facts.ablation_type == "EVOLVE"


def default_attach_to_evolve(facts: DeltaFacts) -> bool:
    return facts.default_type == "ATTACH" and facts.ablation_type == "EVOLVE"


def default_attack_to_evolve(facts: DeltaFacts) -> bool:
    return facts.default_type == "ATTACK" and facts.ablation_type == "EVOLVE"


def always(_: DeltaFacts) -> bool:
    return True


DELTA_CLASSIFICATION_RULES: tuple[DeltaRule, ...] = (
    DeltaRule("no-difference", same_choice, DeltaClassification.NO_DIFFERENCE),
    DeltaRule("not-evolve-related", not_evolve_related, DeltaClassification.NOT_EVOLVE_RELATED),
    DeltaRule("preserve-ko-or-prize-attack", ko_or_prize_attack_available, DeltaClassification.PRESERVED_KO_OR_PRIZE_ATTACK),
    DeltaRule(
        "ablation-skipped-target-creating-play",
        ablation_changes_target_play_to_evolve,
        DeltaClassification.ABLATED_SKIPPED_TARGET_CREATING_PLAY,
    ),
    DeltaRule(
        "default-play-evolved-later-same-turn",
        default_evolved_later_same_turn,
        DeltaClassification.NEUTRAL_EVOLVE_SAME_TURN_ANYWAY,
    ),
    DeltaRule(
        "ablation-skipped-useful-draw-or-search",
        ablation_changes_named_play_to_evolve,
        DeltaClassification.ABLATED_SKIPPED_USEFUL_DRAW_OR_SEARCH,
    ),
    DeltaRule(
        "default-play-evolved-later-same-game",
        default_evolved_later_same_game,
        DeltaClassification.DEFAULT_DELAYED_BUT_RECOVERED,
    ),
    DeltaRule("default-play-to-evolve", default_play_to_evolve, DeltaClassification.GOOD_EVOLVE_CORRECTION),
    DeltaRule("default-attach-to-evolve", default_attach_to_evolve, DeltaClassification.ATTACH_SEQUENCE_DIFFERENCE),
    DeltaRule("default-attack-to-evolve", default_attack_to_evolve, DeltaClassification.UNKNOWN),
    DeltaRule("unknown", always, DeltaClassification.UNKNOWN),
)


def classify_delta(facts: DeltaFacts) -> dict[str, str]:
    for rule in DELTA_CLASSIFICATION_RULES:
        if rule.guard(facts):
            return rule.classify()
    raise AssertionError("unreachable evolve-priority delta classifier")


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


def _raw_observation(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("raw_observation")
    return raw if isinstance(raw, dict) else {}


def _raw_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    select = _raw_observation(frame).get("select") or {}
    return [option for option in (select.get("option") or []) if isinstance(option, dict)]


def _probe_choice(frame: dict[str, Any], policy_spec: str) -> dict[str, Any]:
    probe = probe_frame(frame, policy_spec).to_jsonable()
    chosen = probe.get("chosen_option_summaries") or []
    summary = chosen[0] if chosen and isinstance(chosen[0], dict) else {}
    return {
        "policy": probe.get("policy_spec") or policy_spec,
        "indices": probe.get("chosen_indices") or [],
        "valid_return": probe.get("valid_return"),
        "error": probe.get("error"),
        "summary": summary,
        "option_index": _int_or_none(summary.get("index")) if summary else _first_index(probe.get("chosen_indices")),
        "option_type": summary.get("option_type"),
        "card_id": _int_or_none(summary.get("card_id")),
        "card_name": summary.get("card_name"),
        "attack_id": _int_or_none(summary.get("attack_id")),
        "attack_name": summary.get("attack_name"),
        "target_area": summary.get("target_area"),
        "target_index": _int_or_none(summary.get("target_index")),
        "target_card_id": _int_or_none(summary.get("target_card_id")),
        "target_card_name": summary.get("target_card_name"),
    }


def _first_index(indices: Any) -> int | None:
    if isinstance(indices, list) and indices:
        return _int_or_none(indices[0])
    return None


def _score_modes(policy_label: str) -> dict[str, str]:
    return {
        "attach_timing": "new" if "attach_timing=new" in policy_label else "old",
        "evolve_priority": "new" if "evolve_priority=new" in policy_label else "old",
    }


def _score_map(raw: dict[str, Any], policy_label: str) -> dict[int, Any]:
    modes = _score_modes(policy_label)
    return {
        candidate.index: candidate
        for candidate in score_candidates(
            raw,
            use_card_semantics=True,
            attach_timing=modes["attach_timing"],
            evolve_priority=modes["evolve_priority"],
        )
    }


def _score_payload(candidate: Any | None) -> dict[str, Any]:
    if candidate is None:
        return {"score": None, "reasons": []}
    return {"score": candidate.score, "reasons": list(candidate.reasons)}


def _candidate_payloads(candidates: dict[int, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": candidate.index,
            "option_type": candidate.option_type,
            "score": candidate.score,
            "reasons": list(candidate.reasons),
        }
        for candidate in sorted(candidates.values(), key=lambda item: item.index)
    ]


def _top_candidate(candidates: dict[int, Any]) -> Any | None:
    if not candidates:
        return None
    return min(candidates.values(), key=lambda candidate: (-candidate.score, candidate.index))


def _changed_scores(default_scores: dict[int, Any], ablation_scores: dict[int, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in sorted(set(default_scores) | set(ablation_scores)):
        left = default_scores.get(index)
        right = ablation_scores.get(index)
        if left is None or right is None or left.score != right.score or tuple(left.reasons) != tuple(right.reasons):
            rows.append(
                {
                    "index": index,
                    "option_type": getattr(right or left, "option_type", None),
                    "default_score": getattr(left, "score", None),
                    "ablation_score": getattr(right, "score", None),
                    "delta": None if left is None or right is None else right.score - left.score,
                    "default_reasons": list(getattr(left, "reasons", ())) if left is not None else [],
                    "ablation_reasons": list(getattr(right, "reasons", ())) if right is not None else [],
                }
            )
    return rows


def _frame_groups(frames: list[dict[str, Any]]) -> dict[tuple[str, str, int | None], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        key = (str(frame.get("source_agent") or frame.get("agent_label") or ""), str(frame.get("game_id") or ""), _int_or_none(frame.get("seat")))
        groups[key].append(frame)
    for group in groups.values():
        group.sort(key=lambda item: _int_or_none(item.get("decision_index")) or -1)
    return groups


def _chosen_important_evolve(frame: dict[str, Any], agent_label: str) -> dict[str, Any] | None:
    raw = _raw_observation(frame)
    if not raw:
        return None
    choice = _probe_choice(frame, agent_label)
    if choice["option_type"] != "EVOLVE" or choice["option_index"] is None:
        return None
    scores = _score_map(raw, agent_label)
    for option in _important_evolve_options(frame, scores):
        if option["option_index"] == choice["option_index"]:
            return option
    return None


def _later_evolve(frame: dict[str, Any], group: list[dict[str, Any]], source_agent: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    current_decision = _int_or_none(frame.get("decision_index")) or -1
    current_turn = _int_or_none(frame.get("turn"))
    same_turn: dict[str, Any] | None = None
    same_game: dict[str, Any] | None = None
    later_actions: list[dict[str, Any]] = []
    for later in group:
        later_decision = _int_or_none(later.get("decision_index")) or -1
        if later_decision <= current_decision:
            continue
        if len(later_actions) < 8:
            chosen = later.get("chosen_option_summaries") or []
            later_actions.append(
                {
                    "frame_id": later.get("frame_id"),
                    "turn": later.get("turn"),
                    "decision_index": later.get("decision_index"),
                    "chosen": chosen[0] if chosen else {},
                }
            )
        if _chosen_important_evolve(later, source_agent) is None:
            continue
        if same_game is None:
            same_game = later
        if current_turn == _int_or_none(later.get("turn")) and same_turn is None:
            same_turn = later
    return same_turn, same_game, later_actions


def _plan_payload(raw: dict[str, Any]) -> tuple[Any | None, dict[str, Any], str]:
    if not raw:
        return None, {}, "missing-raw-observation"
    try:
        plan = analyze_board(raw)
    except Exception as exc:
        return None, {}, f"state-machine-error={type(exc).__name__}: {exc}"
    return (
        plan,
        {
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
            "readiness": [to_jsonable(item) for item in getattr(plan, "readiness", ()) or ()],
        },
        describe_plan(plan),
    )


def _own_active(frame: dict[str, Any]) -> dict[str, Any]:
    active = (frame.get("board_summary") or {}).get("own_active")
    return active if isinstance(active, dict) else {}


def _play_indices_from_facts(raw: dict[str, Any], plan: Any | None) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    select = raw.get("select") or {}
    options = select.get("option") or []
    facts = extract_evolve_priority_facts(raw, select, options, plan, use_card_semantics=True)
    decision = classify_evolve_priority(facts)
    return (
        facts.non_target_play_indices,
        facts.target_creating_play_indices,
        {
            "state": decision.state.value,
            "action": decision.action.value,
            "play_score_delta": decision.play_score_delta,
            "reason": decision.reason,
        },
    )


def _evolve_priority_related(default_choice: dict[str, Any], ablation_choice: dict[str, Any], important_options: list[dict[str, Any]], changed_scores: list[dict[str, Any]], default_top: Any | None, ablation_top: Any | None) -> bool:
    if not important_options:
        return False
    if default_choice["option_type"] == "EVOLVE" or ablation_choice["option_type"] == "EVOLVE":
        return True
    if default_choice["option_type"] == "PLAY" and ablation_choice["option_type"] == "EVOLVE":
        return True
    return bool(
        default_top is not None
        and ablation_top is not None
        and default_top.index != ablation_top.index
        and any("evolve-priority-state=" in reason for row in changed_scores for reason in row.get("ablation_reasons") or [])
    )


def _tags(event: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    if event["default_chosen_action_type"] != event["ablation_chosen_action_type"]:
        tags.add(f"{str(event['default_chosen_action_type']).lower()}_to_{str(event['ablation_chosen_action_type']).lower()}")
    if event.get("important_evolve_options"):
        tags.add("important_evolve_legal")
    if event.get("default_chosen_action_type") == "PLAY":
        if event.get("default_chosen_card_id") in TARGET_CREATING_PLAY_IDS:
            tags.add("target_creating_play_skipped")
        else:
            tags.add("non_target_play_skipped")
        if event.get("default_chosen_card_name"):
            tags.add(f"play_card={event['default_chosen_card_name']}")
    if event.get("default_later_evolve_same_turn_frame_id"):
        tags.add("default_later_evolved_same_turn")
    elif event.get("default_later_evolve_same_game_frame_id"):
        tags.add("default_later_evolved_same_game")
    if event.get("ko_attack_available"):
        tags.add("ko_attack_available")
    if event.get("prize_pressure_attack_available"):
        tags.add("prize_pressure_attack_available")
    if event.get("ablation_score_adjusted_top_action"):
        tags.add("ablation_score_adjusted_top_action")
    return sorted(tags)


def build_delta_event(
    frame: dict[str, Any],
    *,
    group: list[dict[str, Any]],
    default_policy: str = DEFAULT_POLICY,
    ablation_policy: str = ABLATION_POLICY,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    raw = _raw_observation(frame)
    if not raw:
        return None
    source_agent = str(frame.get("source_agent") or frame.get("agent_label") or "")
    default_choice = _probe_choice(frame, default_policy)
    ablation_choice = _probe_choice(frame, ablation_policy)
    if default_choice["error"] or ablation_choice["error"]:
        return None

    default_scores = _score_map(raw, default_policy)
    ablation_scores = _score_map(raw, ablation_policy)
    default_top = _top_candidate(default_scores)
    ablation_top = _top_candidate(ablation_scores)
    changed_scores = _changed_scores(default_scores, ablation_scores)
    important_options = _important_evolve_options(frame, ablation_scores)
    best_evolve = _best_evolve_option(important_options)
    best_attack = (_attack_summaries(frame) or [None])[0]
    attack_rows = _attack_summaries(frame)
    plan, plan_summary, plan_text = _plan_payload(raw)
    non_target_play_indices, target_creating_play_indices, decision_payload = _play_indices_from_facts(raw, plan)
    same_turn, same_game, later_actions = _later_evolve(frame, group, source_agent)
    own_active = _own_active(frame)
    same = default_choice["indices"] == ablation_choice["indices"]
    related = _evolve_priority_related(default_choice, ablation_choice, important_options, changed_scores, default_top, ablation_top)
    default_score = _score_payload(default_scores.get(default_choice["option_index"]))
    ablation_score = _score_payload(ablation_scores.get(ablation_choice["option_index"]))
    evolve_score = _score_payload(ablation_scores.get(best_evolve["option_index"] if best_evolve else None))
    ko_available = any(row.get("pressure") == "ko" for row in attack_rows)
    prize_available = any(row.get("pressure") in {"prize_pressure", "meaningful"} for row in attack_rows)
    facts = DeltaFacts(
        same_choice=same,
        evolve_priority_related=related,
        default_type=default_choice["option_type"],
        ablation_type=ablation_choice["option_type"],
        default_card_id=default_choice["card_id"],
        important_evolve_legal=bool(important_options),
        default_later_evolved_same_turn=same_turn is not None,
        default_later_evolved_same_game=same_game is not None,
        ko_or_prize_attack_available=ko_available or prize_available,
        ablation_score_adjusted_top_action=bool(default_top is not None and ablation_top is not None and default_top.index != ablation_top.index),
    )
    classification = classify_delta(facts)
    event = {
        "schema_version": DELTA_EVENT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "source_agent": source_agent,
        "game_id": frame.get("game_id"),
        "matchup": frame.get("matchup"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "board_plan_state": plan_summary.get("state"),
        "describe_plan": plan_text,
        "active_card_id": own_active.get("card_id"),
        "active_card_name": own_active.get("name"),
        "active_energy_count": own_active.get("energy_count"),
        "bench_count": plan_summary.get("bench_count"),
        "energy_attached": plan_summary.get("energy_attached"),
        "legal_attack_available": plan_summary.get("legal_attack_available"),
        "legal_attach_available": plan_summary.get("legal_attach_available"),
        "legal_play_available": plan_summary.get("legal_play_available"),
        "important_evolve_options": important_options,
        "best_visible_attack_pressure": (best_attack or {}).get("pressure"),
        "best_visible_attack": best_attack,
        "ko_attack_available": ko_available,
        "prize_pressure_attack_available": prize_available,
        "default_chosen_action_type": default_choice["option_type"],
        "default_chosen_option_index": default_choice["option_index"],
        "default_chosen_card_id": default_choice["card_id"],
        "default_chosen_card_name": default_choice["card_name"],
        "default_chosen_attack_id": default_choice["attack_id"],
        "default_chosen_attack_name": default_choice["attack_name"],
        "default_chosen_score": default_score["score"],
        "default_score_reasons": default_score["reasons"],
        "ablation_chosen_action_type": ablation_choice["option_type"],
        "ablation_chosen_option_index": ablation_choice["option_index"],
        "ablation_chosen_card_id": ablation_choice["card_id"],
        "ablation_chosen_card_name": ablation_choice["card_name"],
        "ablation_chosen_attack_id": ablation_choice["attack_id"],
        "ablation_chosen_attack_name": ablation_choice["attack_name"],
        "ablation_chosen_score": ablation_score["score"],
        "ablation_score_reasons": ablation_score["reasons"],
        "evolve_option_score": evolve_score["score"],
        "evolve_option_score_reasons": evolve_score["reasons"],
        "default_chosen_minus_evolve_score_gap": None if default_score["score"] is None or evolve_score["score"] is None else default_score["score"] - evolve_score["score"],
        "ablation_chosen_minus_evolve_score_gap": None if ablation_score["score"] is None or evolve_score["score"] is None else ablation_score["score"] - evolve_score["score"],
        "changed_scores": changed_scores,
        "ablation_score_adjusted_top_action": facts.ablation_score_adjusted_top_action,
        "default_top_action_type": getattr(default_top, "option_type", None),
        "default_top_option_index": getattr(default_top, "index", None),
        "ablation_top_action_type": getattr(ablation_top, "option_type", None),
        "ablation_top_option_index": getattr(ablation_top, "index", None),
        "non_target_play_indices": list(non_target_play_indices),
        "target_creating_play_indices": list(target_creating_play_indices),
        "evolve_priority_decision": decision_payload,
        "source_trajectory_later_same_turn_actions": [row for row in later_actions if row.get("turn") == frame.get("turn")],
        "source_trajectory_later_actions": later_actions,
        "default_later_evolve_same_turn_frame_id": same_turn.get("frame_id") if same_turn else None,
        "default_later_evolve_same_game_frame_id": same_game.get("frame_id") if same_game else None,
        "source_trajectory_final_result": frame.get("final_result"),
        "source_trajectory_final_result_reason": frame.get("final_result_reason"),
        "is_delta_event": not same,
        "is_evolve_priority_delta_event": bool(not same and related),
        **classification,
    }
    event["tags"] = _tags(event)
    snapshot = {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "source_agent": source_agent,
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "decision_index": frame.get("decision_index"),
        "board_plan": plan_summary,
        "describe_plan": plan_text,
        "important_evolve_options": important_options,
        "best_visible_attack": best_attack,
        "default_candidates": _candidate_payloads(default_scores),
        "ablation_candidates": _candidate_payloads(ablation_scores),
    }
    return event, snapshot


def analyze_evolve_priority_delta(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    source_agents: Iterable[str] | None = None,
    default_policy: str = DEFAULT_POLICY,
    ablation_policy: str = ABLATION_POLICY,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = set(source_agents or DEFAULT_SOURCE_AGENTS)
    for frame in frames:
        frame.setdefault("source_agent", frame.get("agent_label"))
    groups = _frame_groups(frames)
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    total_probed = 0
    for frame in frames:
        source_agent = str(frame.get("source_agent") or frame.get("agent_label") or "")
        if source_agent not in wanted:
            continue
        total_probed += 1
        key = (source_agent, str(frame.get("game_id") or ""), _int_or_none(frame.get("seat")))
        built = build_delta_event(frame, group=groups.get(key, []), default_policy=default_policy, ablation_policy=ablation_policy)
        if built is None:
            continue
        event, snapshot = built
        if event["is_delta_event"]:
            events.append(event)
            snapshots.append(snapshot)
    summary = summarize_delta_events(
        run_id=run_id,
        frames=frames,
        total_probed=total_probed,
        events=events,
        source_agents=tuple(wanted),
        default_policy=default_policy,
        ablation_policy=ablation_policy,
    )
    return summary, events, snapshots


def _mean(values: list[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[int | float]) -> float | None:
    return float(median(values)) if values else None


def _score_gap_summary_by_class(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int | float]] = defaultdict(list)
    for event in events:
        gap = event.get("default_chosen_minus_evolve_score_gap")
        if gap is not None:
            grouped[str(event.get("classification") or "UNKNOWN")].append(gap)
    return [
        {
            "classification": key,
            "count": len(values),
            "mean_gap": _mean(values),
            "median_gap": _median(values),
            "min_gap": min(values),
            "max_gap": max(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _play_card_bucket(card_id: int | None) -> str:
    return NAMED_PLAY_BUCKETS.get(card_id, "other")


def _report_answers(metrics: dict[str, Any]) -> list[str]:
    action_counts = metrics.get("action_change_counts") or {}
    class_counts = metrics.get("classification_counts") or {}
    play_counts = metrics.get("default_play_card_counts") or {}
    return [
        f"Frames probed: {metrics.get('total_frames_probed', 0)}.",
        f"Delta events: {metrics.get('total_delta_events', 0)}; evolve-priority deltas: {metrics.get('total_evolve_priority_delta_events', 0)}.",
        f"Main action changes: {action_counts}.",
        f"Default PLAY card breakdown in PLAY -> EVOLVE deltas: {play_counts}.",
        f"Classification counts: {class_counts}.",
        f"Default later evolved same turn: {metrics.get('default_later_evolved_same_turn_count', 0)}.",
        f"Target-creating PLAY skipped: {metrics.get('target_creating_play_skipped_count', 0)}; non-target PLAY skipped: {metrics.get('non_target_play_skipped_count', 0)}.",
        f"Interpretation: {metrics.get('interpretation')}.",
        f"Recommendation: {metrics.get('recommendation')}.",
        "Do not change default policy, deck, cg, trainer semantics, parser, ML/search/belief/replay learning, or main.py from this audit.",
    ]


def summarize_delta_events(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    total_probed: int,
    events: list[dict[str, Any]],
    source_agents: tuple[str, ...],
    default_policy: str,
    ablation_policy: str,
) -> dict[str, Any]:
    evolve_events = [event for event in events if event.get("is_evolve_priority_delta_event")]
    action_counts = Counter(
        f"{event.get('default_chosen_action_type')}->{event.get('ablation_chosen_action_type')}"
        for event in evolve_events
    )
    play_to_evolve = [
        event
        for event in evolve_events
        if event.get("default_chosen_action_type") == "PLAY" and event.get("ablation_chosen_action_type") == "EVOLVE"
    ]
    play_counts = Counter(_play_card_bucket(_int_or_none(event.get("default_chosen_card_id"))) for event in play_to_evolve)
    for bucket in ["Lillie's Determination", "Mega Signal", "Waitress", "Cyrano", "Maximum Belt", "other"]:
        play_counts.setdefault(bucket, 0)
    class_counts = Counter(str(event.get("classification") or "UNKNOWN") for event in evolve_events)
    final_split: dict[str, dict[str, int]] = {}
    for event in evolve_events:
        klass = str(event.get("classification") or "UNKNOWN")
        final_split.setdefault(klass, {})
        result = str(event.get("source_trajectory_final_result"))
        final_split[klass][result] = final_split[klass].get(result, 0) + 1
    good = class_counts.get(DeltaClassification.GOOD_EVOLVE_CORRECTION.value, 0)
    premature = (
        class_counts.get(DeltaClassification.PREMATURE_EVOLVE_BEFORE_VALUABLE_PLAY.value, 0)
        + class_counts.get(DeltaClassification.PREMATURE_EVOLVE_BEFORE_TARGET_CREATION.value, 0)
        + class_counts.get(DeltaClassification.ABLATED_SKIPPED_USEFUL_DRAW_OR_SEARCH.value, 0)
        + class_counts.get(DeltaClassification.ABLATED_SKIPPED_TARGET_CREATING_PLAY.value, 0)
    )
    neutral = class_counts.get(DeltaClassification.NEUTRAL_EVOLVE_SAME_TURN_ANYWAY.value, 0)
    if premature > good:
        recommendation = "revise rule"
        interpretation = "the ablation frequently skips useful sequencing or target creation before EVOLVE"
    elif good > 0 and premature == 0:
        recommendation = "expand eval"
        interpretation = "same-frame deltas look mostly benign or corrective, but outcome evidence remains weak"
    else:
        recommendation = "keep experimental"
        interpretation = "the same-frame evidence is mixed and not enough for promotion"
    metrics = {
        "schema_version": DELTA_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "source_agents": list(source_agents),
        "default_policy": default_policy,
        "ablation_policy": ablation_policy,
        "total_frames_collected": len(frames),
        "total_frames_probed": total_probed,
        "total_delta_events": len(events),
        "total_evolve_priority_delta_events": len(evolve_events),
        "default_play_to_ablation_evolve_count": action_counts.get("PLAY->EVOLVE", 0),
        "default_attach_to_ablation_evolve_count": action_counts.get("ATTACH->EVOLVE", 0),
        "default_attack_to_ablation_evolve_count": action_counts.get("ATTACK->EVOLVE", 0),
        "default_evolve_to_ablation_non_evolve_count": sum(
            1
            for event in evolve_events
            if event.get("default_chosen_action_type") == "EVOLVE" and event.get("ablation_chosen_action_type") != "EVOLVE"
        ),
        "good_evolve_correction_count": good,
        "premature_evolve_count": premature,
        "neutral_same_turn_correction_count": neutral,
        "unknown_count": class_counts.get(DeltaClassification.UNKNOWN.value, 0),
        "action_change_counts": dict(action_counts),
        "default_play_card_counts": dict(play_counts),
        "target_creating_play_skipped_count": sum(1 for event in play_to_evolve if event.get("default_chosen_card_id") in TARGET_CREATING_PLAY_IDS),
        "non_target_play_skipped_count": sum(1 for event in play_to_evolve if event.get("default_chosen_card_id") not in TARGET_CREATING_PLAY_IDS),
        "default_later_evolved_same_turn_count": sum(1 for event in evolve_events if event.get("default_later_evolve_same_turn_frame_id")),
        "default_later_evolved_same_game_count": sum(1 for event in evolve_events if event.get("default_later_evolve_same_game_frame_id")),
        "source_trajectory_final_result_split_by_classification": final_split,
        "score_gap_summary_by_classification": _score_gap_summary_by_class(evolve_events),
        "classification_counts": dict(class_counts),
        "interpretation": interpretation,
        "recommendation": recommendation,
    }
    return {"metrics": metrics, "report_answers": _report_answers(metrics)}


def _csv_text(events: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "source_agent",
        "game_id",
        "turn",
        "decision_index",
        "classification",
        "default_chosen_action_type",
        "default_chosen_card_name",
        "ablation_chosen_action_type",
        "ablation_chosen_card_name",
        "default_chosen_score",
        "ablation_chosen_score",
        "evolve_option_score",
        "default_chosen_minus_evolve_score_gap",
        "bench_count",
        "best_visible_attack_pressure",
        "default_later_evolve_same_turn_frame_id",
        "default_later_evolve_same_game_frame_id",
        "source_trajectory_final_result",
        "tags",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for event in events:
        writer.writerow({field: ";".join(event.get(field) or []) if field == "tags" else event.get(field) for field in fields})
    return stream.getvalue()


def _metric_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    keys = [
        "total_frames_probed",
        "total_delta_events",
        "total_evolve_priority_delta_events",
        "default_play_to_ablation_evolve_count",
        "default_attach_to_ablation_evolve_count",
        "default_attack_to_ablation_evolve_count",
        "default_evolve_to_ablation_non_evolve_count",
        "good_evolve_correction_count",
        "premature_evolve_count",
        "neutral_same_turn_correction_count",
        "unknown_count",
        "target_creating_play_skipped_count",
        "non_target_play_skipped_count",
        "default_later_evolved_same_turn_count",
        "default_later_evolved_same_game_count",
        "recommendation",
    ]
    return [[key, metrics.get(key)] for key in keys]


def build_delta_report(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    metrics = summary.get("metrics") or {}
    lines = [title("Deck Track V3.9 Evolve Priority Delta Audit")]
    lines.append(section("Scope"))
    lines.append(
        bullets(
            [
                "Diagnostic-only same-observation probe of `v2.4` and `v2.4[evolve_priority=new]`.",
                "Delta event means the two policies choose different actions on the exact same observation.",
                "Later source trajectory outcomes are recorded as weak evidence only.",
                "No gameplay behavior is changed by this audit.",
            ]
        )
    )
    lines.append(section("Headline Metrics"))
    lines.append(table(["Metric", "Value"], _metric_rows(metrics)))
    lines.append(section("Action Changes"))
    lines.append(table(["Change", "Count"], Counter(metrics.get("action_change_counts", {})).most_common() or [["none", 0]]))
    lines.append(section("Skipped PLAY Cards"))
    lines.append(table(["Card", "Count"], Counter(metrics.get("default_play_card_counts", {})).most_common()))
    lines.append(section("Classifications"))
    lines.append(table(["Class", "Count"], Counter(metrics.get("classification_counts", {})).most_common() or [["none", 0]]))
    lines.append(section("Score Gaps By Classification"))
    lines.append(
        table(
            ["Class", "Count", "Mean", "Median", "Min", "Max"],
            [
                [row["classification"], row["count"], row["mean_gap"], row["median_gap"], row["min_gap"], row["max_gap"]]
                for row in metrics.get("score_gap_summary_by_classification", [])
            ]
            or [["none", 0, "none", "none", "none", "none"]],
        )
    )
    lines.append(section("Questions Answered"))
    lines.append(bullets(summary.get("report_answers") or []))
    lines.append(section("Example Frames"))
    for event in _example_rows(events):
        lines.append(f"### {event.get('frame_id')}\n")
        lines.append(
            bullets(
                [
                    f"class={event.get('classification')} tags={event.get('tags')}",
                    f"default={event.get('default_chosen_action_type')} {event.get('default_chosen_card_name') or event.get('default_chosen_attack_name') or ''} score={event.get('default_chosen_score')}",
                    f"ablation={event.get('ablation_chosen_action_type')} {event.get('ablation_chosen_card_name') or event.get('ablation_chosen_attack_name') or ''} score={event.get('ablation_chosen_score')}",
                    f"board={event.get('board_plan_state')} bench={event.get('bench_count')} attack={event.get('best_visible_attack_pressure')}",
                    f"later_same_turn={event.get('default_later_evolve_same_turn_frame_id')} later_same_game={event.get('default_later_evolve_same_game_frame_id')}",
                    f"changed_scores={event.get('changed_scores')}",
                ]
            )
        )
    return "\n".join(lines)


def _example_rows(events: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def add_matching(predicate: Callable[[dict[str, Any]], bool], count: int) -> None:
        for event in events:
            if len(selected) >= limit or sum(1 for item in selected if predicate(item)) >= count:
                return
            if predicate(event) and event not in selected:
                selected.append(event)

    add_matching(lambda e: e.get("default_chosen_action_type") == "PLAY" and e.get("ablation_chosen_action_type") == "EVOLVE", 5)
    add_matching(lambda e: e.get("default_chosen_card_name") == "Lillie's Determination", 3)
    add_matching(lambda e: e.get("default_chosen_card_name") == "Mega Signal", 3)
    add_matching(lambda e: e.get("default_chosen_card_name") in {"Waitress", "Cyrano"}, 3)
    add_matching(lambda e: e.get("classification") == DeltaClassification.GOOD_EVOLVE_CORRECTION.value, 3)
    add_matching(
        lambda e: e.get("classification")
        in {
            DeltaClassification.ABLATED_SKIPPED_USEFUL_DRAW_OR_SEARCH.value,
            DeltaClassification.ABLATED_SKIPPED_TARGET_CREATING_PLAY.value,
            DeltaClassification.PREMATURE_EVOLVE_BEFORE_TARGET_CREATION.value,
            DeltaClassification.PREMATURE_EVOLVE_BEFORE_VALUABLE_PLAY.value,
        },
        3,
    )
    add_matching(lambda e: e.get("classification") == DeltaClassification.NEUTRAL_EVOLVE_SAME_TURN_ANYWAY.value, 3)
    add_matching(lambda e: "target_creating_play_skipped" in (e.get("tags") or []), 3)
    add_matching(lambda e: e.get("default_chosen_action_type") in {"ATTACH", "ATTACK"}, 3)
    for event in events:
        if len(selected) >= limit:
            break
        if event not in selected:
            selected.append(event)
    return selected[:limit]


def _example_frames_markdown(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    _ = summary
    lines = ["# Example Frames", "", "Same-observation delta examples; source trajectory outcomes are weak evidence only.", ""]
    for event in _example_rows(events):
        lines.extend(
            [
                f"## {event.get('frame_id')}",
                "",
                f"- classification: {event.get('classification')}",
                f"- tags: {event.get('tags')}",
                f"- source: {event.get('source_agent')} game={event.get('game_id')} turn={event.get('turn')} decision={event.get('decision_index')}",
                f"- default: {event.get('default_chosen_action_type')} {event.get('default_chosen_card_name') or event.get('default_chosen_attack_name') or ''} score={event.get('default_chosen_score')}",
                f"- ablation: {event.get('ablation_chosen_action_type')} {event.get('ablation_chosen_card_name') or event.get('ablation_chosen_attack_name') or ''} score={event.get('ablation_chosen_score')}",
                f"- evolve score: {event.get('evolve_option_score')} gaps default={event.get('default_chosen_minus_evolve_score_gap')} ablation={event.get('ablation_chosen_minus_evolve_score_gap')}",
                f"- board: {event.get('board_plan_state')} bench={event.get('bench_count')} active={event.get('active_card_name')} energy={event.get('active_energy_count')}",
                f"- later evolve same turn: {event.get('default_later_evolve_same_turn_frame_id')}",
                f"- later evolve same game: {event.get('default_later_evolve_same_game_frame_id')}",
                "",
            ]
        )
    return "\n".join(lines)


def _readme(command: str) -> str:
    return "\n".join(
        [
            "# BabyMew Evolve Priority Delta Review Bundle",
            "",
            "This bundle contains a diagnostic same-observation probe comparing `v2.4` and `v2.4[evolve_priority=new]`.",
            "",
            f"- generated by: `{command}`",
            "- default behavior is unchanged; this is not a promotion.",
            "",
            "Inspect first:",
            "",
            "1. `executive_summary.md`",
            "2. `delta_summary.json`",
            "3. `delta_events.csv`",
            "4. `example_frames.md`",
            "",
        ]
    )


def _executive_summary(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics") or {}
    return "\n".join(
        [
            "# Executive Summary",
            "",
            "This is a diagnostic delta audit, not a behavior patch.",
            "",
            f"- frames probed: {metrics.get('total_frames_probed')}",
            f"- delta events: {metrics.get('total_delta_events')}",
            f"- evolve-priority delta events: {metrics.get('total_evolve_priority_delta_events')}",
            f"- action changes: {metrics.get('action_change_counts')}",
            f"- classification counts: {metrics.get('classification_counts')}",
            f"- skipped PLAY cards: {metrics.get('default_play_card_counts')}",
            f"- interpretation: {metrics.get('interpretation')}",
            f"- recommendation: {metrics.get('recommendation')}",
            "",
        ]
    )


def _relevant_paths(run_dir: Path, bundle_dir: Path | None) -> str:
    paths = [
        run_dir / "report.md",
        run_dir / "delta_summary.json",
        run_dir / "delta_events.jsonl",
        run_dir / "state_machine_snapshots.jsonl",
        bundle_dir or "",
        Path("deepmew/trace/evolve_priority_delta.py"),
        Path("scripts/diagnose_evolve_priority_delta.py"),
        Path("deepmew/policy_rules/evolve_priority.py"),
        Path("tests/test_evolve_priority_delta.py"),
        Path(".research/TASKS.md"),
        Path(".research/STATE.md"),
        Path(".research/CURRENT_STATE_COMPACT.md"),
        Path("HANDOFF_SUMMARY.md"),
    ]
    return "\n".join(str(path) for path in paths if str(path)) + "\n"


def write_delta_outputs(
    *,
    run_dir: Path,
    run_id: str,
    command: str,
    frames: list[dict[str, Any]],
    source_agents: Iterable[str] | None = None,
    inputs: dict[str, Any] | None = None,
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> EvolvePriorityDeltaResult:
    summary, events, snapshots = analyze_evolve_priority_delta(
        run_id=run_id,
        frames=frames,
        source_agents=source_agents,
    )
    run_path = Path(run_dir)
    outputs = {
        "report_markdown": str(run_path / "report.md"),
        "delta_summary_json": str(run_path / "delta_summary.json"),
        "delta_events_jsonl": str(run_path / "delta_events.jsonl"),
        "delta_events_csv": str(run_path / "delta_events.csv"),
        "state_machine_snapshots_jsonl": str(run_path / "state_machine_snapshots.jsonl"),
        "example_frames_markdown": str(run_path / "example_frames.md"),
    }
    write_json(run_path / "delta_summary.json", summary)
    write_jsonl(run_path / "delta_events.jsonl", events)
    write_text(run_path / "delta_events.csv", _csv_text(events))
    write_jsonl(run_path / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(run_path / "report.md", build_delta_report(summary, events))
    write_markdown(run_path / "example_frames.md", _example_frames_markdown(summary, events))

    bundle_dir = None
    if export_handoff_bundle:
        bundle_dir = export_delta_bundle(
            summary=summary,
            events=events,
            snapshots=snapshots,
            run_dir=run_path,
            command=command,
            out_root=handoff_root,
        )
        outputs["handoff_bundle"] = str(bundle_dir)

    manifest = build_manifest(
        run_id=run_id,
        kind=RUN_KIND,
        command=command,
        inputs=inputs or {"source_agents": list(source_agents or DEFAULT_SOURCE_AGENTS)},
        outputs=outputs,
        summary=summary.get("metrics") or {},
    )
    write_manifest(run_path / "manifest.json", manifest)
    return EvolvePriorityDeltaResult(run_dir=run_path, manifest_path=run_path / "manifest.json", summary=summary, events=events, bundle_dir=bundle_dir)


def export_delta_bundle(
    *,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    run_dir: Path,
    command: str,
    out_root: Path | str = ".research/handoff_bundles",
) -> Path:
    bundle_dir = _create_handoff_dir(out_root=out_root)
    write_markdown(bundle_dir / "README.md", _readme(command))
    write_markdown(bundle_dir / "executive_summary.md", _executive_summary(summary))
    write_json(bundle_dir / "delta_summary.json", summary)
    write_text(bundle_dir / "delta_events.csv", _csv_text(events))
    write_jsonl(bundle_dir / "delta_events.jsonl", events)
    write_jsonl(bundle_dir / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(bundle_dir / "example_frames.md", _example_frames_markdown(summary, events))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(run_dir, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=HANDOFF_KIND,
        command=command,
        inputs={"delta_run": str(run_dir)},
        outputs={name: str(bundle_dir / name) for name in REQUIRED_BUNDLE_FILES if name != "manifest.json"},
        summary=summary.get("metrics") or {},
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def collect_delta_frames(
    *,
    run_id: str,
    source_agents: Iterable[str],
    games: int,
    seed: int,
    max_steps: int,
    limit_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    recorder = TraceRecorder(run_id=run_id, sample_limit=limit_frames)
    pair_summaries = []
    for index, source_agent in enumerate(source_agents):
        pair_seed = seed + index * 10007
        opponent = "first-legal" if source_agent != "first-legal" else DEFAULT_POLICY
        prefix = f"evolvedelta-{index:02d}-{safe_slug(source_agent)}-vs-{safe_slug(opponent)}"
        pair = run_pair(
            source_agent,
            opponent,
            games=games,
            max_steps=max_steps,
            seed=pair_seed,
            trace_recorder=recorder,
            game_id_prefix=prefix,
        )
        pair_summaries.append(pair)
    frames = []
    for frame in recorder.frames:
        payload = frame.to_jsonable()
        payload["source_agent"] = payload.get("agent_label")
        payload["matchup"] = payload.get("game_id", "").split("-000", 1)[0]
        frames.append(payload)
    return frames, pair_summaries, recorder.summary()
