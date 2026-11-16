"""Whole-rulebook diagnostic audit for BabyMew policies.

This module is intentionally diagnostic-only.  It probes several policy specs
on identical decision frames, inventories current rule families, and writes
evidence artifacts without changing gameplay behavior.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from cg.api import OptionType

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
from deepmew.policy.attacks import attack_pressure
from deepmew.policy_rules.attachment_timing import classify_attach_timing, extract_attach_timing_facts
from deepmew.policy_rules.evolve_priority import classify_evolve_priority, extract_evolve_priority_facts
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board, describe_plan
from deepmew.trace.attach_target_quality import build_target_candidates
from deepmew.trace.probe import probe_frame
from deepmew.trace.record import TraceRecorder
from deepmew.trace.schema import to_jsonable


RUN_KIND = "babymew_rulebook_audit"
HANDOFF_KIND = "babymew_rulebook_audit_review"
SUMMARY_SCHEMA_VERSION = "deepmew.rulebook_audit.v1"
DELTA_EVENT_SCHEMA_VERSION = "deepmew.rulebook_delta_event.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "deepmew.rulebook_snapshot.v1"
RULE_INVENTORY_SCHEMA_VERSION = "deepmew.rule_inventory.v1"
RULE_IMPACT_SCHEMA_VERSION = "deepmew.rule_impact.v1"

DEFAULT_POLICIES = (
    "v2.4",
    "first-legal",
    "v2.4[attach_timing=new]",
    "v2.4[evolve_priority=new]",
    "v2.4[semantics=off]",
)
DEFAULT_SOURCE_AGENTS = DEFAULT_POLICIES
DEFAULT_POLICY = "v2.4"

TARGET_CREATING_PLAY_IDS = frozenset({SNOVER_CARD_ID, KYOGRE_CARD_ID})
USEFUL_DRAW_OR_SEARCH_IDS = frozenset(
    {
        LILLIES_DETERMINATION_CARD_ID,
        WAITRESS_CARD_ID,
        CYRANO_CARD_ID,
        MEGA_SIGNAL_CARD_ID,
        MAXIMUM_BELT_CARD_ID,
    }
)
CARD_BUCKETS = {
    LILLIES_DETERMINATION_CARD_ID: "Lillie's Determination",
    WAITRESS_CARD_ID: "Waitress",
    CYRANO_CARD_ID: "Cyrano",
    MEGA_SIGNAL_CARD_ID: "Mega Signal",
    MAXIMUM_BELT_CARD_ID: "Maximum Belt",
    SNOVER_CARD_ID: "Snover",
    KYOGRE_CARD_ID: "Kyogre",
}

REQUIRED_RUN_FILES = [
    "manifest.json",
    "report.md",
    "rule_inventory.json",
    "rule_inventory.md",
    "decision_delta_summary.json",
    "decision_delta_events.jsonl",
    "decision_delta_events.csv",
    "rule_impact_summary.json",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
]
REQUIRED_BUNDLE_FILES = [
    "manifest.json",
    "README.md",
    "executive_summary.md",
    "rule_inventory.md",
    "rule_inventory.json",
    "decision_delta_summary.json",
    "decision_delta_events.csv",
    "decision_delta_events.jsonl",
    "rule_impact_summary.json",
    "state_machine_snapshots.jsonl",
    "example_frames.md",
    "relevant_paths.txt",
]


class DeltaClass(str, Enum):
    NO_DIFFERENCE = "NO_DIFFERENCE"
    FIRST_LEGAL_ATTACHES_DEFAULT_PLAYS = "FIRST_LEGAL_ATTACHES_DEFAULT_PLAYS"
    FIRST_LEGAL_ATTACHES_DEFAULT_ATTACKS = "FIRST_LEGAL_ATTACHES_DEFAULT_ATTACKS"
    FIRST_LEGAL_ATTACHES_DEFAULT_EVOLVES = "FIRST_LEGAL_ATTACHES_DEFAULT_EVOLVES"
    DEFAULT_ATTACKS_FIRST_LEGAL_DEVELOPS = "DEFAULT_ATTACKS_FIRST_LEGAL_DEVELOPS"
    DEFAULT_PLAYS_FIRST_LEGAL_ATTACHES = "DEFAULT_PLAYS_FIRST_LEGAL_ATTACHES"
    DEFAULT_EVOLVES_FIRST_LEGAL_PLAYS = "DEFAULT_EVOLVES_FIRST_LEGAL_PLAYS"
    ATTACH_TIMING_DELTA = "ATTACH_TIMING_DELTA"
    EVOLVE_PRIORITY_DELTA = "EVOLVE_PRIORITY_DELTA"
    ATTACK_PRESSURE_DELTA = "ATTACK_PRESSURE_DELTA"
    SEMANTICS_DELTA = "SEMANTICS_DELTA"
    ATTACH_TARGET_DELTA = "ATTACH_TARGET_DELTA"
    TRAINER_SEQUENCING_DELTA = "TRAINER_SEQUENCING_DELTA"
    STATE_MACHINE_READINESS_DELTA = "STATE_MACHINE_READINESS_DELTA"
    UNKNOWN_DELTA = "UNKNOWN_DELTA"


@dataclass(frozen=True)
class RulebookDeltaFacts:
    default_type: str | None
    first_legal_type: str | None
    attach_timing_type: str | None
    evolve_priority_type: str | None
    semantics_off_type: str | None
    default_card_id: int | None
    any_difference: bool
    first_legal_differs: bool
    attach_timing_differs: bool
    evolve_priority_differs: bool
    semantics_differs: bool
    attach_target_differs: bool
    ko_or_meaningful_attack_available: bool
    attach_legal: bool
    important_evolve_legal: bool
    readiness_state: str | None


@dataclass(frozen=True)
class RulebookDeltaRule:
    name: str
    guard: Callable[[RulebookDeltaFacts], bool]
    classification: DeltaClass
    family: str


@dataclass(frozen=True)
class RulebookAuditResult:
    run_dir: Path
    manifest_path: Path
    summary: dict[str, Any]
    rule_impact_summary: dict[str, Any]
    events: list[dict[str, Any]]
    bundle_dir: Path | None = None


def _no_difference(facts: RulebookDeltaFacts) -> bool:
    return not facts.any_difference


def _first_legal_attaches_default_plays(facts: RulebookDeltaFacts) -> bool:
    return facts.first_legal_type == "ATTACH" and facts.default_type == "PLAY"


def _first_legal_attaches_default_attacks(facts: RulebookDeltaFacts) -> bool:
    return facts.first_legal_type == "ATTACH" and facts.default_type == "ATTACK"


def _first_legal_attaches_default_evolves(facts: RulebookDeltaFacts) -> bool:
    return facts.first_legal_type == "ATTACH" and facts.default_type == "EVOLVE"


def _default_attacks_first_legal_develops(facts: RulebookDeltaFacts) -> bool:
    return facts.default_type == "ATTACK" and facts.first_legal_type in {"ATTACH", "PLAY", "EVOLVE"}


def _default_evolves_first_legal_plays(facts: RulebookDeltaFacts) -> bool:
    return facts.default_type == "EVOLVE" and facts.first_legal_type == "PLAY"


def _attach_timing_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.attach_timing_differs


def _evolve_priority_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.evolve_priority_differs


def _semantics_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.semantics_differs


def _attach_target_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.attach_target_differs


def _attack_pressure_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.ko_or_meaningful_attack_available and facts.default_type == "ATTACK"


def _state_machine_readiness_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.readiness_state in {"ATTACK_READY", "ATTACKER_NEAR_READY", "NO_ATTACKER_READY"} and facts.default_type != facts.first_legal_type


def _trainer_sequencing_delta(facts: RulebookDeltaFacts) -> bool:
    return facts.default_type == "PLAY" or facts.first_legal_type == "PLAY"


def _always(_: RulebookDeltaFacts) -> bool:
    return True


RULEBOOK_DELTA_RULES: tuple[RulebookDeltaRule, ...] = (
    RulebookDeltaRule("no-difference", _no_difference, DeltaClass.NO_DIFFERENCE, "unknown"),
    RulebookDeltaRule(
        "first-legal-attaches-default-plays",
        _first_legal_attaches_default_plays,
        DeltaClass.FIRST_LEGAL_ATTACHES_DEFAULT_PLAYS,
        "trainer_sequencing",
    ),
    RulebookDeltaRule(
        "first-legal-attaches-default-attacks",
        _first_legal_attaches_default_attacks,
        DeltaClass.FIRST_LEGAL_ATTACHES_DEFAULT_ATTACKS,
        "attack_pressure",
    ),
    RulebookDeltaRule(
        "first-legal-attaches-default-evolves",
        _first_legal_attaches_default_evolves,
        DeltaClass.FIRST_LEGAL_ATTACHES_DEFAULT_EVOLVES,
        "attachment_timing_old",
    ),
    RulebookDeltaRule(
        "default-attacks-first-legal-develops",
        _default_attacks_first_legal_develops,
        DeltaClass.DEFAULT_ATTACKS_FIRST_LEGAL_DEVELOPS,
        "attack_pressure",
    ),
    RulebookDeltaRule(
        "default-evolves-first-legal-plays",
        _default_evolves_first_legal_plays,
        DeltaClass.DEFAULT_EVOLVES_FIRST_LEGAL_PLAYS,
        "trainer_sequencing",
    ),
    RulebookDeltaRule("attach-timing-ablation-delta", _attach_timing_delta, DeltaClass.ATTACH_TIMING_DELTA, "attachment_timing_new"),
    RulebookDeltaRule("evolve-priority-ablation-delta", _evolve_priority_delta, DeltaClass.EVOLVE_PRIORITY_DELTA, "evolve_priority_new"),
    RulebookDeltaRule("semantics-off-delta", _semantics_delta, DeltaClass.SEMANTICS_DELTA, "card_semantics"),
    RulebookDeltaRule("attach-target-delta", _attach_target_delta, DeltaClass.ATTACH_TARGET_DELTA, "attach_target"),
    RulebookDeltaRule("attack-pressure-delta", _attack_pressure_delta, DeltaClass.ATTACK_PRESSURE_DELTA, "attack_pressure"),
    RulebookDeltaRule(
        "state-machine-readiness-delta",
        _state_machine_readiness_delta,
        DeltaClass.STATE_MACHINE_READINESS_DELTA,
        "state_machine_readiness",
    ),
    RulebookDeltaRule("trainer-sequencing-delta", _trainer_sequencing_delta, DeltaClass.TRAINER_SEQUENCING_DELTA, "trainer_sequencing"),
    RulebookDeltaRule("unknown-delta", _always, DeltaClass.UNKNOWN_DELTA, "unknown"),
)


RULE_FAMILIES = (
    "base_action_priority",
    "attack_pressure",
    "card_semantics",
    "state_machine_readiness",
    "attach_target",
    "attachment_timing_old",
    "attachment_timing_new",
    "evolve_priority_new",
    "play_identity",
    "trainer_sequencing",
    "first_player_choice",
    "unknown",
)


def classify_rulebook_delta(facts: RulebookDeltaFacts) -> dict[str, str]:
    for rule in RULEBOOK_DELTA_RULES:
        if rule.guard(facts):
            return {
                "classification": rule.classification.value,
                "classification_reason": rule.name,
                "likely_rule_family": rule.family,
            }
    raise AssertionError("unreachable rulebook delta classifier")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_handoff_dir(*, out_root: Path | str = ".research/handoff_bundles", now: datetime | None = None) -> Path:
    created = now or _utc_now()
    bundle_dir = Path(out_root) / created.strftime("%Y%m%d") / make_run_id(HANDOFF_KIND, now=created)
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


def _option_type(option: dict[str, Any]) -> str | None:
    try:
        return OptionType(option.get("type")).name
    except (TypeError, ValueError):
        return None


def _policy_modes(policy: str) -> tuple[bool, str, str]:
    return (
        "semantics=off" not in policy,
        "new" if "attach_timing=new" in policy else "old",
        "new" if "evolve_priority=new" in policy else "old",
    )


def _score_map(raw: dict[str, Any], policy: str) -> dict[int, Any]:
    use_semantics, attach_timing, evolve_priority = _policy_modes(policy)
    if policy == "first-legal":
        return {}
    return {
        candidate.index: candidate
        for candidate in score_candidates(
            raw,
            use_card_semantics=use_semantics,
            attach_timing=attach_timing,
            evolve_priority=evolve_priority,
        )
    }


def _top_candidate(candidates: dict[int, Any]) -> Any | None:
    if not candidates:
        return None
    return min(candidates.values(), key=lambda item: (-item.score, item.index))


def _candidate_payload(candidate: Any | None) -> dict[str, Any]:
    if candidate is None:
        return {"score": None, "reasons": []}
    return {
        "score": getattr(candidate, "score", None),
        "reasons": list(getattr(candidate, "reasons", ()) or ()),
        "option_type": getattr(candidate, "option_type", None),
    }


def _first_index(indices: Any) -> int | None:
    if isinstance(indices, list) and indices:
        return _int_or_none(indices[0])
    return None


def _probe_choice(frame: dict[str, Any], policy: str) -> dict[str, Any]:
    probe = probe_frame(frame, policy).to_jsonable()
    chosen = probe.get("chosen_option_summaries") or []
    summary = chosen[0] if chosen and isinstance(chosen[0], dict) else {}
    index = _int_or_none(summary.get("index")) if summary else _first_index(probe.get("chosen_indices"))
    return {
        "policy": probe.get("policy_spec") or policy,
        "indices": probe.get("chosen_indices") or [],
        "option_index": index,
        "option_type": summary.get("option_type"),
        "card_id": _int_or_none(summary.get("card_id")),
        "card_name": summary.get("card_name"),
        "attack_id": _int_or_none(summary.get("attack_id")),
        "attack_name": summary.get("attack_name"),
        "target_area": summary.get("target_area"),
        "target_index": _int_or_none(summary.get("target_index")),
        "target_card_id": _int_or_none(summary.get("target_card_id")),
        "target_card_name": summary.get("target_card_name"),
        "summary": summary,
        "valid_return": probe.get("valid_return"),
        "error": probe.get("error"),
    }


def _same_choice(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return not left.get("error") and not right.get("error") and left.get("indices") == right.get("indices")


def _score_change_rows(default_scores: dict[int, Any], other_scores: dict[int, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in sorted(set(default_scores) | set(other_scores)):
        left = default_scores.get(index)
        right = other_scores.get(index)
        if left is None or right is None:
            continue
        if left.score != right.score or tuple(left.reasons) != tuple(right.reasons):
            rows.append(
                {
                    "index": index,
                    "option_type": right.option_type,
                    "default_score": left.score,
                    "other_score": right.score,
                    "delta": right.score - left.score,
                    "default_reasons": list(left.reasons),
                    "other_reasons": list(right.reasons),
                }
            )
    return rows


def _frame_groups(frames: list[dict[str, Any]]) -> dict[tuple[str, str, int | None], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        key = (
            str(frame.get("source_agent") or frame.get("agent_label") or ""),
            str(frame.get("game_id") or ""),
            _int_or_none(frame.get("seat")),
        )
        groups[key].append(frame)
    for group in groups.values():
        group.sort(key=lambda item: _int_or_none(item.get("decision_index")) or -1)
    return groups


def _later_actions(frame: dict[str, Any], group: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    current_decision = _int_or_none(frame.get("decision_index")) or -1
    current_turn = _int_or_none(frame.get("turn"))
    rows: list[dict[str, Any]] = []
    for later in group:
        later_decision = _int_or_none(later.get("decision_index")) or -1
        if later_decision <= current_decision:
            continue
        if current_turn != _int_or_none(later.get("turn")):
            continue
        chosen = later.get("chosen_option_summaries") or []
        rows.append(
            {
                "frame_id": later.get("frame_id"),
                "decision_index": later.get("decision_index"),
                "chosen": chosen[0] if chosen else {},
            }
        )
        if len(rows) >= limit:
            break
    return rows


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


def _attach_timing_payload(raw: dict[str, Any], plan: Any | None) -> tuple[dict[str, Any], dict[str, Any]]:
    select = raw.get("select") if isinstance(raw.get("select"), dict) else {}
    options = select.get("option") or []
    facts = extract_attach_timing_facts(raw, select, options, plan, {"use_card_semantics": True})
    decision = classify_attach_timing(facts)
    return to_jsonable(facts), to_jsonable(decision)


def _evolve_priority_payload(raw: dict[str, Any], plan: Any | None) -> tuple[dict[str, Any], dict[str, Any]]:
    select = raw.get("select") if isinstance(raw.get("select"), dict) else {}
    options = select.get("option") or []
    facts = extract_evolve_priority_facts(raw, select, options, plan, use_card_semantics=True)
    decision = classify_evolve_priority(facts)
    return to_jsonable(facts), to_jsonable(decision)


def _attack_pressures(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, option in enumerate((raw.get("select") or {}).get("option") or []):
        if _option_type(option) != "ATTACK":
            continue
        rows.append(
            {
                "index": index,
                "attack_id": _int_or_none(option.get("attackId")),
                "pressure": attack_pressure(option, raw, True),
                "pressure_semantics_off": attack_pressure(option, raw, False),
            }
        )
    return rows


def _best_attach_target(frame: dict[str, Any]) -> dict[str, Any] | None:
    try:
        candidates = build_target_candidates(frame)
    except Exception:
        return None
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item.audit_score, -item.option_index))
    return best.to_jsonable()


def _score_payloads(raw: dict[str, Any], policies: Iterable[str]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for policy in policies:
        scores = _score_map(raw, policy)
        top = _top_candidate(scores)
        payloads[policy] = {
            "top": {
                "index": getattr(top, "index", None),
                "option_type": getattr(top, "option_type", None),
                **_candidate_payload(top),
            },
            "by_index": {
                str(index): {
                    "option_type": candidate.option_type,
                    "score": candidate.score,
                    "reasons": list(candidate.reasons),
                }
                for index, candidate in sorted(scores.items())
            },
        }
    return payloads


def _choice_payload(
    frame: dict[str, Any],
    raw: dict[str, Any],
    policy: str,
    default_scores: dict[int, Any],
) -> dict[str, Any]:
    choice = _probe_choice(frame, policy)
    scores = _score_map(raw, policy)
    candidate = scores.get(choice["option_index"]) if choice["option_index"] is not None else None
    choice.update(_candidate_payload(candidate))
    if policy not in {DEFAULT_POLICY, "first-legal"}:
        choice["score_changes_vs_default"] = _score_change_rows(default_scores, scores)
    else:
        choice["score_changes_vs_default"] = []
    return choice


def _source_result_label(frame: dict[str, Any]) -> str:
    result = frame.get("final_result")
    if result is None:
        return "none"
    seat = _int_or_none(frame.get("seat"))
    if result == 2:
        return "draw"
    if seat is None:
        return str(result)
    return "source_win" if result == seat else "source_loss"


def _facts_from_choices(
    choices: dict[str, dict[str, Any]],
    *,
    attach_legal: bool,
    important_evolve_legal: bool,
    readiness_state: str | None,
    attack_rows: list[dict[str, Any]],
) -> RulebookDeltaFacts:
    default = choices[DEFAULT_POLICY]
    first = choices["first-legal"]
    attach_timing = choices["v2.4[attach_timing=new]"]
    evolve_priority = choices["v2.4[evolve_priority=new]"]
    semantics_off = choices["v2.4[semantics=off]"]
    policy_indices = {policy: tuple(choice.get("indices") or []) for policy, choice in choices.items()}
    target_default = (default.get("option_type"), default.get("target_area"), default.get("target_index"))
    target_new = (attach_timing.get("option_type"), attach_timing.get("target_area"), attach_timing.get("target_index"))
    return RulebookDeltaFacts(
        default_type=default.get("option_type"),
        first_legal_type=first.get("option_type"),
        attach_timing_type=attach_timing.get("option_type"),
        evolve_priority_type=evolve_priority.get("option_type"),
        semantics_off_type=semantics_off.get("option_type"),
        default_card_id=_int_or_none(default.get("card_id")),
        any_difference=len(set(policy_indices.values())) > 1,
        first_legal_differs=not _same_choice(default, first),
        attach_timing_differs=not _same_choice(default, attach_timing),
        evolve_priority_differs=not _same_choice(default, evolve_priority),
        semantics_differs=not _same_choice(default, semantics_off),
        attach_target_differs=target_default != target_new and default.get("option_type") == "ATTACH",
        ko_or_meaningful_attack_available=any(row.get("pressure") in {"ko", "meaningful"} for row in attack_rows),
        attach_legal=attach_legal,
        important_evolve_legal=important_evolve_legal,
        readiness_state=readiness_state,
    )


def _event_tags(event: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    default = event["policy_choices"].get(DEFAULT_POLICY, {})
    first = event["policy_choices"].get("first-legal", {})
    if event.get("classification"):
        tags.add(str(event["classification"]).lower())
    if default.get("option_type") and first.get("option_type") and default.get("option_type") != first.get("option_type"):
        tags.add(f"default_{str(default.get('option_type')).lower()}_first_{str(first.get('option_type')).lower()}")
    if event.get("attach_legal"):
        tags.add("attach_legal")
    if event.get("important_evolve_legal"):
        tags.add("important_evolve_legal")
    if event.get("best_visible_attack_pressure") in {"ko", "meaningful"}:
        tags.add("attack_pressure_available")
    card_id = _int_or_none(default.get("card_id"))
    if card_id in TARGET_CREATING_PLAY_IDS:
        tags.add("target_creating_play")
    if card_id in USEFUL_DRAW_OR_SEARCH_IDS:
        tags.add("useful_draw_or_search_play")
    if event.get("later_same_turn_actions"):
        tags.add("source_later_same_turn_actions")
    return sorted(tags)


def build_rulebook_delta_event(
    frame: dict[str, Any],
    *,
    group: list[dict[str, Any]] | None = None,
    policies: Iterable[str] = DEFAULT_POLICIES,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    raw = _raw_observation(frame)
    if not raw:
        return None
    policies = tuple(policies)
    default_scores = _score_map(raw, DEFAULT_POLICY)
    choices = {policy: _choice_payload(frame, raw, policy, default_scores) for policy in policies}
    if any(choice.get("error") for choice in choices.values()):
        return None

    plan, plan_summary, plan_text = _plan_payload(raw)
    attach_facts, attach_decision = _attach_timing_payload(raw, plan)
    evolve_facts, evolve_decision = _evolve_priority_payload(raw, plan)
    attack_rows = _attack_pressures(raw)
    best_attack_pressure = next((row["pressure"] for row in attack_rows if row.get("pressure") == "ko"), None)
    if best_attack_pressure is None:
        best_attack_pressure = next((row["pressure"] for row in attack_rows if row.get("pressure") == "meaningful"), None)
    if best_attack_pressure is None:
        best_attack_pressure = next((row["pressure"] for row in attack_rows), None)
    facts = _facts_from_choices(
        choices,
        attach_legal=bool(attach_facts.get("attach_legal")),
        important_evolve_legal=bool(evolve_facts.get("important_evolve_indices")),
        readiness_state=plan_summary.get("state"),
        attack_rows=attack_rows,
    )
    classification = classify_rulebook_delta(facts)
    later_same_turn = _later_actions(frame, group or [])
    event = {
        "schema_version": DELTA_EVENT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "source_agent": frame.get("source_agent") or frame.get("agent_label"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "seat": frame.get("seat"),
        "decision_index": frame.get("decision_index"),
        "select_type": frame.get("select_type"),
        "select_context": frame.get("select_context"),
        "board_summary": frame.get("board_summary") or {},
        "hand_visible_play_cards": _visible_play_cards(frame),
        "legal_options": frame.get("legal_options") or [],
        "policy_choices": choices,
        "score_payloads": _score_payloads(raw, policies),
        "attack_pressures": attack_rows,
        "best_visible_attack_pressure": best_attack_pressure,
        "important_evolve_legal": facts.important_evolve_legal,
        "attach_legal": facts.attach_legal,
        "energy_attached": plan_summary.get("energy_attached"),
        "best_attach_target": _best_attach_target(frame),
        "board_plan": plan_summary,
        "board_plan_description": plan_text,
        "attach_timing_decision": attach_decision,
        "evolve_priority_decision": evolve_decision,
        "classification": classification["classification"],
        "classification_reason": classification["classification_reason"],
        "likely_rule_family": classification["likely_rule_family"],
        "is_delta_event": facts.any_difference,
        "later_same_turn_actions": later_same_turn,
        "source_trajectory_final_result": _source_result_label(frame),
        "final_result": frame.get("final_result"),
        "final_result_reason": frame.get("final_result_reason"),
    }
    event["tags"] = _event_tags(event)
    snapshot = {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "source_agent": event["source_agent"],
        "board_plan": plan_summary,
        "board_plan_description": plan_text,
        "attach_timing_facts": attach_facts,
        "attach_timing_decision": attach_decision,
        "evolve_priority_facts": evolve_facts,
        "evolve_priority_decision": evolve_decision,
        "attack_pressures": attack_rows,
    }
    return event, snapshot


def _visible_play_cards(frame: dict[str, Any]) -> list[dict[str, Any]]:
    cards = []
    for option in frame.get("legal_options") or []:
        if option.get("option_type") != "PLAY":
            continue
        card_id = _int_or_none(option.get("card_id"))
        cards.append(
            {
                "index": option.get("index"),
                "card_id": card_id,
                "card_name": option.get("card_name") or CARD_BUCKETS.get(card_id),
                "source_area": option.get("source_area"),
                "source_index": option.get("source_index"),
            }
        )
    return cards


def rule_inventory() -> dict[str, Any]:
    rows = [
        _rule("base_action_priority.attack", "deepmew/policy/v2_scoring.py", "score_option_v2", "ATTACK", "any ATTACK option", "printed/effective damage base score", True, False, "take useful attacks", "supported", "Base damage score still needed; pressure overlays decide timing.", "V2.4 diagnostics"),
        _rule("base_action_priority.play", "deepmew/policy/v2_scoring.py", "score_option_v2", "PLAY", "any PLAY option", "+70 play-setup-card", True, False, "setup cards are generally strong", "stale", "Generic PLAY is too coarse; identity now resolves cards but effects are mostly unknown.", "PLAY identity review"),
        _rule("base_action_priority.evolve", "deepmew/policy/v2_scoring.py", "score_option_v2", "EVOLVE", "any EVOLVE option", "+68 develop-evolution", True, False, "evolution develops board", "supported", "Important Mega-line evolves are useful, but timing is nuanced.", "evolve sequencing audit"),
        _rule("base_action_priority.attach", "deepmew/policy/v2_scoring.py", "score_option_v2", "ATTACH", "any ATTACH option", "+62 develop-energy", True, False, "energy tempo matters", "supported", "Attachment timing is a known gap; target quality is not the largest current issue.", "attachment audits"),
        _rule("attack_pressure.visible_ko", "deepmew/policy/scoring.py", "_score_option", "ATTACK", "MAIN attack can KO opponent active", "+55", True, False, "take visible prizes", "supported", "Prior audits found missed visible KOs were not dominant.", "V2.4 diagnosis"),
        _rule("attack_pressure.meaningful_damage", "deepmew/policy/scoring.py", "_score_option", "ATTACK", "MAIN attack meaningful damage", "+22", True, False, "preserve pressure", "supported", "Works with printed/effective damage, still shallow.", "semantics diagnosis"),
        _rule("attack_pressure.weak_setup_delay", "deepmew/policy/scoring.py", "_score_option", "ATTACK", "weak attack and setup option exists", "-24", True, False, "setup before weak pressure", "supported", "Reduced shallow attack-ready mistakes, not enough to close first-legal gap.", "V2.4 diagnosis"),
        _rule("card_semantics.hammer_lanche", "deepmew/card_semantics/attacks.py", "effective_attack_semantics", "ATTACK", "Hammer-lanche printed zero", "deck-prior effective damage", True, False, "printed-zero attacks can be real attacks", "supported", "Semantics active; not the broad first-legal gap.", "Deck Track V3.1"),
        _rule("card_semantics.riptide", "deepmew/card_semantics/attacks.py", "effective_attack_semantics", "ATTACK", "Riptide visible discard water", "20 x visible Basic Water", True, False, "visible discard damage matters", "instrumentation-limited", "Only visible discard is counted.", "Deck Track V3.1"),
        _rule("state_machine.attack_ready", "deepmew/state_machine/transition.py", "TURN_STATE_RULES", "all", "legal attack or active shortfall 0", "TurnState.ATTACK_READY", True, False, "attacks are available now", "supported", "Table-driven now; still coarse for strategic timing.", "state-machine tests"),
        _rule("state_machine.near_ready", "deepmew/state_machine/transition.py", "TURN_STATE_RULES", "all", "best shortfall <= 1", "TurnState.ATTACKER_NEAR_READY", True, False, "one energy from pressure", "supported", "Useful for attach scoring and timing.", "attachment audits"),
        _rule("attach_target.closest_active", "deepmew/state_machine/effects.py", "attachment_effect", "ATTACH", "attach target improves readiness / active / closest", "+readiness boosts in policy scoring", True, False, "attach to visible attacker", "supported", "Default target quality was high in audit-best agreement.", "Deck Track V3.3"),
        _rule("attachment_timing.old_window", "deepmew/policy/scoring.py", "_score_option", "PLAY/EVOLVE", "attach legal and energy not attached", "-10 attachment-window-still-open", True, False, "do not forget energy attachment", "contradicted", "Too weak to beat generic PLAY often enough.", "Deck Track V3.4"),
        _rule("attachment_timing.new_bundle", "deepmew/policy_rules/attachment_timing.py", "ATTACH_TIMING_RULES", "ATTACH/PLAY/ATTACK", "guard table for attach timing", "+/-20..28 bundled deltas", False, True, "attach before generic development when safe", "dangerous bundle", "Experimental bundle mixes several ideas; evaluation mixed, not promoted.", "Deck Track V3.5"),
        _rule("evolve_priority.new_non_target_play", "deepmew/policy_rules/evolve_priority.py", "EVOLVE_PRIORITY_RULES", "PLAY", "important Mega-line EVOLVE legal and non-target PLAY", "-28 non-target PLAY", False, True, "do important EVOLVE before non-target PLAY", "contradicted", "Too blunt: often skips useful draw/search or compresses same-turn EVOLVE.", "Deck Track V3.9"),
        _rule("play_identity.visible_hand", "deepmew/options/identity.py", "resolve_option_identity", "PLAY", "PLAY index maps to visible hand", "identity metadata only", True, False, "diagnose real trainer cards", "supported", "Corrected earlier generic PLAY instrumentation.", "PLAY identity review"),
        _rule("trainer_sequencing.generic_play", "deepmew/policy/scoring.py", "_score_option", "PLAY", "PLAY before/after ATTACH/EVOLVE", "generic score and small state boosts", True, False, "play setup cards early", "untested", "Trainer effects remain mostly opaque, so same-frame PLAY deltas need caution.", "rulebook audit"),
        _rule("first_player_choice.second", "deepmew/policy/v2_scoring.py", "score_option_v2", "YES/NO", "IS_FIRST context", "+12 to NO", True, False, "prefer second for first attack", "untested", "First-player choice still appears in first-legal gap notes.", "V2.5-pre"),
        _rule("arena_ablation.attach_timing", "deepmew/arena/specs.py", "parse_agent_spec", "policy", "attach_timing=new/old/default", "explicit gated mode", True, True, "test isolated-ish rules", "supported", "Ablation is useful but attach_timing=new is not isolated enough for promotion.", "arena runs"),
        _rule("arena_ablation.evolve_priority", "deepmew/arena/specs.py", "parse_agent_spec", "policy", "evolve_priority=new/old/default", "explicit gated mode", True, True, "test EVOLVE sequencing idea", "supported", "Gated; default remains old.", "evolve priority runs"),
        _rule("arena_ablation.semantics", "deepmew/arena/specs.py", "parse_agent_spec", "policy", "semantics=off/on/default", "toggle card semantics", True, True, "measure printed-zero attack semantics", "supported", "Semantics-on remains default.", "semantics runs"),
    ]
    return {"schema_version": RULE_INVENTORY_SCHEMA_VERSION, "rules": rows}


def _rule(
    rule_id: str,
    module_path: str,
    function_or_table: str,
    affected_action: str,
    trigger_guard: str,
    score_effect: str,
    default_uses: bool,
    ablation_only: bool,
    strategic_belief: str,
    evidence_status: str,
    notes: str,
    related_reports_runs: str,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "module_path": module_path,
        "function_or_table": function_or_table,
        "affected_action": affected_action,
        "trigger_guard": trigger_guard,
        "score_or_effect": score_effect,
        "default_uses": default_uses,
        "ablation_only": ablation_only,
        "strategic_belief": strategic_belief,
        "evidence_status": evidence_status,
        "notes": notes,
        "related_reports_or_runs": related_reports_runs,
    }


def build_rule_inventory_markdown(inventory: dict[str, Any]) -> str:
    rows = [
        [
            row["rule_id"],
            row["module_path"],
            row["function_or_table"],
            row["affected_action"],
            row["score_or_effect"],
            row["default_uses"],
            row["ablation_only"],
            row["evidence_status"],
            row["notes"],
        ]
        for row in inventory.get("rules", [])
    ]
    return "\n".join(
        [
            title("BabyMew Rule Inventory"),
            "This is a diagnostic inventory of visible policy/rulebook logic, not a behavior patch.\n",
            table(
                [
                    "rule_id",
                    "module",
                    "function/table",
                    "action",
                    "score/effect",
                    "default",
                    "ablation",
                    "evidence",
                    "notes",
                ],
                rows,
            ),
        ]
    )


def analyze_rulebook_audit(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    source_agents: Iterable[str] = DEFAULT_SOURCE_AGENTS,
    policies: Iterable[str] = DEFAULT_POLICIES,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    groups = _frame_groups(frames)
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for frame in frames:
        key = (
            str(frame.get("source_agent") or frame.get("agent_label") or ""),
            str(frame.get("game_id") or ""),
            _int_or_none(frame.get("seat")),
        )
        built = build_rulebook_delta_event(frame, group=groups.get(key, []), policies=policies)
        if built is None:
            continue
        event, snapshot = built
        snapshots.append(snapshot)
        if event["is_delta_event"]:
            events.append(event)

    summary = build_decision_delta_summary(
        run_id=run_id,
        frames=frames,
        events=events,
        snapshots=snapshots,
        source_agents=source_agents,
        policies=policies,
    )
    impact = build_rule_impact_summary(events)
    return summary, events, impact, snapshots


def build_decision_delta_summary(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    source_agents: Iterable[str],
    policies: Iterable[str],
) -> dict[str, Any]:
    class_counts = Counter(str(event.get("classification")) for event in events)
    family_counts = Counter(str(event.get("likely_rule_family")) for event in events)
    action_flip_counts = Counter()
    first_gap_counts = Counter()
    final_split: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        default_type = event["policy_choices"][DEFAULT_POLICY].get("option_type")
        final_split[str(event.get("classification"))][str(event.get("source_trajectory_final_result"))] += 1
        for policy, choice in event["policy_choices"].items():
            if policy == DEFAULT_POLICY:
                continue
            other_type = choice.get("option_type")
            if other_type != default_type:
                action_flip_counts[f"{DEFAULT_POLICY}->{policy}:{default_type}->{other_type}"] += 1
        first = event["policy_choices"].get("first-legal", {}).get("option_type")
        if first != default_type:
            first_gap_counts[f"default_{default_type}_first_{first}"] += 1

    metrics = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "source_agents": list(source_agents),
        "policies": list(policies),
        "total_frames_collected": len(frames),
        "total_frames_analyzed": len(snapshots),
        "total_delta_events": len(events),
        "classification_counts": dict(class_counts),
        "likely_rule_family_counts": dict(family_counts),
        "action_flip_counts": dict(action_flip_counts),
        "first_legal_gap_counts": dict(first_gap_counts),
        "attach_legal_delta_events": sum(1 for event in events if event.get("attach_legal")),
        "important_evolve_delta_events": sum(1 for event in events if event.get("important_evolve_legal")),
        "semantics_delta_events": class_counts.get(DeltaClass.SEMANTICS_DELTA.value, 0),
        "attach_timing_delta_events": class_counts.get(DeltaClass.ATTACH_TIMING_DELTA.value, 0),
        "evolve_priority_delta_events": class_counts.get(DeltaClass.EVOLVE_PRIORITY_DELTA.value, 0),
        "source_trajectory_result_split_by_classification": {
            key: dict(value) for key, value in final_split.items()
        },
        "recommendation": _summary_recommendation(class_counts, family_counts),
    }
    return {"metrics": metrics, "report_answers": _report_answers(metrics)}


def _summary_recommendation(class_counts: Counter[str], family_counts: Counter[str]) -> str:
    if family_counts.get("trainer_sequencing", 0) or family_counts.get("attack_pressure", 0):
        return "continue diagnostics around trainer sequencing, attack pressure, and first-choice gaps; do not promote current experimental bundles"
    if class_counts.get(DeltaClass.EVOLVE_PRIORITY_DELTA.value, 0):
        return "revise evolve_priority=new before any promotion"
    if class_counts.get(DeltaClass.ATTACH_TIMING_DELTA.value, 0):
        return "split attach_timing=new into smaller isolated candidates before promotion"
    return "no default change from this audit"


def _report_answers(metrics: dict[str, Any]) -> list[str]:
    first_gap = Counter(metrics.get("first_legal_gap_counts") or {})
    families = Counter(metrics.get("likely_rule_family_counts") or {})
    return [
        f"Frames analyzed: {metrics.get('total_frames_analyzed')} with {metrics.get('total_delta_events')} same-observation delta events.",
        f"Top first-legal gaps: {dict(first_gap.most_common(5))}.",
        f"Top likely rule families: {dict(families.most_common(5))}.",
        "The first-legal gap is not corrected by treating option order as causal; it is reported as same-observation action differences only.",
        "attach_timing=new remains a dangerous bundle; evolve_priority=new remains too blunt and should be revised before any promotion.",
    ]


def build_rule_impact_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for family in RULE_FAMILIES:
        family_events = [event for event in events if event.get("likely_rule_family") == family]
        action_counts = Counter()
        result_split = Counter()
        for event in family_events:
            default_type = event["policy_choices"][DEFAULT_POLICY].get("option_type")
            result_split[str(event.get("source_trajectory_final_result"))] += 1
            for policy, choice in event["policy_choices"].items():
                if policy == DEFAULT_POLICY:
                    continue
                other_type = choice.get("option_type")
                if other_type != default_type:
                    action_counts[f"{default_type}->{other_type}"] += 1
        rows[family] = {
            "frames_active": len(family_events),
            "top_action_flips": sum(action_counts.values()),
            "common_flip_types": dict(action_counts.most_common(8)),
            "source_result_split": dict(result_split),
            "evidence_status": _family_evidence_status(family, len(family_events)),
            "recommendation": _family_recommendation(family, len(family_events)),
            "confidence": _family_confidence(family, len(family_events)),
        }
    return {"schema_version": RULE_IMPACT_SCHEMA_VERSION, "families": rows}


def _family_evidence_status(family: str, count: int) -> str:
    static = {
        "attachment_timing_new": "dangerous bundle",
        "evolve_priority_new": "contradicted",
        "play_identity": "supported",
        "card_semantics": "supported" if count else "instrumentation-limited",
        "first_player_choice": "untested",
        "trainer_sequencing": "untested" if count else "instrumentation-limited",
    }
    return static.get(family, "supported" if count else "instrumentation-limited")


def _family_recommendation(family: str, count: int) -> str:
    recommendations = {
        "attachment_timing_new": "do not promote; split into smaller candidates",
        "evolve_priority_new": "do not promote; revise to preserve useful setup/draw/search",
        "trainer_sequencing": "diagnose trainer identity/effects before score tuning",
        "first_player_choice": "audit separately with same-observation first-choice frames",
        "attach_target": "keep as secondary; current target quality looked mostly good",
    }
    if family in recommendations:
        return recommendations[family]
    if count:
        return "keep observing with equal-budget ablations before changing defaults"
    return "no action from this run"


def _family_confidence(family: str, count: int) -> str:
    if family in {"attachment_timing_new", "evolve_priority_new"}:
        return "medium"
    if count >= 20:
        return "medium"
    if count > 0:
        return "low"
    return "none"


def _csv_text(events: list[dict[str, Any]]) -> str:
    fields = [
        "frame_id",
        "source_agent",
        "game_id",
        "turn",
        "decision_index",
        "classification",
        "likely_rule_family",
        "default_action",
        "first_legal_action",
        "attach_timing_new_action",
        "evolve_priority_new_action",
        "semantics_off_action",
        "best_visible_attack_pressure",
        "attach_legal",
        "important_evolve_legal",
        "source_trajectory_final_result",
        "tags",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for event in events:
        choices = event.get("policy_choices") or {}
        writer.writerow(
            {
                "frame_id": event.get("frame_id"),
                "source_agent": event.get("source_agent"),
                "game_id": event.get("game_id"),
                "turn": event.get("turn"),
                "decision_index": event.get("decision_index"),
                "classification": event.get("classification"),
                "likely_rule_family": event.get("likely_rule_family"),
                "default_action": (choices.get(DEFAULT_POLICY) or {}).get("option_type"),
                "first_legal_action": (choices.get("first-legal") or {}).get("option_type"),
                "attach_timing_new_action": (choices.get("v2.4[attach_timing=new]") or {}).get("option_type"),
                "evolve_priority_new_action": (choices.get("v2.4[evolve_priority=new]") or {}).get("option_type"),
                "semantics_off_action": (choices.get("v2.4[semantics=off]") or {}).get("option_type"),
                "best_visible_attack_pressure": event.get("best_visible_attack_pressure"),
                "attach_legal": event.get("attach_legal"),
                "important_evolve_legal": event.get("important_evolve_legal"),
                "source_trajectory_final_result": event.get("source_trajectory_final_result"),
                "tags": ";".join(event.get("tags") or []),
            }
        )
    return stream.getvalue()


def _metric_rows(summary: dict[str, Any]) -> list[list[Any]]:
    metrics = summary.get("metrics") or {}
    keys = [
        "total_frames_collected",
        "total_frames_analyzed",
        "total_delta_events",
        "attach_legal_delta_events",
        "important_evolve_delta_events",
        "attach_timing_delta_events",
        "evolve_priority_delta_events",
        "semantics_delta_events",
        "recommendation",
    ]
    return [[key, metrics.get(key)] for key in keys]


def build_rulebook_report(
    summary: dict[str, Any],
    inventory: dict[str, Any],
    impact: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    metrics = summary.get("metrics") or {}
    lines = [title("BabyMew Rulebook Reset Audit")]
    lines.append(section("Scope"))
    lines.append(
        bullets(
            [
                "Diagnostic-only whole-rulebook audit of current BabyMew rules and gated ablations.",
                f"Policies probed on the same observations: {', '.join(metrics.get('policies') or [])}.",
                "Delta events are same-observation action differences, not full-game counterfactual claims.",
                "No gameplay/default behavior is changed by this audit.",
            ]
        )
    )
    lines.append(section("Headline Metrics"))
    lines.append(table(["Metric", "Value"], _metric_rows(summary)))
    lines.append(section("Classifications"))
    lines.append(table(["Class", "Count"], Counter(metrics.get("classification_counts") or {}).most_common() or [["none", 0]]))
    lines.append(section("Likely Rule Families"))
    lines.append(table(["Family", "Count"], Counter(metrics.get("likely_rule_family_counts") or {}).most_common() or [["none", 0]]))
    lines.append(section("First-Legal Gap"))
    lines.append(table(["Gap", "Count"], Counter(metrics.get("first_legal_gap_counts") or {}).most_common() or [["none", 0]]))
    lines.append(section("Rule Impact Summary"))
    rows = [
        [
            family,
            data["frames_active"],
            data["top_action_flips"],
            data["evidence_status"],
            data["recommendation"],
            data["confidence"],
        ]
        for family, data in (impact.get("families") or {}).items()
    ]
    lines.append(table(["Family", "Frames", "Flips", "Evidence", "Recommendation", "Confidence"], rows))
    lines.append(section("Inventory Evidence Status"))
    evidence_counts = Counter(row.get("evidence_status") for row in inventory.get("rules", []))
    lines.append(table(["Evidence Status", "Rule Count"], evidence_counts.most_common()))
    lines.append(section("Questions Answered"))
    lines.append(bullets(summary.get("report_answers") or []))
    lines.append(section("Example Frames"))
    for event in _example_rows(events):
        choices = event.get("policy_choices") or {}
        lines.append(f"### {event.get('frame_id')}\n")
        lines.append(
            bullets(
                [
                    f"class={event.get('classification')} family={event.get('likely_rule_family')} tags={event.get('tags')}",
                    f"default={(choices.get(DEFAULT_POLICY) or {}).get('option_type')} {(choices.get(DEFAULT_POLICY) or {}).get('card_name') or (choices.get(DEFAULT_POLICY) or {}).get('attack_name') or ''}",
                    f"first-legal={(choices.get('first-legal') or {}).get('option_type')}",
                    f"attach_timing=new={(choices.get('v2.4[attach_timing=new]') or {}).get('option_type')} evolve_priority=new={(choices.get('v2.4[evolve_priority=new]') or {}).get('option_type')} semantics=off={(choices.get('v2.4[semantics=off]') or {}).get('option_type')}",
                    f"board={event.get('board_plan', {}).get('state')} attack_pressure={event.get('best_visible_attack_pressure')} attach_legal={event.get('attach_legal')} important_evolve={event.get('important_evolve_legal')}",
                    f"later_same_turn={event.get('later_same_turn_actions')}",
                ]
            )
        )
    lines.append(section("Do Not Change"))
    lines.append(
        bullets(
            [
                "Do not promote `v2.4[attach_timing=new]`.",
                "Do not promote `v2.4[evolve_priority=new]`.",
                "Do not change deck.csv, cg/, trainer semantics, parser breadth, ML/search/replay/belief, or main.py eval behavior from this audit.",
            ]
        )
    )
    return "\n".join(lines)


def _example_rows(events: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def add(predicate: Callable[[dict[str, Any]], bool], count: int) -> None:
        for event in events:
            if len([item for item in selected if predicate(item)]) >= count or len(selected) >= limit:
                return
            if predicate(event) and event not in selected:
                selected.append(event)

    add(lambda event: event.get("classification") == DeltaClass.FIRST_LEGAL_ATTACHES_DEFAULT_PLAYS.value, 5)
    add(lambda event: event.get("classification") == DeltaClass.FIRST_LEGAL_ATTACHES_DEFAULT_ATTACKS.value, 5)
    add(lambda event: event.get("classification") == DeltaClass.ATTACH_TIMING_DELTA.value, 4)
    add(lambda event: event.get("classification") == DeltaClass.EVOLVE_PRIORITY_DELTA.value, 4)
    add(lambda event: event.get("classification") == DeltaClass.SEMANTICS_DELTA.value, 4)
    for event in events:
        if len(selected) >= limit:
            break
        if event not in selected:
            selected.append(event)
    return selected[:limit]


def _example_frames_markdown(events: list[dict[str, Any]]) -> str:
    lines = ["# Example Frames", "", "Same-observation delta examples; final results are weak trajectory context only.", ""]
    for event in _example_rows(events):
        choices = event.get("policy_choices") or {}
        default = choices.get(DEFAULT_POLICY) or {}
        first = choices.get("first-legal") or {}
        lines.extend(
            [
                f"## {event.get('frame_id')}",
                "",
                f"- classification: {event.get('classification')}",
                f"- likely rule family: {event.get('likely_rule_family')}",
                f"- default: {default.get('option_type')} {default.get('card_name') or default.get('attack_name') or ''}",
                f"- first-legal: {first.get('option_type')} {first.get('card_name') or first.get('attack_name') or ''}",
                f"- board: {event.get('board_plan', {}).get('state')} attach={event.get('attach_legal')} important_evolve={event.get('important_evolve_legal')}",
                f"- attack pressure: {event.get('best_visible_attack_pressure')}",
                f"- later same turn: {event.get('later_same_turn_actions')}",
                f"- tags: {event.get('tags')}",
                "",
            ]
        )
    return "\n".join(lines)


def _readme(command: str) -> str:
    return "\n".join(
        [
            "# BabyMew Rulebook Reset Audit Bundle",
            "",
            "This bundle contains a diagnostic-only whole-rulebook audit.",
            "",
            f"- generated by: `{command}`",
            "- default behavior is unchanged.",
            "- same-observation deltas are not full-game counterfactual proof.",
            "",
            "Inspect first:",
            "",
            "1. `executive_summary.md`",
            "2. `rule_inventory.md`",
            "3. `decision_delta_summary.json`",
            "4. `rule_impact_summary.json`",
            "5. `example_frames.md`",
            "",
        ]
    )


def _executive_summary(summary: dict[str, Any], impact: dict[str, Any]) -> str:
    metrics = summary.get("metrics") or {}
    families = Counter(metrics.get("likely_rule_family_counts") or {})
    return "\n".join(
        [
            "# Executive Summary",
            "",
            "This is a diagnostic reset audit, not a behavior patch.",
            "",
            f"- frames analyzed: {metrics.get('total_frames_analyzed')}",
            f"- delta events: {metrics.get('total_delta_events')}",
            f"- top rule families: {dict(families.most_common(5))}",
            f"- first-legal gaps: {metrics.get('first_legal_gap_counts')}",
            f"- recommendation: {metrics.get('recommendation')}",
            "",
            "Current conclusion: do not promote either experimental bundle. `attach_timing=new` bundles too many ideas; `evolve_priority=new` is too blunt around useful setup/draw/search.",
            "",
            f"- impact families: {list((impact.get('families') or {}).keys())}",
            "",
        ]
    )


def _relevant_paths(run_dir: Path, bundle_dir: Path | None) -> str:
    paths = [
        run_dir / "report.md",
        run_dir / "rule_inventory.md",
        run_dir / "decision_delta_summary.json",
        run_dir / "rule_impact_summary.json",
        run_dir / "decision_delta_events.jsonl",
        run_dir / "state_machine_snapshots.jsonl",
        bundle_dir or "",
        Path("deepmew/trace/rulebook_audit.py"),
        Path("scripts/audit_babymew_rulebook.py"),
        Path("tests/test_rulebook_audit.py"),
        Path("deepmew/policy/scoring.py"),
        Path("deepmew/policy_rules/attachment_timing.py"),
        Path("deepmew/policy_rules/evolve_priority.py"),
        Path("deepmew/state_machine/transition.py"),
        Path("HANDOFF_SUMMARY.md"),
        Path(".research/CURRENT_STATE_COMPACT.md"),
    ]
    return "\n".join(str(path) for path in paths if str(path)) + "\n"


def export_rulebook_bundle(
    *,
    summary: dict[str, Any],
    inventory: dict[str, Any],
    inventory_markdown: str,
    impact: dict[str, Any],
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    run_dir: Path,
    command: str,
    out_root: Path | str = ".research/handoff_bundles",
) -> Path:
    bundle_dir = _create_handoff_dir(out_root=out_root)
    write_markdown(bundle_dir / "README.md", _readme(command))
    write_markdown(bundle_dir / "executive_summary.md", _executive_summary(summary, impact))
    write_markdown(bundle_dir / "rule_inventory.md", inventory_markdown)
    write_json(bundle_dir / "rule_inventory.json", inventory)
    write_json(bundle_dir / "decision_delta_summary.json", summary)
    write_text(bundle_dir / "decision_delta_events.csv", _csv_text(events))
    write_jsonl(bundle_dir / "decision_delta_events.jsonl", events)
    write_json(bundle_dir / "rule_impact_summary.json", impact)
    write_jsonl(bundle_dir / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(bundle_dir / "example_frames.md", _example_frames_markdown(events))
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(run_dir, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=HANDOFF_KIND,
        command=command,
        inputs={"rulebook_run": str(run_dir)},
        outputs={name: str(bundle_dir / name) for name in REQUIRED_BUNDLE_FILES if name != "manifest.json"},
        summary=summary.get("metrics") or {},
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def write_rulebook_outputs(
    *,
    run_dir: Path,
    run_id: str,
    command: str,
    frames: list[dict[str, Any]],
    source_agents: Iterable[str] = DEFAULT_SOURCE_AGENTS,
    policies: Iterable[str] = DEFAULT_POLICIES,
    inputs: dict[str, Any] | None = None,
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> RulebookAuditResult:
    summary, events, impact, snapshots = analyze_rulebook_audit(
        run_id=run_id,
        frames=frames,
        source_agents=source_agents,
        policies=policies,
    )
    inventory = rule_inventory()
    inventory_markdown = build_rule_inventory_markdown(inventory)
    run_path = Path(run_dir)
    write_json(run_path / "rule_inventory.json", inventory)
    write_markdown(run_path / "rule_inventory.md", inventory_markdown)
    write_json(run_path / "decision_delta_summary.json", summary)
    write_jsonl(run_path / "decision_delta_events.jsonl", events)
    write_text(run_path / "decision_delta_events.csv", _csv_text(events))
    write_json(run_path / "rule_impact_summary.json", impact)
    write_jsonl(run_path / "state_machine_snapshots.jsonl", snapshots)
    write_markdown(run_path / "example_frames.md", _example_frames_markdown(events))
    write_markdown(run_path / "report.md", build_rulebook_report(summary, inventory, impact, events))
    outputs = {
        "report_markdown": str(run_path / "report.md"),
        "rule_inventory_json": str(run_path / "rule_inventory.json"),
        "rule_inventory_markdown": str(run_path / "rule_inventory.md"),
        "decision_delta_summary_json": str(run_path / "decision_delta_summary.json"),
        "decision_delta_events_jsonl": str(run_path / "decision_delta_events.jsonl"),
        "decision_delta_events_csv": str(run_path / "decision_delta_events.csv"),
        "rule_impact_summary_json": str(run_path / "rule_impact_summary.json"),
        "state_machine_snapshots_jsonl": str(run_path / "state_machine_snapshots.jsonl"),
        "example_frames_markdown": str(run_path / "example_frames.md"),
    }
    bundle_dir = None
    if export_handoff_bundle:
        bundle_dir = export_rulebook_bundle(
            summary=summary,
            inventory=inventory,
            inventory_markdown=inventory_markdown,
            impact=impact,
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
        inputs=inputs or {"source_agents": list(source_agents), "policies": list(policies)},
        outputs=outputs,
        summary=summary.get("metrics") or {},
    )
    write_manifest(run_path / "manifest.json", manifest)
    return RulebookAuditResult(
        run_dir=run_path,
        manifest_path=run_path / "manifest.json",
        summary=summary,
        rule_impact_summary=impact,
        events=events,
        bundle_dir=bundle_dir,
    )


def collect_rulebook_frames(
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
        prefix = f"rulebook-{index:02d}-{safe_slug(source_agent)}-vs-{safe_slug(opponent)}"
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
        payload["matchup"] = str(payload.get("game_id", "")).split("-000", 1)[0]
        frames.append(payload)
    return frames, pair_summaries, recorder.summary()
