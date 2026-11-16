"""Attachment timing audit for current-deck deterministic traces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from cg.api import AreaType, OptionType, all_attack, all_card_data

from deepmew.arena import PairSummary, run_pair
from deepmew.artifacts.run_dir import safe_slug
from deepmew.card_semantics import effective_attack_semantics, visible_basic_water_in_discard
from deepmew.deck_profiles import (
    CURRENT_WATER_MEGA_ABOMASNOW,
    CYRANO_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    VISIBLE_SETUP_PLAY_IDS,
    WAITRESS_CARD_ID,
)
from deepmew.options import resolve_option_identity
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board
from deepmew.trace.attach_target_quality import (
    ATTACH_TARGET_POLICIES,
    DEFAULT_MATCHUPS,
    DEFAULT_POLICIES,
    KYOGRE_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_LINE_IDS,
    SEMANTICS_CONTROL_MATCHUP,
    SNOVER_CARD_ID,
    TargetCandidate,
    analyze_target_quality,
    build_probe_outputs,
    build_target_candidates,
    load_jsonl,
    read_probe_run,
)
from deepmew.trace.diff import diff_probe_sets
from deepmew.trace.probe import probe_frames, read_frames
from deepmew.trace.record import TraceRecorder
from deepmew.trace.schema import PolicyDiff, PolicyProbe


TIMING_AUDIT_SCHEMA_VERSION = "deepmew.deck_attachment_timing.v1"
TIMING_EVENT_SCHEMA_VERSION = "deepmew.attachment_timing_event.v1"
RUN_KIND = "deck_attachment_timing"

TIMING_BUCKETS = [
    "good_skip_visible_ko",
    "good_skip_prize_pressure_attack",
    "good_skip_meaningful_attack",
    "good_skip_evolve_to_attacker",
    "good_skip_specific_setup_play_possible",
    "suspicious_skip_for_weak_attack",
    "suspicious_skip_for_named_nonsetup_play",
    "suspicious_skip_for_unresolved_play",
    "suspicious_skip_for_setup",
    "named_setup_play",
    "named_nonsetup_play",
    "unresolved_play",
    "missed_development_attach",
    "mega_line_attach_delayed_possible",
    "near_ready_attacker_attach_delayed",
    "future_attacker_underfed_possible",
    "active_already_ready_attach_still_skipped",
    "first_legal_attached_default_skipped",
    "attach_target_ablation_attached_default_skipped",
    "semantics_control_no_effect",
    "timing_likely_primary",
    "target_likely_secondary",
    "unknown",
]
META_BUCKETS = {
    "first_legal_attached_default_skipped",
    "attach_target_ablation_attached_default_skipped",
    "semantics_control_no_effect",
}
GOOD_SKIP_BUCKETS = {
    "good_skip_visible_ko",
    "good_skip_prize_pressure_attack",
    "good_skip_meaningful_attack",
    "good_skip_evolve_to_attacker",
    "good_skip_specific_setup_play_possible",
}
SUSPICIOUS_SKIP_BUCKETS = {
    "suspicious_skip_for_weak_attack",
    "suspicious_skip_for_named_nonsetup_play",
    "suspicious_skip_for_unresolved_play",
    "suspicious_skip_for_setup",
    "missed_development_attach",
    "mega_line_attach_delayed_possible",
    "near_ready_attacker_attach_delayed",
    "future_attacker_underfed_possible",
    "active_already_ready_attach_still_skipped",
}
PLAY_CARD_LABELS = {
    MEGA_SIGNAL_CARD_ID: "Mega Signal",
    WAITRESS_CARD_ID: "Waitress",
    LILLIES_DETERMINATION_CARD_ID: "Lillie's Determination",
    CYRANO_CARD_ID: "Cyrano",
    MAXIMUM_BELT_CARD_ID: "Maximum Belt",
}
REQUIRED_EXAMPLE_KEYS = [
    "v2.4_skipped_attach_for_play_while_first_legal_attached",
    "v2.4_skipped_attach_for_weak_attack_while_first_legal_attached",
    "v2.4_skipped_attach_for_visible_ko_or_meaningful_attack",
    "v2.4_skipped_attach_while_mega_line_underfed",
    "uncertain_frame",
]

_ATTACK_BY_ID: dict[int, Any] | None = None
_CARD_BY_ID: dict[int, Any] | None = None


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return dict(value)


def _payloads(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [_payload(value) for value in values]


def _probe_key(probe: dict[str, Any]) -> tuple[str, str]:
    return str(probe.get("frame_id") or ""), str(probe.get("policy_spec") or "")


def _chosen_options(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
    if probe is None:
        return []
    return list(probe.get("chosen_option_summaries") or [])


def _chosen_indices(probe: dict[str, Any] | None) -> list[int]:
    if probe is None:
        return []
    return [value for value in (probe.get("chosen_indices") or []) if isinstance(value, int)]


def _chosen_types(probe: dict[str, Any] | None) -> set[str]:
    return {str(option.get("option_type")) for option in _chosen_options(probe)}


def _first_chosen_type(probe: dict[str, Any] | None) -> str:
    options = _chosen_options(probe)
    if not options:
        return "none"
    return str(options[0].get("option_type") or "unknown")


def _chooses_attach(probe: dict[str, Any] | None) -> bool:
    return "ATTACH" in _chosen_types(probe)


def _same_choice(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return False
    return _chosen_indices(left) == _chosen_indices(right) and not left.get("error") and not right.get("error")


def _same_attachment_relevant_choice(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if _same_choice(left, right):
        return True
    return _chooses_attach(left) == _chooses_attach(right)


def _raw_observation(frame: dict[str, Any]) -> dict[str, Any] | None:
    raw = frame.get("raw_observation")
    return raw if isinstance(raw, dict) else None


def _raw_select(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame)
    if raw is None:
        return {}
    select = raw.get("select")
    return select if isinstance(select, dict) else {}


def _raw_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    options = _raw_select(frame).get("option") or []
    return [option for option in options if isinstance(option, dict)]


def _legal_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return list(frame.get("legal_options") or [])


def _legal_attach_indices(frame: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for option in _legal_options(frame):
        if str(option.get("option_type")) != "ATTACH":
            continue
        value = option.get("index")
        if isinstance(value, int):
            indices.append(value)
    if indices:
        return indices
    return [index for index, option in enumerate(_raw_options(frame)) if option.get("type") == OptionType.ATTACH.value]


def _attach_legal(frame: dict[str, Any]) -> bool:
    board = frame.get("board_summary") or {}
    return bool(board.get("attach_legal")) or bool(_legal_attach_indices(frame))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _enum_name(enum_cls: Any, value: Any) -> str | None:
    if value is None:
        return None
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _attack_by_id() -> dict[int, Any]:
    global _ATTACK_BY_ID
    if _ATTACK_BY_ID is None:
        _ATTACK_BY_ID = {int(attack.attackId): attack for attack in all_attack()}
    return _ATTACK_BY_ID


def _card_by_id() -> dict[int, Any]:
    global _CARD_BY_ID
    if _CARD_BY_ID is None:
        _CARD_BY_ID = {int(card.cardId): card for card in all_card_data()}
    return _CARD_BY_ID


def _card_name(card_id: int | None) -> str | None:
    if card_id is None:
        return None
    card = _card_by_id().get(card_id)
    if card is None:
        return None
    return str(getattr(card, "name", getattr(card, "cardName", "")) or "") or None


def _attack_name(attack_id: int | None) -> str | None:
    if attack_id is None:
        return None
    attack = _attack_by_id().get(attack_id)
    if attack is None:
        return None
    return str(getattr(attack, "name", getattr(attack, "attackName", "")) or "") or None


def _printed_damage(attack_id: int | None) -> int | None:
    if attack_id is None:
        return None
    attack = _attack_by_id().get(attack_id)
    if attack is None:
        return None
    return _int_or_none(getattr(attack, "damage", None))


def _own_player(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame) or {}
    current = raw.get("current") or {}
    players = current.get("players") or []
    your_index = _int_or_none(current.get("yourIndex"))
    if your_index is None:
        your_index = _int_or_none(frame.get("seat")) or 0
    if your_index < 0 or your_index >= len(players):
        return {}
    player = players[your_index]
    return player if isinstance(player, dict) else {}


def _opponent_player(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame) or {}
    current = raw.get("current") or {}
    players = current.get("players") or []
    your_index = _int_or_none(current.get("yourIndex")) or 0
    opponent_index = 1 - your_index
    if opponent_index < 0 or opponent_index >= len(players):
        return {}
    player = players[opponent_index]
    return player if isinstance(player, dict) else {}


def _pokemon_card_id(pokemon: dict[str, Any] | None) -> int | None:
    if not pokemon:
        return None
    return _int_or_none(pokemon.get("id", pokemon.get("card_id")))


def _energy_count(pokemon: dict[str, Any] | None) -> int:
    if not pokemon:
        return 0
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)
    energy_cards = pokemon.get("energyCards")
    if isinstance(energy_cards, list):
        return len(energy_cards)
    return int(pokemon.get("energy_count") or 0)


def _own_pokemon(frame: dict[str, Any]) -> list[dict[str, Any]]:
    player = _own_player(frame)
    pokemon: list[dict[str, Any]] = []
    active = player.get("active") or []
    if active and isinstance(active[0], dict):
        pokemon.append({"area": "ACTIVE", "index": 0, **active[0]})
    for index, item in enumerate(player.get("bench") or []):
        if isinstance(item, dict):
            pokemon.append({"area": "BENCH", "index": index, **item})
    if pokemon:
        return pokemon

    board = frame.get("board_summary") or {}
    own_active = board.get("own_active")
    if isinstance(own_active, dict) and own_active.get("card_id") is not None:
        pokemon.append(
            {
                "area": "ACTIVE",
                "index": 0,
                "id": own_active.get("card_id"),
                "energy_count": own_active.get("energy_count") or 0,
            }
        )
    for index, item in enumerate(board.get("own_bench") or []):
        if isinstance(item, dict):
            pokemon.append(
                {
                    "area": "BENCH",
                    "index": index,
                    "id": item.get("card_id"),
                    "energy_count": item.get("energy_count") or 0,
                }
            )
    return pokemon


def _pokemon_at(frame: dict[str, Any], area: str | None, index: int | None) -> dict[str, Any] | None:
    if area is None or index is None:
        return None
    for pokemon in _own_pokemon(frame):
        if str(pokemon.get("area")) == area and int(pokemon.get("index") or 0) == index:
            return pokemon
    return None


def _option_at(frame: dict[str, Any], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    options = _raw_options(frame)
    if 0 <= index < len(options):
        return options[index]
    return None


def _legal_summary_at(frame: dict[str, Any], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    for option in _legal_options(frame):
        if option.get("index") == index:
            return option
    return None


def _opponent_active_hp(frame: dict[str, Any]) -> int | None:
    board = frame.get("board_summary") or {}
    opponent_active = board.get("opponent_active")
    if isinstance(opponent_active, dict):
        hp = _int_or_none(opponent_active.get("hp"))
        if hp is not None:
            return hp
    active = _opponent_player(frame).get("active") or []
    if active and isinstance(active[0], dict):
        return _int_or_none(active[0].get("hp"))
    return None


def _card_id_from_choice(frame: dict[str, Any], index: int | None) -> int | None:
    option = _option_at(frame, index)
    if option:
        card_id = _int_or_none(option.get("cardId", option.get("cardID")))
        if card_id is not None:
            return card_id
    summary = _legal_summary_at(frame, index) or {}
    return _int_or_none(summary.get("card_id"))


def _summary_text(frame: dict[str, Any], index: int | None) -> str:
    parts: list[str] = []
    option = _legal_summary_at(frame, index)
    if isinstance(option, dict):
        parts.extend(str(option.get(key) or "") for key in ("summary", "card_id", "attack_id", "attack_name"))
    raw = _option_at(frame, index)
    if isinstance(raw, dict):
        parts.append(json.dumps(raw, sort_keys=True))
    return " ".join(parts)


def _safe_plan_summary(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame)
    if raw is None:
        return {"state": "unknown"}
    try:
        plan = analyze_board(raw)
    except Exception as exc:
        return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "state": getattr(plan.state, "value", str(plan.state)),
        "energy_attached": plan.energy_attached,
        "legal_attack_available": plan.legal_attack_available,
        "legal_attach_available": plan.legal_attach_available,
        "legal_play_available": plan.legal_play_available,
        "bench_count": plan.bench_count,
        "active_shortfall": plan.active_shortfall,
        "best_shortfall": plan.best_shortfall,
        "closest_area": plan.closest_area,
        "closest_index": plan.closest_index,
    }


def _candidate_json(candidate: TargetCandidate) -> dict[str, Any]:
    return candidate.to_jsonable()


def _target_summary(candidates: list[TargetCandidate]) -> dict[str, Any]:
    if not candidates:
        return {
            "candidate_count": 0,
            "mega_line_underfed_possible": False,
            "near_ready_candidate_count": 0,
            "future_attacker_underfed_possible": False,
            "active_already_ready_with_other_underfed": False,
            "best_candidates": [],
            "candidates": [],
        }
    best_score = max(candidate.audit_score for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if candidate.audit_score == best_score]
    mega_line_underfed = any(
        candidate.card_id in MEGA_LINE_IDS
        and candidate.known_target
        and not candidate.already_ready
        and (candidate.energy_count or 0) < 2
        for candidate in candidates
    )
    near_ready = [candidate for candidate in candidates if candidate.near_ready]
    future_underfed = any(candidate.known_target and not candidate.already_ready for candidate in candidates)
    active_ready = any(candidate.area == "ACTIVE" and candidate.already_ready for candidate in candidates)
    active_ready_other_underfed = active_ready and any(
        candidate.known_target and candidate.area != "ACTIVE" and not candidate.already_ready for candidate in candidates
    )
    return {
        "candidate_count": len(candidates),
        "mega_line_underfed_possible": mega_line_underfed,
        "near_ready_candidate_count": len(near_ready),
        "future_attacker_underfed_possible": future_underfed,
        "active_already_ready_with_other_underfed": active_ready_other_underfed,
        "best_candidates": [_candidate_json(candidate) for candidate in best_candidates],
        "candidates": [_candidate_json(candidate) for candidate in candidates],
    }


def summarize_board_timing(frame: dict[str, Any], candidates: list[TargetCandidate] | None = None) -> dict[str, Any]:
    candidates = candidates if candidates is not None else build_target_candidates(frame)
    raw = _raw_observation(frame) or {}
    board = frame.get("board_summary") or {}
    return {
        "select_context": frame.get("select_context"),
        "turn": frame.get("turn"),
        "seat": frame.get("seat"),
        "plan": _safe_plan_summary(frame),
        "own_active": board.get("own_active"),
        "own_bench": board.get("own_bench"),
        "opponent_active": board.get("opponent_active"),
        "own_discard_basic_water_count": board.get(
            "own_discard_basic_water_count",
            visible_basic_water_in_discard(raw) if raw else 0,
        ),
        "attach_targets": _target_summary(candidates),
    }


def attack_info_for_option(frame: dict[str, Any], option_index: int | None) -> dict[str, Any]:
    option = _option_at(frame, option_index)
    summary = _legal_summary_at(frame, option_index) or {}
    attack_id = _int_or_none((option or {}).get("attackId"))
    if attack_id is None:
        attack_id = _int_or_none(summary.get("attack_id"))
    printed_damage = _printed_damage(attack_id)
    effective_damage = _int_or_none(summary.get("effective_damage"))
    semantics_reason = None
    raw = _raw_observation(frame)
    if option is not None and raw is not None:
        try:
            semantics = effective_attack_semantics(option, raw)
        except Exception:
            semantics = None
        if semantics is not None:
            effective_damage = semantics.effective_damage
            semantics_reason = semantics.reason
    if effective_damage is None:
        effective_damage = printed_damage
    opponent_hp = _opponent_active_hp(frame)
    pressure = "unknown"
    if effective_damage is not None:
        if opponent_hp is not None and opponent_hp > 0:
            if effective_damage >= opponent_hp:
                pressure = "ko"
            elif effective_damage * 2 >= opponent_hp:
                pressure = "prize_pressure"
            elif effective_damage <= 30:
                pressure = "weak"
            else:
                pressure = "unknown"
        elif effective_damage >= 100:
            pressure = "meaningful"
        elif effective_damage <= 30:
            pressure = "weak"
    return {
        "option_index": option_index,
        "attack_id": attack_id,
        "attack_name": _attack_name(attack_id),
        "printed_damage": printed_damage,
        "effective_damage": effective_damage,
        "semantics_reason": semantics_reason,
        "opponent_active_hp": opponent_hp,
        "pressure": pressure,
    }


def play_info_for_option(frame: dict[str, Any], option_index: int | None) -> dict[str, Any]:
    option = _option_at(frame, option_index) or {}
    raw = _raw_observation(frame) or {}
    identity = resolve_option_identity(raw, option)
    card_id = identity.card_id
    text = _summary_text(frame, option_index)
    card_name = identity.card_name or _card_name(card_id)
    setup_card = CURRENT_WATER_MEGA_ABOMASNOW.is_setup_card(card_id)
    if not setup_card and card_id is None:
        setup_card = "Mega Signal" in text or "Waitress" in text
    if identity.resolved and setup_card:
        identity_category = "named_setup_play"
    elif identity.resolved:
        identity_category = "named_nonsetup_play"
    else:
        identity_category = "unresolved_play"
    return {
        "option_index": option_index,
        "resolved": identity.resolved,
        "resolve_reason": identity.reason,
        "source_area": identity.source_area,
        "source_index": identity.source_index,
        "card_id": card_id,
        "card_name": card_name,
        "identity_category": identity_category,
        "specific_setup_play_possible": bool(setup_card),
        "summary_text": text[:300],
    }


def evolve_info_for_option(frame: dict[str, Any], option_index: int | None) -> dict[str, Any]:
    option = _option_at(frame, option_index) or {}
    card_id = _card_id_from_choice(frame, option_index)
    area = _enum_name(AreaType, option.get("inPlayArea"))
    target_index = _int_or_none(option.get("inPlayIndex"))
    if target_index is None:
        target_index = 0
    target = _pokemon_at(frame, area, target_index)
    target_card_id = _pokemon_card_id(target)
    text = _summary_text(frame, option_index)
    attacker_evolve_possible = (
        card_id == MEGA_ABOMASNOW_CARD_ID
        or target_card_id in MEGA_LINE_IDS
        or "Mega Abomasnow" in text
        or "Snover" in text
    )
    return {
        "option_index": option_index,
        "card_id": card_id,
        "card_name": _card_name(card_id),
        "target_area": area,
        "target_index": target_index,
        "target_card_id": target_card_id,
        "target_card_name": _card_name(target_card_id),
        "evolves_visible_attacker_line": bool(attacker_evolve_possible),
        "summary_text": text[:300],
    }


def _choice_summary(probe: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "policy_spec": None if probe is None else probe.get("policy_spec"),
        "chosen_indices": _chosen_indices(probe),
        "chosen_type": _first_chosen_type(probe),
        "chooses_attach": _chooses_attach(probe),
        "error": None if probe is None else probe.get("error"),
    }


def classify_timing_event(
    frame: dict[str, Any],
    frame_probes: dict[str, dict[str, Any]],
    *,
    default_policy: str = "v2.4",
) -> dict[str, Any] | None:
    """Classify one ATTACH-legal frame where default V2.4 skipped ATTACH."""

    if not _attach_legal(frame):
        return None
    default_probe = frame_probes.get(default_policy)
    if default_probe is None or default_probe.get("error") or _chooses_attach(default_probe):
        return None

    first_probe = frame_probes.get("first-legal")
    semantics_probe = frame_probes.get("v2.4[semantics=off]")
    ablation_choices = {
        policy: _choice_summary(frame_probes.get(policy))
        for policy in ATTACH_TARGET_POLICIES
        if frame_probes.get(policy) is not None
    }
    attach_target_ablation_attached = any(value["chooses_attach"] for value in ablation_choices.values())
    chosen_indices = _chosen_indices(default_probe)
    chosen_index = chosen_indices[0] if chosen_indices else None
    chosen_type = _first_chosen_type(default_probe)
    candidates = build_target_candidates(frame)
    target_summary = _target_summary(candidates)
    buckets: set[str] = set()
    attack_info: dict[str, Any] | None = None
    play_info: dict[str, Any] | None = None
    evolve_info: dict[str, Any] | None = None

    if _chooses_attach(first_probe):
        buckets.add("first_legal_attached_default_skipped")
    if attach_target_ablation_attached:
        buckets.add("attach_target_ablation_attached_default_skipped")
    if semantics_probe is not None and _same_attachment_relevant_choice(default_probe, semantics_probe):
        buckets.add("semantics_control_no_effect")

    if target_summary["mega_line_underfed_possible"]:
        buckets.add("mega_line_attach_delayed_possible")
    if target_summary["near_ready_candidate_count"]:
        buckets.add("near_ready_attacker_attach_delayed")
    if target_summary["future_attacker_underfed_possible"]:
        buckets.add("future_attacker_underfed_possible")
        buckets.add("missed_development_attach")
    if target_summary["active_already_ready_with_other_underfed"]:
        buckets.add("active_already_ready_attach_still_skipped")

    if chosen_type == "ATTACK":
        attack_info = attack_info_for_option(frame, chosen_index)
        pressure = attack_info["pressure"]
        if pressure == "ko":
            buckets.add("good_skip_visible_ko")
        elif pressure == "prize_pressure":
            buckets.add("good_skip_prize_pressure_attack")
        elif pressure == "meaningful":
            buckets.add("good_skip_meaningful_attack")
        elif pressure == "weak":
            buckets.add("suspicious_skip_for_weak_attack")
    elif chosen_type == "PLAY":
        play_info = play_info_for_option(frame, chosen_index)
        category = str(play_info.get("identity_category") or "unresolved_play")
        buckets.add(category)
        if category == "named_setup_play" and play_info["specific_setup_play_possible"]:
            buckets.add("good_skip_specific_setup_play_possible")
        elif category == "named_nonsetup_play":
            buckets.add("suspicious_skip_for_named_nonsetup_play")
        else:
            buckets.add("suspicious_skip_for_unresolved_play")
        if target_summary["future_attacker_underfed_possible"]:
            buckets.add("suspicious_skip_for_setup")
    elif chosen_type == "EVOLVE":
        evolve_info = evolve_info_for_option(frame, chosen_index)
        if evolve_info["evolves_visible_attacker_line"]:
            buckets.add("good_skip_evolve_to_attacker")
        else:
            buckets.add("suspicious_skip_for_setup")
    elif chosen_type in {"END", "none"} and target_summary["future_attacker_underfed_possible"]:
        buckets.add("suspicious_skip_for_setup")

    buckets.discard("unknown")
    if not buckets or not (buckets - META_BUCKETS):
        buckets.add("unknown")

    return {
        "schema_version": TIMING_EVENT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "seat": frame.get("seat"),
        "decision_index": frame.get("decision_index"),
        "select_context": frame.get("select_context"),
        "attach_legal": True,
        "default_choice": _choice_summary(default_probe),
        "first_legal_choice": _choice_summary(first_probe),
        "semantics_off_choice": _choice_summary(semantics_probe),
        "attach_target_ablation_choices": ablation_choices,
        "first_legal_attached_default_skipped": _chooses_attach(first_probe),
        "attach_target_ablation_attached_default_skipped": attach_target_ablation_attached,
        "chosen_type": chosen_type,
        "chosen_index": chosen_index,
        "attack_info": attack_info,
        "play_info": play_info,
        "evolve_info": evolve_info,
        "board_timing": summarize_board_timing(frame, candidates),
        "buckets": sorted(buckets),
        "is_good_skip": bool(buckets & GOOD_SKIP_BUCKETS),
        "is_suspicious_skip": bool(buckets & SUSPICIOUS_SKIP_BUCKETS),
    }


def run_trace_matchups(
    *,
    run_id: str,
    matchups: list[tuple[str, str]],
    games: int,
    seed: int,
    max_steps: int,
    limit_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[PairSummary], dict[str, Any]]:
    recorder = TraceRecorder(run_id=run_id, sample_limit=limit_frames)
    pair_summaries: list[PairSummary] = []
    for index, (agent_a, agent_b) in enumerate(matchups):
        pair_seed = seed + index * 10007
        prefix = f"decktime-{index:02d}-{safe_slug(agent_a)}-vs-{safe_slug(agent_b)}"
        pair_summaries.append(
            run_pair(
                agent_a,
                agent_b,
                games=games,
                max_steps=max_steps,
                seed=pair_seed,
                trace_recorder=recorder,
                game_id_prefix=prefix,
            )
        )
    return [frame.to_jsonable() for frame in recorder.frames], pair_summaries, recorder.summary()


def _summarize_pair(pair: PairSummary) -> dict[str, Any]:
    return {
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
    }


def _probe_groups(probes: Iterable[Any]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for probe in _payloads(probes):
        frame_id, policy = _probe_key(probe)
        groups[frame_id][policy] = probe
    return groups


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _event_has(event: dict[str, Any], bucket: str) -> bool:
    return bucket in set(event.get("buckets") or [])


def _add_example(
    examples: dict[str, list[dict[str, Any]]],
    key: str,
    event: dict[str, Any],
    *,
    limit: int = 5,
) -> None:
    if len(examples[key]) >= limit:
        return
    attack = event.get("attack_info") or {}
    play = event.get("play_info") or {}
    examples[key].append(
        {
            "frame_id": event.get("frame_id"),
            "game_id": event.get("game_id"),
            "turn": event.get("turn"),
            "seat": event.get("seat"),
            "chosen_type": event.get("chosen_type"),
            "chosen_index": event.get("chosen_index"),
            "attack": {
                "name": attack.get("attack_name"),
                "effective_damage": attack.get("effective_damage"),
                "opponent_hp": attack.get("opponent_active_hp"),
                "pressure": attack.get("pressure"),
            },
            "play": {
                "card_name": play.get("card_name"),
                "specific_setup_play_possible": play.get("specific_setup_play_possible"),
            },
            "best_attach_targets": event.get("board_timing", {})
            .get("attach_targets", {})
            .get("best_candidates", [])[:2],
            "buckets": event.get("buckets"),
        }
    )


def load_target_quality_reference(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    target_path = Path(path)
    if target_path.is_dir():
        target_path = target_path / "target_quality.json"
    if not target_path.exists():
        return None
    return json.loads(target_path.read_text(encoding="utf-8"))


def _target_reference_summary(reference: dict[str, Any] | None) -> dict[str, Any]:
    if not reference:
        return {"source": None, "default_better_visible_target_available": None, "target_events": None}
    metrics = reference.get("metrics") or {}
    better = metrics.get("better_visible_target_available_count_by_policy") or {}
    return {
        "source": reference.get("run_id"),
        "default_better_visible_target_available": better.get("v2.4", 0),
        "target_events": metrics.get("target_events"),
        "missed_attach_frames_by_default_v24": metrics.get("missed_attach_frames_by_default_v24"),
        "conclusion": metrics.get("timing_vs_target_vs_option_order_conclusion"),
    }


def _conclusion(
    *,
    skipped: int,
    suspicious: int,
    good: int,
    weak_attack: int,
    suspicious_play: int,
    target_better: int | None,
) -> str:
    if skipped == 0:
        return "unknown, no default V2.4 skipped ATTACH frames found"
    if suspicious >= max(8, good) and skipped >= max(10, (target_better or 0) * 2):
        return "attach timing likely primary and target quality likely secondary, possible and not proven causal"
    if (target_better or 0) >= max(8, skipped // 2):
        return "attach target quality remains competitive with timing, possible and not proven causal"
    if weak_attack + suspicious_play >= max(8, good):
        return "weak attack or non-setup/unresolved PLAY timing looks suspicious, possible and not proven causal"
    if good > suspicious:
        return "many skipped ATTACH frames have visible-state good reasons, timing not proven primary"
    return "mixed attachment timing evidence, possible and not proven causal"


def _play_card_bucket(card_id: int | None, card_name: str | None, resolved: bool) -> str:
    if not resolved or card_id is None:
        return "unresolved PLAY"
    if card_id in PLAY_CARD_LABELS:
        return PLAY_CARD_LABELS[card_id]
    return "other named PLAY"


def _play_identity_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    named_card_counter: Counter[tuple[int | None, str | None]] = Counter()
    bucket_counter: Counter[str] = Counter()
    for event in events:
        if event.get("chosen_type") != "PLAY":
            continue
        play = event.get("play_info") or {}
        category = str(play.get("identity_category") or "unresolved_play")
        category_counts[category] += 1
        card_id = _int_or_none(play.get("card_id"))
        card_name = play.get("card_name")
        resolved = bool(play.get("resolved"))
        if resolved and card_id is not None:
            named_card_counter[(card_id, str(card_name or _card_name(card_id) or card_id))] += 1
        bucket_counter[_play_card_bucket(card_id, card_name, resolved)] += 1
    per_card = [
        {"card_id": card_id, "card_name": card_name, "count": count}
        for (card_id, card_name), count in named_card_counter.most_common()
    ]
    return {
        "named_setup_play": category_counts.get("named_setup_play", 0),
        "named_nonsetup_play": category_counts.get("named_nonsetup_play", 0),
        "unresolved_play": category_counts.get("unresolved_play", 0),
        "per_card_counts": per_card,
        "bucket_counts": dict(bucket_counter),
    }


def analyze_attachment_timing(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    probes: Iterable[Any],
    diffs: Iterable[Any],
    policies: list[str],
    pair_summaries: Iterable[PairSummary] | None = None,
    trace_source: str | None = None,
    probe_source: str | None = None,
    target_quality_reference: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probe_payloads = _payloads(probes)
    diff_payloads = _payloads(diffs)
    pair_payloads = [_summarize_pair(pair) for pair in pair_summaries or []]
    probes_by_frame = _probe_groups(probe_payloads)

    events: list[dict[str, Any]] = []
    bucket_counter: Counter[str] = Counter()
    skipped_by_choice_type: Counter[str] = Counter()
    examples_required: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attach_legal_frames = 0

    for frame in frames:
        if not _attach_legal(frame):
            continue
        attach_legal_frames += 1
        frame_id = str(frame.get("frame_id") or "")
        event = classify_timing_event(frame, probes_by_frame.get(frame_id, {}))
        if event is None:
            continue
        events.append(event)
        bucket_counter.update(event["buckets"])
        skipped_by_choice_type[event["chosen_type"]] += 1
        for bucket in event["buckets"]:
            _add_example(examples_by_bucket, bucket, event)

        if event["chosen_type"] == "PLAY" and event["first_legal_attached_default_skipped"]:
            _add_example(examples_required, "v2.4_skipped_attach_for_play_while_first_legal_attached", event)
        if _event_has(event, "suspicious_skip_for_weak_attack") and event["first_legal_attached_default_skipped"]:
            _add_example(examples_required, "v2.4_skipped_attach_for_weak_attack_while_first_legal_attached", event)
        if any(
            _event_has(event, bucket)
            for bucket in ("good_skip_visible_ko", "good_skip_prize_pressure_attack", "good_skip_meaningful_attack")
        ):
            _add_example(examples_required, "v2.4_skipped_attach_for_visible_ko_or_meaningful_attack", event)
        if _event_has(event, "mega_line_attach_delayed_possible"):
            _add_example(examples_required, "v2.4_skipped_attach_while_mega_line_underfed", event)
        if _event_has(event, "unknown"):
            _add_example(examples_required, "uncertain_frame", event)

    for key in REQUIRED_EXAMPLE_KEYS:
        examples_required.setdefault(key, [])

    if target_quality_reference is None:
        try:
            target_quality_reference, _ = analyze_target_quality(
                run_id=f"{run_id}-target-reference",
                frames=frames,
                probes=probe_payloads,
                diffs=diff_payloads,
                policies=policies,
                pair_summaries=None,
                trace_source=trace_source,
                probe_source=probe_source,
            )
        except Exception:
            target_quality_reference = None

    target_reference = _target_reference_summary(target_quality_reference)
    target_better = target_reference["default_better_visible_target_available"]
    suspicious = sum(1 for event in events if event["is_suspicious_skip"])
    good = sum(1 for event in events if event["is_good_skip"])
    weak_attack = bucket_counter.get("suspicious_skip_for_weak_attack", 0)
    suspicious_play = bucket_counter.get("suspicious_skip_for_named_nonsetup_play", 0) + bucket_counter.get(
        "suspicious_skip_for_unresolved_play", 0
    )
    play_identity = _play_identity_summary(events)
    conclusion = _conclusion(
        skipped=len(events),
        suspicious=suspicious,
        good=good,
        weak_attack=weak_attack,
        suspicious_play=suspicious_play,
        target_better=target_better,
    )
    if "timing likely primary" in conclusion:
        bucket_counter["timing_likely_primary"] += 1
        bucket_counter["target_likely_secondary"] += 1

    skipped_attack_pressure = {
        "visible_ko": bucket_counter.get("good_skip_visible_ko", 0),
        "prize_pressure_or_meaningful": bucket_counter.get("good_skip_prize_pressure_attack", 0)
        + bucket_counter.get("good_skip_meaningful_attack", 0),
        "weak_attack": weak_attack,
    }
    metrics = {
        "total_frames_analyzed": len(frames),
        "attach_legal_frames": attach_legal_frames,
        "default_v24_skipped_attach_frames": len(events),
        "first_legal_attached_while_default_skipped": bucket_counter.get("first_legal_attached_default_skipped", 0),
        "attach_ablations_attached_while_default_skipped": bucket_counter.get(
            "attach_target_ablation_attached_default_skipped", 0
        ),
        "default_skipped_for_play": skipped_by_choice_type.get("PLAY", 0),
        "default_skipped_for_attack": skipped_by_choice_type.get("ATTACK", 0),
        "default_skipped_for_evolve": skipped_by_choice_type.get("EVOLVE", 0),
        "default_skipped_for_end_or_other": sum(
            count for option_type, count in skipped_by_choice_type.items() if option_type not in {"PLAY", "ATTACK", "EVOLVE"}
        ),
        "skipped_attach_for_visible_ko": skipped_attack_pressure["visible_ko"],
        "skipped_attach_for_prize_pressure_or_meaningful_attack": skipped_attack_pressure[
            "prize_pressure_or_meaningful"
        ],
        "skipped_attach_for_weak_attack": skipped_attack_pressure["weak_attack"],
        "skipped_attach_for_generic_play": bucket_counter.get("suspicious_skip_for_generic_play", 0),
        "skipped_attach_for_named_setup_play": play_identity["named_setup_play"],
        "skipped_attach_for_named_nonsetup_play": play_identity["named_nonsetup_play"],
        "skipped_attach_for_unresolved_play": play_identity["unresolved_play"],
        "play_skip_counts_by_card": play_identity["per_card_counts"],
        "play_skip_named_bucket_counts": play_identity["bucket_counts"],
        "mega_line_attach_delayed_possible": bucket_counter.get("mega_line_attach_delayed_possible", 0),
        "near_ready_attacker_attach_delayed": bucket_counter.get("near_ready_attacker_attach_delayed", 0),
        "suspicious_skip_count": suspicious,
        "good_skip_count": good,
        "unknown_count": bucket_counter.get("unknown", 0),
        "good_skip_rate": _rate(good, len(events)),
        "suspicious_skip_rate": _rate(suspicious, len(events)),
        "skipped_by_choice_type": dict(skipped_by_choice_type),
        "target_quality_reference": target_reference,
        "conclusion": conclusion,
    }

    audit = {
        "schema_version": TIMING_AUDIT_SCHEMA_VERSION,
        "run_id": run_id,
        "trace_source": trace_source,
        "probe_source": probe_source,
        "policies": policies,
        "frames": len(frames),
        "games": len({frame.get("game_id") for frame in frames}),
        "probes": len(probe_payloads),
        "diffs": len(diff_payloads),
        "probe_errors": sum(1 for probe in probe_payloads if probe.get("error")),
        "pair_summaries": pair_payloads,
        "metrics": metrics,
        "bucket_counts": dict(bucket_counter),
        "required_examples": dict(examples_required),
        "examples_by_bucket": dict(examples_by_bucket),
        "classification_buckets": TIMING_BUCKETS,
        "do_not_implement_yet": [
            "Do not change gameplay policy from this audit-only classifier.",
            "Do not tune scores from this visible-state evidence alone.",
            "Do not treat possible tags as proven causal.",
            "Do not change Hammer-lanche or Riptide semantics here.",
            "Do not add trainer semantics here.",
            "Do not change deck.csv or deck loading here.",
        ],
    }
    return audit, events


def build_timing_from_existing(
    *,
    run_id: str,
    trace_run: str | Path,
    probe_run: str | Path | None,
    policies: list[str],
    limit_frames: int | None = None,
    target_quality_run: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    frames = read_frames(trace_run, limit=limit_frames)
    if probe_run:
        probes, diffs = read_probe_run(probe_run)
    else:
        probe_objects, diff_objects = build_probe_outputs(frames, policies)
        probes = [probe.to_jsonable() for probe in probe_objects]
        diffs = [diff.to_jsonable() for diff in diff_objects]
    target_reference = load_target_quality_reference(target_quality_run)
    audit, events = analyze_attachment_timing(
        run_id=run_id,
        frames=frames,
        probes=probes,
        diffs=diffs,
        policies=policies,
        trace_source=str(trace_run),
        probe_source=str(probe_run),
        target_quality_reference=target_reference,
    )
    return frames, probes, diffs, audit, events


def _pair_rows(audit: dict[str, Any]) -> list[list[Any]]:
    rows = [
        [
            pair["agent_a"],
            pair["agent_b"],
            f"{pair['a_wins']}-{pair['b_wins']}-{pair['draws']}",
            f"{pair['completed']}/{pair['games']}",
            pair["crashes"],
            pair["timeouts"],
            f"{pair['avg_turns']:.2f}",
        ]
        for pair in audit.get("pair_summaries", [])
    ]
    return rows or [["unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"]]


def _counter_rows(counter_dict: dict[str, Any], *, limit: int = 16) -> list[list[Any]]:
    counter = Counter({str(key): int(value) for key, value in counter_dict.items()})
    rows = [[key, value] for key, value in counter.most_common(limit)]
    return rows or [["none", 0]]


def _examples_table(examples: dict[str, list[dict[str, Any]]]) -> str:
    rows: list[list[Any]] = []
    ordered_keys = [key for key in REQUIRED_EXAMPLE_KEYS if key in examples]
    ordered_keys.extend(key for key in examples if key not in ordered_keys)
    for key in ordered_keys:
        values = examples.get(key) or []
        if not values:
            rows.append([key, "none observed", "none", "none", "none", "none", "none", "none"])
            continue
        for example in values[:3]:
            attack = example.get("attack") or {}
            best_targets = example.get("best_attach_targets") or []
            target_role = "none"
            target_shortfall = "none"
            if best_targets:
                target_role = str(best_targets[0].get("primary_role") or "unknown")
                target_shortfall = str(best_targets[0].get("before_shortfall"))
            rows.append(
                [
                    key,
                    example.get("frame_id"),
                    example.get("chosen_type"),
                    attack.get("pressure", "none"),
                    attack.get("effective_damage", "none"),
                    target_role,
                    target_shortfall,
                    ",".join(example.get("buckets") or []),
                ]
            )
    return table(
        ["example", "frame_id", "choice", "attack_pressure", "eff_damage", "best_attach_role", "shortfall", "buckets"],
        rows or [["none", "none", "none", "none", "none", "none", "none", "none"]],
    )


def build_attachment_timing_report(audit: dict[str, Any]) -> str:
    metrics = audit["metrics"]
    target_ref = metrics["target_quality_reference"]
    lines = [
        title("Deck Track V3.4 Attachment Timing Audit"),
        f"- run_id: `{audit['run_id']}`\n",
        f"- frames: {audit['frames']}\n",
        f"- games_with_frames: {audit['games']}\n",
        f"- probes: {audit['probes']}\n",
        f"- probe_errors: {audit['probe_errors']}\n",
        f"- trace_source: `{audit.get('trace_source')}`\n",
        f"- probe_source: `{audit.get('probe_source')}`\n",
        "\n",
        section("Evidence Boundary"),
        "This report uses an audit-only classifier, visible-state evidence, and conservative possible tags.\n\n",
        "A suspicious label is not proven causal. Hidden hand contents, hidden deck order, future draws, trainer text, and full-game counterfactuals are outside this audit.\n\n",
        "The audit separates attachment timing from attach target quality. Target-quality reference numbers are included only to keep those hypotheses distinct.\n\n",
        section("Matchups"),
        table(["agent_a", "agent_b", "record", "completed", "crashes", "timeouts", "avg_turns"], _pair_rows(audit)),
        "\n",
        section("Core Metrics"),
        table(
            ["metric", "value"],
            [
                ["total_frames_analyzed", metrics["total_frames_analyzed"]],
                ["attach_legal_frames", metrics["attach_legal_frames"]],
                ["default_v24_skipped_attach_frames", metrics["default_v24_skipped_attach_frames"]],
                ["first_legal_attached_while_default_skipped", metrics["first_legal_attached_while_default_skipped"]],
                ["attach_ablations_attached_while_default_skipped", metrics["attach_ablations_attached_while_default_skipped"]],
                ["default_skipped_for_play", metrics["default_skipped_for_play"]],
                ["default_skipped_for_attack", metrics["default_skipped_for_attack"]],
                ["default_skipped_for_evolve", metrics["default_skipped_for_evolve"]],
                ["default_skipped_for_end_or_other", metrics["default_skipped_for_end_or_other"]],
                ["skipped_attach_for_visible_ko", metrics["skipped_attach_for_visible_ko"]],
                [
                    "skipped_attach_for_prize_pressure_or_meaningful_attack",
                    metrics["skipped_attach_for_prize_pressure_or_meaningful_attack"],
                ],
                ["skipped_attach_for_weak_attack", metrics["skipped_attach_for_weak_attack"]],
                ["skipped_attach_for_named_setup_play", metrics["skipped_attach_for_named_setup_play"]],
                ["skipped_attach_for_named_nonsetup_play", metrics["skipped_attach_for_named_nonsetup_play"]],
                ["skipped_attach_for_unresolved_play", metrics["skipped_attach_for_unresolved_play"]],
                ["mega_line_attach_delayed_possible", metrics["mega_line_attach_delayed_possible"]],
                ["near_ready_attacker_attach_delayed", metrics["near_ready_attacker_attach_delayed"]],
                ["suspicious_skip_count", metrics["suspicious_skip_count"]],
                ["good_skip_count", metrics["good_skip_count"]],
                ["unknown_count", metrics["unknown_count"]],
                ["conclusion", metrics["conclusion"]],
            ],
        ),
        "\n",
        section("Timing vs Target Reference"),
        table(
            ["source", "default_better_visible_target", "target_events", "target_missed_attach", "target_conclusion"],
            [
                [
                    target_ref.get("source"),
                    target_ref.get("default_better_visible_target_available"),
                    target_ref.get("target_events"),
                    target_ref.get("missed_attach_frames_by_default_v24"),
                    target_ref.get("conclusion"),
                ]
            ],
        ),
        "\n",
        section("Skipped Choice Mix"),
        table(["choice_type", "count"], _counter_rows(metrics["skipped_by_choice_type"])),
        "\n",
        section("PLAY Identity Counts"),
        table(["category", "count"], _counter_rows(metrics.get("play_skip_named_bucket_counts", {}))),
        "\n",
        "Per-card named PLAY skips:\n\n",
        table(
            ["card_id", "card_name", "count"],
            [
                [row.get("card_id"), row.get("card_name"), row.get("count")]
                for row in metrics.get("play_skip_counts_by_card", [])
            ]
            or [["none", "none", 0]],
        ),
        "\n",
        section("Bucket Counts"),
        table(["bucket", "count"], _counter_rows(audit["bucket_counts"])),
        "\n",
        section("Required Example Frames"),
        _examples_table(audit.get("required_examples", {})),
        "\n",
        section("Good Skips"),
        bullets(
            [
                "good_skip_visible_ko means the skipped attachment protected a visible KO attack.",
                "good_skip_prize_pressure_attack and good_skip_meaningful_attack mean visible damage looked high enough to justify attacking.",
                "good_skip_evolve_to_attacker and good_skip_specific_setup_play_possible mean the skipped attachment may have allowed concrete setup first.",
            ]
        ),
        "\n",
        section("Suspicious Skips"),
        bullets(
            [
                "suspicious_skip_for_weak_attack means ATTACH was skipped for an attack with weak visible pressure.",
                "suspicious_skip_for_named_nonsetup_play means ATTACH was skipped for a resolved PLAY card that is not a current setup card.",
                "suspicious_skip_for_unresolved_play means ATTACH was skipped for PLAY whose card identity could not be resolved from visible hand.",
                "mega_line_attach_delayed_possible, near_ready_attacker_attach_delayed, and future_attacker_underfed_possible are possible timing issues, not proven causal.",
            ]
        ),
        "\n",
        section("Answers"),
        bullets(
            [
                f"Is attachment timing a likely primary issue? {metrics['conclusion']}.",
                f"Does the audit distinguish timing from target quality? yes, target reference default better target count is {target_ref.get('default_better_visible_target_available')}.",
                f"How often did first-legal attach while default skipped? {metrics['first_legal_attached_while_default_skipped']} frames.",
                f"How often did attach-target ablations attach while default skipped? {metrics['attach_ablations_attached_while_default_skipped']} frames.",
                "Would an exact policy patch be justified? Only if future work accepts this audit-only classifier as sufficient evidence for a narrow timing rule.",
            ]
        ),
        "\n",
    ]
    return "".join(lines)
