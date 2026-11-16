from __future__ import annotations

from typing import Any, Callable

from cg.api import OptionType, SelectContext, SelectType
from deepmew.card_semantics import effective_attack_semantics
from deepmew.policy.attacks import attack_base_score, attack_damage, attack_pressure, has_main_setup_option
from deepmew.policy.enums import RISKY_ACTIONS, enum_name, option_type
from deepmew.policy.models import Candidate
from deepmew.policy.v2_scoring import score_option_v2
from deepmew.policy_rules.attachment_timing import (
    apply_attach_timing_adjustment,
    classify_attach_timing,
    extract_attach_timing_facts,
)
from deepmew.policy_rules.evolve_priority import (
    apply_evolve_priority_adjustment,
    classify_evolve_priority,
    extract_evolve_priority_facts,
)
from deepmew.state_machine import TurnState, analyze_board, attachment_effect


def _score_option(
    option: dict[str, Any],
    select_type: str,
    context: str,
    obs: dict[str, Any],
    use_card_semantics: bool = True,
) -> tuple[int, list[str]]:
    score, reasons = score_option_v2(option, select_type, context)
    current_option_type = option_type(option)
    plan = analyze_board(obs)

    if current_option_type == "ATTACK" and use_card_semantics:
        semantics = effective_attack_semantics(option, obs)
        if semantics is not None:
            printed_damage = attack_damage(option.get("attackId"))
            printed_score, _ = attack_base_score(printed_damage)
            effective_score, _ = attack_base_score(semantics.effective_damage)
            score += effective_score - printed_score
            reasons.append(f"semantic-effective-damage={semantics.effective_damage}")
            reasons.append(semantics.reason)

    if plan.state == TurnState.FALLBACK:
        return score, reasons

    if current_option_type == "END" and plan.legal_attach_available and not plan.energy_attached:
        score -= 60
        reasons.append("do-not-waste-attachment-window")

    if current_option_type == "ATTACH":
        effect = attachment_effect(plan, option)
        if plan.state in {TurnState.NO_ATTACKER_READY, TurnState.ATTACKER_NEAR_READY}:
            if effect.improves_readiness:
                score += 45
                reasons.append("attachment-reduces-attack-shortfall")
            if effect.enables_attack:
                score += 40
                reasons.append("attachment-enables-attacker")
            if effect.target_is_active:
                score += 18
                reasons.append("develop-active-attacker")
            if effect.target_is_closest:
                score += 14
                reasons.append("develop-closest-attacker")
        elif plan.state == TurnState.ATTACK_READY:
            if effect.improves_readiness:
                score += 14
                reasons.append("attachment-maintains-future-pressure")
            if effect.enables_attack:
                score += 10
                reasons.append("attachment-readies-backup")
            if effect.target_is_active and effect.improves_readiness:
                score += 8
                reasons.append("develop-active-attacker")
        if not effect.known_target:
            score += 5
            reasons.append("energy-tempo-visible-attach")

        if plan.state == TurnState.NO_ATTACKER_READY:
            score += 50
            reasons.append("state-goal-find-attacker")
        elif plan.state == TurnState.ATTACKER_NEAR_READY:
            score += 35
            reasons.append("state-goal-finish-attacker")
        elif plan.state == TurnState.ATTACK_READY:
            score += 5
            reasons.append("state-goal-maintain-pressure")

    elif current_option_type == "ATTACK":
        if plan.state == TurnState.ATTACK_READY:
            score += 20
            reasons.append("state-goal-attack-ready")
        if select_type == "MAIN" and context == "MAIN":
            pressure = attack_pressure(option, obs, use_card_semantics)
            if pressure == "ko":
                score += 55
                reasons.append("prize-pressure-visible-ko")
            elif pressure == "meaningful":
                score += 22
                reasons.append("prize-pressure-meaningful-damage")
            elif pressure == "weak" and has_main_setup_option(obs):
                score -= 24
                reasons.append("sequence-setup-before-weak-attack")
        if plan.bench_count == 0 and plan.legal_play_available:
            score -= 10
            reasons.append("avoid-lone-active-before-setup")
        elif plan.legal_attach_available and not plan.energy_attached and plan.active_shortfall != 0:
            score -= 25
            reasons.append("consider-attachment-before-attack")

    elif current_option_type in {"PLAY", "EVOLVE", "ABILITY", "SKILL"}:
        if plan.state in {TurnState.NO_ATTACKER_READY, TurnState.ATTACKER_NEAR_READY}:
            score += 12
            reasons.append("state-goal-development")
        if select_type == "MAIN" and context == "MAIN" and plan.state == TurnState.ATTACK_READY and current_option_type in {"PLAY", "EVOLVE"}:
            score += 8
            reasons.append("sequence-development-before-weak-pressure")
        if current_option_type == "PLAY" and plan.bench_count == 0 and plan.legal_attack_available:
            score += 20
            reasons.append("setup-before-attacking-with-lone-active")
        if plan.legal_attach_available and not plan.energy_attached and current_option_type in {"PLAY", "EVOLVE"}:
            score -= 10
            reasons.append("attachment-window-still-open")

    elif current_option_type in RISKY_ACTIONS:
        if plan.legal_attach_available or plan.legal_attack_available:
            score -= 30
            reasons.append("avoid-risk-while-safer-tempo-exists")

    return score, reasons


def _score_candidates_with(obs: dict[str, Any], scorer: Callable[[dict[str, Any], str, str], tuple[int, list[str]]]) -> list[Candidate]:
    select = obs.get("select")
    if select is None:
        return []
    select_type = enum_name(SelectType, select.get("type"))
    context = enum_name(SelectContext, select.get("context"))
    candidates = []
    for index, option in enumerate(select.get("option") or []):
        current_option_type = enum_name(OptionType, option.get("type"))
        score, reasons = scorer(option, select_type, context)
        candidates.append(
            Candidate(
                index=index,
                option=option,
                option_type=current_option_type,
                select_type=select_type,
                context=context,
                score=score,
                reasons=tuple(reasons),
            )
        )
    return candidates


def score_candidates_v2(obs: dict[str, Any]) -> list[Candidate]:
    return _score_candidates_with(obs, score_option_v2)


def score_candidates(
    obs: dict[str, Any],
    use_card_semantics: bool = True,
    attach_timing: str = "old",
    evolve_priority: str = "old",
) -> list[Candidate]:
    candidates = _score_candidates_with(
        obs,
        lambda option, select_type, context: _score_option(
            option,
            select_type,
            context,
            obs,
            use_card_semantics=use_card_semantics,
        ),
    )
    select = obs.get("select") or {}
    options = select.get("option") or []
    plan = analyze_board(obs)

    if evolve_priority == "new":
        facts = extract_evolve_priority_facts(
            obs,
            select,
            options,
            plan,
            use_card_semantics=use_card_semantics,
        )
        decision = classify_evolve_priority(facts)
        candidates = apply_evolve_priority_adjustment(candidates, decision, facts)

    if attach_timing != "new":
        return candidates

    config = {"use_card_semantics": use_card_semantics}
    facts = extract_attach_timing_facts(obs, select, options, plan, config)
    decision = classify_attach_timing(facts)
    return apply_attach_timing_adjustment(candidates, options, decision, config, obs)
