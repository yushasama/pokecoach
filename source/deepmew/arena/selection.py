from __future__ import annotations

import random
from typing import Any

import main
from cg.api import AreaType, OptionType, SelectContext
from deepmew.arena.models import AblationConfig, AgentSpec
from deepmew.policy import Candidate, choose_action, choose_action_v2, guard_selection, score_candidates
from deepmew.policy.greedy import choose_greedy_action
from deepmew.turn_plan import choose_v4_action


SETUP_TYPES = {"PLAY", "EVOLVE", "ATTACH", "ABILITY", "SKILL"}


def _option_type_name(option: dict[str, Any]) -> str:
    value = option.get("type")
    try:
        return OptionType(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _context_name(select: dict[str, Any]) -> str:
    value = select.get("context")
    try:
        return SelectContext(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _choose_first(select: dict[str, Any], wants: str) -> list[int] | None:
    if _context_name(select) != "IS_FIRST":
        return None
    desired = "YES" if wants == "yes" else "NO"
    for index, option in enumerate(select.get("option") or []):
        if _option_type_name(option) == desired:
            return guard_selection([index], select)
    return None


def _choose_attach_target(select: dict[str, Any], target: str) -> list[int] | None:
    options = select.get("option") or []
    attach_indices = [index for index, option in enumerate(options) if _option_type_name(option) == "ATTACH"]
    if not attach_indices:
        return None
    if target == "first":
        return guard_selection([attach_indices[0]], select)
    for index in attach_indices:
        try:
            area = AreaType(options[index].get("inPlayArea")).name
        except (TypeError, ValueError):
            area = str((options[index].get("inPlayArea") or "")).upper()
        if target == "active" and area == "ACTIVE":
            return guard_selection([index], select)
        if target == "bench" and area == "BENCH":
            return guard_selection([index], select)
    return None


def _semantics_enabled(config: AblationConfig) -> bool:
    return config.semantics != "off"


def _attach_timing_mode(config: AblationConfig) -> str:
    return "new" if config.attach_timing == "new" else "old"


def _evolve_priority_mode(config: AblationConfig) -> str:
    return "new" if config.evolve_priority == "new" else "old"


def _choose_with_adjusted_scores(obs: dict[str, Any], config: AblationConfig) -> list[int] | None:
    if not any((config.attach_bonus, config.attack_bonus, config.setup_bonus)):
        return None
    select = obs.get("select") or {}
    candidates = []
    for candidate in score_candidates(
        obs,
        use_card_semantics=_semantics_enabled(config),
        attach_timing=_attach_timing_mode(config),
        evolve_priority=_evolve_priority_mode(config),
    ):
        score = candidate.score
        reasons = list(candidate.reasons)
        if candidate.option_type == "ATTACH" and config.attach_bonus:
            score += config.attach_bonus
            reasons.append(f"ablate-attach-bonus={config.attach_bonus}")
        if candidate.option_type == "ATTACK" and config.attack_bonus:
            score += config.attack_bonus
            reasons.append(f"ablate-attack-bonus={config.attack_bonus}")
        if candidate.option_type in SETUP_TYPES and config.setup_bonus:
            score += config.setup_bonus
            reasons.append(f"ablate-setup-bonus={config.setup_bonus}")
        candidates.append(
            Candidate(
                index=candidate.index,
                option=candidate.option,
                option_type=candidate.option_type,
                select_type=candidate.select_type,
                context=candidate.context,
                score=score,
                reasons=tuple(reasons),
            )
        )
    if not candidates:
        return None

    ordered = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.index))
    min_count = max(0, min(int(select.get("minCount") or 0), len(ordered)))
    max_count = max(min_count, min(int(select.get("maxCount") or 0), len(ordered)))
    if max_count == 0:
        return guard_selection([], select)
    if min_count == max_count:
        selection = [candidate.index for candidate in ordered[:max_count]]
    else:
        positive = [candidate.index for candidate in ordered if candidate.score > 0]
        selection = positive[: max(min_count, min(max_count, len(positive)))]
    return guard_selection(selection, select)


def select_action(spec: AgentSpec, obs: dict[str, Any], rng: random.Random) -> list[int]:
    select = obs.get("select")
    if select is None:
        return main.read_deck_csv()

    if spec.ablation.first_choice != "default":
        choice = _choose_first(select, spec.ablation.first_choice)
        if choice is not None:
            return choice

    if spec.ablation.attach_target != "default":
        choice = _choose_attach_target(select, spec.ablation.attach_target)
        if choice is not None:
            return choice

    adjusted = _choose_with_adjusted_scores(obs, spec.ablation)
    if adjusted is not None:
        return adjusted

    options = select.get("option") or []
    if spec.base == "random":
        max_count = min(int(select.get("maxCount") or 0), len(options))
        return guard_selection(rng.sample(list(range(len(options))), max_count), select)
    if spec.base == "first-legal":
        max_count = min(int(select.get("maxCount") or 0), len(options))
        return guard_selection(list(range(max_count)), select)
    if spec.base == "greedy":
        return choose_greedy_action(obs)
    if spec.base == "legacy-v2":
        return choose_action_v2(obs)
    if spec.base == "v2.4":
        return choose_action(
            obs,
            use_card_semantics=_semantics_enabled(spec.ablation),
            attach_timing=_attach_timing_mode(spec.ablation),
            evolve_priority=_evolve_priority_mode(spec.ablation),
        )
    if spec.base == "v4":
        return choose_v4_action(obs)
    raise ValueError(f"unknown agent base: {spec.base}")
