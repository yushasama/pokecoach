from __future__ import annotations

from typing import Any

from cg.api import OptionType
from deepmew.policy.attacks import attack_base_score, attack_damage
from deepmew.policy.enums import KNOWN_OPTION_TYPES, enum_name


def score_option_v2(option: dict[str, Any], select_type: str, context: str) -> tuple[int, list[str]]:
    option_type = enum_name(OptionType, option.get("type"))
    score = 0
    reasons: list[str] = []

    if option.get("type") not in KNOWN_OPTION_TYPES:
        return -100, ["unknown-option-type"]

    if option_type == "ATTACK":
        damage = attack_damage(option.get("attackId"))
        attack_score, reason = attack_base_score(damage)
        score += attack_score
        reasons.append(reason)
    elif option_type == "SKILL":
        score += 72
        reasons.append("use-visible-skill")
    elif option_type == "EVOLVE":
        score += 68
        reasons.append("develop-evolution")
    elif option_type == "ATTACH":
        score += 62
        reasons.append("develop-energy")
    elif option_type == "PLAY":
        score += 70
        reasons.append("play-setup-card")
    elif option_type == "ABILITY":
        score += 52
        reasons.append("use-ability")
    elif option_type == "END":
        score -= 45
        reasons.append("end-low-priority")
    elif option_type in {"DISCARD", "RETREAT"}:
        score -= 35
        reasons.append("risky-resource-action")
    elif option_type in {"YES", "NO"}:
        if context == "IS_FIRST":
            score += 12 if option_type == "NO" else 0
            reasons.append("prefer-second-for-first-attack")
        else:
            score += 5 if option_type == "YES" else 0
            reasons.append("low-stakes-choice")
    elif option_type == "NUMBER":
        number = int(option.get("number") or 0)
        score += min(number, 3) * 8
        reasons.append("prefer-positive-count")
    elif option_type == "CARD":
        score += 20
        reasons.append("context-card-choice")
    elif option_type in {"TOOL_CARD", "ENERGY_CARD", "ENERGY", "SPECIAL_CONDITION"}:
        score += 5
        reasons.append("contextual-subchoice")
    else:
        score -= 25
        reasons.append("unclassified-action")

    if context == "SETUP_ACTIVE_POKEMON":
        score += 80
        reasons.append("must-establish-active")
    elif context == "TO_ACTIVE":
        score += 75
        reasons.append("replace-active")
    elif context == "TO_HAND":
        score += 58
        reasons.append("draw-search-prize")
    elif context == "SETUP_BENCH_POKEMON":
        score += 18
        reasons.append("setup-bench")
    elif context in {"EVOLVE", "EVOLVES_FROM", "EVOLVES_TO"}:
        score += 45
        reasons.append("evolution-context")
    elif context in {"ATTACK", "DAMAGE"}:
        score += 35
        reasons.append("attack-context")
    elif context in {"DISCARD", "DISCARD_CARD_OR_ATTACHED_CARD", "DISCARD_ENERGY", "DISCARD_ENERGY_CARD", "DISCARD_TOOL_CARD"}:
        score -= 45
        reasons.append("discard-context")
    elif context in {"SWITCH", "RETREAT"}:
        score -= 20
        reasons.append("movement-context")

    if select_type in {"CARD", "COUNT"} and context in {"TO_HAND", "DRAW_COUNT"}:
        score += 25
        reasons.append("resource-acquisition")

    return score, reasons
