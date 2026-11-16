from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

from cg.api import AreaType, OptionType, SelectContext, SelectType, all_attack
from deepmew.card_semantics import effective_attack_semantics
from deepmew.deck_profiles import (
    CURRENT_WATER_MEGA_ABOMASNOW,
    KYOGRE_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_LINE_IDS,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
    VISIBLE_SETUP_PLAY_IDS,
    WAITRESS_CARD_ID,
)
from deepmew.options import resolve_option_identity
from deepmew.state_machine import BoardPlan, attachment_effect


_ATTACK_DAMAGE: dict[int, int] | None = None


class AttachTimingState(str, Enum):
    UNKNOWN = "UNKNOWN"
    NO_OPINION = "NO_OPINION"
    FORCED_OR_TRIVIAL = "FORCED_OR_TRIVIAL"
    NO_ATTACHMENT_WINDOW = "NO_ATTACHMENT_WINDOW"
    TAKE_VISIBLE_KO = "TAKE_VISIBLE_KO"
    TAKE_PRIZE_PRESSURE = "TAKE_PRIZE_PRESSURE"
    IMPORTANT_EVOLVE_READY = "IMPORTANT_EVOLVE_READY"
    SPECIFIC_SETUP_PLAY_READY = "SPECIFIC_SETUP_PLAY_READY"
    WEAK_ATTACK_SHOULD_WAIT = "WEAK_ATTACK_SHOULD_WAIT"
    GENERIC_PLAY_SHOULD_WAIT = "GENERIC_PLAY_SHOULD_WAIT"
    ATTACH_TO_NEAR_READY_ATTACKER = "ATTACH_TO_NEAR_READY_ATTACKER"
    ATTACH_TO_MEGA_LINE = "ATTACH_TO_MEGA_LINE"
    ATTACH_TO_FUTURE_ATTACKER = "ATTACH_TO_FUTURE_ATTACKER"
    ATTACH_OPTIONAL = "ATTACH_OPTIONAL"


class AttachTimingAction(str, Enum):
    NO_OPINION = "NO_OPINION"
    PRESERVE = "PRESERVE"
    PREFER_ATTACK = "PREFER_ATTACK"
    PREFER_ATTACH = "PREFER_ATTACH"
    PREFER_EVOLVE = "PREFER_EVOLVE"
    PREFER_SPECIFIC_PLAY = "PREFER_SPECIFIC_PLAY"


@dataclass(frozen=True)
class AttachCandidateFacts:
    option_index: int
    target_area: str | None
    target_index: int | None
    target_card_id: int | None
    target_attached_energy: int | None
    target_shortfall_before: int | None
    target_shortfall_after: int | None
    improves_readiness: bool
    enables_attack: bool
    target_is_active: bool
    target_is_closest: bool
    target_is_mega_line: bool
    advances_underfed_attacker: bool
    priority: int


@dataclass(frozen=True)
class AttachTimingFacts:
    select_type: str | None
    context: str | None
    option_count: int
    attach_legal: bool
    energy_attached: bool
    generic_play_available: bool
    specific_setup_play_available: bool
    important_evolve_available: bool
    visible_ko_attack_available: bool
    prize_pressure_attack_available: bool
    weak_attack_available: bool
    meaningful_attack_available: bool
    near_ready_attacker_available: bool
    future_attacker_available: bool
    underfed_mega_line_available: bool
    attach_candidates: tuple[AttachCandidateFacts, ...] = ()
    best_attach: AttachCandidateFacts | None = None
    unknown: bool = False


@dataclass(frozen=True)
class AttachTimingDecision:
    state: AttachTimingState
    action: AttachTimingAction
    target_option_index: int | None = None
    reason: str = ""
    attach_score_delta: int = 0
    play_score_delta: int = 0
    attack_score_delta: int = 0


DeltaValue = int | Callable[[AttachTimingFacts], int]


@dataclass(frozen=True)
class AttachTimingRule:
    name: str
    guard: Callable[[AttachTimingFacts], bool]
    state: AttachTimingState
    action: AttachTimingAction
    attach_score_delta: DeltaValue = 0
    play_score_delta: DeltaValue = 0
    attack_score_delta: DeltaValue = 0

    def to_decision(self, facts: AttachTimingFacts) -> AttachTimingDecision:
        return AttachTimingDecision(
            state=self.state,
            action=self.action,
            target_option_index=facts.best_attach.option_index if facts.best_attach is not None else None,
            reason=self.name,
            attach_score_delta=_resolve_delta(self.attach_score_delta, facts),
            play_score_delta=_resolve_delta(self.play_score_delta, facts),
            attack_score_delta=_resolve_delta(self.attack_score_delta, facts),
        )


def _resolve_delta(value: DeltaValue, facts: AttachTimingFacts) -> int:
    return value(facts) if callable(value) else value


def _enum_name(enum_cls: Any, value: Any) -> str | None:
    if value is None:
        return None
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _option_type(option: dict[str, Any]) -> str | None:
    return _enum_name(OptionType, option.get("type"))


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return AreaType(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _card_id(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    card_id = value.get("cardId", value.get("cardID"))
    try:
        return int(card_id) if card_id is not None else None
    except (TypeError, ValueError):
        return None


def _option_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _semantics_enabled(config: Any) -> bool:
    if isinstance(config, dict):
        return bool(config.get("use_card_semantics", True))
    return bool(getattr(config, "use_card_semantics", True))


def _attack_damage(attack_id: Any) -> int | None:
    global _ATTACK_DAMAGE
    if attack_id is None:
        return None
    if _ATTACK_DAMAGE is None:
        _ATTACK_DAMAGE = {attack.attackId: attack.damage for attack in all_attack()}
    try:
        return _ATTACK_DAMAGE.get(int(attack_id))
    except (TypeError, ValueError):
        return None


def _attack_damage_for_option(option: dict[str, Any], observation: dict[str, Any], config: Any) -> int | None:
    if _semantics_enabled(config):
        semantics = effective_attack_semantics(option, observation)
        if semantics is not None:
            return semantics.effective_damage
    return _attack_damage(option.get("attackId"))


def _opponent_active_hp(observation: dict[str, Any]) -> int | None:
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        your_index = int(current.get("yourIndex") or 0)
    except (TypeError, ValueError):
        your_index = 0
    opponent_index = 1 - your_index
    if opponent_index < 0 or opponent_index >= len(players):
        return None
    active = (players[opponent_index] or {}).get("active") or []
    if not active or active[0] is None:
        return None
    try:
        return int(active[0].get("hp"))
    except (TypeError, ValueError):
        return None


def _attack_pressure(option: dict[str, Any], observation: dict[str, Any], config: Any) -> str:
    damage = _attack_damage_for_option(option, observation, config)
    if damage is None:
        return "unknown"
    opponent_hp = _opponent_active_hp(observation)
    if opponent_hp is not None and opponent_hp > 0:
        if damage >= opponent_hp:
            return "ko"
        if damage * 2 >= opponent_hp:
            return "meaningful"
        return "weak"
    if damage >= 100:
        return "meaningful"
    if damage <= 30:
        return "weak"
    return "unknown"


def _readiness_for_target(board_state: BoardPlan | None, option: dict[str, Any]) -> Any | None:
    if board_state is None:
        return None
    area = _area_name(option.get("inPlayArea"))
    index = _option_index(option.get("inPlayIndex")) or 0
    return next(
        (
            item
            for item in getattr(board_state, "readiness", ()) or ()
            if item.area == area and item.index == index
        ),
        None,
    )


def _is_specific_setup_play(option: dict[str, Any], observation: dict[str, Any] | None = None) -> bool:
    if _option_type(option) != "PLAY":
        return False
    card_id = _card_id(option)
    if observation is not None:
        identity = resolve_option_identity(observation, option)
        card_id = identity.card_id
    return CURRENT_WATER_MEGA_ABOMASNOW.is_setup_card(card_id)


def _is_important_evolve(board_state: BoardPlan | None, option: dict[str, Any]) -> bool:
    if _option_type(option) != "EVOLVE":
        return False
    if _card_id(option) == MEGA_ABOMASNOW_CARD_ID:
        return True
    target = _readiness_for_target(board_state, option)
    return bool(target is not None and CURRENT_WATER_MEGA_ABOMASNOW.is_mega_line(target.card_id))


def _attach_candidate(board_state: BoardPlan | None, option_index: int, option: dict[str, Any]) -> AttachCandidateFacts:
    target = _readiness_for_target(board_state, option)
    effect = attachment_effect(board_state, option) if board_state is not None else None
    target_card_id = getattr(target, "card_id", None)
    attached_energy = getattr(target, "attached_energy", None)
    target_is_mega_line = CURRENT_WATER_MEGA_ABOMASNOW.is_mega_line(target_card_id)
    improves = bool(getattr(effect, "improves_readiness", False))
    enables = bool(getattr(effect, "enables_attack", False))
    active = bool(getattr(effect, "target_is_active", False))
    closest = bool(getattr(effect, "target_is_closest", False))
    before = getattr(effect, "before_shortfall", None)
    after = getattr(effect, "after_shortfall", None)
    underfed = bool(target_is_mega_line and attached_energy is not None and attached_energy < 2)
    advances = bool(enables or improves or underfed)
    priority = (
        (100 if enables else 0)
        + (50 if improves else 0)
        + (25 if closest else 0)
        + (18 if active else 0)
        + (15 if underfed else 0)
        - (before or 0)
    )
    return AttachCandidateFacts(
        option_index=option_index,
        target_area=getattr(target, "area", _area_name(option.get("inPlayArea"))),
        target_index=getattr(target, "index", _option_index(option.get("inPlayIndex"))),
        target_card_id=target_card_id,
        target_attached_energy=attached_energy,
        target_shortfall_before=before,
        target_shortfall_after=after,
        improves_readiness=improves,
        enables_attack=enables,
        target_is_active=active,
        target_is_closest=closest,
        target_is_mega_line=target_is_mega_line,
        advances_underfed_attacker=advances,
        priority=priority,
    )


def extract_attach_timing_facts(
    observation: dict[str, Any],
    select: dict[str, Any] | None,
    options: list[dict[str, Any]],
    board_state: BoardPlan | None,
    config: Any,
) -> AttachTimingFacts:
    select = select or {}
    options = options or []
    select_type = _enum_name(SelectType, select.get("type"))
    context = _enum_name(SelectContext, select.get("context"))
    unknown = select_type is None or context is None

    attach_candidates = tuple(
        _attach_candidate(board_state, index, option)
        for index, option in enumerate(options)
        if _option_type(option) == "ATTACH"
    )
    best_attach = max(attach_candidates, key=lambda item: (item.priority, -item.option_index), default=None)

    attack_pressures = [
        _attack_pressure(option, observation, config)
        for option in options
        if _option_type(option) == "ATTACK"
    ]
    generic_play_available = any(
        _option_type(option) == "PLAY" and not _is_specific_setup_play(option, observation)
        for option in options
    )
    specific_setup_play_available = any(_is_specific_setup_play(option, observation) for option in options)
    important_evolve_available = any(_is_important_evolve(board_state, option) for option in options)
    energy_attached = bool(getattr(board_state, "energy_attached", False))

    return AttachTimingFacts(
        select_type=select_type,
        context=context,
        option_count=len(options),
        attach_legal=bool(attach_candidates),
        energy_attached=energy_attached,
        generic_play_available=generic_play_available,
        specific_setup_play_available=specific_setup_play_available,
        important_evolve_available=important_evolve_available,
        visible_ko_attack_available="ko" in attack_pressures,
        prize_pressure_attack_available="meaningful" in attack_pressures,
        weak_attack_available="weak" in attack_pressures,
        meaningful_attack_available=any(item in {"ko", "meaningful"} for item in attack_pressures),
        near_ready_attacker_available=any(item.enables_attack for item in attach_candidates),
        future_attacker_available=any(item.improves_readiness for item in attach_candidates),
        underfed_mega_line_available=any(
            item.target_is_mega_line
            and item.target_attached_energy is not None
            and item.target_attached_energy < 2
            for item in attach_candidates
        ),
        attach_candidates=attach_candidates,
        best_attach=best_attach,
        unknown=unknown,
    )


def is_unknown_selection(facts: AttachTimingFacts) -> bool:
    return facts.unknown


def is_not_main_phase(facts: AttachTimingFacts) -> bool:
    return facts.select_type != "MAIN" or facts.context != "MAIN"


def is_forced_or_trivial(facts: AttachTimingFacts) -> bool:
    return facts.option_count <= 1


def has_no_attachment_window(facts: AttachTimingFacts) -> bool:
    return not facts.attach_legal or facts.energy_attached


def has_visible_ko_attack(facts: AttachTimingFacts) -> bool:
    return facts.visible_ko_attack_available


def has_prize_pressure_attack(facts: AttachTimingFacts) -> bool:
    return facts.prize_pressure_attack_available


def has_important_evolve(facts: AttachTimingFacts) -> bool:
    return facts.important_evolve_available


def has_specific_setup_play(facts: AttachTimingFacts) -> bool:
    return facts.specific_setup_play_available


def weak_attack_and_near_ready_attacker(facts: AttachTimingFacts) -> bool:
    return facts.weak_attack_available and facts.near_ready_attacker_available


def generic_play_and_underfed_mega_line(facts: AttachTimingFacts) -> bool:
    return facts.generic_play_available and facts.underfed_mega_line_available


def has_near_ready_attacker(facts: AttachTimingFacts) -> bool:
    return facts.near_ready_attacker_available


def has_underfed_mega_line(facts: AttachTimingFacts) -> bool:
    return facts.underfed_mega_line_available


def has_future_attacker(facts: AttachTimingFacts) -> bool:
    return facts.future_attacker_available


def weak_attack_delta(facts: AttachTimingFacts) -> int:
    return -18 if facts.weak_attack_available else 0


def future_attacker_weak_attack_delta(facts: AttachTimingFacts) -> int:
    return -12 if facts.weak_attack_available else 0


def always(_: AttachTimingFacts) -> bool:
    return True


ATTACH_TIMING_RULES: tuple[AttachTimingRule, ...] = (
    AttachTimingRule(
        name="unknown-selection-shape",
        guard=is_unknown_selection,
        state=AttachTimingState.UNKNOWN,
        action=AttachTimingAction.NO_OPINION,
    ),
    AttachTimingRule(
        name="not-main-phase",
        guard=is_not_main_phase,
        state=AttachTimingState.NO_OPINION,
        action=AttachTimingAction.NO_OPINION,
    ),
    AttachTimingRule(
        name="forced-or-trivial",
        guard=is_forced_or_trivial,
        state=AttachTimingState.FORCED_OR_TRIVIAL,
        action=AttachTimingAction.PRESERVE,
    ),
    AttachTimingRule(
        name="no-attachment-window",
        guard=has_no_attachment_window,
        state=AttachTimingState.NO_ATTACHMENT_WINDOW,
        action=AttachTimingAction.NO_OPINION,
    ),
    AttachTimingRule(
        name="visible-ko-before-attach",
        guard=has_visible_ko_attack,
        state=AttachTimingState.TAKE_VISIBLE_KO,
        action=AttachTimingAction.PREFER_ATTACK,
    ),
    AttachTimingRule(
        name="prize-pressure-before-attach",
        guard=has_prize_pressure_attack,
        state=AttachTimingState.TAKE_PRIZE_PRESSURE,
        action=AttachTimingAction.PREFER_ATTACK,
    ),
    AttachTimingRule(
        name="important-evolve-before-attach",
        guard=has_important_evolve,
        state=AttachTimingState.IMPORTANT_EVOLVE_READY,
        action=AttachTimingAction.PREFER_EVOLVE,
        attach_score_delta=-28,
        play_score_delta=-12,
    ),
    AttachTimingRule(
        name="specific-setup-play-before-attach",
        guard=has_specific_setup_play,
        state=AttachTimingState.SPECIFIC_SETUP_PLAY_READY,
        action=AttachTimingAction.PREFER_SPECIFIC_PLAY,
    ),
    AttachTimingRule(
        name="near-ready-attach-before-weak-attack",
        guard=weak_attack_and_near_ready_attacker,
        state=AttachTimingState.WEAK_ATTACK_SHOULD_WAIT,
        action=AttachTimingAction.PREFER_ATTACH,
        attach_score_delta=28,
        play_score_delta=-28,
        attack_score_delta=-18,
    ),
    AttachTimingRule(
        name="mega-line-attach-before-generic-play",
        guard=generic_play_and_underfed_mega_line,
        state=AttachTimingState.GENERIC_PLAY_SHOULD_WAIT,
        action=AttachTimingAction.PREFER_ATTACH,
        attach_score_delta=28,
        play_score_delta=-28,
    ),
    AttachTimingRule(
        name="attach-enables-near-ready-attacker",
        guard=has_near_ready_attacker,
        state=AttachTimingState.ATTACH_TO_NEAR_READY_ATTACKER,
        action=AttachTimingAction.PREFER_ATTACH,
        attach_score_delta=28,
        play_score_delta=-28,
        attack_score_delta=weak_attack_delta,
    ),
    AttachTimingRule(
        name="attach-progresses-underfed-mega-line",
        guard=has_underfed_mega_line,
        state=AttachTimingState.ATTACH_TO_MEGA_LINE,
        action=AttachTimingAction.PREFER_ATTACH,
        attach_score_delta=28,
        play_score_delta=-28,
    ),
    AttachTimingRule(
        name="attach-progresses-future-attacker",
        guard=has_future_attacker,
        state=AttachTimingState.ATTACH_TO_FUTURE_ATTACKER,
        action=AttachTimingAction.PREFER_ATTACH,
        attach_score_delta=20,
        play_score_delta=-20,
        attack_score_delta=future_attacker_weak_attack_delta,
    ),
    AttachTimingRule(
        name="attach-optional",
        guard=always,
        state=AttachTimingState.ATTACH_OPTIONAL,
        action=AttachTimingAction.NO_OPINION,
    ),
)


def classify_attach_timing(facts: AttachTimingFacts) -> AttachTimingDecision:
    for rule in ATTACH_TIMING_RULES:
        if rule.guard(facts):
            return rule.to_decision(facts)
    raise AssertionError("unreachable attach-timing rule table")


def _candidate_index(candidate: Any) -> int:
    return int(getattr(candidate, "index"))


def _candidate_option_type(candidate: Any) -> str | None:
    return getattr(candidate, "option_type", None)


def _with_adjustment(candidate: Any, delta: int, reason: str, decision: AttachTimingDecision) -> Any:
    if delta == 0:
        return candidate
    reasons = tuple(getattr(candidate, "reasons", ()))
    next_reasons = reasons + (
        f"attach-timing-state={decision.state.value}",
        decision.reason,
        reason,
    )
    return replace(candidate, score=int(getattr(candidate, "score")) + delta, reasons=next_reasons)


def _adjust_for_prefer_attach(
    candidate: Any,
    option: dict[str, Any],
    candidate_type: str | None,
    candidate_index: int,
    decision: AttachTimingDecision,
    observation: dict[str, Any] | None,
) -> Any:
    if candidate_type == "ATTACH" and (
        decision.target_option_index is None or candidate_index == decision.target_option_index
    ):
        return _with_adjustment(
            candidate,
            decision.attach_score_delta,
            "attach-timing-before-generic-play",
            decision,
        )
    if candidate_type == "PLAY" and not _is_specific_setup_play(option, observation):
        return _with_adjustment(
            candidate,
            decision.play_score_delta,
            "delay-generic-play-for-underfed-attacker",
            decision,
        )
    if candidate_type == "ATTACK":
        return _with_adjustment(
            candidate,
            decision.attack_score_delta,
            "delay-weak-attack-for-attachment",
            decision,
        )
    return candidate


def _adjust_for_prefer_evolve(
    candidate: Any,
    option: dict[str, Any],
    candidate_type: str | None,
    candidate_index: int,
    decision: AttachTimingDecision,
    observation: dict[str, Any] | None,
) -> Any:
    _ = candidate_index
    if candidate_type == "ATTACH":
        return _with_adjustment(
            candidate,
            decision.attach_score_delta,
            "attach-timing-preserve-evolve",
            decision,
        )
    if candidate_type == "PLAY" and not _is_specific_setup_play(option, observation):
        return _with_adjustment(
            candidate,
            decision.play_score_delta,
            "delay-generic-play-for-evolve",
            decision,
        )
    return candidate


AttachAdjustmentHandler = Callable[[Any, dict[str, Any], str | None, int, AttachTimingDecision, dict[str, Any] | None], Any]

ATTACH_ADJUSTMENT_DISPATCH: dict[AttachTimingAction, AttachAdjustmentHandler] = {
    AttachTimingAction.PREFER_ATTACH: _adjust_for_prefer_attach,
    AttachTimingAction.PREFER_EVOLVE: _adjust_for_prefer_evolve,
}


def apply_attach_timing_adjustment(
    base_scores: list[Any],
    options: list[dict[str, Any]],
    decision: AttachTimingDecision,
    config: Any,
    observation: dict[str, Any] | None = None,
) -> list[Any]:
    _ = config
    handler = ATTACH_ADJUSTMENT_DISPATCH.get(decision.action)
    if handler is None:
        return base_scores

    adjusted: list[Any] = []
    for candidate in base_scores:
        candidate_type = _candidate_option_type(candidate)
        candidate_index = _candidate_index(candidate)
        option = options[candidate_index] if 0 <= candidate_index < len(options) else {}
        next_candidate = handler(candidate, option, candidate_type, candidate_index, decision, observation)
        adjusted.append(next_candidate)
    return adjusted
