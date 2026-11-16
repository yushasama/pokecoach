"""Transparent strategic action scorer (matchup threats -> action scores + reason tags).

``score_action(candidate, context)`` returns an :class:`ActionScore` with a numeric
score, reason tags, a per-component breakdown, and a confidence. Every rule is
explicit and reuses MatchupThreatInference v0's ANSWER_MATRIX so the semantic
separations hold identically:

* a Battle Cage-style play (damage-counter prevention) facing ordinary bench *attack*
  damage answers nothing -> ``BAD_WRONG_PROTECTION_TYPE`` (never ANSWER_BENCH_ATTACK_DAMAGE);
* a Rabsca-style play (attack-effect protection) facing Ability counters answers
  nothing -> ``BAD_WRONG_PROTECTION_TYPE`` (never ANSWER_ABILITY_COUNTERS);
* Ability prevention facing Ability counters -> ``ANSWER_ABILITY_COUNTERS``;
* tool removal / stadium bump / energy denial only earn their tag when the
  opponent's dependency is load-bearing;
* a one-of tech answer that answers no live load-bearing threat -> ``BAD_LOW_VALUE_TECH_TIMING``.

No win rate / EV is ever computed. No archetype name is ever read.
"""

from __future__ import annotations

from deepmew.strategic_action_scoring.schemas import (
    ANSWER_MATRIX,
    ActionCandidate,
    ActionScore,
    BENCH_THREATS,
    KIND_ATTACK_ADVANCE,
    KIND_BENCH_PLAY,
    KIND_PLAY_ANSWER,
    KIND_RESOURCE_SPEND,
    KIND_SETUP_ENGINE,
    MatchupContext,
    PROTECTION_COVERAGE,
    ScoreComponent,
    ScorerConfig,
    SPREAD_THREATS,
    STRUCTURAL_ANSWER_THREATS,
    THREAT_ANSWER_TAG,
    reason_sort_key,
    weakest,
)

_BASE_BY_KIND = {
    KIND_PLAY_ANSWER: "base_play_answer",
    KIND_ATTACK_ADVANCE: "base_attack_advance",
    KIND_SETUP_ENGINE: "base_setup_engine",
    KIND_BENCH_PLAY: "base_bench_play",
    KIND_RESOURCE_SPEND: "base_resource_spend",
}


def _is_load_bearing(item, config: ScorerConfig) -> bool:
    return bool(item) and item.is_primary and item.strength >= config.load_bearing_min


def score_action(
    candidate: ActionCandidate, context: MatchupContext, config: ScorerConfig | None = None
) -> ActionScore:
    config = config or ScorerConfig()
    components: list[ScoreComponent] = []
    tags: set[str] = set()

    base = getattr(config, _BASE_BY_KIND.get(candidate.action_kind, "base_play_answer"))
    components.append(ScoreComponent("base_" + candidate.action_kind, base))
    score = base

    present = context.opponent_primary_threats()
    subject_exposes = context.subject_exposes()
    present_spread = bool(SPREAD_THREATS & set(present))
    answered_live = False
    used_threat_items = []

    # --- answering opponent threats with provided coverage --------------------
    for cov in candidate.provides_coverage:
        for threat_cat, item in present.items():
            rule = ANSWER_MATRIX.get(threat_cat)
            tag = THREAT_ANSWER_TAG.get(threat_cat)
            if rule is None or tag is None:
                continue
            # Structural answers only count when the dependency is load-bearing.
            if threat_cat in STRUCTURAL_ANSWER_THREATS and not _is_load_bearing(item, config):
                continue
            if cov in rule.full:
                bonus = config.answer_full_core if candidate.is_core else config.answer_full_tech
                components.append(ScoreComponent(f"{tag}(full)", bonus))
                score += bonus
                tags.add(tag)
                answered_live = True
                used_threat_items.append(item)
            elif cov in rule.partial:
                components.append(ScoreComponent(f"{tag}(partial)", config.answer_partial))
                score += config.answer_partial
                tags.add(tag)
                answered_live = True
                used_threat_items.append(item)

    # --- wrong protection type: brought protection, answered nothing present --
    brings_protection = bool(PROTECTION_COVERAGE & set(candidate.provides_coverage))
    present_bench = bool(BENCH_THREATS & set(present))
    if brings_protection and present_bench and not answered_live:
        components.append(ScoreComponent("BAD_WRONG_PROTECTION_TYPE", config.wrong_type_penalty))
        score += config.wrong_type_penalty
        tags.add("BAD_WRONG_PROTECTION_TYPE")

    # --- protect a fragile bench with the right protection --------------------
    right_bench_protection = bool(
        {"bench_attack_damage_protection", "damage_counter_protection"} & set(candidate.provides_coverage)
    )
    if right_bench_protection and "fragile_bench_liability" in subject_exposes and present_spread:
        components.append(ScoreComponent("PROTECT_FRAGILE_BENCH", config.protect_fragile_bonus))
        score += config.protect_fragile_bonus
        tags.add("PROTECT_FRAGILE_BENCH")

    # --- tempo / development --------------------------------------------------
    if candidate.advances_main_attacker:
        components.append(ScoreComponent("ADVANCE_MAIN_ATTACKER", config.advance_main_bonus))
        score += config.advance_main_bonus
        tags.add("ADVANCE_MAIN_ATTACKER")
    if candidate.sets_up_engine:
        components.append(ScoreComponent("SET_UP_ENGINE", config.setup_engine_bonus))
        score += config.setup_engine_bonus
        tags.add("SET_UP_ENGINE")

    # --- mistakes -------------------------------------------------------------
    if candidate.benches_fragile and present_spread:
        components.append(ScoreComponent("BAD_OVERBENCH_INTO_SPREAD", config.overbench_penalty))
        score += config.overbench_penalty
        tags.add("BAD_OVERBENCH_INTO_SPREAD")

    # A one-of tech answer that answered no live load-bearing threat is mistimed.
    if candidate.action_kind == KIND_PLAY_ANSWER and not candidate.is_core and not answered_live:
        components.append(ScoreComponent("BAD_LOW_VALUE_TECH_TIMING", config.low_value_tech_penalty))
        score += config.low_value_tech_penalty
        tags.add("BAD_LOW_VALUE_TECH_TIMING")

    if candidate.discards_core_piece:
        components.append(ScoreComponent("BAD_DISCARD_CORE_PIECE", config.discard_core_penalty))
        score += config.discard_core_penalty
        tags.add("BAD_DISCARD_CORE_PIECE")
    if candidate.resource_mistimed:
        components.append(ScoreComponent("BAD_RESOURCE_TIMING", config.resource_timing_penalty))
        score += config.resource_timing_penalty
        tags.add("BAD_RESOURCE_TIMING")

    # --- confidence -----------------------------------------------------------
    confidence = weakest(candidate.card_confidence, context.opponent.threat_confidence)
    for item in used_threat_items:
        if item.review_flagged:
            confidence = weakest(confidence, "medium")

    return ActionScore(
        candidate_id=candidate.candidate_id,
        card_id=candidate.card_id,
        card_name=candidate.card_name,
        action_kind=candidate.action_kind,
        subject_deck_id=context.subject.deck_id,
        opponent_deck_id=context.opponent.deck_id,
        score=score,
        reason_tags=sorted(tags, key=reason_sort_key),
        components=components,
        confidence=confidence,
    )
