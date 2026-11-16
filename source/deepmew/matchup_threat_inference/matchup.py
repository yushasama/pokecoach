"""Subject-vs-opponent coverage-gap analysis (the relational matchup layer).

Given a subject deck's threat surface and an opponent deck's threat surface, decide,
for each of the opponent's primary threats, whether the subject's coverage answers it
(full / partial / none) and emit structured coverage-gap tags. This is where the
semantic separations bite:

* counter-placement prevention (Battle Cage) does NOT answer ordinary bench *attack*
  damage  ->  ``battle_cage_does_not_answer_ordinary_bench_damage`` + ``no_bench_attack_damage_answer``;
* attack-effect bench protection (Rabsca) does NOT answer Ability-placed counters
  ->  ``rabsca_does_not_answer_ability_counters`` + ``no_ability_counter_answer``;
* Ability prevention (Psyduck/Damp) DOES answer Ability damage-counter threat.

A one-of tech answer is recorded but never counts as a *full* reliable answer
(``reliable=False``), so it cannot paper over a gap. No win-rate / EV is computed.
Archetype labels are passed through for display only.
"""

from __future__ import annotations

from deepmew.matchup_threat_inference.schemas import (
    ANSWER_MATRIX,
    AnsweredThreat,
    DeckThreatProfile,
    MatchupConfig,
    MatchupPairing,
    threat_sort_key,
    weakest,
    lower_one,
)


def _classify_answer(
    threat_category: str, coverage: dict, config: MatchupConfig
) -> tuple[str, list[str]]:
    """Return (status, coverage categories used). ``reliable`` full coverage = full;
    a present-but-tech full answer or any present partial answer = partial."""
    rule = ANSWER_MATRIX.get(threat_category)
    if rule is None:
        return "none", []
    reliable_full = [c for c in rule.full if c in coverage and coverage[c].reliable]
    tech_full = [c for c in rule.full if c in coverage and not coverage[c].reliable]
    partial = [c for c in rule.partial if c in coverage]
    if reliable_full:
        return "full", sorted(reliable_full)
    if tech_full or partial:
        return "partial", sorted(set(tech_full) | set(partial))
    return "none", []


def analyze_matchup(
    subject: DeckThreatProfile, opponent: DeckThreatProfile, config: MatchupConfig | None = None
) -> MatchupPairing:
    config = config or MatchupConfig()
    coverage = subject.coverage_map()
    subject_present = {t.category for t in subject.presents if t.is_primary}
    subject_exposes = {t.category for t in subject.exposes if t.is_primary}

    pairing = MatchupPairing(
        subject_deck_id=subject.deck_id,
        opponent_deck_id=opponent.deck_id,
        subject_label=subject.archetype_label,
        opponent_label=opponent.archetype_label,
    )

    # Primary threats the opponent brings (offensive + structural).
    opp_threats = sorted(
        {t.category for t in opponent.all_threats() if t.is_primary},
        key=threat_sort_key,
    )
    pairing.opponent_threats = opp_threats

    gaps: list[str] = []
    review_reasons: list[str] = list(subject.manual_review_reasons[:0])  # start empty, fill below
    confidence = "high"

    has_reliable = lambda cat: cat in coverage and coverage[cat].reliable  # noqa: E731

    for threat in opp_threats:
        status, used = _classify_answer(threat, coverage, config)
        pairing.answered.append(AnsweredThreat(threat_category=threat, status=status, answered_by=used))
        rule = ANSWER_MATRIX.get(threat)
        if status == "full" or rule is None:
            continue
        # No full answer: emit the generic gap (when one is defined for this threat).
        if rule.gap:
            gaps.append(rule.gap)
        # Wrong-protection-type named gaps (the v1 conflation fixes).
        if threat == "bench_attack_damage_threat" and has_reliable("damage_counter_protection"):
            gaps.append("battle_cage_does_not_answer_ordinary_bench_damage")
        if threat == "ability_damage_counter_threat" and has_reliable("attack_effect_protection"):
            gaps.append("rabsca_does_not_answer_ability_counters")

    # Compound subject-liability gaps.
    opp_present = {t.category for t in opponent.presents if t.is_primary}
    spread = {"bench_attack_damage_threat", "bench_damage_counter_threat"}
    if (
        "fragile_bench_liability" in subject_exposes
        and (opp_present & spread)
        and not (has_reliable("bench_attack_damage_protection") or has_reliable("damage_counter_protection"))
    ):
        gaps.append("fragile_bench_into_spread")
    subject_setup_reliant = _is_setup_reliant(subject, config)
    if subject_setup_reliant and "lock_control_threat" in opp_present and not has_reliable("switching_outs"):
        gaps.append("setup_deck_into_lock")

    pairing.coverage_gaps = _dedup_sorted(gaps)

    # Confidence + manual-review propagation (test 11): if a load-bearing opponent
    # threat or a subject coverage item used here is review-flagged / low-confidence,
    # lower the matchup confidence and surface the reason.
    used_coverage = {c for a in pairing.answered for c in a.answered_by}
    for item in opponent.all_threats():
        if item.category in opp_threats and (item.review_flagged or item.confidence in ("low", "medium")):
            confidence = weakest(confidence, lower_one(item.confidence) if item.review_flagged else item.confidence)
            for reason in item.review_reasons:
                review_reasons.append(f"opponent threat {item.category}: {reason}")
    for cat in sorted(used_coverage):
        item = coverage.get(cat)
        if item and (item.review_flagged or item.confidence in ("low", "medium")):
            confidence = weakest(confidence, lower_one(item.confidence) if item.review_flagged else item.confidence)
            for reason in item.review_reasons:
                review_reasons.append(f"subject coverage {cat}: {reason}")
    if subject.profile_confidence == "low" or opponent.profile_confidence == "low":
        confidence = weakest(confidence, "low")

    pairing.matchup_confidence = confidence
    pairing.manual_review_reasons = _dedup_sorted(review_reasons)[:6]
    # Reason tags = the answered-threat statuses + the gap tags (Phase-4 inputs).
    answered_tags = [f"{a.status.upper()}:{a.threat_category}" for a in pairing.answered if a.status != "none"]
    pairing.reason_tags = _dedup_sorted(pairing.coverage_gaps + answered_tags)
    return pairing


def _is_setup_reliant(profile: DeckThreatProfile, config: MatchupConfig) -> bool:
    """A deck is setup-reliant if it exposes ability-engine reliance or runs a
    Stage-2 line heavy enough to need a turn to come online. Derived from threat
    evidence, not from any archetype name."""
    for item in profile.exposes:
        if item.category in ("fragile_bench_liability",) and item.strength >= config.bench_fragility_min:
            return True
    # ability_engine reliance shows up as an offensive ability_engine_threat too.
    for item in profile.presents:
        if item.category == "ability_engine_threat" and item.is_primary:
            return True
    return False


def _dedup_sorted(values: list[str]) -> list[str]:
    return sorted(set(values))
