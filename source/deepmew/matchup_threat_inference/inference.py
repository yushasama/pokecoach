"""Top-level MatchupThreatInference v0: deck profiles -> threat surfaces -> pairings.

For each mined decklist we reuse DeckProfileInference v0 (stage 2) to get the
count-weighted DeckProfile, then derive a threat surface (presents/exposes) and a
coverage surface. We also build a bounded, label-free set of subject-vs-opponent
pairings so the relational coverage-gap categories have artifact evidence.

Determinism: decks are ordered by ``deck_hash`` and numbered ``deck_0000`` to match
DeckProfileInference v0 deck_ids; opponents are de-duplicated by their threat/coverage
*signature* (never by archetype name), so the pairing set is reproducible and diverse.
"""

from __future__ import annotations

from deepmew.deck_profile_inference.inference import infer_profile
from deepmew.deck_profile_inference.schemas import DeckList, ProfileConfig, V2Card
from deepmew.matchup_threat_inference.matchup import analyze_matchup
from deepmew.matchup_threat_inference.schemas import (
    DeckThreatProfile,
    MatchupConfig,
    MatchupPairing,
    coverage_sort_key,
    threat_sort_key,
    weakest,
)
from deepmew.matchup_threat_inference.surfaces import (
    build_coverage_surface,
    build_threat_surface,
    resolve,
)


def build_threat_profile(
    deck: DeckList, v2map: dict[str, V2Card], config: MatchupConfig, deck_id: str
) -> DeckThreatProfile:
    profile = infer_profile(deck, v2map, ProfileConfig(), deck_id=deck_id)
    resolved = resolve(deck, v2map)
    presents, exposes = build_threat_surface(resolved, profile, config)
    coverage = build_coverage_surface(resolved, config)

    # threat_confidence: weakest of profile confidence and the primary-threat confidences.
    threat_confidence = profile.profile_confidence
    for item in presents + exposes:
        if item.is_primary:
            threat_confidence = weakest(threat_confidence, item.confidence)

    reason_tags = sorted(
        [t.category for t in presents if t.is_primary] + [t.category for t in exposes if t.is_primary],
        key=threat_sort_key,
    )
    reason_tags += sorted([c.category for c in coverage if c.reliable], key=coverage_sort_key)

    review: list[str] = []
    for item in presents + exposes:
        if item.is_primary:
            review.extend(item.review_reasons)
    for item in coverage:
        if item.reliable:
            review.extend(item.review_reasons)
    if profile.manual_review_reasons:
        review.append("deck profile flagged for review (see DeckProfileInference v0)")

    return DeckThreatProfile(
        deck_id=deck_id,
        deck_hash=deck.deck_hash,
        archetype_label=deck.archetype,
        split=deck.split,
        profile_confidence=profile.profile_confidence,
        threat_confidence=threat_confidence,
        presents=presents,
        exposes=exposes,
        coverage=coverage,
        reason_tags=_dedup_keep_order(reason_tags),
        manual_review_reasons=_dedup_keep_order(review)[:6],
    )


def build_all_profiles(
    decks: list[DeckList], v2map: dict[str, V2Card], config: MatchupConfig | None = None
) -> list[DeckThreatProfile]:
    config = config or MatchupConfig()
    ordered = sorted(decks, key=lambda d: d.deck_hash)
    return [build_threat_profile(deck, v2map, config, f"deck_{i:04d}") for i, deck in enumerate(ordered)]


def _signature(profile: DeckThreatProfile) -> tuple:
    """Label-free signature of a deck's *threat* role: its primary threats
    (presents + exposes). A pairing's result depends only on what the opponent
    threatens with — not on the opponent's own coverage — so deduping opponents by
    this signature is both correct and keeps the pairing set bounded."""
    return tuple(sorted(t.category for t in profile.all_threats() if t.is_primary))


def select_opponents(profiles: list[DeckThreatProfile]) -> list[DeckThreatProfile]:
    """One representative opponent per unique primary-threat signature (lowest
    deck_id wins). Deterministic and independent of archetype labels."""
    seen: dict[tuple, DeckThreatProfile] = {}
    for profile in sorted(profiles, key=lambda p: p.deck_id):
        sig = _signature(profile)
        if sig not in seen:
            seen[sig] = profile
    return sorted(seen.values(), key=lambda p: p.deck_id)


def build_pairings(
    profiles: list[DeckThreatProfile], config: MatchupConfig | None = None
) -> list[MatchupPairing]:
    config = config or MatchupConfig()
    opponents = select_opponents(profiles)
    pairings: list[MatchupPairing] = []
    for subject in sorted(profiles, key=lambda p: p.deck_id):
        for opponent in opponents:
            if opponent.deck_id == subject.deck_id:
                continue
            pairings.append(analyze_matchup(subject, opponent, config))
    pairings.sort(key=lambda p: (p.subject_deck_id, p.opponent_deck_id))
    return pairings


def _dedup_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
