"""Deck archetype + variant classification from anchor evidence and lift.

Terminal cards dominate. Engine packages can be primary when Pokemon evidence is
weak. Staples are downweighted. Unknown / mixed / engine_first are first-class
honest outputs, never trash bins.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from deepmew.meta_kg.schemas import (
    AnchorDefinition,
    AnchorScore,
    ArchetypeAssignment,
    CardId,
    K2ABuildConfig,
    MIXED_ARCHETYPE,
    STATUS_CONFIDENT,
    STATUS_ENGINE_FIRST,
    STATUS_LOW_CONFIDENCE,
    STATUS_MIXED,
    STATUS_UNKNOWN,
    UNKNOWN_ARCHETYPE,
    UniqueDeck,
)


def score_deck_against_anchor(
    deck: UniqueDeck,
    anchor: AnchorDefinition,
    staple_scores: dict[CardId, float],
    config: K2ABuildConfig,
) -> AnchorScore:
    cards = deck.cards
    if anchor.is_engine:
        engine_present = [c for c in anchor.engine_card_ids if c in cards]
        engine_fraction = len(engine_present) / max(1, len(anchor.engine_card_ids))
        anchor_copies = sum(cards.get(c, 0) for c in anchor.engine_card_ids)
        copy_bonus = min(config.max_copy_bonus, config.copy_weight * anchor_copies)
        evidence = engine_present
        staple_penalty = sum(staple_scores.get(c, 0.0) for c in evidence) * 0.25
        identity_score = config.engine_weight * engine_fraction
        score = identity_score + copy_bonus - staple_penalty
        return AnchorScore(
            anchor_id=anchor.anchor_id,
            label=anchor.label,
            anchor_type=anchor.anchor_type,
            score=max(0.0, score),
            terminal_score=identity_score,
            line_score=0.0,
            support_score=0.0,
            copy_score=copy_bonus,
            staple_penalty=staple_penalty,
            evidence_card_ids=evidence,
            priority=anchor.priority,
            terminal_is_wincon=True,
        )

    terminal_present = [c for c in anchor.terminal_card_ids if c in cards]
    line_present = [c for c in anchor.line_card_ids if c in cards]
    support_present = [c for c in anchor.support_card_ids if c in cards]

    terminal_fraction = len(terminal_present) / max(1, len(anchor.terminal_card_ids)) if anchor.terminal_card_ids else 0.0
    line_fraction = len(line_present) / max(1, len(anchor.line_card_ids)) if anchor.line_card_ids else 0.0
    support_fraction = len(support_present) / max(1, len(anchor.support_card_ids)) if anchor.support_card_ids else 0.0

    anchor_card_ids = anchor.terminal_card_ids + anchor.line_card_ids + anchor.support_card_ids
    anchor_copies = sum(cards.get(c, 0) for c in anchor_card_ids)
    copy_bonus = min(config.max_copy_bonus, config.copy_weight * anchor_copies)

    evidence = terminal_present + line_present + support_present
    staple_penalty = sum(staple_scores.get(c, 0.0) for c in evidence) * 0.25

    eff_terminal_weight = config.terminal_weight if anchor.terminal_is_wincon else config.terminal_weight * config.nonwincon_terminal_scale
    terminal_score = eff_terminal_weight * terminal_fraction
    line_score = config.line_weight * line_fraction
    support_score = config.support_weight * support_fraction
    score = terminal_score + line_score + support_score + copy_bonus - staple_penalty

    return AnchorScore(
        anchor_id=anchor.anchor_id,
        label=anchor.label,
        anchor_type=anchor.anchor_type,
        score=max(0.0, score),
        terminal_score=terminal_score,
        line_score=line_score,
        support_score=support_score,
        copy_score=copy_bonus,
        staple_penalty=staple_penalty,
        evidence_card_ids=evidence,
        priority=anchor.priority,
        terminal_is_wincon=anchor.terminal_is_wincon,
    )


def compute_confidence(
    top: AnchorScore | None,
    second: AnchorScore | None,
    deck: UniqueDeck,
    config: K2ABuildConfig,
) -> float:
    if top is None or top.score <= 0:
        return 0.0
    strength = min(1.0, top.score / max(config.primary_threshold, 1e-9))
    if second is None or second.score <= 0:
        margin = 1.0
    else:
        margin = min(1.0, max(0.0, (top.score - second.score) / max(top.score, 1e-9)))
    terminal_evidence = 1.0 if top.terminal_score > 0 else 0.0
    support_consistency = 1.0 if (top.support_score > 0 or top.line_score > 0) else 0.0
    artifact_support = min(1.0, math.log1p(deck.artifact_count) / math.log(10))
    staple_penalty = min(0.25, top.staple_penalty / max(top.score, 1e-9))
    confidence = (
        0.35 * strength
        + 0.25 * margin
        + 0.20 * terminal_evidence
        + 0.15 * support_consistency
        + 0.05 * artifact_support
        - staple_penalty
    )
    return max(0.0, min(1.0, confidence))


def _sorted_scores(scores: list[AnchorScore]) -> list[AnchorScore]:
    # Highest score wins; on ties, higher-priority (manual > auto > fallback)
    # anchors win so a configured core beats an auto support Pokemon.
    return sorted(scores, key=lambda s: (-s.score, -s.priority, s.anchor_id))


def assign_deck_archetype(
    deck: UniqueDeck,
    anchors: list[AnchorDefinition],
    staple_scores: dict[CardId, float],
    config: K2ABuildConfig,
) -> ArchetypeAssignment:
    all_scores = _sorted_scores(
        [score_deck_against_anchor(deck, anchor, staple_scores, config) for anchor in anchors]
    )
    top = all_scores[0] if all_scores else None
    second = all_scores[1] if len(all_scores) > 1 else None

    rank_bucket_summary = dict(deck.rank_buckets)
    anchor_scores = {s.anchor_id: round(s.score, 4) for s in all_scores[:5] if s.score > 0}

    if top is None or top.score < config.secondary_threshold:
        return _build_assignment(
            deck, UNKNOWN_ARCHETYPE, None, UNKNOWN_ARCHETYPE, 0.0, STATUS_UNKNOWN,
            core_cards=[], rank_bucket_summary=rank_bucket_summary, anchor_scores=anchor_scores,
            notes="no anchor exceeded the weak threshold",
        )

    confidence = compute_confidence(top, second, deck, config)
    primary = top.label
    secondary: str | None = None
    variant = primary

    if top.anchor_type == "engine" and top.score >= config.engine_primary_threshold:
        status = STATUS_ENGINE_FIRST
    elif top.score < config.primary_threshold or confidence < config.confident_confidence_threshold:
        status = STATUS_LOW_CONFIDENCE
    else:
        status = STATUS_CONFIDENT

    # Only a win-condition-grade second core can promote a deck to secondary/mixed.
    # Non-win-condition support Pokemon become package cards, not co-cores.
    if (
        second is not None
        and second.score >= config.secondary_threshold
        and second.anchor_id != top.anchor_id
        and second.terminal_is_wincon
    ):
        ratio = second.score / max(top.score, 1e-9)
        if ratio >= config.mixed_ratio:
            status = STATUS_MIXED
            primary = MIXED_ARCHETYPE if False else primary  # keep strongest core primary
            secondary = second.label
            variant = f"{top.label} / {second.label}"
        elif ratio >= config.secondary_ratio:
            secondary = second.label
            variant = f"{top.label} / {second.label}"

    notes = _notes(top, second, status)
    core_cards = sorted(set(top.evidence_card_ids), key=_card_sort_key)
    if secondary is not None and second is not None:
        core_cards = sorted(set(core_cards) | set(second.evidence_card_ids), key=_card_sort_key)

    return _build_assignment(
        deck, primary, secondary, variant, round(confidence, 4), status,
        core_cards=core_cards, rank_bucket_summary=rank_bucket_summary, anchor_scores=anchor_scores,
        notes=notes,
    )


def _card_sort_key(card_id: CardId) -> tuple[int, Any]:
    try:
        return (0, int(card_id))
    except (TypeError, ValueError):
        return (1, str(card_id))


def _notes(top: AnchorScore, second: AnchorScore | None, status: str) -> str:
    parts = [f"top={top.label}:{top.score:.2f}"]
    if second is not None and second.score > 0:
        parts.append(f"second={second.label}:{second.score:.2f}")
    parts.append(f"status={status}")
    return "; ".join(parts)


def _build_assignment(
    deck: UniqueDeck,
    primary: str,
    secondary: str | None,
    variant: str,
    confidence: float,
    status: str,
    *,
    core_cards: list[CardId],
    rank_bucket_summary: dict[str, int],
    anchor_scores: dict[str, float],
    notes: str,
    cap: int = 10,
) -> ArchetypeAssignment:
    return ArchetypeAssignment(
        deck_hash=deck.deck_hash,
        primary_archetype=primary,
        secondary_package=secondary,
        variant_label=variant,
        confidence=confidence,
        classification_status=status,
        core_cards=core_cards,
        package_cards=[],
        staple_cards=[],
        tech_cards=[],
        representative_team_ids=deck.team_ids[:cap],
        representative_player_ids=deck.player_ids[:cap],
        representative_episode_ids=deck.episode_ids[:cap],
        artifact_count=deck.artifact_count,
        episode_count=deck.episode_count,
        rank_bucket_summary=rank_bucket_summary,
        anchor_scores=anchor_scores,
        notes=notes,
    )


def classify_decks(
    unique_decks: dict[str, UniqueDeck],
    anchors: list[AnchorDefinition],
    staple_scores: dict[CardId, float],
    config: K2ABuildConfig,
) -> list[ArchetypeAssignment]:
    assignments = [
        assign_deck_archetype(unique_decks[h], anchors, staple_scores, config)
        for h in sorted(unique_decks)
    ]
    return assignments


def enrich_card_roles(
    assignments: list[ArchetypeAssignment],
    unique_decks: dict[str, UniqueDeck],
    lift_rows: list[dict[str, Any]],
    staple_card_ids: set[CardId],
    config: K2ABuildConfig,
) -> None:
    """Fill staple_cards / package_cards / tech_cards per deck using lift roles."""
    role_by_archetype_card: dict[tuple[str, CardId], str] = {}
    signature_cards: dict[str, set[CardId]] = defaultdict(set)
    for row in lift_rows:
        key = (row["archetype"], row["card_id"])
        role_by_archetype_card[key] = row["role_guess"]
        if row.get("is_signature"):
            signature_cards[row["archetype"]].add(row["card_id"])

    for assignment in assignments:
        deck = unique_decks.get(assignment.deck_hash)
        if deck is None:
            continue
        archetype = assignment.primary_archetype
        core_set = set(assignment.core_cards)
        staple_present: list[CardId] = []
        package_present: list[CardId] = []
        tech_present: list[CardId] = []
        for cid in deck.cards:
            if cid in staple_card_ids:
                staple_present.append(cid)
                continue
            role = role_by_archetype_card.get((archetype, cid))
            if role in ("support", "core") and cid not in core_set:
                package_present.append(cid)
            elif role == "tech":
                tech_present.append(cid)
        # Promote archetype signatures into core if present in the deck.
        for cid in signature_cards.get(archetype, ()):  # type: ignore[arg-type]
            if cid in deck.cards and cid not in core_set:
                assignment.core_cards.append(cid)
                core_set.add(cid)
        assignment.core_cards = sorted(set(assignment.core_cards), key=_card_sort_key)
        assignment.staple_cards = sorted(set(staple_present), key=_card_sort_key)
        assignment.package_cards = sorted(set(package_present), key=_card_sort_key)
        assignment.tech_cards = sorted(set(tech_present), key=_card_sort_key)
