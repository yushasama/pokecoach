"""Conditional probability and lift of cards within provisional archetypes.

``lift(card, archetype) = P(card | archetype) / P(card overall)`` separates
archetype signatures (low frequency, high lift) from generic staples (high
frequency, low lift). Probabilities are smoothed so small archetypes stay sane.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from deepmew.meta_kg.schemas import (
    ArchetypeAssignment,
    CardId,
    K2ABuildConfig,
    UniqueDeck,
)

EPS = 1e-12


def smoothed_global_prob(card_deck_count: int, total_decks: int, alpha: float) -> float:
    return (card_deck_count + alpha) / (total_decks + 2 * alpha)


def smoothed_conditional_prob(card_count_in_archetype: int, archetype_deck_count: int, alpha: float) -> float:
    return (card_count_in_archetype + alpha) / (archetype_deck_count + 2 * alpha)


def lift_value(p_cond: float, p_global: float) -> float:
    return (p_cond + EPS) / (p_global + EPS)


def log_lift_value(p_cond: float, p_global: float) -> float:
    return math.log(p_cond + EPS) - math.log(p_global + EPS)


def compute_card_lift(
    assignments: list[ArchetypeAssignment],
    unique_decks: dict[str, UniqueDeck],
    config: K2ABuildConfig,
    *,
    staple_scores: dict[CardId, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute one lift row per (archetype, card) above reporting thresholds.

    Archetype label per deck is the assignment's primary_archetype. Decks
    labelled ``Unknown`` still contribute to overall probabilities but get their
    own ``Unknown`` archetype bucket so signatures stay comparable.
    """
    alpha = config.lift_alpha
    total_decks = max(1, len(unique_decks))
    staple_scores = staple_scores or {}

    global_deck_count: dict[CardId, int] = defaultdict(int)
    for deck in unique_decks.values():
        for cid in deck.cards:
            global_deck_count[cid] += 1

    archetype_decks: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        archetype_decks[assignment.primary_archetype].append(assignment.deck_hash)

    rows: list[dict[str, Any]] = []
    for archetype, deck_hashes in archetype_decks.items():
        archetype_deck_count = len(deck_hashes)
        card_count_in_archetype: dict[CardId, int] = defaultdict(int)
        for deck_hash in deck_hashes:
            deck = unique_decks.get(deck_hash)
            if deck is None:
                continue
            for cid in deck.cards:
                card_count_in_archetype[cid] += 1
        for cid, in_count in card_count_in_archetype.items():
            p_cond = smoothed_conditional_prob(in_count, archetype_deck_count, alpha)
            p_global = smoothed_global_prob(global_deck_count.get(cid, 0), total_decks, alpha)
            lift = lift_value(p_cond, p_global)
            cond_freq = in_count / archetype_deck_count if archetype_deck_count else 0.0
            if lift < config.min_report_lift and cond_freq < config.min_report_conditional_freq:
                continue
            is_signature = cond_freq >= config.signature_min_conditional_freq and lift >= config.signature_min_lift
            is_tech = cond_freq <= config.tech_max_conditional_freq and lift >= config.tech_min_lift
            staple_score = staple_scores.get(cid, 0.0)
            role = _role_guess(cond_freq, lift, staple_score, is_signature, is_tech, config)
            rows.append(
                {
                    "archetype": archetype,
                    "card_id": cid,
                    "deck_count_in_archetype": in_count,
                    "archetype_deck_count": archetype_deck_count,
                    "global_deck_count": global_deck_count.get(cid, 0),
                    "unique_deck_count": total_decks,
                    "p_card_given_archetype": round(p_cond, 6),
                    "p_card_overall": round(p_global, 6),
                    "lift": round(lift, 4),
                    "log_lift": round(log_lift_value(p_cond, p_global), 4),
                    "cond_freq": round(cond_freq, 4),
                    "role_guess": role,
                    "staple_score": round(staple_score, 4),
                    "is_signature": is_signature,
                    "is_tech_candidate": is_tech,
                }
            )
    rows.sort(key=lambda r: (r["archetype"], -r["lift"], -r["cond_freq"], r["card_id"]))
    return rows


def _role_guess(
    cond_freq: float,
    lift: float,
    staple_score: float,
    is_signature: bool,
    is_tech: bool,
    config: K2ABuildConfig,
) -> str:
    if staple_score >= 0.5:
        return "staple"
    if is_signature:
        return "core"
    if is_tech:
        return "tech"
    if lift >= config.package_min_lift and cond_freq >= config.package_min_conditional_freq:
        return "support"
    return "unknown"
