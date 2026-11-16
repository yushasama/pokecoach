"""Weighted deck similarity (top-k neighbors) with staple downweighting.

Raw Jaccard is reported alongside the weighted score for debugging. Shared
staples are surfaced separately from shared core/package cards so a staple
bridge is visible rather than silently merging unrelated decks.
"""

from __future__ import annotations

from typing import Any

from deepmew.meta_kg.schemas import (
    ArchetypeAssignment,
    CardId,
    K2ABuildConfig,
    UniqueDeck,
)


def weighted_jaccard(a: dict[CardId, int], b: dict[CardId, int], weights: dict[CardId, float]) -> float:
    keys = set(a) | set(b)
    num = 0.0
    den = 0.0
    for cid in keys:
        w = weights.get(cid, 1.0)
        wa = w * a.get(cid, 0)
        wb = w * b.get(cid, 0)
        num += min(wa, wb)
        den += max(wa, wb)
    return 0.0 if den == 0 else num / den


def raw_jaccard(a: dict[CardId, int], b: dict[CardId, int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_deck_similarity(
    unique_decks: dict[str, UniqueDeck],
    card_weights: dict[CardId, float],
    assignments: list[ArchetypeAssignment],
    staple_card_ids: set[CardId],
    config: K2ABuildConfig,
) -> list[dict[str, Any]]:
    assignment_by_hash = {a.deck_hash: a for a in assignments}
    core_by_hash = {a.deck_hash: set(a.core_cards) | set(a.package_cards) for a in assignments}
    deck_items = sorted(unique_decks.items())
    rows: list[dict[str, Any]] = []

    for hash_a, deck_a in deck_items:
        scored: list[tuple[float, str]] = []
        for hash_b, deck_b in deck_items:
            if hash_a == hash_b:
                continue
            sim = weighted_jaccard(deck_a.cards, deck_b.cards, card_weights)
            if sim > 0:
                scored.append((sim, hash_b))
        scored.sort(key=lambda item: (-item[0], item[1]))
        a_assignment = assignment_by_hash.get(hash_a)
        for rank, (sim, hash_b) in enumerate(scored[: config.similarity_top_k], start=1):
            deck_b = unique_decks[hash_b]
            b_assignment = assignment_by_hash.get(hash_b)
            shared = set(deck_a.cards) & set(deck_b.cards)
            shared_staples = sorted(shared & staple_card_ids, key=_sort_key)
            shared_core = sorted((shared & (core_by_hash.get(hash_a, set()) | core_by_hash.get(hash_b, set()))) - staple_card_ids, key=_sort_key)
            shared_package = sorted((shared - set(shared_core) - set(shared_staples)), key=_sort_key)
            diff_sig = sorted(((core_by_hash.get(hash_a, set()) ^ core_by_hash.get(hash_b, set())) - staple_card_ids), key=_sort_key)
            rows.append(
                {
                    "deck_hash": hash_a,
                    "neighbor_deck_hash": hash_b,
                    "weighted_similarity": round(sim, 6),
                    "raw_jaccard": round(raw_jaccard(deck_a.cards, deck_b.cards), 6),
                    "shared_core_cards": shared_core,
                    "shared_package_cards": shared_package,
                    "shared_staples": shared_staples,
                    "different_signature_cards": diff_sig[:20],
                    "same_primary_archetype": bool(
                        a_assignment and b_assignment
                        and a_assignment.primary_archetype == b_assignment.primary_archetype
                    ),
                    "same_variant_label": bool(
                        a_assignment and b_assignment
                        and a_assignment.variant_label == b_assignment.variant_label
                    ),
                    "rank": rank,
                }
            )
    rows.sort(key=lambda r: (r["deck_hash"], r["rank"]))
    return rows


def _sort_key(card_id: CardId) -> tuple[int, Any]:
    try:
        return (0, int(card_id))
    except (TypeError, ValueError):
        return (1, str(card_id))
