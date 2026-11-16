"""Support package discovery and variant labelling.

A v0 package is the cohesive set of high-lift, non-staple cards for an archetype.
Variants refine a deck's label using a secondary core (already set during
classification) or, failing that, the archetype's support/energy package.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from deepmew.meta_kg.schemas import (
    ArchetypeAssignment,
    CardId,
    K2ABuildConfig,
    MIXED_ARCHETYPE,
    UNKNOWN_ARCHETYPE,
    UniqueDeck,
)


def _package_type_guess(card_names: list[str]) -> str:
    joined = " ".join(card_names).lower()
    if "energy" in joined:
        return "energy_engine"
    trainer_tokens = ("ball", "boss", "order", "research", "gear", "poffin", "switch", "candy")
    if any(token in joined for token in trainer_tokens):
        return "trainer_engine"
    return "pokemon_secondary"


def discover_support_packages(
    assignments: list[ArchetypeAssignment],
    unique_decks: dict[str, UniqueDeck],
    lift_rows: list[dict[str, Any]],
    staple_card_ids: set[CardId],
    card_names: dict[CardId, str],
    config: K2ABuildConfig,
) -> list[dict[str, Any]]:
    archetype_of_deck = {a.deck_hash: a.primary_archetype for a in assignments}
    decks_by_archetype: dict[str, list[str]] = defaultdict(list)
    for deck_hash, archetype in archetype_of_deck.items():
        decks_by_archetype[archetype].append(deck_hash)

    lift_by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lift_rows:
        lift_by_archetype[row["archetype"]].append(row)

    rows: list[dict[str, Any]] = []
    for archetype, candidates in sorted(lift_by_archetype.items()):
        if archetype in (UNKNOWN_ARCHETYPE, MIXED_ARCHETYPE):
            continue
        selected = [
            row for row in candidates
            if row["card_id"] not in staple_card_ids
            and float(row["lift"]) >= config.package_min_lift
            and float(row["cond_freq"]) >= config.package_min_conditional_freq
        ]
        if not selected:
            continue
        selected.sort(key=lambda r: (-float(r["lift"]), -float(r["cond_freq"]), r["card_id"]))
        selected = selected[: config.package_max_cards]
        card_ids = [row["card_id"] for row in selected]
        names = [card_names.get(cid, f"Unknown Card {cid}") for cid in card_ids]

        archetype_decks = decks_by_archetype.get(archetype, [])
        archetype_deck_count = max(1, len(archetype_decks))
        needed = max(1, len(card_ids) // 2)
        rep_hashes = [
            deck_hash for deck_hash in archetype_decks
            if sum(1 for cid in card_ids if cid in unique_decks[deck_hash].cards) >= needed
        ]
        mean_log_lift = sum(math.log(max(float(r["lift"]), 1e-9)) for r in selected) / len(selected)
        cohesion = min(float(r["cond_freq"]) for r in selected)
        package_score = round(mean_log_lift * cohesion, 4)
        deck_count = len(rep_hashes)
        rows.append(
            {
                "package_id": f"pkg_{_slug(archetype)}",
                "package_name": f"{archetype} Support Package",
                "owner_archetype": archetype,
                "package_type": _package_type_guess(names),
                "card_ids": card_ids,
                "card_names": names,
                "deck_count": deck_count,
                "artifact_count": sum(unique_decks[h].artifact_count for h in rep_hashes),
                "conditional_freq": round(deck_count / archetype_deck_count, 4),
                "global_freq": round(sum(float(r["p_card_overall"]) for r in selected) / len(selected), 6),
                "lift": round(sum(float(r["lift"]) for r in selected) / len(selected), 4),
                "cohesion_score": round(cohesion, 4),
                "package_score": package_score,
                "representative_deck_hashes": sorted(rep_hashes)[: config.representative_cap],
                "notes": f"{len(card_ids)} high-lift non-staple cards for {archetype}",
            }
        )
    rows.sort(key=lambda r: (r["owner_archetype"], -r["package_score"]))
    return rows


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "archetype"


def assign_variants(
    assignments: list[ArchetypeAssignment],
    support_packages: list[dict[str, Any]],
    card_names: dict[CardId, str],
    config: K2ABuildConfig,
) -> None:
    """Refine variant labels using support packages when no secondary core exists."""
    package_by_archetype = {pkg["owner_archetype"]: pkg for pkg in support_packages}
    for assignment in assignments:
        if assignment.secondary_package:
            continue  # multi-core secondary already wins
        if assignment.classification_status in ("unknown",):
            continue
        package = package_by_archetype.get(assignment.primary_archetype)
        if not package:
            continue
        package_cards = set(assignment.package_cards)
        if not package_cards:
            continue
        # If a deck strongly expresses the support package, tag it as a package variant.
        present = package_cards & set(package["card_ids"])
        if len(present) >= max(1, len(package["card_ids"]) // 3):
            assignment.variant_label = f"{assignment.primary_archetype} / {_short_package_label(package, card_names)}"


def _short_package_label(package: dict[str, Any], card_names: dict[CardId, str]) -> str:
    package_type = package.get("package_type", "support")
    label = {
        "energy_engine": "Energy Engine",
        "trainer_engine": "Trainer Engine",
        "pokemon_secondary": "Support Package",
    }.get(package_type, "Support Package")
    return label
