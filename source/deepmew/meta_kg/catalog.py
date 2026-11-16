"""Canonical card catalog: one row per card ID with global counts/frequencies.

Card IDs are canonical and never dropped. Missing names recover to
``Unknown Card <card_id>``. Type/stage guesses are best-effort name heuristics
and are intentionally non-blocking for classification.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from deepmew.meta_kg.schemas import (
    BuildWarning,
    CardCatalogRow,
    CardId,
    DeckArtifact,
    UniqueDeck,
)

_ENERGY_HINTS = ("Energy",)
_TRAINER_HINTS = (
    "Ball", "Boss", "Order", "Switch", "Stretcher", "Poffin", "Hammer", "Belt",
    "Determination", "Compassion", "Candy", "Research", "Gear", "Vessel",
    "Stadium", "Tower", "Catcher", "Rope", "Turbo", "Capsule",
)


def _type_guess(name: str, name_missing: bool) -> str:
    if name_missing:
        return "unknown"
    lowered = name.lower()
    if any(hint.lower() in lowered for hint in _ENERGY_HINTS):
        return "energy"
    if any(hint.lower() in lowered for hint in _TRAINER_HINTS):
        return "trainer"
    # Pokemon names rarely contain these tokens; default to pokemon when a name
    # exists but no trainer/energy hint matched. This is a non-blocking guess.
    return "pokemon"


def best_card_name(card_id: CardId, observed_names: Counter) -> tuple[str, bool, list[str]]:
    non_empty = [name for name in observed_names if name and str(name).strip()]
    if not non_empty:
        return f"Unknown Card {card_id}", True, []
    best = max(non_empty, key=lambda name: (observed_names[name], name))
    alternates = sorted(name for name in non_empty if name != best)
    return best, False, alternates


def build_card_catalog(
    artifacts: list[DeckArtifact],
    unique_decks: dict[str, UniqueDeck],
    warnings: dict[str, BuildWarning] | None = None,
    card_meta: dict[CardId, Any] | None = None,
) -> tuple[list[CardCatalogRow], dict[CardId, list[str]]]:
    """Return (catalog_rows, name_alternates_by_card_id).

    When ``card_meta`` (from the engine card data) is provided, ``type_guess`` and
    ``stage_guess`` come from real card categories and ``metadata_status`` becomes
    ``enriched``; otherwise both fall back to name heuristics (``payload_only``).
    """
    card_meta = card_meta or {}
    unique_deck_count = max(1, len(unique_decks))
    artifact_count_total = max(1, len(artifacts))

    name_votes: dict[CardId, Counter] = defaultdict(Counter)
    copies_total: dict[CardId, int] = defaultdict(int)
    artifact_count: dict[CardId, int] = defaultdict(int)
    episode_ids: dict[CardId, set[str]] = defaultdict(set)
    players: dict[CardId, set[str]] = defaultdict(set)

    for artifact in artifacts:
        for cid, count in artifact.cards.items():
            copies_total[cid] += count
            artifact_count[cid] += 1
            if artifact.episode_id:
                episode_ids[cid].add(artifact.episode_id)
            if artifact.player_id:
                players[cid].add(artifact.player_id)
            name = artifact.names.get(cid)
            if name:
                name_votes[cid][name] += 1

    deck_count: dict[CardId, int] = defaultdict(int)
    for deck in unique_decks.values():
        for cid in deck.cards:
            deck_count[cid] += 1

    all_card_ids = set(copies_total) | set(deck_count)
    rows: list[CardCatalogRow] = []
    alternates_map: dict[CardId, list[str]] = {}
    for cid in all_card_ids:
        name, missing, alternates = best_card_name(cid, name_votes.get(cid, Counter()))
        if missing and warnings is not None:
            warnings.setdefault("missing_card_name", BuildWarning("missing_card_name", "card id has no display name")).bump(cid)
        if alternates and warnings is not None:
            warnings.setdefault("card_name_conflict", BuildWarning("card_name_conflict", "card id observed with multiple names")).bump(cid)
            alternates_map[cid] = alternates
        present_artifacts = artifact_count.get(cid, 0)
        meta = card_meta.get(cid)
        if meta is not None:
            type_guess = meta.type_guess
            stage_guess = meta.stage_guess
            metadata_status = "enriched"
        else:
            type_guess = _type_guess(name, missing)
            stage_guess = "unknown"
            metadata_status = "payload_only"
        rows.append(
            CardCatalogRow(
                card_id=cid,
                card_name=name,
                copies_total=copies_total.get(cid, 0),
                deck_count=deck_count.get(cid, 0),
                artifact_count=present_artifacts,
                episode_count=len(episode_ids.get(cid, ())),
                player_team_count=len(players.get(cid, ())),
                global_deck_freq=round(deck_count.get(cid, 0) / unique_deck_count, 6),
                global_artifact_freq=round(present_artifacts / artifact_count_total, 6),
                avg_copies_when_present=round(copies_total.get(cid, 0) / present_artifacts, 4) if present_artifacts else 0.0,
                name_missing=missing,
                metadata_status=metadata_status,
                type_guess=type_guess,
                stage_guess=stage_guess,
            )
        )

    rows.sort(key=lambda row: (-row.deck_count, _id_sort_key(row.card_id)))
    return rows, alternates_map


def _id_sort_key(card_id: CardId) -> tuple[int, Any]:
    """Sort numeric IDs numerically, fall back to string ordering."""
    try:
        return (0, int(card_id))
    except (TypeError, ValueError):
        return (1, str(card_id))


def catalog_row_to_dict(row: CardCatalogRow) -> dict[str, Any]:
    return {
        "card_id": row.card_id,
        "card_name": row.card_name,
        "copies_total": row.copies_total,
        "deck_count": row.deck_count,
        "artifact_count": row.artifact_count,
        "episode_count": row.episode_count,
        "player_team_count": row.player_team_count,
        "global_deck_freq": row.global_deck_freq,
        "global_artifact_freq": row.global_artifact_freq,
        "avg_copies_when_present": row.avg_copies_when_present,
        "name_missing": row.name_missing,
        "metadata_status": row.metadata_status,
        "type_guess": row.type_guess,
        "stage_guess": row.stage_guess,
    }


CATALOG_FIELDS = [
    "card_id", "card_name", "copies_total", "deck_count", "artifact_count",
    "episode_count", "player_team_count", "global_deck_freq", "global_artifact_freq",
    "avg_copies_when_present", "name_missing", "metadata_status", "type_guess", "stage_guess",
]
