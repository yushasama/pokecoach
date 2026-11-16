"""Deterministic archetype -> broad goal_of_deck mapping (K4.1).

goal_of_deck is intentionally broad (a deck-plan family, not a hyper-specific
archetype name). Mappings are curated from the K2A archetype vocabulary; archetypes
without a curated entry fall back to a transparent name-token heuristic, and decks
that match nothing emit ``unknown`` with zero confidence rather than a confident
guess. This module is pure string logic -- no engine, no replay I/O.
"""

from __future__ import annotations

from deepmew.imitation_fine.schemas import UNKNOWN

# Curated, high-confidence archetype -> goal_of_deck. Keys are normalized
# (lowercased, curly apostrophes folded to straight) so display variants match.
_CURATED: dict[str, str] = {
    "hop's trevenant": "control_stall",
    "mega lucario ex": "setup_sweeper",
    "mega abomasnow ex": "setup_sweeper",
    "mega kangaskhan ex": "setup_sweeper",
    "mega starmie ex": "tempo_pressure",
    "mega froslass ex": "disruption",
    "dragapult ex": "spread_damage",
    "dusknoir": "spread_damage",
    "alakazam": "spread_damage",
    "iono's bellibolt ex": "tempo_pressure",
    "cornerstone mask ogerpon ex": "control_stall",
    "ethan's typhlosion": "engine_setup",
    "cynthia's garchomp ex": "setup_sweeper",
    "barbaracle": "setup_sweeper",
    "seaking": "engine_setup",
    "togekiss": "toolbox_flex",
}

CURATED_CONFIDENCE = 0.9
HEURISTIC_CONFIDENCE = 0.5


def normalize_archetype(archetype: str) -> str:
    return (archetype or "").strip().lower().replace("’", "'")


def goal_of_deck(archetype: str) -> tuple[str, float]:
    """Return (goal_of_deck, confidence) for an archetype name."""
    key = normalize_archetype(archetype)
    if not key or key == UNKNOWN or key == "mixed":
        return UNKNOWN, 0.0

    curated = _CURATED.get(key)
    if curated is not None:
        return curated, CURATED_CONFIDENCE

    # Transparent fallback: a named ex/Mega line is almost always a board-building
    # single-attacker sweeper; everything else stays honestly unknown.
    if "mega" in key and "ex" in key:
        return "setup_sweeper", HEURISTIC_CONFIDENCE
    if key.endswith(" ex") or " ex " in key:
        return "setup_sweeper", HEURISTIC_CONFIDENCE
    return UNKNOWN, 0.0
