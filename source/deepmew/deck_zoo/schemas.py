"""K3 Deck Zoo schemas: curated representative decks and arena opponent sets.

The Deck Zoo is seeded from K2A outputs (deck_archetypes, archetype_summary,
deck_similarity), not guessed from raw replays. It selects representative decks
per archetype/variant, flags near-duplicates, assigns train/eval/holdout splits,
and groups decks into named arena opponent sets. Offline/training-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARTIFACT_VERSION = "deepmew.deck_zoo.v1"
DECK_ZOO_KIND = "deck_zoo"

OUTPUT_FILES = {
    "deck_zoo": "deck_zoo.csv",
    "deck_zoo_summary": "deck_zoo_summary.csv",
    "arena_opponent_sets": "arena_opponent_sets.csv",
    "arena_matchups": "arena_matchups.csv",
    "deck_zoo_decklists": "deck_zoo_decklists.jsonl",
    "deck_zoo_report": "deck_zoo_report.md",
}

# Opponent set names (stable contract for K2B arena / K4 imitation consumers).
SET_TOP_META = "top_meta"
SET_VARIANT_DIVERSITY = "variant_diversity"
SET_ANTI_OVERFIT_HOLDOUT = "anti_overfit_holdout"
SET_UNKNOWN_MIXED = "unknown_mixed"
SET_ENGINE_FIRST = "engine_first"

SPLIT_TRAIN = "train"
SPLIT_EVAL = "eval"
SPLIT_HOLDOUT = "holdout"


@dataclass
class DeckZooConfig:
    # Near-duplicate clustering (within an archetype) on weighted similarity.
    dup_similarity_threshold: float = 0.85
    # Max canonical representatives kept per archetype.
    max_reps_per_archetype: int = 8
    # Cumulative artifact share covered by the top_meta opponent set.
    top_meta_cumulative_share: float = 0.85
    # Minimum confidence for a deck to enter top_meta.
    top_meta_min_confidence: float = 0.5
    # An archetype needs at least this many unique decks to contribute variants.
    variant_min_archetype_decks: int = 4
    # Stratified split fractions (per archetype; best exemplar always trains).
    holdout_fraction: float = 0.25
    eval_fraction: float = 0.20
    # Representative score weights.
    weight_confidence: float = 0.5
    weight_artifact: float = 0.3
    weight_episode: float = 0.2
    # Suggested arena games per matchup (advisory only).
    suggested_games_per_matchup: int = 20


@dataclass
class DeckZooEntry:
    deck_hash: str
    archetype: str
    variant_label: str
    classification_status: str
    confidence: float
    representative_score: float
    artifact_count: int
    episode_count: int
    deck_count_in_cluster: int
    is_canonical: bool
    is_near_duplicate: bool
    cluster_id: str
    split: str
    core_cards: list[str] = field(default_factory=list)
    package_cards: list[str] = field(default_factory=list)
    staple_cards: list[str] = field(default_factory=list)
    tech_cards: list[str] = field(default_factory=list)
    representative_player_ids: list[str] = field(default_factory=list)
    representative_episode_ids: list[str] = field(default_factory=list)
    card_count: int = 0
    reason_selected: str = ""
