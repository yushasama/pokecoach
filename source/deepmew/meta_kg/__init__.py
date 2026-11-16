"""K2A meta knowledge graph: mined ladder corpus -> machine-readable meta map.

Offline/training-only. This package must never be imported by the submission
runtime (`main.py`) or the battle engine (`cg`). It converts mined deck
artifacts into a card catalog, deck archetypes, variants, support packages,
staples, card lift, weighted deck similarity, and a live-belief payload consumed
by K2B (Real-meta Arena / ReplayPolicy), K3 (Deck Zoo), K4 (imitation), K5
(offline RL), and K6 (submission loop).
"""

from __future__ import annotations

from deepmew.meta_kg.build import K2ABuildResult, build_meta_kg
from deepmew.meta_kg.live_belief import (
    ArchetypeBeliefResult,
    LiveArchetypeBelief,
    build_live_belief_payload,
    live_belief_from_payload,
)
from deepmew.meta_kg.schemas import (
    ARTIFACT_VERSION,
    AnchorDefinition,
    ArchetypeAssignment,
    CardCatalogRow,
    DeckArtifact,
    K2ABuildConfig,
    OUTPUT_FILES,
    UniqueDeck,
)

__all__ = [
    "ARTIFACT_VERSION",
    "OUTPUT_FILES",
    "AnchorDefinition",
    "ArchetypeAssignment",
    "ArchetypeBeliefResult",
    "CardCatalogRow",
    "DeckArtifact",
    "K2ABuildConfig",
    "K2ABuildResult",
    "LiveArchetypeBelief",
    "UniqueDeck",
    "build_live_belief_payload",
    "build_meta_kg",
    "live_belief_from_payload",
]
