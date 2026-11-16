"""DeckProfileInference v0: function tags -> deck profile.

Second stage of the meta pipeline:

    card text -> function tags -> deck profile -> matchup threats
                 -> action scores -> reason tags

This package only owns ``function tags -> deck profile``. It aggregates
CardFunctionClassifier v2 tags over mined 60-card decklists into generalized,
count-weighted deck profiles (plans, dependencies, choke points, reason tags),
keeping the three kinds of bench pressure and the kinds of protection separate.

Offline/training-only. Archetype labels are carried through as metadata, never
used as inference logic; no action scores, trained labels, deck-name scripts, or
matchup win rates are produced. Never imported by the submission runtime
(`main.py`) or the battle engine (`cg`) (INVARIANT #7/#8).
"""

from __future__ import annotations

from deepmew.deck_profile_inference.card_functions import DEFAULT_V2_CSV, load_v2_cards
from deepmew.deck_profile_inference.decklists import find_latest_decklists, load_decklists
from deepmew.deck_profile_inference.inference import infer_all, infer_profile
from deepmew.deck_profile_inference.schemas import (
    ARTIFACT_VERSION,
    CardRef,
    DeckList,
    DeckProfile,
    ProfileConfig,
    V2Card,
)

__all__ = [
    "ARTIFACT_VERSION",
    "CardRef",
    "DeckList",
    "DeckProfile",
    "ProfileConfig",
    "V2Card",
    "DEFAULT_V2_CSV",
    "load_v2_cards",
    "load_decklists",
    "find_latest_decklists",
    "infer_profile",
    "infer_all",
]
