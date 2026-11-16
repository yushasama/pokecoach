"""CardFunctionClassifier v2: card text -> function tags.

First stage of the meta pipeline:

    card text -> function tags -> deck profile -> matchup threats
                 -> action scores -> reason tags

This package only owns ``card text -> function tags``. It parses
``EN_Card_Data.csv`` (2,022 rows = 1,267 unique cards, one row per move/Ability)
into card-type-aware sections, applies transparent text rules with explicit
negative guards, and emits structured parse events, function tags, confidence,
and manual-audit metadata.

Offline/training-only. Never imported by the submission runtime (`main.py`) or
the battle engine (`cg`): it reads a data file unavailable at evaluation time
(INVARIANT #7) and respects the one-way module dependency direction (INVARIANT #8).
"""

from __future__ import annotations

from deepmew.card_functions.classify import classify_all, classify_card
from deepmew.card_functions.parsing import DEFAULT_CSV_PATH, load_card_inputs
from deepmew.card_functions.priority import load_priority_hints
from deepmew.card_functions.rules import normalize, scan_card
from deepmew.card_functions.schemas import (
    ARTIFACT_VERSION,
    TAG_ORDER,
    CardInput,
    ClassifiedCard,
    ClassifierConfig,
    ParseEvent,
    PriorityHints,
)

__all__ = [
    "ARTIFACT_VERSION",
    "TAG_ORDER",
    "CardInput",
    "ClassifiedCard",
    "ClassifierConfig",
    "ParseEvent",
    "PriorityHints",
    "DEFAULT_CSV_PATH",
    "classify_all",
    "classify_card",
    "load_card_inputs",
    "load_priority_hints",
    "normalize",
    "scan_card",
]
