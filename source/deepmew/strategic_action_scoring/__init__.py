"""StrategicActionScorer v0 — offline ``matchup threats -> action scores`` (stage 4).

Transparent, explainable, offline. Never imported by ``main.py`` or ``cg``; trains no
labels; emits no win rates; uses no archetype/deck names as logic.
"""

from __future__ import annotations

from deepmew.strategic_action_scoring.candidates import build_candidates
from deepmew.strategic_action_scoring.dataset import build_score_rows, select_representative_opponents
from deepmew.strategic_action_scoring.scorer import score_action
from deepmew.strategic_action_scoring.schemas import (
    ANSWER_REASON_TAGS,
    BAD_REASON_TAGS,
    REASON_TAGS,
    THREAT_ANSWER_TAG,
    ActionCandidate,
    ActionScore,
    MatchupContext,
    ScoreComponent,
    ScorerConfig,
    KIND_ATTACK_ADVANCE,
    KIND_BENCH_PLAY,
    KIND_PLAY_ANSWER,
    KIND_RESOURCE_SPEND,
    KIND_SETUP_ENGINE,
)

__all__ = [
    "REASON_TAGS",
    "ANSWER_REASON_TAGS",
    "BAD_REASON_TAGS",
    "THREAT_ANSWER_TAG",
    "ActionCandidate",
    "ActionScore",
    "ScoreComponent",
    "MatchupContext",
    "ScorerConfig",
    "score_action",
    "build_candidates",
    "build_score_rows",
    "select_representative_opponents",
    "KIND_PLAY_ANSWER",
    "KIND_ATTACK_ADVANCE",
    "KIND_SETUP_ENGINE",
    "KIND_BENCH_PLAY",
    "KIND_RESOURCE_SPEND",
]
