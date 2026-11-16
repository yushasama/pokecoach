"""MatchupThreatInference v0 — offline ``deck profile -> matchup threats`` (stage 3).

Public surface kept small and stable for downstream (Phase 4) consumers. This
package is offline/training-only and must never be imported by ``main.py`` or ``cg``.
"""

from __future__ import annotations

from deepmew.matchup_threat_inference.matchup import analyze_matchup
from deepmew.matchup_threat_inference.schemas import (
    ANSWER_MATRIX,
    COVERAGE_CATEGORIES,
    COVERAGE_GAP_CATEGORIES,
    THREAT_CATEGORIES,
    AnsweredThreat,
    CoverageItem,
    DeckThreatProfile,
    MatchupConfig,
    MatchupPairing,
    ThreatItem,
)
from deepmew.matchup_threat_inference.surfaces import (
    build_coverage_surface,
    build_threat_surface,
    resolve,
)
from deepmew.matchup_threat_inference.inference import (
    build_all_profiles,
    build_pairings,
    build_threat_profile,
    select_opponents,
)

__all__ = [
    "ANSWER_MATRIX",
    "THREAT_CATEGORIES",
    "COVERAGE_CATEGORIES",
    "COVERAGE_GAP_CATEGORIES",
    "ThreatItem",
    "CoverageItem",
    "DeckThreatProfile",
    "AnsweredThreat",
    "MatchupPairing",
    "MatchupConfig",
    "resolve",
    "build_threat_surface",
    "build_coverage_surface",
    "build_threat_profile",
    "build_all_profiles",
    "select_opponents",
    "build_pairings",
    "analyze_matchup",
]
