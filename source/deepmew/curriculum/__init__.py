"""Curriculum arena (offline/eval-only, INVARIANT #8): a difficulty ladder that
composes the four "harder opponent" levers -- deck bands, pilot tiers, structural
counters, and PFSP league growth -- and auto-advances as the candidate improves.

Never imported by `main.py`/`cg`. See schemas.py for the INVARIANT #2 contract.
"""

from deepmew.curriculum.best_response import (
    BrainRanking,
    ExploiterResult,
    best_response_summary,
    rank_brains,
    select_adversarial,
    top_counters,
)
from deepmew.curriculum.ladder import (
    advance,
    default_curriculum,
    guard_invariant_2,
    is_complete,
)
from deepmew.curriculum.league import League
from deepmew.curriculum.schemas import (
    FROZEN_KINDS,
    PILOT_TIERS,
    CurriculumConfig,
    CurriculumRung,
    LeagueMember,
    RungResult,
    pilot_rank,
)

__all__ = [
    "FROZEN_KINDS",
    "PILOT_TIERS",
    "BrainRanking",
    "CurriculumConfig",
    "CurriculumRung",
    "ExploiterResult",
    "League",
    "LeagueMember",
    "RungResult",
    "advance",
    "best_response_summary",
    "default_curriculum",
    "guard_invariant_2",
    "is_complete",
    "pilot_rank",
    "rank_brains",
    "select_adversarial",
    "top_counters",
]
