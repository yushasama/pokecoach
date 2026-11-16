"""K1.7 rank metadata: materialize scout leaderboard ranks onto mined artifacts.

The scout layer stored a ``team_name -> leaderboard_rank`` table that was never
joined onto deck artifacts (per-deck rank was empty; per-game rating is null). This
package materializes that join into generated rank tables, supplies a
``(episode_id, seat) -> RankInfo`` map for the K4.1 imitation join, and exposes the
bucket/weight/teacher-factor vocabulary. Offline/research-only: nothing here is
imported by the submission runtime, and original deck summaries are never modified.

Rank = leaderboard standing at scout time (1-300), NOT per-game ELO/MMR. Rank-aware
meta is a rank-known scout-subset view (~45% coverage); raw frequency meta stays valid.
"""

from __future__ import annotations

from deepmew.rank_meta.backfill import RankMetaResult, build_rank_metadata
from deepmew.rank_meta.loader import (
    load_episode_seat_rank,
    load_team_rank_map,
    rank_info_for,
)
from deepmew.rank_meta.schemas import (
    RANK_WEIGHT,
    TEACHER_RANK_FACTOR,
    RankInfo,
    TeamRank,
    leaderboard_rank_weight,
    rank_bucket,
    teacher_rank_factor,
)

__all__ = [
    "RANK_WEIGHT",
    "TEACHER_RANK_FACTOR",
    "RankInfo",
    "RankMetaResult",
    "TeamRank",
    "build_rank_metadata",
    "leaderboard_rank_weight",
    "load_episode_seat_rank",
    "load_team_rank_map",
    "rank_bucket",
    "rank_info_for",
    "teacher_rank_factor",
]
