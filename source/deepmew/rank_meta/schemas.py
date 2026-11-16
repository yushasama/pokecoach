"""K1.7 rank-metadata vocabulary: leaderboard buckets, weights, and join records.

Rank here is a team's LEADERBOARD STANDING at scout/scrape time (1-300), recovered
from the scout layer -- NOT a per-game ELO/MMR and NOT a per-decision rating. Two
distinct weight schemes live here on purpose:

- ``RANK_WEIGHT`` scales rank-weighted *meta aggregates* (how much a ranked pilot's
  deck counts toward the rank-weighted archetype share);
- ``TEACHER_RANK_FACTOR`` scales K5-lite *teacher reward weight* (how much a ranked
  pilot's decision is trusted as a teacher).

Both fall back to a neutral value for rank-missing rows so partial coverage (~45%)
never silently discards data. Pure stdlib -- no engine, no replay I/O, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARTIFACT_VERSION = "deepmew.rank_metadata.v1"
RANK_META_KIND = "rank_metadata"

RANK_MISSING = "rank_missing"

# Ordered (inclusive_upper_rank, bucket) thresholds. Ranks above the last bound
# fall into the final bucket (the mined corpus tops out at rank 300).
RANK_BUCKET_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (10, "top10"),
    (25, "top25"),
    (50, "top50"),
    (100, "top100"),
    (300, "top300"),
)
RANK_BUCKET_ORDER: tuple[str, ...] = ("top10", "top25", "top50", "top100", "top300", RANK_MISSING)

# Meta-aggregate weight: rank-weighted artifact share for archetype/deck summaries.
RANK_WEIGHT: dict[str, float] = {
    "top10": 5.0, "top25": 3.0, "top50": 2.0, "top100": 1.5, "top300": 1.0, RANK_MISSING: 0.0,
}

# K5-lite teacher reward factor: how much a ranked pilot's decision is trusted.
# Neutral (1.0) when rank is missing so partial coverage never drops rows.
TEACHER_RANK_FACTOR: dict[str, float] = {
    "top10": 1.50, "top25": 1.30, "top50": 1.15, "top100": 1.05, "top300": 1.00, RANK_MISSING: 1.00,
}


def rank_bucket(rank: int | None) -> str:
    """Map a leaderboard rank to its coarse bucket (``rank_missing`` if None)."""
    if rank is None:
        return RANK_MISSING
    for upper, bucket in RANK_BUCKET_THRESHOLDS:
        if rank <= upper:
            return bucket
    return "top300"


def leaderboard_rank_weight(rank: int | None) -> float:
    """Meta-aggregate weight for a rank (0.0 when missing)."""
    return RANK_WEIGHT[rank_bucket(rank)]


def teacher_rank_factor(bucket: str) -> float:
    """K5-lite teacher reward factor for a bucket (1.0 when missing/unknown)."""
    return TEACHER_RANK_FACTOR.get(bucket, 1.0)


@dataclass(frozen=True)
class TeamRank:
    """A scout-resolved leaderboard standing for one team name."""

    team_name: str
    rank: int | None
    team_id: str | None
    score: str | None
    source_range: str | None
    scout_source_manifest: str
    ambiguous: bool = False

    @property
    def bucket(self) -> str:
        return rank_bucket(self.rank)


@dataclass(frozen=True)
class RankInfo:
    """The rank view attached to one deck artifact / pilot seat."""

    rank: int | None = None
    bucket: str = RANK_MISSING
    leaderboard_weight: float = 0.0
    teacher_factor: float = 1.0
    team_id: str | None = None
    score: str | None = None
    missing: bool = True
    ambiguous: bool = False

    @classmethod
    def from_team_rank(cls, team_rank: "TeamRank | None") -> "RankInfo":
        if team_rank is None or team_rank.rank is None:
            ambiguous = bool(team_rank.ambiguous) if team_rank else False
            return cls(missing=True, ambiguous=ambiguous,
                       team_id=team_rank.team_id if team_rank else None)
        bucket = team_rank.bucket
        return cls(
            rank=team_rank.rank,
            bucket=bucket,
            leaderboard_weight=RANK_WEIGHT[bucket],
            teacher_factor=TEACHER_RANK_FACTOR[bucket],
            team_id=team_rank.team_id,
            score=team_rank.score,
            missing=False,
            ambiguous=team_rank.ambiguous,
        )


MISSING_RANK_INFO = RankInfo()


@dataclass
class RankMetaWarning:
    code: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)

    def bump(self, example: str | None = None) -> None:
        self.count += 1
        if example is not None and len(self.examples) < 10 and example not in self.examples:
            self.examples.append(example)
