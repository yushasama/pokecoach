"""Pure contracts for current external-rating replay overlays.

External Kaggle ratings are labels and sampling context only. They never share
scale, thresholds, or update equations with internal panel Bradley-Terry values.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from deepmew.kaggle_replays.scout import LeaderboardRow

BUCKET_LABELS_DESCENDING = (
    "R0_CURRENT_FRONTIER",
    "R1_CURRENT_STRONG",
    "R2_CURRENT_INTERMEDIATE",
    "R3_CURRENT_LOWER",
)
MIN_PLAYERS_PER_BUCKET = 20
MIN_DECISIONS_PER_BUCKET = 1_000
DEFAULT_THRESHOLDS_ASCENDING = (800.0, 900.0, 1_000.0)
RATING_SCALE_BOUNDARY = (
    "EXTERNAL_KAGGLE_CURRENT_RATING_ONLY; MUST_NOT_BE_COMPARED_TO_OR_COMBINED_WITH_"
    "INTERNAL_PANEL_BRADLEY_TERRY_VALUES"
)


class CurrentRatingError(ValueError):
    """Raised when a current-rating artifact cannot be made fail-closed."""


@dataclass(frozen=True)
class ValidatedRatingRow:
    rank: int
    team_id: str
    team_name: str
    score: float
    submission_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "score": self.score,
            "submission_date": self.submission_date,
        }


@dataclass(frozen=True)
class ReplayIdentityEvidence:
    source_game_id: str
    actor_seat: int
    replay_team_name: str
    candidate_team_ids: tuple[str, ...]
    decision_count: int
    replay_timestamp_utc: str | None = None


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def normalize_identity_name(value: str) -> str:
    return " ".join(str(value).casefold().split())


def validate_leaderboard(rows: Sequence[LeaderboardRow]) -> list[ValidatedRatingRow]:
    if not rows:
        raise CurrentRatingError("leaderboard is empty")

    validated: list[ValidatedRatingRow] = []
    seen_team_ids: set[str] = set()
    previous_score = math.inf
    for expected_rank, row in enumerate(rows, start=1):
        team_id = str(row.team_id).strip()
        team_name = str(row.team_name).strip()
        if not team_id.isdigit() or int(team_id) <= 0:
            raise CurrentRatingError(f"rank {expected_rank}: team_id must be a positive integer")
        if team_id in seen_team_ids:
            raise CurrentRatingError(f"duplicate team_id {team_id}")
        if row.rank != expected_rank or row.row_index != expected_rank:
            raise CurrentRatingError(
                f"leaderboard rank must be contiguous: expected {expected_rank}, got rank={row.rank} row={row.row_index}"
            )
        if not team_name:
            raise CurrentRatingError(f"team_id {team_id}: empty team_name")
        try:
            score = float(str(row.score).strip())
        except (TypeError, ValueError) as exc:
            raise CurrentRatingError(f"team_id {team_id}: score is not numeric: {row.score!r}") from exc
        if not math.isfinite(score):
            raise CurrentRatingError(f"team_id {team_id}: score is not finite")
        submission_date = str(row.submission_date or "").strip()
        if not submission_date:
            raise CurrentRatingError(f"team_id {team_id}: missing submission_date")
        try:
            datetime.fromisoformat(submission_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CurrentRatingError(f"team_id {team_id}: invalid submission_date {submission_date!r}") from exc
        if score > previous_score:
            raise CurrentRatingError(
                f"leaderboard score order is not non-increasing at rank {expected_rank}: {score} > {previous_score}"
            )
        seen_team_ids.add(team_id)
        previous_score = score
        validated.append(
            ValidatedRatingRow(
                rank=expected_rank,
                team_id=team_id,
                team_name=team_name,
                score=score,
                submission_date=submission_date,
            )
        )

    return validated


def resolve_replay_identity(
    evidence: ReplayIdentityEvidence,
    aliases_by_team_id: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    replay_name = normalize_identity_name(evidence.replay_team_name)
    matches = []
    for team_id in evidence.candidate_team_ids:
        aliases = {normalize_identity_name(alias) for alias in aliases_by_team_id.get(team_id, ())}
        if replay_name and replay_name in aliases:
            matches.append(team_id)
    unique_matches = sorted(set(matches), key=lambda value: int(value))
    if len(unique_matches) == 1:
        status = "RESOLVED_DIRECT_EPISODE_TEAM_ID_AND_EXACT_SEAT_NAME"
        team_id = unique_matches[0]
        confidence = 1.0
    elif unique_matches:
        status = "AMBIGUOUS_MULTIPLE_DIRECT_TEAM_IDS_MATCH_SEAT_NAME"
        team_id = None
        confidence = 0.0
    else:
        status = "UNRESOLVED_NO_DIRECT_TEAM_ID_MATCHING_SEAT_NAME"
        team_id = None
        confidence = 0.0

    return {
        "source_game_id": evidence.source_game_id,
        "actor_seat": evidence.actor_seat,
        "replay_team_name": evidence.replay_team_name,
        "candidate_team_ids": list(evidence.candidate_team_ids),
        "decision_count": evidence.decision_count,
        "replay_timestamp_utc": evidence.replay_timestamp_utc,
        "status": status,
        "current_rating_classification": (
            "CURRENT_RATING_RESOLVED" if team_id is not None else "CURRENT_RATING_UNRESOLVED"
        ),
        "team_id": team_id,
        "mapping_confidence": confidence,
    }


def summarize_rating_history(observations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_timestamp: dict[str, float] = {}
    for observation in observations:
        timestamp = str(observation.get("snapshot_timestamp_utc") or "").strip()
        try:
            score = float(observation["score"])
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(score):
            by_timestamp[timestamp] = score
    ordered = sorted(by_timestamp.items(), key=lambda item: item[0])
    if not ordered:
        return {
            "observation_count": 0,
            "historical_peak": None,
            "recent_trend": {"status": "NOT_INSTRUMENTED"},
            "volatility": {"status": "NOT_INSTRUMENTED"},
        }

    peak_timestamp, peak_score = max(ordered, key=lambda item: item[1])
    if len(ordered) >= 2:
        first_timestamp, first_score = ordered[-2]
        last_timestamp, last_score = ordered[-1]
        elapsed_days = max(
            (datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
             - datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))).total_seconds() / 86_400.0,
            1e-9,
        )
        trend = {
            "status": "MEASURED",
            "score_delta": last_score - first_score,
            "score_delta_per_day": (last_score - first_score) / elapsed_days,
        }
    else:
        trend = {"status": "INSUFFICIENT_SNAPSHOTS"}
    score_deltas = [ordered[index][1] - ordered[index - 1][1] for index in range(1, len(ordered))]
    if len(score_deltas) >= 2:
        volatility = {
            "status": "MEASURED",
            "score_delta_population_stddev": statistics.pstdev(score_deltas),
        }
    else:
        volatility = {"status": "INSUFFICIENT_SNAPSHOTS"}

    return {
        "observation_count": len(ordered),
        "historical_peak": {"score": peak_score, "snapshot_timestamp_utc": peak_timestamp},
        "recent_trend": trend,
        "volatility": volatility,
    }


def assign_bucket(score: float, thresholds_ascending: Sequence[float]) -> str:
    if len(thresholds_ascending) != 3:
        raise CurrentRatingError("exactly three thresholds are required for R0..R3")
    low, middle, high = thresholds_ascending
    if score < low:
        return "R3_CURRENT_LOWER"
    if score < middle:
        return "R2_CURRENT_INTERMEDIATE"
    if score < high:
        return "R1_CURRENT_STRONG"

    return "R0_CURRENT_FRONTIER"


def rating_distribution_summary(rows: Sequence[Mapping[str, Any]], score_field: str) -> dict[str, Any]:
    scores = sorted(float(row[score_field]) for row in rows)
    if not scores or any(not math.isfinite(score) for score in scores):
        raise CurrentRatingError("rating distribution requires at least one finite score")

    def quantile(fraction: float) -> float:
        position = fraction * (len(scores) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return scores[lower]
        weight = position - lower

        return scores[lower] * (1.0 - weight) + scores[upper] * weight

    return {
        "count": len(scores),
        "minimum": scores[0],
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "maximum": scores[-1],
    }


def freeze_rating_buckets(
    players: Sequence[Mapping[str, Any]],
    full_ladder: Sequence[Mapping[str, Any]],
    *,
    thresholds_ascending: Sequence[float] = DEFAULT_THRESHOLDS_ASCENDING,
    min_players_per_bucket: int = MIN_PLAYERS_PER_BUCKET,
    min_decisions_per_bucket: int = MIN_DECISIONS_PER_BUCKET,
) -> dict[str, Any]:
    if min_players_per_bucket <= 0 or min_decisions_per_bucket <= 0:
        raise CurrentRatingError("minimum bucket supports must be positive")
    thresholds = [float(value) for value in thresholds_ascending]
    if len(thresholds) != 3 or thresholds != sorted(set(thresholds)):
        raise CurrentRatingError("rating thresholds must contain exactly three increasing values")
    if not full_ladder:
        raise CurrentRatingError("full current ladder is required before freezing external bands")
    full_ladder_support = {label: {"players": 0} for label in BUCKET_LABELS_DESCENDING}
    for row in full_ladder:
        label = assign_bucket(float(row["score"]), thresholds)
        full_ladder_support[label]["players"] += 1
    linked_support = {
        label: {"players": 0, "decisions": 0}
        for label in BUCKET_LABELS_DESCENDING
    }
    for player in players:
        label = assign_bucket(float(player["current_external_rating"]), thresholds)
        linked_support[label]["players"] += 1
        linked_support[label]["decisions"] += int(player["decision_count"])
    failures = {
        label: values
        for label, values in linked_support.items()
        if values["players"] < min_players_per_bucket or values["decisions"] < min_decisions_per_bucket
    }
    if failures:
        raise CurrentRatingError(f"prospective bucket support failed: {failures}")

    return {
        "schema_version": "deepmew.current_external_rating_buckets.v1",
        "status": "FROZEN_PROSPECTIVELY_BEFORE_MODEL_RESULTS",
        "rating_scale_boundary": RATING_SCALE_BOUNDARY,
        "algorithm": "fixed interpretable 100-point external-rating bands inspected on the full current ladder",
        "threshold_rationale": (
            "800/900/1000 are round, stable, interpretable external-score boundaries; they are accepted only "
            "after recording the full current ladder distribution and proving minimum replay-linked player and "
            "decision support in every band"
        ),
        "thresholds_ascending": thresholds,
        "bucket_order_high_to_low": list(BUCKET_LABELS_DESCENDING),
        "full_current_ladder_distribution": rating_distribution_summary(full_ladder, "score"),
        "replay_linked_rating_distribution": rating_distribution_summary(players, "current_external_rating"),
        "minimum_support": {
            "players_per_bucket": min_players_per_bucket,
            "decisions_per_bucket": min_decisions_per_bucket,
        },
        "full_current_ladder_support": full_ladder_support,
        "replay_linked_support": linked_support,
    }


def behavior_weighting_spec() -> dict[str, Any]:
    return {
        "schema_version": "deepmew.current_behavior_weighting_spec.v1",
        "status": "FROZEN_PREREQUISITE_NOT_YET_APPLIED_TO_TRAINING",
        "formula": (
            "weight = recency_weight * rating_stability_weight * deck_relevance_weight * "
            "playstyle_continuity_weight * source_reliability_weight"
        ),
        "components": {
            "recency_weight": {"formula": "max(0.10, 2 ** (-age_days / 45.0))", "half_life_days": 45.0},
            "rating_stability_weight": {
                "formula": "1 / (1 + score_delta_stddev / 100)",
                "insufficient_snapshot_value": 0.75,
            },
            "deck_relevance_weight": {
                "formula": "clip(recent_same_deck_family_share, 0.25, 1.0)",
                "cold_start_value": 0.75,
                "measurement": "causal share over the frozen recent replay window",
            },
            "playstyle_continuity_weight": {
                "formula": "clip(1 - normalized_action_family_js_divergence, 0.25, 1.0)",
                "cold_start_value": 0.75,
                "measurement": "causal action-family distribution similarity between adjacent frozen windows",
            },
            "source_reliability_weight": {
                "direct_episode_team_id_and_exact_seat_name": 1.0,
                "unresolved_or_ambiguous": 0.0,
            },
        },
        "guards": [
            "all component evidence must be causal at the decision timestamp",
            "unresolved or ambiguous identities are excluded, never guessed",
            "external ratings remain separate from internal panel Bradley-Terry values",
            "report per-bucket effective sample size before any training authorization",
        ],
    }


def compare_overlays(previous: Mapping[str, Any] | None, current_players: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current = {str(player["team_id"]): player for player in current_players}
    previous_players = (previous or {}).get("players", [])
    old = {str(player["team_id"]): player for player in previous_players}
    shared = sorted(set(old) & set(current), key=int)
    migrations = []
    for team_id in shared:
        old_bucket = old[team_id].get("current_bucket")
        new_bucket = current[team_id].get("current_bucket")
        if old_bucket != new_bucket:
            migrations.append({"team_id": team_id, "from": old_bucket, "to": new_bucket})

    return {
        "added_team_ids": sorted(set(current) - set(old), key=int),
        "removed_team_ids": sorted(set(old) - set(current), key=int),
        "bucket_migrations": migrations,
        "shared_player_count": len(shared),
        "bucket_migration_share": len(migrations) / len(shared) if shared else None,
        "required_response": (
            "REPORT_ONLY; mark affected behavior policies and exploiters for re-evaluation; "
            "do not retrain automatically"
        ),
    }
