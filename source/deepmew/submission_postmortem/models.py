"""Typed data contracts shared by the submission postmortem pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "deepmew.submission_postmortem.v0"
ALIGNMENT_RULE_VERSION = "deepmew.kaggle.causal_previous_step.module_1_32_0.v0"


@dataclass(frozen=True)
class CatalogAgent:
    """Submission metadata for one seat in a Kaggle episode."""

    index: int
    submission_id: str | None = None
    team_id: str | None = None
    team_name: str | None = None
    reward: float | None = None
    state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "submission_id": self.submission_id,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "reward": self.reward,
            "state": self.state,
        }


@dataclass(frozen=True)
class CatalogEpisode:
    """One episode returned by Kaggle's submission episode catalog."""

    episode_id: str
    create_time: str | None = None
    end_time: str | None = None
    state: str | None = None
    is_validation: bool = False
    agents: tuple[CatalogAgent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "create_time": self.create_time,
            "end_time": self.end_time,
            "state": self.state,
            "is_validation": self.is_validation,
            "agents": [agent.to_dict() for agent in self.agents],
        }


@dataclass(frozen=True)
class CatalogQuery:
    """Composable catalog filters; families combine with AND, values with OR."""

    episode_ids: frozenset[str] = frozenset()
    episode_min: int | None = None
    episode_max: int | None = None
    ordinal_min: int | None = None
    ordinal_max: int | None = None
    opponent_terms: tuple[str, ...] = ()
    opponent_team_ids: frozenset[str] = frozenset()
    opponent_submission_ids: frozenset[str] = frozenset()
    include_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_ids": sorted(self.episode_ids),
            "episode_min": self.episode_min,
            "episode_max": self.episode_max,
            "ordinal_min": self.ordinal_min,
            "ordinal_max": self.ordinal_max,
            "opponent_terms": list(self.opponent_terms),
            "opponent_team_ids": sorted(self.opponent_team_ids),
            "opponent_submission_ids": sorted(self.opponent_submission_ids),
            "include_validation": self.include_validation,
        }


@dataclass(frozen=True)
class ReplayDeck:
    """Postgame deck metadata; never used as a policy-rescore feature."""

    seat: int
    team_name: str
    stable_hash: str
    card_ids: tuple[int, ...]
    card_counts: Mapping[int, int]
    name_counts: Mapping[str, int]

    def to_dict(self, *, include_cards: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seat": self.seat,
            "team_name": self.team_name,
            "stable_hash": self.stable_hash,
            "card_counts": {str(key): value for key, value in sorted(self.card_counts.items())},
            "name_counts": dict(sorted(self.name_counts.items())),
            "provenance": "posthoc_deck_metadata",
        }
        if include_cards:
            result["card_ids"] = list(self.card_ids)

        return result


@dataclass(frozen=True)
class ReplayDecision:
    """One causally aligned action request and its selected indices."""

    decision_id: str
    episode_id: str
    request_step: int
    action_step: int
    seat: int
    action: tuple[int, ...]
    observation: Mapping[str, Any] = field(repr=False)
    observation_sha256: str = ""
    result_status: str | None = None
    result_reward: float | None = None
    select_type: str | int | None = None
    select_context: str | int | None = None
    min_count: int = 0
    max_count: int = 0
    legal_count: int = 0
    information_mode: str = "actor_visible"

    def to_dict(self, *, include_observation: bool = False) -> dict[str, Any]:
        result = {
            "decision_id": self.decision_id,
            "episode_id": self.episode_id,
            "request_step": self.request_step,
            "action_step": self.action_step,
            "seat": self.seat,
            "action": list(self.action),
            "observation_sha256": self.observation_sha256,
            "result_status": self.result_status,
            "result_reward": self.result_reward,
            "select_type": self.select_type,
            "select_context": self.select_context,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "legal_count": self.legal_count,
            "information_mode": self.information_mode,
            "provenance": "actor_visible",
        }
        if include_observation:
            result["observation"] = dict(self.observation)

        return result


@dataclass(frozen=True)
class ReplayEpisode:
    """Safe parsed replay and the decisions admitted by its alignment rule."""

    source_path: str
    source_sha256: str
    replay_id: str
    episode_id: str
    module_version: str
    team_names: tuple[str, ...]
    rewards: tuple[float | None, ...]
    statuses: tuple[str | None, ...]
    decks: tuple[ReplayDeck, ...]
    actor_seats: tuple[int, ...]
    seat_binding_status: str
    self_mirror: bool
    alignment_status: str
    decisions: tuple[ReplayDecision, ...]
    errors: tuple[str, ...] = ()

    def decisions_for_actor(self) -> tuple[ReplayDecision, ...]:
        actor_seats = set(self.actor_seats)

        return tuple(decision for decision in self.decisions if decision.seat in actor_seats)

    def to_dict(self, *, include_decisions: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "replay_id": self.replay_id,
            "episode_id": self.episode_id,
            "module_version": self.module_version,
            "team_names": list(self.team_names),
            "rewards": list(self.rewards),
            "statuses": list(self.statuses),
            "decks": [deck.to_dict() for deck in self.decks],
            "actor_seats": list(self.actor_seats),
            "seat_binding_status": self.seat_binding_status,
            "self_mirror": self.self_mirror,
            "alignment_status": self.alignment_status,
            "decision_count": len(self.decisions),
            "actor_decision_count": len(self.decisions_for_actor()),
            "errors": list(self.errors),
        }
        if include_decisions:
            result["decisions"] = [decision.to_dict() for decision in self.decisions]

        return result


def finite_float(value: Any) -> float | None:
    """Return a JSON-safe finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None

    return number


def integer_sequence(value: Any) -> tuple[int, ...] | None:
    """Return an exact integer sequence without silently coercing booleans."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        result.append(item)

    return tuple(result)
