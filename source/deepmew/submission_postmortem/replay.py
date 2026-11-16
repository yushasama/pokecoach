"""Fail-closed CABT 1.32.0 replay extraction for submission postmortems.

Kaggle stores the action returned for ``steps[t - 1][seat].observation`` on
``steps[t][seat].action``.  The observation on step ``t`` is the resulting
state.  This module owns that narrow convention for CABT module 1.32.0 only;
other source versions are rejected instead of guessed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from deepmew.kaggle_replays.decks import DeckExtractionError, extract_decklists
from deepmew.meta_kg.io import stable_deck_hash
from deepmew.replay.information_boundary import (
    InformationMode,
    assert_deployable_observation,
    classify_observation_visibility,
)
from deepmew.submission_postmortem.models import (
    CatalogEpisode,
    ReplayDecision,
    ReplayDeck,
    ReplayEpisode,
    integer_sequence,
)


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_ENVIRONMENT_NAME = "cabt"
SUPPORTED_ENVIRONMENT_VERSION = "1.0.0"
SUPPORTED_MODULE_VERSION = "1.32.0"
SETUP_DECK_CARD_COUNT = 60
FORBIDDEN_PROJECTED_FIELDS = frozenset({"logs", "search_begin_input"})
_EPISODE_FILENAME = re.compile(r"episode-(?P<episode_id>\d+)-replay\.json$")


class ReplayExtractionError(ValueError):
    """Raised when replay-level provenance or source scope is not trustworthy."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if normalized != normalized or normalized in (float("inf"), float("-inf")):
        return None

    return normalized


def project_actor_observation(source_observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the local certified actor-visible projection used for rescoring.

    Raw CABT 1.32.0 observations contain ``search_begin_input`` and therefore
    classify as privileged.  Logs and other opaque/private surfaces are also
    excluded before the tracked information-boundary validator is invoked.
    """

    if not isinstance(source_observation, Mapping):
        raise ReplayExtractionError("source observation must be an object")
    projected = copy.deepcopy(dict(source_observation))
    for field in FORBIDDEN_PROJECTED_FIELDS:
        projected.pop(field, None)
    current = projected.get("current")
    if not isinstance(current, dict):
        raise ReplayExtractionError("source observation current must be an object")
    current.pop("looking", None)
    select = projected.get("select")
    if not isinstance(select, dict):
        raise ReplayExtractionError("source observation select must be an object")
    select.pop("deck", None)
    try:
        assert_deployable_observation(
            projected,
            expected_mode=InformationMode.ACTOR_VISIBLE.value,
        )
    except ValueError as exc:
        raise ReplayExtractionError(
            f"actor observation failed the information boundary: {exc}"
        ) from exc
    if classify_observation_visibility(projected) != InformationMode.ACTOR_VISIBLE.value:
        raise ReplayExtractionError("actor observation was not exactly actor_visible")

    return projected


def stable_multiset_hash(card_ids: Sequence[int]) -> str:
    """Return the repository's order-independent deck multiset identity."""

    normalized = integer_sequence(card_ids)
    if normalized is None or not normalized:
        raise ReplayExtractionError("submitted deck must be a non-empty integer sequence")

    return stable_deck_hash(dict(Counter(normalized)))


def _metadata_value(data: Mapping[str, Any], key: str) -> Any:
    return data.get(key)


def _validate_source_scope(data: Mapping[str, Any]) -> str:
    observed = {
        "schema_version": _strict_integer(_metadata_value(data, "schema_version")),
        "name": _metadata_value(data, "name"),
        "version": _metadata_value(data, "version"),
        "module_version": _metadata_value(data, "module_version"),
    }
    expected = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "name": SUPPORTED_ENVIRONMENT_NAME,
        "version": SUPPORTED_ENVIRONMENT_VERSION,
        "module_version": SUPPORTED_MODULE_VERSION,
    }
    if observed != expected:
        raise ReplayExtractionError(
            "unsupported replay source scope: "
            f"observed={json.dumps(observed, sort_keys=True)} "
            f"expected={json.dumps(expected, sort_keys=True)}"
        )

    return SUPPORTED_MODULE_VERSION


def _path_episode_id(path: Path) -> str | None:
    match = _EPISODE_FILENAME.search(path.name)

    return match.group("episode_id") if match is not None else None


def _episode_id(
    data: Mapping[str, Any],
    path: Path,
    catalog_episode: CatalogEpisode | None,
) -> str:
    info = data.get("info")
    raw_value = info.get("EpisodeId") if isinstance(info, Mapping) else None
    raw_episode_id = str(raw_value) if raw_value is not None else _path_episode_id(path)
    if catalog_episode is not None:
        catalog_id = str(catalog_episode.episode_id)
        if raw_episode_id is not None and raw_episode_id != catalog_id:
            raise ReplayExtractionError(
                f"catalog episode {catalog_id} conflicts with replay episode {raw_episode_id}"
            )

        return catalog_id
    if raw_episode_id:
        return raw_episode_id

    raise ReplayExtractionError("numeric Kaggle episode id is unavailable")


def _team_names(data: Mapping[str, Any]) -> tuple[str, ...]:
    info = data.get("info")
    values = info.get("TeamNames") if isinstance(info, Mapping) else None
    if not isinstance(values, list) or len(values) != 2 or any(
        not isinstance(value, str) for value in values
    ):
        raise ReplayExtractionError("replay TeamNames must contain exactly two strings")

    return tuple(values)


def _reward_values(data: Mapping[str, Any]) -> tuple[float | None, ...]:
    values = data.get("rewards")
    if not isinstance(values, list) or len(values) != 2:
        return (None, None)

    return tuple(_finite_float(value) for value in values)


def _status_values(data: Mapping[str, Any]) -> tuple[str | None, ...]:
    values = data.get("statuses")
    if not isinstance(values, list) or len(values) != 2:
        return (None, None)

    return tuple(value if isinstance(value, str) else None for value in values)


def _replay_decks(data: dict[str, Any]) -> tuple[tuple[ReplayDeck, ...], tuple[str, ...]]:
    try:
        extracted = extract_decklists(data)
    except DeckExtractionError as exc:
        return (), (f"deck_extraction_failed:{exc}",)
    decks = []
    for deck in extracted:
        card_ids = tuple(sorted(deck.card_ids))
        decks.append(ReplayDeck(
            seat=int(deck.player_index),
            team_name=str(deck.team_name),
            stable_hash=stable_multiset_hash(card_ids),
            card_ids=card_ids,
            card_counts=dict(deck.card_counts),
            name_counts=dict(deck.name_counts),
        ))

    return tuple(sorted(decks, key=lambda deck: deck.seat)), ()


def _select_request(agent: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    if agent.get("status") != "ACTIVE":
        return None
    observation = agent.get("observation")
    if not isinstance(observation, Mapping):
        return None
    select = observation.get("select")
    if not isinstance(select, Mapping) or not isinstance(select.get("option"), list):
        return None

    return observation, select


def _result_field_error(
    current_agent: Mapping[str, Any],
    *,
    episode_id: str,
    action_step: int,
    seat: int,
) -> str | None:
    if not isinstance(current_agent.get("observation"), Mapping):
        return f"{episode_id}:step_{action_step}:s{seat}:result_observation_missing"
    reward = current_agent.get("reward")
    if reward is not None and _finite_float(reward) is None:
        return f"{episode_id}:step_{action_step}:s{seat}:result_reward_invalid"

    return None


def _align_decisions(
    data: Mapping[str, Any],
    *,
    episode_id: str,
) -> tuple[tuple[ReplayDecision, ...], tuple[str, ...]]:
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return (), ("replay_steps_missing_or_too_short",)
    decisions: list[ReplayDecision] = []
    errors: list[str] = []
    for action_step in range(1, len(steps)):
        prior_step = steps[action_step - 1]
        current_step = steps[action_step]
        if (
            not isinstance(prior_step, list)
            or not isinstance(current_step, list)
            or len(prior_step) != 2
            or len(current_step) != 2
        ):
            errors.append(f"{episode_id}:step_{action_step}:agent_step_shape_invalid")
            continue
        for seat in (0, 1):
            prior_agent = prior_step[seat]
            current_agent = current_step[seat]
            if not isinstance(prior_agent, Mapping) or not isinstance(current_agent, Mapping):
                errors.append(f"{episode_id}:step_{action_step}:s{seat}:agent_shape_invalid")
                continue
            action = integer_sequence(current_agent.get("action"))
            request = _select_request(prior_agent)
            if request is None:
                prior_observation = prior_agent.get("observation")
                prior_select = (
                    prior_observation.get("select")
                    if isinstance(prior_observation, Mapping)
                    else None
                )
                if (
                    action is not None
                    and len(action) == SETUP_DECK_CARD_COUNT
                    and prior_select is None
                ):
                    continue
                if prior_agent.get("status") == "ACTIVE" and isinstance(
                    prior_select,
                    Mapping,
                ):
                    errors.append(
                        f"{episode_id}:step_{action_step}:s{seat}:prior_request_invalid"
                    )
                    continue
                if action:
                    errors.append(f"{episode_id}:step_{action_step}:s{seat}:prior_request_missing")
                continue
            observation, select = request
            if action is None:
                errors.append(f"{episode_id}:step_{action_step}:s{seat}:action_missing_or_invalid")
                continue
            options = select.get("option")
            minimum = _strict_integer(select.get("minCount"))
            maximum = _strict_integer(select.get("maxCount"))
            if (
                not isinstance(options, list)
                or any(not isinstance(option, Mapping) for option in options)
                or minimum is None
                or maximum is None
                or minimum < 0
                or maximum < minimum
            ):
                errors.append(f"{episode_id}:step_{action_step}:s{seat}:select_contract_invalid")
                continue
            if not minimum <= len(action) <= maximum:
                errors.append(f"{episode_id}:step_{action_step}:s{seat}:choice_bounds_invalid")
                continue
            if any(index < 0 or index >= len(options) for index in action):
                errors.append(f"{episode_id}:step_{action_step}:s{seat}:choice_index_invalid")
                continue
            result_error = _result_field_error(
                current_agent,
                episode_id=episode_id,
                action_step=action_step,
                seat=seat,
            )
            if result_error is not None:
                errors.append(result_error)
                continue
            try:
                projected = project_actor_observation(observation)
            except ReplayExtractionError as exc:
                errors.append(
                    f"{episode_id}:step_{action_step}:s{seat}:projection_invalid:{exc}"
                )
                continue
            current = projected.get("current")
            actor = _strict_integer(current.get("yourIndex")) if isinstance(current, Mapping) else None
            if actor != seat:
                errors.append(f"{episode_id}:step_{action_step}:s{seat}:seat_identity_invalid")
                continue
            status = current_agent.get("status")
            decisions.append(ReplayDecision(
                decision_id=f"{episode_id}:step_{action_step:06d}:s{seat}",
                episode_id=episode_id,
                request_step=action_step - 1,
                action_step=action_step,
                seat=seat,
                action=action,
                observation=projected,
                observation_sha256=_canonical_json_sha256(projected),
                result_status=status if isinstance(status, str) else None,
                result_reward=_finite_float(current_agent.get("reward")),
                select_type=select.get("type"),
                select_context=select.get("context"),
                min_count=minimum,
                max_count=maximum,
                legal_count=len(options),
            ))
    if errors:
        return (), tuple(errors)
    if not decisions:
        return (), ("no_causally_aligned_decisions",)

    return tuple(decisions), ()


def _resolve_submitted_hash(
    submitted_deck_hash: str | None,
    submitted_card_ids: Sequence[int] | None,
) -> str | None:
    derived = (
        stable_multiset_hash(submitted_card_ids)
        if submitted_card_ids is not None
        else None
    )
    if submitted_deck_hash is not None and derived is not None and submitted_deck_hash != derived:
        raise ReplayExtractionError("submitted deck hash conflicts with submitted card ids")

    return submitted_deck_hash or derived


def _bind_actor_seats(
    decks: Sequence[ReplayDeck],
    *,
    submission_id: str | None,
    catalog_episode: CatalogEpisode | None,
    submitted_deck_hash: str | None,
) -> tuple[tuple[int, ...], str, bool]:
    if catalog_episode is not None and submission_id is not None:
        normalized_submission_id = str(submission_id)
        catalog_matches = sorted({
            int(agent.index)
            for agent in catalog_episode.agents
            if agent.index in (0, 1)
            and agent.submission_id is not None
            and str(agent.submission_id) == normalized_submission_id
        })
        if catalog_matches:
            seats = tuple(catalog_matches)
            self_mirror = seats == (0, 1)

            return (
                seats,
                "catalog_submission_id_exact_self_mirror"
                if self_mirror
                else "catalog_submission_id_exact",
                self_mirror,
            )
    if submitted_deck_hash is not None:
        seats = tuple(sorted(
            deck.seat for deck in decks if deck.stable_hash == submitted_deck_hash
        ))
        if seats:
            self_mirror = seats == (0, 1)

            return (
                seats,
                "submitted_deck_hash_exact_self_mirror"
                if self_mirror
                else "submitted_deck_hash_exact",
                self_mirror,
            )

    return (), "unresolved", False


def load_replay(
    path: str | Path,
    *,
    submission_id: str | None = None,
    catalog_episode: CatalogEpisode | None = None,
    submitted_deck_hash: str | None = None,
    submitted_card_ids: Sequence[int] | None = None,
) -> ReplayEpisode:
    """Load, hash-bind, align, project, and seat-bind one CABT 1.32.0 replay."""

    source_path = Path(path).resolve()
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise ReplayExtractionError(f"unable to read replay: {source_path}") from exc
    source_sha256 = _file_sha256(raw)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayExtractionError(f"invalid replay JSON: {source_path}") from exc
    if not isinstance(data, dict):
        raise ReplayExtractionError("replay root must be an object")
    module_version = _validate_source_scope(data)
    episode_id = _episode_id(data, source_path, catalog_episode)
    replay_id = data.get("id")
    if not isinstance(replay_id, str) or not replay_id:
        raise ReplayExtractionError("replay id is missing")
    decks, deck_errors = _replay_decks(data)
    decisions, alignment_errors = _align_decisions(data, episode_id=episode_id)
    resolved_hash = _resolve_submitted_hash(submitted_deck_hash, submitted_card_ids)
    actor_seats, seat_status, self_mirror = _bind_actor_seats(
        decks,
        submission_id=submission_id,
        catalog_episode=catalog_episode,
        submitted_deck_hash=resolved_hash,
    )

    return ReplayEpisode(
        source_path=str(source_path),
        source_sha256=source_sha256,
        replay_id=replay_id,
        episode_id=episode_id,
        module_version=module_version,
        team_names=_team_names(data),
        rewards=_reward_values(data),
        statuses=_status_values(data),
        decks=decks,
        actor_seats=actor_seats,
        seat_binding_status=seat_status,
        self_mirror=self_mirror,
        alignment_status=(
            "structurally_verified_unadmitted"
            if not alignment_errors
            else "rejected"
        ),
        decisions=decisions,
        errors=(*deck_errors, *alignment_errors),
    )


def extract_replay(
    path: str | Path,
    **kwargs: Any,
) -> ReplayEpisode:
    """Compatibility-named wrapper around :func:`load_replay`."""

    return load_replay(path, **kwargs)


__all__ = [
    "ReplayExtractionError",
    "SETUP_DECK_CARD_COUNT",
    "SUPPORTED_ENVIRONMENT_NAME",
    "SUPPORTED_ENVIRONMENT_VERSION",
    "SUPPORTED_MODULE_VERSION",
    "SUPPORTED_SCHEMA_VERSION",
    "extract_replay",
    "load_replay",
    "project_actor_observation",
    "stable_multiset_hash",
]
