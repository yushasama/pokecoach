"""Typed, lossless adapters for admitted source evidence.

This module extracts source facts only.  It never decides whether a selected
action was legal or whether two neighboring events should be paired.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from deepmew.replay.contract.identities import identity, sha256_json
from deepmew.replay.contract.schemas import ObservationAssetReference, SourceEvent
from deepmew.replay.contract.kaggle_sources import KAGGLE_REPLAY_SOURCE_SCHEMAS
from deepmew.replay.information_boundary import InformationMode, assert_deployable_observation


SOURCE_OBSERVATION_FIELD_BY_SCHEMA = {
    "deepmew.replay_action_row.v1": {
        "raw_observation": "state_json",
        "state_json": "state_json",
    },
    "deepmew.refgame_decision_row.rebuilt.v1": {"raw_observation": "raw_observation"},
    "deepmew.refgame_decision_row.rebuilt.v2": {"raw_observation": "raw_observation"},
    "deepmew.refgames_factory.decision.v0": {"raw_observation": "raw_observation"},
    "schema1:cabt1.0.0:module1.30.2": {
        "raw_episode_agent_observation": "observation",
    },
    "schema1:cabt1.0.0:module1.32.0": {
        "raw_episode_agent_observation": "observation",
    },
}

VISIBILITY_SCHEMA_VERSION = "deepmew.canonical_actor_feature_observation.v1"
OBSERVATION_ASSET_IDENTITY_VERSION = "deepmew.observation_asset_identity.v1"
SOURCE_EVENT_IDENTITY_VERSION = "deepmew.canonical_source_event_identity.v1"

_PHYSICAL_PATH_FIELDS = frozenset({
    "path",
    "source_path",
    "source_artifact_path",
})
_SOURCE_ROW_COORDINATE_FIELDS = frozenset({
    "binding_id",
    "local_row",
    "global_row",
})
_STORAGE_OWNERSHIP_FIELDS = _PHYSICAL_PATH_FIELDS | _SOURCE_ROW_COORDINATE_FIELDS
_SUPPORTED_INDEX_SPACES = frozenset({"option_position", "source_index"})
_MISSING = object()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _nonnegative_integer(value: Any) -> int | None:
    resolved = _integer(value)
    if resolved is None or resolved < 0:
        return None

    return resolved


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None

    return value


def _exact_typed_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            match_index = next(
                (
                    index
                    for index, (right_key, _) in enumerate(unmatched)
                    if type(left_key) is type(right_key) and left_key == right_key
                ),
                None,
            )
            if match_index is None:
                return False
            _, right_value = unmatched.pop(match_index)
            if not _exact_typed_equal(left_value, right_value):
                return False

        return not unmatched
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_typed_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )

    return bool(left == right)


def _without_storage_ownership(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_storage_ownership(item)
            for key, item in value.items()
            if key not in _STORAGE_OWNERSHIP_FIELDS
        }
    if isinstance(value, list):
        return [_without_storage_ownership(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_storage_ownership(item) for item in value)

    return value


def compute_source_row_sha256(source_record: Mapping[str, Any]) -> str:
    """Hash the exact canonical source-row payload, including retained provenance."""

    if not isinstance(source_record, Mapping):
        raise ValueError("source_record must be an object")

    return sha256_json(source_record)


def _semantic_source_record_sha256(source_record: Mapping[str, Any]) -> str:
    semantic_record = {
        key: value
        for key, value in source_record.items()
        if key not in _SOURCE_ROW_COORDINATE_FIELDS
    }

    return sha256_json(_without_storage_ownership(semantic_record))


def _observation_asset_semantic_payload(
    asset: ObservationAssetReference | None,
) -> Mapping[str, Any] | None:
    if asset is None:
        return None

    return {
        "identity_version": OBSERVATION_ASSET_IDENTITY_VERSION,
        "admission_identity": asset.admission_identity,
        "source_sha256": asset.source_sha256,
        "source_line": asset.source_line,
        "source_content_sha256": asset.source_content_sha256,
        "content_sha256": asset.content_sha256,
        "source_game_id": asset.source_game_id,
        "source_event_position": asset.source_event_position,
        "actor_seat": asset.actor_seat,
        "information_mode": asset.information_mode,
        "observation_schema_version": asset.observation_schema_version,
        "visibility_schema_version": asset.visibility_schema_version,
        "feature_version": asset.feature_version,
    }


def compute_observation_asset_id(asset: ObservationAssetReference) -> str:
    """Return the path- and rematerialization-independent observation identity."""

    if not isinstance(asset, ObservationAssetReference):
        raise ValueError("asset must be an ObservationAssetReference")

    return identity("obs1", _observation_asset_semantic_payload(asset))


def _source_event_observation_asset_payload(
    asset: ObservationAssetReference | None,
) -> Mapping[str, Any] | None:
    semantic_payload = _observation_asset_semantic_payload(asset)
    if semantic_payload is None or asset is None:
        return None

    return {
        **semantic_payload,
        "observation_id": asset.observation_id,
    }


def _source_event_identity_payload(event: SourceEvent) -> Mapping[str, Any]:
    return {
        "identity_version": SOURCE_EVENT_IDENTITY_VERSION,
        "admission_identity": event.admission_identity,
        "source_id": event.source_id,
        "source_format": event.source_format,
        "source_version": event.source_version,
        "source_game_id": event.source_game_id,
        "source_native_game_id": event.source_native_game_id,
        "source_event_position": event.source_event_position,
        "actor_seat": event.actor_seat,
        "controlled_player": event.controlled_player,
        "observation_asset_sha256": sha256_json(
            _source_event_observation_asset_payload(event.observation_asset)
        ),
        "observation_payload_sha256": (
            sha256_json(event.observation)
            if event.observation is not None
            else None
        ),
        "stored_selected_choices": event.stored_selected_choices,
        "stored_index_space": event.stored_index_space,
        "legal_evidence_sha256": sha256_json(event.legacy_legal_actions),
        "transition_sha256": sha256_json(
            _without_storage_ownership(event.transition)
        ),
        "information_mode": event.information_mode,
        "action_schema_version": event.action_schema_version,
        "observation_schema_version": event.observation_schema_version,
        "semantic_source_record_sha256": _semantic_source_record_sha256(
            event.source_record
        ),
        "source_artifact_sha256": event.source_artifact_sha256,
        "source_corrupt": event.source_corrupt,
    }


def compute_source_event_id(event: SourceEvent) -> str:
    """Recompute the semantic event identity without shard or path ownership."""

    if not isinstance(event, SourceEvent):
        raise ValueError("event must be a SourceEvent")

    return identity("event1", _source_event_identity_payload(event))


def _source_schema_evidence(
    transition: Mapping[str, Any],
    source_record: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    values = []
    failures = []
    for owner, field in (
        (transition, "source_schema"),
        (source_record, "schema_version"),
    ):
        if field not in owner:
            continue
        value = owner.get(field)
        if _nonempty_string(value) is None:
            failures.append("source_schema_invalid")
            continue
        values.append(value)
    if not values:
        failures.append("source_schema_missing")
    elif any(
        not _exact_typed_equal(value, values[0])
        for value in values[1:]
    ):
        failures.append("source_schema_conflict")

    return str(values[0]) if values else "", tuple(failures)


def source_event_integrity_failures(event: SourceEvent) -> tuple[str, ...]:
    """Return deterministic content-integrity failures for a typed source event."""

    if not isinstance(event, SourceEvent):
        return ("not_source_event",)
    failures: list[str] = []
    if not _is_lower_sha256(event.admission_identity):
        failures.append("admission_identity_not_lower_sha256")
    if not isinstance(event.binding_id, str) or not event.binding_id:
        failures.append("binding_id_invalid")
    for field in ("local_row", "global_row", "source_event_position"):
        if _nonnegative_integer(getattr(event, field)) is None:
            failures.append(f"{field}_not_nonnegative_integer")
    for field in (
        "source_id",
        "source_format",
        "source_version",
        "source_game_id",
        "information_mode",
        "action_schema_version",
        "observation_schema_version",
    ):
        if _nonempty_string(getattr(event, field)) is None:
            failures.append(f"{field}_invalid")
    if _nonnegative_integer(event.actor_seat) is None:
        failures.append("actor_seat_not_nonnegative_integer")
    if _nonnegative_integer(event.controlled_player) is None:
        failures.append("controlled_player_not_nonnegative_integer")
    if _nonempty_string(event.source_native_game_id) is None:
        failures.append("source_native_game_id_invalid")
    if event.observation is not None and not isinstance(event.observation, Mapping):
        failures.append("observation_not_object")
    if event.stored_selected_choices is not None and (
        not isinstance(event.stored_selected_choices, tuple)
        or any(
            _nonnegative_integer(choice) is None
            for choice in event.stored_selected_choices
        )
    ):
        failures.append("stored_selected_choices_invalid")
    if (
        event.stored_index_space is not None
        and event.stored_index_space not in _SUPPORTED_INDEX_SPACES
    ):
        failures.append("stored_index_space_invalid")
    if not isinstance(event.legacy_legal_actions, tuple) or any(
        not isinstance(option, Mapping)
        for option in event.legacy_legal_actions
    ):
        failures.append("legacy_legal_actions_invalid")
    if type(event.source_corrupt) is not bool:
        failures.append("source_corrupt_not_boolean")
    elif event.source_corrupt:
        failures.append("source_corrupt")
    if not isinstance(event.transition, Mapping):
        failures.append("transition_not_object")
    if not isinstance(event.source_record, Mapping):
        failures.append("source_record_not_object")
    if not _is_lower_sha256(event.source_row_sha256):
        failures.append("source_row_sha256_invalid")
    elif isinstance(event.source_record, Mapping) and (
        event.source_row_sha256 != compute_source_row_sha256(event.source_record)
    ):
        failures.append("source_row_sha256_mismatch")
    if (
        event.source_artifact_sha256 is not None
        and not _is_lower_sha256(event.source_artifact_sha256)
    ):
        failures.append("source_artifact_sha256_invalid")
    asset = event.observation_asset
    if asset is not None:
        if not isinstance(asset, ObservationAssetReference):
            failures.append("observation_asset_invalid")
            asset = None
    if asset is not None:
        for field in ("source_sha256", "source_content_sha256", "content_sha256"):
            if not _is_lower_sha256(getattr(asset, field)):
                failures.append(f"observation_asset_{field}_invalid")
        if asset.tensor_sha256 is not None and not _is_lower_sha256(asset.tensor_sha256):
            failures.append("observation_asset_tensor_sha256_invalid")
        for field in ("source_line", "source_event_position", "actor_seat"):
            if _nonnegative_integer(getattr(asset, field)) is None:
                failures.append(f"observation_asset_{field}_invalid")
        if type(asset.reusable_admission_asset) is not bool:
            failures.append("observation_asset_reusable_flag_invalid")
        if asset.observation_id != compute_observation_asset_id(asset):
            failures.append("observation_asset_id_mismatch")
        if asset.observation_schema_version != event.observation_schema_version:
            failures.append("observation_asset_schema_version_mismatch")
        if asset.visibility_schema_version != VISIBILITY_SCHEMA_VERSION:
            failures.append("observation_asset_visibility_schema_version_mismatch")

    if isinstance(event.transition, Mapping) and isinstance(event.source_record, Mapping):
        source_schema, schema_failures = _source_schema_evidence(
            event.transition,
            event.source_record,
        )
        failures.extend(schema_failures)
        if source_schema and event.source_version != source_schema:
            failures.append("source_version_evidence_mismatch")
        source_observation, reference, field_registered = _source_observation(
            event.transition,
            event.source_record,
            source_schema,
        )
        if not field_registered:
            failures.append("source_observation_field_unregistered")
        if source_observation is None:
            failures.append("source_observation_missing")
        elif asset is None:
            failures.append("observation_asset_missing")
        projected_observation = None
        if source_observation is not None:
            try:
                projected_observation = project_actor_observation(source_observation)
            except ValueError:
                failures.append("source_observation_projection_invalid")
        if projected_observation is not None and not _exact_typed_equal(
            projected_observation,
            event.observation,
        ):
            failures.append("observation_payload_evidence_mismatch")

        if not isinstance(reference, Mapping):
            failures.append("source_observation_reference_missing")
        else:
            reference_path = _nonempty_string(reference.get("source_path"))
            if reference_path is None:
                failures.append("source_observation_path_invalid")
            elif asset is not None and asset.source_path != reference_path:
                failures.append("observation_asset_source_path_mismatch")
            reference_line = _nonnegative_integer(reference.get("source_line"))
            if reference_line is None:
                failures.append("source_observation_line_invalid")
            elif asset is not None and asset.source_line != reference_line:
                failures.append("observation_asset_source_line_mismatch")
            source_hashes = []
            for owner, field in (
                (reference, "source_sha256"),
                (event.transition, "source_hash"),
            ):
                if field not in owner:
                    continue
                value = owner.get(field)
                if not _is_lower_sha256(value):
                    failures.append("source_hash_evidence_invalid")
                    continue
                source_hashes.append(value)
            if not source_hashes:
                failures.append("source_hash_evidence_missing")
            elif any(
                not _exact_typed_equal(value, source_hashes[0])
                for value in source_hashes[1:]
            ):
                failures.append("source_hash_evidence_conflict")
            content_sha256 = reference.get("content_sha256")
            if not _is_lower_sha256(content_sha256):
                failures.append("source_content_sha256_invalid")
            elif source_observation is not None and (
                sha256_json(source_observation) != content_sha256
            ):
                failures.append("source_content_sha256_mismatch")
            if asset is not None:
                if source_hashes and asset.source_sha256 != source_hashes[0]:
                    failures.append("observation_asset_source_sha256_mismatch")
                if (
                    _is_lower_sha256(content_sha256)
                    and asset.source_content_sha256 != content_sha256
                ):
                    failures.append("observation_asset_source_content_sha256_mismatch")
                expected_content_sha256 = sha256_json({
                    "source_content_sha256": content_sha256,
                    "visibility_schema_version": VISIBILITY_SCHEMA_VERSION,
                })
                if asset.content_sha256 != expected_content_sha256:
                    failures.append("observation_asset_content_sha256_mismatch")
                if asset.admission_identity != event.admission_identity:
                    failures.append("observation_asset_admission_identity_mismatch")
                if asset.source_game_id != event.source_game_id:
                    failures.append("observation_asset_source_game_id_mismatch")
                if asset.source_event_position != event.source_event_position:
                    failures.append("observation_asset_source_event_position_mismatch")
                if asset.actor_seat != event.actor_seat:
                    failures.append("observation_asset_actor_seat_mismatch")
                if asset.information_mode != event.information_mode:
                    failures.append("observation_asset_information_mode_mismatch")

        if asset is not None and event.source_format in {
            "refgame",
            "kaggle_episode_replay",
        }:
            if event.source_artifact_path != asset.source_path:
                failures.append("source_artifact_path_mismatch")
            if event.source_artifact_sha256 != asset.source_sha256:
                failures.append("source_artifact_sha256_mismatch")
        elif event.source_format == "kaggle_replay":
            record_artifact_path = event.source_record.get("source_path")
            if record_artifact_path is not None and (
                event.source_artifact_path != record_artifact_path
            ):
                failures.append("source_artifact_path_mismatch")
            record_artifact_sha256 = event.source_record.get(
                "source_artifact_sha256",
            )
            if record_artifact_sha256 is not None and (
                event.source_artifact_sha256 != record_artifact_sha256
            ):
                failures.append("source_artifact_sha256_mismatch")

        selected, index_space, chosen_corrupt = _selected_choices(
            event.transition,
            event.source_record,
            source_schema,
        )
        if chosen_corrupt:
            failures.append("chosen_evidence_invalid_or_conflicting")
        if not _exact_typed_equal(selected, event.stored_selected_choices):
            failures.append("stored_selected_choices_evidence_mismatch")
        if not _exact_typed_equal(index_space, event.stored_index_space):
            failures.append("stored_index_space_evidence_mismatch")

        legal, legal_corrupt = _legal_evidence(
            event.transition,
            event.source_record,
            source_schema,
        )
        if legal_corrupt:
            failures.append("legal_evidence_invalid_or_conflicting")
        if not _exact_typed_equal(legal, event.legacy_legal_actions):
            failures.append("legacy_legal_actions_evidence_mismatch")
    if event.event_id != compute_source_event_id(event):
        failures.append("event_id_mismatch")

    return tuple(dict.fromkeys(failures))


def project_actor_observation(source_observation: Any) -> Mapping[str, Any]:
    """Return a non-mutating certified view without copying opaque bulk fields."""

    if not isinstance(source_observation, Mapping):
        raise ValueError("source observation must be an object")
    projected = dict(source_observation)
    projected.pop("logs", None)
    projected.pop("search_begin_input", None)
    current = projected.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("source observation current must be an object")
    projected_current = dict(current)
    projected_current.pop("looking", None)
    projected["current"] = projected_current
    select = projected.get("select")
    if isinstance(select, Mapping):
        projected_select = dict(select)
        projected_select.pop("deck", None)
        projected["select"] = projected_select
    assert_deployable_observation(
        projected,
        expected_mode=InformationMode.ACTOR_VISIBLE.value,
    )

    return projected


def _source_observation(
    transition: Mapping[str, Any],
    source_record: Mapping[str, Any],
    source_schema: str,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, bool]:
    state_before = transition.get("state_before")
    reference = state_before.get("source_observation") if isinstance(state_before, Mapping) else None
    if not isinstance(reference, Mapping):
        return None, None, False
    requested_field = reference.get("source_field")
    registered_fields = SOURCE_OBSERVATION_FIELD_BY_SCHEMA.get(source_schema)
    if (
        not isinstance(registered_fields, Mapping)
        or not isinstance(requested_field, str)
        or requested_field not in registered_fields
    ):
        return None, reference, False
    actual_field = registered_fields[requested_field]
    observation = source_record.get(actual_field)

    return (
        observation if isinstance(observation, Mapping) else None,
        reference,
        True,
    )


def _parse_action_evidence(
    value: Any,
    *,
    default_index_space: str,
) -> tuple[Mapping[str, Any] | None, bool]:
    if not isinstance(value, Mapping):
        return None, False
    allowed_fields = frozenset({"indices", "index_space", "semantics"})
    if frozenset(value) - allowed_fields:
        return None, False
    raw_values = value.get("indices")
    if not isinstance(raw_values, list) or any(
        _nonnegative_integer(item) is None for item in raw_values
    ):
        return None, False
    raw_index_space = value.get("index_space", default_index_space)
    if (
        not isinstance(raw_index_space, str)
        or raw_index_space not in _SUPPORTED_INDEX_SPACES
    ):
        return None, False
    semantics = value.get("semantics", _MISSING)
    if semantics is not _MISSING and (
        not isinstance(semantics, list)
        or any(not isinstance(item, Mapping) for item in semantics)
    ):
        return None, False

    return {
        "indices": tuple(raw_values),
        "index_space": raw_index_space,
        "semantics": semantics,
    }, True


def _selected_choices(
    transition: Mapping[str, Any],
    source_record: Mapping[str, Any],
    source_schema: str,
) -> tuple[tuple[int, ...] | None, str | None, bool]:
    evidence: list[Mapping[str, Any]] = []
    mapping_evidence: list[Mapping[str, Any]] = []
    source_corrupt = False
    raw_action_absent = False
    if source_schema in KAGGLE_REPLAY_SOURCE_SCHEMAS:
        if "action" not in source_record:
            source_corrupt = True
        else:
            raw_action = source_record.get("action")
            if raw_action is None:
                raw_action_absent = True
            elif not isinstance(raw_action, list) or any(
                _nonnegative_integer(item) is None for item in raw_action
            ):
                source_corrupt = True
            else:
                evidence.append({
                    "indices": tuple(raw_action),
                    "index_space": "option_position",
                    "semantics": _MISSING,
                })
    if "chosen_indices" in source_record:
        raw_values = source_record.get("chosen_indices")
        if not isinstance(raw_values, list) or any(
            _nonnegative_integer(item) is None for item in raw_values
        ):
            source_corrupt = True
        else:
            evidence.append({
                "indices": tuple(raw_values),
                "index_space": "option_position",
                "semantics": _MISSING,
            })
    for owner in (source_record, transition):
        if "chosen_action" not in owner:
            continue
        parsed, valid = _parse_action_evidence(
            owner.get("chosen_action"),
            default_index_space="source_index",
        )
        if not valid or parsed is None:
            source_corrupt = True
            continue
        evidence.append(parsed)
        mapping_evidence.append(parsed)
    if raw_action_absent and evidence:
        source_corrupt = True
    if not evidence:
        return None, None, source_corrupt
    reference = evidence[0]
    for candidate in evidence[1:]:
        if (
            not _exact_typed_equal(candidate["indices"], reference["indices"])
            or not _exact_typed_equal(
                candidate["index_space"],
                reference["index_space"],
            )
        ):
            source_corrupt = True
    if len(mapping_evidence) > 1:
        semantics_present = [
            candidate["semantics"] is not _MISSING
            for candidate in mapping_evidence
        ]
        if any(semantics_present) and not all(semantics_present):
            source_corrupt = True
        present_semantics = [
            candidate["semantics"]
            for candidate in mapping_evidence
            if candidate["semantics"] is not _MISSING
        ]
        if present_semantics and any(
            not _exact_typed_equal(candidate, present_semantics[0])
            for candidate in present_semantics[1:]
        ):
            source_corrupt = True

    return (
        tuple(reference["indices"]),
        str(reference["index_space"]),
        source_corrupt,
    )


def _legal_evidence(
    transition: Mapping[str, Any],
    source_record: Mapping[str, Any],
    source_schema: str,
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    evidence = []
    source_corrupt = False
    if source_schema in KAGGLE_REPLAY_SOURCE_SCHEMAS:
        source_observation = source_record.get("observation")
        select = (
            source_observation.get("select")
            if isinstance(source_observation, Mapping)
            else _MISSING
        )
        if select is _MISSING or select is None:
            evidence.append([])
        elif not isinstance(select, Mapping):
            source_corrupt = True
        else:
            raw_options = select.get("option", [])
            if not isinstance(raw_options, list) or any(
                not isinstance(option, Mapping) for option in raw_options
            ):
                source_corrupt = True
            else:
                evidence.append(raw_options)
    for owner, field in (
        (source_record, "legal_set"),
        (source_record, "legal_options_json"),
        (transition, "legal_action_set"),
    ):
        if field not in owner:
            continue
        value = owner.get(field)
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(option, Mapping) for option in value)
        ):
            source_corrupt = True
            continue
        evidence.append(value)
    if not evidence:
        return (), source_corrupt
    reference = evidence[0]
    if any(
        not _exact_typed_equal(candidate, reference)
        for candidate in evidence[1:]
    ):
        source_corrupt = True
    selected = next(
        (candidate for candidate in evidence if isinstance(candidate, list)),
        None,
    )
    legacy = (
        tuple(dict(option) for option in selected)
        if isinstance(selected, list)
        else ()
    )

    return legacy, source_corrupt


def source_event_from_admission(
    *,
    admission_identity: str,
    binding_id: str,
    local_row: int,
    global_row: int,
    transition: Mapping[str, Any],
    source_record: Mapping[str, Any],
    verify_source_content: bool = True,
) -> SourceEvent:
    if type(verify_source_content) is not bool:
        raise ValueError("verify_source_content must be a boolean")
    source_corrupt = False
    if not _is_lower_sha256(admission_identity):
        source_corrupt = True
    if not isinstance(binding_id, str) or not binding_id:
        source_corrupt = True
    if _nonnegative_integer(local_row) is None or _nonnegative_integer(global_row) is None:
        source_corrupt = True

    source_schema, schema_failures = _source_schema_evidence(
        transition,
        source_record,
    )
    source_corrupt = source_corrupt or bool(schema_failures)

    source_observation, reference, field_registered = _source_observation(
        transition,
        source_record,
        source_schema,
    )
    if not field_registered:
        source_corrupt = True
    raw_source_id = transition.get("source_id", _MISSING)
    if raw_source_id is _MISSING:
        source_id = binding_id.split(":", 1)[0] if isinstance(binding_id, str) else ""
    else:
        source_id = _nonempty_string(raw_source_id) or ""
        if not source_id:
            source_corrupt = True
    native_game_values = []
    for owner, field in (
        (source_record, "game_id"),
        (transition, "source_game_id"),
    ):
        if field not in owner:
            continue
        value = owner.get(field)
        if _nonempty_string(value) is None:
            source_corrupt = True
            continue
        native_game_values.append(value)
    if native_game_values and any(
        not _exact_typed_equal(value, native_game_values[0])
        for value in native_game_values[1:]
    ):
        source_corrupt = True
    canonical_game_value = transition.get("canonical_game_id", _MISSING)
    if canonical_game_value is not _MISSING and _nonempty_string(canonical_game_value) is None:
        source_corrupt = True
    source_native_game_id = (
        str(native_game_values[0])
        if native_game_values
        else str(canonical_game_value) if _nonempty_string(canonical_game_value) else ""
    )
    source_game_id = (
        str(canonical_game_value)
        if _nonempty_string(canonical_game_value)
        else source_native_game_id
    )
    if not source_game_id:
        source_corrupt = True

    source_position = None
    for owner, field in (
        (source_record, "step"),
        (source_record, "decision_index"),
        (transition, "global_decision_index"),
    ):
        if field not in owner:
            continue
        candidate = _nonnegative_integer(owner.get(field))
        if candidate is None:
            source_corrupt = True
            continue
        if source_position is None:
            source_position = candidate
    actor = _nonnegative_integer(transition.get("actor"))
    controlled = None
    observation = None
    if source_observation is not None:
        try:
            observation = project_actor_observation(source_observation)
        except ValueError:
            source_corrupt = True
            observation = None
    current = observation.get("current") if isinstance(observation, Mapping) else None
    if isinstance(current, Mapping):
        controlled = _nonnegative_integer(current.get("yourIndex"))
    source_corrupt = source_corrupt or any(
        value is None
        for value in (
            source_observation,
            observation,
            reference,
            source_position,
            actor,
            controlled,
        )
    )
    source_row_sha256 = compute_source_row_sha256(source_record)
    information_mode_value = transition.get("information_mode")
    information_mode = _nonempty_string(information_mode_value) or ""
    if not information_mode:
        source_corrupt = True
    action_schema_value = transition.get("schema_version")
    action_schema_version = _nonempty_string(action_schema_value) or ""
    if not action_schema_version:
        source_corrupt = True
    record_information_mode = source_record.get("information_mode", _MISSING)
    if record_information_mode is not _MISSING and not _exact_typed_equal(
        record_information_mode,
        information_mode_value,
    ):
        source_corrupt = True

    observation_asset = None
    if (
        isinstance(reference, Mapping)
        and source_observation is not None
        and observation is not None
        and source_position is not None
        and actor is not None
    ):
        source_sha_values = []
        for owner, field in (
            (reference, "source_sha256"),
            (transition, "source_hash"),
        ):
            if field not in owner:
                continue
            value = owner.get(field)
            if not _is_lower_sha256(value):
                source_corrupt = True
                continue
            source_sha_values.append(value)
        if source_sha_values and any(
            not _exact_typed_equal(value, source_sha_values[0])
            for value in source_sha_values[1:]
        ):
            source_corrupt = True
        source_sha256 = str(source_sha_values[0]) if source_sha_values else ""
        source_path = _nonempty_string(reference.get("source_path")) or ""
        source_line = _nonnegative_integer(reference.get("source_line"))
        raw_content_sha256 = reference.get("content_sha256")
        source_content_sha256 = (
            str(raw_content_sha256)
            if _is_lower_sha256(raw_content_sha256)
            else ""
        )
        content_sha256 = sha256_json({
            "source_content_sha256": source_content_sha256,
            "visibility_schema_version": VISIBILITY_SCHEMA_VERSION,
        })
        if not source_sha256 or not source_path or source_line is None or not source_content_sha256:
            source_corrupt = True
        elif verify_source_content and sha256_json(source_observation) != source_content_sha256:
            source_corrupt = True
        observation_asset = ObservationAssetReference(
            admission_identity=admission_identity,
            source_sha256=source_sha256,
            source_path=source_path,
            source_line=source_line,
            source_content_sha256=source_content_sha256,
            content_sha256=content_sha256,
            source_game_id=source_game_id,
            source_event_position=source_position,
            actor_seat=actor,
            information_mode=information_mode,
            observation_schema_version=source_schema,
            visibility_schema_version=VISIBILITY_SCHEMA_VERSION,
            asset_origin="admission_v0",
            reusable_admission_asset=True,
            feature_version=(
                str(transition["state_before"].get("representation"))
                if isinstance(transition.get("state_before"), Mapping)
                and _nonempty_string(transition["state_before"].get("representation"))
                else None
            ),
            observation_id="",
        )
        observation_asset = replace(
            observation_asset,
            observation_id=compute_observation_asset_id(observation_asset),
        )
    selected, index_space, chosen_corrupt = _selected_choices(
        transition,
        source_record,
        source_schema,
    )
    legacy_legal, legal_corrupt = _legal_evidence(
        transition,
        source_record,
        source_schema,
    )
    source_corrupt = source_corrupt or chosen_corrupt or legal_corrupt
    source_format = "kaggle_replay" if source_id == "kaggle_ingest" else "refgame"
    record_artifact_path = source_record.get("source_path", _MISSING)
    if record_artifact_path is not _MISSING and _nonempty_string(record_artifact_path) is None:
        source_corrupt = True
    if source_format == "kaggle_replay":
        source_artifact_path = (
            str(record_artifact_path)
            if _nonempty_string(record_artifact_path)
            else None
        )
        raw_artifact_sha = source_record.get("source_artifact_sha256", _MISSING)
        if raw_artifact_sha is not _MISSING and not _is_lower_sha256(raw_artifact_sha):
            source_corrupt = True
        source_artifact_sha256 = (
            str(raw_artifact_sha)
            if _is_lower_sha256(raw_artifact_sha)
            else None
        )
    else:
        source_artifact_path = (
            observation_asset.source_path
            if observation_asset is not None
            else None
        )
        source_artifact_sha256 = (
            observation_asset.source_sha256
            if observation_asset is not None
            else None
        )
    event = SourceEvent(
        admission_identity=admission_identity,
        binding_id=binding_id,
        local_row=local_row,
        global_row=global_row,
        source_id=source_id,
        source_format=source_format,
        source_version=source_schema,
        source_game_id=source_game_id,
        source_native_game_id=source_native_game_id,
        source_event_position=source_position if source_position is not None else -1,
        actor_seat=actor,
        controlled_player=controlled,
        observation_asset=observation_asset,
        observation=dict(observation) if observation is not None else None,
        stored_selected_choices=selected,
        stored_index_space=index_space,
        legacy_legal_actions=legacy_legal,
        transition=dict(transition),
        information_mode=information_mode,
        action_schema_version=action_schema_version,
        observation_schema_version=source_schema,
        source_row_sha256=source_row_sha256,
        source_record=dict(source_record),
        source_artifact_path=source_artifact_path,
        source_artifact_sha256=source_artifact_sha256,
        source_corrupt=source_corrupt,
        event_id="",
    )

    return replace(event, event_id=compute_source_event_id(event))


__all__ = [
    "OBSERVATION_ASSET_IDENTITY_VERSION",
    "SOURCE_OBSERVATION_FIELD_BY_SCHEMA",
    "SOURCE_EVENT_IDENTITY_VERSION",
    "VISIBILITY_SCHEMA_VERSION",
    "compute_observation_asset_id",
    "compute_source_event_id",
    "compute_source_row_sha256",
    "project_actor_observation",
    "source_event_integrity_failures",
    "source_event_from_admission",
]
