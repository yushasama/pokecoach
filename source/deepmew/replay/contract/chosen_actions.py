"""Canonical chosen-action reconstruction with no silent repair."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from deepmew.replay.contract.identities import identity
from deepmew.replay.contract.legal_actions import allowed_option_ids
from deepmew.replay.contract.schemas import (
    CanonicalChosenAction,
    CanonicalChosenStage,
    CanonicalLegalActionSet,
    ChoicePresence,
    OrderingSignificance,
    ValidationFailure,
)


CHOSEN_ACTION_RECONSTRUCTOR_VERSION = "deepmew.chosen_action_reconstructor.v1"


def _unique_failures(values: Sequence[ValidationFailure]) -> tuple[ValidationFailure, ...]:
    return tuple(dict.fromkeys(values))


def _legacy_position_map(
    legacy_options: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, int], bool]:
    mapping: dict[int, int] = {}
    ambiguous = False
    for position, option in enumerate(legacy_options):
        raw_index = option.get("index", position)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            ambiguous = True
            continue
        if raw_index in mapping:
            ambiguous = True
            continue
        mapping[raw_index] = position

    return mapping, ambiguous


def _source_positions(
    values: tuple[int, ...],
    *,
    index_space: str | None,
    legacy_options: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, ...], tuple[ValidationFailure, ...]]:
    if index_space in (None, "option_position"):
        return values, ()
    if index_space != "source_index":
        return (), (ValidationFailure.UNRESOLVED_ACTION,)
    mapping, ambiguous = _legacy_position_map(legacy_options)
    failures = []
    if ambiguous:
        failures.append(ValidationFailure.UNRESOLVED_ACTION)
    positions = []
    for value in values:
        position = mapping.get(value)
        if position is None:
            failures.append(ValidationFailure.CHOICE_NOT_LEGAL)
        else:
            positions.append(position)

    return tuple(positions), _unique_failures(failures)


def reconstruct_chosen_action(
    legal_actions: CanonicalLegalActionSet,
    stage_selected_values: Sequence[Sequence[int] | None],
    *,
    stage_index_spaces: Sequence[str | None] | None = None,
    legacy_stage_options: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    source_action_template: str | None = None,
) -> CanonicalChosenAction:
    failures: list[ValidationFailure] = list(legal_actions.failures)
    if source_action_template is not None and source_action_template != legal_actions.action_template:
        failures.append(ValidationFailure.ACTION_TEMPLATE_NOT_LEGAL)
    if len(stage_selected_values) != len(legal_actions.stages):
        failures.append(ValidationFailure.UNRESOLVED_ACTION)
    index_spaces = tuple(stage_index_spaces or (None,) * len(stage_selected_values))
    legacy_options = tuple(legacy_stage_options or ((),) * len(stage_selected_values))
    if len(index_spaces) != len(stage_selected_values) or len(legacy_options) != len(stage_selected_values):
        failures.append(ValidationFailure.UNRESOLVED_ACTION)
    chosen_stages = []
    prior: tuple[tuple[str, ...], ...] = ()
    for stage_index, stage in enumerate(legal_actions.stages):
        selected_raw = stage_selected_values[stage_index] if stage_index < len(stage_selected_values) else None
        index_space = index_spaces[stage_index] if stage_index < len(index_spaces) else None
        stage_legacy = legacy_options[stage_index] if stage_index < len(legacy_options) else ()
        if selected_raw is None:
            selected_values: tuple[int, ...] = ()
            omitted_optional = stage.presence == ChoicePresence.OPTIONAL
            if not omitted_optional:
                failures.append(ValidationFailure.MISSING_REQUIRED_CHOICE)
        else:
            omitted_optional = False
            if any(isinstance(value, bool) or not isinstance(value, int) for value in selected_raw):
                failures.append(ValidationFailure.UNRESOLVED_ACTION)
                selected_values = ()
            else:
                selected_values = tuple(int(value) for value in selected_raw)
        if len(selected_values) < stage.minimum_count:
            failures.append(
                ValidationFailure.MISSING_REQUIRED_CHOICE
                if not selected_values and stage.minimum_count > 0
                else ValidationFailure.TOO_FEW_CHOICES
            )
        if len(selected_values) > stage.maximum_count:
            failures.append(ValidationFailure.TOO_MANY_CHOICES)
        positions, position_failures = _source_positions(
            selected_values,
            index_space=index_space,
            legacy_options=stage_legacy,
        )
        failures.extend(position_failures)
        if not stage.repetition_allowed and len(positions) != len(set(positions)):
            failures.append(ValidationFailure.DUPLICATE_FORBIDDEN_CHOICE)
        option_by_position = {option.position: option for option in stage.options}
        selected_ids = []
        for position in positions:
            option = option_by_position.get(position)
            if option is None:
                failures.append(ValidationFailure.CHOICE_NOT_LEGAL)
            else:
                selected_ids.append(option.option_id)
        allowed = allowed_option_ids(stage, prior)
        if allowed is None:
            failures.append(ValidationFailure.UNRESOLVED_ACTION)
        elif not set(selected_ids).issubset(allowed):
            failures.append(ValidationFailure.CHOICE_NOT_LEGAL)
        if len(selected_ids) > 1 and stage.ordering_significance == OrderingSignificance.UNKNOWN:
            failures.append(ValidationFailure.CHOICE_ORDER)
        identity_ids = tuple(selected_ids)
        if stage.ordering_significance in {
            OrderingSignificance.NOT_APPLICABLE,
            OrderingSignificance.UNORDERED,
        }:
            identity_ids = tuple(sorted(identity_ids))
        chosen_stages.append(CanonicalChosenStage(
            stage_id=stage.stage_id,
            selected_option_ids=tuple(selected_ids),
            source_selected_values=selected_values,
            omitted_optional=omitted_optional,
        ))
        prior += (identity_ids,)
    unique_failures = _unique_failures(failures)
    complete = not unique_failures
    chosen_action_id = None
    if complete:
        chosen_action_id = identity("chosen1", {
            "reconstructor_version": CHOSEN_ACTION_RECONSTRUCTOR_VERSION,
            "legal_action_set_id": legal_actions.legal_action_set_id,
            "action_template": legal_actions.action_template,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "selected_option_ids": (
                        tuple(sorted(stage.selected_option_ids))
                        if legal_stage.ordering_significance in {
                            OrderingSignificance.NOT_APPLICABLE,
                            OrderingSignificance.UNORDERED,
                        }
                        else stage.selected_option_ids
                    ),
                    "omitted_optional": stage.omitted_optional,
                }
                for stage, legal_stage in zip(chosen_stages, legal_actions.stages)
            ],
        })

    return CanonicalChosenAction(
        action_template=legal_actions.action_template,
        stages=tuple(chosen_stages),
        chosen_action_id=chosen_action_id,
        complete=complete,
        failures=unique_failures,
    )


__all__ = [
    "CHOSEN_ACTION_RECONSTRUCTOR_VERSION",
    "reconstruct_chosen_action",
]
