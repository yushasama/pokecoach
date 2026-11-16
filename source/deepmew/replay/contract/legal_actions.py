"""Canonical typed legal-action reconstruction.

The CABT adapter is deliberately explicit.  Duplicate selections are forbidden
by the vendored 0.1.0 API, ``SKILL_ORDER`` is ordered by its documented source
context, and ``TO_DECK_BOTTOM`` multi-select order remains unknown/fail-closed.
All other CABT multi-select contexts use set semantics in this V1 adapter.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from deepmew.replay.contract.identities import identity, sha256_json
from deepmew.replay.contract.schemas import (
    CanonicalLegalActionSet,
    CanonicalLegalOption,
    ChoicePresence,
    ChoiceStageContract,
    ConditionalOptionSet,
    OrderingSignificance,
    ValidationFailure,
)


CABT_LEGAL_ACTION_ADAPTER_VERSION = "deepmew.cabt_legal_action_adapter.v1"
REFGAME_SEMANTIC_CROSSWALK_VERSION = "deepmew.refgame_semantic_crosswalk.v1"

SELECT_TYPE_NAMES = {
    0: "MAIN",
    1: "CARD",
    2: "ATTACHED_CARD",
    3: "CARD_OR_ATTACHED_CARD",
    4: "ENERGY",
    5: "SKILL",
    6: "ATTACK",
    7: "EVOLVE",
    8: "COUNT",
    9: "YES_NO",
    10: "SPECIAL_CONDITION",
}

SELECT_CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE_POKEMON",
    2: "SETUP_BENCH_POKEMON",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    9: "TO_DECK",
    10: "TO_DECK_BOTTOM",
    11: "TO_PRIZE",
    12: "NOT_MOVE",
    13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY",
    15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER",
    17: "HEAL",
    18: "EVOLVES_FROM",
    19: "EVOLVES_TO",
    20: "DEVOLVE",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    23: "DETACH_FROM",
    24: "LOOK",
    25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD",
    27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY",
    33: "SWITCH_ENERGY",
    34: "SKILL_ORDER",
    35: "ATTACK",
    36: "DISABLE_ATTACK",
    37: "EVOLVE",
    38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST",
    42: "MULLIGAN",
    43: "ACTIVATE",
    44: "FIRST_EFFECT",
    45: "MORE_DEVOLVE",
    46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}

OPTION_TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}

AREA_TYPE_NAMES = {
    1: "DECK",
    2: "HAND",
    3: "DISCARD",
    4: "ACTIVE",
    5: "BENCH",
    6: "PRIZE",
    7: "STADIUM",
    8: "ENERGY",
    9: "TOOL",
    10: "PRE_EVOLUTION",
    11: "PLAYER",
    12: "LOOKING",
}

SPECIAL_CONDITION_TYPE_NAMES = {
    0: "POISON",
    1: "BURN",
    2: "SLEEP",
    3: "PARALYZE",
    4: "CONFUSE",
}

# Frozen from cg.api.SelectType's documented OptionType compatibility.
SELECT_OPTION_TYPES = {
    0: frozenset({7, 8, 9, 10, 11, 12, 13, 14}),
    1: frozenset({3}),
    2: frozenset({4, 5}),
    3: frozenset({3, 4, 5}),
    4: frozenset({6}),
    5: frozenset({15}),
    6: frozenset({13}),
    7: frozenset({9}),
    8: frozenset({0}),
    9: frozenset({1, 2}),
    10: frozenset({16}),
}

# ``cg.api.Option`` is a tagged payload. Known optional fields may serialize
# as null, but the documented semantic fields for an OptionType must not.
CABT_OPTION_FIELDS = frozenset({
    "type",
    "number",
    "area",
    "index",
    "playerIndex",
    "toolIndex",
    "energyIndex",
    "count",
    "inPlayArea",
    "inPlayIndex",
    "attackId",
    "cardId",
    "serial",
    "specialConditionType",
})
CABT_INTEGER_OPTION_FIELDS = frozenset({
    "number",
    "index",
    "playerIndex",
    "toolIndex",
    "energyIndex",
    "count",
    "inPlayIndex",
    "attackId",
    "cardId",
    "serial",
})
CABT_OPTION_REQUIRED_FIELDS = {
    0: frozenset({"number"}),
    1: frozenset(),
    2: frozenset(),
    3: frozenset({"area", "index", "playerIndex"}),
    4: frozenset({"area", "index", "playerIndex", "toolIndex"}),
    5: frozenset({"area", "index", "playerIndex", "energyIndex"}),
    6: frozenset({"area", "index", "playerIndex", "energyIndex", "count"}),
    7: frozenset({"index"}),
    8: frozenset({"area", "index", "inPlayArea", "inPlayIndex"}),
    9: frozenset({"area", "index", "inPlayArea", "inPlayIndex"}),
    10: frozenset({"area", "index"}),
    11: frozenset({"area", "index"}),
    12: frozenset(),
    13: frozenset({"attackId"}),
    14: frozenset(),
    15: frozenset({"cardId", "serial"}),
    16: frozenset({"specialConditionType"}),
}

# These fields form the frozen semantic identity emitted by historical
# refgame producers.  Card names are presentation metadata and deliberately do
# not influence canonical legal-action identity.
REFGAME_CORE_SEMANTIC_FIELDS = (
    "index",
    "option_type",
    "select_type",
    "select_context",
    "semantic_key",
    "source_area",
    "source_index",
    "source_card_id",
    "target_area",
    "target_index",
    "target_card_id",
    "attack_id",
)
REFGAME_DISPLAY_ONLY_SEMANTIC_FIELDS = frozenset({
    "source_card_name",
    "target_card_name",
})

ORDERED_CONTEXTS = frozenset({34})
UNKNOWN_MULTI_ORDER_CONTEXTS = frozenset({10})


class LegalActionContractError(ValueError):
    """Raised when a legal-action source cannot form a typed contract."""


class StaleLegalSetError(LegalActionContractError):
    """Raised when well-formed semantic evidence contradicts raw CABT state."""


def _optional_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None

    return _integer(value, field=field)


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)

    return getattr(value, key, default)


def _card_id(value: Any) -> int | None:
    if value is None:
        return None
    raw = _mapping_value(
        value,
        "id",
        _mapping_value(value, "cardId", _mapping_value(value, "cardID")),
    )
    if raw is None:
        return None

    return _integer(raw, field="actor_observation card id")


def _actor_players(actor_observation: Mapping[str, Any]) -> tuple[Sequence[Any], int]:
    current = actor_observation.get("current")
    if not isinstance(current, Mapping):
        raise LegalActionContractError("actor_observation.current must be an object")
    players = current.get("players")
    if not isinstance(players, list):
        raise LegalActionContractError("actor_observation.current.players must be a list")
    actor_index = _integer(current.get("yourIndex"), field="actor_observation.current.yourIndex")
    if actor_index < 0 or actor_index >= len(players):
        raise LegalActionContractError("actor_observation.current.yourIndex is out of range")

    return players, actor_index


def _player_by_index(
    actor_observation: Mapping[str, Any],
    player_index: int | None,
) -> Any:
    players, actor_index = _actor_players(actor_observation)
    resolved_index = actor_index if player_index is None else player_index
    if resolved_index < 0 or resolved_index >= len(players):
        raise LegalActionContractError("raw option playerIndex is out of range")

    return players[resolved_index]


def _area_card(
    actor_observation: Mapping[str, Any],
    *,
    area: str,
    index: int | None,
    player_index: int | None,
) -> Any | None:
    if not area or index is None:
        return None
    field_by_area = {
        "HAND": "hand",
        "DECK": "deck",
        "DISCARD": "discard",
        "LOST": "lost",
        "PRIZE": "prize",
        "ACTIVE": "active",
        "BENCH": "bench",
    }
    field = field_by_area.get(area)
    if field is None:
        return None
    values = _mapping_value(_player_by_index(actor_observation, player_index), field)
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return None

    return values[index]


def _hand_source(
    actor_observation: Mapping[str, Any],
    raw_option: Mapping[str, Any],
) -> tuple[int | None, str, int | None]:
    source_index = _optional_integer(raw_option.get("index"), field="select.option.index")
    if source_index is None or source_index < 0:
        return None, "HAND", source_index
    card = _area_card(
        actor_observation,
        area="HAND",
        index=source_index,
        player_index=None,
    )

    return _card_id(card), "HAND", source_index


def _target_fields(
    actor_observation: Mapping[str, Any],
    raw_option: Mapping[str, Any],
    *,
    player_index: int | None,
) -> tuple[str, int | None, int | None]:
    raw_area = raw_option.get("inPlayArea", raw_option.get("targetArea"))
    target_area = _enum_name(raw_area, AREA_TYPE_NAMES, prefix="AREA") if raw_area is not None else ""
    raw_index = raw_option.get("inPlayIndex", raw_option.get("targetIndex"))
    target_index = _optional_integer(raw_index, field="select.option.inPlayIndex")
    if target_area == "ACTIVE" and target_index is None:
        target_index = 0
    card = _area_card(
        actor_observation,
        area=target_area,
        index=target_index,
        player_index=player_index,
    )

    return target_area, target_index, _card_id(card)


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LegalActionContractError(f"{field} must be an integer")

    return value


def _known_enum_integer(
    value: Any,
    names: Mapping[int, str],
    *,
    field: str,
) -> int:
    resolved = _integer(value, field=field)
    if resolved not in names:
        raise LegalActionContractError(f"{field} is outside the frozen CABT enum")

    return resolved


def exact_json_value_equal(left: Any, right: Any) -> bool:
    """Compare JSON-domain values without Python's bool/int/float aliases."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if frozenset(left) != frozenset(right):
            return False

        return all(exact_json_value_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False

        return all(
            exact_json_value_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )

    return left == right


def _validate_cabt_option_fields(
    raw_option: Mapping[str, Any],
    *,
    option_type: int,
    position: int,
) -> None:
    unsupported = frozenset(raw_option) - CABT_OPTION_FIELDS
    if unsupported:
        raise LegalActionContractError(
            f"select.option[{position}] has unsupported fields: {sorted(unsupported)}"
        )
    required = CABT_OPTION_REQUIRED_FIELDS[option_type]
    missing = frozenset(
        field
        for field in required
        if field not in raw_option or raw_option[field] is None
    )
    if missing:
        raise LegalActionContractError(
            f"select.option[{position}] is missing required fields: {sorted(missing)}"
        )
    for field in CABT_INTEGER_OPTION_FIELDS:
        value = raw_option.get(field)
        if value is None:
            continue
        resolved = _integer(value, field=f"select.option[{position}].{field}")
        if resolved < 0:
            raise LegalActionContractError(
                f"select.option[{position}].{field} must be non-negative"
            )
    for field, names in (
        ("area", AREA_TYPE_NAMES),
        ("inPlayArea", AREA_TYPE_NAMES),
        ("specialConditionType", SPECIAL_CONDITION_TYPE_NAMES),
    ):
        value = raw_option.get(field)
        if value is not None:
            _known_enum_integer(
                value,
                names,
                field=f"select.option[{position}].{field}",
            )


def _validate_cabt_options(
    raw_options: Sequence[Any],
    *,
    select_type: int,
) -> tuple[Mapping[str, Any], ...]:
    if not raw_options:
        raise LegalActionContractError("CABT select.option must be non-empty")
    validated = []
    for position, raw_option in enumerate(raw_options):
        if not isinstance(raw_option, Mapping):
            raise LegalActionContractError("CABT select.option entries must be objects")
        option_type = _known_enum_integer(
            raw_option.get("type"),
            OPTION_TYPE_NAMES,
            field=f"select.option[{position}].type",
        )
        if option_type not in SELECT_OPTION_TYPES[select_type]:
            raise LegalActionContractError(
                f"select.option[{position}].type is incompatible with select.type"
            )
        _validate_cabt_option_fields(
            raw_option,
            option_type=option_type,
            position=position,
        )
        validated.append(raw_option)

    return tuple(validated)


def _enum_name(value: Any, names: Mapping[int, str], *, prefix: str) -> str:
    if isinstance(value, bool):
        return f"{prefix}_UNKNOWN_BOOL"
    if isinstance(value, int):
        return names.get(value, f"{prefix}_UNKNOWN_{value}")

    return f"{prefix}_{str(value).strip() or 'UNKNOWN'}"


def build_refgame_semantic_legal_options(
    actor_observation: Mapping[str, Any],
    *,
    select: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Derive the frozen refgame semantic table from actor-visible CABT data.

    ``select`` may be supplied by a caller that has already extracted it, but
    it must still equal ``actor_observation.select``.  This prevents a legal
    table from one observation from being crosswalked against another.
    """

    if not isinstance(actor_observation, Mapping):
        raise LegalActionContractError("actor_observation must be an object")
    observation_select = actor_observation.get("select")
    if not isinstance(observation_select, Mapping):
        raise LegalActionContractError("actor_observation.select must be an object")
    raw_select = observation_select if select is None else select
    if not isinstance(raw_select, Mapping):
        raise LegalActionContractError("CABT select payload must be an object")
    if dict(raw_select) != dict(observation_select):
        raise StaleLegalSetError("CABT select does not belong to actor_observation")
    select_type_value = _known_enum_integer(
        raw_select.get("type"),
        SELECT_TYPE_NAMES,
        field="select.type",
    )
    select_context_value = _known_enum_integer(
        raw_select.get("context"),
        SELECT_CONTEXT_NAMES,
        field="select.context",
    )
    select_type = _enum_name(select_type_value, SELECT_TYPE_NAMES, prefix="SELECT")
    select_context = _enum_name(select_context_value, SELECT_CONTEXT_NAMES, prefix="CONTEXT")
    raw_options = raw_select.get("option")
    if not isinstance(raw_options, list) or not raw_options:
        raise LegalActionContractError("CABT select.option must be a non-empty list")
    validated_options = _validate_cabt_options(
        raw_options,
        select_type=select_type_value,
    )
    rows = []
    for position, raw_value in enumerate(validated_options):
        raw_option = dict(raw_value)
        option_type_value = _known_enum_integer(
            raw_option.get("type"),
            OPTION_TYPE_NAMES,
            field="select.option.type",
        )
        option_type = _enum_name(option_type_value, OPTION_TYPE_NAMES, prefix="OPTION")
        raw_source_area = raw_option.get("area")
        source_area = (
            _enum_name(raw_source_area, AREA_TYPE_NAMES, prefix="AREA")
            if raw_source_area is not None
            else ""
        )
        source_index = _optional_integer(raw_option.get("index"), field="select.option.index")
        source_card_id = None
        target_area = ""
        target_index = None
        target_card_id = None
        attack_id = None
        if option_type in {"PLAY", "EVOLVE"}:
            source_card_id, source_area, source_index = _hand_source(
                actor_observation,
                raw_option,
            )
        if option_type == "EVOLVE" and source_card_id is None:
            source_card_id = _card_id({
                "id": raw_option.get("cardId", raw_option.get("cardID")),
            })
        player_index = _optional_integer(
            raw_option.get("playerIndex"),
            field="select.option.playerIndex",
        )
        if source_card_id is None and source_area:
            source_card_id = _card_id(_area_card(
                actor_observation,
                area=source_area,
                index=source_index,
                player_index=player_index,
            ))
        if option_type == "ATTACK":
            attack_id = _optional_integer(
                raw_option.get("attackId", raw_option.get("attackID")),
                field="select.option.attackId",
            )
            if attack_id is not None and attack_id < 0:
                attack_id = None
        if option_type in {"ATTACH", "EVOLVE"}:
            target_area, target_index, target_card_id = _target_fields(
                actor_observation,
                raw_option,
                player_index=player_index,
            )
        if target_card_id is None and target_area:
            target_card_id = _card_id(_area_card(
                actor_observation,
                area=target_area,
                index=target_index,
                player_index=player_index,
            ))
        if option_type in {"PLAY", "EVOLVE", "CARD"} and source_card_id is not None:
            semantic_key = f"{option_type}:{source_card_id}"
        elif option_type == "ATTACH" and target_area:
            semantic_key = f"ATTACH:{target_area}:{target_card_id or target_index or 0}"
        elif option_type == "ATTACK" and attack_id is not None:
            semantic_key = f"ATTACK:{attack_id}"
        elif option_type in {"YES", "NO", "END", "RETREAT"}:
            semantic_key = option_type
        else:
            semantic_key = f"{option_type}:{position}"
        rows.append({
            "index": position,
            "option_type": option_type,
            "select_type": select_type,
            "select_context": select_context,
            "semantic_key": semantic_key,
            "source_area": source_area,
            "source_index": source_index,
            "source_card_id": source_card_id,
            "target_area": target_area,
            "target_index": target_index,
            "target_card_id": target_card_id,
            "attack_id": attack_id,
        })

    return tuple(rows)


def _verified_semantic_options(
    legal_options: Sequence[Mapping[str, Any]],
    expected_options: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if len(legal_options) != len(expected_options):
        raise StaleLegalSetError("semantic option count differs from the source observation")
    core_fields = frozenset(REFGAME_CORE_SEMANTIC_FIELDS)
    allowed_fields = core_fields | REFGAME_DISPLAY_ONLY_SEMANTIC_FIELDS
    verified = []
    for position, (source_option, expected_option) in enumerate(
        zip(legal_options, expected_options),
    ):
        if not isinstance(source_option, Mapping):
            raise LegalActionContractError("semantic legal_options entries must be objects")
        source_fields = frozenset(source_option)
        missing = core_fields - source_fields
        unsupported = source_fields - allowed_fields
        if missing:
            raise LegalActionContractError(
                f"semantic option {position} is missing core fields: {sorted(missing)}"
            )
        if unsupported:
            raise LegalActionContractError(
                f"semantic option {position} has unsupported fields: {sorted(unsupported)}"
            )
        source_core = {
            field: source_option[field]
            for field in REFGAME_CORE_SEMANTIC_FIELDS
        }
        expected_core = {
            field: expected_option[field]
            for field in REFGAME_CORE_SEMANTIC_FIELDS
        }
        if not exact_json_value_equal(source_core, expected_core):
            differing = [
                field
                for field in REFGAME_CORE_SEMANTIC_FIELDS
                if not exact_json_value_equal(
                    source_core[field],
                    expected_core[field],
                )
            ]
            raise StaleLegalSetError(
                f"semantic option {position} differs from actor observation: {differing}"
            )
        verified.append(expected_core)

    return tuple(verified)


def cabt_ordering_significance(context: Any, maximum_count: int) -> OrderingSignificance:
    if maximum_count <= 1:
        return OrderingSignificance.NOT_APPLICABLE
    if context in ORDERED_CONTEXTS:
        return OrderingSignificance.ORDERED
    if context in UNKNOWN_MULTI_ORDER_CONTEXTS:
        return OrderingSignificance.UNKNOWN

    return OrderingSignificance.UNORDERED


def semantic_ordering_significance(context: Any, maximum_count: int) -> OrderingSignificance:
    """Map a normalized semantic context onto the same CABT ordering rule."""

    if maximum_count <= 1:
        return OrderingSignificance.NOT_APPLICABLE
    normalized = str(context or "").strip().upper()
    if normalized == "SKILL_ORDER":
        return OrderingSignificance.ORDERED
    if normalized == "TO_DECK_BOTTOM":
        return OrderingSignificance.UNKNOWN

    return OrderingSignificance.UNORDERED


def _selection_count(stage: ChoiceStageContract) -> int | None:
    option_count = len(stage.options)
    minimum = stage.minimum_count
    maximum = stage.maximum_count
    ordering = stage.ordering_significance
    if ordering == OrderingSignificance.UNKNOWN:
        return None
    total = 0
    for selected_count in range(minimum, maximum + 1):
        if stage.repetition_allowed:
            if ordering == OrderingSignificance.ORDERED:
                total += option_count ** selected_count
            elif selected_count == 0:
                total += 1
            elif option_count:
                total += math.comb(option_count + selected_count - 1, selected_count)
        elif ordering == OrderingSignificance.ORDERED:
            total += math.perm(option_count, selected_count)
        else:
            total += math.comb(option_count, selected_count)

    return total


def _stage_selections(stage: ChoiceStageContract) -> Iterable[tuple[str, ...]]:
    option_ids = tuple(option.option_id for option in stage.options)
    for selected_count in range(stage.minimum_count, stage.maximum_count + 1):
        if stage.repetition_allowed:
            if stage.ordering_significance == OrderingSignificance.ORDERED:
                yield from itertools.product(option_ids, repeat=selected_count)
            else:
                yield from itertools.combinations_with_replacement(option_ids, selected_count)
        elif stage.ordering_significance == OrderingSignificance.ORDERED:
            yield from itertools.permutations(option_ids, selected_count)
        else:
            yield from itertools.combinations(option_ids, selected_count)


def allowed_option_ids(
    stage: ChoiceStageContract,
    prior: tuple[tuple[str, ...], ...],
) -> frozenset[str] | None:
    if not stage.conditional_options:
        return frozenset(option.option_id for option in stage.options)
    matches = [entry for entry in stage.conditional_options if entry.prior_stage_option_ids == prior]
    if len(matches) != 1:
        return None

    return frozenset(matches[0].option_ids)


def complete_action_count(stages: Sequence[ChoiceStageContract]) -> int | None:
    """Return the exact number of complete actions, or ``None`` if unresolved."""

    if not stages:
        return None
    if any(stage.ordering_significance == OrderingSignificance.UNKNOWN for stage in stages):
        return None
    if not any(stage.conditional_options for stage in stages):
        counts = [_selection_count(stage) for stage in stages]
        if any(count is None for count in counts):
            return None

        return math.prod(int(count) for count in counts)

    total = 0

    def visit(stage_index: int, prior: tuple[tuple[str, ...], ...]) -> None:
        nonlocal total
        if stage_index == len(stages):
            total += 1

            return
        stage = stages[stage_index]
        allowed = allowed_option_ids(stage, prior)
        if allowed is None:
            raise LegalActionContractError("conditional option path is absent or ambiguous")
        for selected in _stage_selections(stage):
            if set(selected).issubset(allowed):
                visit(stage_index + 1, prior + (selected,))

    try:
        visit(0, ())
    except LegalActionContractError:
        return None

    return total


def build_choice_stage(
    *,
    observation_id: str,
    stage_index: int,
    stage_name: str,
    raw_options: Sequence[Mapping[str, Any]],
    minimum_count: int,
    maximum_count: int,
    repetition_allowed: bool,
    ordering_significance: OrderingSignificance,
    conditional_options: Sequence[ConditionalOptionSet] = (),
) -> ChoiceStageContract:
    minimum = _integer(minimum_count, field="minimum_count")
    maximum = _integer(maximum_count, field="maximum_count")
    if minimum < 0 or maximum < minimum:
        raise LegalActionContractError("choice bounds are invalid")
    if not repetition_allowed and maximum > len(raw_options):
        raise LegalActionContractError("maximum_count exceeds unique option count")
    stage_id = identity("stage1", {
        "observation_id": observation_id,
        "stage_index": stage_index,
        "stage_name": stage_name,
    })
    options = []
    for position, raw_option in enumerate(raw_options):
        if not isinstance(raw_option, Mapping):
            raise LegalActionContractError("legal option must be an object")
        payload = dict(raw_option)
        payload_sha256 = sha256_json(payload)
        option_id = identity("option1", {
            "stage_id": stage_id,
            "position": position,
            "payload_sha256": payload_sha256,
        })
        options.append(CanonicalLegalOption(
            stage_id=stage_id,
            position=position,
            option_id=option_id,
            payload_sha256=payload_sha256,
            payload=payload,
        ))

    return ChoiceStageContract(
        stage_id=stage_id,
        stage_index=stage_index,
        stage_name=stage_name,
        presence=ChoicePresence.REQUIRED if minimum > 0 else ChoicePresence.OPTIONAL,
        minimum_count=minimum,
        maximum_count=maximum,
        repetition_allowed=bool(repetition_allowed),
        ordering_significance=ordering_significance,
        options=tuple(options),
        conditional_options=tuple(conditional_options),
    )


def build_cabt_legal_action_set(
    observation_id: str,
    select: Mapping[str, Any],
    *,
    ordering_significance: OrderingSignificance | None = None,
) -> CanonicalLegalActionSet:
    if not isinstance(select, Mapping):
        raise LegalActionContractError("CABT select payload must be an object")
    raw_options = select.get("option")
    if not isinstance(raw_options, list):
        raise LegalActionContractError("CABT select.option must be a list")
    minimum = _integer(select.get("minCount"), field="select.minCount")
    maximum = _integer(select.get("maxCount"), field="select.maxCount")
    select_type = _known_enum_integer(
        select.get("type"),
        SELECT_TYPE_NAMES,
        field="select.type",
    )
    select_context = _known_enum_integer(
        select.get("context"),
        SELECT_CONTEXT_NAMES,
        field="select.context",
    )
    validated_options = _validate_cabt_options(
        raw_options,
        select_type=select_type,
    )
    action_family = _enum_name(select_type, SELECT_TYPE_NAMES, prefix="SELECT")
    action_template = _enum_name(select_context, SELECT_CONTEXT_NAMES, prefix="CONTEXT")
    ordering = ordering_significance or cabt_ordering_significance(select_context, maximum)
    stage = build_choice_stage(
        observation_id=observation_id,
        stage_index=0,
        stage_name=action_template,
        raw_options=validated_options,
        minimum_count=minimum,
        maximum_count=maximum,
        repetition_allowed=False,
        ordering_significance=ordering,
    )
    failures = ()
    if ordering == OrderingSignificance.UNKNOWN:
        failures = (ValidationFailure.CHOICE_ORDER,)
    count = complete_action_count((stage,))
    payload = {
        "adapter_version": CABT_LEGAL_ACTION_ADAPTER_VERSION,
        "observation_id": observation_id,
        "action_family": action_family,
        "action_template": action_template,
        "stages": (stage,),
        "source_select_type": select_type,
        "source_select_context": select_context,
    }

    return CanonicalLegalActionSet(
        observation_id=observation_id,
        action_family=action_family,
        action_template=action_template,
        stages=(stage,),
        complete_action_count=count,
        legal_action_set_id=identity("legal1", payload),
        source_select_type=select_type,
        source_select_context=select_context,
        failures=failures,
    )


def build_multi_event_cabt_legal_action_set(
    observation_id: str,
    request_observations: Sequence[tuple[str, Mapping[str, Any]]],
    stage_selected_values: Sequence[Sequence[int] | None],
    *,
    stage_index_spaces: Sequence[str | None] | None = None,
) -> CanonicalLegalActionSet:
    """Build the exact staged contract witnessed by an explicit event chain.

    Request ``i`` supplies the legal options answered by action event ``i``.
    A later request is therefore conditional on the actual canonical choices
    made at all earlier stages.  Only the observed conditional path is known;
    unobserved sibling paths are never invented and make the global complete
    action count unresolved unless every preceding path is forced.
    """

    requests = tuple(request_observations)
    selected_values = tuple(stage_selected_values)
    index_spaces = tuple(stage_index_spaces or (None,) * len(selected_values))
    if not requests or len(requests) != len(selected_values):
        raise LegalActionContractError(
            "multi-event request and selected-stage counts must match"
        )
    if len(index_spaces) != len(selected_values):
        raise LegalActionContractError(
            "multi-event index-space and selected-stage counts must match"
        )
    if requests[0][0] != observation_id:
        raise StaleLegalSetError(
            "first multi-event request does not belong to the pre-action observation"
        )

    stages: list[ChoiceStageContract] = []
    failures: list[ValidationFailure] = []
    canonical_prior: list[tuple[str, ...]] = []
    select_types: list[int] = []
    select_contexts: list[int] = []
    path_is_proven = True
    for stage_index, (request_observation_id, select) in enumerate(requests):
        if not isinstance(request_observation_id, str) or not request_observation_id:
            raise LegalActionContractError(
                "multi-event request observation identity is absent"
            )
        if not isinstance(select, Mapping):
            raise LegalActionContractError("multi-event CABT select must be an object")
        raw_options = select.get("option")
        if not isinstance(raw_options, list):
            raise LegalActionContractError("multi-event CABT select.option must be a list")
        minimum = _integer(select.get("minCount"), field="select.minCount")
        maximum = _integer(select.get("maxCount"), field="select.maxCount")
        select_type = _known_enum_integer(
            select.get("type"),
            SELECT_TYPE_NAMES,
            field="select.type",
        )
        select_context = _known_enum_integer(
            select.get("context"),
            SELECT_CONTEXT_NAMES,
            field="select.context",
        )
        validated_options = _validate_cabt_options(
            raw_options,
            select_type=select_type,
        )
        select_types.append(select_type)
        select_contexts.append(select_context)
        stage_name = _enum_name(
            select_context,
            SELECT_CONTEXT_NAMES,
            prefix="CONTEXT",
        )
        ordering = cabt_ordering_significance(select_context, maximum)
        stage = build_choice_stage(
            observation_id=request_observation_id,
            stage_index=stage_index,
            stage_name=stage_name,
            raw_options=validated_options,
            minimum_count=minimum,
            maximum_count=maximum,
            repetition_allowed=False,
            ordering_significance=ordering,
        )
        if stage_index and path_is_proven:
            stage = replace(
                stage,
                conditional_options=(ConditionalOptionSet(
                    prior_stage_option_ids=tuple(canonical_prior),
                    option_ids=tuple(option.option_id for option in stage.options),
                ),),
            )
        elif stage_index:
            failures.append(ValidationFailure.UNRESOLVED_ACTION)
        if ordering == OrderingSignificance.UNKNOWN:
            failures.append(ValidationFailure.CHOICE_ORDER)

        stages.append(stage)
        values = selected_values[stage_index]
        index_space = index_spaces[stage_index]
        option_by_position = {option.position: option.option_id for option in stage.options}
        if (
            values is None
            or index_space not in (None, "option_position")
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        ):
            path_is_proven = False
            canonical_prior.append(())
            continue
        option_ids = tuple(option_by_position.get(int(value)) for value in values)
        if any(option_id is None for option_id in option_ids):
            path_is_proven = False
            canonical_prior.append(())
            continue
        canonical_ids = tuple(str(option_id) for option_id in option_ids)
        if ordering in {
            OrderingSignificance.NOT_APPLICABLE,
            OrderingSignificance.UNORDERED,
        }:
            canonical_ids = tuple(sorted(canonical_ids))
        canonical_prior.append(canonical_ids)

    action_families = tuple(
        _enum_name(value, SELECT_TYPE_NAMES, prefix="SELECT")
        for value in select_types
    )
    action_templates = tuple(
        _enum_name(value, SELECT_CONTEXT_NAMES, prefix="CONTEXT")
        for value in select_contexts
    )
    action_family = "MULTI_EVENT[" + ">".join(action_families) + "]"
    action_template = "MULTI_EVENT[" + ">".join(action_templates) + "]"
    exact_count = complete_action_count(stages) if path_is_proven else None
    payload = {
        "adapter_version": CABT_LEGAL_ACTION_ADAPTER_VERSION,
        "source_adapter": "explicit_multi_event_cabt.v1",
        "observation_id": observation_id,
        "request_observation_ids": tuple(value[0] for value in requests),
        "action_family": action_family,
        "action_template": action_template,
        "stages": tuple(stages),
        "source_select_types": tuple(select_types),
        "source_select_contexts": tuple(select_contexts),
    }

    return CanonicalLegalActionSet(
        observation_id=observation_id,
        action_family=action_family,
        action_template=action_template,
        stages=tuple(stages),
        complete_action_count=exact_count,
        legal_action_set_id=identity("legal1", payload),
        source_select_type=">".join(str(value) for value in select_types),
        source_select_context=">".join(str(value) for value in select_contexts),
        failures=tuple(dict.fromkeys(failures)),
    )


def build_semantic_legal_action_set(
    observation_id: str,
    legal_options: Sequence[Mapping[str, Any]],
    action_mask: Mapping[str, Any],
    *,
    actor_observation: Mapping[str, Any] | None = None,
    select: Mapping[str, Any] | None = None,
    ordering_significance: OrderingSignificance | None = None,
) -> CanonicalLegalActionSet:
    """Build the typed contract for normalized refgame decision rows.

    The source row must carry the full semantic option table and exact action
    bounds.  The actor-visible observation independently reconstructs the
    semantic table.  Display-only card names are ignored explicitly; all
    frozen core fields and their positions must match exactly.
    """

    if not isinstance(action_mask, Mapping):
        raise LegalActionContractError("semantic action_mask must be an object")
    options = tuple(legal_options)
    if not options or any(not isinstance(option, Mapping) for option in options):
        raise LegalActionContractError("semantic legal_options must be non-empty objects")
    if not isinstance(actor_observation, Mapping):
        raise LegalActionContractError("semantic actor_observation must be an object")
    minimum = _integer(action_mask.get("min_count"), field="action_mask.min_count")
    maximum = _integer(action_mask.get("max_count"), field="action_mask.max_count")
    allowed_mask_fields = frozenset({"n_options", "min_count", "max_count", "legal"})
    unsupported_mask_fields = frozenset(action_mask) - allowed_mask_fields
    if unsupported_mask_fields:
        raise LegalActionContractError(
            f"semantic action_mask has unsupported fields: {sorted(unsupported_mask_fields)}"
        )
    n_options = _integer(action_mask.get("n_options"), field="action_mask.n_options")
    if n_options != len(options):
        raise StaleLegalSetError("action_mask.n_options differs from the semantic option table")
    raw_mask = action_mask.get("legal")
    if raw_mask is not None:
        if (
            not isinstance(raw_mask, list)
            or len(raw_mask) < n_options
            or any(not isinstance(value, bool) for value in raw_mask)
        ):
            raise LegalActionContractError("action_mask.legal must be a boolean list")
        expected_mask = [True] * n_options + [False] * (len(raw_mask) - n_options)
        if raw_mask != expected_mask:
            raise StaleLegalSetError(
                "action_mask.legal contradicts the semantic option table"
            )
    observation_select = actor_observation.get("select")
    if not isinstance(observation_select, Mapping):
        raise LegalActionContractError("actor_observation.select must be an object")
    raw_select = observation_select if select is None else select
    if not isinstance(raw_select, Mapping):
        raise LegalActionContractError("CABT select payload must be an object")
    raw_minimum = _integer(raw_select.get("minCount"), field="select.minCount")
    raw_maximum = _integer(raw_select.get("maxCount"), field="select.maxCount")
    if raw_minimum != minimum or raw_maximum != maximum:
        raise StaleLegalSetError("semantic bounds differ from the source observation")
    expected_options = build_refgame_semantic_legal_options(
        actor_observation,
        select=raw_select,
    )
    verified_options = _verified_semantic_options(options, expected_options)
    raw_type = _known_enum_integer(
        raw_select.get("type"),
        SELECT_TYPE_NAMES,
        field="select.type",
    )
    raw_context = _known_enum_integer(
        raw_select.get("context"),
        SELECT_CONTEXT_NAMES,
        field="select.context",
    )
    option_types = {str(option["option_type"]) for option in verified_options}
    action_family = next(iter(option_types)) if len(option_types) == 1 else "MIXED"
    action_template = _enum_name(raw_context, SELECT_CONTEXT_NAMES, prefix="CONTEXT")
    ordering = ordering_significance or semantic_ordering_significance(action_template, maximum)
    stage = build_choice_stage(
        observation_id=observation_id,
        stage_index=0,
        stage_name=action_template,
        raw_options=verified_options,
        minimum_count=minimum,
        maximum_count=maximum,
        repetition_allowed=False,
        ordering_significance=ordering,
    )
    failures = ()
    if ordering == OrderingSignificance.UNKNOWN:
        failures = (ValidationFailure.CHOICE_ORDER,)
    payload = {
        "adapter_version": CABT_LEGAL_ACTION_ADAPTER_VERSION,
        "source_adapter": REFGAME_SEMANTIC_CROSSWALK_VERSION,
        "observation_id": observation_id,
        "action_family": action_family,
        "action_template": action_template,
        "stages": (stage,),
        "source_select_type": raw_type,
        "source_select_context": raw_context,
    }

    return CanonicalLegalActionSet(
        observation_id=observation_id,
        action_family=action_family,
        action_template=action_template,
        stages=(stage,),
        complete_action_count=complete_action_count((stage,)),
        legal_action_set_id=identity("legal1", payload),
        source_select_type=payload["source_select_type"],
        source_select_context=payload["source_select_context"],
        failures=failures,
    )


__all__ = [
    "CABT_LEGAL_ACTION_ADAPTER_VERSION",
    "REFGAME_CORE_SEMANTIC_FIELDS",
    "REFGAME_DISPLAY_ONLY_SEMANTIC_FIELDS",
    "REFGAME_SEMANTIC_CROSSWALK_VERSION",
    "LegalActionContractError",
    "StaleLegalSetError",
    "allowed_option_ids",
    "build_cabt_legal_action_set",
    "build_choice_stage",
    "build_multi_event_cabt_legal_action_set",
    "build_refgame_semantic_legal_options",
    "build_semantic_legal_action_set",
    "cabt_ordering_significance",
    "complete_action_count",
    "exact_json_value_equal",
    "semantic_ordering_significance",
]
