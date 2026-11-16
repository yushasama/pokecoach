"""Leakage-safe sequence views over public prospective trace rows."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable

from deepmew.replay.prospective_trace import (
    PUBLIC_PROJECTION_VERSION,
    PUBLIC_ROW_SCHEMA_VERSION,
    assert_public_trace_row_safe,
)


SEQUENCE_SCHEMA_VERSION = "deepmew.temporal_public_sequence.v1"
SPLIT_SCHEMA_VERSION = "deepmew.temporal_public_split.v1"

MODEL_INPUT_FIELDS: tuple[str, ...] = (
    "public_observation_before",
    "previous_public_events",
    "focal_actor_seat",
    "controller_seat",
    "controller_is_focal",
    "phase",
    "turn",
    "elapsed_decisions",
)
LABEL_FIELDS: tuple[str, ...] = ("terminal_actor_win_probability",)
AUDIT_ONLY_SOURCE_FIELDS: tuple[str, ...] = (
    "public_observation_after",
    "public_chosen_action",
    "public_event_delta",
    "end_of_actor_turn_public_observation",
    "next_same_actor_public_observation",
    "result",
    "result_reason",
    "game_completed_at_utc",
    "rating_snapshot",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _record_seat_identity(mapping: dict[int, Any], seat: int, value: Any, *, field: str) -> None:
    normalized = copy.deepcopy(value)
    if seat in mapping and mapping[seat] != normalized:
        raise ValueError(f"{field} changed within game for seat {seat}")
    mapping[seat] = normalized


def actor_relative_terminal_label(result: Any, focal_actor_seat: int) -> float:
    """Map CABT's terminal winner index to a focal-actor probability label."""
    terminal = _as_int(result)
    if focal_actor_seat not in (0, 1):
        raise ValueError(f"focal actor seat must be 0 or 1, got {focal_actor_seat}")
    if terminal == 2:
        return 0.5
    if terminal not in (0, 1):
        raise ValueError(f"prospective sequence requires terminal result 0, 1, or 2; got {result!r}")

    return float(terminal == focal_actor_seat)


def _ordered_game_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((copy.deepcopy(row) for row in rows), key=lambda row: int(row["decision_index"]))
    if not ordered:
        raise ValueError("cannot build a temporal sequence from zero rows")
    game_ids = {str(row.get("game_id") or "") for row in ordered}
    if len(game_ids) != 1 or "" in game_ids:
        raise ValueError(f"temporal sequence rows must belong to one game: {sorted(game_ids)}")
    indices = [_as_int(row.get("decision_index")) for row in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError(f"decision_index must be contiguous from zero, got {indices[:10]}")
    decision_ids = [str(row.get("decision_id") or "") for row in ordered]
    if "" in decision_ids or len(set(decision_ids)) != len(decision_ids):
        raise ValueError("temporal sequence has missing or duplicate decision IDs")
    tranches = {
        (_as_int(row.get("tranche_sequence")), str(row.get("tranche_id") or ""))
        for row in ordered
    }
    if len(tranches) != 1:
        raise ValueError(f"one game cannot cross prospective tranches: {sorted(tranches)}")
    results = {_as_int(row.get("result")) for row in ordered}
    if len(results) != 1:
        raise ValueError(f"terminal result changed within game rows: {sorted(results)}")
    terminal_result = next(iter(results))
    if terminal_result not in (0, 1, 2):
        raise ValueError(
            f"temporal sequence requires a completed terminal result 0, 1, or 2; "
            f"got {terminal_result!r}"
        )
    for row in ordered:
        if row.get("schema_version") != PUBLIC_ROW_SCHEMA_VERSION:
            raise ValueError(f"temporal sequence requires {PUBLIC_ROW_SCHEMA_VERSION}")
        if row.get("information_mode") != "public_belief":
            raise ValueError("temporal sequence requires public_belief information_mode")
        if row.get("public_projection_version") != PUBLIC_PROJECTION_VERSION:
            raise ValueError(f"temporal sequence requires {PUBLIC_PROJECTION_VERSION}")
        for safety_field in ("crashed", "timed_out", "illegal"):
            if row.get(safety_field) is not False:
                raise ValueError(
                    f"temporal sequence requires explicit {safety_field}=false on every row"
                )
        if not isinstance(row.get("game_completed_at_utc"), str) or not row["game_completed_at_utc"].strip():
            raise ValueError("temporal sequence requires an explicit game completion timestamp")
        assert_public_trace_row_safe(row)
    if ordered[-1].get("terminated") is not True:
        raise ValueError("temporal sequence requires the final decision to carry terminated=true")

    return ordered


def build_focal_actor_sequence(
    rows: Iterable[dict[str, Any]],
    *,
    focal_actor_seat: int,
) -> dict[str, Any]:
    """Build one causal public-history sequence from a fixed actor perspective."""
    ordered = _ordered_game_rows(rows)
    terminal_label = actor_relative_terminal_label(ordered[-1]["result"], focal_actor_seat)
    first = ordered[0]
    policy_by_seat: dict[int, Any] = {}
    deck_by_seat: dict[int, Any] = {}
    for row in ordered:
        actor_seat = int(row["actor_seat"])
        opponent_seat = 1 - actor_seat
        _record_seat_identity(policy_by_seat, actor_seat, row.get("actor_policy"), field="policy")
        _record_seat_identity(policy_by_seat, opponent_seat, row.get("opponent_policy"), field="policy")
        _record_seat_identity(deck_by_seat, actor_seat, row.get("actor_deck"), field="deck")
        _record_seat_identity(deck_by_seat, opponent_seat, row.get("opponent_deck"), field="deck")
    previous_events: list[dict[str, Any]] = []
    steps = []
    for elapsed, row in enumerate(ordered):
        controller = _as_int(row.get("actor_seat"))
        model_input = {
            "public_observation_before": copy.deepcopy(_as_dict(row.get("public_observation_before"))),
            "previous_public_events": copy.deepcopy(previous_events),
            "focal_actor_seat": focal_actor_seat,
            "controller_seat": controller,
            "controller_is_focal": controller == focal_actor_seat,
            "phase": str(row.get("phase") or "UNKNOWN"),
            "turn": _as_int(row.get("turn"), 0),
            "elapsed_decisions": elapsed,
        }
        if tuple(model_input) != MODEL_INPUT_FIELDS:
            raise RuntimeError("temporal model-input field contract drifted")
        steps.append({
            "decision_index": elapsed,
            "source_decision_id": str(row["decision_id"]),
            "input": model_input,
            "label": {"terminal_actor_win_probability": terminal_label},
        })
        previous_events = copy.deepcopy(_as_list(row.get("public_event_delta")))

    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "public_projection_version": PUBLIC_PROJECTION_VERSION,
        "sequence_id": f"{first['game_id']}:focal-{focal_actor_seat}",
        "game_id": str(first["game_id"]),
        "focal_actor_seat": focal_actor_seat,
        "tranche_id": str(first["tranche_id"]),
        "tranche_sequence": _as_int(first["tranche_sequence"]),
        "league_snapshot_id": _as_dict(first.get("generation_provenance")).get("league_snapshot_id"),
        "focal_policy": copy.deepcopy(policy_by_seat.get(focal_actor_seat)),
        "opponent_policy": copy.deepcopy(policy_by_seat.get(1 - focal_actor_seat)),
        "focal_deck": copy.deepcopy(deck_by_seat.get(focal_actor_seat)),
        "opponent_deck": copy.deepcopy(deck_by_seat.get(1 - focal_actor_seat)),
        "steps": steps,
    }


def build_temporal_sequences(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build two fixed-perspective causal sequences per complete game."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        game_id = str(row.get("game_id") or "")
        if not game_id:
            raise ValueError("prospective row is missing game_id")
        grouped[game_id].append(row)
    sequences = []
    for game_id in sorted(grouped):
        for focal_actor_seat in (0, 1):
            sequences.append(build_focal_actor_sequence(
                grouped[game_id],
                focal_actor_seat=focal_actor_seat,
            ))

    return sequences


def split_sequences_by_tranche(
    sequences: Iterable[dict[str, Any]],
    *,
    train_tranches: set[str],
    validation_tranches: set[str],
    test_tranches: set[str],
) -> dict[str, Any]:
    """Assign whole games to explicit tranche-based splits and reject overlap."""
    split_sets = {
        "train": set(train_tranches),
        "validation": set(validation_tranches),
        "test": set(test_tranches),
    }
    names = list(split_sets)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            overlap = split_sets[name].intersection(split_sets[other])
            if overlap:
                raise ValueError(f"tranche split overlap between {name} and {other}: {sorted(overlap)}")
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    game_split: dict[str, str] = {}
    for sequence in sequences:
        tranche = str(sequence.get("tranche_id") or "")
        matches = [name for name, values in split_sets.items() if tranche in values]
        if len(matches) != 1:
            raise ValueError(f"tranche {tranche!r} must map to exactly one split")
        split = matches[0]
        game_id = str(sequence["game_id"])
        prior = game_split.setdefault(game_id, split)
        if prior != split:
            raise ValueError(f"game {game_id} crossed sequence splits")
        outputs[split].append(copy.deepcopy(sequence))

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "tranches": {name: sorted(values) for name, values in split_sets.items()},
        "counts": {name: len(values) for name, values in outputs.items()},
        "sequences": outputs,
    }
