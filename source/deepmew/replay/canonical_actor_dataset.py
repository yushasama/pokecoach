"""Canonical actor-visible dataset schemas for the value/Q experiment.

Two logical schemas over one physical layout:

- ``deepmew.temporal_actor_sequence.v1``: per (game, seat) causal decision
  sequences. Rows inside one block are one actor's decisions in order; the
  pre-action actor-visible observation is represented by the canonical
  feature vector; the current chosen action is NEVER part of the V input.
- ``deepmew.actor_transition.v1``: one explicit transition sidecar record per
  decision. Numeric before/after state representations live in the NPZ part;
  exact chosen/legal actions and content-addressed source references live in
  the sidecar. The next-same-actor link is explicit and never fabricated.
- ``deepmew.action_option.v1``: one exact source option per legal action where
  the source actually persisted a legal set.

Physical layout: compressed NPZ parts (feature matrix + per-row metadata)
plus a games index (one record per game-seat block) plus a fail-closed unit
manifest. Raw observations stay in the immutable source shards; every part
is bound to its source by hash in the manifest.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from deepmew.replay.information_boundary import (
    InformationMode,
    UNKNOWN_VISIBILITY,
    classify_decision_row_visibility,
)


SEQUENCE_SCHEMA_VERSION = "deepmew.temporal_actor_sequence.v1"
TRANSITION_SCHEMA_VERSION = "deepmew.actor_transition.v1"
ACTION_OPTION_SCHEMA_VERSION = "deepmew.action_option.v1"
UNIT_MANIFEST_SCHEMA_VERSION = "deepmew.value_q_conversion_unit.v1"
CATALOG_SCHEMA_VERSION = "deepmew.value_q_dataset_catalog.v1"

# Feature layout of the canonical per-decision vector (float32).
FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("r15_state_v2", 49),   # scripts/run_unified_r3_value._features (includes second-order 14)
    ("encode_base", 84),    # scripts/run_unified_r4_ablation.encode_base (ctx/options/scalars)
    ("hand_composition", 11),  # own-hand cardType counts (7) + basic/evolution/ex + hand size
)
FEATURE_DIM = sum(width for _, width in FEATURE_BLOCKS)
S1_SLICE = (0, 49)          # existing ValueMLP feature prefix
S2_SLICE = (0, FEATURE_DIM)  # richer actor-visible snapshot

# Persisted CABT rows carry opaque/debug surfaces that are not certified as
# causal model inputs. In particular, prior prospective-v1 work found private
# identities in logs/looking. Canonical features are built only after these
# surfaces are removed; the raw source is referenced by hash, never copied.
OPAQUE_OBSERVATION_FIELDS = ("logs", "search_begin_input")
OPAQUE_CURRENT_FIELDS = ("looking",)
OPAQUE_SELECT_FIELDS = ("deck",)


class QuarantineReason:
    NOT_ACTOR_VISIBLE = "not_actor_visible"
    UNKNOWN_VISIBILITY = "unknown_visibility"
    PRIVILEGED = "privileged_oracle_content"
    SAFETY = "safety_flagged_game"
    NO_TERMINAL = "no_terminal_result"
    LABEL_INCONSISTENT = "label_inconsistent_across_rows"
    NOT_COMPLEMENTARY = "seat_labels_not_complementary"
    ORDER_BROKEN = "decision_order_broken"
    PARSE_ERROR = "parse_error"
    MISSING_FIELDS = "missing_required_fields"
    INVALID_ACTOR = "invalid_actor"
    ACTION_CONTRACT = "invalid_action_contract"
    DUPLICATE_GAME = "duplicate_game"


def canonical_actor_feature_observation(observation: Any) -> dict[str, Any]:
    """Project a validated actor observation onto the certified feature surface."""
    if not isinstance(observation, dict):
        raise ValueError("canonical feature observation must be an object")
    projected = copy.deepcopy(observation)
    for field in OPAQUE_OBSERVATION_FIELDS:
        projected.pop(field, None)
    current = projected.get("current")
    if not isinstance(current, dict):
        raise ValueError("canonical feature observation requires current")
    for field in OPAQUE_CURRENT_FIELDS:
        current.pop(field, None)
    select = projected.get("select")
    if isinstance(select, dict):
        for field in OPAQUE_SELECT_FIELDS:
            select.pop(field, None)

    return projected


def refgame_actor_label(outcome: Any) -> float | None:
    """Actor-relative terminal label from a refgame ``outcome`` dict."""
    if not isinstance(outcome, dict):
        return None
    flags = [bool(outcome.get(name)) for name in ("won", "lost", "draw")]
    if sum(flags) != 1:
        return None
    if flags[2]:
        return 0.5
    if flags[0]:
        return 1.0
    if flags[1]:
        return 0.0

    return None


def refgame_safety_flagged(outcome: Any) -> bool:
    if not isinstance(outcome, dict):
        return True

    safety_fields = (
        "crashed",
        "timed_out",
        "illegal_action",
        "illegal_actions",
        "safety_failed",
    )

    return any(bool(outcome.get(field)) for field in safety_fields)


def kaggle_actor_label(result: Any, player_id: Any) -> float | None:
    """Actor-relative label from a Kaggle winner-index ``result`` field."""
    try:
        terminal = int(result)
        seat = int(player_id)
    except (TypeError, ValueError):
        return None
    if seat not in (0, 1):
        return None
    if terminal == 2:
        return 0.5
    if terminal not in (0, 1):
        return None

    return float(terminal == seat)


@dataclass
class GameCheckResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_seat_sequences(rows_by_seat: dict[int, list[dict[str, Any]]]) -> GameCheckResult:
    """Integrity-check one game's per-seat row lists (already label-mapped).

    Each row dict must carry ``decision_index`` (int) and ``z`` (float label).
    Checks: per-seat strictly increasing decision order, per-seat label
    consistency, and cross-seat complementarity when both seats are present.
    """
    reasons: list[str] = []
    labels: dict[int, float] = {}
    for seat, rows in rows_by_seat.items():
        if seat not in (0, 1):
            reasons.append(QuarantineReason.INVALID_ACTOR)
        if not rows:
            reasons.append(QuarantineReason.MISSING_FIELDS)
            continue
        indices = [row["decision_index"] for row in rows]
        if any(not isinstance(index, int) for index in indices) or any(
            b <= a for a, b in zip(indices, indices[1:])
        ):
            reasons.append(QuarantineReason.ORDER_BROKEN)
        seat_labels = {row["z"] for row in rows}
        if len(seat_labels) != 1:
            reasons.append(QuarantineReason.LABEL_INCONSISTENT)
        else:
            labels[seat] = next(iter(seat_labels))
    if len(labels) == 2:
        z0, z1 = labels.get(0), labels.get(1)
        if z0 is None or z1 is None or abs((z0 + z1) - 1.0) > 1e-9:
            reasons.append(QuarantineReason.NOT_COMPLEMENTARY)

    return GameCheckResult(ok=not reasons, reasons=sorted(set(reasons)))


def validate_action_contract(chosen_action: Any, legal_actions: Any) -> bool:
    """Return whether a persisted action contract is exact and self-consistent."""
    if not isinstance(chosen_action, dict) or not isinstance(legal_actions, list) or not legal_actions:
        return False
    indices = chosen_action.get("indices")
    if not isinstance(indices, list):
        return False
    if not indices and chosen_action.get("allows_empty_choice") is not True:
        return False
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        return False
    if chosen_action.get("index_space") == "option_position":
        legal_indices = set(range(len(legal_actions)))
    else:
        legal_indices = {
            option.get("index", position) if isinstance(option, dict) else None
            for position, option in enumerate(legal_actions)
        }
    if None in legal_indices or not set(indices).issubset(legal_indices):
        return False

    return all(isinstance(option, dict) for option in legal_actions)


def validate_transition_record(record: dict[str, Any]) -> None:
    """Fail closed on the stable, source-independent transition envelope."""
    required = (
        "schema_version",
        "canonical_game_id",
        "decision_id",
        "actor",
        "global_decision_index",
        "actor_decision_index",
        "state_before",
        "chosen_action",
        "legal_action_set",
        "next_same_actor_state",
        "transition_horizon_logical_decisions",
        "terminal",
        "terminal_actor_relative_result",
        "immediate_reward",
        "information_mode",
        "source_hash",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"transition missing required fields: {missing}")
    if record["schema_version"] != TRANSITION_SCHEMA_VERSION:
        raise ValueError(f"unexpected transition schema {record['schema_version']}")
    if record["information_mode"] != InformationMode.ACTOR_VISIBLE.value:
        raise ValueError("deployable transition must be actor_visible")
    if record["actor"] not in (0, 1):
        raise ValueError("transition actor must be seat 0 or 1")
    if not validate_action_contract(record["chosen_action"], record["legal_action_set"]):
        raise ValueError("transition action contract is invalid")
    terminal = bool(record["terminal"])
    next_state = record["next_same_actor_state"]
    horizon = record["transition_horizon_logical_decisions"]
    if terminal and (next_state is not None or horizon is not None):
        raise ValueError("terminal transition cannot link a next-same-actor state")
    if not terminal and (not isinstance(next_state, dict) or not isinstance(horizon, int) or horizon <= 0):
        raise ValueError("nonterminal transition requires a positive next-same-actor link")


def validate_sequence_record(record: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "canonical_game_id",
        "decision_id",
        "information_mode",
        "source_id",
        "source_schema",
        "source_game_id",
        "actor",
        "actor_decision_index",
        "global_decision_index",
        "state_before",
        "terminal_actor_relative_result",
        "game_completion_status",
        "safety_status",
        "source_hash",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"sequence row missing required fields: {missing}")
    if record["schema_version"] != SEQUENCE_SCHEMA_VERSION:
        raise ValueError(f"unexpected sequence schema {record['schema_version']}")
    if record["information_mode"] != InformationMode.ACTOR_VISIBLE.value:
        raise ValueError("deployable sequence row must be actor_visible")
    if record["actor"] not in (0, 1):
        raise ValueError("sequence actor must be seat 0 or 1")
    if record["game_completion_status"] != "completed" or record["safety_status"] != "clean":
        raise ValueError("canonical deployable sequence requires completed safety-clean game")


def validate_action_option_record(record: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "canonical_game_id",
        "decision_id",
        "actor",
        "global_decision_index",
        "actor_decision_index",
        "option_position",
        "source_option_index",
        "option",
        "is_chosen",
        "information_mode",
        "source_hash",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"action option missing required fields: {missing}")
    if record["schema_version"] != ACTION_OPTION_SCHEMA_VERSION:
        raise ValueError(f"unexpected action option schema {record['schema_version']}")
    if record["information_mode"] != InformationMode.ACTOR_VISIBLE.value:
        raise ValueError("deployable action option must be actor_visible")
    if record["actor"] not in (0, 1) or not isinstance(record["option"], dict):
        raise ValueError("action option actor/option is invalid")


def deployable_row_visibility(row: dict[str, Any]) -> tuple[bool, str]:
    """Fail-closed visibility gate for one persisted decision row.

    Returns (accepted, classification). Only actor_visible rows enter the
    canonical deployable dataset; strict-public rows would lack legal actions
    and are handled by the separate prospective lane.
    """
    classification = classify_decision_row_visibility(row)
    if classification == InformationMode.ACTOR_VISIBLE.value:
        return True, classification

    return False, classification


def quarantine_reason_for_visibility(classification: str) -> str:
    if classification == InformationMode.PRIVILEGED_ORACLE.value:
        return QuarantineReason.PRIVILEGED
    if classification == UNKNOWN_VISIBILITY:
        return QuarantineReason.UNKNOWN_VISIBILITY

    return QuarantineReason.NOT_ACTOR_VISIBLE
