"""Schema objects for deterministic decision traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


DECISION_FRAME_SCHEMA_VERSION = "deepmew.decision_frame.v1"
OPTION_SUMMARY_SCHEMA_VERSION = "deepmew.option_summary.v1"
POLICY_PROBE_SCHEMA_VERSION = "deepmew.policy_probe.v1"
POLICY_DIFF_SCHEMA_VERSION = "deepmew.policy_diff.v1"


class TraceSchemaError(ValueError):
    """Raised when a trace payload uses an unsupported schema."""


@dataclass
class OptionSummary:
    index: int
    option_type: str
    summary: str
    attack_id: int | None = None
    card_id: int | None = None
    card_name: str | None = None
    area: str | None = None
    target: str | None = None
    attack_name: str | None = None
    effective_damage: int | None = None
    printed_damage: int | None = None
    resolved: bool | None = None
    resolve_reason: str | None = None
    source_area: str | None = None
    source_index: int | None = None
    target_area: str | None = None
    target_index: int | None = None
    target_card_id: int | None = None
    target_card_name: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class DecisionFrame:
    schema_version: str
    run_id: str
    game_id: str
    frame_id: str
    seed: int | None
    seat: int | None
    turn: int | None
    decision_index: int
    agent_spec: str
    agent_label: str
    policy_version: str | None
    deck_id: str | None
    deck_path: str | None
    select_type: str
    select_context: str
    min_count: int
    max_count: int
    legal_options: list[OptionSummary]
    chosen_indices: list[int]
    chosen_option_summaries: list[OptionSummary]
    board_summary: dict[str, Any]
    final_result: int | None
    final_result_reason: str | None
    raw_observation: Any
    notes: list[str]
    information_mode: str = "public_belief"

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class PolicyProbe:
    schema_version: str
    frame_id: str
    policy_spec: str
    chosen_indices: list[int]
    chosen_option_summaries: list[OptionSummary]
    valid_return: bool
    error: str | None
    notes: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class PolicyDiff:
    schema_version: str
    frame_id: str
    base_policy: str
    compare_policy: str
    same_choice: bool
    base_choice: list[int]
    compare_choice: list[int]
    diff_tags: list[str]
    notes: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def require_schema(payload: dict[str, Any], expected_schema: str) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    if schema_version != expected_schema:
        raise TraceSchemaError(f"unsupported schema_version {schema_version!r}, expected {expected_schema!r}")
    return payload


def validate_frame_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return require_schema(payload, DECISION_FRAME_SCHEMA_VERSION)


def validate_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return require_schema(payload, POLICY_PROBE_SCHEMA_VERSION)


def validate_diff_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return require_schema(payload, POLICY_DIFF_SCHEMA_VERSION)
