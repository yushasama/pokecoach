"""Training-artifact information boundary for public and oracle observations."""

from __future__ import annotations

import copy
import json
from dataclasses import InitVar, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "deepmew.information_boundary.v1"
MODE_CONTRACT_VERSION = "deepmew.information_mode_contract.v2"


class InformationMode(str, Enum):
    PUBLIC_BELIEF = "public_belief"
    ACTOR_VISIBLE = "actor_visible"
    PRIVILEGED_ORACLE = "privileged_oracle"
    PRIVILEGED_SIMULATOR_TEACHER = "privileged_simulator_teacher"


# Classification outcome for observations whose visibility cannot be proven.
# It is deliberately NOT an InformationMode: unproven rows are quarantined,
# never labeled with a trainable mode.
UNKNOWN_VISIBILITY = "unknown_visibility"

# Modes a deployable model may train on. Privileged rows must never reach a
# deployable trainer.
DEPLOYABLE_MODES = frozenset({InformationMode.PUBLIC_BELIEF.value, InformationMode.ACTOR_VISIBLE.value})
PRIVILEGED_MODES = frozenset(
    {
        InformationMode.PRIVILEGED_ORACLE.value,
        InformationMode.PRIVILEGED_SIMULATOR_TEACHER.value,
    }
)

_ISOLATED_TEACHER_TOKEN = object()


@dataclass(frozen=True)
class InformationBoundary:
    mode: InformationMode = InformationMode.PUBLIC_BELIEF
    _isolated_teacher_token: InitVar[object | None] = None

    def __post_init__(self, _isolated_teacher_token: object | None) -> None:
        if (
            self.mode is InformationMode.PRIVILEGED_SIMULATOR_TEACHER
            and _isolated_teacher_token is not _ISOLATED_TEACHER_TOKEN
        ):
            raise ValueError(
                "privileged_simulator_teacher requires the explicit isolated-teacher constructor"
            )

    @classmethod
    def from_reveal_flag(cls, reveal_opponent_private_state: bool) -> "InformationBoundary":
        mode = InformationMode.PRIVILEGED_ORACLE if reveal_opponent_private_state else InformationMode.PUBLIC_BELIEF

        return cls(mode=mode)

    @classmethod
    def for_isolated_simulator_teacher(cls) -> "InformationBoundary":
        """Construct the research-only boundary used by an isolated branch teacher."""

        return cls(
            mode=InformationMode.PRIVILEGED_SIMULATOR_TEACHER,
            _isolated_teacher_token=_ISOLATED_TEACHER_TOKEN,
        )

    @property
    def reveals_opponent_private_state(self) -> bool:
        return self.mode is InformationMode.PRIVILEGED_ORACLE

    @property
    def is_privileged_simulator_teacher(self) -> bool:
        return self.mode is InformationMode.PRIVILEGED_SIMULATOR_TEACHER

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "information_mode": self.mode.value,
            "reveals_opponent_private_state": self.reveals_opponent_private_state,
        }
        if self.is_privileged_simulator_teacher:
            metadata.update(
                {
                    "mode_contract_version": MODE_CONTRACT_VERSION,
                    "isolated_research_only": True,
                    "labels_generated_by_privileged_teacher": True,
                    "teacher_target_semantics": "oracle_world_supervision",
                    "student_information_mode": InformationMode.ACTOR_VISIBLE.value,
                    "accesses_privileged_simulator_state": True,
                    "hidden_state_persistence_allowed": False,
                }
            )

        return metadata

    def artifact_observation(
        self,
        agent_observation: dict[str, Any],
        *,
        visualizer_payload: str | None = None,
    ) -> dict[str, Any]:
        public = redact_public_observation(agent_observation)
        if self.is_privileged_simulator_teacher:
            if visualizer_payload is not None:
                raise ValueError(
                    "privileged_simulator_teacher artifact projection forbids private visualizer payloads"
                )
            assert_deployable_observation(
                public,
                expected_mode=InformationMode.ACTOR_VISIBLE.value,
            )

            return public
        if not self.reveals_opponent_private_state:
            return public
        if visualizer_payload is None:
            raise ValueError("privileged_oracle observation requires an explicit visualizer payload")
        oracle = merge_visualizer_private_state(public, visualizer_payload)
        if "search_begin_input" in agent_observation:
            oracle["search_begin_input"] = copy.deepcopy(agent_observation["search_begin_input"])

        return oracle


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def redact_public_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Return an actor observation with non-CABT private extensions removed."""
    redacted = copy.deepcopy(observation)
    redacted.pop("search_begin_input", None)
    current = _as_dict(redacted.get("current"))
    players = _as_list(current.get("players"))
    actor = current.get("yourIndex")
    try:
        actor_index = int(actor)
    except (TypeError, ValueError):
        actor_index = None
    for index, player_value in enumerate(players):
        player = _as_dict(player_value)
        player.pop("deck", None)
        player.pop("deckOrder", None)
        player.pop("trueDeckOrder", None)
        if actor_index is None or index != actor_index:
            player["hand"] = None

    return redacted


def _latest_visualizer_observation(payload: str) -> dict[str, Any]:
    decoded = json.loads(payload)
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("visualizer payload must be a non-empty observation list")
    latest = decoded[-1]
    if not isinstance(latest, dict):
        raise ValueError("latest visualizer observation must be an object")

    return latest


def merge_visualizer_private_state(
    public_observation: dict[str, Any],
    visualizer_payload: str,
) -> dict[str, Any]:
    """Merge only private hand/deck zones from CABT VisualizeData."""
    merged = copy.deepcopy(public_observation)
    visualizer = _latest_visualizer_observation(visualizer_payload)
    public_players = _as_list(_as_dict(merged.get("current")).get("players"))
    private_players = _as_list(_as_dict(visualizer.get("current")).get("players"))
    if len(public_players) != len(private_players):
        raise ValueError("visualizer player count does not match agent observation")
    for public_value, private_value in zip(public_players, private_players):
        public_player = _as_dict(public_value)
        private_player = _as_dict(private_value)
        if "hand" in private_player:
            public_player["hand"] = copy.deepcopy(private_player["hand"])
        if "deck" in private_player:
            public_player["deck"] = copy.deepcopy(private_player["deck"])

    return merged


def require_oracle_output_path(path: str | Path, boundary: InformationBoundary) -> None:
    if boundary.is_privileged_simulator_teacher:
        raise ValueError("privileged_simulator_teacher cannot persist hidden-state observations")
    if boundary.reveals_opponent_private_state and "oracle" not in str(path).lower():
        raise ValueError("privileged_oracle output path must contain 'oracle'")


def reject_privileged_rows(rows: list[dict[str, Any]], *, allow_privileged: bool) -> None:
    teacher = sum(
        row.get("information_mode") == InformationMode.PRIVILEGED_SIMULATOR_TEACHER.value for row in rows
    )
    privileged = sum(row.get("information_mode") in PRIVILEGED_MODES for row in rows)
    if teacher:
        raise ValueError(
            f"public trainer rejected {privileged} privileged rows, including {teacher} "
            "privileged_simulator_teacher rows; isolated-teacher rows cannot enter ordinary replay trainers"
        )
    if privileged and not allow_privileged:
        raise ValueError(
            f"public trainer rejected {privileged} privileged rows; "
            "use an explicit privileged-training override only for oracle research"
        )


_PRIVATE_DECK_FIELDS = ("deck", "deckOrder", "trueDeckOrder")

# Severity order used when merging classifications across multiple persisted
# observations of one decision row (pre-action, post-action, ...).
_VISIBILITY_SEVERITY = {
    InformationMode.PUBLIC_BELIEF.value: 0,
    InformationMode.ACTOR_VISIBLE.value: 1,
    UNKNOWN_VISIBILITY: 2,
    InformationMode.PRIVILEGED_ORACLE.value: 3,
    InformationMode.PRIVILEGED_SIMULATOR_TEACHER.value: 4,
}


def classify_observation_visibility(observation: Any) -> str:
    """Classify a persisted observation's true visibility, fail-closed.

    Returns one of the InformationMode values or UNKNOWN_VISIBILITY. Anything
    not provably deployable is escalated: opponent hand content or any deck
    ordering field -> privileged_oracle; malformed structure or a non-null
    prize identity (hidden from both players in CABT unless revealed through
    a channel we cannot prove public) -> unknown_visibility.
    """
    if not isinstance(observation, dict):
        return UNKNOWN_VISIBILITY
    if observation.get("search_begin_input") is not None:
        return InformationMode.PRIVILEGED_ORACLE.value
    current = _as_dict(observation.get("current"))
    players = current.get("players")
    if not isinstance(players, list) or not players:
        return UNKNOWN_VISIBILITY
    try:
        actor_index = int(current.get("yourIndex"))
    except (TypeError, ValueError):
        return UNKNOWN_VISIBILITY
    if actor_index < 0 or actor_index >= len(players):
        return UNKNOWN_VISIBILITY

    privileged = False
    actor_hand_visible = False
    for index, player_value in enumerate(players):
        if not isinstance(player_value, dict):
            return UNKNOWN_VISIBILITY
        for field in _PRIVATE_DECK_FIELDS:
            if player_value.get(field) is not None:
                privileged = True
        prize = player_value.get("prize")
        if isinstance(prize, list) and any(entry is not None for entry in prize):
            return UNKNOWN_VISIBILITY
        hand = player_value.get("hand")
        if index == actor_index:
            if isinstance(hand, list):
                actor_hand_visible = True
            elif hand is not None:
                return UNKNOWN_VISIBILITY
        else:
            if isinstance(hand, list):
                privileged = True
            elif hand is not None:
                return UNKNOWN_VISIBILITY

    if privileged:
        return InformationMode.PRIVILEGED_ORACLE.value
    if actor_hand_visible:
        return InformationMode.ACTOR_VISIBLE.value

    return InformationMode.PUBLIC_BELIEF.value


def classify_decision_row_visibility(
    row: dict[str, Any],
    *,
    observation_fields: tuple[str, ...] = ("raw_observation", "state_json", "post_observation"),
) -> str:
    """Classify a persisted decision row across all its stored observations.

    An explicit privileged stamp always wins (never downgrade a privileged
    artifact). A deployable stamp never launders content: the observation-level
    classification is authoritative, so legacy rows stamped `public_belief` by
    the v1 boundary reclassify to actor_visible when they retain the actor's
    own hand, and any opponent-private content escalates regardless of stamp.
    """
    stamped = row.get("information_mode")
    if stamped in PRIVILEGED_MODES:
        return str(stamped)
    if stamped is not None and stamped not in DEPLOYABLE_MODES:
        return UNKNOWN_VISIBILITY

    observed: str | None = None
    for field in observation_fields:
        value = row.get(field)
        if value is None:
            continue
        classification = classify_observation_visibility(value)
        if observed is None or _VISIBILITY_SEVERITY[classification] > _VISIBILITY_SEVERITY[observed]:
            observed = classification
    if observed is None:
        return UNKNOWN_VISIBILITY

    return observed


def assert_deployable_observation(observation: Any, *, expected_mode: str | None = None) -> str:
    """Fail-closed gate for observations entering a deployable dataset."""
    classification = classify_observation_visibility(observation)
    if classification not in DEPLOYABLE_MODES:
        raise ValueError(f"observation classified {classification}; deployable data requires {sorted(DEPLOYABLE_MODES)}")
    if expected_mode is not None and classification != expected_mode:
        raise ValueError(f"observation classified {classification}, expected {expected_mode}")

    return classification
