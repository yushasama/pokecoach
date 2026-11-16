"""Isolated fresh-simulator executor for aggregate oracle-world supervision.

This module never accepts hidden zones as public arguments.  The one permitted
private surface is an opaque zero-argument ``begin_search`` closure owned by the
fresh simulator process.  Returned objects contain terminal scalars, public
trajectory receipts, booleans, and counts only.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from deepmew.q_guided.branching import (
    PrivilegedTeacherExecutionError,
    build_root_consistency_receipt,
    project_student_input,
    student_input_bytes,
)
from deepmew.replay.information_boundary import InformationBoundary
from deepmew.replay.prospective_trace import (
    actor_private_observation,
    public_trace_observation,
)


PRIVILEGED_TEACHER_MODE = "privileged_simulator_teacher"


class SimulatorTeacherSafetyError(RuntimeError):
    """Fail-closed branch error whose message never contains engine payloads."""


@dataclass(frozen=True)
class TerminalRollout:
    terminal_result: int
    terminal_payoff: float
    logical_decision_steps: int
    public_trajectory_sha256: str


@dataclass(frozen=True)
class SimulatorTeacherReceipt:
    isolated_mode_active: bool
    actor_visible_input_stable: bool
    reconstructed_roots_match: bool
    live_root_unchanged: bool
    begin_search_count: int
    search_end_count: int
    search_release_count: int
    rollout_count: int
    immediate_measurement_count: int


def _as_observation(state: Any) -> dict[str, Any]:
    observation = getattr(state, "observation", state)
    if dataclasses.is_dataclass(observation):
        observation = dataclasses.asdict(observation)
    if not isinstance(observation, Mapping):
        raise SimulatorTeacherSafetyError("search returned an invalid observation")

    return copy.deepcopy(dict(observation))


def _result(observation: Mapping[str, Any]) -> int:
    current = observation.get("current")
    value = current.get("result") if isinstance(current, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimulatorTeacherSafetyError("search observation has an invalid result")

    return value


def _actor(observation: Mapping[str, Any]) -> int:
    current = observation.get("current")
    value = current.get("yourIndex") if isinstance(current, Mapping) else None
    if isinstance(value, bool) or value not in (0, 1):
        raise SimulatorTeacherSafetyError("search observation has an invalid actor")

    return int(value)


def _search_id(state: Any) -> int:
    value = getattr(state, "searchId", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimulatorTeacherSafetyError("search state has an invalid search ID")

    return value


def _terminal_payoff(result: int, root_actor: int) -> float:
    if result == root_actor:
        return 1.0
    if result in (0, 1):
        return 0.0
    if result == 2:
        return 0.5

    raise SimulatorTeacherSafetyError("terminal search result is unsupported")


def _mask_face_down_entries(reconstructed: Any, expected: Any) -> Any:
    """Restore the live actor's visibility before comparing a search root."""
    if expected is None:
        return None
    if isinstance(reconstructed, list) and isinstance(expected, list):
        if len(reconstructed) != len(expected):
            return copy.deepcopy(reconstructed)

        return [
            _mask_face_down_entries(reconstructed_value, expected_value)
            for reconstructed_value, expected_value in zip(reconstructed, expected, strict=True)
        ]
    if isinstance(reconstructed, Mapping) and isinstance(expected, Mapping):
        masked = copy.deepcopy(dict(reconstructed))
        for key, expected_value in expected.items():
            if key in masked:
                masked[key] = _mask_face_down_entries(masked[key], expected_value)

        return masked

    return copy.deepcopy(reconstructed)


def _actor_visible_search_root(
    reconstructed: Mapping[str, Any],
    expected_actor_root: Mapping[str, Any],
) -> dict[str, Any]:
    masked = _mask_face_down_entries(reconstructed, expected_actor_root)
    if not isinstance(masked, Mapping):
        raise SimulatorTeacherSafetyError("search root visibility projection failed")

    return project_student_input(masked)


def _public_trace_digest(observations: Sequence[Mapping[str, Any]]) -> str:
    public = [public_trace_observation(dict(observation)) for observation in observations]
    encoded = json.dumps(
        public,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    return hashlib.sha256(encoded).hexdigest()


class PrivilegedSimulatorTeacher:
    """Run fresh sibling searches without retaining or returning hidden state."""

    def __init__(
        self,
        actor_observation: Mapping[str, Any],
        *,
        begin_search: Callable[[], Any],
        step_search: Callable[[Any, Sequence[int]], Any],
        end_search: Callable[[], None],
        verify_live_root_unchanged: Callable[[], bool],
        release_search: Callable[[int], None] | None = None,
    ) -> None:
        if not all(callable(value) for value in (
            begin_search,
            step_search,
            end_search,
            verify_live_root_unchanged,
        )):
            raise TypeError("teacher engine surfaces must be callable")
        if release_search is not None and not callable(release_search):
            raise TypeError("release_search must be callable when provided")
        boundary = InformationBoundary.for_isolated_simulator_teacher()
        safe_root = boundary.artifact_observation(dict(actor_observation))
        self._safe_root = project_student_input(safe_root)
        self._student_bytes = student_input_bytes(self._safe_root)
        self._root_actor = _actor(self._safe_root)
        self._begin_search = begin_search
        self._step_search = step_search
        self._end_search = end_search
        self._verify_live_root_unchanged = verify_live_root_unchanged
        self._release_search = release_search
        self._root_state: Any | None = None
        self._closed = False
        self._begin_count = 0
        self._end_count = 0
        self._release_count = 0
        self._rollout_count = 0
        self._immediate_measurement_count = 0
        self._root_checks_passed = True
        self._live_root_unchanged = True

    def __repr__(self) -> str:
        return (
            "PrivilegedSimulatorTeacher("
            f"mode={PRIVILEGED_TEACHER_MODE!r}, rollouts={self._rollout_count}, "
            f"closed={self._closed})"
        )

    @property
    def actor_visible_root(self) -> dict[str, Any]:
        return copy.deepcopy(self._safe_root)

    def _require_open(self) -> None:
        if self._closed:
            raise SimulatorTeacherSafetyError("privileged simulator teacher is closed")

    def _verify_live_root(self, failure_message: str) -> None:
        self._require_open()
        try:
            unchanged = bool(self._verify_live_root_unchanged())
        except Exception:
            raise SimulatorTeacherSafetyError("live root verification failed") from None
        self._live_root_unchanged &= unchanged
        if not unchanged:
            raise SimulatorTeacherSafetyError(failure_message)

    def _expire_engine_surfaces(self) -> None:
        self._root_state = None
        self._begin_search = _expired
        self._step_search = _expired
        self._end_search = _expired
        self._verify_live_root_unchanged = _expired
        self._release_search = None

    def _end_preserved_root(self) -> None:
        if self._root_state is None or self._end_count:
            return
        try:
            self._end_search()
        except Exception:
            raise SimulatorTeacherSafetyError("SearchEnd failed") from None
        self._end_count += 1

    def _reject_preserved_root(self, message: str) -> None:
        self._root_checks_passed = False
        try:
            self._end_preserved_root()
        finally:
            self._closed = True
            self._expire_engine_surfaces()

        raise SimulatorTeacherSafetyError(message)

    def _preserved_root(self) -> Any:
        """Lazily create and validate the one SearchBegin root for all children."""
        self._require_open()
        if self._root_state is not None:
            return self._root_state
        self._verify_live_root("live simulator root changed before SearchBegin")
        try:
            state = self._begin_search()
        except Exception:
            raise SimulatorTeacherSafetyError("preserved root SearchBegin failed") from None
        if state is None:
            raise SimulatorTeacherSafetyError("preserved root SearchBegin returned no state")
        self._begin_count += 1
        self._root_state = state
        try:
            reconstructed = _as_observation(state)
            visible = _actor_visible_search_root(reconstructed, self._safe_root)
            receipt = build_root_consistency_receipt(self._safe_root, visible)
        except Exception:
            self._reject_preserved_root("preserved root consistency check failed")
        root_matches = bool(
            receipt.actor_visible_root_matches
            and receipt.legal_options_match
            and receipt.player_count_matches
        )
        self._root_checks_passed &= root_matches
        if not root_matches:
            self._reject_preserved_root(
                "preserved search root does not match the actor-visible root"
            )

        return self._root_state

    def _release_child(self, child_root: Any) -> None:
        if self._release_search is None:
            return
        child_id = _search_id(child_root)
        try:
            self._release_search(child_id)
        except Exception:
            raise SimulatorTeacherSafetyError("SearchRelease failed for a child branch") from None
        self._release_count += 1

    def measure_immediate_successor(
        self,
        root_action: Sequence[int],
        *,
        scalar_scorer: Callable[[dict[str, Any], int], Sequence[float]],
        expected_score_count: int,
    ) -> tuple[float, ...]:
        """Measure one actor-visible child and return finite scalars only."""
        self._require_open()
        if not callable(scalar_scorer):
            raise TypeError("scalar_scorer must be callable")
        if (
            isinstance(expected_score_count, bool)
            or not isinstance(expected_score_count, int)
            or expected_score_count <= 0
        ):
            raise ValueError("expected_score_count must be a positive integer")
        if isinstance(root_action, (str, bytes)) or not isinstance(root_action, Sequence):
            raise TypeError("root_action must be a sequence of option indices")
        action = list(root_action)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in action):
            raise TypeError("root_action must contain integer option indices")
        preserved_root = self._preserved_root()
        self._verify_live_root("live simulator root changed before an immediate measurement")
        child_root: Any | None = None
        try:
            try:
                child_root = self._step_search(preserved_root, action)
            except Exception:
                raise SimulatorTeacherSafetyError("forced root action failed") from None
            if child_root is None or child_root is preserved_root:
                raise SimulatorTeacherSafetyError("forced root action did not create a child state")
            if _search_id(child_root) == _search_id(preserved_root):
                raise SimulatorTeacherSafetyError("forced root action reused the preserved root state")
            try:
                raw_successor = _as_observation(child_root)
                boundary = InformationBoundary.for_isolated_simulator_teacher()
                safe_successor = project_student_input(
                    boundary.artifact_observation(
                        actor_private_observation(raw_successor)
                    )
                )
                measured = scalar_scorer(safe_successor, self._root_actor)
                if isinstance(measured, (str, bytes)) or not isinstance(measured, Sequence):
                    raise TypeError
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in measured
                ):
                    raise TypeError
                scores = tuple(float(value) for value in measured)
                if (
                    len(scores) != expected_score_count
                    or any(not math.isfinite(value) for value in scores)
                ):
                    raise ValueError
            except Exception:
                raise SimulatorTeacherSafetyError(
                    "immediate successor scorer returned invalid scalars"
                ) from None
        finally:
            if child_root is not None:
                self._release_child(child_root)
        self._verify_live_root("live simulator root changed after an immediate measurement")
        self._immediate_measurement_count += 1

        return scores

    def rollout_terminal(
        self,
        root_action: Sequence[int],
        *,
        seed: int,
        continuation_selector: Callable[[dict[str, Any], random.Random], Sequence[int]],
        max_steps: int,
    ) -> TerminalRollout:
        """Force one root action and continue to terminal under a passed policy."""
        self._require_open()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not callable(continuation_selector):
            raise TypeError("continuation_selector must be callable")
        action = list(root_action)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in action):
            raise TypeError("root_action must contain integer option indices")
        rng = random.Random(seed)
        public_path: list[dict[str, Any]] = []
        preserved_root = self._preserved_root()
        self._verify_live_root("live simulator root changed before a sibling")
        child_root: Any | None = None
        try:
            try:
                state = self._step_search(preserved_root, action)
            except Exception:
                raise SimulatorTeacherSafetyError("forced root action failed") from None
            if state is None or state is preserved_root:
                raise SimulatorTeacherSafetyError("forced root action did not create a child state")
            if _search_id(state) == _search_id(preserved_root):
                raise SimulatorTeacherSafetyError("forced root action reused the preserved root state")
            child_root = state
            observation = _as_observation(state)
            public_path.append(observation)
            steps = 1
            while _result(observation) == -1 and steps < max_steps:
                try:
                    selected = list(continuation_selector(observation, rng))
                    state = self._step_search(state, selected)
                except SimulatorTeacherSafetyError:
                    raise
                except Exception:
                    raise SimulatorTeacherSafetyError("continuation search step failed") from None
                observation = _as_observation(state)
                public_path.append(observation)
                steps += 1
            result = _result(observation)
            if result == -1:
                raise SimulatorTeacherSafetyError("branch exceeded the logical-decision limit")
            payoff = _terminal_payoff(result, self._root_actor)
            digest = _public_trace_digest(public_path)
        finally:
            if child_root is not None:
                self._release_child(child_root)
        self._verify_live_root("live simulator root changed after a sibling")
        self._rollout_count += 1

        return TerminalRollout(
            terminal_result=result,
            terminal_payoff=payoff,
            logical_decision_steps=steps,
            public_trajectory_sha256=digest,
        )

    def receipt(self) -> SimulatorTeacherReceipt:
        stable = student_input_bytes(self._safe_root) == self._student_bytes

        return SimulatorTeacherReceipt(
            isolated_mode_active=True,
            actor_visible_input_stable=stable,
            reconstructed_roots_match=self._root_checks_passed,
            live_root_unchanged=self._live_root_unchanged,
            begin_search_count=self._begin_count,
            search_end_count=self._end_count,
            search_release_count=self._release_count,
            rollout_count=self._rollout_count,
            immediate_measurement_count=self._immediate_measurement_count,
        )

    def close(self) -> SimulatorTeacherReceipt:
        if self._closed:
            return self.receipt()
        try:
            self._end_preserved_root()
        finally:
            self._closed = True
            self._expire_engine_surfaces()

        return self.receipt()

    def __enter__(self) -> "PrivilegedSimulatorTeacher":
        self._require_open()

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _expired(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs

    raise PrivilegedTeacherExecutionError("privileged teacher engine closure expired")
