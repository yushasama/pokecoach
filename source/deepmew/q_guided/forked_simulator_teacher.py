"""POSIX fork-clone executor for aggregate privileged-teacher rollouts.

The parent creates and validates exactly one CABT ``SearchBegin`` root.  Each
sibling then runs in a forked child whose address space starts from that exact
in-memory root.  The only child-to-parent channel is a bounded pipe containing
one fixed-schema JSON result made of terminal scalars, a step count, and a
public-trajectory digest.

Hidden zones and engine payloads are inherited only through the operating
system process snapshot.  They are never arguments to this API and never
cross IPC.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import select
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from deepmew.q_guided.branching import (
    PrivilegedTeacherExecutionError,
    build_root_consistency_receipt,
    project_student_input,
    student_input_bytes,
)
from deepmew.q_guided.simulator_teacher import (
    TerminalRollout,
    _actor,
    _actor_visible_search_root,
    _as_observation,
    _public_trace_digest,
    _result,
    _search_id,
    _terminal_payoff,
)
from deepmew.replay.information_boundary import InformationBoundary


DEFAULT_CHILD_TIMEOUT_SECONDS = 60.0
DEFAULT_IPC_PAYLOAD_CAP_BYTES = 4096
MIN_IPC_PAYLOAD_CAP_BYTES = 256
MAX_IPC_PAYLOAD_CAP_BYTES = 65536
POLL_INTERVAL_SECONDS = 0.05

_SUCCESS_PAYLOAD_FIELDS = frozenset({
    "status",
    "terminal_result",
    "terminal_payoff",
    "logical_decision_steps",
    "public_trajectory_sha256",
})
_ERROR_PAYLOAD = b'{"status":"error"}'


class ForkedSimulatorTeacherSafetyError(RuntimeError):
    """Fail-closed fork/IPC error whose message contains no child payload."""


class _ChildTimeout(RuntimeError):
    pass


class _ChildPayloadRejected(RuntimeError):
    pass


def _fork_available() -> bool:
    return os.name == "posix" and hasattr(os, "fork")


@dataclass(frozen=True)
class ForkedSimulatorTeacherReceipt:
    """Persistence-safe execution facts; fields are booleans and counts only."""

    isolated_mode_active: bool
    fork_clone_active: bool
    actor_visible_input_stable: bool
    reconstructed_roots_match: bool
    live_root_unchanged: bool
    begin_search_count: int
    search_end_count: int
    fork_count: int
    rollout_count: int
    failed_rollout_count: int
    timeout_count: int
    payload_rejection_count: int


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    return converted


def _require_payload_cap(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ipc_payload_cap_bytes must be an integer")
    if not MIN_IPC_PAYLOAD_CAP_BYTES <= value <= MAX_IPC_PAYLOAD_CAP_BYTES:
        raise ValueError(
            "ipc_payload_cap_bytes must be within the fixed safety bounds"
        )

    return value


def _terminal_payload(rollout: TerminalRollout) -> bytes:
    payload = {"status": "ok", **asdict(rollout)}

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _decode_terminal_payload(payload: bytes, *, max_steps: int) -> TerminalRollout:
    """Decode the exclusive scalar-only IPC schema and reject every extension."""
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _ChildPayloadRejected from None
    if not isinstance(decoded, dict):
        raise _ChildPayloadRejected
    if decoded.get("status") == "error" and set(decoded) == {"status"}:
        raise ForkedSimulatorTeacherSafetyError("forked sibling failed")
    if set(decoded) != _SUCCESS_PAYLOAD_FIELDS or decoded.get("status") != "ok":
        raise _ChildPayloadRejected
    result = decoded.get("terminal_result")
    payoff = decoded.get("terminal_payoff")
    steps = decoded.get("logical_decision_steps")
    digest = decoded.get("public_trajectory_sha256")
    if isinstance(result, bool) or result not in (0, 1, 2):
        raise _ChildPayloadRejected
    if isinstance(payoff, bool) or not isinstance(payoff, (int, float)):
        raise _ChildPayloadRejected
    numeric_payoff = float(payoff)
    if not math.isfinite(numeric_payoff) or numeric_payoff not in (0.0, 0.5, 1.0):
        raise _ChildPayloadRejected
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise _ChildPayloadRejected
    if not 1 <= steps <= max_steps:
        raise _ChildPayloadRejected
    if not isinstance(digest, str) or len(digest) != 64:
        raise _ChildPayloadRejected
    if any(character not in "0123456789abcdef" for character in digest):
        raise _ChildPayloadRejected

    return TerminalRollout(
        terminal_result=result,
        terminal_payoff=numeric_payoff,
        logical_decision_steps=steps,
        public_trajectory_sha256=digest,
    )


def _write_all(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("pipe write failed")
        offset += written


def _silence_child_output() -> None:
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        if descriptor not in (1, 2):
            os.close(descriptor)


def _run_child_rollout(
    preserved_root: Any,
    root_action: Sequence[int],
    *,
    seed: int,
    continuation_selector: Callable[[dict[str, Any], random.Random], Sequence[int]],
    max_steps: int,
    root_actor: int,
    step_search: Callable[[Any, Sequence[int]], Any],
) -> TerminalRollout:
    rng = random.Random(seed)
    public_path: list[dict[str, Any]] = []
    state = step_search(preserved_root, list(root_action))
    if state is None or state is preserved_root:
        raise ForkedSimulatorTeacherSafetyError(
            "forced root action did not create a child state"
        )
    if _search_id(state) == _search_id(preserved_root):
        raise ForkedSimulatorTeacherSafetyError(
            "forced root action reused the preserved root state"
        )
    observation = _as_observation(state)
    public_path.append(observation)
    steps = 1
    while _result(observation) == -1 and steps < max_steps:
        selected = list(continuation_selector(observation, rng))
        state = step_search(state, selected)
        observation = _as_observation(state)
        public_path.append(observation)
        steps += 1
    result = _result(observation)
    if result == -1:
        raise ForkedSimulatorTeacherSafetyError(
            "branch exceeded the logical-decision limit"
        )

    return TerminalRollout(
        terminal_result=result,
        terminal_payoff=_terminal_payoff(result, root_actor),
        logical_decision_steps=steps,
        public_trajectory_sha256=_public_trace_digest(public_path),
    )


def _child_main(
    write_descriptor: int,
    preserved_root: Any,
    root_action: Sequence[int],
    *,
    seed: int,
    continuation_selector: Callable[[dict[str, Any], random.Random], Sequence[int]],
    max_steps: int,
    root_actor: int,
    step_search: Callable[[Any, Sequence[int]], Any],
    payload_cap: int,
) -> None:
    exit_code = 1
    payload = _ERROR_PAYLOAD
    try:
        _silence_child_output()
        rollout = _run_child_rollout(
            preserved_root,
            root_action,
            seed=seed,
            continuation_selector=continuation_selector,
            max_steps=max_steps,
            root_actor=root_actor,
            step_search=step_search,
        )
        candidate = _terminal_payload(rollout)
        if len(candidate) <= payload_cap:
            payload = candidate
            exit_code = 0
    except BaseException:
        payload = _ERROR_PAYLOAD
        exit_code = 1
    try:
        _write_all(write_descriptor, payload)
    except BaseException:
        exit_code = 1
    finally:
        try:
            os.close(write_descriptor)
        finally:
            os._exit(exit_code)


def _kill_and_reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _receive_child_payload(
    pid: int,
    read_descriptor: int,
    *,
    timeout_seconds: float,
    payload_cap: int,
) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    payload = bytearray()
    child_status: int | None = None
    reached_eof = False
    try:
        while child_status is None or not reached_eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _kill_and_reap(pid)
                raise _ChildTimeout
            readable, _, _ = select.select(
                [read_descriptor],
                [],
                [],
                min(POLL_INTERVAL_SECONDS, remaining),
            )
            if readable:
                chunk = os.read(read_descriptor, min(65536, payload_cap + 1 - len(payload)))
                if chunk:
                    payload.extend(chunk)
                    if len(payload) > payload_cap:
                        _kill_and_reap(pid)
                        raise _ChildPayloadRejected
                else:
                    reached_eof = True
            if child_status is None:
                waited_pid, status = os.waitpid(pid, os.WNOHANG)
                if waited_pid == pid:
                    child_status = status
        if child_status is None:
            raise ForkedSimulatorTeacherSafetyError("forked sibling wait failed")
        if not os.WIFEXITED(child_status) or os.WEXITSTATUS(child_status) != 0:
            raise ForkedSimulatorTeacherSafetyError("forked sibling failed")

        return bytes(payload)
    except (_ChildTimeout, _ChildPayloadRejected, ForkedSimulatorTeacherSafetyError):
        raise
    except BaseException:
        if child_status is None:
            _kill_and_reap(pid)
        raise ForkedSimulatorTeacherSafetyError(
            "forked sibling IPC failed"
        ) from None
    finally:
        try:
            os.close(read_descriptor)
        except OSError:
            pass


class ForkedSimulatorTeacher:
    """Fork every sibling from one validated, unmodified parent search root."""

    def __init__(
        self,
        actor_observation: Mapping[str, Any],
        *,
        begin_search: Callable[[], Any],
        step_search: Callable[[Any, Sequence[int]], Any],
        end_search: Callable[[], None],
        verify_live_root_unchanged: Callable[[], bool],
        child_timeout_seconds: float = DEFAULT_CHILD_TIMEOUT_SECONDS,
        ipc_payload_cap_bytes: int = DEFAULT_IPC_PAYLOAD_CAP_BYTES,
    ) -> None:
        if not all(callable(value) for value in (
            begin_search,
            step_search,
            end_search,
            verify_live_root_unchanged,
        )):
            raise TypeError("teacher engine surfaces must be callable")
        timeout = _require_positive_float(
            child_timeout_seconds,
            "child_timeout_seconds",
        )
        payload_cap = _require_payload_cap(ipc_payload_cap_bytes)
        boundary = InformationBoundary.for_isolated_simulator_teacher()
        safe_root = boundary.artifact_observation(dict(actor_observation))
        self._safe_root = project_student_input(safe_root)
        self._student_bytes = student_input_bytes(self._safe_root)
        self._root_actor = _actor(self._safe_root)
        self._begin_search = begin_search
        self._step_search = step_search
        self._end_search = end_search
        self._verify_live_root_unchanged = verify_live_root_unchanged
        self._child_timeout_seconds = timeout
        self._ipc_payload_cap_bytes = payload_cap
        self._root_state: Any | None = None
        self._closed = False
        self._begin_count = 0
        self._end_count = 0
        self._fork_count = 0
        self._successful_rollout_count = 0
        self._failed_rollout_count = 0
        self._timeout_count = 0
        self._payload_rejection_count = 0
        self._root_checks_passed = True
        self._live_root_unchanged = True

    def __repr__(self) -> str:
        return (
            "ForkedSimulatorTeacher("
            f"forks={self._fork_count}, "
            f"rollouts={self._successful_rollout_count}, "
            f"closed={self._closed})"
        )

    @property
    def actor_visible_root(self) -> dict[str, Any]:
        return copy.deepcopy(self._safe_root)

    def _require_open(self) -> None:
        if self._closed:
            raise ForkedSimulatorTeacherSafetyError(
                "forked simulator teacher is closed"
            )

    def _verify_live_root(self, failure_message: str) -> None:
        self._require_open()
        try:
            unchanged = bool(self._verify_live_root_unchanged())
        except Exception:
            raise ForkedSimulatorTeacherSafetyError(
                "live root verification failed"
            ) from None
        self._live_root_unchanged &= unchanged
        if not unchanged:
            raise ForkedSimulatorTeacherSafetyError(failure_message)

    def _expire_engine_surfaces(self) -> None:
        self._root_state = None
        self._begin_search = _expired
        self._step_search = _expired
        self._end_search = _expired
        self._verify_live_root_unchanged = _expired

    def _end_preserved_root(self) -> None:
        if self._root_state is None or self._end_count:
            return
        try:
            self._end_search()
        except Exception:
            raise ForkedSimulatorTeacherSafetyError("SearchEnd failed") from None
        self._end_count += 1

    def _reject_preserved_root(self, message: str) -> None:
        self._root_checks_passed = False
        try:
            self._end_preserved_root()
        finally:
            self._closed = True
            self._expire_engine_surfaces()

        raise ForkedSimulatorTeacherSafetyError(message)

    def _preserved_root(self) -> Any:
        self._require_open()
        if self._root_state is not None:
            return self._root_state
        self._verify_live_root("live simulator root changed before SearchBegin")
        try:
            state = self._begin_search()
        except Exception:
            raise ForkedSimulatorTeacherSafetyError(
                "preserved root SearchBegin failed"
            ) from None
        if state is None:
            raise ForkedSimulatorTeacherSafetyError(
                "preserved root SearchBegin returned no state"
            )
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

    def _fork_rollout(
        self,
        preserved_root: Any,
        root_action: Sequence[int],
        *,
        seed: int,
        continuation_selector: Callable[[dict[str, Any], random.Random], Sequence[int]],
        max_steps: int,
    ) -> bytes:
        if not _fork_available():
            raise ForkedSimulatorTeacherSafetyError(
                "POSIX fork is unavailable for sibling cloning"
            )
        try:
            read_descriptor, write_descriptor = os.pipe()
        except OSError:
            raise ForkedSimulatorTeacherSafetyError(
                "forked sibling pipe creation failed"
            ) from None
        try:
            pid = os.fork()
        except BaseException:
            os.close(read_descriptor)
            os.close(write_descriptor)
            raise ForkedSimulatorTeacherSafetyError(
                "forked sibling creation failed"
            ) from None
        if pid == 0:
            try:
                os.close(read_descriptor)
                _child_main(
                    write_descriptor,
                    preserved_root,
                    root_action,
                    seed=seed,
                    continuation_selector=continuation_selector,
                    max_steps=max_steps,
                    root_actor=self._root_actor,
                    step_search=self._step_search,
                    payload_cap=self._ipc_payload_cap_bytes,
                )
            except BaseException:
                os._exit(1)
        self._fork_count += 1
        try:
            os.close(write_descriptor)
        except OSError:
            _kill_and_reap(pid)
            try:
                os.close(read_descriptor)
            except OSError:
                pass
            raise ForkedSimulatorTeacherSafetyError(
                "forked sibling pipe setup failed"
            ) from None

        return _receive_child_payload(
            pid,
            read_descriptor,
            timeout_seconds=self._child_timeout_seconds,
            payload_cap=self._ipc_payload_cap_bytes,
        )

    def rollout_terminal(
        self,
        root_action: Sequence[int],
        *,
        seed: int,
        continuation_selector: Callable[[dict[str, Any], random.Random], Sequence[int]],
        max_steps: int,
    ) -> TerminalRollout:
        """Run one scalar-only sibling in a fork of the exact parent root."""
        self._require_open()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not callable(continuation_selector):
            raise TypeError("continuation_selector must be callable")
        if isinstance(root_action, (str, bytes)) or not isinstance(root_action, Sequence):
            raise TypeError("root_action must be a sequence of option indices")
        action = list(root_action)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in action):
            raise TypeError("root_action must contain integer option indices")
        if not _fork_available():
            raise ForkedSimulatorTeacherSafetyError(
                "POSIX fork is unavailable for sibling cloning"
            )
        preserved_root = self._preserved_root()
        self._verify_live_root("live simulator root changed before a sibling")
        fork_count_before = self._fork_count
        try:
            payload = self._fork_rollout(
                preserved_root,
                action,
                seed=seed,
                continuation_selector=continuation_selector,
                max_steps=max_steps,
            )
            rollout = _decode_terminal_payload(payload, max_steps=max_steps)
            self._verify_live_root("live simulator root changed after a sibling")
        except _ChildTimeout:
            self._timeout_count += 1
            self._failed_rollout_count += 1
            raise ForkedSimulatorTeacherSafetyError(
                "forked sibling timed out"
            ) from None
        except _ChildPayloadRejected:
            self._payload_rejection_count += 1
            self._failed_rollout_count += 1
            raise ForkedSimulatorTeacherSafetyError(
                "forked sibling returned an invalid IPC payload"
            ) from None
        except ForkedSimulatorTeacherSafetyError:
            if self._fork_count > fork_count_before:
                self._failed_rollout_count += 1
            raise
        self._successful_rollout_count += 1

        return rollout

    def receipt(self) -> ForkedSimulatorTeacherReceipt:
        stable = student_input_bytes(self._safe_root) == self._student_bytes

        return ForkedSimulatorTeacherReceipt(
            isolated_mode_active=True,
            fork_clone_active=_fork_available(),
            actor_visible_input_stable=stable,
            reconstructed_roots_match=self._root_checks_passed,
            live_root_unchanged=self._live_root_unchanged,
            begin_search_count=self._begin_count,
            search_end_count=self._end_count,
            fork_count=self._fork_count,
            rollout_count=self._successful_rollout_count,
            failed_rollout_count=self._failed_rollout_count,
            timeout_count=self._timeout_count,
            payload_rejection_count=self._payload_rejection_count,
        )

    def close(self) -> ForkedSimulatorTeacherReceipt:
        if self._closed:
            return self.receipt()
        try:
            self._end_preserved_root()
        finally:
            self._closed = True
            self._expire_engine_surfaces()

        return self.receipt()

    def __enter__(self) -> "ForkedSimulatorTeacher":
        self._require_open()

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _expired(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs

    raise PrivilegedTeacherExecutionError("privileged teacher engine closure expired")
