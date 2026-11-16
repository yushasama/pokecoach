"""Torch-free seat adapters shared by the frozen league and the RL game farm.

Moved verbatim from ``policies.py`` / ``temporal.py`` (both still re-export
them) so farm worker processes can run reference opponents and actor-visible
histories without importing torch.
"""

from __future__ import annotations

import os
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.schema import Payload, payload


class SeatPolicy(Protocol):
    label: str

    def select(self, observation: Payload, rng: random.Random) -> list[int]: ...


def greedy_fallback(observation: Payload) -> list[int]:
    from deepmew.policy.greedy import choose_greedy_action
    return choose_greedy_action(observation)


@contextmanager
def reference_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class ReferenceSeatPolicy:
    def __init__(self, teacher: SeatPolicy, directory: Path) -> None:
        self.teacher, self.directory, self.label = teacher, directory, teacher.label
        self.counts: Counter[str] = Counter()

    @property
    def fallback_errors(self) -> int:
        return int(getattr(self.teacher, '_fallback_errors', 0))

    def select(self, observation: Payload, rng: random.Random) -> list[int]:
        # The native dataclass decoder requires these keys even when unavailable.
        # Explicit absence preserves the common projection; hidden payloads stay out.
        visible = dict(observation)
        if isinstance(observation.get('current'), dict):
            visible['current'] = {**observation['current'], 'looking': None}
        if isinstance(observation.get('select'), dict):
            visible['select'] = {**observation['select'], 'deck': None}
            # Native DECK=1 / LOOKING=12 options require excluded zone buffers.
            # Keep these requests on the same declared baseline as candidates.
            if any(option.get('area') in (1, 12) for option in observation['select'].get('option', [])):
                self.counts['fallback:restricted_zone'] += 1
                return greedy_fallback(observation)
        previous_errors = self.fallback_errors
        with reference_directory(self.directory):
            selected = self.teacher.select(visible, rng)
        if self.fallback_errors != previous_errors:
            raise RuntimeError('reference notebook failed; proxy fallback rejected')
        self.counts['notebook_choices'] += 1
        return selected


class IndependentSeat:
    def __init__(self, policy: SeatPolicy, seed: int, max_seconds: float = 5.0) -> None:
        self.policy, self.label = policy, policy.label
        self.rng = random.Random(seed)
        self.python_state = random.Random(seed).getstate()
        self.numpy_state = np.random.RandomState(seed % 2**32).get_state()
        self.calls = 0
        self.max_seconds = max_seconds
        self.latencies: list[float] = []

    def select(self, observation: Payload, ignored_shared_rng: random.Random) -> list[int]:
        # Some frozen notebooks use module-global randomness; isolate that too.
        previous_python, previous_numpy = random.getstate(), np.random.get_state()
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        started = time.perf_counter()
        try:
            self.calls += 1
            contract = load_contract()
            visible = payload(contract.project_actor(observation))
            visible['logs'] = contract.public_logs(observation.get('logs', []))
            return self.policy.select(visible, self.rng)
        finally:
            elapsed = time.perf_counter() - started
            self.latencies.append(elapsed)
            self.python_state, self.numpy_state = random.getstate(), np.random.get_state()
            random.setstate(previous_python)
            np.random.set_state(previous_numpy)
            if elapsed > self.max_seconds:
                raise TimeoutError('frozen per-decision budget exceeded')


class ActorNotifyingSeat:
    def __init__(self, policy: SelectPolicy, observers: tuple[FullPrefixLivePolicy, ...]) -> None:
        self.policy, self.observers, self.label = policy, observers, policy.label

    def select(self, observation: Payload, rng: random.Random) -> list[int]:
        current = observation.get('current')
        if not isinstance(current, dict) or type(current.get('yourIndex')) is not int:
            raise ValueError('native observation has no actor identity')
        for observer in self.observers:
            observer.observe(observation)
        return self.policy.select(observation, rng)
