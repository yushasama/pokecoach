"""Compact controlled environments for RL critical-path validation.

These environments use the same reset/step/select shape as `EnvAdapter`, but
they are tiny enough to prove PPO learning, mask use, delayed credit, and
vectorized reset isolation quickly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cg.api import OptionType, SelectContext
from deepmew.rl.env_adapter import StepResult
from deepmew.rl.observation import ObservationSchema
from deepmew.rl.rewards import RewardContext


def _select(n_options: int, *, min_count: int = 1, max_count: int = 1) -> dict[str, Any]:
    return {
        "type": 0,
        "context": SelectContext.MAIN.value,
        "minCount": min_count,
        "maxCount": max_count,
        "option": [{"type": OptionType.PLAY.value, "test_index": index} for index in range(n_options)],
    }


@dataclass(frozen=True)
class FeatureVectorEncoder:
    dim: int

    @property
    def schema(self) -> ObservationSchema:
        return ObservationSchema(tuple(f"controlled_{index}" for index in range(self.dim)))

    def encode(self, obs: dict[str, Any], context: dict[str, Any] | None = None) -> list[float]:
        values = list(obs.get("features") or [])
        if len(values) != self.dim:
            raise ValueError(f"controlled feature dim mismatch: {len(values)} != {self.dim}")
        return [float(value) for value in values]


@dataclass(frozen=True)
class ControlledReward:
    name: str = "controlled_reward"

    def __call__(self, ctx: RewardContext) -> float:
        if ctx.next_obs and "_reward" in ctx.next_obs:
            return float(ctx.next_obs["_reward"])
        return 0.0


class _BaseControlledEnv:
    our_seat = 0
    opponent_label = "controlled"

    def __init__(self, *, seed: int = 0):
        self.rng = random.Random(seed)
        self._obs: dict[str, Any] | None = None
        self._done = False

    @property
    def select(self) -> dict[str, Any] | None:
        return None if self._obs is None else self._obs.get("select")

    def close(self) -> None:
        self._obs = None
        self._done = True


class FixedMaskedBanditEnv(_BaseControlledEnv):
    """One-step deterministic bandit with one optimal legal action."""

    def __init__(self, *, optimal_index: int = 2, n_options: int = 4, seed: int = 0):
        super().__init__(seed=seed)
        self.optimal_index = optimal_index
        self.n_options = n_options

    def reset(self) -> StepResult:
        self._done = False
        self._obs = {"features": [1.0, 0.0, 0.0, 0.0], "select": _select(self.n_options)}
        return StepResult(obs=self._obs, done=False, result=None)

    def step(self, selection: list[int]) -> StepResult:
        reward = 1.0 if selection == [self.optimal_index] else 0.0
        self._done = True
        self._obs = {"features": [1.0, 0.0, 0.0, 0.0], "select": _select(self.n_options), "_reward": reward}
        return StepResult(obs=self._obs, done=True, result=0 if reward > 0 else 1)


class ChangingMaskContextualBanditEnv(_BaseControlledEnv):
    """Optimal high-index action is unavailable in one context."""

    def __init__(self, *, seed: int = 0):
        super().__init__(seed=seed)
        self.context = 0

    def reset(self) -> StepResult:
        self._done = False
        self.context = 1 - self.context
        n_options = 3 if self.context == 0 else 2
        self._obs = {
            "features": [1.0 if self.context == 0 else 0.0, 1.0 if self.context == 1 else 0.0, 0.0, 0.0],
            "select": _select(n_options),
        }
        return StepResult(obs=self._obs, done=False, result=None)

    def step(self, selection: list[int]) -> StepResult:
        optimal = 2 if self.context == 0 else 1
        reward = 1.0 if selection == [optimal] else 0.0
        self._done = True
        assert self._obs is not None
        self._obs = {**self._obs, "_reward": reward}
        return StepResult(obs=self._obs, done=True, result=0 if reward > 0 else 1)


class TwoStepDelayedCreditEnv(_BaseControlledEnv):
    """First-step choice controls terminal reward; first step has no reward."""

    def __init__(self, *, seed: int = 0):
        super().__init__(seed=seed)
        self.first_action = 0
        self.stage = 0

    def reset(self) -> StepResult:
        self._done = False
        self.stage = 0
        self.first_action = 0
        self._obs = {"features": [0.0, 0.0, 1.0, 0.0], "select": _select(2)}
        return StepResult(obs=self._obs, done=False, result=None)

    def step(self, selection: list[int]) -> StepResult:
        if self.stage == 0:
            self.first_action = selection[0] if selection else 0
            self.stage = 1
            self._obs = {"features": [0.0, 0.0, 0.0, 1.0], "select": _select(1), "_reward": 0.0}
            return StepResult(obs=self._obs, done=False, result=None)
        reward = 1.0 if self.first_action == 1 else 0.0
        self._done = True
        self._obs = {"features": [0.0, 0.0, 0.0, 1.0], "select": _select(1), "_reward": reward}
        return StepResult(obs=self._obs, done=True, result=0 if reward > 0 else 1)


class VectorResetIsolationEnv(_BaseControlledEnv):
    """Stateful env whose feature vector exposes seed/state for isolation tests."""

    def __init__(self, *, env_id: int, seed: int = 0):
        super().__init__(seed=seed)
        self.env_id = env_id
        self.counter = 0
        self.our_seat = env_id % 2
        self.opponent_label = f"controlled-{env_id}"

    def _make_obs(self, reward: float = 0.0) -> dict[str, Any]:
        return {
            "features": [float(self.env_id), float(self.counter), 0.0, 1.0],
            "select": _select(2),
            "_reward": reward,
        }

    def reset(self) -> StepResult:
        self._done = False
        self.counter = self.rng.randint(1, 1000)
        self._obs = self._make_obs()
        return StepResult(obs=self._obs, done=False, result=None)

    def step(self, selection: list[int]) -> StepResult:
        self.counter += 1
        done = self.counter % 3 == 0
        self._done = done
        self._obs = self._make_obs(reward=1.0 if selection == [1] else 0.0)
        return StepResult(obs=self._obs, done=done, result=0 if done else None)


class MultiSelectEchoEnv(_BaseControlledEnv):
    """One-step env that requires a two-item multiselect action."""

    opponent_label = "controlled-multiselect"

    def reset(self) -> StepResult:
        self._done = False
        self._obs = {"features": [0.0, 1.0, 0.0, 1.0], "select": _select(4, min_count=2, max_count=2)}
        return StepResult(obs=self._obs, done=False, result=None)

    def step(self, selection: list[int]) -> StepResult:
        reward = 1.0 if len(selection) == 2 and len(set(selection)) == 2 else 0.0
        self._done = True
        self._obs = {
            "features": [0.0, 1.0, 0.0, 1.0],
            "select": _select(4, min_count=2, max_count=2),
            "_reward": reward,
        }
        return StepResult(obs=self._obs, done=True, result=0 if reward > 0 else 1)
