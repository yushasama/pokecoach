"""EnvAdapter: a single-agent view over the Pokemon engine vs a fixed opponent.

Wraps ``battle_start`` / ``battle_select`` / ``battle_finish`` and exposes a
gym-like ``reset`` / ``step`` / legal-action / terminal interface. The RL agent
controls one seat; the opponent (a fixed policy) plays the other, advanced
internally between RL decisions.

The adapter deliberately knows NOTHING about PPO, observation encoding, action
masking policy, reward shaping, DSU, belief, or support labels. It returns raw
observations and the engine's terminal status; encoders/maskers/rewards live in
their own modules. INVARIANT #1: it never crashes the caller (engine exceptions
are caught and surfaced as a terminal `crashed` step) and always cleans up via
``battle_finish``. Root ``deck.csv`` is only a declared shipping/failure-baseline
surface; guarded reference/top-player opponents must expose an explicit deck and
never fall back to the root deck silently.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import main
from cg.game import battle_finish, battle_select, battle_start
from deepmew.rl.deck_identity import is_guarded_policy_id


@dataclass
class StepResult:
    obs: dict[str, Any] | None
    done: bool
    result: int | None  # winning seat (0/1), 2 draw, None unfinished
    crashed: bool = False
    timed_out: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class EnvAdapter:
    """One in-progress battle from the RL seat's point of view."""

    def __init__(
        self,
        opponent: Any,
        *,
        our_seat: int = 0,
        our_deck: list[int] | None = None,
        opp_deck: list[int] | None = None,
        max_steps: int = 2000,
        opponent_seed: int = 0,
        phase_timer: Any | None = None,
    ):
        if our_seat not in (0, 1):
            raise ValueError("our_seat must be 0 or 1")
        self.opponent = opponent
        self.our_seat = our_seat
        opponent_deck = getattr(opponent, "deck", None)
        if opp_deck is None and opponent_deck is None:
            label = str(getattr(opponent, "label", getattr(opponent, "policy_id", "")))
            if is_guarded_policy_id(label):
                raise ValueError(
                    f"{label}: guarded opponent is missing explicit deck; refusing root deck.csv fallback"
                )
        self.our_deck = list(our_deck) if our_deck is not None else main.read_deck_csv()
        self.opp_deck = (
            list(opp_deck)
            if opp_deck is not None
            else list(opponent_deck)
            if opponent_deck is not None
            else main.read_deck_csv()
        )
        self.max_steps = max_steps
        self._opp_rng = random.Random(opponent_seed)
        self._obs: dict[str, Any] | None = None
        self._started = False
        self._steps = 0
        self._terminal: StepResult | None = None
        self._phase_timer = phase_timer

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> StepResult:
        self.close()
        self._opp_rng = random.Random(self._opp_rng.randint(0, 2**31 - 1))
        self._steps = 0
        self._terminal = None
        reset = getattr(self.opponent, "reset", None)
        if callable(reset):
            reset()
        if self.our_seat == 0:
            deck0, deck1 = self.our_deck, self.opp_deck
        else:
            deck0, deck1 = self.opp_deck, self.our_deck
        try:
            with self._phase("engine_reset_start"):
                obs, start_data = battle_start(deck0, deck1)
            if obs is None:
                raise RuntimeError(
                    f"BattleStart failed: errorPlayer={start_data.errorPlayer}, "
                    f"errorType={start_data.errorType}"
                )
            self._obs = obs
            self._started = True
            return self._advance_to_agent()
        except Exception as exc:  # INVARIANT #1: never propagate an engine crash
            self._obs = None
            return self._finish_crash(exc)

    def close(self) -> None:
        if self._started and self._obs is not None:
            try:
                battle_finish()
            except Exception:
                pass
        self._started = False
        self._obs = None

    # -- decision interface --------------------------------------------------
    @property
    def select(self) -> dict[str, Any] | None:
        if self._obs is None:
            return None
        return self._obs.get("select")

    @property
    def obs(self) -> dict[str, Any] | None:
        return self._obs

    def terminal(self) -> StepResult | None:
        return self._terminal

    @property
    def selection_steps(self) -> int:
        """Return all engine selections submitted in the current episode."""

        return self._steps

    def step(self, selection: list[int]) -> StepResult:
        """Apply the RL agent's selection, then advance the opponent."""
        if self._terminal is not None:
            return self._terminal
        if not self._started or self._obs is None:
            return self._finish_crash(RuntimeError("step before reset"))
        try:
            with self._phase("engine_action_decoding"):
                submitted = [int(index) for index in selection]
            with self._phase("engine_action_submission"):
                next_obs = battle_select(submitted)
            with self._phase("engine_state_copying"):
                self._obs = next_obs
            with self._phase("engine_misc"):
                self._steps += 1
            return self._advance_to_agent()
        except Exception as exc:
            return self._finish_crash(exc)

    # -- internals -----------------------------------------------------------
    def _phase(self, name: str):
        if self._phase_timer is None:
            return nullcontext()
        return self._phase_timer.phase(name)

    def _advance_to_agent(self) -> StepResult:
        """Play opponent decisions until it is the RL seat's turn or terminal."""
        try:
            while True:
                with self._phase("engine_terminal_check"):
                    if self._steps >= self.max_steps:
                        return self._finish_timeout()
                    current = (self._obs or {}).get("current") or {}
                    result = int(current.get("result", -1))
                if result != -1:
                    return self._finish_result(result)
                with self._phase("engine_turn_owner_check"):
                    player_index = int(current.get("yourIndex") or 0)
                if player_index == self.our_seat:
                    return StepResult(obs=self._obs, done=False, result=None)
                with self._phase("engine_opponent_policy"):
                    selected = self.opponent.select(self._obs, self._opp_rng)
                with self._phase("engine_opponent_action_submission"):
                    next_obs = battle_select(selected)
                with self._phase("engine_state_copying"):
                    self._obs = next_obs
                with self._phase("engine_misc"):
                    self._steps += 1
        except Exception as exc:
            return self._finish_crash(exc)

    def _finish_result(self, result: int) -> StepResult:
        step = StepResult(obs=self._obs, done=True, result=result)
        self._terminal = step
        self.close()
        return step

    def _finish_timeout(self) -> StepResult:
        step = StepResult(
            obs=self._obs, done=True, result=None, timed_out=True,
            info={"reason": f"exceeded {self.max_steps} selections"},
        )
        self._terminal = step
        self.close()
        return step

    def _finish_crash(self, exc: Exception) -> StepResult:
        step = StepResult(
            obs=None, done=True, result=None, crashed=True,
            info={"error": f"{type(exc).__name__}: {exc}"},
        )
        self._terminal = step
        self.close()
        return step
