"""Parametrized real-meta game runner: our deck/policy vs an opponent deck/policy.

Mirrors `deepmew.arena.runner` but instantiates `battle_start(deck0, deck1)` with
two explicit 60-card lists so DeepMew (on root `deck.csv`) can play a K3 Deck Zoo
opponent deck. Reuses the existing `select_action` policy dispatch for spec-based
agents and `ReplayAgent.select` for replay opponents. Eval-only; the runtime
submission path is untouched.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from cg.game import battle_finish, battle_select, battle_start
from deepmew.arena.models import AgentSpec
from deepmew.arena.selection import select_action
from deepmew.arena.specs import parse_agent_spec
from deepmew.policy import valid_selection


class SeatAgent(Protocol):
    label: str

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]: ...


@dataclass
class SpecAgent:
    """Adapter so an arena AgentSpec (v2.4 / first-legal / ...) fits SeatAgent."""

    spec: AgentSpec
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.spec.label

    @classmethod
    def from_str(cls, spec: str) -> "SpecAgent":
        return cls(spec=parse_agent_spec(spec))

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        return select_action(self.spec, obs, rng)


@dataclass(frozen=True)
class RealMetaGameRecord:
    """One explicit-deck arena game, including safety termination details."""

    game_index: int
    seat_our: int
    rng_seed: int | None
    result: int | None
    turns: int
    crashed: bool
    timed_out: bool
    illegal: bool
    elapsed_seconds: float
    error: str | None = None
    illegal_agent: str | None = None

    @property
    def outcome(self) -> str:
        if self.illegal:
            return "illegal"
        if self.crashed:
            return "crash"
        if self.timed_out:
            return "timeout"
        if self.result == 2:
            return "draw"
        if self.result == self.seat_our:
            return "win"
        if self.result in (0, 1):
            return "loss"

        return "incomplete"

    def __iter__(self):
        """Preserve the legacy five-value unpacking contract."""
        yield self.result
        yield self.turns
        yield self.crashed
        yield self.timed_out
        yield self.error


@dataclass
class MatchupSummary:
    our_label: str
    opp_label: str
    opp_deck_hash: str
    opp_archetype: str
    bucket: str
    games: int
    completed: int
    our_wins: int
    opp_wins: int
    draws: int
    crashes: int
    timeouts: int
    avg_turns: float
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)
    illegal: int = 0
    attempted: int = 0
    stopped_early: bool = False
    records: list[RealMetaGameRecord] = field(default_factory=list)


def run_real_meta_game(
    our_agent: SeatAgent,
    opp_agent: SeatAgent,
    our_deck: list[int],
    opp_deck: list[int],
    *,
    seat_our: int,
    max_steps: int,
    rng: random.Random,
    game_index: int = 0,
    rng_seed: int | None = None,
    on_observation: Callable[[dict[str, Any]], None] | None = None,
) -> RealMetaGameRecord:
    """Run one game and return result plus explicit safety instrumentation."""
    if seat_our == 0:
        deck0, deck1, agent0, agent1 = our_deck, opp_deck, our_agent, opp_agent
    else:
        deck0, deck1, agent0, agent1 = opp_deck, our_deck, opp_agent, our_agent

    started = time.perf_counter()
    obs = None
    try:
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            raise RuntimeError(
                f"BattleStart failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
            )
        for _ in range(max_steps):
            if on_observation is not None:
                on_observation(obs)
            current = obs.get("current") or {}
            if current.get("result", -1) != -1:
                result = int(current["result"])
                if result not in (0, 1, 2):
                    raise RuntimeError(f"engine returned unknown terminal result {result}")
                return RealMetaGameRecord(
                    game_index=game_index,
                    seat_our=seat_our,
                    rng_seed=rng_seed,
                    result=result,
                    turns=int(current.get("turn") or 0),
                    crashed=False,
                    timed_out=False,
                    illegal=False,
                    elapsed_seconds=time.perf_counter() - started,
                )
            player_index = int(current.get("yourIndex") or 0)
            agent = agent0 if player_index == 0 else agent1
            selected = agent.select(obs, rng)
            select = obs.get("select")
            if isinstance(select, dict) and not valid_selection(selected, select):
                return RealMetaGameRecord(
                    game_index=game_index,
                    seat_our=seat_our,
                    rng_seed=rng_seed,
                    result=None,
                    turns=int(current.get("turn") or 0),
                    crashed=False,
                    timed_out=False,
                    illegal=True,
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"invalid selection from {agent.label}: {selected!r}",
                    illegal_agent=agent.label,
                )
            obs = battle_select(selected)
        return RealMetaGameRecord(
            game_index=game_index,
            seat_our=seat_our,
            rng_seed=rng_seed,
            result=None,
            turns=max_steps,
            crashed=False,
            timed_out=True,
            illegal=False,
            elapsed_seconds=time.perf_counter() - started,
            error=f"game exceeded {max_steps} selections",
        )
    except Exception as exc:
        return RealMetaGameRecord(
            game_index=game_index,
            seat_our=seat_our,
            rng_seed=rng_seed,
            result=None,
            turns=0,
            crashed=True,
            timed_out=False,
            illegal=False,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if obs is not None:
            try:
                battle_finish()
            except Exception as exc:
                current = obs.get("current") or {} if isinstance(obs, dict) else {}
                return RealMetaGameRecord(
                    game_index=game_index,
                    seat_our=seat_our,
                    rng_seed=rng_seed,
                    result=None,
                    turns=int(current.get("turn") or 0),
                    crashed=True,
                    timed_out=False,
                    illegal=False,
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"battle_finish {type(exc).__name__}: {exc}",
                )


def run_real_meta_matchup(
    our_agent: SeatAgent,
    opp_agent: SeatAgent,
    our_deck: list[int],
    opp_deck: list[int],
    *,
    games: int,
    bucket: str,
    opp_deck_hash: str,
    opp_archetype: str,
    seed: int = 0,
    max_steps: int = 5000,
    game_seeds: Sequence[int] | None = None,
    stop_on_safety_failure: bool = False,
) -> MatchupSummary:
    if games < 0:
        raise ValueError("games must be non-negative")
    if game_seeds is not None and len(game_seeds) != games:
        raise ValueError(f"game_seeds has {len(game_seeds)} entries; expected {games}")

    shared_rng = random.Random(seed)
    started = time.perf_counter()
    our_wins = opp_wins = draws = crashes = timeouts = illegal = completed = turns = 0
    errors: list[str] = []
    records: list[RealMetaGameRecord] = []
    for index in range(games):
        seat_our = index % 2
        game_seed = int(game_seeds[index]) if game_seeds is not None else None
        rng = random.Random(game_seed) if game_seed is not None else shared_rng
        record = run_real_meta_game(
            our_agent, opp_agent, our_deck, opp_deck,
            seat_our=seat_our, max_steps=max_steps, rng=rng,
            game_index=index, rng_seed=game_seed if game_seed is not None else seed,
        )
        records.append(record)
        if record.illegal:
            illegal += 1
            if record.error and len(errors) < 5:
                errors.append(record.error)
            break
        if record.crashed:
            crashes += 1
            if record.error and len(errors) < 5:
                errors.append(record.error)
            if stop_on_safety_failure:
                break
            continue
        if record.timed_out:
            timeouts += 1
            if record.error and len(errors) < 5:
                errors.append(record.error)
            if stop_on_safety_failure:
                break
            continue
        completed += 1
        turns += record.turns
        if record.result == 2:
            draws += 1
        elif record.result == seat_our:
            our_wins += 1
        else:
            opp_wins += 1

    attempted = len(records)
    return MatchupSummary(
        our_label=our_agent.label,
        opp_label=opp_agent.label,
        opp_deck_hash=opp_deck_hash,
        opp_archetype=opp_archetype,
        bucket=bucket,
        games=games,
        completed=completed,
        our_wins=our_wins,
        opp_wins=opp_wins,
        draws=draws,
        crashes=crashes,
        timeouts=timeouts,
        avg_turns=turns / completed if completed else 0.0,
        elapsed_seconds=time.perf_counter() - started,
        errors=errors,
        illegal=illegal,
        attempted=attempted,
        stopped_early=attempted < games,
        records=records,
    )
