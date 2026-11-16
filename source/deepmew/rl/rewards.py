"""Reward functions, ablated individually.

RL0 uses ``terminal_win_loss`` only: +1 win / -1 loss / 0 draw, plus an
illegal/crash/timeout penalty. No card-semantic or hand-authored strategic
reward is allowed in RL0. Shaping (e.g. prize differential) is an additive
module enabled by config; the trainer never changes when the reward swaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _prizes_remaining(obs: dict[str, Any] | None, our_seat: int) -> int | None:
    if not isinstance(obs, dict):
        return None
    current = obs.get("current") or {}
    players = current.get("players") or []
    if len(players) < 2 or our_seat not in (0, 1):
        return None
    me = players[our_seat] if isinstance(players[our_seat], dict) else {}
    prize = me.get("prize")
    return len(prize) if isinstance(prize, list) else None


def _opp_prizes_remaining(obs: dict[str, Any] | None, our_seat: int) -> int | None:
    if not isinstance(obs, dict):
        return None
    current = obs.get("current") or {}
    players = current.get("players") or []
    if len(players) < 2 or our_seat not in (0, 1):
        return None
    opp = players[1 - our_seat] if isinstance(players[1 - our_seat], dict) else {}
    prize = opp.get("prize")
    return len(prize) if isinstance(prize, list) else None


@dataclass(frozen=True)
class RewardContext:
    prev_obs: dict[str, Any] | None
    next_obs: dict[str, Any] | None
    our_seat: int
    done: bool
    result: int | None  # winning seat (0/1), 2 for draw, None if unfinished
    crashed: bool = False
    timed_out: bool = False
    illegal_attempts: int = 0


class RewardFn(Protocol):
    name: str

    def __call__(self, ctx: RewardContext) -> float: ...


@dataclass(frozen=True)
class TerminalWinLoss:
    """+1 win / -1 loss / 0 draw, with illegal/crash/timeout penalties only."""

    name: str = "terminal_win_loss"
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    crash_penalty: float = -1.0
    timeout_penalty: float = -1.0
    illegal_penalty: float = -0.05

    def __call__(self, ctx: RewardContext) -> float:
        reward = self.illegal_penalty * max(0, ctx.illegal_attempts)
        if ctx.crashed:
            return reward + self.crash_penalty
        if ctx.timed_out:
            return reward + self.timeout_penalty
        if not ctx.done:
            return reward
        if ctx.result == 2 or ctx.result is None:
            return reward + self.draw
        return reward + (self.win if ctx.result == ctx.our_seat else self.loss)


@dataclass(frozen=True)
class PrizeDeltaShaping:
    """Terminal win/loss + a small per-step prize-differential shaping term.

    Wraps a terminal reward and adds ``coef * (my_prizes_taken -
    opp_prizes_taken)`` for the step, where prizes_taken = prev_remaining -
    next_remaining (taking your OWN prizes, i.e. your remaining pile shrinking, is
    progress toward winning). This is the only shaping enabled at RL2; it is
    ablated as a unit and computed from public prize counts only.
    """

    base: RewardFn
    coef: float = 0.25
    name: str = "terminal_win_loss+prize_delta"

    def __call__(self, ctx: RewardContext) -> float:
        reward = self.base(ctx)
        my_prev = _prizes_remaining(ctx.prev_obs, ctx.our_seat)
        my_next = _prizes_remaining(ctx.next_obs, ctx.our_seat)
        opp_prev = _opp_prizes_remaining(ctx.prev_obs, ctx.our_seat)
        opp_next = _opp_prizes_remaining(ctx.next_obs, ctx.our_seat)
        if None in (my_prev, my_next, opp_prev, opp_next):
            return reward
        my_taken = max(0, my_prev - my_next)  # type: ignore[operator]
        opp_taken = max(0, opp_prev - opp_next)  # type: ignore[operator]
        return reward + self.coef * (my_taken - opp_taken)


# Game constants that bound each potential term to roughly [-1, 1] before weighting.
MAX_PRIZES = 6
MAX_BENCH = 5
MAX_POKEMON_IN_PLAY = 6  # 1 active + 5 bench
ENERGY_TEMPO_CAP = 12  # matches the public_belief opp_revealed_energy cap


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _player(obs: dict[str, Any] | None, seat: int) -> dict[str, Any] | None:
    if not isinstance(obs, dict):
        return None
    current = obs.get("current") or {}
    players = current.get("players") or []
    if len(players) < 2 or seat not in (0, 1):
        return None

    return players[seat] if isinstance(players[seat], dict) else {}


def _active_hp_ratio(active: list) -> float | None:
    pokemon = active[0] if active and isinstance(active[0], dict) else {}
    max_hp = int(pokemon.get("maxHp") or 0)
    if max_hp <= 0:
        return None
    cur = int(pokemon.get("hp")) if pokemon.get("hp") is not None else max(0, max_hp - int(pokemon.get("damage") or 0))

    return max(0.0, min(1.0, cur / max_hp))


def _energy_in_play(active: list, bench: list) -> int:
    total = 0
    for pokemon in active + bench:
        if isinstance(pokemon, dict):
            total += len(_as_list(pokemon.get("energyCards")) or _as_list(pokemon.get("energies")))

    return total


@dataclass(frozen=True)
class PlayerBoardView:
    """Public, bounded board facts for one player, read from a player dict."""

    prizes: int
    bench: int
    pokemon_in_play: int
    energy_in_play: int
    active_hp_ratio: float | None


def _player_board_view(obs: dict[str, Any] | None, seat: int) -> PlayerBoardView | None:
    player = _player(obs, seat)
    if player is None:
        return None

    active = _as_list(player.get("active"))
    bench = _as_list(player.get("bench"))
    has_active = 1 if active else 0

    return PlayerBoardView(
        prizes=len(_as_list(player.get("prize"))),
        bench=len(bench),
        pokemon_in_play=has_active + len(bench),
        energy_in_play=_energy_in_play(active, bench),
        active_hp_ratio=_active_hp_ratio(active),
    )


@dataclass(frozen=True)
class PotentialShapingWeights:
    """Named, owned coefficients for the potential terms.

    Because the shaping is potential-based (Ng 1999), these weights do NOT change
    the optimal policy; they only shape learning speed. `coef` scales the whole
    potential so the per-step shaping magnitude stays well below the +/-1 terminal.
    """

    coef: float = 0.1
    prize_advantage: float = 1.0
    bench_presence: float = 0.3
    board_presence: float = 0.3
    energy_tempo: float = 0.2
    active_hp: float = 0.3


def _hp_or_zero(ratio: float | None) -> float:
    return ratio if ratio is not None else 0.0


def _board_potential(obs: dict[str, Any] | None, our_seat: int, weights: PotentialShapingWeights) -> float:
    me = _player_board_view(obs, our_seat)
    opp = _player_board_view(obs, 1 - our_seat)
    if me is None or opp is None:
        return 0.0

    prize = weights.prize_advantage * (opp.prizes - me.prizes) / MAX_PRIZES
    bench = weights.bench_presence * (me.bench - opp.bench) / MAX_BENCH
    board = weights.board_presence * (me.pokemon_in_play - opp.pokemon_in_play) / MAX_POKEMON_IN_PLAY
    energy = weights.energy_tempo * (me.energy_in_play - opp.energy_in_play) / ENERGY_TEMPO_CAP
    active_hp = weights.active_hp * (_hp_or_zero(me.active_hp_ratio) - _hp_or_zero(opp.active_hp_ratio))

    return weights.coef * (prize + bench + board + energy + active_hp)


@dataclass(frozen=True)
class PotentialBasedShaping:
    """Terminal win/loss + Ng-1999 potential-based shaping ``gamma*Phi(s') - Phi(s)``.

    Optimal policy is provably unchanged (the discounted shaping return telescopes
    to the constant ``-Phi(s_0)``). ``Phi`` is bounded and PUBLIC-only (prize /
    bench / board / energy / active-hp differential); ``Phi(terminal) == 0`` so the
    +/-1 terminal signal is never distorted. ``gamma`` MUST equal the trainer's GAE
    ``gamma`` (``MaskedPPOConfig.gamma``) or the invariance breaks; a test guards it.
    """

    base: RewardFn
    weights: PotentialShapingWeights = PotentialShapingWeights()
    gamma: float = 0.99
    name: str = "terminal_win_loss+potential_shaping"

    def __call__(self, ctx: RewardContext) -> float:
        reward = self.base(ctx)
        phi_prev = _board_potential(ctx.prev_obs, ctx.our_seat, self.weights)
        phi_next = 0.0 if ctx.done else _board_potential(ctx.next_obs, ctx.our_seat, self.weights)

        return reward + self.gamma * phi_next - phi_prev


def make_reward(name: str) -> RewardFn:
    if name == "terminal_win_loss":
        return TerminalWinLoss()
    if name == "terminal_win_loss+prize_delta":
        return PrizeDeltaShaping(base=TerminalWinLoss())
    if name == "terminal_win_loss+potential_shaping":
        return PotentialBasedShaping(base=TerminalWinLoss())
    raise ValueError(f"unknown reward function: {name!r}")
