"""Evaluator: play a policy against a fixed opponent suite, report per bucket.

The evaluator never updates the policy — it only plays the frozen exam arena and
reports per-opponent win rates (the "actual play strength" signal DEC-040 cares
about). For the v0 scaffold the policy under test is the untrained
`MaskedRandomPolicy` (no trained checkpoint exists yet); a real backend would
load a checkpoint and pass its greedy/sampled policy through the same interface.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from deepmew.rl.action_mask import ActionMasker
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.deck_identity import agent_deck_identity_from_config, read_deck_any
from deepmew.rl.env_adapter import EnvAdapter
from deepmew.rl.observation import ObservationEncoder
from deepmew.rl.opponents import OpponentAgent, make_eval_suite
from deepmew.rl.registry import build_encoder, build_masker, build_reward
from deepmew.rl.rewards import RewardFn
from deepmew.rl.trainer import MaskedRandomPolicy, Policy, play_episode


@dataclass
class EvalResult:
    experiment_id: str
    policy_name: str
    games_per_opponent: int
    observation_dim: int
    per_opponent: dict[str, dict[str, Any]] = field(default_factory=dict)
    crashes: int = 0
    timeouts: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "policy_name": self.policy_name,
            "games_per_opponent": self.games_per_opponent,
            "observation_dim": self.observation_dim,
            "per_opponent": self.per_opponent,
            "crashes": self.crashes,
            "timeouts": self.timeouts,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


class Evaluator:
    def __init__(
        self,
        config: RLExperimentConfig,
        *,
        encoder: ObservationEncoder | None = None,
        masker: ActionMasker | None = None,
        reward_fn: RewardFn | None = None,
        suite: list[OpponentAgent] | None = None,
        seed: int = 0,
        max_steps_per_game: int | None = None,
    ):
        self.config = config
        self.encoder = encoder or build_encoder(config)
        self.masker = masker or build_masker(config)
        self.reward_fn = reward_fn or build_reward(config)
        self.suite = suite if suite is not None else make_eval_suite()
        if suite is None:
            self.suite = make_eval_suite(config.opponent_sampler)
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps_per_game = max_steps_per_game or config.max_steps_per_game
        self.agent_deck_identity = agent_deck_identity_from_config(config)
        self.our_deck = read_deck_any(config.agent_deck_path)

    def evaluate(self, *, games_per_opponent: int, policy: Policy | None = None) -> EvalResult:
        policy = policy or MaskedRandomPolicy(rng=random.Random(self.seed + 7))
        policy_name = type(policy).__name__
        result = EvalResult(
            experiment_id=self.config.experiment_id,
            policy_name=policy_name,
            games_per_opponent=games_per_opponent,
            observation_dim=self.encoder.schema.dim,
        )
        started = time.perf_counter()
        for opponent in self.suite:
            identity = opponent.deck_identity.to_dict()
            bucket = {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "crashes": 0,
                "timeouts": 0,
                "agent_policy_id": self.config.experiment_id,
                "agent_deck_id": self.agent_deck_identity.deck_id,
                "agent_deck_hash": self.agent_deck_identity.deck_hash,
                "opponent_policy_id": identity["policy_id"],
                "opponent_deck_id": identity["deck_id"],
                "opponent_deck_hash": identity["deck_hash"],
                "opponent_pairing_role": identity["pairing_role"],
                "opponent_deck_status": identity["deck_status"],
                "deck_source_path": identity["deck_source_path"],
            }
            for game_index in range(games_per_opponent):
                our_seat = game_index % 2
                env = EnvAdapter(
                    opponent=opponent,
                    our_seat=our_seat,
                    our_deck=self.our_deck,
                    opp_deck=opponent.deck,
                    max_steps=self.max_steps_per_game,
                    opponent_seed=self.rng.randint(0, 2**31 - 1),
                )
                outcome = play_episode(
                    env,
                    encoder=self.encoder,
                    masker=self.masker,
                    reward_fn=self.reward_fn,
                    policy=policy,
                    our_seat=our_seat,
                )
                bucket["games"] += 1
                if outcome.crashed:
                    bucket["crashes"] += 1
                    result.crashes += 1
                elif outcome.timed_out:
                    bucket["timeouts"] += 1
                    result.timeouts += 1
                elif outcome.is_win:
                    bucket["wins"] += 1
                elif outcome.is_loss:
                    bucket["losses"] += 1
                elif outcome.is_draw:
                    bucket["draws"] += 1
            completed = bucket["wins"] + bucket["losses"] + bucket["draws"]
            bucket["win_rate"] = round(bucket["wins"] / completed, 6) if completed else 0.0
            result.per_opponent[opponent.label] = bucket
        result.elapsed_seconds = time.perf_counter() - started
        return result
