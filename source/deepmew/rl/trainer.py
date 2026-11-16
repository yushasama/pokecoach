"""Algorithm backends behind a single, narrow training interface.

`TrainerBackend` (aliased `AlgorithmBackend`) is the **algorithm** abstraction:
the thing that consumes `RLComponents` (env adapter + encoder + masker + reward +
opponent sampler + policy) and turns experience into a policy update. It is
deliberately NOT a "PPO project": the same interface is meant to host PPO, A2C,
IMPALA/V-trace, offline RL, behavior cloning, ranking, IL+RL hybrids, and a
PufferLib / CleanRL / SB3 backend later. The env adapter never learns which
algorithm runs.

No real RL dependency is vendored — SB3 / CleanRL / PufferLib / gym are not
installed, and pulling one in silently would be a dependency explosion. The
shipped `SmokeTrainer` is dependency-free: it drives the real
env/encoder/mask/reward/opponent objects with an untrained masked policy to
prove the interfaces are real enough for a gradient backend to slot in, WITHOUT
starting any long training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from deepmew.rl.action_mask import ActionMask, ActionMasker
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.deck_identity import agent_deck_identity_from_config, read_deck_any
from deepmew.rl.env_adapter import EnvAdapter, StepResult
from deepmew.rl.observation import ObservationEncoder
from deepmew.rl.rewards import RewardContext, RewardFn


class Policy(Protocol):
    def act(self, features: list[float], mask: ActionMask) -> list[int]: ...


class DirectSelectionPolicy(Protocol):
    def select(
        self,
        features: list[float],
        mask: ActionMask,
        select: dict[str, Any],
    ) -> list[int]: ...


@dataclass
class MaskedRandomPolicy:
    """Untrained control policy: a uniform-random order over LEGAL options only.

    Returns a best-first priority over option indices restricted to the legal
    set, so the `ActionMasker` can only ever yield a legal selection. Ignores
    `features` (no parameters) but receives them so the encode->policy->mask path
    is exercised exactly as a trained net would use it.
    """

    rng: random.Random

    def act(self, features: list[float], mask: ActionMask) -> list[int]:
        legal = mask.legal_indices()
        self.rng.shuffle(legal)
        return legal


@dataclass(frozen=True)
class RLComponents:
    config: RLExperimentConfig
    encoder: ObservationEncoder
    masker: ActionMasker
    reward_fn: RewardFn
    opponent_sampler: Any


@dataclass
class EpisodeOutcome:
    our_seat: int
    decision_steps: int
    reward: float
    illegal_attempts: int
    result: int | None
    crashed: bool
    timed_out: bool
    partial: bool

    @property
    def is_win(self) -> bool:
        return (
            not self.partial
            and not self.crashed
            and not self.timed_out
            and self.result is not None
            and self.result != 2
            and self.result == self.our_seat
        )

    @property
    def is_loss(self) -> bool:
        return (
            not self.partial
            and not self.crashed
            and not self.timed_out
            and self.result is not None
            and self.result != 2
            and self.result != self.our_seat
        )

    @property
    def is_draw(self) -> bool:
        return (
            not self.partial
            and not self.crashed
            and not self.timed_out
            and self.result == 2
        )


def play_episode(
    env: EnvAdapter,
    *,
    encoder: ObservationEncoder,
    masker: ActionMasker,
    reward_fn: RewardFn,
    policy: Policy,
    our_seat: int,
    step_budget: int | None = None,
) -> EpisodeOutcome:
    """Run one episode from reset to terminal (or until `step_budget` is hit).

    Shared by `SmokeTrainer` and `Evaluator` so the rollout loop is defined once.
    Every selection passes through the masker (INVARIANT #1: legal-only).
    """
    step = env.reset()
    prev_obs = step.obs
    decision_steps = 0
    reward_total = 0.0
    illegal_total = 0
    partial = False

    while not step.done:
        select = env.select
        features = encoder.encode(step.obs or {})
        mask = masker.build_mask(select)
        direct_select = getattr(policy, "select", None)
        if callable(direct_select):
            selection = masker.sanitize(direct_select(features, mask, select or {}), select or {})
        else:
            priority = policy.act(features, mask)
            selection = masker.select_from_priority(priority, select or {})
        illegal = 0 if masker.is_legal(selection, select or {}) else 1
        next_step = env.step(selection)
        reward_total += reward_fn(
            RewardContext(
                prev_obs=prev_obs,
                next_obs=next_step.obs if next_step.obs is not None else step.obs,
                our_seat=our_seat,
                done=next_step.done,
                result=next_step.result,
                crashed=next_step.crashed,
                timed_out=next_step.timed_out,
                illegal_attempts=illegal,
            )
        )
        decision_steps += 1
        illegal_total += illegal
        prev_obs = step.obs
        step = next_step
        if step_budget is not None and decision_steps >= step_budget and not step.done:
            env.close()
            partial = True
            break

    return EpisodeOutcome(
        our_seat=our_seat,
        decision_steps=decision_steps,
        reward=reward_total,
        illegal_attempts=illegal_total,
        result=step.result,
        crashed=step.crashed,
        timed_out=step.timed_out,
        partial=partial,
    )


@dataclass
class RLRunResult:
    experiment_id: str
    backend: str
    dry_run: bool
    episodes: int = 0
    decision_steps: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    crashes: int = 0
    timeouts: int = 0
    illegal_attempts: int = 0
    total_reward: float = 0.0
    observation_dim: int = 0
    per_opponent: dict[str, dict[str, int]] = field(default_factory=dict)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.episodes if self.episodes else 0.0

    def record(self, outcome: EpisodeOutcome, opponent_label: str, opponent_identity: dict[str, Any] | None = None) -> None:
        bucket = self.per_opponent.setdefault(
            opponent_label,
            {"episodes": 0, "wins": 0, "losses": 0, "draws": 0, "crashes": 0, "timeouts": 0},
        )
        if opponent_identity:
            for source, target in {
                "policy_id": "opponent_policy_id",
                "deck_id": "opponent_deck_id",
                "deck_hash": "opponent_deck_hash",
                "pairing_role": "opponent_pairing_role",
                "deck_status": "opponent_deck_status",
                "deck_source_path": "deck_source_path",
            }.items():
                bucket.setdefault(target, opponent_identity.get(source, ""))
        bucket["episodes"] += 1
        self.episodes += 1
        self.decision_steps += outcome.decision_steps
        self.illegal_attempts += outcome.illegal_attempts
        self.total_reward += outcome.reward
        if outcome.crashed:
            self.crashes += 1
            bucket["crashes"] += 1
        elif outcome.timed_out:
            self.timeouts += 1
            bucket["timeouts"] += 1
        elif outcome.is_win:
            self.wins += 1
            bucket["wins"] += 1
        elif outcome.is_loss:
            self.losses += 1
            bucket["losses"] += 1
        elif outcome.is_draw:
            self.draws += 1
            bucket["draws"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "backend": self.backend,
            "dry_run": self.dry_run,
            "episodes": self.episodes,
            "completed": self.completed,
            "decision_steps": self.decision_steps,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "crashes": self.crashes,
            "timeouts": self.timeouts,
            "illegal_attempts": self.illegal_attempts,
            "total_reward": round(self.total_reward, 6),
            "mean_reward": round(self.mean_reward, 6),
            "observation_dim": self.observation_dim,
            "per_opponent": self.per_opponent,
            "training_metrics": self.training_metrics,
            "notes": self.notes,
        }


class TrainerBackend(Protocol):
    name: str

    def run(self, *, total_steps: int, dry_run: bool) -> RLRunResult: ...


# The algorithm-backend abstraction. `TrainerBackend` is the historical name;
# `AlgorithmBackend` is the preferred alias going forward (PPO/A2C/IMPALA/offline
# RL/BC/ranking/Puffer/CleanRL/SB3 all implement this same protocol).
AlgorithmBackend = TrainerBackend


class SmokeTrainer:
    """Dependency-free backend that exercises the full loop for a few steps."""

    name = "smoke"

    def __init__(
        self,
        components: RLComponents,
        *,
        seed: int = 0,
        max_steps_per_game: int | None = None,
    ):
        self.components = components
        self.seed = seed
        self.rng = random.Random(seed)
        self.policy: Policy = MaskedRandomPolicy(rng=random.Random(seed + 1))
        self.max_steps_per_game = (
            max_steps_per_game
            if max_steps_per_game is not None
            else components.config.max_steps_per_game
        )
        self.agent_deck_identity = agent_deck_identity_from_config(components.config)
        self.our_deck = read_deck_any(components.config.agent_deck_path)

    def run(self, *, total_steps: int, dry_run: bool) -> RLRunResult:
        comp = self.components
        result = RLRunResult(
            experiment_id=comp.config.experiment_id,
            backend=self.name,
            dry_run=dry_run,
            observation_dim=comp.encoder.schema.dim,
        )
        comp.opponent_sampler.reset(self.seed)
        if dry_run:
            result.notes.append("dry-run: components assembled, no episodes played")
            return result

        episode_index = 0
        while result.decision_steps < total_steps:
            sampled = comp.opponent_sampler.sample()
            our_seat = episode_index % 2
            env = EnvAdapter(
                opponent=sampled.agent,
                our_seat=our_seat,
                our_deck=self.our_deck,
                opp_deck=sampled.agent.deck,
                max_steps=self.max_steps_per_game,
                opponent_seed=self.rng.randint(0, 2**31 - 1),
            )
            remaining = total_steps - result.decision_steps
            outcome = play_episode(
                env,
                encoder=comp.encoder,
                masker=comp.masker,
                reward_fn=comp.reward_fn,
                policy=self.policy,
                our_seat=our_seat,
                step_budget=remaining,
            )
            result.record(outcome, sampled.agent.label, sampled.agent.deck_identity.to_dict())
            episode_index += 1
        return result


TRAINER_BACKENDS: dict[str, type] = {"smoke": SmokeTrainer}


def make_trainer(
    backend: str,
    components: RLComponents,
    *,
    seed: int = 0,
    **kwargs: Any,
) -> TrainerBackend:
    if backend == "masked_ppo":
        from deepmew.rl.algorithms.masked_ppo import MaskedPPOTrainer

        return MaskedPPOTrainer(components, seed=seed, **kwargs)
    if backend in {"masked_ppo_puffer_advantage", "puffer_advantage_cpu", "puffer"}:
        from deepmew.rl.algorithms.puffer_backend import PufferAdvantageBackend

        return PufferAdvantageBackend(components, seed=seed, requested_backend=backend, **kwargs)
    if backend == "puffer_vectorized":
        from deepmew.rl.algorithms.puffer_backend import PufferVectorizedBackend

        return PufferVectorizedBackend(components, seed=seed, **kwargs)
    if backend not in TRAINER_BACKENDS:
        raise ValueError(
            f"unknown algorithm/trainer backend {backend!r}; available: {sorted(TRAINER_BACKENDS)}. "
            "Available lazy backends: ['masked_ppo', 'masked_ppo_puffer_advantage', 'puffer_vectorized']."
        )
    return TRAINER_BACKENDS[backend](components, seed=seed)
