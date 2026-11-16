"""Transparent masked PPO reference backend.

This is intentionally small and inspectable. It supports the live DeepMew
selection contract: a decision is a bounded list of option indices, not always a
single categorical action. The policy samples an autoregressive sequence with a
STOP token when `minCount <= len(selection) < maxCount`, and PPO recomputes the
exact log probability of that full submitted selection.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from deepmew.policy.selection import valid_selection
from deepmew.rl.action_mask import ActionMask, ActionMasker
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.decks import read_config_deck
from deepmew.rl.env_adapter import EnvAdapter, StepResult
from deepmew.rl.metadata import build_run_metadata
from deepmew.rl.observation import ObservationEncoder
from deepmew.rl.rewards import RewardContext, RewardFn
from deepmew.rl.trainer import EpisodeOutcome, RLComponents, RLRunResult


class PPOValidationError(RuntimeError):
    """Raised when a numerical or action-contract invariant is violated."""


class TrainEnv(Protocol):
    our_seat: int
    opponent_label: str

    @property
    def select(self) -> dict[str, Any] | None: ...

    def reset(self) -> StepResult: ...

    def step(self, selection: list[int]) -> StepResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class MaskedPPOConfig:
    obs_dim: int
    action_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 128
    rollout_steps: int = 256
    ent_coef: float = 0.01
    # Anti-collapse: when set, add a hinge penalty entropy_floor_coef * relu(floor - mean_entropy)
    # so policy entropy does not crater toward zero early (exploration lever, RL4 arm B/F).
    entropy_floor: float | None = None
    entropy_floor_coef: float = 0.5
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    seed: int = 0
    device: str = "cpu"
    advantage_backend: str = "torch"


@dataclass
class PPOUpdateStats:
    update_count: int = 0
    rollout_steps: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approximate_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    gradient_norm: float = 0.0
    parameter_norm: float = 0.0
    parameter_delta_norm: float = 0.0
    learner_update_seconds: float = 0.0
    advantage_backend: str = "torch"

    def to_dict(self) -> dict[str, Any]:
        return {key: round(value, 6) if isinstance(value, float) else value for key, value in asdict(self).items()}


@dataclass
class PPOTrainingState:
    global_step: int = 0
    update_count: int = 0
    episodes: int = 0
    best_eval_score: float | None = None
    last_update: PPOUpdateStats = field(default_factory=PPOUpdateStats)


@dataclass
class RolloutBatch:
    obs: list[list[float]] = field(default_factory=list)
    masks: list[ActionMask] = field(default_factory=list)
    selects: list[dict[str, Any]] = field(default_factory=list)
    actions: list[list[int]] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    seats: list[int] = field(default_factory=list)
    opponents: list[str] = field(default_factory=list)
    contexts: list[Any] = field(default_factory=list)
    env_wait_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stop_index = action_dim
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim + 1)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        _assert_finite("observation", obs)
        hidden = self.body(obs)
        logits = self.policy(hidden)
        values = self.value(hidden).squeeze(-1)
        _assert_finite("policy_logits", logits)
        _assert_finite("values", values)
        return logits, values


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise PPOValidationError(f"{name} contains NaN/Inf")


def _parameter_norm(model: nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        total = total + torch.sum(param.detach() ** 2)
    return float(torch.sqrt(total).detach().cpu())


def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1).cpu() for param in model.parameters()])


def _masked_logits(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.dtype != torch.bool:
        valid = valid.bool()
    if not bool(valid.any()):
        raise PPOValidationError("all-invalid action mask")
    masked = logits.masked_fill(~valid, -1.0e9)
    _assert_finite("masked_logits", masked)
    return masked


def _valid_next_tokens(mask: ActionMask, chosen: set[int], action_dim: int) -> torch.Tensor:
    valid = torch.zeros(action_dim + 1, dtype=torch.bool)
    selected_count = len(chosen)
    if selected_count >= mask.max_count:
        valid[action_dim] = True
        return valid
    for index in range(min(mask.n_options, action_dim)):
        if mask.legal[index] and index not in chosen:
            valid[index] = True
    if selected_count >= mask.min_count:
        valid[action_dim] = True
    return valid


@dataclass(frozen=True)
class ActionSample:
    selection: list[int]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class AutoregressiveSelectionDistribution:
    def __init__(self, model: ActorCritic, config: MaskedPPOConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

    def sample(
        self,
        obs: Sequence[float],
        mask: ActionMask,
        *,
        deterministic: bool = False,
    ) -> ActionSample:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, values = self.model(obs_tensor)
        logits = logits[0]
        value = values[0]
        chosen: list[int] = []
        chosen_set: set[int] = set()
        log_prob = torch.zeros((), device=self.device)
        entropy = torch.zeros((), device=self.device)

        while True:
            valid = _valid_next_tokens(mask, chosen_set, self.config.action_dim).to(self.device)
            masked = _masked_logits(logits, valid)
            dist = Categorical(logits=masked)
            token = torch.argmax(masked) if deterministic else dist.sample()
            _assert_finite("action_log_prob", dist.log_prob(token))
            log_prob = log_prob + dist.log_prob(token)
            entropy = entropy + dist.entropy()
            index = int(token.detach().cpu())
            if index == self.config.action_dim:
                break
            chosen.append(index)
            chosen_set.add(index)
            if len(chosen) >= mask.max_count:
                break
            if len(chosen) > self.config.action_dim:
                raise PPOValidationError("autoregressive selection exceeded action dimension")

        return ActionSample(selection=chosen, log_prob=log_prob, entropy=entropy, value=value)

    def evaluate(
        self,
        obs_batch: torch.Tensor,
        masks: Sequence[ActionMask],
        actions: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_batch, values = self.model(obs_batch)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for logits, mask, action in zip(logits_batch, masks, actions):
            chosen_set: set[int] = set()
            log_prob = torch.zeros((), device=self.device)
            entropy = torch.zeros((), device=self.device)
            for raw_index in action:
                index = int(raw_index)
                valid = _valid_next_tokens(mask, chosen_set, self.config.action_dim).to(self.device)
                if index >= self.config.action_dim or not bool(valid[index]):
                    raise PPOValidationError(
                        f"stored action {list(action)} disagrees with mask n={mask.n_options} "
                        f"min={mask.min_count} max={mask.max_count}"
                    )
                dist = Categorical(logits=_masked_logits(logits, valid))
                token = torch.as_tensor(index, dtype=torch.long, device=self.device)
                log_prob = log_prob + dist.log_prob(token)
                entropy = entropy + dist.entropy()
                chosen_set.add(index)

            if len(action) < mask.max_count:
                valid = _valid_next_tokens(mask, chosen_set, self.config.action_dim).to(self.device)
                stop = self.config.action_dim
                if not bool(valid[stop]):
                    raise PPOValidationError(
                        f"stored action {list(action)} stopped before minCount={mask.min_count}"
                    )
                dist = Categorical(logits=_masked_logits(logits, valid))
                token = torch.as_tensor(stop, dtype=torch.long, device=self.device)
                log_prob = log_prob + dist.log_prob(token)
                entropy = entropy + dist.entropy()
            log_probs.append(log_prob)
            entropies.append(entropy)

        out_log_probs = torch.stack(log_probs)
        out_entropies = torch.stack(entropies)
        _assert_finite("recomputed_log_probs", out_log_probs)
        _assert_finite("recomputed_entropies", out_entropies)
        return out_log_probs, out_entropies, values


class PPOPolicy:
    """Frozen policy object usable by `play_episode` / `Evaluator`."""

    def __init__(
        self,
        model: ActorCritic,
        config: MaskedPPOConfig,
        *,
        deterministic: bool = True,
    ):
        self.model = model
        self.config = config
        self.deterministic = deterministic
        self.dist = AutoregressiveSelectionDistribution(model, config)

    def select(self, features: list[float], mask: ActionMask, select: dict[str, Any]) -> list[int]:
        with torch.no_grad():
            sample = self.dist.sample(features, mask, deterministic=self.deterministic)
        if not valid_selection(sample.selection, select):
            raise PPOValidationError(f"policy produced illegal selection {sample.selection}")
        return sample.selection

    def act(self, features: list[float], mask: ActionMask) -> list[int]:
        with torch.no_grad():
            return self.dist.sample(features, mask, deterministic=self.deterministic).selection


class SampledBattleEnv:
    """EnvAdapter wrapper that samples a fresh opponent/seat on every reset."""

    def __init__(
        self,
        components: RLComponents,
        *,
        seed: int,
        max_steps_per_game: int,
    ):
        self.components = components
        self.rng = random.Random(seed)
        self.max_steps_per_game = max_steps_per_game
        self.our_deck = read_config_deck(components.config, role="agent")
        self._env: EnvAdapter | None = None
        self.episode_index = 0
        self.our_seat = 0
        self.opponent_label = "unknown"

    @property
    def select(self) -> dict[str, Any] | None:
        return self._env.select if self._env is not None else None

    def reset(self) -> StepResult:
        self.close()
        sampled = self.components.opponent_sampler.sample()
        self.our_seat = self.episode_index % 2
        self.opponent_label = sampled.agent.label
        self.episode_index += 1
        self._env = EnvAdapter(
            opponent=sampled.agent,
            our_seat=self.our_seat,
            our_deck=self.our_deck,
            max_steps=self.max_steps_per_game,
            opponent_seed=self.rng.randint(0, 2**31 - 1),
        )
        return self._env.reset()

    def step(self, selection: list[int]) -> StepResult:
        if self._env is None:
            return StepResult(
                obs=None,
                done=True,
                result=None,
                crashed=True,
                info={"error": "step before reset"},
            )
        return self._env.step(selection)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        for attempt in range(20):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def save_checkpoint(
    path: Path | str,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    algo_config: MaskedPPOConfig,
    experiment_config: RLExperimentConfig | None,
    state: PPOTrainingState,
    schema_names: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": "deepmew.masked_ppo_checkpoint.v0",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "algo_config": asdict(algo_config),
        "experiment_config": experiment_config.to_dict() if experiment_config is not None else None,
        "state": asdict(state),
        "schema_names": list(schema_names),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "saved_at": time.time(),
        "extra": extra or {},
    }
    _atomic_torch_save(payload, Path(path))


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_model_from_checkpoint(path: Path | str, *, device: str = "cpu") -> tuple[ActorCritic, MaskedPPOConfig, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=device)
    algo_config = MaskedPPOConfig(**{**payload["algo_config"], "device": device})
    model = ActorCritic(algo_config.obs_dim, algo_config.action_dim, algo_config.hidden_dim)
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(device))
    model.eval()
    return model, algo_config, payload


def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y.detach().cpu()) <= 1e-12:
        return 0.0
    residual_var = torch.var(y_true - y_pred, unbiased=False)
    return float((1.0 - residual_var / (var_y + 1e-8)).detach().cpu())


def _compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[step]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    _assert_finite("advantages", advantages)
    _assert_finite("returns", returns)
    return advantages, returns


def _compute_advantages(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    algo_config: MaskedPPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if algo_config.advantage_backend == "torch":
        return _compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_value=next_value,
            gamma=algo_config.gamma,
            gae_lambda=algo_config.gae_lambda,
        )
    if algo_config.advantage_backend == "puffer_cpu":
        from deepmew.rl.algorithms.puffer_support import compute_puffer_cpu_gae

        return compute_puffer_cpu_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_value=next_value,
            gamma=algo_config.gamma,
            gae_lambda=algo_config.gae_lambda,
        )
    raise PPOValidationError(f"unknown advantage backend: {algo_config.advantage_backend!r}")


def _update_ppo(
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    algo_config: MaskedPPOConfig,
    batch: RolloutBatch,
    next_value: float,
    update_count: int,
) -> PPOUpdateStats:
    device = torch.device(algo_config.device)
    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=device)
    values = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)
    _assert_finite("rollout_rewards", rewards)
    _assert_finite("old_log_probs", old_log_probs)
    _assert_finite("old_values", values)

    advantages, returns = _compute_advantages(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=torch.as_tensor(next_value, dtype=torch.float32, device=device),
        algo_config=algo_config,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    _assert_finite("normalized_advantages", advantages)

    dist = AutoregressiveSelectionDistribution(model, algo_config)
    batch_size = len(batch)
    indices = torch.arange(batch_size, device=device)
    before = _flatten_params(model)
    started = time.perf_counter()

    last_policy_loss = last_value_loss = last_entropy = last_kl = last_clip_fraction = 0.0
    last_grad_norm = 0.0
    for _epoch in range(algo_config.update_epochs):
        perm = indices[torch.randperm(batch_size, device=device)]
        for start in range(0, batch_size, algo_config.minibatch_size):
            mb = perm[start:start + algo_config.minibatch_size]
            mb_list = [int(i.detach().cpu()) for i in mb]
            new_log_probs, entropy, new_values = dist.evaluate(
                obs[mb],
                [batch.masks[i] for i in mb_list],
                [batch.actions[i] for i in mb_list],
            )
            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = torch.exp(log_ratio)
            _assert_finite("ppo_ratio", ratio)

            mb_adv = advantages[mb]
            pg_loss_1 = -mb_adv * ratio
            pg_loss_2 = -mb_adv * torch.clamp(ratio, 1.0 - algo_config.clip_coef, 1.0 + algo_config.clip_coef)
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * ((new_values - returns[mb]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + algo_config.vf_coef * value_loss - algo_config.ent_coef * entropy_loss
            if algo_config.entropy_floor is not None:
                loss = loss + algo_config.entropy_floor_coef * torch.relu(algo_config.entropy_floor - entropy_loss)
            _assert_finite("ppo_loss", loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise PPOValidationError(f"gradient for {name} contains NaN/Inf")
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), algo_config.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise PPOValidationError("gradient norm is NaN/Inf")
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > algo_config.clip_coef).float().mean()
            last_policy_loss = float(policy_loss.detach().cpu())
            last_value_loss = float(value_loss.detach().cpu())
            last_entropy = float(entropy_loss.detach().cpu())
            last_kl = float(approx_kl.detach().cpu())
            last_clip_fraction = float(clip_fraction.detach().cpu())
            last_grad_norm = float(grad_norm.detach().cpu())
        if algo_config.target_kl is not None and last_kl > algo_config.target_kl:
            break

    after = _flatten_params(model)
    delta_norm = float(torch.linalg.vector_norm(after - before).detach().cpu())
    with torch.no_grad():
        _, final_values = model(obs)
    return PPOUpdateStats(
        update_count=update_count,
        rollout_steps=len(batch),
        policy_loss=last_policy_loss,
        value_loss=last_value_loss,
        entropy=last_entropy,
        approximate_kl=last_kl,
        clip_fraction=last_clip_fraction,
        explained_variance=_explained_variance(final_values.detach(), returns.detach()),
        gradient_norm=last_grad_norm,
        parameter_norm=_parameter_norm(model),
        parameter_delta_norm=delta_norm,
        learner_update_seconds=time.perf_counter() - started,
        advantage_backend=algo_config.advantage_backend,
    )


def _reward_from_step(
    reward_fn: RewardFn,
    *,
    prev_obs: dict[str, Any] | None,
    next_step: StepResult,
    current_obs: dict[str, Any] | None,
    our_seat: int,
    illegal: int,
) -> float:
    return float(
        reward_fn(
            RewardContext(
                prev_obs=prev_obs,
                next_obs=next_step.obs if next_step.obs is not None else current_obs,
                our_seat=our_seat,
                done=next_step.done,
                result=next_step.result,
                crashed=next_step.crashed,
                timed_out=next_step.timed_out,
                illegal_attempts=illegal,
            )
        )
    )


def train_masked_ppo(
    *,
    env_fns: Sequence[Callable[[], TrainEnv]],
    encoder: ObservationEncoder,
    masker: ActionMasker,
    reward_fn: RewardFn,
    algo_config: MaskedPPOConfig,
    total_steps: int,
    experiment_config: RLExperimentConfig | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = False,
    checkpoint_extra: dict[str, Any] | None = None,
) -> tuple[RLRunResult, ActorCritic, torch.optim.Optimizer, PPOTrainingState]:
    """Train PPO over one or more independent environments."""
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    _seed_everything(algo_config.seed)
    device = torch.device(algo_config.device)
    model = ActorCritic(algo_config.obs_dim, algo_config.action_dim, algo_config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=algo_config.learning_rate, eps=1e-5)
    state = PPOTrainingState()

    if checkpoint_path is not None and resume and Path(checkpoint_path).exists():
        payload = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        loaded = payload.get("state", {})
        state = PPOTrainingState(
            global_step=int(loaded.get("global_step", 0)),
            update_count=int(loaded.get("update_count", 0)),
            episodes=int(loaded.get("episodes", 0)),
            best_eval_score=loaded.get("best_eval_score"),
        )
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        random.setstate(payload["python_rng_state"])

    result = RLRunResult(
        experiment_id=experiment_config.experiment_id if experiment_config is not None else "controlled_masked_ppo",
        backend="masked_ppo",
        dry_run=False,
        observation_dim=algo_config.obs_dim,
    )
    envs = [fn() for fn in env_fns]
    current_steps = [env.reset() for env in envs]
    episode_reward = [0.0 for _ in envs]
    episode_decisions = [0 for _ in envs]
    episode_illegal = [0 for _ in envs]
    prev_obs = [step.obs for step in current_steps]
    dist = AutoregressiveSelectionDistribution(model, algo_config)
    started = time.perf_counter()
    # Per-chunk accumulator of (explained_variance, value_loss, batch_len). The
    # last update of a chunk can be a tiny tail rollout (e.g. chunk_steps not a
    # multiple of rollout_steps), so single-update EV/value_loss are a noisy,
    # small-sample estimate. We report a batch-weighted chunk mean instead and
    # keep the last-update value under an explicit key for transparency.
    chunk_update_stats: list[tuple[float, float, int]] = []

    try:
        while result.decision_steps < total_steps:
            batch = RolloutBatch()
            rollout_target = min(algo_config.rollout_steps, total_steps - result.decision_steps)
            while len(batch) < rollout_target:
                for env_index, env in enumerate(envs):
                    if len(batch) >= rollout_target:
                        break
                    step = current_steps[env_index]
                    if step.done:
                        current_steps[env_index] = env.reset()
                        prev_obs[env_index] = current_steps[env_index].obs
                        continue
                    obs = step.obs or {}
                    select = env.select or {}
                    features = encoder.encode(obs)
                    if len(features) != algo_config.obs_dim:
                        raise PPOValidationError(
                            f"observation schema mismatch: got {len(features)} expected {algo_config.obs_dim}"
                        )
                    mask = masker.build_mask(select)
                    sample = dist.sample(features, mask)
                    if not valid_selection(sample.selection, select):
                        raise PPOValidationError(f"sampled illegal selection {sample.selection}")

                    env_started = time.perf_counter()
                    next_step = env.step(sample.selection)
                    batch.env_wait_seconds += time.perf_counter() - env_started
                    illegal = 0 if masker.is_legal(sample.selection, select) else 1
                    reward = _reward_from_step(
                        reward_fn,
                        prev_obs=prev_obs[env_index],
                        next_step=next_step,
                        current_obs=obs,
                        our_seat=env.our_seat,
                        illegal=illegal,
                    )
                    batch.obs.append(list(features))
                    batch.masks.append(mask)
                    batch.selects.append(select)
                    batch.actions.append(sample.selection)
                    batch.log_probs.append(float(sample.log_prob.detach().cpu()))
                    batch.values.append(float(sample.value.detach().cpu()))
                    batch.rewards.append(reward)
                    batch.dones.append(next_step.done)
                    batch.seats.append(env.our_seat)
                    batch.opponents.append(env.opponent_label)
                    batch.contexts.append(select.get("context"))

                    result.decision_steps += 1
                    state.global_step += 1
                    episode_decisions[env_index] += 1
                    episode_reward[env_index] += reward
                    episode_illegal[env_index] += illegal
                    prev_obs[env_index] = obs
                    current_steps[env_index] = next_step

                    if next_step.done:
                        outcome = EpisodeOutcome(
                            our_seat=env.our_seat,
                            decision_steps=episode_decisions[env_index],
                            reward=episode_reward[env_index],
                            illegal_attempts=episode_illegal[env_index],
                            result=next_step.result,
                            crashed=next_step.crashed,
                            timed_out=next_step.timed_out,
                            partial=False,
                        )
                        counted_steps = result.decision_steps
                        result.record(outcome, env.opponent_label)
                        # PPO already counts each collected transition as the
                        # step budget. Episode recording owns W/L/reward, but
                        # must not double-count completed episode length.
                        result.decision_steps = counted_steps
                        state.episodes += 1
                        episode_reward[env_index] = 0.0
                        episode_decisions[env_index] = 0
                        episode_illegal[env_index] = 0
                        current_steps[env_index] = env.reset()
                        prev_obs[env_index] = current_steps[env_index].obs

            with torch.no_grad():
                next_features: list[float] | None = None
                for step in current_steps:
                    if not step.done and step.obs is not None:
                        next_features = encoder.encode(step.obs)
                        break
                if next_features is None:
                    next_value = 0.0
                else:
                    _, next_values = model(torch.as_tensor(next_features, dtype=torch.float32, device=device))
                    next_value = float(next_values[0].detach().cpu())
            state.update_count += 1
            update = _update_ppo(
                model=model,
                optimizer=optimizer,
                algo_config=algo_config,
                batch=batch,
                next_value=next_value,
                update_count=state.update_count,
            )
            update.rollout_steps = len(batch)
            state.last_update = update
            result.training_metrics = update.to_dict()
            chunk_update_stats.append((update.explained_variance, update.value_loss, len(batch)))
            result.notes = [
                f"update_count={state.update_count}",
                f"policy_loss={update.policy_loss:.6f}",
                f"value_loss={update.value_loss:.6f}",
                f"entropy={update.entropy:.6f}",
                f"approximate_kl={update.approximate_kl:.6f}",
                f"parameter_delta_norm={update.parameter_delta_norm:.6f}",
                f"env_wait_seconds={batch.env_wait_seconds:.6f}",
                f"advantage_backend={update.advantage_backend}",
            ]
            if checkpoint_path is not None:
                extra = dict(checkpoint_extra or {})
                extra["elapsed_seconds"] = time.perf_counter() - started
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    algo_config=algo_config,
                    experiment_config=experiment_config,
                    state=state,
                    schema_names=encoder.schema.names,
                    extra=extra,
                )
    finally:
        for env in envs:
            env.close()

    if chunk_update_stats:
        total_w = sum(w for _, _, w in chunk_update_stats) or 1
        ev_chunk = sum(ev * w for ev, _, w in chunk_update_stats) / total_w
        vl_chunk = sum(vl * w for _, vl, w in chunk_update_stats) / total_w
        # Preserve the last-update values, then surface the representative
        # batch-weighted chunk means as the primary EV/value_loss.
        result.training_metrics["explained_variance_last_update"] = result.training_metrics.get("explained_variance")
        result.training_metrics["value_loss_last_update"] = result.training_metrics.get("value_loss")
        result.training_metrics["explained_variance"] = round(ev_chunk, 6)
        result.training_metrics["value_loss"] = round(vl_chunk, 6)
        result.training_metrics["chunk_update_count"] = len(chunk_update_stats)
        result.training_metrics["chunk_total_transitions"] = total_w

    elapsed = max(1e-9, time.perf_counter() - started)
    result.notes.append(f"steps_per_sec={result.decision_steps / elapsed:.6f}")
    return result, model, optimizer, state


class MaskedPPOTrainer:
    """Existing TrainerBackend-compatible wrapper for the real engine."""

    name = "masked_ppo"

    def __init__(
        self,
        components: RLComponents,
        *,
        seed: int = 0,
        max_steps_per_game: int | None = None,
        device: str = "cpu",
        checkpoint_path: Path | str | None = None,
        resume: bool = False,
        rollout_steps: int = 256,
        num_envs: int = 1,
        advantage_backend: str = "torch",
        env_vectorization: str = "local",
        ent_coef: float | None = None,
        entropy_floor: float | None = None,
        entropy_floor_coef: float | None = None,
    ):
        self.components = components
        self.seed = seed
        self.max_steps_per_game = max_steps_per_game or components.config.max_steps_per_game
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.resume = resume
        self.rollout_steps = rollout_steps
        self.num_envs = max(1, int(num_envs))
        self.advantage_backend = advantage_backend
        self.env_vectorization = env_vectorization
        self.ent_coef = ent_coef
        self.entropy_floor = entropy_floor
        self.entropy_floor_coef = entropy_floor_coef
        self.model: ActorCritic | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.state: PPOTrainingState | None = None

    def _algo_config(self) -> MaskedPPOConfig:
        defaults = MaskedPPOConfig(obs_dim=1)
        return MaskedPPOConfig(
            obs_dim=self.components.encoder.schema.dim,
            action_dim=self.components.masker.max_options,
            seed=self.seed,
            device=self.device,
            rollout_steps=self.rollout_steps,
            advantage_backend=self.advantage_backend,
            ent_coef=defaults.ent_coef if self.ent_coef is None else self.ent_coef,
            entropy_floor=self.entropy_floor,
            entropy_floor_coef=(
                defaults.entropy_floor_coef if self.entropy_floor_coef is None else self.entropy_floor_coef
            ),
        )

    def _vector_backend_label(self) -> str:
        if self.env_vectorization == "subprocess":
            return "subprocess_vector_env"
        return "single_process_env"

    def _run_metadata(self) -> dict[str, Any]:
        puffer_status = "not_attempted"
        if self.advantage_backend == "puffer_cpu":
            puffer_status = "not_vectorized_puffer_cpu_advantage_only"
        return build_run_metadata(
            config=self.components.config,
            schema_names=self.components.encoder.schema.names,
            max_options=self.components.masker.max_options,
            trainer_backend="masked_ppo_torch",
            advantage_backend=self.advantage_backend,
            vector_backend=self._vector_backend_label(),
            puffer_vectorized_status=puffer_status,
        )

    def run(self, *, total_steps: int, dry_run: bool) -> RLRunResult:
        if dry_run:
            return RLRunResult(
                experiment_id=self.components.config.experiment_id,
                backend=self.name,
                dry_run=True,
                observation_dim=self.components.encoder.schema.dim,
                notes=["dry-run: masked PPO components assembled, no rollout/update"],
            )
        if self.num_envs != 1 and self.env_vectorization != "subprocess":
            result = RLRunResult(
                experiment_id=self.components.config.experiment_id,
                backend=self.name,
                dry_run=False,
                observation_dim=self.components.encoder.schema.dim,
                notes=[
                    "BLOCKER: real-engine SampledBattleEnv currently supports num_envs=1 only; "
                    "num_envs>1 can corrupt the engine cleanup path during battle_finish. "
                    "Use the controlled vector-isolation test envs for multi-env validation until "
                    "the live engine wrapper has isolated process/vector reset ownership."
                ],
            )
            result.training_metrics["real_engine_vector_env_status"] = "blocked_num_envs_gt_1"
            result.training_metrics["vector_backend"] = self._vector_backend_label()
            result.training_metrics["puffer_vectorized_status"] = "not_attempted"
            return result
        if self.env_vectorization not in {"local", "subprocess"}:
            result = RLRunResult(
                experiment_id=self.components.config.experiment_id,
                backend=self.name,
                dry_run=False,
                observation_dim=self.components.encoder.schema.dim,
                notes=[f"BLOCKER: unknown env_vectorization={self.env_vectorization!r}"],
            )
            result.training_metrics["real_engine_vector_env_status"] = "blocked_unknown_vectorization"
            result.training_metrics["vector_backend"] = self._vector_backend_label()
            result.training_metrics["puffer_vectorized_status"] = "not_attempted"
            return result
        self.components.opponent_sampler.reset(self.seed)
        if self.env_vectorization == "subprocess":
            from deepmew.rl.vector_env import SubprocessEnvProxy, sampled_battle_env_spec

            env_fns = [
                lambda offset=offset: SubprocessEnvProxy(
                    offset,
                    sampled_battle_env_spec(
                        experiment=self.components.config.experiment_id,
                        seed=self.seed,
                        max_steps_per_game=self.max_steps_per_game,
                        env_id=offset,
                    ),
                )
                for offset in range(self.num_envs)
            ]
        else:
            env_fns = [
                lambda offset=offset: SampledBattleEnv(
                    self.components,
                    seed=self.seed + 11 + offset,
                    max_steps_per_game=self.max_steps_per_game,
                )
                for offset in range(self.num_envs)
            ]
        run_metadata = self._run_metadata()
        result, model, optimizer, state = train_masked_ppo(
            env_fns=env_fns,
            encoder=self.components.encoder,
            masker=self.components.masker,
            reward_fn=self.components.reward_fn,
            algo_config=self._algo_config(),
            total_steps=total_steps,
            experiment_config=self.components.config,
            checkpoint_path=self.checkpoint_path,
            resume=self.resume,
            checkpoint_extra={
                "run_metadata": run_metadata,
                "backend_metadata": {
                    "trainer_backend": run_metadata["trainer_backend"],
                    "advantage_backend": run_metadata["advantage_backend"],
                    "vector_backend": run_metadata["vector_backend"],
                    "puffer_vectorized_status": run_metadata["puffer_vectorized_status"],
                },
            },
        )
        self.model = model
        self.optimizer = optimizer
        self.state = state
        result.training_metrics["env_vectorization"] = self.env_vectorization
        result.training_metrics["num_envs_requested"] = self.num_envs
        result.training_metrics.update(
            {
                "trainer_backend": run_metadata["trainer_backend"],
                "advantage_backend": run_metadata["advantage_backend"],
                "vector_backend": run_metadata["vector_backend"],
                "puffer_vectorized_status": run_metadata["puffer_vectorized_status"],
                "agent_deck_hash": run_metadata["agent_deck_hash"],
                "opponent_deck_hash": run_metadata["opponent_deck_hash"],
                "obs_schema_hash": run_metadata["obs_schema_hash"],
                "action_schema_hash": run_metadata["action_schema_hash"],
            }
        )
        return result
