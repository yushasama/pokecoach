"""R4F-native terminal-reward PPO for research league learners.

The adapter deliberately consumes already actor-visible R4F option features.
It has no simulator, deck, opponent, runtime-policy, or submission dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.distributions import Categorical


R4F_SOURCE_SCHEMA_VERSION = "deepmew.r9.r4f_retrain.v0"
R4F_MODEL_KIND = "r4f_intent_autoregressive"
R4F_PPO_CHECKPOINT_SCHEMA_VERSION = "deepmew.r4f_terminal_ppo_checkpoint.v0"
R4F_PPO_MODEL_KIND = "r4f_terminal_actor_critic"


class R4FPPOValidationError(ValueError):
    """Raised when an R4F/PPO boundary would make training unsafe."""


@dataclass(frozen=True)
class R4FActorConfig:
    input_dim: int
    hidden_dim: int
    depth: int
    embed_dim: int
    dropout: float
    intent_dim: int


@dataclass(frozen=True)
class FrozenR4FState:
    """One immutable actor-input probe, including its selection prefix."""

    option_features: torch.Tensor
    min_count: int
    max_count: int
    selected: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActorEquivalenceProof:
    passed: bool
    source_checkpoint_sha256: str
    panel_sha256: str
    states_checked: int
    actor_tensors_checked: int
    max_logit_abs_delta: float
    max_intent_abs_delta: float
    greedy_choices_match: bool


@dataclass(frozen=True)
class R4FPPOConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    critic_hidden_dim: int = 128
    device: str = "cpu"


@dataclass(frozen=True)
class R4FActionSample:
    selection: tuple[int, ...]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class R4FRolloutStep:
    option_features: torch.Tensor
    min_count: int
    max_count: int
    selection: tuple[int, ...]
    old_log_prob: float
    old_value: float


@dataclass(frozen=True)
class R4FTerminalEpisode:
    steps: tuple[R4FRolloutStep, ...]
    terminal_reward: float


@dataclass(frozen=True)
class R4FPPOUpdateStats:
    episodes: int
    decisions: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    gradient_norm: float
    actor_parameter_delta_norm: float
    critic_parameter_delta_norm: float


@dataclass
class R4FPPOState:
    update_count: int = 0
    episode_count: int = 0
    decision_count: int = 0


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise R4FPPOValidationError(f"{name} contains NaN/Inf")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _actor_config(payload: Mapping[str, Any]) -> R4FActorConfig:
    if payload.get("schema_version") != R4F_SOURCE_SCHEMA_VERSION:
        raise R4FPPOValidationError("source checkpoint is not the frozen R4F retrain schema")
    if payload.get("model_kind") != R4F_MODEL_KIND:
        raise R4FPPOValidationError("source checkpoint is not an R4F intent-autoregressive actor")
    intent_vocab = payload.get("intent_vocab")
    if not isinstance(intent_vocab, (list, tuple)) or not intent_vocab:
        raise R4FPPOValidationError("source checkpoint lacks a non-empty intent vocabulary")
    config = R4FActorConfig(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
        embed_dim=int(payload["embed_dim"]),
        dropout=float(payload.get("dropout", 0.0)),
        intent_dim=len(intent_vocab),
    )
    if min(config.input_dim, config.hidden_dim, config.depth, config.embed_dim, config.intent_dim) <= 0:
        raise R4FPPOValidationError("source checkpoint has an invalid R4F architecture")
    if config.dropout != 0.0:
        raise R4FPPOValidationError("on-policy R4F adapter requires deterministic zero-dropout actor logits")

    return config


def _encoder(config: R4FActorConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = config.input_dim
    for _ in range(config.depth):
        layers.append(nn.Linear(current_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        current_dim = config.hidden_dim
    layers.append(nn.Linear(current_dim, config.embed_dim))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class R4FActor(nn.Module):
    """Tensor-key-compatible copy of the historical R4F actor."""

    def __init__(self, config: R4FActorConfig):
        super().__init__()
        self.config = config
        self.encoder = _encoder(config)
        self.option_head = nn.Sequential(
            nn.Linear(config.embed_dim + 5, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(config.embed_dim + 5, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.intent_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.intent_dim),
        )

    def encode_options(self, option_features: torch.Tensor) -> torch.Tensor:
        if option_features.ndim != 3 or option_features.shape[1] <= 0:
            raise R4FPPOValidationError("option features must have shape [batch, options, input_dim]")
        if option_features.shape[2] != self.config.input_dim:
            raise R4FPPOValidationError(
                f"R4F option feature width {option_features.shape[2]} != {self.config.input_dim}"
            )
        _assert_finite("option_features", option_features)
        batch, options, width = option_features.shape
        flat = option_features.reshape(batch * options, width)

        return self.encoder(flat).reshape(batch, options, self.config.embed_dim)

    def score_embeddings(self, embeddings: torch.Tensor, option_meta: torch.Tensor) -> torch.Tensor:
        if option_meta.shape != (*embeddings.shape[:2], 5):
            raise R4FPPOValidationError("option metadata must have shape [batch, options, 5]")
        _assert_finite("option_meta", option_meta)
        option_scores = self.option_head(torch.cat([embeddings, option_meta], dim=-1)).squeeze(-1)
        pooled = embeddings.mean(dim=1)
        step_meta = option_meta[:, 0, :]
        stop_score = self.stop_head(torch.cat([pooled, step_meta], dim=-1)).squeeze(-1).unsqueeze(1)

        return torch.cat([option_scores, stop_score], dim=1)

    def forward(self, option_features: torch.Tensor, option_meta: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode_options(option_features)

        return self.score_embeddings(embeddings, option_meta)

    def intent_logits(self, option_features: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode_options(option_features)

        return self.intent_head(embeddings.mean(dim=1))


def _load_actor(payload: Mapping[str, Any]) -> R4FActor:
    config = _actor_config(payload)
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise R4FPPOValidationError("source checkpoint lacks model_state")
    actor = R4FActor(config)
    actor.load_state_dict(dict(state), strict=True)
    for name, value in actor.state_dict().items():
        _assert_finite(f"actor tensor {name}", value)
    actor.eval()

    return actor


def _validated_counts(n_options: int, min_count: int, max_count: int) -> tuple[int, int]:
    if n_options <= 0:
        raise R4FPPOValidationError("R4F decision requires at least one legal option")
    if not 0 <= min_count <= max_count <= n_options:
        raise R4FPPOValidationError(
            f"invalid selection counts min={min_count} max={max_count} options={n_options}"
        )

    return min_count, max_count


def build_option_meta(
    n_options: int,
    selected: Sequence[int],
    min_count: int,
    max_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rebuild the exact five autoregressive prefix fields used by R4F."""

    min_count, max_count = _validated_counts(n_options, min_count, max_count)
    chosen = tuple(int(index) for index in selected)
    chosen_set = set(chosen)
    if len(chosen) != len(chosen_set) or any(index < 0 or index >= n_options for index in chosen):
        raise R4FPPOValidationError(f"invalid R4F selection prefix {chosen}")
    if len(chosen) > max_count:
        raise R4FPPOValidationError(f"selection prefix {chosen} exceeds max_count={max_count}")
    selected_count = len(chosen_set)
    common = (
        min(len(chosen), 8) / 8.0,
        min(selected_count, 8) / 8.0,
        max(0, min_count - selected_count) / n_options,
        max(0, max_count - selected_count) / n_options,
    )
    meta = torch.empty((n_options, 5), dtype=dtype, device=device)
    for option_index in range(n_options):
        meta[option_index] = torch.tensor(
            (1.0 if option_index in chosen_set else 0.0, *common),
            dtype=dtype,
            device=device,
        )

    return meta


def _valid_tokens(n_options: int, selected: Sequence[int], min_count: int, max_count: int, device: torch.device) -> torch.Tensor:
    min_count, max_count = _validated_counts(n_options, min_count, max_count)
    chosen = set(int(index) for index in selected)
    valid = torch.zeros(n_options + 1, dtype=torch.bool, device=device)
    if len(chosen) < max_count:
        for index in range(n_options):
            valid[index] = index not in chosen
    if len(chosen) >= min_count:
        valid[n_options] = True
    if not bool(valid.any().item()):
        raise R4FPPOValidationError("R4F decision has no valid next token")

    return valid


def _masked_logits(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~valid, -1.0e9)
    _assert_finite("masked_logits", masked)

    return masked


class R4FActorCritic(nn.Module):
    def __init__(self, actor: R4FActor, critic_hidden_dim: int):
        super().__init__()
        if critic_hidden_dim <= 0:
            raise R4FPPOValidationError("critic_hidden_dim must be positive")
        self.actor = actor
        self.critic = nn.Sequential(
            nn.Linear(actor.config.embed_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def encode_decision(self, option_features: torch.Tensor) -> torch.Tensor:
        if option_features.ndim != 2:
            raise R4FPPOValidationError("one R4F decision must have shape [options, input_dim]")

        return self.actor.encode_options(option_features.unsqueeze(0))[0]

    def value_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        value = self.critic(embeddings.mean(dim=0)).squeeze(-1)
        _assert_finite("critic_value", value)

        return value

    def logits_from_embeddings(
        self,
        embeddings: torch.Tensor,
        selected: Sequence[int],
        min_count: int,
        max_count: int,
    ) -> torch.Tensor:
        meta = build_option_meta(
            embeddings.shape[0],
            selected,
            min_count,
            max_count,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        logits = self.actor.score_embeddings(embeddings.unsqueeze(0), meta.unsqueeze(0))[0]
        _assert_finite("actor_logits", logits)

        return logits


class R4FSelectionDistribution:
    def __init__(self, model: R4FActorCritic):
        self.model = model

    def sample(
        self,
        option_features: torch.Tensor,
        min_count: int,
        max_count: int,
        *,
        deterministic: bool = False,
    ) -> R4FActionSample:
        device = next(self.model.parameters()).device
        features = torch.as_tensor(option_features, dtype=torch.float32, device=device)
        embeddings = self.model.encode_decision(features)
        value = self.model.value_from_embeddings(embeddings)
        selected: list[int] = []
        log_prob = torch.zeros((), device=device)
        entropy = torch.zeros((), device=device)
        while True:
            logits = self.model.logits_from_embeddings(embeddings, selected, min_count, max_count)
            valid = _valid_tokens(len(embeddings), selected, min_count, max_count, device)
            dist = Categorical(logits=_masked_logits(logits, valid))
            token = torch.argmax(_masked_logits(logits, valid)) if deterministic else dist.sample()
            log_prob = log_prob + dist.log_prob(token)
            entropy = entropy + dist.entropy()
            index = int(token.detach().cpu())
            if index == len(embeddings):
                break
            selected.append(index)
            if len(selected) >= max_count:
                break
        _assert_finite("sample_log_prob", log_prob)
        _assert_finite("sample_entropy", entropy)

        return R4FActionSample(tuple(selected), log_prob, entropy, value)

    def evaluate(self, step: R4FRolloutStep) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        features = torch.as_tensor(step.option_features, dtype=torch.float32, device=device)
        embeddings = self.model.encode_decision(features)
        n_options = len(embeddings)
        _validated_counts(n_options, step.min_count, step.max_count)
        selected: list[int] = []
        log_prob = torch.zeros((), device=device)
        entropy = torch.zeros((), device=device)
        for raw_index in step.selection:
            index = int(raw_index)
            logits = self.model.logits_from_embeddings(
                embeddings,
                selected,
                step.min_count,
                step.max_count,
            )
            valid = _valid_tokens(n_options, selected, step.min_count, step.max_count, device)
            if index < 0 or index >= n_options or not bool(valid[index].item()):
                raise R4FPPOValidationError(f"stored selection {step.selection} violates its R4F mask")
            dist = Categorical(logits=_masked_logits(logits, valid))
            token = torch.as_tensor(index, dtype=torch.long, device=device)
            log_prob = log_prob + dist.log_prob(token)
            entropy = entropy + dist.entropy()
            selected.append(index)
        if len(selected) < step.max_count:
            logits = self.model.logits_from_embeddings(
                embeddings,
                selected,
                step.min_count,
                step.max_count,
            )
            valid = _valid_tokens(n_options, selected, step.min_count, step.max_count, device)
            if not bool(valid[n_options].item()):
                raise R4FPPOValidationError(
                    f"stored selection {step.selection} stops before min_count={step.min_count}"
                )
            dist = Categorical(logits=_masked_logits(logits, valid))
            stop = torch.as_tensor(n_options, dtype=torch.long, device=device)
            log_prob = log_prob + dist.log_prob(stop)
            entropy = entropy + dist.entropy()
        value = self.model.value_from_embeddings(embeddings)
        _assert_finite("evaluated_log_prob", log_prob)
        _assert_finite("evaluated_entropy", entropy)

        return log_prob, entropy, value


def _panel_sha256(states: Sequence[FrozenR4FState]) -> str:
    digest = hashlib.sha256()
    for state in states:
        features = torch.as_tensor(state.option_features, dtype=torch.float32).detach().cpu().contiguous()
        header = {
            "shape": list(features.shape),
            "min_count": state.min_count,
            "max_count": state.max_count,
            "selected": list(state.selected),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
        digest.update(features.numpy().tobytes(order="C"))

    return digest.hexdigest()


def prove_actor_equivalence(
    actor: R4FActor,
    source_payload: Mapping[str, Any],
    frozen_states: Sequence[FrozenR4FState],
    *,
    source_checkpoint_sha256: str,
) -> ActorEquivalenceProof:
    """Require exact tensors and outputs before the first PPO update."""

    if not frozen_states:
        raise R4FPPOValidationError("actor equivalence requires a non-empty frozen state panel")
    reference = _load_actor(source_payload)
    actor.eval()
    reference.eval()
    candidate_state = actor.state_dict()
    reference_state = reference.state_dict()
    if candidate_state.keys() != reference_state.keys():
        raise R4FPPOValidationError("R4F actor tensor keys changed during initialization")
    for name in candidate_state:
        if not torch.equal(candidate_state[name].detach().cpu(), reference_state[name].detach().cpu()):
            raise R4FPPOValidationError(f"R4F actor tensor {name} changed during initialization")
    max_logit_delta = 0.0
    max_intent_delta = 0.0
    greedy_match = True
    with torch.no_grad():
        for state in frozen_states:
            features = torch.as_tensor(state.option_features, dtype=torch.float32).detach().cpu()
            if features.ndim != 2 or features.shape[1] != actor.config.input_dim:
                raise R4FPPOValidationError("frozen actor state has an incompatible option-feature shape")
            meta = build_option_meta(
                features.shape[0],
                state.selected,
                state.min_count,
                state.max_count,
                device=features.device,
                dtype=features.dtype,
            )
            candidate_logits = actor(features.unsqueeze(0), meta.unsqueeze(0))[0]
            reference_logits = reference(features.unsqueeze(0), meta.unsqueeze(0))[0]
            candidate_intent = actor.intent_logits(features.unsqueeze(0))[0]
            reference_intent = reference.intent_logits(features.unsqueeze(0))[0]
            max_logit_delta = max(
                max_logit_delta,
                float((candidate_logits - reference_logits).abs().max().item()),
            )
            max_intent_delta = max(
                max_intent_delta,
                float((candidate_intent - reference_intent).abs().max().item()),
            )
            valid = _valid_tokens(
                features.shape[0],
                state.selected,
                state.min_count,
                state.max_count,
                features.device,
            )
            candidate_choice = int(torch.argmax(_masked_logits(candidate_logits, valid)).item())
            reference_choice = int(torch.argmax(_masked_logits(reference_logits, valid)).item())
            greedy_match = greedy_match and candidate_choice == reference_choice
    passed = max_logit_delta == 0.0 and max_intent_delta == 0.0 and greedy_match
    proof = ActorEquivalenceProof(
        passed=passed,
        source_checkpoint_sha256=source_checkpoint_sha256,
        panel_sha256=_panel_sha256(frozen_states),
        states_checked=len(frozen_states),
        actor_tensors_checked=len(candidate_state),
        max_logit_abs_delta=max_logit_delta,
        max_intent_abs_delta=max_intent_delta,
        greedy_choices_match=greedy_match,
    )
    if not proof.passed:
        raise R4FPPOValidationError(f"R4F actor equivalence failed: {proof}")

    return proof


def _flatten_parameters(module: nn.Module) -> torch.Tensor:
    values = [parameter.detach().reshape(-1).cpu() for parameter in module.parameters()]

    return torch.cat(values) if values else torch.empty(0)


def _terminal_advantages(
    episodes: Sequence[R4FTerminalEpisode],
    config: R4FPPOConfig,
    device: torch.device,
) -> tuple[list[R4FRolloutStep], torch.Tensor, torch.Tensor]:
    steps: list[R4FRolloutStep] = []
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for episode in episodes:
        if not episode.steps:
            raise R4FPPOValidationError("terminal PPO episode has no decisions")
        if float(episode.terminal_reward) not in {-1.0, 0.0, 1.0}:
            raise R4FPPOValidationError("terminal game reward must be exactly -1, 0, or +1")
        values = torch.tensor([step.old_value for step in episode.steps], dtype=torch.float32, device=device)
        _assert_finite("old_values", values)
        episode_advantages = torch.zeros_like(values)
        last_gae = torch.zeros((), device=device)
        for index in reversed(range(len(episode.steps))):
            is_terminal = index == len(episode.steps) - 1
            reward = float(episode.terminal_reward) if is_terminal else 0.0
            next_value = torch.zeros((), device=device) if is_terminal else values[index + 1]
            next_non_terminal = 0.0 if is_terminal else 1.0
            delta = reward + config.gamma * next_value * next_non_terminal - values[index]
            last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
            episode_advantages[index] = last_gae
        steps.extend(episode.steps)
        advantages.extend(episode_advantages.unbind())
        returns.extend((episode_advantages + values).unbind())
    stacked_advantages = torch.stack(advantages)
    stacked_returns = torch.stack(returns)
    _assert_finite("terminal_advantages", stacked_advantages)
    _assert_finite("terminal_returns", stacked_returns)

    return steps, stacked_advantages, stacked_returns


def _actor_payload_template(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "model_kind",
        "input_dim",
        "hidden_dim",
        "depth",
        "embed_dim",
        "dropout",
        "feature_mode",
        "intent_vocab",
        "state_schema",
        "option_schema",
    )

    return {key: source_payload[key] for key in keys if key in source_payload}


class R4FNativePPO:
    """Verified R4F actor plus a fresh critic and terminal-reward PPO updater."""

    def __init__(
        self,
        model: R4FActorCritic,
        optimizer: torch.optim.Optimizer,
        config: R4FPPOConfig,
        proof: ActorEquivalenceProof,
        source_payload_template: Mapping[str, Any],
        state: R4FPPOState | None = None,
    ):
        if not proof.passed:
            raise R4FPPOValidationError("PPO updates require a passing actor-equivalence proof")
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.proof = proof
        self.source_payload_template = dict(source_payload_template)
        self.state = state or R4FPPOState()
        self.distribution = R4FSelectionDistribution(model)

    @classmethod
    def from_r4f_checkpoint(
        cls,
        checkpoint: Path | str,
        frozen_states: Sequence[FrozenR4FState],
        config: R4FPPOConfig | None = None,
    ) -> "R4FNativePPO":
        path = Path(checkpoint).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise R4FPPOValidationError("R4F checkpoint payload must be a mapping")
        actor = _load_actor(payload)
        source_sha256 = _sha256_file(path)
        proof = prove_actor_equivalence(
            actor,
            payload,
            frozen_states,
            source_checkpoint_sha256=source_sha256,
        )
        ppo_config = config or R4FPPOConfig()
        model = R4FActorCritic(actor, ppo_config.critic_hidden_dim).to(torch.device(ppo_config.device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=ppo_config.learning_rate,
            weight_decay=ppo_config.weight_decay,
        )

        return cls(model, optimizer, ppo_config, proof, _actor_payload_template(payload))

    def sample(
        self,
        option_features: torch.Tensor,
        min_count: int,
        max_count: int,
        *,
        deterministic: bool = False,
    ) -> R4FActionSample:
        with torch.no_grad():
            return self.distribution.sample(
                option_features,
                min_count,
                max_count,
                deterministic=deterministic,
            )

    def update(self, episodes: Sequence[R4FTerminalEpisode]) -> R4FPPOUpdateStats:
        if not episodes:
            raise R4FPPOValidationError("terminal PPO update requires at least one episode")
        device = torch.device(self.config.device)
        steps, advantages, returns = _terminal_advantages(episodes, self.config, device)
        old_log_probs = torch.tensor(
            [step.old_log_prob for step in steps],
            dtype=torch.float32,
            device=device,
        )
        _assert_finite("old_log_probs", old_log_probs)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        actor_before = _flatten_parameters(self.model.actor)
        critic_before = _flatten_parameters(self.model.critic)
        indices = torch.arange(len(steps), device=device)
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        kl_value = 0.0
        gradient_norm_value = 0.0
        self.model.train()
        for _ in range(self.config.update_epochs):
            permutation = indices[torch.randperm(len(indices), device=device)]
            for start in range(0, len(indices), self.config.minibatch_size):
                minibatch = permutation[start:start + self.config.minibatch_size]
                evaluated = [self.distribution.evaluate(steps[int(index.detach().cpu())]) for index in minibatch]
                new_log_probs = torch.stack([value[0] for value in evaluated])
                entropies = torch.stack([value[1] for value in evaluated])
                new_values = torch.stack([value[2] for value in evaluated])
                log_ratio = new_log_probs - old_log_probs[minibatch]
                ratio = torch.exp(log_ratio)
                _assert_finite("ppo_ratio", ratio)
                minibatch_advantages = advantages[minibatch]
                unclipped = -minibatch_advantages * ratio
                clipped = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()
                value_loss = 0.5 * ((new_values - returns[minibatch]) ** 2).mean()
                entropy = entropies.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                _assert_finite("ppo_loss", loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None:
                        _assert_finite(f"gradient {name}", parameter.grad)
                gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                _assert_finite("gradient_norm", gradient_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                policy_loss_value = float(policy_loss.detach().cpu())
                value_loss_value = float(value_loss.detach().cpu())
                entropy_value = float(entropy.detach().cpu())
                kl_value = float(approximate_kl.detach().cpu())
                gradient_norm_value = float(gradient_norm.detach().cpu())
        self.model.eval()
        actor_after = _flatten_parameters(self.model.actor)
        critic_after = _flatten_parameters(self.model.critic)
        actor_delta = float(torch.linalg.vector_norm(actor_after - actor_before).item())
        critic_delta = float(torch.linalg.vector_norm(critic_after - critic_before).item())
        if not math.isfinite(actor_delta) or not math.isfinite(critic_delta):
            raise R4FPPOValidationError("PPO parameter delta is non-finite")
        self.state.update_count += 1
        self.state.episode_count += len(episodes)
        self.state.decision_count += len(steps)

        return R4FPPOUpdateStats(
            episodes=len(episodes),
            decisions=len(steps),
            policy_loss=policy_loss_value,
            value_loss=value_loss_value,
            entropy=entropy_value,
            approximate_kl=kl_value,
            gradient_norm=gradient_norm_value,
            actor_parameter_delta_norm=actor_delta,
            critic_parameter_delta_norm=critic_delta,
        )

    def actor_export_payload(self, *, training_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return an actor-only payload accepted by historical R4F loaders."""

        payload = dict(self.source_payload_template)
        payload["schema_version"] = R4F_SOURCE_SCHEMA_VERSION
        payload["model_kind"] = R4F_MODEL_KIND
        payload["model_state"] = {
            name: value.detach().cpu().clone()
            for name, value in self.model.actor.state_dict().items()
        }
        payload["train"] = {
            "kind": "terminal_reward_ppo",
            "episodes": self.state.episode_count,
            "decisions": self.state.decision_count,
            "updates": self.state.update_count,
            **dict(training_metadata or {}),
        }
        payload["initialization"] = {
            "kind": "exact_r4f_actor",
            "source_checkpoint_sha256": self.proof.source_checkpoint_sha256,
            "frozen_panel_sha256": self.proof.panel_sha256,
            "actor_equivalence": asdict(self.proof),
        }

        return payload

    def export_actor(self, path: Path | str, *, training_metadata: Mapping[str, Any] | None = None) -> None:
        _atomic_torch_save(self.actor_export_payload(training_metadata=training_metadata), Path(path))

    def save_training_checkpoint(self, path: Path | str, *, extra: Mapping[str, Any] | None = None) -> None:
        payload = {
            "schema_version": R4F_PPO_CHECKPOINT_SCHEMA_VERSION,
            "model_kind": R4F_PPO_MODEL_KIND,
            "actor_config": asdict(self.model.actor.config),
            "actor_state": {
                name: value.detach().cpu().clone()
                for name, value in self.model.actor.state_dict().items()
            },
            "critic_state": {
                name: value.detach().cpu().clone()
                for name, value in self.model.critic.state_dict().items()
            },
            "optimizer_state": self.optimizer.state_dict(),
            "ppo_config": asdict(self.config),
            "training_state": asdict(self.state),
            "initialization_proof": asdict(self.proof),
            "source_payload_template": dict(self.source_payload_template),
            "extra": dict(extra or {}),
        }
        _atomic_torch_save(payload, Path(path))

    @classmethod
    def load_training_checkpoint(
        cls,
        path: Path | str,
        *,
        device: str | None = None,
    ) -> "R4FNativePPO":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema_version") != R4F_PPO_CHECKPOINT_SCHEMA_VERSION:
            raise R4FPPOValidationError("not an R4F terminal-PPO training checkpoint")
        config_values = dict(payload["ppo_config"])
        if device is not None:
            config_values["device"] = device
        config = R4FPPOConfig(**config_values)
        actor = R4FActor(R4FActorConfig(**payload["actor_config"]))
        actor.load_state_dict(payload["actor_state"], strict=True)
        model = R4FActorCritic(actor, config.critic_hidden_dim)
        model.critic.load_state_dict(payload["critic_state"], strict=True)
        model.to(torch.device(config.device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        proof = ActorEquivalenceProof(**payload["initialization_proof"])
        state = R4FPPOState(**payload["training_state"])
        model.eval()

        return cls(
            model,
            optimizer,
            config,
            proof,
            payload["source_payload_template"],
            state,
        )
