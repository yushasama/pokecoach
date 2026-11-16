"""Value architectures over actor-visible canonical features.

Every model maps a padded sequence batch ``[B, T, D]`` and a boolean
``sequence_mask [B, T]`` to per-step win-probability logits ``[B, T]``.
Snapshot/window models see only the current or preceding steps. Recurrent
models are strictly unidirectional and initialize a fresh hidden state for
every batch element, where one batch element is one game-seat sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn


ARCHS = ("s1", "s2", "s3", "s4", "s5")
MODEL_VARIANTS = ("default", "matched", "custom")
DEFAULT_MATCHED_PARAMETER_BUDGET = 100_000


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for S1/S2 per-step snapshot MLPs."""

    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.0


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for S3 fixed causal-window MLPs."""

    window: int = 8
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.0


@dataclass(frozen=True)
class RecurrentConfig:
    """Configuration for S4/S5 unidirectional recurrent models."""

    hidden_size: int = 128
    head_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


ModelConfig = SnapshotConfig | WindowConfig | RecurrentConfig


DEFAULT_MODEL_CONFIGS: Mapping[str, ModelConfig] = MappingProxyType({
    "s1": SnapshotConfig(),
    "s2": SnapshotConfig(),
    "s3": WindowConfig(),
    "s4": RecurrentConfig(),
    "s5": RecurrentConfig(),
})


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_dropout(dropout: float) -> None:
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


def _validate_hidden_dims(hidden_dims: tuple[int, ...]) -> None:
    if not isinstance(hidden_dims, tuple) or not hidden_dims:
        raise ValueError("hidden_dims must be a non-empty tuple")
    for width in hidden_dims:
        _validate_positive_int(width, "hidden width")


def _validate_sequence_inputs(
    x: torch.Tensor,
    sequence_mask: torch.Tensor,
    input_dim: int,
) -> None:
    """Fail closed on the shared padded-sequence contract.

    Valid timesteps must form one contiguous prefix per batch element. This
    prevents a masked gap from being interpreted as a real zero-valued state
    by window or recurrent encoders.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, T, D], got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError("x must be a floating-point tensor")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("x must contain at least one sequence and timestep")
    if x.shape[2] != input_dim:
        raise ValueError(f"x feature dimension {x.shape[2]} does not match {input_dim}")
    if sequence_mask.ndim != 2 or tuple(sequence_mask.shape) != tuple(x.shape[:2]):
        raise ValueError(
            f"sequence_mask must have shape {tuple(x.shape[:2])}, got {tuple(sequence_mask.shape)}"
        )
    if sequence_mask.dtype is not torch.bool:
        raise TypeError("sequence_mask must have dtype torch.bool")
    if sequence_mask.device != x.device:
        raise ValueError("x and sequence_mask must be on the same device")
    if bool((sequence_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every sequence_mask row must contain at least one valid timestep")
    if sequence_mask.shape[1] > 1:
        has_valid_after_padding = sequence_mask[:, 1:] & ~sequence_mask[:, :-1]
        if bool(has_valid_after_padding.any().item()):
            raise ValueError("sequence_mask valid timesteps must form a contiguous prefix")


class SnapshotMLP(nn.Module):
    """Per-step MLP (S1 on the prefix, S2 on the full feature vector)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        _validate_hidden_dims(hidden_dims)
        _validate_dropout(dropout)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        _validate_sequence_inputs(x, sequence_mask, self.input_dim)
        logits = self.net(x).squeeze(-1)

        return logits.masked_fill(~sequence_mask, 0.0)


class CausalWindowMLP(nn.Module):
    """S3: fixed causal window of the last L steps, padded before t=0."""

    def __init__(
        self,
        input_dim: int,
        window: int = 8,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        _validate_positive_int(window, "window")
        _validate_hidden_dims(hidden_dims)
        _validate_dropout(dropout)
        self.window = window
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = float(dropout)
        self.mlp = SnapshotMLP(input_dim * window, hidden_dims, dropout)

    def forward(self, x: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        _validate_sequence_inputs(x, sequence_mask, self.input_dim)
        batch, horizon, dim = x.shape
        x = x * sequence_mask[..., None]
        padded = torch.cat([x.new_zeros(batch, self.window - 1, dim), x], dim=1)
        # windows[:, t] = concat(x[t-L+1 .. t]) with zeros before block start.
        windows = padded.unfold(1, self.window, 1)  # [B, T, D, L]
        windows = windows.permute(0, 1, 3, 2).reshape(batch, horizon, self.window * dim)
        logits = self.mlp(windows, sequence_mask)

        return logits.masked_fill(~sequence_mask, 0.0)


class RecurrentValue(nn.Module):
    """S4 (GRU) / S5 (LSTM): unidirectional recurrent per-step value head."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        cell: str = "gru",
        head: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        _validate_positive_int(hidden_size, "hidden_size")
        _validate_positive_int(head, "head")
        _validate_positive_int(num_layers, "num_layers")
        _validate_dropout(dropout)
        if cell not in {"gru", "lstm"}:
            raise ValueError(f"unknown recurrent cell {cell}")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        recurrent_dropout = float(dropout) if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=False,
        )
        head_layers: list[nn.Module] = [nn.Linear(hidden_size, head), nn.ReLU()]
        if dropout > 0:
            head_layers.append(nn.Dropout(float(dropout)))
        head_layers.append(nn.Linear(head, 1))
        self.head = nn.Sequential(*head_layers)
        self.cell = cell
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.head_dim = head
        self.num_layers = num_layers
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        _validate_sequence_inputs(x, sequence_mask, self.input_dim)
        x = x * sequence_mask[..., None]
        output, _ = self.rnn(x)
        logits = self.head(output).squeeze(-1)

        return logits.masked_fill(~sequence_mask, 0.0)


def _snapshot_parameter_count(input_dim: int, width: int) -> int:
    second = max(1, width // 2)

    return (input_dim + 1) * width + (width + 1) * second + second + 1


def _recurrent_parameter_count(input_dim: int, hidden_size: int, gates: int) -> int:
    head_dim = max(1, hidden_size // 2)
    recurrent = gates * hidden_size * (input_dim + hidden_size + 2)
    head = (hidden_size + 1) * head_dim + head_dim + 1

    return recurrent + head


def _closest_width(parameter_budget: int, count_parameters) -> int:
    """Find the compact width whose analytic parameter count is nearest budget."""
    _validate_positive_int(parameter_budget, "parameter_budget")
    upper = 1
    while count_parameters(upper) < parameter_budget:
        upper *= 2
        if upper > 65_536:
            raise ValueError("parameter_budget is outside the supported matching range")
    candidates = range(1, upper + 1)

    return min(candidates, key=lambda width: (abs(count_parameters(width) - parameter_budget), width))


def matched_model_config(
    arch: str,
    feature_dim: int,
    s1_dim: int,
    parameter_budget: int = DEFAULT_MATCHED_PARAMETER_BUDGET,
    *,
    window: int = 8,
) -> ModelConfig:
    """Return a causal architecture config nearest a common parameter budget.

    The matched family uses two-layer MLP bottlenecks ``(w, floor(w/2))`` and
    recurrent heads of ``floor(hidden/2)``. Width is selected analytically for
    each architecture, so comparisons remain meaningful when feature widths or
    fixed-window lengths change.
    """
    _validate_build_dimensions(arch, feature_dim, s1_dim)
    _validate_positive_int(parameter_budget, "parameter_budget")
    _validate_positive_int(window, "window")
    if arch in {"s1", "s2", "s3"}:
        base_dim = s1_dim if arch == "s1" else feature_dim
        input_dim = base_dim * window if arch == "s3" else base_dim
        width = _closest_width(
            parameter_budget,
            lambda candidate: _snapshot_parameter_count(input_dim, candidate),
        )
        hidden_dims = (width, max(1, width // 2))
        if arch == "s3":
            return WindowConfig(window=window, hidden_dims=hidden_dims)

        return SnapshotConfig(hidden_dims=hidden_dims)
    gates = 3 if arch == "s4" else 4
    hidden_size = _closest_width(
        parameter_budget,
        lambda candidate: _recurrent_parameter_count(feature_dim, candidate, gates),
    )

    return RecurrentConfig(hidden_size=hidden_size, head_dim=max(1, hidden_size // 2))


def _validate_build_dimensions(arch: str, feature_dim: int, s1_dim: int) -> None:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch}")
    _validate_positive_int(feature_dim, "feature_dim")
    _validate_positive_int(s1_dim, "s1_dim")
    if s1_dim > feature_dim:
        raise ValueError("s1_dim cannot exceed feature_dim")


def _build_from_config(arch: str, input_dim: int, config: ModelConfig) -> nn.Module:
    if arch in {"s1", "s2"}:
        if not isinstance(config, SnapshotConfig):
            raise TypeError(f"{arch} requires SnapshotConfig")

        return SnapshotMLP(input_dim, config.hidden_dims, config.dropout)
    if arch == "s3":
        if not isinstance(config, WindowConfig):
            raise TypeError("s3 requires WindowConfig")

        return CausalWindowMLP(input_dim, config.window, config.hidden_dims, config.dropout)
    if not isinstance(config, RecurrentConfig):
        raise TypeError(f"{arch} requires RecurrentConfig")
    cell = "gru" if arch == "s4" else "lstm"

    return RecurrentValue(
        input_dim,
        hidden_size=config.hidden_size,
        cell=cell,
        head=config.head_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def build_model(
    arch: str,
    feature_dim: int,
    s1_dim: int,
    *,
    variant: str = "default",
    config: ModelConfig | None = None,
    parameter_budget: int | None = None,
    matched_window: int = 8,
) -> tuple[nn.Module, int]:
    """Build one screening model and return ``(model, consumed_input_dim)``.

    Existing three-argument calls retain the original default architectures.
    ``variant='matched'`` derives a parameter-budget-matched configuration;
    ``variant='custom'`` requires an explicit architecture-specific config.
    """
    _validate_build_dimensions(arch, feature_dim, s1_dim)
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown model variant {variant}")
    if variant == "default":
        if config is not None or parameter_budget is not None:
            raise ValueError("default variant does not accept config or parameter_budget")
        resolved_config = DEFAULT_MODEL_CONFIGS[arch]
    elif variant == "matched":
        if config is not None:
            raise ValueError("matched variant derives its config and does not accept config")
        resolved_config = matched_model_config(
            arch,
            feature_dim,
            s1_dim,
            parameter_budget if parameter_budget is not None else DEFAULT_MATCHED_PARAMETER_BUDGET,
            window=matched_window,
        )
    else:
        if config is None:
            raise ValueError("custom variant requires config")
        if parameter_budget is not None:
            raise ValueError("custom variant does not accept parameter_budget")
        resolved_config = config
    input_dim = s1_dim if arch == "s1" else feature_dim
    model = _build_from_config(arch, input_dim, resolved_config)

    return model, input_dim


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable and frozen parameters in a model."""

    return int(sum(parameter.numel() for parameter in model.parameters()))
