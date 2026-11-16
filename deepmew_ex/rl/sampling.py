"""Draw structured actions from ``TemporalPolicy`` heads with exact log-probabilities.

The distributions are exactly the ones whose negative log-likelihood
``TemporalPolicy.stage_loss`` / ``batched_stage_losses`` train:

* unordered stage: P(S) = exp(sum_{i in S} l_i + c_|S|) / Z over subsets with
  minimum <= |S| <= maximum (the count is drawn first from its marginal, then
  the subset by conditional-Poisson sampling over elementary symmetric sums);
* ordered stage: autoregressive softmax over options plus a stop action, with
  no-repeat masking and the decoder GRU advancing on each pick.

Unordered arithmetic runs in float64 on the CPU (options are few); the
returned log-probability is what PPO stores as the behaviour log-probability.
``greedy=True`` reproduces ``TemporalPolicy.decode`` choices exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from deepmew_ex.datasets.v2.learning.trajectory import StageInput
from deepmew_ex.models.temporal import TemporalPolicy


@dataclass(frozen=True)
class Draw:
    selected: tuple[tuple[int, ...], ...]
    logp: float
    entropy: float | None  # exact for single-choice decisions, None otherwise


def is_simple(stages: tuple[StageInput, ...]) -> bool:
    return len(stages) == 1 and stages[0].minimum == stages[0].maximum == 1 and len(stages[0].options) > 0


def count_scores(model: TemporalPolicy, state: torch.Tensor, maximum: int, options: int) -> torch.Tensor:
    counts = torch.arange(maximum + 1, device=state.device, dtype=state.dtype)
    scale = max(options, 1)
    features = torch.stack((counts / scale, (counts / scale).square(), torch.log1p(counts)), dim=1)
    return features @ model.count(state)


def elementary(logits: np.ndarray, top: int) -> np.ndarray:
    """log e_k(exp(logits)) for k = 0..top (``-inf`` beyond len(logits))."""
    coefficients = np.full(top + 1, -np.inf)
    coefficients[0] = 0.0
    for index, logit in enumerate(logits):
        for count in range(min(top, index + 1), 0, -1):
            coefficients[count] = np.logaddexp(coefficients[count], coefficients[count - 1] + logit)
    return coefficients


def subset_logp(logits: np.ndarray, counts: np.ndarray, selected: tuple[int, ...], minimum: int, maximum: int) -> float:
    total = elementary(logits, maximum)[minimum:maximum + 1] + counts[minimum:maximum + 1]
    normalizer = np.logaddexp.reduce(total)
    return float(logits[list(selected)].sum() + counts[len(selected)] - normalizer)


def sample_subset(logits: np.ndarray, counts: np.ndarray, minimum: int, maximum: int, rng: np.random.Generator) -> tuple[int, ...]:
    total = elementary(logits, maximum)[minimum:maximum + 1] + counts[minimum:maximum + 1]
    probabilities = np.exp(total - np.logaddexp.reduce(total))
    size = minimum + int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
    # suffix[i, r] = log e_r over items i..n-1
    count = len(logits)
    suffix = np.full((count + 1, size + 1), -np.inf)
    suffix[count, 0] = 0.0
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1]
        suffix[index, 1:] = np.logaddexp(suffix[index + 1, 1:], logits[index] + suffix[index + 1, :-1])
    chosen, remaining = [], size
    for index in range(count):
        if remaining == 0:
            break
        take = np.exp(logits[index] + suffix[index + 1, remaining - 1] - suffix[index, remaining])
        if rng.random() < take:
            chosen.append(index)
            remaining -= 1
    if remaining:
        raise RuntimeError('conditional subset sampler ran out of items')
    return tuple(chosen)


def greedy_subset(logits: np.ndarray, counts: np.ndarray, minimum: int, maximum: int) -> tuple[int, ...]:
    """``TemporalPolicy.decode_stage`` unordered rule (stable descending rank, best count)."""
    ranked = np.argsort(-logits, kind='stable')
    scores = [counts[count] + logits[ranked[:count]].sum() for count in range(minimum, maximum + 1)]
    size = minimum + int(np.argmax(scores))
    return tuple(sorted(int(value) for value in ranked[:size]))


@torch.no_grad()
def draw(model: TemporalPolicy, state: torch.Tensor, records: torch.Tensor, stages: tuple[StageInput, ...], rng: np.random.Generator, greedy: bool = False) -> Draw:
    """One decision: every stage in order, state advancing exactly as in ``decode``/``losses``."""
    if not stages:
        raise ValueError('decision without action stages')
    result, logp = [], 0.0
    entropy = None
    for stage in stages:
        options = records[list(stage.options)] if stage.options else records.new_zeros((0, records.shape[1]))
        if not stage.ordered:
            if not 0 <= stage.minimum <= stage.maximum <= len(stage.options):
                raise ValueError('invalid subset bounds')
            logits = model.scorer(options, state).double().cpu().numpy() if len(stage.options) else np.zeros(0)
            counts = count_scores(model, state, stage.maximum, len(stage.options)).double().cpu().numpy()
            if not np.isfinite(logits).all() or not np.isfinite(counts).all():
                raise FloatingPointError('nonfinite subset scores')
            selected = greedy_subset(logits, counts, stage.minimum, stage.maximum) if greedy else sample_subset(logits, counts, stage.minimum, stage.maximum, rng)
            logp += subset_logp(logits, counts, selected, stage.minimum, stage.maximum)
            if is_simple(stages):
                probabilities = np.exp(logits - np.logaddexp.reduce(logits))
                entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum())
            after = model.decoder(options[list(selected)].mean(dim=0), state) if selected else state
        else:
            hidden, picks = state, []
            for step in range(stage.maximum):
                scores = torch.cat((model.scorer(options, hidden), model.stop(hidden)))
                allowed = torch.ones(len(stage.options) + 1, device=state.device, dtype=torch.bool)
                if not stage.repeated and picks:
                    allowed[picks] = False
                allowed[-1] = step >= stage.minimum
                logs = F.log_softmax(scores.masked_fill(~allowed, -torch.inf), dim=0)
                values = logs.double().cpu().numpy()
                if greedy:
                    target = int(np.argmax(scores.masked_fill(~allowed, -torch.inf).cpu().numpy()))
                else:
                    probabilities = np.exp(values)
                    target = int(rng.choice(len(values), p=probabilities / probabilities.sum()))
                logp += float(values[target])
                if target == len(stage.options):
                    break
                picks.append(target)
                hidden = model.decoder(options[target], hidden)
            selected = tuple(picks)
            if not stage.minimum <= len(selected) <= stage.maximum:
                raise ValueError('sampled ordered action violates count bounds')
            after = hidden
        result.append(selected)
        if selected:
            state = after
    if not np.isfinite(logp):
        raise FloatingPointError('nonfinite action log-probability')
    return Draw(tuple(result), logp, entropy)
