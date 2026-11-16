from __future__ import annotations

import torch

from deepmew_ex.models.mlp import DeepMewEX
from deepmew_ex.models.semantic import SemanticBC
from deepmew_ex.training.bc import apply_gradients, set_seed


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = torch.tensor([[[2., 2., .1], [3., 3., .2]]])
    options = torch.tensor([[[[4., 4., .4]], [[4., 5., .5]], [[4., 6., .6]]]])
    history = torch.tensor([[[[2., 3., .3]], [[3., 4., .4]]]])
    return current, options, history, torch.ones(1, 3, dtype=torch.bool)


def test_batched_shared_scorer_matches_original_per_decision() -> None:
    set_seed(42)
    model = DeepMewEX(option_dim=7, value_dim=5, hidden_dim=8).eval()
    options, values = torch.randn(3, 4, 7), torch.randn(3, 5)
    assert torch.allclose(model(options, values), torch.stack([model(option, value) for option, value in zip(options, values, strict=True)]), atol=1e-6)


def test_semantic_option_permutation_and_parameter_updates() -> None:
    set_seed(42)
    model = SemanticBC(16, 16, hidden=16, embedding=8)
    current, options, history, mask = inputs()
    permutation = torch.tensor([2, 0, 1])
    before = model(current, options, history, mask).detach()
    permuted = model(current, options[:, permutation], history, mask[:, permutation]).detach()
    assert torch.allclose(permuted, before[:, permutation], atol=1e-6)
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(8):
        logits = model(current, options, history, mask)
        apply_gradients(torch.nn.functional.cross_entropy(logits, torch.tensor([1])), model, optimizer, 1.0)
    assert any(not torch.equal(value, parameters[name]) for name, value in model.named_parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.nn.functional.cross_entropy(model(current, options, history, mask), torch.tensor([1])) < torch.nn.functional.cross_entropy(before, torch.tensor([1]))


def test_null_history_is_invariant_and_ordered_history_is_consumed() -> None:
    current, options, history, mask = inputs()
    for treatment in ('null', 'ordered'):
        set_seed(42)
        model = SemanticBC(16, 16, hidden=16, embedding=8, history=treatment).eval()
        left = model(current, options, history, mask)
        right = model(current, options, history.flip(1), mask)
        assert torch.equal(left, right) if treatment == 'null' else not torch.allclose(left, right, atol=1e-8)
