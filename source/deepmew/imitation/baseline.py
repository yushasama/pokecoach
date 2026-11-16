"""Count-table behavior-cloning baselines over action categories.

Each model predicts the chosen OptionType from a conditioning key, backing off to
coarser keys when a cell is unseen or too sparse. Evaluation is top-1 accuracy of
predicting the real player's actual action category on a held-out split -- a cheap,
deterministic imitation signal that needs no game simulation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from deepmew.imitation.schemas import ImitationExample, ImitationConfig


class CountTablePolicy:
    """argmax_a P(action | key), with deterministic tie-breaks."""

    def __init__(self, key_fn: Callable[[ImitationExample], tuple], min_count: int = 1):
        self.key_fn = key_fn
        self.min_count = min_count
        self.tables: dict[tuple, Counter] = defaultdict(Counter)

    def fit(self, examples: Iterable[ImitationExample]) -> "CountTablePolicy":
        for ex in examples:
            self.tables[self.key_fn(ex)][ex.chosen_option_type] += 1
        return self

    def predict(self, ex: ImitationExample) -> str | None:
        table = self.tables.get(self.key_fn(ex))
        if not table or sum(table.values()) < self.min_count:
            return None
        # Deterministic: highest count, then action name.
        return max(table.items(), key=lambda kv: (kv[1], kv[0]))[0]


@dataclass
class ModelEval:
    name: str
    accuracy: float
    n: int
    coverage: float  # fraction of test examples the conditioned model could answer


def _accuracy(predictions: list[tuple[str | None, str]]) -> tuple[float, float]:
    if not predictions:
        return 0.0, 0.0
    answered = [(p, y) for p, y in predictions if p is not None]
    correct = sum(1 for p, y in answered if p == y)
    accuracy = correct / len(predictions)  # unanswered count as wrong
    coverage = len(answered) / len(predictions)
    return accuracy, coverage


def evaluate_baselines(
    train: list[ImitationExample],
    test: list[ImitationExample],
    config: ImitationConfig,
) -> list[ModelEval]:
    """Fit BC count tables on train, report top-1 accuracy on test with backoff."""
    global_table = CountTablePolicy(lambda e: (), min_count=1).fit(train)
    ctx = CountTablePolicy(lambda e: (e.select_context,), min_count=config.min_cell_count).fit(train)
    ctx_arch = CountTablePolicy(lambda e: (e.select_context, e.archetype), min_count=config.min_cell_count).fit(train)
    ctx_arch_opp = CountTablePolicy(
        lambda e: (e.select_context, e.archetype, e.opponent_archetype), min_count=config.min_cell_count
    ).fit(train)

    def backoff(*policies: CountTablePolicy) -> Callable[[ImitationExample], str | None]:
        def predict(ex: ImitationExample) -> str | None:
            for policy in policies:
                pred = policy.predict(ex)
                if pred is not None:
                    return pred
            return None
        return predict

    models = {
        "global_most_common": backoff(global_table),
        "by_context": backoff(ctx, global_table),
        "by_context_archetype": backoff(ctx_arch, ctx, global_table),
        "by_context_archetype_opponent": backoff(ctx_arch_opp, ctx_arch, ctx, global_table),
    }

    results: list[ModelEval] = []
    for name, predict in models.items():
        preds = [(predict(ex), ex.chosen_option_type) for ex in test]
        acc, cov = _accuracy(preds)
        results.append(ModelEval(name=name, accuracy=round(acc, 4), n=len(test), coverage=round(cov, 4)))
    return results
