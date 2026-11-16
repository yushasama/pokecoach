from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from deepmew_ex.datasets.v2.learning.semantic import leaves
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, StageInput, StageTarget
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.rl.farm import InferenceServer, Net
from deepmew_ex.rl.incremental import tables_of
from deepmew_ex.rl.ppo import ValueHead, policy_terms


def live_game(seed: int, steps: int) -> tuple[ActorSequence, list[tuple[int, int, tuple[StageInput, ...]]]]:
    """Live-shaped game: request event, its option records, then a past-action event; mixed action kinds."""
    rng = random.Random(seed)
    records, events, decisions = [], [], []
    kinds = [(1, 1, False), (0, 1, False), (0, 2, False), (2, 2, False), (0, 2, True), (1, 3, True)]
    for step in range(steps):
        events.append(len(records))
        records.append(list(leaves({'state': {'turn': step, 'hp': rng.randint(0, 300), 'card': f'card-{rng.randint(0, 30)}'}})))
        count = rng.randint(1, 5)
        options = tuple(range(len(records), len(records) + count))
        for _ in range(count):
            records.append(list(leaves({'option': {'card': f'card-{rng.randint(0, 30)}', 'slot': rng.randint(0, 5)}})))
        minimum, maximum, ordered = rng.choice([kind for kind in kinds if kind[1] <= count])
        decisions.append((len(events) - 1, len(records), (StageInput(options, minimum, maximum, ordered, False),)))
        events.append(len(records))
        records.append(list(leaves({'past_action': {'step': step}})))
    return ActorSequence('live', 0, 'test', 'train', tuple(records), tuple(events), (), (), (), True), decisions


@pytest.mark.parametrize('greedy', [False, True])
def test_server_replies_match_training_likelihood_and_value(greedy: bool) -> None:
    torch.manual_seed(2)
    policy = TemporalPolicy(hidden=32, embedding=8, layers=2, heads=4).eval()
    head = ValueHead(32).eval()
    server = InferenceServer({'current': Net(policy, head)}, seed=0)
    games = [live_game(seed, steps) for seed, steps in ((1, 9), (2, 4), (3, 14))]
    sent = [[0, 0] for _ in games]
    replies: list[list[tuple]] = [[] for _ in games]
    for step in range(max(len(decisions) for _, decisions in games)):
        requests, owners = [], []
        for game, (sequence, decisions) in enumerate(games):
            if step >= len(decisions):
                continue
            event_index, record_end, stages = decisions[step]
            event_end = event_index + 1
            requests.append(((game,), 'current', tables_of(list(sequence.records[sent[game][0]:record_end])), list(sequence.events[sent[game][1]:event_end]), event_index, stages, greedy))
            sent[game] = [record_end, event_end]
            owners.append(game)
        for game, reply in zip(owners, server.serve(requests), strict=True):
            replies[game].append(reply)
    assert server.stats['subset'] and server.stats['structured']
    for game, (sequence, decisions) in enumerate(games):
        predictions = []
        for index, ((event_index, _, stages), reply) in enumerate(zip(decisions, replies[game], strict=True)):
            targets = tuple(StageTarget(stage.options, selected, stage.minimum, stage.maximum, stage.ordered, stage.repeated) for stage, selected in zip(stages, reply[0], strict=True))
            predictions.append(Prediction(f'd{index}', event_index, targets, '', True))
            if greedy:
                assert reply[0] == policy.decode(sequence, event_index, stages)
        with torch.no_grad():
            logp, entropy, values = policy_terms(policy, head, [replace(sequence, predictions=tuple(predictions))])
        np.testing.assert_allclose([reply[1] for reply in replies[game]], logp.numpy(), atol=5e-5)
        np.testing.assert_allclose([reply[2] for reply in replies[game]], values.numpy(), atol=5e-5)
        for reply, spread in zip(replies[game], entropy.numpy(), strict=True):
            if reply[3] is not None:
                assert reply[3] == pytest.approx(float(spread), abs=5e-5)
    server.drop([(game,) for game in range(len(games))])
    server.refresh()
