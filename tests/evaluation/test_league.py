from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

from deepmew_ex.datasets.v2.learning.semantic import Vocabulary, leaves
from deepmew_ex.evaluation.league.policies import IndependentSeat, ReferenceSeatPolicy, SemanticLivePolicy
from deepmew_ex.evaluation.league.runner import summarize
from deepmew_ex.models.semantic import SemanticBC
from deepmew_ex.training.bc import set_seed
from deepmew_ex.training.semantic import make_batch


def observation() -> dict[str, object]:
    return {'current': {'yourIndex': 0, 'players': [{'hand': [], 'prize': [None]}, {'hand': None, 'prize': [None]}]}, 'select': {'type': 9, 'context': 41, 'minCount': 1, 'maxCount': 1, 'option': [{'type': 1}, {'type': 2}]}, 'logs': [], 'search_begin_input': 'PRIVATE-SENTINEL'}


def test_seat_randomness_and_privacy_are_independent() -> None:
    class RandomPolicy:
        label = 'random-fixture'

        def select(self, obs: dict[str, object], rng: random.Random) -> list[int]:
            assert 'search_begin_input' not in obs
            assert obs['current']['players'][1].get('hand') is None
            return [rng.randint(0, 10000), int(random.random() * 10000), int(np.random.random() * 10000)]
    first = IndependentSeat(RandomPolicy(), 11)
    second = IndependentSeat(RandomPolicy(), 22)
    control = IndependentSeat(RandomPolicy(), 11)
    shared = random.Random(777)
    actual = []
    for _ in range(3):
        actual.append(first.select(observation(), shared))
        second.select(observation(), shared)
    assert actual == [control.select(observation(), shared) for _ in range(3)]
    assert first.calls == second.calls == 3


def test_live_and_offline_history_encodings_have_logit_parity() -> None:
    current = {'state': {'hand': [12]}, 'request': {'type': 9}, 'public_log_snapshot': [{'type': 4}]}
    options = [{'type': 1}, {'type': 2}]
    prior = [{'state': {'hand': [10 + index]}, 'past_action': [{'type': index}], 'public_log_snapshot': []} for index in range(2)]
    records = [list(leaves(current)), *[list(leaves(option)) for option in options], *[list(leaves(record)) for record in prior]]
    vocab = Vocabulary.build(records)
    encoded = [vocab.encode(record) for record in records]
    class MemoryDataset:
        window = 2

        def record(self, row: dict[str, object], index: int) -> np.ndarray:
            return encoded[index]
    row = {'state_record': 0, 'option_records': [1, 2], 'history_records': [3, 4], 'target': 0, 'actor_order': 7}
    for treatment in ('snapshot', 'null', 'ordered', 'shuffled'):
        set_seed(42)
        model = SemanticBC(len(vocab.paths) + 2, len(vocab.values) + 2, hidden=16, embedding=8, history=treatment).eval()
        policy = SemanticLivePolicy(treatment, model, vocab, 2, 42)
        assert not policy.previous
        policy.previous = deque(prior, maxlen=2)
        policy.actor_order = 7
        batch = make_batch(MemoryDataset(), [row], treatment, 42)
        with torch.no_grad():
            assert torch.equal(policy.score(current, options), batch.logits(model)[0])


def test_league_uncertainty_and_denominators() -> None:
    rows = [{'participant': participant, 'opponent': opponent, 'outcome': outcome} for participant in ('history', 'control') for opponent in ('D1', 'H2') for outcome in ('win', 'loss', 'draw')]
    report = summarize(rows)
    assert len(report['matchups']) == 4
    assert all(row['games'] == 3 and row['score_rate'] == .5 for row in report['matchups'])
    assert report['aggregate']['history']['equal_opponent_score'] == .5
    assert report['aggregate']['history']['opponent_cluster_bootstrap95'] == [.5, .5]


def test_reference_runtime_deck_is_scoped_and_restores_cwd(tmp_path: Path) -> None:
    class Teacher:
        label = 'notebook-fixture'
        _fallback_errors = 3

        def select(self, obs: dict[str, object], rng: random.Random) -> list[int]:
            assert Path('deck.csv').read_text() == 'declared frozen deck'
            return [0]
    (tmp_path / 'deck.csv').write_text('declared frozen deck')
    before = Path.cwd()
    policy = ReferenceSeatPolicy(Teacher(), tmp_path)
    assert policy.select({}, random.Random(42)) == [0]
    assert Path.cwd() == before and policy.fallback_errors == 3


def test_reference_decoder_gets_explicit_absence_without_private_payloads(tmp_path: Path) -> None:
    class StrictTeacher:
        label = 'strict-decoder-fixture'
        _fallback_errors = 0

        def select(self, obs: dict[str, object], rng: random.Random) -> list[int]:
            assert obs['current']['looking'] is None
            assert obs['select']['deck'] is None
            assert obs['current']['players'][1]['hand'] is None
            return [0]
    original = observation()
    projected = {key: value for key, value in original.items() if key != 'search_begin_input'}
    policy = ReferenceSeatPolicy(StrictTeacher(), tmp_path)
    assert policy.select(projected, random.Random(42)) == [0]
    assert 'looking' not in projected['current'] and 'deck' not in projected['select']


def test_reference_execution_error_cannot_silently_become_a_proxy(tmp_path: Path) -> None:
    class BrokenTeacher:
        label = 'broken-notebook-fixture'
        _fallback_errors = 0

        def select(self, obs: dict[str, object], rng: random.Random) -> list[int]:
            self._fallback_errors += 1
            return [0]
    with pytest.raises(RuntimeError, match='proxy fallback rejected'):
        ReferenceSeatPolicy(BrokenTeacher(), tmp_path).select(observation(), random.Random(42))


def test_reference_restricted_zone_uses_declared_fallback_without_notebook_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deepmew_ex.evaluation.league import seats as policies  # ReferenceSeatPolicy lives in seats.py
    class UnavailableZoneTeacher:
        label = 'restricted-zone-fixture'
        _fallback_errors = 0

        def select(self, obs: dict[str, object], rng: random.Random) -> list[int]:
            raise AssertionError('restricted buffers must not reach the notebook')
    monkeypatch.setattr(policies, 'greedy_fallback', lambda obs: [1])
    policy = ReferenceSeatPolicy(UnavailableZoneTeacher(), tmp_path)
    for area in (1, 12):
        obs = observation()
        obs['select']['option'][0]['area'] = area
        assert policy.select(obs, random.Random(42)) == [1]
    assert policy.counts == {'fallback:restricted_zone': 2} and policy.fallback_errors == 0
