from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.learning.raw_recovery import recover_raw_histories
from deepmew_ex.datasets.v2.schema import Episode
from deepmew_ex.registry.store import canonical
from tests.datasets.v2.conftest import source_row, synthetic_replay


def source_replay() -> tuple[Episode, bytes]:
    replay = synthetic_replay('restored-raw-identities')
    for step in replay['steps']:
        for seat, agent in enumerate(step):
            agent['observation']['logs'][0].update({'cardId': 123 + seat, 'serial': 456 + seat})
            agent['observation']['current']['looking'] = None
            agent['observation']['select']['deck'] = None
    raw = canonical(replay)
    return convert_episode(raw, source_row(raw, Path('unused'), replay['id']), load_contract()), raw


def test_raw_recovery_restores_visible_draw_identity_and_all_exact_actions() -> None:
    episode, raw = source_replay()
    sequences = recover_raw_histories(episode, raw, 'train')
    assert sum(len(sequence.predictions) for sequence in sequences) == len(episode.frames)
    assert sum(len(sequence.preserved_transition_ids) for sequence in sequences) == len(episode.frames)
    for sequence in sequences:
        assert 'PRIVATE' not in str(sequence.records)
        assert any(path.endswith('/cardId') and value == f'int:{123 + sequence.actor}' for record in sequence.records for path, value, _ in record)
        assert not any(path.startswith('/public_log_snapshot') and value == f'int:{124 - sequence.actor}' for record in sequence.records for path, value, _ in record)
        assert len(sequence.events) == 2 * len(sequence.predictions)


def test_raw_recovery_rejects_changed_source_bytes() -> None:
    episode, raw = source_replay()
    with pytest.raises(ValueError, match='source identity'):
        recover_raw_histories(episode, raw + b' ', 'train')


def test_raw_recovery_never_invents_missing_global_requests() -> None:
    episode, raw = source_replay()
    with pytest.raises(ValueError):
        recover_raw_histories(replace(episode, frames=(episode.frames[0], *episode.frames[2:])), raw, 'train')
