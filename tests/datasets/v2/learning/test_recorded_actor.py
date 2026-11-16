from __future__ import annotations

import copy

import pytest

from deepmew_ex.datasets.v2.learning.recorded_actor import observation_hash, recover_recorded_actor_game
from deepmew_ex.datasets.v2.schema import Payload
from tests.datasets.v2.learning.test_recovery import reference_game
from tests.datasets.v2.corpus.conftest import corpus_inventory


@pytest.fixture
def recorded(reference_game: tuple[list[Payload], list[Payload], Payload]) -> tuple[list[Payload], list[Payload]]:
    frames, rows, _ = reference_game
    for frame, row in zip(frames, rows, strict=True):
        frame['source_schema'] = row['schema_version'] = 'deepmew.refgame_decision_row.rebuilt.v1'
    return frames, rows


def test_recorded_teacher_history_preserves_masks_and_admits_no_opponent_timeline(recorded: tuple[list[Payload], list[Payload]]) -> None:
    result = recover_recorded_actor_game(*recorded)
    assert result.recorded_requests == 2
    assert [row.loss_eligible for row in result.sequence.predictions] == [True, False]
    assert len(result.sequence.events) == 4
    assert not result.training_admitted
    assert 'not_proven_complete' in result.coverage


@pytest.mark.parametrize('defect', ['gap', 'mixed_seed', 'mixed_actor', 'timeout', 'wrong_outcome'])
def test_recorded_actor_integrity_rejections(recorded: tuple[list[Payload], list[Payload]], defect: str) -> None:
    frames, rows = recorded
    if defect == 'gap':
        rows[1]['decision_index'] += 1
    elif defect == 'mixed_seed':
        rows[1]['seed'] += 1
    elif defect == 'mixed_actor':
        rows[1]['seat'] = 1
    elif defect == 'timeout':
        rows[1]['outcome']['timed_out'] = True
    else:
        frames[1]['terminal_target'] = 0.0
    with pytest.raises(ValueError):
        recover_recorded_actor_game(frames, rows)


def test_exact_post_state_is_bound_but_never_leaks_other_observer_private_state(recorded: tuple[list[Payload], list[Payload]]) -> None:
    frames, rows = recorded
    for frame, row in zip(frames, rows, strict=True):
        frame['source_schema'] = row['schema_version'] = 'deepmew.refgame_decision_row.rebuilt.v2'
    after = copy.deepcopy(rows[0]['raw_observation'])
    after['current']['yourIndex'] = 1
    after['current']['players'].reverse()
    after['current']['players'][1]['hand'][0]['id'] = 'PRIVATE-POST-CARD'
    rows[0].update({'post_observation': after, 'post_actor_seat': 1})
    result = recover_recorded_actor_game(frames, rows)
    assert result.post_state_receipts[0]['content_sha256'] == observation_hash(after)
    assert result.post_state_receipts[0]['observer'] == 1
    assert 'PRIVATE-POST-CARD' not in str(result.sequence.records)


def test_explicit_post_reference_cannot_change_bytes(recorded: tuple[list[Payload], list[Payload]]) -> None:
    frames, rows = recorded
    for frame, row in zip(frames, rows, strict=True):
        frame['source_schema'] = row['schema_version'] = 'deepmew.refgame_decision_row.rebuilt.v2'
    rows[0].update({'post_observation': copy.deepcopy(rows[0]['raw_observation']), 'post_actor_seat': 0})
    frames[0]['action_result']['state_after_reference'] = {**frames[0]['state_before'], 'source_field': 'post_observation', 'content_sha256': '0' * 64}
    with pytest.raises(ValueError, match='post-state identity'):
        recover_recorded_actor_game(frames, rows)
