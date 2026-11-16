from __future__ import annotations

import copy
import gzip
import json
import hashlib
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.corpus.records import normalize_reference
from deepmew_ex.datasets.v2.learning.recovery import recover_reference_game
from deepmew_ex.datasets.v2.schema import Payload
from tests.datasets.v2.corpus.conftest import corpus_inventory


@pytest.fixture
def reference_game(corpus_inventory: Path) -> tuple[list[Payload], list[Payload], Payload]:
    inventory = json.loads(corpus_inventory.read_text())
    part = inventory['parts'][0]
    sources = {row['path']: row for row in inventory['sources']}
    with gzip.open(part['files']['part_0000.transitions.jsonl.gz']['path'], 'rt') as handle:
        transitions = [json.loads(line) for line in handle]
    with gzip.open(part['files']['part_0000.sequences.jsonl.gz']['path'], 'rt') as handle:
        sequences = [json.loads(line) for line in handle]
    frames = [normalize_reference(transition, sequence, sources, 'train', 'train', index == 0, 'unit') for index, (transition, sequence) in enumerate(zip(transitions, sequences, strict=True))]
    rows = [json.loads(line) for line in Path(inventory['sources'][0]['path']).read_text().splitlines()]
    for row in rows:
        row.update({'seed': 700000, 'outcome': {'result': 0, 'crashed': False, 'timed_out': False}})
    summary = {'schema_version': 'deepmew.refgames_factory.game.v0', 'game_id': 'game', 'game_index': 0, 'seed': 700000, 'decisions': 2, 'result': 0, 'crashed': False, 'timed_out': False, 'error': ''}
    return frames, rows, summary


def test_recovery_preserves_masked_context_without_inventing_results(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    result = recover_reference_game(*reference_game)
    sequence, = result.sequences
    assert result.recorded_requests == result.missing_immediate_post_states == 2
    assert [row.loss_eligible for row in sequence.predictions] == [True, False]
    assert len(sequence.events) == 4 and len(sequence.predictions) == 2
    assert not sequence.preserved_transition_ids
    assert 'PRIVATE' not in str(sequence.records) and 'search_begin_input' not in str(sequence.records)
    assert 'observed_action_result' not in str(sequence.records)
    assert 'past_action' in str(sequence.records)
    assert result.admission.startswith('candidate_requires')


@pytest.mark.parametrize('defect', ['missing', 'order', 'actor', 'game', 'seed', 'terminal', 'observation', 'action', 'post_state'])
def test_recovery_rejects_broken_source_evidence(reference_game: tuple[list[Payload], list[Payload], Payload], defect: str) -> None:
    frames, rows, summary = copy.deepcopy(reference_game)
    if defect == 'missing':
        rows.pop()
    elif defect == 'order':
        rows.reverse()
    elif defect == 'actor':
        frames[1]['actor_order'] = 0
    elif defect == 'game':
        rows[1]['game_id'] = 'other'
    elif defect == 'seed':
        rows[1]['seed'] += 1
    elif defect == 'terminal':
        summary['crashed'] = True
    elif defect == 'observation':
        rows[1]['raw_observation']['current']['turn'] += 1
    elif defect == 'action':
        rows[1]['chosen_action']['indices'] = [99]
    else:
        frames[1]['action_result']['state_after_reference'] = {'source_line': 3}
    with pytest.raises(ValueError):
        recover_reference_game(frames, rows, summary)


def test_reserved_split_wins_for_whole_game(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    frames, rows, summary = reference_game
    frames[1]['effective_split'] = 'holdout'
    result = recover_reference_game(frames, rows, summary)
    assert result.effective_split == 'holdout'
    assert all(sequence.split == 'holdout' for sequence in result.sequences)


def test_exclusion_changes_loss_not_history(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    original = recover_reference_game(*reference_game)
    signature = original.sequences[0].predictions[0].signature
    masked = recover_reference_game(*reference_game, frozenset({signature}))
    assert not any(row.loss_eligible for row in masked.sequences[0].predictions)
    assert masked.sequences[0].records == original.sequences[0].records


def test_canonical_storage_order_does_not_change_causal_history(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    frames, rows, summary = reference_game
    original = recover_reference_game(frames, rows, summary)
    assert recover_reference_game(list(reversed(frames)), rows, summary) == original
    frames[1]['global_order'] = frames[0]['global_order']
    with pytest.raises(ValueError, match='reference coordinate'):
        recover_reference_game(frames, rows, summary)


def test_cumulative_public_stream_cannot_introduce_future_events(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    frames, rows, summary = reference_game
    for row in rows:
        row['ordered_public_reveal_stream'] = []
    assert recover_reference_game(frames, rows, summary).public_events == 0
    rows[0]['ordered_public_reveal_stream'] = [{'event_index': 0, 'type': 999, 'playerIndex': 1}]
    with pytest.raises(ValueError, match='recorded causal prefix'):
        recover_reference_game(frames, rows, summary)


def test_intervening_public_events_cross_seats_without_private_identity(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    from deepmew_ex.evaluation.league.temporal import FullPrefixHistory
    frames, rows, summary = reference_game
    event = {'type': 999, 'playerIndex': 0, 'cardId': 'PRIVATE-CARD', 'serial': 'PRIVATE-SERIAL'}
    rows[0]['raw_observation']['logs'] = [event]
    rows[1]['seat'] = frames[1]['actor_seat'] = 1
    rows[1]['raw_observation']['current']['yourIndex'] = 1
    rows[1]['raw_observation']['current']['players'].reverse()
    frames[1]['actor_order'] = 0
    frames[1]['terminal_target'] = 0.0
    for frame, row in zip(frames, rows, strict=True):
        row['ordered_public_reveal_stream'] = [{'event_index': 0, **event}]
        frame['state_before']['content_sha256'] = hashlib.sha256(json.dumps(row['raw_observation'], sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    result = recover_reference_game(frames, rows, summary)
    assert result.public_events == 1
    other = result.sequences[1]
    assert any(path == '/public_log_snapshot/0/type' and value == 'int:999' for path, value, _ in other.records[0])
    assert 'PRIVATE-CARD' not in str(other.records) and 'PRIVATE-SERIAL' not in str(other.records)
    histories = [FullPrefixHistory(public_event_mode=True) for _ in (0, 1)]
    for row in rows:
        for history in histories:
            history.observe(row['raw_observation'])
        history = histories[row['seat']]
        decision = history.prepare(row['raw_observation'])
        history.commit(decision, [row['chosen_action']['indices']])
    for sequence in result.sequences:
        assert tuple(histories[sequence.actor].records) == sequence.records
        assert tuple(histories[sequence.actor].events) == sequence.events
