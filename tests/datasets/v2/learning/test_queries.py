from __future__ import annotations

from dataclasses import replace

import pytest

from deepmew_ex.datasets.v2.learning.queries import actor_queries, enrich_query_views
from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from tests.datasets.v2.learning.test_admission import episode


def observation() -> dict[str, object]:
    return {'current': {'yourIndex': 0, 'looking': [None, {'id': 88, 'serial': 7, 'playerIndex': 1}]}, 'select': {'deck': [{'id': 12, 'serial': 34, 'playerIndex': 0}, {'id': 56, 'serial': 78, 'playerIndex': 0}]}, 'search_begin_input': 'PRIVATE-SIMULATOR'}


def test_query_card_indices_and_facedown_placeholders_are_preserved() -> None:
    value = actor_queries(observation(), 0)
    assert value['deck'][1] == {'id': 56, 'serial': 78, 'playerIndex': 0}
    assert value['looking'][0] is None
    assert 'PRIVATE-SIMULATOR' not in str(value)
    with pytest.raises(ValueError, match='active observer'):
        actor_queries(observation(), 1)


@pytest.mark.parametrize('bad', ['opaque', [{'id': True, 'serial': 34, 'playerIndex': 0}], [{'id': 12, 'serial': 34, 'playerIndex': 0, 'hiddenExtra': 9}], [None]])
def test_untyped_or_unknown_query_fields_are_not_silently_learned(bad: object) -> None:
    value = observation()
    value['select']['deck'] = bad
    with pytest.raises(ValueError):
        actor_queries(value, 0)


def test_query_enrichment_preserves_targets_masks_and_other_records() -> None:
    sequence = actor_sequences(episode(0), 'train')[0]
    sequence = replace(sequence, predictions=(replace(sequence.predictions[0], loss_eligible=False), *sequence.predictions[1:]))
    upgraded = enrich_query_views(sequence, [observation() for _ in sequence.predictions])
    assert upgraded.predictions == sequence.predictions and upgraded.events == sequence.events
    current = {sequence.events[row.event_index] for row in sequence.predictions}
    for index, (before, after) in enumerate(zip(sequence.records, upgraded.records, strict=True)):
        if index not in current:
            assert before == after
        else:
            assert after[-len(before):] == before
    with pytest.raises(ValueError, match='already present'):
        enrich_query_views(upgraded, [observation() for _ in sequence.predictions])
