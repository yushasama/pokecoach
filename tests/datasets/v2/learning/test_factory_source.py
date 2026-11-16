from __future__ import annotations

import copy
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.learning.factory_source import recover_factory_source_game
from deepmew_ex.datasets.v2.learning.recovery import recover_reference_game
from deepmew_ex.datasets.v2.schema import Payload
from tests.datasets.v2.corpus.conftest import corpus_inventory
from tests.datasets.v2.learning.test_recovery import reference_game


def test_uncatalogued_factory_preserves_known_recovery_features_and_masks(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    frames, rows, summary = reference_game
    for row, frame in zip(rows, frames, strict=True):
        row['legal_set'] = frame['legal_actions']
    expected = recover_reference_game(frames, rows, summary)
    actual = recover_factory_source_game(rows, summary, Path('source.gz'), 'a' * 64, 1, 'train', (True, False))
    assert actual.recorded_requests == expected.recorded_requests
    assert actual.sequences[0].records == expected.sequences[0].records
    assert actual.sequences[0].events == expected.sequences[0].events
    assert [row.loss_eligible for row in actual.sequences[0].predictions] == [True, False]
    assert actual.game_id != expected.game_id


def test_reused_native_labels_do_not_merge_different_source_generations(reference_game: tuple[list[Payload], list[Payload], Payload]) -> None:
    frames, rows, summary = reference_game
    for row, frame in zip(rows, frames, strict=True):
        row['legal_set'] = frame['legal_actions']
    first = recover_factory_source_game(rows, summary, Path('first.gz'), 'a' * 64, 1, 'holdout', (True, True))
    second = recover_factory_source_game(rows, summary, Path('second.gz'), 'b' * 64, 1, 'holdout', (True, True))
    assert first.game_id != second.game_id
    assert first.sequences[0].records == second.sequences[0].records
    assert first.sequences[0].split == second.sequences[0].split == 'holdout'


@pytest.mark.parametrize('defect', ['legal', 'chosen', 'missing', 'terminal', 'mask'])
def test_uncatalogued_factory_still_rejects_broken_evidence(reference_game: tuple[list[Payload], list[Payload], Payload], defect: str) -> None:
    frames, rows, summary = copy.deepcopy(reference_game)
    for row, frame in zip(rows, frames, strict=True):
        row['legal_set'] = frame['legal_actions']
    masks = (True, False)
    if defect == 'legal':
        rows[0]['legal_set'] = []
    elif defect == 'chosen':
        rows[0]['chosen_action']['indices'] = [999]
    elif defect == 'missing':
        rows.pop()
        masks = (True,)
    elif defect == 'terminal':
        summary['timed_out'] = True
    else:
        masks = (True,)
    with pytest.raises(ValueError):
        recover_factory_source_game(rows, summary, Path('source.gz'), 'a' * 64, 1, 'train', masks)
