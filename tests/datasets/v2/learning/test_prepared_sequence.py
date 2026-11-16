from __future__ import annotations

from dataclasses import asdict

import orjson
import pytest

from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences, sequence_from_payload
from tests.datasets.v2.learning.test_admission import episode


def test_prepared_roundtrip_preserves_exact_optional_multiselect_and_history() -> None:
    for sequence in actor_sequences(episode(), 'train'):
        assert sequence_from_payload(orjson.loads(orjson.dumps(asdict(sequence)))) == sequence


@pytest.mark.parametrize('defect', ['future_index', 'unordered', 'nonfinite', 'bad_action', 'duplicate_prediction'])
def test_prepared_reader_rejects_corrupt_coordinates(defect: str) -> None:
    value = orjson.loads(orjson.dumps(asdict(actor_sequences(episode(), 'train')[0])))
    if defect == 'future_index':
        value['events'][-1] = len(value['records'])
    elif defect == 'unordered':
        value['events'].reverse()
    elif defect == 'nonfinite':
        value['records'][0][0][2] = float('nan')
    elif defect == 'bad_action':
        value['predictions'][0]['stages'][0]['selected'] = [10000]
    else:
        value['predictions'].append(value['predictions'][0])
    with pytest.raises(ValueError):
        sequence_from_payload(value)
