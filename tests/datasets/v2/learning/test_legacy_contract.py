from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.learning.legacy_contract import SPECIFICATION_SHA256, recorded_legacy_contract
from deepmew_ex.datasets.v2.learning.raw_recovery import recover_raw_histories
from deepmew_ex.registry.store import canonical, digest
from tests.datasets.v2.conftest import source_row, synthetic_replay


def legacy_replay() -> dict[str, object]:
    replay = synthetic_replay('legacy-compatibility', '1.30.1')
    replay['specification'] = json.loads(Path(__file__).with_name('legacy_specification.json').read_text())
    for step in replay['steps']:
        for agent in step:
            agent['observation']['current']['looking'] = None
            agent['observation']['select']['deck'] = None
            agent['observation']['logs'] = []
    return replay


def test_explicit_legacy_parser_preserves_native_version_and_full_histories() -> None:
    replay = legacy_replay()
    assert digest(canonical(replay['specification'])) == SPECIFICATION_SHA256
    base, bound = load_contract(), recorded_legacy_contract()
    assert base.source_contract(replay) is None
    raw = canonical(replay)
    episode = convert_episode(raw, source_row(raw, Path('unused'), replay['id'], '1.30.1'), bound)
    assert episode.module_version == '1.30.1'
    assert sum(len(sequence.predictions) for sequence in recover_raw_histories(episode, raw, 'train')) == len(episode.frames)
    assert base.source_contract(replay) is None
    assert bound.hashes['recorded_legacy_specification'] == SPECIFICATION_SHA256


@pytest.mark.parametrize('defect', ['specification', 'missing_specification', 'schema', 'name', 'version', 'module'])
def test_legacy_compatibility_does_not_guess_unknown_contracts(defect: str) -> None:
    replay = legacy_replay()
    if defect == 'specification':
        replay['specification']['action']['type'] = 'object'
    elif defect == 'missing_specification':
        replay.pop('specification')
    else:
        replay[{'schema': 'schema_version', 'module': 'module_version'}.get(defect, defect)] = 'unknown'
    assert recorded_legacy_contract().source_contract(replay) is None


def test_existing_supported_native_contract_is_unchanged() -> None:
    replay = synthetic_replay('supported', '1.30.2')
    assert recorded_legacy_contract().source_contract(replay) == load_contract().source_contract(replay)


def test_legacy_conversion_still_rejects_invalid_action() -> None:
    replay = legacy_replay()
    replay['steps'][1][0]['action'] = [999]
    raw = canonical(replay)
    with pytest.raises(ValueError):
        convert_episode(raw, source_row(raw, Path('unused'), replay['id'], '1.30.1'), recorded_legacy_contract())
