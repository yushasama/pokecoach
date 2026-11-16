from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.config import GoldenConfig
from deepmew_ex.datasets.v2.conversion import run_golden
from deepmew_ex.datasets.v2.reader import iter_episodes
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.store import file_hash, verify_run


def test_conversion_immutable_inputs_outputs_and_partition_isolation(golden_inventory: Path, tmp_path: Path) -> None:
    before = {path: file_hash(path) for path in tmp_path.glob('*.json')}
    config = GoldenConfig(golden_inventory, 'first', OutputPaths(tmp_path / 'outputs'))
    path = run_golden(config)
    assert before == {item: file_hash(item) for item in before}
    all_episodes = list(iter_episodes(path))
    groups = {split: {episode.source_native_game_id for episode in iter_episodes(path, split=split)} for split in ('golden_development', 'golden_validation')}
    assert not groups['golden_development'] & groups['golden_validation']
    assert len(all_episodes) == 2 and sum(map(len, groups.values())) == 2
    assert verify_run(path)['optimizer_updates'] == 0
    with pytest.raises(FileExistsError):
        run_golden(config)
    second = run_golden(GoldenConfig(golden_inventory, 'second', config.paths))
    assert json.loads((path / 'conversion.json').read_text())['identity_root_sha256'] == json.loads((second / 'conversion.json').read_text())['identity_root_sha256']
    with (path / 'episode-000.json').open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(ValueError, match='integrity'):
        list(iter_episodes(path))


def test_unsupported_objective_precedes_all_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='unsupported V2 objective'):
        list(iter_episodes(tmp_path / 'missing', 'td_transition'))


@pytest.mark.parametrize('failure', ['bytes', 'size', 'evidence'])
def test_failed_attempt_is_sealed(golden_inventory: Path, tmp_path: Path, failure: str) -> None:
    inventory = json.loads(golden_inventory.read_text())
    if failure == 'bytes':
        path = Path(inventory['sources'][0]['path'])
        path.write_bytes(path.read_bytes().replace(b'cabt', b'nope'))
    elif failure == 'size':
        inventory['sources'][0]['byte_size'] += 1
    else:
        inventory['evidence'] = [{'path': str(golden_inventory), 'sha256': '0' * 64}]
    golden_inventory.write_text(json.dumps(inventory))
    config = GoldenConfig(golden_inventory, 'rejected', OutputPaths(tmp_path / 'outputs'))
    with pytest.raises(ValueError):
        run_golden(config)
    assert verify_run(config.output)['status'] == 'failed'
    assert json.loads((config.output / 'failure.json').read_text())['completed_episodes'] == 0


def test_parser_drift_and_duplicate_sources_precede_allocation(golden_inventory: Path, tmp_path: Path) -> None:
    inventory = json.loads(golden_inventory.read_text())
    inventory['parser_hashes'] = {}
    golden_inventory.write_text(json.dumps(inventory))
    config = GoldenConfig(golden_inventory, 'parser-drift', OutputPaths(tmp_path / 'outputs'))
    with pytest.raises(ValueError, match='binding'):
        run_golden(config)
    assert not config.output.exists()
    inventory['sources'].append(inventory['sources'][0])
    golden_inventory.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='duplicate'):
        run_golden(config)


def test_audited_parser_does_not_import_engine_or_trainer() -> None:
    code = "from deepmew_ex.datasets.v2.contract import load_contract; import sys; load_contract(); assert not any(n == 'cg' or n.startswith(('cg.', 'torch', 'deepmew.')) for n in sys.modules)"
    subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True, text=True)
