from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.indexes import lookup_run, rebuild
from deepmew_ex.registry.store import RunStore, atomic_new, file_hash, verify_run


def test_exclusive_immutable_dummy_runs_and_resume_link(tmp_path: Path) -> None:
    paths = OutputPaths(tmp_path / 'outputs')
    first = RunStore.allocate(paths, 'family', 'variant', 'first')
    first.write('config.json', {'seed': 42})
    first.seal({'run': {'run_id': 'first', 'family_id': 'family', 'variant_id': 'variant'}, 'artifacts': []})
    before = file_hash(first.path / 'manifest.json')
    second = RunStore.continue_attempt(paths, 'family', 'variant', 'second', first.path)
    resume = json.loads((second.path / 'resume.json').read_text())
    assert resume['parent_run_id'] == 'first' and not resume['optimizer_rng_restored']
    second.seal({'run': {'run_id': 'second'}, 'artifacts': []})
    verify_run(first.path)
    verify_run(second.path)
    assert before == file_hash(first.path / 'manifest.json')
    with pytest.raises(ValueError, match='immutable'):
        first.write('extra.json', {})
    with pytest.raises(ValueError, match='immutable'):
        first.seal({})
    with pytest.raises(FileExistsError):
        RunStore.allocate(paths, 'family', 'variant', 'first')


@pytest.mark.parametrize('component', ['../escape', '..', '/absolute', 'a/b', 'a\\b', 'x:drive', ''])
def test_path_traversal(tmp_path: Path, component: str) -> None:
    with pytest.raises(ValueError):
        RunStore.allocate(OutputPaths(tmp_path), component, 'variant', 'run')


def test_interruption_corruption_and_write_collision(tmp_path: Path) -> None:
    store = RunStore.allocate(OutputPaths(tmp_path), 'family', 'variant', 'run')
    with pytest.raises(ValueError, match='incomplete'):
        verify_run(store.path)
    with patch('os.replace', side_effect=OSError('interrupted publication')):
        with pytest.raises(OSError, match='interrupted'):
            store.write('interrupted.json', {})
    assert not (store.path / 'interrupted.json').exists()
    assert not list(store.path.glob('*.tmp'))
    assert not list(store.path.glob('*.lock'))
    store.write('value.json', {'value': 1})
    with pytest.raises(FileExistsError):
        store.write('value.json', {'value': 2})
    with pytest.raises(ValueError, match='traversal'):
        store.write('../escape.json', {})
    store.seal({})
    (store.path / 'value.json').write_text('corruption')
    with pytest.raises(ValueError, match='integrity'):
        verify_run(store.path)
    (store.path / 'value.json').unlink()
    with pytest.raises(ValueError, match='integrity'):
        verify_run(store.path)


def test_index_rebuild_and_artifact_to_run_dataset_links(s0_run: Path) -> None:
    manifest = verify_run(s0_run)
    indexes = rebuild(s0_run.parents[2])
    assert any(record['run_id'] == manifest['run']['run_id'] for record in indexes['runs'])
    assert any(record['model_spec_id'] == manifest['run']['model_spec_id'] for record in indexes['models'])
    for artifact in manifest['artifacts']:
        assert artifact['run_id'] == manifest['run']['run_id']
        assert artifact['sha256'] == file_hash(s0_run / artifact['path'])
        assert lookup_run(s0_run.parents[2], artifact['artifact_id']) == s0_run
    assert lookup_run(s0_run.parents[2], manifest['run']['model_id']) == s0_run
    assert lookup_run(s0_run.parents[2], manifest['evaluation']['evaluation_id']) == s0_run
    lineage = json.loads((s0_run / 'lineage.json').read_text())
    assert any(edge['relation'] == 'uses_dataset' and edge['source'] == manifest['run']['run_id'] for edge in lineage['edges'])


def test_variant_cannot_change_model_specification(tmp_path: Path) -> None:
    paths = OutputPaths(tmp_path / 'outputs')
    first = RunStore.allocate(paths, 'family', 'variant-v1', 'one', specification={'model_spec_id': 'width64'})
    first.seal({})
    with pytest.raises(ValueError, match='new variant'):
        RunStore.allocate(paths, 'family', 'variant-v1', 'two', specification={'model_spec_id': 'width32'})
    assert not paths.run('family', 'variant-v1', 'two').exists()
    second = RunStore.allocate(paths, 'family', 'variant-v2', 'two', specification={'model_spec_id': 'width32'})
    second.seal({})
    verify_run(first.path)
    verify_run(second.path)


def test_nested_manifest_is_in_integrity_scope(tmp_path: Path) -> None:
    store = RunStore.allocate(OutputPaths(tmp_path), 'family', 'variant', 'run')
    store.write('checkpoints/manifest.json', {'checkpoint': 'original'})
    manifest = store.seal({})
    assert 'checkpoints/manifest.json' in manifest['files_sha256']
    (store.path / 'checkpoints/manifest.json').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        verify_run(store.path)
