from __future__ import annotations

from pathlib import Path

import pytest

from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.indexes import lookup_run, rebuild
from deepmew_ex.registry.store import RunStore


def test_model_index_does_not_read_unrelated_dataset_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.registry.indexes as owner
    paths = OutputPaths(tmp_path / 'outputs')
    store = RunStore.allocate(paths, 'deepmew_ex', 'dataset', 'large-corpus')
    store.write_bytes('payload.bin', b'not model metadata')
    store.seal({'kind': 'v2_full_corpus_port'})
    def forbidden(path: Path) -> dict[str, object]:
        raise AssertionError('model lookup read unrelated dataset bytes')
    monkeypatch.setattr(owner, 'verify_run', forbidden)
    assert rebuild(paths.root) == {'families': [], 'models': [], 'runs': []}
    with pytest.raises(ValueError, match='unknown or ambiguous'):
        lookup_run(paths.root, 'nonexistent-model')
