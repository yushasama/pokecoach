from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.corpus.config import CorpusConfig
from deepmew_ex.datasets.v2.corpus.runner import run_corpus
from deepmew_ex.datasets.v2.learning.recovery_index import build_reference_index
from deepmew_ex.experiments.config import OutputPaths
from tests.datasets.v2.corpus.conftest import corpus_inventory


def test_complete_index_preserves_masked_rows_and_resumes_committed_part(corpus_inventory: Path, tmp_path: Path) -> None:
    corpus = run_corpus(CorpusConfig(corpus_inventory, 'recovery-index', OutputPaths(tmp_path / 'outputs'), workers=1, min_free_bytes=0))
    index = tmp_path / 'references.sqlite3'

    def interrupt(event: dict[str, object]) -> None:
        raise RuntimeError('after committed part')

    with pytest.raises(RuntimeError, match='committed part'):
        build_reference_index(corpus, index, lambda: None, interrupt)
    receipt = build_reference_index(corpus, index, lambda: None, lambda event: None)
    assert receipt['reference_presentations'] == receipt['unique_references'] == 2
    assert receipt['parts'] == 1 and receipt['source_files'] == 1
    assert receipt['masked_rows_retained_for_context'] and not receipt['training_admitted']
    connection = sqlite3.connect(index)
    assert connection.execute('SELECT split,COUNT(*) FROM refs GROUP BY split').fetchall() == [('validation', 2)]
    connection.close()


def test_index_rejects_corpus_metadata_drift(corpus_inventory: Path, tmp_path: Path) -> None:
    corpus = run_corpus(CorpusConfig(corpus_inventory, 'index-drift', OutputPaths(tmp_path / 'outputs'), workers=1, min_free_bytes=0))
    (corpus / 'port.json').write_text('{}')
    with pytest.raises(ValueError, match='metadata changed'):
        build_reference_index(corpus, tmp_path / 'unused.sqlite3', lambda: None, lambda event: None)
    assert not (tmp_path / 'unused.sqlite3').exists()


def test_index_resume_binds_split_priority_dependency(corpus_inventory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.recovery_index as owner
    corpus = run_corpus(CorpusConfig(corpus_inventory, 'index-dependency', OutputPaths(tmp_path / 'outputs'), workers=1, min_free_bytes=0))
    index = tmp_path / 'references.sqlite3'
    build_reference_index(corpus, index, lambda: None, lambda event: None)
    original = owner.file_hash
    monkeypatch.setattr(owner, 'file_hash', lambda path: '0' * 64 if path.name == 'admission.py' else original(path))
    with pytest.raises(ValueError, match='identity changed'):
        build_reference_index(corpus, index, lambda: None, lambda event: None)
