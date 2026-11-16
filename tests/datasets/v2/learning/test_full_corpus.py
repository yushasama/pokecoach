from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from deepmew_ex.datasets.v2.corpus.config import CorpusConfig
from deepmew_ex.datasets.v2.corpus.runner import run_corpus
from deepmew_ex.datasets.v2.learning.full_corpus import FullCorpus, index_reference
from deepmew_ex.experiments.config import OutputPaths
from tests.datasets.v2.corpus.conftest import corpus_inventory


def test_all_reference_rows_guard_holdouts_even_when_old_target_mask_is_false(corpus_inventory: Path, tmp_path: Path) -> None:
    root = run_corpus(CorpusConfig(corpus_inventory, 'full-test', OutputPaths(tmp_path / 'outputs'), workers=1, min_free_bytes=0))
    output = tmp_path / 'stream'
    output.mkdir()
    corpus = FullCorpus(root, output, lambda event, fields: None)
    corpus.index_snapshots()
    corpus.audit_raw()
    corpus.audit_reserved_snapshots()
    assert len(corpus.signatures) == 2
    assert corpus.connection.execute('SELECT COUNT(*) FROM refs').fetchone()[0] == 2
    sequences = list(corpus.snapshots('validation'))
    assert len(sequences) == 2
    assert [row.predictions[0].loss_eligible for row in sequences] == [True, False]
    assert all(not row.history_available for row in sequences)
    assert 'PRIVATE' not in str([row.records for row in sequences])
    assert not list(corpus.snapshots('train'))
    corpus.close()


def test_bound_parser_drift_rejected_before_source_resolution(corpus_inventory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = run_corpus(CorpusConfig(corpus_inventory, 'parser-test', OutputPaths(tmp_path / 'outputs'), workers=1, min_free_bytes=0))
    import deepmew_ex.datasets.v2.learning.full_corpus as owner
    monkeypatch.setattr(owner, 'binding_hashes', lambda: {'changed.py': '0' * 64})
    with pytest.raises(ValueError, match='parser changed'):
        FullCorpus(root, tmp_path, lambda event, fields: None)


@pytest.mark.parametrize('splits', [('train', 'holdout'), ('holdout', 'train')])
def test_duplicate_reference_uses_reserved_split_independent_of_order(splits: tuple[str, str]) -> None:
    connection = sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE refs (id TEXT PRIMARY KEY,source TEXT,line INTEGER,split TEXT,family TEXT,priority INTEGER,payload BLOB)')
    for split in splits:
        frame = {'frame_id': 'same', 'state_before': {'source_path': 'same.jsonl', 'source_line': 1}, 'effective_split': split, 'source_family': 'fixture'}
        index_reference(connection, frame, split.encode())
    assert connection.execute('SELECT split,payload FROM refs').fetchall() == [('holdout', b'holdout')]
    connection.close()
