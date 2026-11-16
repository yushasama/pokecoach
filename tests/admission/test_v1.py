from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from deepmew_ex.admission.v1 import audit_v1
from deepmew_ex.experiments.config import PROJECT_ROOT
from deepmew_ex.registry.store import file_hash, verify_run
from .conftest import CorpusFixture


def test_read_only_full_audit_support_and_selection(corpus: CorpusFixture) -> None:
    before = {path.name: file_hash(path) for path in corpus.config.corpus.iterdir()}
    path = audit_v1(corpus.config)
    report = json.loads((path / 'audit.json').read_text())
    assert verify_run(path)['status'] == 'completed'
    assert report['status'] == 'passed' and report['rows'] == 24 and report['games'] == 6
    for split in ('train', 'validation', 'holdout'):
        support = report['splits'][split]
        assert support['raw'] == 8 and support['eligible'] == 4
        assert support['excluded'] == {'forced': 1, 'zero_actor_weight': 1, 'optional_empty': 1, 'multi_select': 1, 'unsupported_selection': 0}
        assert sum(support['excluded'].values()) + support['eligible'] == support['raw']
    assert report['complete_action_count_differs_from_candidate_count'] == 6
    assert report['proposed_membership']['train']['rows'] == 3
    assert report['proposed_membership']['validation']['rows'] == 2
    assert report['optimizer_updates'] == 0 and not report['model_imported']
    assert not report['capabilities']['temporal_bc']
    assert report['visibility'] == 'receipt_supported_not_reproduced'
    assert {path.name: file_hash(path) for path in corpus.config.corpus.iterdir()} == before
    second = audit_v1(replace(corpus.config, audit_id='fixture-repeat'))
    assert (path / 'proposed-train-membership.json').read_bytes() == (second / 'proposed-train-membership.json').read_bytes()
    with pytest.raises(FileExistsError):
        audit_v1(corpus.config)


@pytest.mark.parametrize('kind', ['hash', 'offset', 'target', 'nonfinite', 'selection_cardinality', 'duplicate_id', 'split_leak', 'frozen_seed', 'extra_rows', 'empty_eligible'])
def test_corruption_and_bad_rows_fail_with_immutable_receipt(corpus: CorpusFixture, kind: str) -> None:
    arrays = corpus.arrays()
    metadata = corpus.metadata()
    if kind == 'hash':
        (corpus.config.corpus / 'shard_00000.npz').write_bytes(b'corrupt')
    elif kind == 'offset':
        arrays['option_offsets'][2] = -1
    elif kind == 'target':
        arrays['selection_indices'][0] = 999
    elif kind == 'nonfinite':
        arrays['value_features'][0, 0] = np.inf
    elif kind == 'selection_cardinality':
        arrays['min_count'][3] = 1
    elif kind == 'duplicate_id':
        metadata[1]['stable_row_id'] = metadata[0]['stable_row_id']
    elif kind == 'split_leak':
        metadata[8]['source_game_id'] = metadata[0]['source_game_id']
    elif kind == 'frozen_seed':
        metadata[0]['seed'] = 40009
    elif kind == 'extra_rows':
        metadata.append(metadata[-1])
    elif kind == 'empty_eligible':
        arrays['actor_loss_weight'][:] = 0
    if kind != 'hash':
        corpus.write_arrays(arrays)
        corpus.write_metadata(metadata)
        if kind == 'empty_eligible':
            manifest = json.loads(corpus.config.expected_manifest.read_text())
            manifest['shards'][0]['actor_rows'] = 0
            corpus.config.expected_manifest.write_text(json.dumps(manifest))
            expected = json.loads(corpus.config.validation_receipt.read_text())
            expected.update(actor_rows=0, forced_rows=24)
            corpus.config.validation_receipt.write_text(json.dumps(expected))
        corpus.rebind()
    with pytest.raises((ValueError, EOFError)):
        audit_v1(corpus.config)
    sealed = verify_run(corpus.config.output)
    assert sealed['status'] == 'failed'
    assert sealed['optimizer_updates'] == 0
    assert 'failure.json' in sealed['files_sha256']


def test_missing_or_mismatched_manifest_and_output_separation(corpus: CorpusFixture) -> None:
    original = corpus.config.corpus / 'EXPORT_MANIFEST.json'
    original.write_text('{}')
    with pytest.raises(ValueError, match='trusted'):
        audit_v1(corpus.config)
    with pytest.raises(ValueError, match='separate'):
        replace(corpus.config, output_paths=type(corpus.config.output_paths)(corpus.config.corpus)).validate()


def test_cli_audit_cannot_import_torch_or_engine(corpus: CorpusFixture) -> None:
    script = '''
import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','cg'):
        raise AssertionError('model/engine imported')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from deepmew_ex.cli import main
main(sys.argv[1:])
'''
    args = ['dataset', 'audit-v1', '--root', str(corpus.config.root), '--audit-id', 'cli-fixture', '--output-root', str(corpus.config.output_paths.root), '--manifest', str(corpus.config.expected_manifest), '--validation-receipt', str(corpus.config.validation_receipt)]
    result = subprocess.run([sys.executable, '-c', script, *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['result']['optimizer_updates'] == 0
