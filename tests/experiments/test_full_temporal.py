from __future__ import annotations

from deepmew_ex.datasets.v2.learning.history import LEGACY_INPUT_CONTRACT, QUERY_INPUT_CONTRACT

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

import pytest

from deepmew_ex.training.bitmap_ledger import verify_run_ledger
import torch

from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, actor_sequences
from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.experiments.v2.full import FullTrainingConfig, train
from tests.datasets.v2.learning.test_admission import episode


class FixtureCorpus:
    def __init__(self, root: Path, output: Path, progress: Callable[[str, dict[str, object]], None]) -> None:
        self.root = root
        self.raw = [1, 2]
        self.report = {'catalog_rows': 4, 'raw_quarantine_reasons': {}}
        self.parser_hashes = binding_hashes()
        self.counts: dict[str, int] = {}
        self.reserved_audited = False

    def audit_raw(self) -> None:
        pass

    def index_snapshots(self) -> None:
        pass

    def audit_reserved_snapshots(self) -> None:
        self.reserved_audited = True

    def raw_sequences(self, split: str) -> Iterator[ActorSequence]:
        assert self.reserved_audited
        for index in (0, 1) if split == 'train' else (2,):
            yield from actor_sequences(episode(index), split)

    def snapshots(self, split: str) -> Iterator[ActorSequence]:
        raise AssertionError('temporal trainer must never consume history-free snapshots')

    def close(self) -> None:
        pass


def fixture_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FullTrainingConfig:
    import deepmew_ex.experiments.v2.full as runner
    monkeypatch.setattr(runner, 'FullCorpus', FixtureCorpus)
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'manifest.json').write_text('{}')
    return FullTrainingConfig(str(corpus), str(tmp_path / 'outputs'), device='cpu', hidden=16, embedding=4, layers=1, heads=2, snapshot_batch=2, checkpoint_updates=2, minimum_free_bytes=1, max_wall_seconds=120)


@pytest.mark.parametrize('input_contract', [LEGACY_INPUT_CONTRACT, QUERY_INPUT_CONTRACT])
def test_real_prepared_reader_runs_two_complete_epochs_without_snapshot_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_contract: str) -> None:
    from deepmew_ex.datasets.v2.contract import binding_hashes
    from deepmew_ex.datasets.v2.learning.prepared import prepare_membership
    from deepmew_ex.registry.store import RunStore, file_hash
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'manifest.json').write_text('{}')
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    rows = [sequence for index, split in enumerate(('train', 'train', 'validation')) for sequence in actor_sequences(episode(index), split)]
    if input_contract == QUERY_INPUT_CONTRACT:
        from deepmew_ex.datasets.v2.learning.raw_recovery import recover_raw_histories
        from tests.datasets.v2.learning.test_raw_recovery import source_replay
        source_episode, raw = source_replay()
        recovered = recover_raw_histories(source_episode, raw, 'train')
        rows = [replace(sequence, game_id=f'fixture-{index}', split=split, predictions=tuple(replace(row, row_id=f'{index}:{row.row_id}') for row in sequence.predictions)) for index, split in enumerate(('train', 'train', 'validation')) for sequence in recovered]
        # Distinct observed states prevent intentional reserved-prefix masking of
        # these three fixture games; preserve the query-aware record structure.
        rows = [replace(sequence, records=tuple([*record, ('/fixture_game', f'str:{sequence.game_id}', 0.0)] if record_index == sequence.events[0] else record for record_index, record in enumerate(sequence.records))) for sequence in rows]
    with sqlite3.connect(prepared / 'membership.sqlite3') as connection:
        membership = prepare_membership(connection, rows, lambda: None)
    RunStore(prepared).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': input_contract, 'parser_hashes': binding_hashes(), 'original_corpus_manifest_sha256': file_hash(original / 'manifest.json'), 'membership': membership, 'source_coverage': {'catalog_rows': 12, 'scope': 'synthetic trainer input-contract serialization fixture; feature parity tested separately'}})
    config = FullTrainingConfig(str(original), str(tmp_path / 'outputs'), device='cpu', hidden=16, embedding=4, layers=1, heads=2, epochs=2, temporal_batch_size=3, checkpoint_updates=2, minimum_free_bytes=1, max_wall_seconds=120, prepared_corpus=str(prepared))
    result = train(config, 'prepared-two-epochs')
    expected = json.loads((result / 'expected-membership.json').read_text())
    assert expected['expected_presentations'] == 2 * expected['unique_eligible_decisions'] > 0
    assert verify_run_ledger(result)['presentations'] == expected['expected_presentations']
    assert not (result / 'snapshot-scope.json').exists()
    assert json.loads((result / 'prepared-admission.json').read_text())['manifest_sha256'] == file_hash(prepared / 'manifest.json')
    assert json.loads((result / 'evaluations/selected-reload.json').read_text())['reloaded_from_checkpoint']
    from deepmew_ex.evaluation.league.temporal import load_full_prefix_checkpoint
    selected = json.loads((result / 'selection.json').read_text())['selected']
    model, evidence = load_full_prefix_checkpoint(Path(selected['checkpoint']), selected['sha256'], 'cpu')
    assert evidence['input_contract'] == input_contract
    assert evidence['public_event_mode'] == (input_contract == QUERY_INPUT_CONTRACT)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    saved = torch.load(selected['checkpoint'], weights_only=True)
    assert saved['input_contract'] == input_contract
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'load', lambda *args, **kwargs: {**saved, 'input_contract': 'unknown'})
        with pytest.raises(ValueError, match='input contract'):
            load_full_prefix_checkpoint(Path(selected['checkpoint']), selected['sha256'], 'cpu')


@pytest.mark.parametrize('workers', [0, 2])
def test_seeded_shuffle_consumes_every_prepared_target_each_epoch(tmp_path: Path, workers: int) -> None:
    from deepmew_ex.datasets.v2.learning.prepared import prepare_membership
    from deepmew_ex.registry.store import RunStore, file_hash
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'manifest.json').write_text('{}')
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    rows = [sequence for index, split in enumerate(('train', 'train', 'validation')) for sequence in actor_sequences(episode(index), split)]
    with sqlite3.connect(prepared / 'membership.sqlite3') as connection:
        membership = prepare_membership(connection, rows, lambda: None)
    RunStore(prepared).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': LEGACY_INPUT_CONTRACT, 'parser_hashes': binding_hashes(), 'original_corpus_manifest_sha256': file_hash(original / 'manifest.json'), 'membership': membership, 'source_coverage': {'catalog_rows': 12, 'scope': 'seeded-order trainer fixture'}})
    config = FullTrainingConfig(str(original), str(tmp_path / 'outputs'), device='cpu', hidden=16, embedding=4, layers=1, heads=2, epochs=2, temporal_batch_size=1, checkpoint_updates=2, minimum_free_bytes=1, max_wall_seconds=120, prepared_corpus=str(prepared), shuffle_train_sequences=True, loader_workers=workers)
    result = train(config, f'prepared-shuffled-w{workers}')
    expected = json.loads((result / 'expected-membership.json').read_text())
    proof = verify_run_ledger(result)
    assert proof['missing'] == 0 and proof['presentations'] == expected['expected_presentations'] == 2 * expected['unique_eligible_decisions'] > 0
    assert proof['per_epoch'] == [expected['unique_eligible_decisions']] * 2
    # The exact proof is the bitmap ledger; no per-decision database is written.
    assert not (result / 'consumption.sqlite3').exists()
    assert json.loads((result / 'config.json').read_text())['shuffle_train_sequences'] is True


def test_light_launcher_keeps_torch_out_of_spawned_worker_imports() -> None:
    import subprocess
    import sys
    probe = 'import sys, runpy; runpy.run_module("deepmew_ex.experiments.v2.launch_full", run_name="__mp_main__"); print("torch" in sys.modules)'
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'False'


def test_seeded_shuffle_requires_prepared_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='prepared corpus'):
        FullTrainingConfig(str(tmp_path), str(tmp_path / 'outputs'), shuffle_train_sequences=True).validate()
    with pytest.raises(ValueError, match='shuffle'):
        FullTrainingConfig(str(tmp_path), str(tmp_path / 'outputs'), prepared_corpus='x', shuffle_train_sequences=1).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='prepared corpus'):
        FullTrainingConfig(str(tmp_path), str(tmp_path / 'outputs'), loader_workers=2).validate()
    for workers in (-1, 33, True):
        with pytest.raises(ValueError, match='loader_workers'):
            FullTrainingConfig(str(tmp_path), str(tmp_path / 'outputs'), prepared_corpus='x', loader_workers=workers).validate()  # type: ignore[arg-type]


def test_training_complete_coverage_and_final_checkpoint_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture_config(tmp_path, monkeypatch)
    result = train(config, 'complete')
    manifest = json.loads((result / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    receipt = json.loads((result / 'lineage-receipt.json').read_text())
    assert receipt['unique_examples_actually_consumed'] == 8
    assert json.loads((result / 'expected-membership.json').read_text())['unique_eligible_decisions'] == 8
    for population in ('raw',):
        evaluation = json.loads((result / f'evaluations/final-{population}.json').read_text())
        assert evaluation['checkpoint_sha256'] == receipt['checkpoint_hash']
        assert evaluation['rows'] > 0 and evaluation['reloaded_from_checkpoint']


def test_resumed_best_from_failed_parent_is_evaluable_in_completed_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deepmew_ex.evaluation.league.temporal import load_full_prefix_checkpoint
    from deepmew_ex.registry.store import file_hash
    config = fixture_config(tmp_path, monkeypatch)
    parent = train(config, 'selected-parent')
    manifest_path = parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    checkpoint = parent / manifest['checkpoint']
    selected_parent = json.loads((parent / 'selection.json').read_text())['selected']
    manifest['status'] = 'failed'
    manifest_path.write_text(json.dumps(manifest))
    child = train(config, 'selected-child', checkpoint)
    selected = json.loads((child / 'selection.json').read_text())['selected']
    assert Path(selected['checkpoint']).parent == child / 'checkpoints'
    assert selected['sha256'] == selected_parent['sha256'] == file_hash(Path(selected['checkpoint']))
    assert json.loads((child / 'selected-origin.json').read_text())['path'] == selected_parent['checkpoint']
    _, evidence = load_full_prefix_checkpoint(Path(selected['checkpoint']), selected['sha256'], 'cpu')
    assert evidence['completed_training_run']


def test_owner_stop_preserves_committed_optimizer_and_exact_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deepmew_ex.registry.store import verify_run
    config = replace(fixture_config(tmp_path, monkeypatch), epochs=2, temporal_batch_size=3)
    baseline = train(config, 'owner-stop-baseline')
    stopped = tmp_path / 'outputs/deepmew_ex/v2_full_temporal_v1/owner-stop'
    original = torch.optim.AdamW.step

    def request_stop(optimizer: torch.optim.AdamW, closure: Callable[[], float] | None = None) -> object:
        result = original(optimizer, closure)
        (stopped / 'STOP').write_text('usage checkpoint reached', encoding='utf-8')
        return result

    monkeypatch.setattr(torch.optim.AdamW, 'step', request_stop)
    with pytest.raises(InterruptedError, match='owner or usage checkpoint'):
        train(config, 'owner-stop')
    assert verify_run(stopped)['status'] == 'failed'
    failure = json.loads((stopped / 'failure.json').read_bytes())
    assert failure['updates'] == 1 and not failure['interrupted_mid_update']
    saved_path, = (stopped / 'checkpoints').glob('interrupted-*.pt')
    saved = torch.load(saved_path, weights_only=True)
    assert verify_run_ledger(stopped)['presentations'] == saved['presentations']
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    resumed = train(config, 'owner-stop-resumed', saved_path)
    first = torch.load(baseline / verify_run(baseline)['checkpoint'], weights_only=True)
    second = torch.load(resumed / verify_run(resumed)['checkpoint'], weights_only=True)
    assert first['presentations'] == second['presentations']
    assert first['scheduler'] == second['scheduler']
    assert torch.equal(first['torch_rng'], second['torch_rng'])
    for name in first['model']:
        torch.testing.assert_close(first['model'][name], second['model'][name], rtol=0, atol=0)
    assert first['optimizer']['param_groups'] == second['optimizer']['param_groups']
    for parameter, state in first['optimizer']['state'].items():
        for key, value in state.items():
            actual = second['optimizer']['state'][parameter][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual, rtol=0, atol=0)
            else:
                assert value == actual


def test_preflight_detects_entire_game_dropped_by_training_iterator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture_config(tmp_path, monkeypatch)
    original = FixtureCorpus.raw_sequences
    train_passes = 0

    def drop_after_preflight(corpus: FixtureCorpus, split: str) -> Iterator[ActorSequence]:
        nonlocal train_passes
        if split == 'train':
            train_passes += 1
        rows = list(original(corpus, split))
        yield from rows[2:] if split == 'train' and train_passes > 1 else rows

    monkeypatch.setattr(FixtureCorpus, 'raw_sequences', drop_after_preflight)
    with pytest.raises(ValueError, match='coverage mismatch: missing=4'):
        train(config, 'dropped-game')
    failure = tmp_path / 'outputs/deepmew_ex/v2_full_temporal_v1/dropped-game/manifest.json'
    assert json.loads(failure.read_text())['status'] == 'failed'


def test_two_complete_epochs_count_presentations_separately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(fixture_config(tmp_path, monkeypatch), epochs=2, temporal_batch_size=3)
    result = train(config, 'two-epochs')
    expected = json.loads((result / 'expected-membership.json').read_text())
    assert expected['unique_eligible_decisions'] == 8 and expected['expected_presentations'] == 16
    manifest = json.loads((result / 'manifest.json').read_text())
    checkpoint = torch.load(result / manifest['checkpoint'], weights_only=True)
    assert checkpoint['updates'] == 4 and checkpoint['scheduler']['presentations'] == 16
    selection = json.loads((result / 'selection.json').read_text())
    evaluations = [json.loads((result / f'evaluations/epoch-{epoch}-raw.json').read_text()) for epoch in range(2)]
    assert selection['selected']['loss'] == min(row['loss'] for row in evaluations)


def tamper_first_context(run: Path) -> None:
    """Increment the events count of update 1's first context row inside ledger.bin."""
    from deepmew_ex.training.bitmap_ledger import RECORD_HEADER
    data = bytearray((run / 'ledger.bin').read_bytes())
    _, update, _, count, contexts = RECORD_HEADER.unpack_from(data, 0)
    assert update == 1 and contexts
    offset = RECORD_HEADER.size + 4 * count + 4 * 3
    data[offset:offset + 4] = (int.from_bytes(data[offset:offset + 4], 'little') + 1).to_bytes(4, 'little')
    (run / 'ledger.bin').write_bytes(bytes(data))


def test_resume_rejects_tampered_parent_ledger_before_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture_config(tmp_path, monkeypatch)
    baseline = train(config, 'ledger-parent')
    checkpoint = baseline / json.loads((baseline / 'manifest.json').read_text())['checkpoint']
    tamper_first_context(baseline)

    def forbidden_step(optimizer: torch.optim.AdamW, closure: Callable[[], float] | None = None) -> object:
        raise AssertionError('tampered resume must fail before an update')

    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden_step)
    with pytest.raises(ValueError, match='run integrity mismatch'):
        train(config, 'tampered-resume', checkpoint)


def test_resume_rejects_changed_numerical_runtime_before_optimizer_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture_config(tmp_path, monkeypatch)
    baseline = train(config, 'numerical-parent')
    checkpoint = baseline / json.loads((baseline / 'manifest.json').read_text())['checkpoint']
    previous = torch.get_float32_matmul_precision()
    def forbidden_step(optimizer: torch.optim.AdamW, closure: Callable[[], float] | None = None) -> object:
        raise AssertionError('numerical drift must fail before an update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden_step)
    try:
        torch.set_float32_matmul_precision('medium' if previous != 'medium' else 'highest')
        with pytest.raises(ValueError, match='numerical runtime policy changed'):
            train(config, 'numerical-drift', checkpoint)
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(os.environ.get('DEEPMEW_TEST_CUDA') != '1', reason='explicit bounded CUDA verification only'))])
def test_interrupted_optimizer_is_not_checkpointed_and_resume_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    config = replace(fixture_config(tmp_path, monkeypatch), epochs=2, temporal_batch_size=3, device=device)
    monkeypatch.setattr('deepmew_ex.models.temporal.ACTIVATION_LEAF_BUDGET', 64)
    baseline = train(config, 'baseline')
    original = torch.optim.AdamW.step
    calls = 0

    def fail_second(optimizer: torch.optim.AdamW, closure: Callable[[], float] | None = None) -> object:
        nonlocal calls
        result = original(optimizer, closure)
        calls += 1
        if calls == 2:
            raise RuntimeError('injected interruption after parameter mutation')
        return result

    monkeypatch.setattr(torch.optim.AdamW, 'step', fail_second)
    with pytest.raises(RuntimeError, match='injected interruption'):
        train(config, 'interrupted')
    failed = tmp_path / 'outputs/deepmew_ex/v2_full_temporal_v1/interrupted'
    assert json.loads((failed / 'failure.json').read_text())['interrupted_mid_update']
    checkpoints = list((failed / 'checkpoints').glob('*.pt'))
    assert len(checkpoints) == 1 and checkpoints[0].name.endswith('-u1.pt')
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    resumed = train(config, 'resumed', checkpoints[0])
    baseline_manifest = json.loads((baseline / 'manifest.json').read_text())
    resumed_manifest = json.loads((resumed / 'manifest.json').read_text())
    first = torch.load(baseline / baseline_manifest['checkpoint'], weights_only=True)
    second = torch.load(resumed / resumed_manifest['checkpoint'], weights_only=True)
    assert first['updates'] == second['updates'] and first['presentations'] == second['presentations']
    assert first['scheduler'] == second['scheduler']
    assert first['best_validation']['loss'] == second['best_validation']['loss']
    assert torch.equal(first['torch_rng'], second['torch_rng'])
    assert len(first['cuda_rng']) == len(second['cuda_rng'])
    assert all(torch.equal(left, right) for left, right in zip(first['cuda_rng'], second['cuda_rng'], strict=True))
    assert first['optimizer']['param_groups'] == second['optimizer']['param_groups']
    for parameter, state in first['optimizer']['state'].items():
        for key, value in state.items():
            actual = second['optimizer']['state'][parameter][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual, rtol=0, atol=0)
            else:
                assert value == actual
    for name in first['model']:
        torch.testing.assert_close(first['model'][name], second['model'][name], rtol=0, atol=0)


@pytest.mark.parametrize('condition', ['clean', 'corrupt', 'active'])
def test_unsealed_parent_uses_authenticated_checkpoint_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str) -> None:
    from deepmew_ex.training.recovery import TrainingLease
    config = replace(fixture_config(tmp_path, monkeypatch), epochs=2, temporal_batch_size=3)
    parent = train(config, 'unsealed-parent')
    manifest = json.loads((parent / 'manifest.json').read_text())
    expected = torch.load(parent / manifest['checkpoint'], weights_only=True)
    checkpoint = next((parent / 'checkpoints').glob('periodic-*-u1.pt'))
    (parent / 'manifest.json').unlink()
    if condition == 'corrupt':
        tamper_first_context(parent)
        with pytest.raises(ValueError, match='committed checkpoint identity'):
            train(config, 'unsealed-corrupt', checkpoint)
        return
    if condition == 'active':
        lease = TrainingLease(parent, require_existing=True)
        try:
            with pytest.raises(ValueError, match='writer is still active'):
                train(config, 'unsealed-active', checkpoint)
        finally:
            lease.close()
        return
    resumed = train(config, 'unsealed-resumed', checkpoint)
    actual_manifest = json.loads((resumed / 'manifest.json').read_text())
    actual = torch.load(resumed / actual_manifest['checkpoint'], weights_only=True)
    assert actual['scheduler'] == expected['scheduler']
    assert actual['ledger_digest'] == expected['ledger_digest']
    assert actual['deadline_epoch'] == expected['deadline_epoch']
    for name in expected['model']:
        torch.testing.assert_close(actual['model'][name], expected['model'][name], rtol=0, atol=0)
    assert json.loads((resumed / 'resume.json').read_text())['unsealed_parent']
