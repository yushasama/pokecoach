from __future__ import annotations

from deepmew_ex.datasets.v2.learning.history import LEGACY_INPUT_CONTRACT, QUERY_INPUT_CONTRACT

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.prepared import PreparedCorpus, prefix_signatures, prepare_membership
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, actor_sequences
from deepmew_ex.registry.store import RunStore
from tests.datasets.v2.learning.test_admission import episode


def sample(index: int, split: str = 'train') -> ActorSequence:
    return actor_sequences(episode(index), split)[0]


def renamed(sequence: ActorSequence, game: str) -> ActorSequence:
    return replace(sequence, game_id=game, predictions=tuple(replace(row, row_id=game + ':' + str(index)) for index, row in enumerate(sequence.predictions)))


def prepare(tmp_path: Path, sequences: list[ActorSequence]) -> tuple[sqlite3.Connection, dict[str, object]]:
    connection = sqlite3.connect(tmp_path / 'membership.sqlite3')
    return connection, prepare_membership(connection, sequences, lambda: None)


def test_identical_current_state_with_different_history_remains_distinct(tmp_path: Path) -> None:
    first = sample(0)
    second = renamed(first, 'different-past')
    records = [list(record) for record in second.records]
    records[second.events[0]].append(('/earlier_effect', 'str:changed', 0.0))
    second = replace(second, records=tuple(records))
    assert [row.signature for row in first.predictions] == [row.signature for row in second.predictions]
    assert [signature for _, signature in prefix_signatures(first)] != [signature for _, signature in prefix_signatures(second)]
    connection, receipt = prepare(tmp_path, [first, second])
    assert receipt['reasons'] == {'admitted': len(first.predictions) + len(second.predictions)}
    connection.close()


def test_future_event_and_chosen_target_do_not_change_earlier_input_fingerprint() -> None:
    sequence = sample(0)
    records = [list(record) for record in sequence.records]
    records[sequence.events[-1]].append(('/future', 'str:unseen', 0.0))
    changed = replace(sequence, records=tuple(records))
    assert next(prefix_signatures(sequence))[1] == next(prefix_signatures(changed))[1]
    stage = sequence.predictions[0].stages[0]
    prediction = replace(sequence.predictions[0], stages=(replace(stage, selected=tuple(reversed(stage.selected))),))
    assert next(prefix_signatures(sequence))[1] == next(prefix_signatures(replace(sequence, predictions=(prediction, *sequence.predictions[1:]))))[1]


def test_exact_history_duplicates_preserve_context_and_all_valid_masks(tmp_path: Path) -> None:
    first = sample(0)
    masked = replace(first, predictions=tuple(replace(row, row_id='masked:' + row.row_id, loss_eligible=False) for row in first.predictions))
    copy = renamed(first, 'duplicate')
    connection, receipt = prepare(tmp_path, [masked, first, copy])
    assert receipt['reasons'] == {'admitted': len(first.predictions), 'source_mask': len(first.predictions), 'duplicate_actor_history': len(first.predictions)}
    assert connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] == 3
    connection.close()


def test_reserved_prefix_and_whole_game_split_win(tmp_path: Path) -> None:
    first = sample(0)
    reserved = replace(renamed(first, 'held'), split='holdout')
    same_game_other_actor = replace(sample(1), game_id=first.game_id, actor=1, split='validation')
    connection, receipt = prepare(tmp_path, [first, reserved, same_game_other_actor])
    assert connection.execute('SELECT split FROM sequences WHERE ordinal=0').fetchone()[0] == 'validation'
    assert connection.execute("SELECT COUNT(*) FROM predictions WHERE sequence_id=0 AND reason='reserved_causal_prefix'").fetchone()[0] == len(first.predictions)
    connection.close()


def test_history_free_candidate_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='causal history'):
        prepare(tmp_path, [replace(sample(0), history_available=False)])


def test_bounded_publication_transactions_preserve_exact_payloads_and_hashes(tmp_path: Path) -> None:
    from dataclasses import asdict
    import orjson
    import zstandard
    from deepmew_ex.registry.store import digest
    sequences = [renamed(sample(0), f'game-{index}') for index in range(130)]
    connection = sqlite3.connect(tmp_path / 'membership.sqlite3')
    commits: list[str] = []
    connection.set_trace_callback(lambda sql: commits.append(sql) if sql == 'COMMIT' else None)
    result = prepare_membership(connection, sequences, lambda: None)
    assert len(commits) == 4  # Three insertion batches and global admission.
    assert result['reasons']['admitted'] == len(sequences[0].predictions)
    for ordinal, payload, content in connection.execute('SELECT ordinal,payload,content FROM sequences ORDER BY ordinal'):
        sequence = sequences[ordinal]
        assert zstandard.ZstdDecompressor().decompress(payload) == orjson.dumps(asdict(sequence))
        semantic = {'records': sequence.records, 'events': sequence.events, 'targets': [{'event': row.event_index, 'stages': [asdict(stage) for stage in row.stages]} for row in sequence.predictions]}
        assert content == digest(orjson.dumps(semantic))
    connection.close()


def test_interrupted_publication_rolls_back_uncommitted_batch_without_admission(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / 'membership.sqlite3')
    calls = 0

    def stop() -> None:
        nonlocal calls
        calls += 1
        if calls == 66:
            raise InterruptedError('bounded publication pause')

    with pytest.raises(InterruptedError, match='bounded publication pause'):
        prepare_membership(connection, (renamed(sample(0), f'game-{index}') for index in range(130)), stop)
    assert not connection.in_transaction
    assert connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] == 64
    assert connection.execute('SELECT SUM(eligible) FROM predictions').fetchone()[0] == 0
    assert connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    connection.close()


def test_global_admission_uses_covering_metadata_without_per_prediction_sorts(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / 'membership.sqlite3')
    statements: list[str] = []
    connection.set_trace_callback(lambda sql: statements.append(sql) if sql.startswith(('INSERT INTO reservations SELECT', 'UPDATE predictions SET reason=')) else None)
    prepare_membership(connection, [sample(0), renamed(sample(0), 'copy'), sample(1, 'validation')], lambda: None)
    assert len(statements) == 2
    plans = [[row[3] for row in connection.execute('EXPLAIN QUERY PLAN ' + statement)] for statement in statements]
    assert all('TEMP B-TREE' not in detail and 'USING INTEGER PRIMARY KEY' not in detail for plan in plans for detail in plan)
    assert any('COVERING INDEX prediction_signatures' in detail for detail in plans[0])
    assert any('COVERING INDEX sequence_metadata' in detail for detail in plans[0])
    assert any('COVERING INDEX sequence_contents' in detail for detail in plans[1])
    assert any('COVERING INDEX prediction_eligible_positions' in detail for detail in plans[1])
    connection.close()


def test_reader_rejects_unsealed_or_nonadmitted_candidates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='incomplete run'):
        PreparedCorpus(tmp_path)
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': False})
    with pytest.raises(ValueError, match='sealed admitted'):
        PreparedCorpus(tmp_path)


def test_sealed_reader_preserves_every_context_record_and_source_mask(tmp_path: Path) -> None:
    sequence = sample(0)
    masked = replace(sequence, predictions=(replace(sequence.predictions[0], loss_eligible=False), *sequence.predictions[1:]))
    connection, receipt = prepare(tmp_path, [masked])
    connection.close()
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': LEGACY_INPUT_CONTRACT, 'parser_hashes': binding_hashes(), 'membership': receipt})
    reader = PreparedCorpus(tmp_path)
    assert list(reader.sequences('train')) == [masked]
    assert list(reader.sequences('holdout')) == []
    reader.close()


def test_seeded_order_yields_identical_admitted_population_in_deterministic_permutation(tmp_path: Path) -> None:
    def distinct(index: int) -> ActorSequence:
        sequence = renamed(sample(0), f'game-{index}')
        records = [list(record) for record in sequence.records]
        records[sequence.events[0]].append(('/fixture_game', f'str:{index}', 0.0))
        return replace(sequence, records=tuple(records))

    rows = [distinct(index) for index in range(24)]
    masked = replace(rows[5], predictions=(replace(rows[5].predictions[0], loss_eligible=False), *rows[5].predictions[1:]))
    rows[5] = masked
    connection, receipt = prepare(tmp_path, [*rows, renamed(rows[3], 'duplicate-of-3'), replace(distinct(99), split='validation')])
    connection.close()
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': LEGACY_INPUT_CONTRACT, 'parser_hashes': binding_hashes(), 'membership': receipt})
    reader = PreparedCorpus(tmp_path)
    canonical = list(reader.sequences('train'))
    first = list(reader.sequences('train', order_seed=7))
    again = list(reader.sequences('train', order_seed=7))
    other = list(reader.sequences('train', order_seed=8))
    assert len(canonical) == 24 and [row.game_id for row in canonical] == [row.game_id for row in rows]
    assert first == again
    assert sorted(first, key=lambda row: row.game_id) == sorted(canonical, key=lambda row: row.game_id)
    assert [row.game_id for row in first] != [row.game_id for row in canonical]
    assert [row.game_id for row in first] != [row.game_id for row in other]
    assert next(row for row in first if row.game_id == 'game-5').predictions == masked.predictions
    assert [row.game_id for row in reader.sequences('validation', order_seed=7)] == ['game-99']
    with pytest.raises(ValueError, match='integer'):
        next(reader.sequences('train', order_seed=1.5))  # type: ignore[arg-type]
    reader.close()


def test_parallel_loader_yields_serial_sequences_in_serial_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.prepared as prepared_module

    def distinct(index: int) -> ActorSequence:
        sequence = renamed(sample(0), f'game-{index}')
        records = [list(record) for record in sequence.records]
        records[sequence.events[0]].append(('/fixture_game', f'str:{index}', 0.0))
        return replace(sequence, records=tuple(records))

    rows = [distinct(index) for index in range(20)]
    rows[4] = replace(rows[4], predictions=(replace(rows[4].predictions[0], loss_eligible=False), *rows[4].predictions[1:]))
    connection, receipt = prepare(tmp_path, [*rows, renamed(rows[2], 'duplicate-of-2'), replace(distinct(99), split='validation')])
    connection.close()
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': LEGACY_INPUT_CONTRACT, 'parser_hashes': binding_hashes(), 'membership': receipt})
    # A window of one in-flight task per worker forces interleaved submit/consume.
    monkeypatch.setattr(prepared_module, 'LOADER_WINDOW_PER_WORKER', 1)
    reader = PreparedCorpus(tmp_path)
    for seed in (None, 7):
        assert list(reader.sequences('train', order_seed=seed, workers=2)) == list(reader.sequences('train', order_seed=seed))
    assert list(reader.sequences('validation', workers=3)) == list(reader.sequences('validation'))
    partial_read = reader.sequences('train', order_seed=7, workers=2)
    assert next(partial_read) == next(reader.sequences('train', order_seed=7))
    partial_read.close()
    for workers in (-1, 33, 1.0):
        with pytest.raises(ValueError, match='workers'):
            next(reader.sequences('train', workers=workers))  # type: ignore[arg-type]
    reader.close()


def test_duplicate_actor_histories_coalesce_before_training_preflight(tmp_path: Path) -> None:
    from deepmew_ex.training.coverage import prepare_expected
    sequence = sample(0)
    masked = replace(sequence, predictions=tuple(replace(row, row_id='masked:' + row.row_id, loss_eligible=False) for row in sequence.predictions))
    connection, receipt = prepare(tmp_path, [masked, sequence, renamed(sequence, 'duplicate')])
    connection.close()
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': LEGACY_INPUT_CONTRACT, 'parser_hashes': binding_hashes(), 'membership': receipt})
    reader = PreparedCorpus(tmp_path)
    rows = list(reader.sequences('train'))
    assert len(rows) == 1 and rows[0].records == sequence.records
    assert all(row.loss_eligible for row in rows[0].predictions)
    ledger = sqlite3.connect(':memory:')
    ledger.execute('CREATE TABLE expected(epoch,phase,id,game,family,update_id)')
    assert prepare_expected(ledger, rows, 2, lambda: None)['expected_presentations'] == 2 * len(sequence.predictions)
    reader.close()
    ledger.close()


def test_conflicting_mask_on_same_source_coordinate_is_conservative(tmp_path: Path) -> None:
    sequence = sample(0)
    masked = replace(sequence, predictions=tuple(replace(row, loss_eligible=False) for row in sequence.predictions))
    connection, receipt = prepare(tmp_path, [masked, sequence])
    assert receipt['reasons'] == {'source_mask': 2 * len(sequence.predictions)}
    connection.close()


@pytest.mark.parametrize('input_contract', [None, 'unknown'])
def test_prepared_reader_requires_explicit_supported_input_contract(tmp_path: Path, input_contract: str | None) -> None:
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'parser_hashes': binding_hashes(), 'input_contract': input_contract})
    with pytest.raises(ValueError, match='input contract'):
        PreparedCorpus(tmp_path)


def test_query_contract_rejects_mislabeled_legacy_records(tmp_path: Path) -> None:
    connection, receipt = prepare(tmp_path, [sample(0)])
    connection.close()
    RunStore(tmp_path).seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'parser_hashes': binding_hashes(), 'input_contract': QUERY_INPUT_CONTRACT, 'membership': receipt})
    reader = PreparedCorpus(tmp_path)
    try:
        with pytest.raises(ValueError, match='features do not match'):
            list(reader.sequences('train'))
    finally:
        reader.close()
