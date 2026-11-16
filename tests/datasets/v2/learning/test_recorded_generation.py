from __future__ import annotations

import copy
import gzip
import sqlite3
from pathlib import Path

import orjson
import pytest
import zstandard

from deepmew_ex.datasets.v2.learning.prepared import CandidateCache, cached_sequences, publish_prepared
from deepmew_ex.datasets.v2.learning.recorded_generation import RecordedShard, prepare_recorded_generation
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import canonical, file_hash
from tests.datasets.v2.learning.test_prepared_publication import publication
from tests.datasets.v2.learning.test_recorded_actor import recorded
from tests.datasets.v2.learning.test_recovery import reference_game
from tests.datasets.v2.corpus.conftest import corpus_inventory

GenerationInput = tuple[tuple[RecordedShard, ...], Path, str, Path, str]


def generation(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path, defect: str | None = None, reserved: bool = False) -> GenerationInput:
    frames, rows = copy.deepcopy(recorded)
    root = tmp_path / 'generation'
    (root / 'shards').mkdir(parents=True)
    index = tmp_path / 'index.sqlite3'
    connection = sqlite3.connect(index)
    connection.executescript('CREATE TABLE sources(path TEXT,sha TEXT); CREATE TABLE refs(source TEXT,line INTEGER,payload BLOB);')
    sources = []
    for ordinal, (frame, row) in enumerate(zip(frames, rows, strict=True)):
        row.update(game_id='r15-hard-ref-000000', game_index=0)
        frame['native_game_binding'].update(raw_game_id=row['game_id'], canonical_native_game_id=row['game_id'], source_game_index=0)
        frame.update(source_family='r15_hard', episode_id=f'physical-shard-{ordinal}:game', actor_order=0, effective_split='holdout' if reserved and ordinal == 1 else 'train')
        if ordinal == 1:
            if defect == 'gap':
                row['decision_index'] = frame['global_order'] = 2
            elif defect == 'repeated':
                row['decision_index'] = frame['global_order'] = 0
            elif defect == 'seed':
                row['seed'] += 1
            elif defect == 'actor':
                row['seat'] = 1 - row['seat']
            elif defect == 'local_order':
                frame['actor_order'] = 1
            elif defect == 'mixed_frame_group':
                frame['source_family'] = 'different_generation'
            elif defect == 'outcome':
                row['outcome']['result'] = 1 - row['outcome']['result']
        source = root / 'shards' / f'decisions_shard_{ordinal:03d}.jsonl.gz'
        with gzip.open(source, 'wb') as handle:
            handle.write(orjson.dumps(row) + b'\n')
        sha = file_hash(source)
        frame['state_before'].update(source_path=str(source), source_sha256=sha, source_line=1)
        connection.execute('INSERT INTO sources VALUES (?,?)', (str(source), sha))
        connection.execute('INSERT INTO refs VALUES (?,?,?)', (str(source), 1, zstandard.ZstdCompressor().compress(orjson.dumps(frame))))
        sources.append(RecordedShard(source, sha))
    connection.commit()
    connection.close()
    manifest = root / 'manifest.json'
    manifest.write_bytes(canonical({'schema_version': 'deepmew.r15.hard_refgames.v0', 'games_played': 1, 'rows': 2, 'base_seed': rows[0]['seed']}))
    return tuple(sources), manifest, file_hash(manifest), index, file_hash(index)


def test_rejoins_physical_shards_preserving_original_frames_masks_and_restrictive_split(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path) -> None:
    inputs = generation(recorded, tmp_path, reserved=True)
    output = tmp_path / 'candidate.sqlite3'
    receipt = prepare_recorded_generation(*inputs, output, lambda: None, lambda row: None)
    assert (receipt['games'], receipt['requests'], receipt['joined_boundary_games']) == (1, 2, 1)
    assert not receipt['training_admitted']
    assert prepare_recorded_generation(*inputs, output, lambda: None, lambda row: None) == receipt
    sequences = list(cached_sequences(output, receipt['cache_sha256'], 'recorded_group', lambda: None))
    assert len(sequences) == 1 and sequences[0].split == 'holdout'
    assert [row.row_id for row in sequences[0].predictions] == [frame['frame_id'] for frame in recorded[0]]
    assert [row.loss_eligible for row in sequences[0].predictions] == [True, False]
    assert len(sequences[0].events) == 4
    with sqlite3.connect(output.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        assembly = orjson.loads(connection.execute('SELECT assembly FROM games').fetchone()[0])
        assert [segment['global_orders'] for segment in assembly['segments']] == [[0], [1]]
        assert [segment['actor_orders'] for segment in assembly['segments']] == [[0], [0]]
        assert [segment['canonical_episode'] for segment in assembly['segments']] == ['physical-shard-0:game', 'physical-shard-1:game']
    connection.close()


@pytest.mark.parametrize('defect', ['gap', 'repeated', 'seed', 'actor', 'local_order', 'mixed_frame_group', 'outcome'])
def test_cross_shard_repair_never_renumbers_missing_or_conflicting_source_history(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path, defect: str) -> None:
    inputs = generation(recorded, tmp_path, defect)
    output = tmp_path / 'rejected.sqlite3'
    with pytest.raises(ValueError):
        prepare_recorded_generation(*inputs, output, lambda: None, lambda row: None)
    with sqlite3.connect(output) as connection:
        assert connection.execute('SELECT COUNT(*) FROM completion').fetchone()[0] == 0


def test_omitted_physical_shard_fails_before_candidate_creation(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path) -> None:
    inputs = generation(recorded, tmp_path)
    output = tmp_path / 'missing.sqlite3'
    with pytest.raises(ValueError, match='every ordered physical shard'):
        prepare_recorded_generation(inputs[0][:1], *inputs[1:], output, lambda: None, lambda row: None)
    assert not output.exists()


def test_joined_teacher_generation_still_cannot_supply_full_history_training_loss(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path) -> None:
    inputs = generation(recorded, tmp_path)
    output = tmp_path / 'teacher.sqlite3'
    receipt = prepare_recorded_generation(*inputs, output, lambda: None, lambda row: None)
    cache = CandidateCache(output, receipt['cache_sha256'], 'recorded_group')
    store, inventory, sha = publication(tmp_path, (cache,))
    with pytest.raises(ValueError, match='no eligible training'):
        publish_prepared(store, (cache,), inventory, sha, lambda: None)
    assert not (store.path / 'manifest.json').exists()


def test_resume_checks_preserved_original_frame_mapping(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path) -> None:
    inputs = generation(recorded, tmp_path)
    output = tmp_path / 'candidate.sqlite3'
    prepare_recorded_generation(*inputs, output, lambda: None, lambda row: None)
    with sqlite3.connect(output) as connection:
        connection.execute("UPDATE games SET assembly='changed'")
    connection.close()
    with pytest.raises(ValueError, match='resume membership'):
        prepare_recorded_generation(*inputs, output, lambda: None, lambda row: None)
