"""Rejoin R15 teacher games split by the generator's row-count shard rotation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.recorded_actor import recover_recorded_actor_game
from deepmew_ex.datasets.v2.learning.recorded_cache import LockedReferenceIndex, recorded_groups
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.records import validate_hash
from deepmew_ex.registry.store import canonical, digest, file_hash


@dataclass(frozen=True)
class RecordedShard:
    path: Path
    sha256: str


def generation_groups(sources: tuple[RecordedShard, ...], index: sqlite3.Connection, staging_root: Path, check: Callable[[], None]) -> Iterator[tuple[list[Payload], list[Payload], list[Payload]]]:
    """Join adjacent native identities only; never invent missing coordinates."""
    frames: list[Payload] = []
    rows: list[Payload] = []
    segments: list[Payload] = []
    prior: tuple[object, ...] | None = None
    for source in sources:
        for first, fragment_frames, fragment_rows in recorded_groups(source.path, index, staging_root, check):
            row = fragment_rows[0]
            key = tuple(row.get(name) for name in ('game_id', 'game_index', 'seed', 'seat'))
            if prior is not None and key != prior:
                yield frames, rows, segments
                frames, rows, segments = [], [], []
            if any(frame['source_family'] != 'r15_hard' or frame['actor_order'] != position or frame['episode_id'] != fragment_frames[0]['episode_id'] or frame['state_before']['source_sha256'] != source.sha256 for position, frame in enumerate(fragment_frames)):
                raise ValueError('R15 fragment canonical identity or local actor order changed')
            if any(tuple(value.get(name) for name in ('game_id', 'game_index', 'seed', 'seat')) != key for value in fragment_rows):
                raise ValueError('mixed R15 native fragment identity')
            segments.append({'source_path': str(source.path), 'source_sha256': source.sha256, 'first_line': first, 'rows': len(fragment_rows), 'canonical_episode': fragment_frames[0]['episode_id'], 'original_frames_sha256': digest(canonical(fragment_frames)), 'global_orders': [frame['global_order'] for frame in fragment_frames], 'actor_orders': [frame['actor_order'] for frame in fragment_frames], 'frame_ids': [frame['frame_id'] for frame in fragment_frames]})
            frames.extend(fragment_frames)
            rows.extend(fragment_rows)
            prior = key
    if rows:
        yield frames, rows, segments


def prepare_recorded_generation(sources: tuple[RecordedShard, ...], manifest_path: Path, manifest_sha256: str, index_path: Path, index_sha256: str, output: Path, check: Callable[[], None], progress: Callable[[dict[str, object]], None], locked_index: LockedReferenceIndex | None = None) -> dict[str, object]:
    """Preserve original canonical frames; derive only joined episode/actor order.

    This reconstructs recorded teacher actions across physical file boundaries.
    Opponent and zero-option prompts remain unproven and cannot enter fitting.
    """
    for sha in (manifest_sha256, index_sha256, *(source.sha256 for source in sources)):
        validate_hash(sha)
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != manifest_sha256:
        raise ValueError('R15 generation manifest changed')
    manifest = orjson.loads(manifest_bytes)
    if manifest.get('schema_version') != 'deepmew.r15.hard_refgames.v0' or any(type(manifest.get(key)) is not int or manifest[key] < 0 for key in ('games_played', 'rows', 'base_seed')):
        raise ValueError('unsupported R15 generation contract')
    expected_paths = tuple(manifest_path.parent / 'shards' / f'decisions_shard_{ordinal:03d}.jsonl.gz' for ordinal in range(len(sources)))
    if not sources or tuple(source.path.resolve() for source in sources) != tuple(path.resolve() for path in expected_paths) or {path.resolve() for path in expected_paths} != {path.resolve() for path in (manifest_path.parent / 'shards').glob('decisions_shard_*.jsonl.gz')}:
        raise ValueError('R15 generation requires every ordered physical shard')
    if output.resolve() in {manifest_path.resolve(), index_path.resolve(), *(source.path.resolve() for source in sources)}:
        raise ValueError('R15 output would overwrite an input')
    for source in sources:
        check()
        if file_hash(source.path) != source.sha256:
            raise ValueError('R15 source identity changed')
    if locked_index is not None:
        locked_index.verify(index_path, index_sha256)
    elif file_hash(index_path) != index_sha256:
        raise ValueError('R15 canonical index changed')
    root = Path(__file__).parents[1]
    modules = ('learning/recorded_generation.py', 'learning/recorded_cache.py', 'learning/recorded_actor.py', 'learning/actor_events.py', 'learning/queries.py', 'learning/trajectory.py', 'learning/semantic.py', 'learning/admission.py', 'corpus/reader.py', 'conversion.py', 'schema.py', 'contract.py')
    identity = {'version': 1, 'generation_manifest': str(manifest_path), 'generation_manifest_sha256': manifest_sha256, 'sources': [{'path': str(source.path), 'sha256': source.sha256} for source in sources], 'reference_index_sha256': index_sha256, 'parser': binding_hashes(), 'code': {name: file_hash(root / name) for name in modules}, 'contract': 'recorded_teacher_generation_cross_shard_v1'}
    generation_sha = digest(canonical({'manifest': manifest_sha256, 'sources': identity['sources']}))
    index = sqlite3.connect(index_path.resolve().as_uri() + ('?mode=ro&immutable=1' if locked_index is not None else '?mode=ro'), uri=True)
    connection = sqlite3.connect(output)
    try:
        if any(index.execute('SELECT sha FROM sources WHERE path=?', (str(source.path),)).fetchone() != (source.sha256,) for source in sources):
            raise ValueError('R15 source differs from frozen canonical inventory')
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS metadata(identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS games(game TEXT PRIMARY KEY,split TEXT NOT NULL,requests INTEGER NOT NULL,assembly TEXT NOT NULL,member_sha TEXT NOT NULL); CREATE TABLE IF NOT EXISTS sequences(game TEXT NOT NULL,actor INTEGER NOT NULL,events INTEGER NOT NULL,records INTEGER NOT NULL,eligible INTEGER NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(game,actor)); CREATE TABLE IF NOT EXISTS completion(receipt TEXT NOT NULL);')
        encoded_identity = canonical(identity).decode()
        previous = connection.execute('SELECT identity FROM metadata').fetchall()
        if not previous:
            with connection:
                connection.execute('INSERT INTO metadata VALUES (?)', (encoded_identity,))
        elif previous != [(encoded_identity,)]:
            raise ValueError('R15 recovery input or producer identity changed')
        encoder = zstandard.ZstdCompressor(level=3)
        games, requests, joined = 0, 0, 0
        for frames, rows, segments in generation_groups(sources, index, output.parent, check):
            check()
            first = rows[0]
            if first['game_index'] != games or first['seed'] != manifest['base_seed'] + games or first['game_id'] != f'r15-hard-ref-{games:06d}':
                raise ValueError('R15 generation game order, seed or native identity changed')
            if [row['decision_index'] for row in rows] != list(range(len(rows))) or [frame['global_order'] for frame in frames] != list(range(len(rows))):
                raise ValueError('R15 joined history has missing, repeated or reordered original decisions')
            game = f'recorded-generation:{generation_sha}:{first["game_id"]}'
            assembly = canonical({'derived_episode': game, 'mapping': 'actor_order := unchanged original global_order; source/frame identities and masks unchanged', 'segments': segments}).decode()
            prior = connection.execute('SELECT requests,assembly,member_sha,split FROM games WHERE game=?', (game,)).fetchone()
            if prior is not None:
                members = connection.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=?', (game,)).fetchall()
                actual = digest(canonical([*members[0][:-1], digest(members[0][-1]), assembly, prior[3]])) if len(members) == 1 else None
                if prior[:2] != (len(rows), assembly) or prior[2] != actual:
                    raise ValueError('R15 resume membership or preserved assembly changed')
            else:
                derived = [{**frame, 'episode_id': game, 'actor_order': frame['global_order']} for frame in frames]
                recovered = recover_recorded_actor_game(derived, rows)
                sequence = recovered.sequence
                encoded = encoder.compress(orjson.dumps(asdict(sequence)))
                member = (sequence.actor, len(sequence.events), len(sequence.records), sum(row.loss_eligible for row in sequence.predictions), encoded)
                member_sha = digest(canonical([*member[:-1], digest(encoded), assembly, sequence.split]))
                with connection:
                    connection.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?)', (game, *member))
                    connection.execute('INSERT INTO games VALUES (?,?,?,?,?)', (game, sequence.split, len(rows), assembly, member_sha))
            games += 1
            requests += len(rows)
            joined += len(segments) > 1
            if games % 100 == 0:
                progress({'games': games, 'requests': requests, 'joined_boundary_games': joined, 'training_admitted': False})
        if (games, requests) != (manifest['games_played'], manifest['rows']) or connection.execute('SELECT COUNT(*) FROM games').fetchone()[0] != games or connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] != games:
            raise ValueError('R15 complete generation membership differs from its manifest')
        check()
        if file_hash(manifest_path) != manifest_sha256 or any(file_hash(source.path) != source.sha256 for source in sources) or connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('R15 source mutation or cache integrity failure')
        if locked_index is not None:
            locked_index.verify(index_path, index_sha256)
        elif file_hash(index_path) != index_sha256:
            raise ValueError('R15 canonical index changed during recovery')
        receipt = {'identity': identity, 'games': games, 'requests': requests, 'sequences': games, 'joined_boundary_games': joined, 'eligible_before_global_admission': connection.execute('SELECT SUM(eligible) FROM sequences').fetchone()[0], 'coverage': 'complete recorded teacher generation; opponent and zero-option prompts remain unproven', 'training_admitted': False}
        encoded_receipt = canonical(receipt).decode()
        with connection:
            prior = connection.execute('SELECT receipt FROM completion').fetchall()
            if not prior:
                connection.execute('INSERT INTO completion VALUES (?)', (encoded_receipt,))
            elif prior != [(encoded_receipt,)]:
                raise ValueError('R15 completion receipt changed')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
        index.close()
    return {**receipt, 'cache_sha256': file_hash(output)}
