"""Complete factory archives recovered under an explicit lineage split mapping."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path, PureWindowsPath

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.factory_source import SourceStreamAudit, factory_games, factory_rows, recover_factory_source_game
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.registry.records import validate_hash
from deepmew_ex.registry.store import canonical, digest, file_hash


def factory_manifest_shards(root: Path, manifest_sha256: str, check: Callable[[], None]) -> tuple[ReferenceShard, ...]:
    """Read the original factory manifest's hashes and verified summary counts.

    This historical schema has no newer shard_validation list. Decision bodies
    remain unadmitted until prepare_factory_shard verifies their exact bytes.
    """
    validate_hash(manifest_sha256)
    manifest_path = root / 'manifest.json'
    check()
    if file_hash(manifest_path) != manifest_sha256:
        raise ValueError('factory manifest identity changed')
    manifest = orjson.loads(manifest_path.read_bytes())
    if manifest.get('schema_version') != 'deepmew.refgames_factory.manifest.v0' or manifest.get('status') != 'complete':
        raise ValueError('unsupported or incomplete original factory manifest')
    workers = manifest.get('workers')
    if type(workers) is not int or workers < 1:
        raise ValueError('invalid factory manifest worker count')
    summary = manifest['summary']
    hashes: dict[str, str] = {}
    for location, value in summary['shard_hashes'].items():
        name = PureWindowsPath(location).name
        if name in hashes or not isinstance(value, str):
            raise ValueError('ambiguous factory manifest shard identity')
        hashes[name] = value.removeprefix('sha256:')
        validate_hash(hashes[name])
    expected = {f'{kind}_shard_{ordinal:03d}.jsonl.gz' for kind in ('games', 'decisions') for ordinal in range(workers)}
    if set(hashes) != expected:
        raise ValueError('factory manifest shard population differs from workers')
    shards = []
    native_ids: set[str] = set()
    coordinates: set[tuple[int, int]] = set()
    for ordinal in range(workers):
        check()
        summaries = root / 'shards' / f'games_shard_{ordinal:03d}.jsonl.gz'
        decisions = summaries.with_name(f'decisions_shard_{ordinal:03d}.jsonl.gz')
        if file_hash(summaries) != hashes[summaries.name]:
            raise ValueError('factory summary identity changed')
        games, requests = 0, 0
        for _, row in factory_rows(summaries, SourceStreamAudit(), check):
            native, count = row.get('game_id'), row.get('decisions')
            key = row.get('seed'), row.get('game_index')
            if not isinstance(native, str) or not native or native in native_ids or key in coordinates or any(type(value) is not int for value in key) or type(count) is not int or count < 0:
                raise ValueError('invalid or duplicate factory summary identity/count')
            native_ids.add(native)
            coordinates.add(key)
            games += 1
            requests += count
        if file_hash(summaries) != hashes[summaries.name]:
            raise ValueError('factory summary changed during inspection')
        shards.append(ReferenceShard(decisions, hashes[decisions.name], summaries, hashes[summaries.name], requests, games))
    games, requests = sum(row.games for row in shards), sum(row.decision_rows for row in shards)
    if any(type(value) is not int or value != games for value in (manifest.get('target_games'), summary.get('target_games'), summary.get('completed_games'), summary.get('game_rows'))) or type(summary.get('decision_rows')) is not int or summary['decision_rows'] != requests:
        raise ValueError('factory manifest population disagrees with verified summaries')
    if file_hash(manifest_path) != manifest_sha256:
        raise ValueError('factory manifest changed during inspection')
    return tuple(shards)


def prepare_factory_shard(shard: ReferenceShard, lineage_splits: Mapping[tuple[int, int], str], lineage_sha256: str, output: Path, check: Callable[[], None], progress: Callable[[dict[str, object]], None], allow_truncated: bool = False) -> dict[str, object]:
    """A caller must supply approved lineage assignments, not inferred new splits.

    Interrupted-container recovery is explicit and emits only complete games.
    Every incomplete game and byte tail remains in the completion audit.
    """
    validate_hash(lineage_sha256)
    if type(allow_truncated) is not bool:
        raise ValueError('factory truncated-container mode must be explicit boolean')
    if output.resolve() in {shard.decisions.resolve(), shard.summaries.resolve()}:
        raise ValueError('factory candidate output would overwrite an input')
    if any(len(key) != 2 or any(type(value) is not int for value in key) or split not in SPLIT_PRIORITY for key, split in lineage_splits.items()):
        raise ValueError('invalid factory lineage split mapping')
    for path, expected in ((shard.decisions, shard.decisions_sha256), (shard.summaries, shard.summaries_sha256)):
        check()
        if file_hash(path) != expected:
            raise ValueError('factory recovery source hash changed')
    summaries = {}
    summary_audit, decision_audit = SourceStreamAudit(), SourceStreamAudit()
    for _, summary in factory_rows(shard.summaries, summary_audit, check, allow_truncated):
        native = summary.get('game_id')
        key = summary.get('seed'), summary.get('game_index')
        if not isinstance(native, str) or native in summaries or any(type(value) is not int for value in key) or key not in lineage_splits:
            raise ValueError('factory summary has duplicate identity or unresolved lineage')
        summaries[native] = summary
    if len(summaries) != shard.games or sum(summary['decisions'] for summary in summaries.values()) != shard.decision_rows:
        raise ValueError('factory summary population differs from expected archive')
    root = Path(__file__).parents[1]
    modules = ('learning/factory_cache.py', 'learning/factory_source.py', 'learning/recovery_cache.py', 'learning/recovery.py', 'learning/actor_events.py', 'learning/queries.py', 'learning/trajectory.py', 'learning/semantic.py', 'learning/admission.py', 'corpus/reader.py', 'corpus/records.py', 'conversion.py', 'schema.py', 'contract.py')
    identity = {'shard': {**asdict(shard), 'decisions': str(shard.decisions), 'summaries': str(shard.summaries)}, 'lineage_sha256': lineage_sha256, 'split_mapping_sha256': digest(canonical([[*key, split] for key, split in sorted(lineage_splits.items())])), 'parser': binding_hashes(), 'code': {name: file_hash(root / name) for name in modules}, 'allow_truncated_container': allow_truncated, 'version': 2}
    connection = sqlite3.connect(output)
    encoder = zstandard.ZstdCompressor(level=3)
    try:
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS metadata(identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS games(game TEXT PRIMARY KEY,native TEXT UNIQUE NOT NULL,first_line INTEGER NOT NULL,requests INTEGER NOT NULL,split TEXT NOT NULL,sequence_identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS sequences(game TEXT NOT NULL,actor INTEGER NOT NULL,events INTEGER NOT NULL,records INTEGER NOT NULL,eligible INTEGER NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(game,actor)); CREATE TABLE IF NOT EXISTS completion(receipt TEXT NOT NULL);')
        encoded = canonical(identity).decode()
        previous = connection.execute('SELECT identity FROM metadata').fetchone()
        if previous is None:
            connection.execute('INSERT INTO metadata VALUES (?)', (encoded,))
            connection.commit()
        elif previous != (encoded,):
            raise ValueError('factory candidate source/code/lineage identity changed')
        seen: set[str] = set()
        incomplete: list[dict[str, object]] = []
        requests = 0
        for first, rows in factory_games(shard.decisions, decision_audit, check, allow_truncated):
            check()
            native = rows[0]['game_id']
            if native in seen:
                raise ValueError('factory game repeated')
            seen.add(native)
            if native not in summaries or len(rows) != summaries[native]['decisions']:
                if not allow_truncated:
                    raise ValueError('factory source game is incomplete or missing terminal summary')
                incomplete.append({'native_game_id': native, 'first_line': first, 'observed_rows': len(rows), 'summary_decisions': None if native not in summaries else summaries[native]['decisions'], 'reason': 'missing_terminal_summary' if native not in summaries else 'incomplete_recorded_requests'})
                continue
            summary = summaries[native]
            split = lineage_splits[summary['seed'], summary['game_index']]
            requests += len(rows)
            previous = connection.execute('SELECT game,first_line,requests,split,sequence_identity FROM games WHERE native=?', (native,)).fetchone()
            if previous is not None:
                members = connection.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=? ORDER BY actor', (previous[0],)).fetchall()
                actual = canonical([(*row[:-1], digest(row[-1])) for row in members]).decode()
                if previous[1:4] != (first, len(rows), split) or actual != previous[4]:
                    raise ValueError('factory candidate committed membership changed')
                continue
            recovered = recover_factory_source_game(rows, summary, shard.decisions, shard.decisions_sha256, first, split, (True,) * len(rows))
            with connection:
                members = []
                for sequence in recovered.sequences:
                    member = (sequence.actor, len(sequence.events), len(sequence.records), sum(row.loss_eligible for row in sequence.predictions), encoder.compress(orjson.dumps(asdict(sequence))))
                    connection.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?)', (recovered.game_id, *member))
                    members.append((*member[:-1], digest(member[-1])))
                connection.execute('INSERT INTO games VALUES (?,?,?,?,?,?)', (recovered.game_id, native, first, len(rows), split, canonical(members).decode()))
            if len(seen) % 100 == 0:
                progress({'games': len(seen), 'requests': requests, 'training_admitted': False})
        missing = sorted(set(summaries) - seen)
        complete_games = len(seen) - len(incomplete)
        if (not allow_truncated and (requests != shard.decision_rows or seen != set(summaries))) or connection.execute('SELECT COUNT(*) FROM games').fetchone()[0] != complete_games:
            raise ValueError('factory source population mismatch')
        if connection.execute('SELECT COUNT(*) FROM sequences LEFT JOIN games USING(game) WHERE games.game IS NULL').fetchone()[0] or connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('factory candidate database integrity failure')
        for path, expected in ((shard.decisions, shard.decisions_sha256), (shard.summaries, shard.summaries_sha256)):
            check()
            if file_hash(path) != expected:
                raise ValueError('factory source changed during recovery')
        receipt = {'identity': identity, 'games': complete_games, 'requests': requests, 'sequences': connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0], 'eligible_before_global_admission': connection.execute('SELECT COALESCE(SUM(eligible),0) FROM sequences').fetchone()[0], 'training_admitted': False, 'coverage': 'complete recorded request/action log with source-qualified game identity; unrecorded engine microsteps remain unknown', 'source_masks': 'new uncatalogued coordinates only; cross-format historical masks and reserved-content checks still required at global admission', 'source_streams': {'summaries': asdict(summary_audit), 'decisions': asdict(decision_audit)}, 'incomplete_games': incomplete, 'summary_games_without_recorded_rows': missing}
        encoded = canonical(receipt).decode()
        previous = connection.execute('SELECT receipt FROM completion').fetchone()
        if previous is None:
            connection.execute('INSERT INTO completion VALUES (?)', (encoded,))
            connection.commit()
        elif previous != (encoded,):
            raise ValueError('factory candidate completion changed')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
    return {**receipt, 'cache_sha256': file_hash(output)}
