"""Resumable source-bound native replay candidates, including explicit rejections."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import zstandard

from deepmew_ex.datasets.v2.conversion import convert_episode, game_split
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.legacy_contract import recorded_legacy_contract
from deepmew_ex.datasets.v2.learning.raw_recovery import inherit_raw_masks, recover_raw_histories
from deepmew_ex.registry.store import canonical, digest, file_hash


@dataclass(frozen=True)
class RawSource:
    path: Path
    sha256: str
    native_game_id: str
    module_version: str
    split: str


def raw_reference_paths(corpus: Path, index: sqlite3.Connection, check: Callable[[], None]) -> tuple[dict[str, tuple[str, ...]], tuple[dict[str, str], ...]]:
    """Find every historical Kaggle table through its bound catalog owner.

    The caller verifies the immutable reference-index file once. Source-leading
    queries use its existing covering index, avoiding a full 23 GB table scan
    for each native replay. Filenames are never used to infer game identity.
    """
    manifest = orjson.loads((corpus / 'manifest.json').read_bytes())
    if file_hash(corpus / 'inventory.json') != manifest['files_sha256']['inventory.json']:
        raise ValueError('raw reference inventory changed')
    inventory = orjson.loads((corpus / 'inventory.json').read_bytes())
    catalog_path = Path(inventory['catalog']['path'])
    if file_hash(catalog_path) != inventory['catalog']['sha256']:
        raise ValueError('raw reference catalog changed')
    catalog = orjson.loads(catalog_path.read_bytes())
    sources: dict[str, str] = {}
    for unit in catalog['included_units']:
        if unit['source_id'] != 'kaggle_ingest':
            continue
        check()
        path = (catalog_path.parent / unit['manifest_path']).resolve()
        if not path.is_relative_to(catalog_path.parent.resolve()) or file_hash(path) != unit['manifest_sha256']:
            raise ValueError('raw reference canonical unit identity changed')
        metadata = orjson.loads(path.read_bytes())
        if metadata['source_id'] != 'kaggle_ingest' or metadata['status'] != 'completed':
            raise ValueError('raw reference canonical unit is not completed Kaggle evidence')
        for source in metadata['source_fingerprint']:
            location = source['path']
            if location in sources and sources[location] != source['sha256']:
                raise ValueError('raw reference source identities conflict')
            sources[location] = source['sha256']
    games: dict[str, list[str]] = {}
    unrepresented: list[dict[str, str]] = []
    for path, sha in sorted(sources.items()):
        check()
        binding = index.execute('SELECT sha FROM sources WHERE path=?', (path,)).fetchone()
        if binding is None:
            # Historical whole-game quarantines have no indexed references.
            # Preserve this unresolved source explicitly for the coverage audit;
            # absence from the index is not proof of absent history or masks.
            unrepresented.append({'path': path, 'sha256': sha})
            continue
        if binding != (sha,):
            raise ValueError('raw reference index differs from canonical source binding')
        for (native,) in index.execute('SELECT DISTINCT game FROM refs WHERE source=?', (path,)):
            games.setdefault(native, []).append(path)
    return {native: tuple(paths) for native, paths in games.items()}, tuple(unrepresented)


def freeze_raw_sources(corpus: Path, check: Callable[[], None], progress: Callable[[dict[str, object]], None]) -> tuple[tuple[RawSource, ...], dict[str, object]]:
    """Reuse the frozen port's assignments and resolve quarantined source metadata.

    This is inventory preparation only. Every original byte hash is verified
    again by the recovery worker before conversion; no historical rejection is
    silently promoted and no target mask is inferred from a game-level split.
    """
    manifest_path = corpus / 'manifest.json'
    manifest_sha = file_hash(manifest_path)
    manifest = orjson.loads(manifest_path.read_bytes())
    if manifest.get('status') != 'completed' or manifest.get('kind') != 'v2_full_corpus_port':
        raise ValueError('raw source plan requires a completed historical port')
    for name in ('inventory.json', 'port.json'):
        if file_hash(corpus / name) != manifest['files_sha256'][name]:
            raise ValueError('raw source plan historical metadata changed')
    inventory = orjson.loads((corpus / 'inventory.json').read_bytes())
    report = orjson.loads((corpus / 'port.json').read_bytes())
    if report['raw_native_game_conflicts']:
        raise ValueError('raw source native identity conflicts require investigation')
    results = {row['source']['sha256']: row for row in report['results'] if row['kind'] == 'raw'}
    available = [row for row in inventory['replays'] if row['status'] == 'available' and not row.get('alias_of')]
    if len({row['sha256'] for row in available}) != len(available) or set(results) != {row['sha256'] for row in available}:
        raise ValueError('raw source plan population differs from original port')
    sources = []
    decoder = zstandard.ZstdDecompressor()
    for ordinal, source in enumerate(available, 1):
        check()
        result = results[source['sha256']]
        if result['source'] != source:
            raise ValueError('raw source plan path or identity differs from inventory')
        if result['status'] == 'admitted_actor_visible_episode':
            child = Path(result['path']).resolve()
            if not child.is_relative_to(corpus.resolve()) or file_hash(child / 'manifest.json') != result['manifest_sha256']:
                raise ValueError('raw source plan child receipt mismatch')
            child_manifest = orjson.loads((child / 'manifest.json').read_bytes())
            episode_path = child / 'episode.json.zst'
            if file_hash(episode_path) != child_manifest['files_sha256']['episode.json.zst']:
                raise ValueError('raw source plan preserved episode changed')
            episode = orjson.loads(decoder.decompress(episode_path.read_bytes()))
            native, module = episode['source_native_game_id'], episode['module_version']
            if episode['source_sha256'] != source['sha256'] or native != result['native_game_id']:
                raise ValueError('raw source plan episode lineage differs')
        elif result['status'] == 'quarantined':
            raw = Path(source['path']).read_bytes()
            if digest(raw) != source['sha256']:
                raise ValueError('raw source plan quarantined source changed')
            replay = orjson.loads(raw)
            native, module = replay.get('id'), replay.get('module_version')
            del raw, replay
        else:
            raise ValueError('unknown raw source historical classification')
        if not isinstance(native, str) or not native or not isinstance(module, str) or not module:
            raise ValueError('raw source metadata cannot be identified')
        # Exact frozen fallback from corpus/workers.py:raw_episode, not a new
        # partition of the recovered population. Existing reservations win.
        number = int(hashlib.sha256(native.encode()).hexdigest(), 16) % 10
        split = inventory['raw_splits'].get(native, 'train' if number < 8 else 'validation' if number == 8 else 'holdout')
        if split not in SPLIT_PRIORITY or result.get('effective_split', split) != split:
            raise ValueError('raw source plan would change a historical split')
        sources.append(RawSource(Path(source['path']), source['sha256'], native, module, split))
        if ordinal % 100 == 0:
            progress({'metadata_sources': ordinal, 'total': len(available), 'training_admitted': False})
    if len({source.native_game_id for source in sources}) != len(sources) or file_hash(manifest_path) != manifest_sha:
        raise ValueError('raw source plan ambiguous native identity or manifest drift')
    receipt = {'original_corpus_manifest_sha256': manifest_sha, 'sources': len(sources), 'path_aliases': sum(bool(row.get('alias_of')) for row in inventory['replays']), 'unavailable': [row for row in inventory['replays'] if row['status'] != 'available'], 'historical_quarantine_reasons': report['raw_quarantine_reasons'], 'training_admitted': False, 'scope': 'source metadata and original game splits only; full recovery, historical mask joins and global admission still required'}
    return tuple(sources), receipt


def prepare_raw_sources(sources: tuple[RawSource, ...], original_manifest: Path, manifest_sha256: str, output: Path, check: Callable[[], None], progress: Callable[[dict[str, object]], None], reference_index: Path | None = None, reference_index_sha256: str | None = None) -> dict[str, object]:
    if not sources or len({source.sha256 for source in sources}) != len(sources) or any(source.split not in SPLIT_PRIORITY for source in sources):
        raise ValueError('raw recovery requires unique sources and explicit original splits')
    if output.resolve() in {original_manifest.resolve(), *(source.path.resolve() for source in sources)}:
        raise ValueError('raw candidate output would overwrite an input')
    check()
    if file_hash(original_manifest) != manifest_sha256:
        raise ValueError('raw recovery original manifest changed')
    if (reference_index is None) != (reference_index_sha256 is None):
        raise ValueError('raw historical mask index requires its exact identity')
    if reference_index is not None and (output.resolve() == reference_index.resolve() or file_hash(reference_index) != reference_index_sha256):
        raise ValueError('raw historical mask index identity or output collision')
    reference_paths: dict[str, tuple[str, ...]] = {}
    unrepresented: tuple[dict[str, str], ...] = ()
    index = None if reference_index is None else sqlite3.connect(reference_index.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if index is not None:
            reference_paths, unrepresented = raw_reference_paths(original_manifest.parent, index, check)
    except BaseException:
        if index is not None:
            index.close()
        raise
    root = Path(__file__).parents[1]
    modules = ('learning/raw_cache.py', 'learning/raw_recovery.py', 'learning/legacy_contract.py', 'learning/history.py', 'learning/queries.py', 'learning/actor_events.py', 'learning/trajectory.py', 'learning/semantic.py', 'learning/admission.py', 'conversion.py', 'schema.py', 'reader.py', 'contract.py')
    contract = recorded_legacy_contract()
    identity = {'original_manifest_sha256': manifest_sha256, 'sources': [{**asdict(source), 'path': str(source.path.resolve())} for source in sources], 'code': {name: file_hash(root / name) for name in modules}, 'parser': contract.hashes, 'reference_index_sha256': reference_index_sha256, 'historical_reference_map_sha256': digest(canonical([reference_paths, unrepresented])), 'version': 2}
    connection = sqlite3.connect(output)
    encoder = zstandard.ZstdCompressor(level=3)
    try:
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS metadata(identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS sources(sha TEXT PRIMARY KEY,game TEXT NOT NULL,split TEXT NOT NULL,status TEXT NOT NULL,reason TEXT,requests INTEGER,episode BLOB,payload BLOB NOT NULL,mask_receipt TEXT NOT NULL,content_sha TEXT NOT NULL); CREATE TABLE IF NOT EXISTS completion(receipt TEXT NOT NULL);')
        encoded = canonical(identity).decode()
        previous = connection.execute('SELECT identity FROM metadata').fetchone()
        if previous is None:
            connection.execute('INSERT INTO metadata VALUES (?)', (encoded,))
            connection.commit()
        elif previous != (encoded,):
            raise ValueError('raw candidate preparation identity changed')
        for ordinal, source in enumerate(sources, 1):
            check()
            raw = source.path.read_bytes()
            if digest(raw) != source.sha256:
                raise ValueError(f'raw source identity changed: {source.path}')
            previous = connection.execute('SELECT game,split,status,reason,requests,episode,payload,mask_receipt,content_sha FROM sources WHERE sha=?', (source.sha256,)).fetchone()
            if previous is not None:
                member = [*previous[:5], None if previous[5] is None else digest(previous[5]), digest(previous[6]), previous[7]]
                if previous[:2] != (source.native_game_id, source.split) or digest(canonical(member)) != previous[8]:
                    raise ValueError('raw candidate committed membership changed')
                continue
            replay = json.loads(raw)
            if replay.get('id') != source.native_game_id or replay.get('module_version') != source.module_version:
                raise ValueError('raw source metadata differs from frozen identity')
            status, reason, requests, episode_bytes = 'candidate', None, None, None
            mask_receipt: dict[str, object] = {'status': 'not_reconciled' if index is None else 'no_candidate'}
            payload = encoder.compress(orjson.dumps([]))
            if contract.source_contract(replay) is None:
                status, reason = 'unsupported', 'source_contract_not_proven'
            else:
                metadata = {'family': f'kaggle_cabt_{source.module_version}', 'module_version': source.module_version, 'path': str(source.path), 'sha256': source.sha256, 'native_game_id': source.native_game_id, 'split': game_split(source.native_game_id), 'asset_id': source.path.name}
                try:
                    episode = convert_episode(raw, metadata, contract, max_steps=len(replay.get('steps', [])))
                    requests = len(episode.frames)
                    sequences = recover_raw_histories(episode, raw, source.split)
                except ValueError as error:
                    # This is an explicit source rejection, never authority to
                    # change its data or to claim the remaining corpus complete.
                    status, reason = 'rejected', str(error)
                else:
                    if index is not None:
                        paths = reference_paths.get(source.native_game_id, ())
                        decoder = zstandard.ZstdDecompressor()
                        references = [orjson.loads(decoder.decompress(row[0])) for path in paths for row in index.execute("SELECT payload FROM refs WHERE source=? AND game=? AND family='kaggle_ingest' ORDER BY id", (path, source.native_game_id))]
                        sequences, mask_receipt = inherit_raw_masks(episode, raw, sequences, references, {path: Path(path).read_bytes() for path in paths})
                        if mask_receipt['effective_split'] != source.split:
                            raise ValueError('raw historical reference conflicts with frozen effective split')
                        mask_receipt['status'] = 'applied'
                    episode_bytes = encoder.compress(episode.to_bytes())
                    payload = encoder.compress(orjson.dumps([asdict(sequence) for sequence in sequences]))
            mask_json = canonical(mask_receipt).decode()
            member = [source.native_game_id, source.split, status, reason, requests, None if episode_bytes is None else digest(episode_bytes), digest(payload), mask_json]
            with connection:
                connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)', (source.sha256, source.native_game_id, source.split, status, reason, requests, episode_bytes, payload, mask_json, digest(canonical(member))))
            if ordinal % 25 == 0:
                progress({'sources': ordinal, 'total': len(sources), 'training_admitted': False})
        if connection.execute('SELECT COUNT(*) FROM sources').fetchone()[0] != len(sources) or connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('raw recovery population or database integrity mismatch')
        check()
        if file_hash(original_manifest) != manifest_sha256:
            raise ValueError('raw recovery original manifest changed during preparation')
        receipt = {'identity': identity, 'source_counts': dict(connection.execute('SELECT status,COUNT(*) FROM sources GROUP BY status')), 'candidate_requests': connection.execute("SELECT COALESCE(SUM(requests),0) FROM sources WHERE status='candidate'").fetchone()[0], 'reasons': connection.execute("SELECT status,reason,COUNT(*) FROM sources WHERE status!='candidate' GROUP BY status,reason").fetchall(), 'historical_masks': {'status': 'applied' if index is not None else 'not_reconciled', 'unrepresented_sources': unrepresented}, 'training_admitted': False, 'scope': 'complete recorded native request/action histories; unrecorded engine microsteps not inferred; every supplied source retained as candidate, unsupported or rejected; historical masks and complete global admission are required before training'}
        encoded = canonical(receipt).decode()
        previous = connection.execute('SELECT receipt FROM completion').fetchone()
        if previous is None:
            connection.execute('INSERT INTO completion VALUES (?)', (encoded,))
            connection.commit()
        elif previous != (encoded,):
            raise ValueError('raw recovery completion identity changed')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
        if index is not None:
            index.close()
    return {**receipt, 'cache_sha256': file_hash(output)}
