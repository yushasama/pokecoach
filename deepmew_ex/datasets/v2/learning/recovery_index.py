"""Resumable, split-preserving index of every frozen canonical reference.

Unlike the legacy training index, masked targets remain available as temporal
context. Indexing is preparation only; it grants no source-history capability.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import orjson
import zstandard

from deepmew_ex.datasets.v2.corpus.reader import compressed_rows
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import canonical, digest, file_hash


def store_reference(connection: sqlite3.Connection, frame: Payload, compressor: zstandard.ZstdCompressor) -> None:
    reference = frame['state_before']
    split = frame['effective_split']
    row = (frame['frame_id'], reference['source_path'], reference['source_line'], frame['native_game_binding']['raw_game_id'], frame['global_order'], frame['actor_seat'], split, SPLIT_PRIORITY[split], frame['source_family'], compressor.compress(orjson.dumps(frame)))
    try:
        connection.execute('INSERT INTO refs VALUES (?,?,?,?,?,?,?,?,?,?)', row)
    except sqlite3.IntegrityError:
        previous = orjson.loads(zstandard.ZstdDecompressor().decompress(connection.execute('SELECT payload FROM refs WHERE id=?', (frame['frame_id'],)).fetchone()[0]))
        keys = ('native_game_binding', 'global_order', 'actor_seat', 'legal_actions', 'chosen_action')
        if any(previous[key] != frame[key] for key in keys) or previous['state_before']['content_sha256'] != reference['content_sha256']:
            raise ValueError('duplicate reference has conflicting source evidence')
        selected = frame if SPLIT_PRIORITY[split] > SPLIT_PRIORITY[previous['effective_split']] else previous
        selected['target_mask']['source_target_eligible'] = previous['target_mask']['source_target_eligible'] is True and frame['target_mask']['source_target_eligible'] is True
        connection.execute('UPDATE refs SET split=?,priority=?,payload=? WHERE id=?', (selected['effective_split'], SPLIT_PRIORITY[selected['effective_split']], compressor.compress(orjson.dumps(selected)), frame['frame_id']))


def build_reference_index(corpus: Path, output: Path, check_budget: Callable[[], None], progress: Callable[[dict[str, object]], None]) -> dict[str, object]:
    corpus, output = corpus.resolve(), output.resolve()
    manifest = orjson.loads((corpus / 'manifest.json').read_bytes())
    if manifest['status'] != 'completed' or manifest['kind'] != 'v2_full_corpus_port':
        raise ValueError('completed frozen corpus required')
    for name in ('inventory.json', 'port.json'):
        if file_hash(corpus / name) != manifest['files_sha256'][name]:
            raise ValueError('frozen corpus metadata changed')
    inventory = orjson.loads((corpus / 'inventory.json').read_bytes())
    report = orjson.loads((corpus / 'port.json').read_bytes())
    results = {row['part_id']: row for row in report['results'] if row['kind'] == 'canonical'}
    dependencies = (Path(__file__).with_name('admission.py'), Path(__file__).parents[1] / 'corpus/reader.py', Path(__file__).parents[3] / 'registry/store.py')
    identity = {'corpus_manifest_sha256': file_hash(corpus / 'manifest.json'), 'index_code_sha256': file_hash(Path(__file__)), 'dependencies_sha256': {path.name: file_hash(path) for path in dependencies}, 'split_priority': SPLIT_PRIORITY, 'version': 2}
    connection = sqlite3.connect(output)
    try:
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-65536; CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS refs (id TEXT PRIMARY KEY,source TEXT NOT NULL,line INTEGER NOT NULL,game TEXT NOT NULL,global_order INTEGER NOT NULL,actor INTEGER NOT NULL,split TEXT NOT NULL,priority INTEGER NOT NULL,family TEXT NOT NULL,payload BLOB NOT NULL); CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY,sha TEXT NOT NULL); CREATE TABLE IF NOT EXISTS parts (id TEXT PRIMARY KEY,sha TEXT NOT NULL,rows INTEGER NOT NULL);')
        previous = connection.execute('SELECT identity FROM metadata').fetchone()
        encoded_identity = canonical(identity).decode()
        if previous is None:
            connection.execute('INSERT INTO metadata VALUES (?)', (encoded_identity,))
            connection.commit()
        elif previous[0] != encoded_identity:
            raise ValueError('recovery index code or corpus identity changed')
        sources = dict(connection.execute('SELECT path,sha FROM sources'))
        completed = dict(connection.execute('SELECT id,sha FROM parts'))
        compressor = zstandard.ZstdCompressor(level=1)
        for ordinal, part in enumerate(inventory['parts'], 1):
            check_budget()
            path = Path(results[part['part_id']]['path']) / 'frames.jsonl.zst'
            relative = path.resolve().relative_to(corpus).as_posix()
            expected = manifest['files_sha256'][relative]
            if part['part_id'] in completed:
                if completed[part['part_id']] != expected:
                    raise ValueError('completed reference part identity changed')
                continue
            if file_hash(path) != expected:
                raise ValueError('canonical reference shard changed')
            seen = 0
            with connection:
                for frame in compressed_rows(path):
                    source = frame['state_before']
                    location, sha = source['source_path'], source['source_sha256']
                    if location in sources and sources[location] != sha:
                        raise ValueError('conflicting source file identity')
                    if location not in sources:
                        connection.execute('INSERT INTO sources VALUES (?,?)', (location, sha))
                        sources[location] = sha
                    store_reference(connection, frame, compressor)
                    seen += 1
                    if seen % 5000 == 0:
                        check_budget()
                if seen != part['rows']:
                    raise ValueError('canonical part row-count mismatch')
                connection.execute('INSERT INTO parts VALUES (?,?,?)', (part['part_id'], expected, seen))
            progress({'phase': 'index', 'parts': ordinal, 'total_parts': len(inventory['parts']), 'last_part_rows': seen})
        check_budget()
        connection.execute('CREATE INDEX IF NOT EXISTS source_rows ON refs(source,line)')
        connection.execute('CREATE INDEX IF NOT EXISTS game_rows ON refs(source,game,global_order)')
        connection.commit()
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('recovery index integrity failure')
        receipt = {'identity': identity, 'parts': connection.execute('SELECT COUNT(*) FROM parts').fetchone()[0], 'reference_presentations': connection.execute('SELECT SUM(rows) FROM parts').fetchone()[0], 'unique_references': connection.execute('SELECT COUNT(*) FROM refs').fetchone()[0], 'source_files': connection.execute('SELECT COUNT(*) FROM sources').fetchone()[0], 'by_family_split': connection.execute('SELECT family,split,COUNT(*) FROM refs GROUP BY family,split').fetchall(), 'training_admitted': False, 'masked_rows_retained_for_context': True}
        if receipt['reference_presentations'] != report['catalog_rows']:
            raise ValueError('index did not traverse the complete frozen catalog')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
    return {**receipt, 'index_sha256': file_hash(output)}
