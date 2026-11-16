"""Stream the frozen corpus's raw trajectories and resolved snapshot sources."""

from __future__ import annotations

import gzip
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import orjson
import zstandard

from deepmew_ex.datasets.v2.corpus.reader import compressed_rows, resolve_row
from deepmew_ex.datasets.v2.contract import binding_hashes, load_contract
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.semantic import actor_state, input_signature, leaves, relative_players
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, actor_sequences, structured_target
from deepmew_ex.datasets.v2.reader import validate_episode
from deepmew_ex.datasets.v2.schema import Episode, Payload
from deepmew_ex.registry.store import canonical, digest, file_hash


def index_reference(connection: sqlite3.Connection, frame: Payload, encoded: bytes) -> bool:
    """Duplicate source coordinates always inherit the most reserved split."""
    reference = frame['state_before']
    cursor = connection.execute('INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET split=excluded.split,family=excluded.family,priority=excluded.priority,payload=excluded.payload WHERE excluded.priority>refs.priority', (frame['frame_id'], reference['source_path'], reference['source_line'], frame['effective_split'], frame['source_family'], SPLIT_PRIORITY[frame['effective_split']], encoded))
    return bool(cursor.rowcount)


class FullCorpus:
    def __init__(self, root: Path, output: Path, progress: Callable[[str, dict[str, object]], None]) -> None:
        self.root, self.output, self.progress = root, output, progress
        self.manifest = orjson.loads((root / 'manifest.json').read_bytes())
        if self.manifest['status'] != 'completed' or self.manifest['kind'] != 'v2_full_corpus_port':
            raise ValueError('a completed V2 corpus is required')
        for name in ('port.json', 'inventory.json'):
            if file_hash(root / name) != self.manifest['files_sha256'][name]:
                raise ValueError('corpus metadata changed')
        self.report = orjson.loads((root / 'port.json').read_bytes())
        inventory = orjson.loads((root / 'inventory.json').read_bytes())
        self.parser_hashes = binding_hashes()
        if inventory['parser_hashes'] != self.parser_hashes:
            raise ValueError('hash-bound source parser changed since corpus conversion')
        self.raw = [row for row in self.report['results'] if row.get('status') == 'admitted_actor_visible_episode']
        self.raw_ids = {row['native_game_id'] for row in self.raw}
        self.raw_by_id = {row['native_game_id']: row for row in self.raw}
        self.counts: Counter[str] = Counter()
        self.signatures: dict[str, int] = {}
        self.excluded_games: set[str] = set()
        self.snapshot_index = output / 'snapshot-index.sqlite3'
        self.connection: sqlite3.Connection | None = None

    def episode(self, entry: Payload) -> Episode:
        path = Path(entry['path']) / 'episode.json.zst'
        relative = path.resolve().relative_to(self.root.resolve()).as_posix()
        if file_hash(path) != self.manifest['files_sha256'][relative]:
            raise ValueError('raw episode changed')
        episode = Episode.from_bytes(zstandard.ZstdDecompressor().decompress(path.read_bytes()))
        if episode.source_native_game_id != entry['native_game_id'] or episode.source_sha256 != entry['source']['sha256']:
            raise ValueError('raw episode source identity changed')
        validate_episode(episode)
        return episode

    def audit_raw(self) -> None:
        """No selection limits: build reserved-split protection for every action type."""
        games, contents, game_contents = {}, {}, {}
        for ordinal, entry in enumerate(self.raw, 1):
            episode, split = self.episode(entry), entry['effective_split']
            if episode.source_native_game_id in games:
                raise ValueError('duplicate raw native game')
            games[episode.source_native_game_id] = split
            rank = SPLIT_PRIORITY[split]
            sequences = actor_sequences(episode, split)
            content_sha = digest(canonical([sequence.records for sequence in sequences]))
            game_contents[episode.source_native_game_id] = content_sha
            contents[content_sha] = max(rank, contents.get(content_sha, -1))
            for sequence in sequences:
                self.counts[f'raw:{split}:requests'] += len(sequence.predictions)
                self.counts[f'raw:{split}:recorded_post_states'] += len(sequence.preserved_transition_ids)
                self.counts[f'raw:{split}:non_request_post_states'] += len(sequence.unavailable_post_states)
                for prediction in sequence.predictions:
                    self.signatures[prediction.signature] = max(rank, self.signatures.get(prediction.signature, -1))
            if ordinal % 50 == 0:
                self.progress('audit_raw', {'games': ordinal, 'total_games': len(self.raw)})
        self.excluded_games = {game for game, content in game_contents.items() if contents[content] > SPLIT_PRIORITY[games[game]]}

    def raw_sequences(self, split: str) -> Iterator[ActorSequence]:
        excluded = {signature for signature, rank in self.signatures.items() if rank > SPLIT_PRIORITY[split]}
        for entry in self.raw:
            if entry['effective_split'] == split and entry['native_game_id'] not in self.excluded_games:
                yield from actor_sequences(self.episode(entry), split, excluded)

    def index_snapshots(self) -> None:
        if self.snapshot_index.exists():
            raise FileExistsError('snapshot index already exists; resume must use its recorded identity')
        connection = sqlite3.connect(self.snapshot_index)
        self.connection = connection
        # Rebuildable index only: a crash invalidates preparation, never model
        # recovery. Keep the consumption/checkpoint ledger durable in the runner.
        connection.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-262144; CREATE TABLE refs (id TEXT PRIMARY KEY, source TEXT NOT NULL, line INTEGER NOT NULL, split TEXT NOT NULL, family TEXT NOT NULL, priority INTEGER NOT NULL, payload BLOB NOT NULL); CREATE TABLE sources (path TEXT PRIMARY KEY, sha TEXT NOT NULL);')
        codec = zstandard.ZstdCompressor(level=1)
        source_hashes: dict[str, str] = {}
        for part_index, entry in enumerate(row for row in self.report['results'] if row['kind'] == 'canonical'):
            path = Path(entry['path']) / 'frames.jsonl.zst'
            relative = path.resolve().relative_to(self.root.resolve()).as_posix()
            if file_hash(path) != self.manifest['files_sha256'][relative]:
                raise ValueError('canonical reference shard changed')
            for frame in compressed_rows(path):
                family, split = frame['source_family'], frame['effective_split']
                prefix = f'snapshot:{family}:{split}'
                self.counts[prefix + ':references'] += 1
                native = frame['native_game_binding']['raw_game_id']
                # Raw episodes already supply exact actions, chronology and split.
                if native in self.raw_ids:
                    raw = self.raw_by_id[native]
                    if SPLIT_PRIORITY[split] > SPLIT_PRIORITY[raw['effective_split']]:
                        raw['effective_split'] = split
                    self.counts[prefix + ':covered_by_raw_game'] += 1
                    continue
                if frame['source_schema'] == 'deepmew.replay_action_row.v1':
                    self.counts[prefix + ':unresolved_prior_request_alignment'] += 1
                    continue
                if split == 'train' and frame['target_mask']['source_target_eligible'] is not True:
                    self.counts[prefix + ':historical_split_exclusion'] += 1
                    continue
                reference = frame['state_before']
                source = reference['source_path']
                prior = source_hashes.get(source)
                if prior is not None and prior != reference['source_sha256']:
                    raise ValueError('conflicting source file hashes')
                if prior is None:
                    connection.execute('INSERT INTO sources VALUES (?, ?)', (source, reference['source_sha256']))
                    source_hashes[source] = reference['source_sha256']
                inserted = index_reference(connection, frame, codec.compress(orjson.dumps(frame)))
                self.counts[prefix + (':indexed_or_reserved' if inserted else ':exact_duplicate_reference')] += 1
            connection.commit()
            self.progress('index_snapshots', {'parts': part_index + 1, 'counts': dict(self.counts)})
        connection.execute('CREATE INDEX source_lines ON refs(source,line)')
        connection.commit()
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('prepared snapshot index failed integrity check')
        self.progress('snapshot_index_complete', {'unique_references': connection.execute('SELECT COUNT(*) FROM refs').fetchone()[0]})

    def snapshots(self, split: str) -> Iterator[ActorSequence]:
        if self.connection is None:
            self.connection = sqlite3.connect(self.snapshot_index)
        connection = self.connection
        codec = zstandard.ZstdDecompressor()
        for source, expected in connection.execute('SELECT path,sha FROM sources ORDER BY path').fetchall():
            path = Path(source)
            if file_hash(path) != expected:
                raise ValueError('snapshot source changed or unavailable')
            cursor = connection.execute('SELECT line,payload FROM refs WHERE source=? AND split!=? ORDER BY line,id' if split == 'reserved' else 'SELECT line,payload FROM refs WHERE source=? AND split=? ORDER BY line,id', (source, 'train' if split == 'reserved' else split))
            pending = cursor.fetchone()
            if pending is None:
                continue
            opener = gzip.open if path.suffix == '.gz' else open
            self.progress('resolve_snapshot_source', {'source': source, 'split': split})
            with opener(path, 'rb') as handle:
                for line_number, line in enumerate(handle, 1):
                    if pending is None:
                        break
                    if line_number != pending[0]:
                        continue
                    raw = orjson.loads(line)
                    while pending is not None and pending[0] == line_number:
                        frame = orjson.loads(codec.decompress(pending[1]))
                        resolved = resolve_row(frame, raw)
                        records: list[list[tuple[str, str, float]]] = []
                        projected = load_contract().project_actor(raw[frame['state_before']['source_field']])
                        context = relative_players({key: value for key, value in projected['select'].items() if key != 'option'}, resolved.state.actor_seat)
                        public_logs = load_contract().public_logs(raw[frame['state_before']['source_field']].get('logs', []))
                        records.append(list(leaves({'state': actor_state(resolved.state), 'request': context, 'public_log_snapshot': relative_players(public_logs, resolved.state.actor_seat), 'history_available': False})))
                        stages = structured_target(resolved.legal_actions, resolved.target_action, resolved.state.actor_seat, records)
                        options = [relative_players(option['payload'], resolved.state.actor_seat) for stage in resolved.legal_actions['stages'] for option in stage['options']]
                        signature = input_signature(actor_state(resolved.state), context, options)
                        actual_split = frame['effective_split']
                        eligible = resolved.legal_actions['complete_action_count'] != 1 and self.signatures.get(signature, -1) <= SPLIT_PRIORITY[actual_split] and frame['target_mask']['source_target_eligible'] is True
                        prediction = Prediction(resolved.frame_id, 0, stages, signature, eligible)
                        yield ActorSequence(resolved.episode_id, resolved.state.actor_seat, frame['source_family'], actual_split, tuple(records), (0,), (prediction,), (), (), False)
                        pending = cursor.fetchone()
            if pending is not None or file_hash(path) != expected:
                raise ValueError('missing snapshot reference or source mutation')

    def audit_reserved_snapshots(self) -> None:
        for ordinal, sequence in enumerate(self.snapshots('reserved'), 1):
            for prediction in sequence.predictions:
                self.signatures[prediction.signature] = max(SPLIT_PRIORITY[sequence.split], self.signatures.get(prediction.signature, -1))
            if ordinal % 5000 == 0:
                self.progress('audit_reserved_snapshots', {'rows': ordinal, 'guarded_signatures': len(self.signatures)})

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
