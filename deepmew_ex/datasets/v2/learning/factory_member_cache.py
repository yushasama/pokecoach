"""Source-bound recovery of independently complete factory gzip game members."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.factory_members import verified_members
from deepmew_ex.datasets.v2.learning.factory_source import SourceStreamAudit, factory_rows, recover_factory_source_game
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, sequence_from_payload
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.records import validate_hash
from deepmew_ex.registry.store import canonical, digest, file_hash


def member_coordinates(sequences: tuple[ActorSequence, ...], rows: list[Payload], source_sha: str, member_start: int) -> tuple[ActorSequence, ...]:
    """Keep features intact; identify targets by original compressed byte/member line."""
    result = []
    for sequence in sequences:
        coordinates = [index + 1 for index, row in enumerate(rows) if row['seat'] == sequence.actor]
        if len(coordinates) != len(sequence.predictions):
            raise ValueError('member actor/request count differs from recovered sequence')
        predictions = tuple(replace(prediction, row_id='factory-member-frame:' + digest(canonical([source_sha, member_start, line, sequence.actor]))) for prediction, line in zip(sequence.predictions, coordinates, strict=True))
        mapping = {old.row_id: new.row_id for old, new in zip(sequence.predictions, predictions, strict=True)}
        gaps = []
        for gap in sequence.unavailable_post_states:
            if not gap.startswith('unavailable-post-state:'):
                gaps.append(gap)
                continue
            original = gap.removeprefix('unavailable-post-state:')
            if original not in mapping:
                raise ValueError('member post-state gap has an unknown prediction identity')
            gaps.append('unavailable-post-state:' + mapping[original])
        result.append(replace(sequence, predictions=predictions, unavailable_post_states=tuple(gaps)))
    return tuple(result)


def prepare_factory_members(shard: ReferenceShard, lineage_splits: Mapping[tuple[int, int], str], lineage_sha256: str, output: Path, check: Callable[[], None], progress: Callable[[dict[str, object]], None], parent: Path | None = None, parent_sha256: str | None = None) -> dict[str, object]:
    validate_hash(lineage_sha256)
    if output.resolve() in {shard.decisions.resolve(), shard.summaries.resolve()} or (parent is None) != (parent_sha256 is None) or (parent is not None and output.resolve() == parent.resolve()):
        raise ValueError('member recovery input/output collision or incomplete parent binding')
    if any(len(key) != 2 or any(type(value) is not int for value in key) or split not in SPLIT_PRIORITY for key, split in lineage_splits.items()):
        raise ValueError('member recovery requires explicit valid lineage splits')
    for path, sha in ((shard.decisions, shard.decisions_sha256), (shard.summaries, shard.summaries_sha256)):
        check()
        if file_hash(path) != sha:
            raise ValueError('member recovery source identity changed')
    summaries: dict[str, Payload] = {}
    for _, row in factory_rows(shard.summaries, SourceStreamAudit(), check):
        native = row.get('game_id')
        if not isinstance(native, str) or native in summaries or (row.get('seed'), row.get('game_index')) not in lineage_splits:
            raise ValueError('member summary identity or lineage is unresolved')
        summaries[native] = row
    if len(summaries) != shard.games or sum(row['decisions'] for row in summaries.values()) != shard.decision_rows:
        raise ValueError('member summary population mismatch')
    root = Path(__file__).parents[1]
    modules = ('learning/factory_member_cache.py', 'learning/factory_members.py', 'learning/factory_source.py', 'learning/recovery.py', 'learning/actor_events.py', 'learning/queries.py', 'learning/trajectory.py', 'learning/semantic.py', 'learning/admission.py', 'corpus/reader.py', 'corpus/records.py', 'conversion.py', 'schema.py', 'contract.py')
    identity = {'shard': {**asdict(shard), 'decisions': str(shard.decisions), 'summaries': str(shard.summaries)}, 'lineage_sha256': lineage_sha256, 'split_mapping_sha256': digest(canonical([[*key, value] for key, value in sorted(lineage_splits.items())])), 'parser': binding_hashes(), 'code': {name: file_hash(root / name) for name in modules}, 'version': 1, 'coordinates': 'original source SHA256 + gzip member compressed start byte + member-local JSON line + actor', 'parent': None if parent is None else {'path': str(parent), 'sha256': parent_sha256}}
    previous_cache = None
    if parent is not None:
        check()
        wal = Path(str(parent) + '-wal')
        if (wal.exists() and wal.stat().st_size) or file_hash(parent) != parent_sha256:
            raise ValueError('member parent cache changed or is live')
        previous_cache = sqlite3.connect(parent.resolve().as_uri() + '?mode=ro', uri=True)
        old = orjson.loads(previous_cache.execute('SELECT identity FROM metadata').fetchone()[0])
        if old['version'] != 2 or old['shard'] != identity['shard'] or old['lineage_sha256'] != lineage_sha256 or old['split_mapping_sha256'] != identity['split_mapping_sha256'] or old['parser'] != identity['parser']:
            previous_cache.close()
            raise ValueError('member parent source/split/parser mismatch')
        for name, sha in old['code'].items():
            if file_hash(root / name) != sha:
                previous_cache.close()
                raise ValueError('member parent producer changed')
    connection = sqlite3.connect(output)
    encoder, decoder = zstandard.ZstdCompressor(level=3), zstandard.ZstdDecompressor()
    try:
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS metadata(identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS members(start INTEGER PRIMARY KEY,end INTEGER NOT NULL,receipt TEXT NOT NULL); CREATE TABLE IF NOT EXISTS games(game TEXT PRIMARY KEY,native TEXT UNIQUE NOT NULL,member_start INTEGER NOT NULL,requests INTEGER NOT NULL,split TEXT NOT NULL,rows_sha TEXT NOT NULL,sequence_identity TEXT NOT NULL,reused_parent INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS sequences(game TEXT NOT NULL,actor INTEGER NOT NULL,events INTEGER NOT NULL,records INTEGER NOT NULL,eligible INTEGER NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(game,actor)); CREATE TABLE IF NOT EXISTS completion(receipt TEXT NOT NULL);')
        encoded_identity = canonical(identity).decode()
        old_identity = connection.execute('SELECT identity FROM metadata').fetchone()
        if old_identity is None:
            connection.execute('INSERT INTO metadata VALUES (?)', (encoded_identity,)); connection.commit()
        elif old_identity != (encoded_identity,):
            raise ValueError('member cache source/code/lineage identity changed')

        def store_member(receipt: dict[str, object]) -> None:
            encoded = canonical(receipt).decode()
            existing = connection.execute('SELECT end,receipt FROM members WHERE start=?', (receipt['start'],)).fetchone()
            if existing is None:
                with connection:
                    connection.execute('INSERT INTO members VALUES (?,?,?)', (receipt['start'], receipt['end'], encoded))
            elif existing != (receipt['end'], encoded):
                raise ValueError('member byte partition changed on resume')

        # Recheck compressed CRCs on resume. Reuse committed features, never replay
        # conversion merely because the archive had a later interrupted member.
        first_line = 1
        prefix_intact = True
        def rejected(receipt: dict[str, object]) -> None:
            nonlocal prefix_intact
            prefix_intact = False
            store_member(receipt)

        for member in verified_members(shard.decisions, output.parent, check, rejected):
            check()
            rows = [orjson.loads(line) for line in member.stream]
            if not rows or any(not isinstance(row, dict) for row in rows) or len({row.get('game_id') for row in rows}) != 1:
                raise ValueError('member is not one complete JSON game group; investigate before alternate grouping')
            native = rows[0]['game_id']
            base = {'start': member.start, 'end': member.end, 'sha256': member.sha256, 'decoded_bytes': member.decoded_bytes, 'native': native, 'observed_requests': len(rows)}
            if native not in summaries or len(rows) != summaries[native]['decisions']:
                store_member({**base, 'status': 'incomplete_game', 'reason': 'missing_terminal_summary' if native not in summaries else 'request_count_mismatch'})
                first_line += len(rows)
                continue
            summary = summaries[native]
            split = lineage_splits[summary['seed'], summary['game_index']]
            rows_sha = digest(orjson.dumps(rows))
            previous = connection.execute('SELECT game,requests,split,rows_sha,sequence_identity,member_start FROM games WHERE native=?', (native,)).fetchone()
            if previous is not None:
                members = connection.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=? ORDER BY actor', (previous[0],)).fetchall()
                if previous[1:4] != (len(rows), split, rows_sha) or canonical([(*row[:-1], digest(row[-1])) for row in members]).decode() != previous[4]:
                    raise ValueError('member game has conflicting source content or committed features')
                store_member({**base, 'status': 'candidate' if previous[5] == member.start else 'duplicate_verified_game', 'game': previous[0]})
                first_line += len(rows)
                continue
            parent_game = None if previous_cache is None or not prefix_intact else previous_cache.execute('SELECT game,first_line,requests,split,sequence_identity FROM games WHERE native=?', (native,)).fetchone()
            if parent_game is None:
                sequences = recover_factory_source_game(rows, summary, shard.decisions, shard.decisions_sha256, 1, split, (True,) * len(rows)).sequences
            else:
                members = previous_cache.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=? ORDER BY actor', (parent_game[0],)).fetchall()
                if parent_game[1:4] != (first_line, len(rows), split) or canonical([(*row[:-1], digest(row[-1])) for row in members]).decode() != parent_game[4]:
                    raise ValueError('member parent prefix does not match source coordinates/features')
                sequences = tuple(sequence_from_payload(orjson.loads(decoder.decompress(row[-1]))) for row in members)
                for sequence in sequences:
                    positions = [index for index, row in enumerate(rows) if row['seat'] == sequence.actor]
                    expected = ['factory-frame:' + digest(canonical([shard.decisions_sha256, first_line + index, sequence.actor, index])) for index in positions]
                    if [row.row_id for row in sequence.predictions] != expected:
                        raise ValueError('member parent target order changed')
            sequences = member_coordinates(sequences, rows, shard.decisions_sha256, member.start)
            game = sequences[0].game_id
            with connection:
                members = []
                for sequence in sequences:
                    row = (sequence.actor, len(sequence.events), len(sequence.records), sum(p.loss_eligible for p in sequence.predictions), encoder.compress(orjson.dumps(asdict(sequence))))
                    connection.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?)', (game, *row))
                    members.append((*row[:-1], digest(row[-1])))
                connection.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?)', (game, native, member.start, len(rows), split, rows_sha, canonical(members).decode(), int(parent_game is not None)))
            store_member({**base, 'status': 'candidate', 'game': game})
            first_line += len(rows)
            games, requests = connection.execute('SELECT COUNT(*),SUM(requests) FROM games').fetchone()
            if games % 100 == 0:
                progress({'games': games, 'requests': requests, 'member_end': member.end, 'training_admitted': False})
        end = 0
        statuses: dict[str, int] = {}
        for start, next_end, encoded in connection.execute('SELECT start,end,receipt FROM members ORDER BY start'):
            if start != end or next_end <= start:
                raise ValueError('member source byte coverage has a gap or overlap')
            end = next_end
            status = orjson.loads(encoded)['status']
            statuses[status] = statuses.get(status, 0) + 1
        if end != shard.decisions.stat().st_size or connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('member source coverage or cache integrity failure')
        missing = sorted(set(summaries) - {row[0] for row in connection.execute('SELECT native FROM games')})
        for path, sha in ((shard.decisions, shard.decisions_sha256), (shard.summaries, shard.summaries_sha256)):
            check()
            if file_hash(path) != sha:
                raise ValueError('member source changed during recovery')
        games, requests, reused = connection.execute('SELECT COUNT(*),COALESCE(SUM(requests),0),COALESCE(SUM(reused_parent),0) FROM games').fetchone()
        receipt = {'identity': identity, 'games': games, 'requests': requests, 'sequences': connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0], 'eligible_before_global_admission': connection.execute('SELECT COALESCE(SUM(eligible),0) FROM sequences').fetchone()[0], 'member_statuses': statuses, 'source_bytes_accounted': end, 'summary_games_without_complete_recovery': missing, 'reused_parent_games': reused, 'training_admitted': False, 'coverage': 'CRC-verified complete recorded game members; rejected byte spans preserved in original source and ledger; no invented missing events'}
        encoded = canonical(receipt).decode()
        previous = connection.execute('SELECT receipt FROM completion').fetchone()
        if previous is None:
            connection.execute('INSERT INTO completion VALUES (?)', (encoded,)); connection.commit()
        elif previous != (encoded,):
            raise ValueError('member completion changed')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
        if previous_cache is not None:
            previous_cache.close()
    return {**receipt, 'cache_sha256': file_hash(output)}
