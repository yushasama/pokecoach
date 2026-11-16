"""Stream verified reference archives into resumable, non-admitted temporal candidates."""

from __future__ import annotations

import gzip
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.recovery import RecoveredGame, recover_reference_game
from deepmew_ex.datasets.v2.learning.queries import actor_queries, enrich_query_views
from deepmew_ex.datasets.v2.learning.trajectory import sequence_from_payload
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import canonical, digest, file_hash


@dataclass(frozen=True)
class ReferenceShard:
    decisions: Path
    decisions_sha256: str
    summaries: Path
    summaries_sha256: str
    decision_rows: int
    games: int


def source_games(path: Path, check_budget: Callable[[], None]) -> Iterator[tuple[int, list[Payload]]]:
    """Keep one complete game in memory; never silently stitch separated fragments."""
    opener = gzip.open if path.suffix == '.gz' else Path.open
    with opener(path, 'rb') as handle:
        rows: list[Payload] = []
        first = 1
        for line_number, line in enumerate(handle, 1):
            row = orjson.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('game_id'), str):
                raise ValueError(f'invalid source game at line {line_number}')
            if rows and row['game_id'] != rows[0]['game_id']:
                yield first, rows
                rows, first = [], line_number
            rows.append(row)
            if line_number % 1000 == 0:
                check_budget()
        if rows:
            yield first, rows


def prepare_reference_shard(shard: ReferenceShard, reference_index: Path, index_sha256: str, output: Path, check_budget: Callable[[], None], progress: Callable[[dict[str, object]], None], prior_cache: Path | None = None, prior_cache_sha256: str | None = None) -> dict[str, object]:
    """All outputs are candidates until cross-source split/dedup admission completes.

    A committed game is a restart unit, never training authority. Any source or
    contract disagreement raises with its game coordinate and leaves evidence.
    """
    if output.resolve() in {path.resolve() for path in (reference_index, shard.decisions, shard.summaries, *(() if prior_cache is None else (prior_cache,)))}:
        raise ValueError('candidate output would overwrite a recovery input')
    for path, expected in ((reference_index, index_sha256), (shard.decisions, shard.decisions_sha256), (shard.summaries, shard.summaries_sha256)):
        check_budget()
        if file_hash(path) != expected:
            raise ValueError(f'recovery input hash mismatch: {path}')
    if (prior_cache is None) != (prior_cache_sha256 is None) or prior_cache is not None and file_hash(prior_cache) != prior_cache_sha256:
        raise ValueError('query enrichment parent cache identity mismatch')
    if prior_cache is not None:
        wal = prior_cache.with_name(prior_cache.name + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('query enrichment parent has uncheckpointed writes')
    modules = ('learning/recovery_cache.py', 'learning/recovery.py', 'learning/actor_events.py', 'learning/queries.py', 'learning/trajectory.py', 'learning/semantic.py', 'learning/admission.py', 'corpus/reader.py', 'corpus/records.py', 'conversion.py', 'schema.py', 'contract.py')
    identity = {'shard': {**asdict(shard), 'decisions': str(shard.decisions), 'summaries': str(shard.summaries)}, 'index_sha256': index_sha256, 'code': {name: file_hash(Path(__file__).parents[1] / name) for name in modules}, 'parser': binding_hashes(), 'version': 4, 'query_enrichment_parent': None if prior_cache is None else {'path': str(prior_cache.resolve()), 'sha256': prior_cache_sha256}}
    opener = gzip.open if shard.summaries.suffix == '.gz' else Path.open
    summaries: dict[str, Payload] = {}
    with opener(shard.summaries, 'rb') as handle:
        for line in handle:
            row = orjson.loads(line)
            game = row['game_id']
            if game in summaries:
                raise ValueError('duplicate terminal summary')
            summaries[game] = row
    if len(summaries) != shard.games:
        raise ValueError('summary population mismatch')
    index = sqlite3.connect(reference_index.resolve().as_uri() + '?mode=ro', uri=True)
    connection = sqlite3.connect(output)
    parent = None
    try:
        parent_games: set[str] = set()
        if prior_cache is not None:
            parent = sqlite3.connect(prior_cache.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
            previous_identity = orjson.loads(parent.execute('SELECT identity FROM metadata').fetchone()[0])
            if previous_identity.get('version') != 3 or any(previous_identity.get(key) != identity[key] for key in ('shard', 'index_sha256', 'parser')):
                raise ValueError('query enrichment parent source or parser mismatch')
            previous_code = previous_identity.get('code', {})
            migration_modules = {'learning/recovery_cache.py', 'learning/recovery.py'}
            if set(previous_code) != set(identity['code']) - {'learning/queries.py'} or any(previous_code[name] != sha for name, sha in identity['code'].items() if name not in migration_modules | {'learning/queries.py'}):
                raise ValueError('query enrichment unchanged semantic dependency drift')
            parent_games = {row[0] for row in parent.execute('SELECT game FROM games')}
            if parent.execute('SELECT COUNT(*) FROM sequences LEFT JOIN games USING(game) WHERE games.game IS NULL').fetchone()[0] or parent.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('query enrichment parent integrity failure')
        expected_source = index.execute('SELECT sha FROM sources WHERE path=?', (str(shard.decisions),)).fetchone()
        if expected_source != (shard.decisions_sha256,):
            raise ValueError('source does not match frozen reference inventory')
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS games (game TEXT PRIMARY KEY,first_line INTEGER NOT NULL,requests INTEGER NOT NULL,split TEXT NOT NULL,result INTEGER NOT NULL,missing_post_states INTEGER NOT NULL,public_events INTEGER NOT NULL,sequence_identity TEXT NOT NULL,identity_retained INTEGER NOT NULL,identity_withheld INTEGER NOT NULL,query_cards INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS sequences (game TEXT NOT NULL,actor INTEGER NOT NULL,events INTEGER NOT NULL,records INTEGER NOT NULL,eligible INTEGER NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(game,actor)); CREATE TABLE IF NOT EXISTS completion (receipt TEXT NOT NULL);')
        encoded = canonical(identity).decode()
        previous = connection.execute('SELECT identity FROM metadata').fetchone()
        if previous is None:
            connection.execute('INSERT INTO metadata VALUES (?)', (encoded,))
            connection.commit()
        elif previous != (encoded,):
            raise ValueError('candidate preparation identity changed')
        completed = {row[0]: row[1:] for row in connection.execute('SELECT game,first_line,requests,sequence_identity FROM games')}
        seen: set[str] = set()
        count = 0
        compressor, decoder = zstandard.ZstdCompressor(level=3), zstandard.ZstdDecompressor()
        for first, rows in source_games(shard.decisions, check_budget):
            check_budget()
            game = rows[0]['game_id']
            if game in seen or game not in summaries:
                raise ValueError(f'repeated or unbound source game: {game}')
            seen.add(game)
            count += len(rows)
            if game in completed:
                source_first, requests, expected_members = completed[game]
                actual_members = [(*row[:-1], digest(row[-1])) for row in connection.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=? ORDER BY actor', (game,))]
                if (source_first, requests) != (first, len(rows)) or canonical(actual_members).decode() != expected_members:
                    raise ValueError('resumed game membership or payload identity changed')
                continue
            try:
                if game in parent_games and parent is not None:
                    cached = parent.execute('SELECT first_line,requests,split,result,missing_post_states,public_events,sequence_identity,identity_retained,identity_withheld FROM games WHERE game=?', (game,)).fetchone()
                    members = parent.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=? ORDER BY actor', (game,)).fetchall()
                    if cached[:2] != (first, len(rows)) or cached[3] != summaries[game]['result'] or canonical([(*row[:-1], digest(row[-1])) for row in members]).decode() != cached[6]:
                        raise ValueError('query enrichment parent game membership changed')
                    originals = [sequence_from_payload(orjson.loads(decoder.decompress(row[-1]))) for row in members]
                    if {sequence.actor for sequence in originals} != {row['seat'] for row in rows} or any(sequence.game_id != game or sequence.split != cached[2] for sequence in originals):
                        raise ValueError('query enrichment parent actor identity changed')
                    sequences = tuple(enrich_query_views(sequence, [row['raw_observation'] for row in rows if row['seat'] == sequence.actor]) for sequence in originals)
                    recovered = RecoveredGame(game, sequences, len(rows), cached[4], cached[3], cached[2], cached[5], cached[7], cached[8])
                else:
                    frames = [orjson.loads(decoder.decompress(row[0])) for row in index.execute('SELECT payload FROM refs WHERE source=? AND game=? ORDER BY global_order', (str(shard.decisions), game))]
                    if sorted(frame['state_before']['source_line'] for frame in frames) != list(range(first, first + len(rows))):
                        raise ValueError('canonical references do not cover the complete source game')
                    recovered = recover_reference_game(frames, rows, summaries[game])
            except ValueError as error:
                raise ValueError(f'{game} at source line {first}: {error}') from error
            with connection:
                members = []
                for sequence in recovered.sequences:
                    row = (sequence.actor, len(sequence.events), len(sequence.records), sum(row.loss_eligible for row in sequence.predictions), compressor.compress(orjson.dumps(asdict(sequence))))
                    connection.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?)', (game, *row))
                    members.append((*row[:-1], digest(row[-1])))
                query_cards = sum(sum(card is not None for card in values) for row in rows for values in actor_queries(row['raw_observation'], row['seat']).values() if isinstance(values, list))
                connection.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)', (game, first, len(rows), recovered.effective_split, recovered.terminal_result, recovered.missing_immediate_post_states, recovered.public_events, canonical(members).decode(), recovered.event_identity_fields_retained, recovered.event_identity_fields_withheld, query_cards))
            if len(seen) % 100 == 0:
                progress({'phase': 'recover_candidates', 'source': str(shard.decisions), 'games': len(seen), 'requests': count, 'training_admitted': False})
        if count != shard.decision_rows or seen != set(summaries) or not set(completed) <= seen or not parent_games <= seen:
            raise ValueError('source population or terminal-summary coverage mismatch')
        if connection.execute('SELECT COUNT(*) FROM sequences LEFT JOIN games USING(game) WHERE games.game IS NULL').fetchone()[0]:
            raise ValueError('orphan sequence in candidate cache')
        for path, expected in ((shard.decisions, shard.decisions_sha256), (shard.summaries, shard.summaries_sha256)):
            check_budget()
            if file_hash(path) != expected:
                raise ValueError('source changed during recovery')
        if prior_cache is not None and file_hash(prior_cache) != prior_cache_sha256:
            raise ValueError('query enrichment parent changed during recovery')
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('candidate cache integrity failure')
        receipt = {'identity': identity, 'games': len(seen), 'requests': count, 'sequences': connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0], 'eligible_before_global_admission': connection.execute('SELECT SUM(eligible) FROM sequences').fetchone()[0], 'missing_immediate_post_states': connection.execute('SELECT SUM(missing_post_states) FROM games').fetchone()[0], 'recorded_public_events': connection.execute('SELECT SUM(public_events) FROM games').fetchone()[0], 'event_identity_fields_retained': connection.execute('SELECT SUM(identity_retained) FROM games').fetchone()[0], 'event_identity_fields_withheld': connection.execute('SELECT SUM(identity_withheld) FROM games').fetchone()[0], 'coverage': 'complete_recorded_request_log; own-observation event identities and observer-witnessed cross-seat identities; intermediate_engine_events_not_proven; unwitnessed cross-seat identities withheld', 'training_admitted': False}
        receipt.update({'query_card_presentations': connection.execute('SELECT SUM(query_cards) FROM games').fetchone()[0], 'query_views': 'native actor-visible deck-search and looking cards; no cross-seat sharing', 'games_enriched_from_verified_v3': len(parent_games)})
        with connection:
            previous_receipt = connection.execute('SELECT receipt FROM completion').fetchone()
            if previous_receipt is None:
                connection.execute('INSERT INTO completion VALUES (?)', (canonical(receipt).decode(),))
            elif previous_receipt != (canonical(receipt).decode(),):
                raise ValueError('completed candidate receipt changed')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
        index.close()
        if parent is not None:
            parent.close()
    return {**receipt, 'cache_sha256': file_hash(output)}
