"""Frozen temporal membership and streaming access to admitted full histories."""

from __future__ import annotations

import hashlib
import multiprocessing
import multiprocessing.pool
import random
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.history import INPUT_CONTRACTS, QUERY_INPUT_CONTRACT
from deepmew_ex.datasets.v2.learning.legacy_contract import recorded_legacy_contract
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, sequence_from_payload
from deepmew_ex.registry.records import validate_hash
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash, verify_run

CandidateKind = Literal['reference', 'factory', 'factory_members', 'recorded', 'recorded_group', 'raw']


@dataclass(frozen=True)
class CandidateCache:
    path: Path
    sha256: str
    kind: CandidateKind


def publish_prepared(store: RunStore, caches: tuple[CandidateCache, ...], inventory: Path, inventory_sha256: str, check: Callable[[], None], recover_from_publisher: str | None = None) -> dict[str, object]:
    """Publish every cache in the frozen inventory, or leave an unsealed attempt.

    The caller reconciles source-universe coverage and binds that evidence in
    the inventory. This function enforces exact inventory consumption, cache
    integrity, conservative teacher exclusion and global membership. It cannot
    prove that an inventory includes historical sources unknown to its author.
    """
    validate_hash(inventory_sha256)
    inventory_bytes = inventory.read_bytes()
    if digest(inventory_bytes) != inventory_sha256:
        raise ValueError('prepared source inventory changed')
    plan = orjson.loads(inventory_bytes)
    supplied = [{'path': str(cache.path.resolve()), 'sha256': cache.sha256, 'kind': cache.kind} for cache in caches]
    if not caches or plan.get('candidate_caches') != supplied or len({row['path'] for row in supplied}) != len(caches):
        raise ValueError('prepared candidates must exactly cover the frozen source inventory')
    if plan.get('input_contract') != QUERY_INPUT_CONTRACT or plan.get('source_reconciliation_complete') is not True or not plan.get('coverage_evidence'):
        raise ValueError('prepared publication requires reconciled coverage and explicit query contract')
    original = plan.get('original_corpus', {})
    validate_hash(original.get('sha256'))
    original_path = Path(original['manifest'])
    original_bytes = original_path.read_bytes()
    if digest(original_bytes) != original['sha256']:
        raise ValueError('original corpus identity changed')
    original_manifest = orjson.loads(original_bytes)
    report_path = original_path.parent / 'port.json'
    report_bytes = report_path.read_bytes()
    report_sha = digest(report_bytes)
    if original_manifest.get('status') != 'completed' or original_manifest.get('kind') != 'v2_full_corpus_port' or original_manifest.get('files_sha256', {}).get('port.json') != report_sha:
        raise ValueError('original corpus report is not bound to the completed port')
    catalog_rows = orjson.loads(report_bytes)['catalog_rows']
    if type(catalog_rows) is not int or catalog_rows < 0:
        raise ValueError('invalid original catalog population')
    for evidence in plan['coverage_evidence']:
        validate_hash(evidence['sha256'])
        if file_hash(Path(evidence['path'])) != evidence['sha256']:
            raise ValueError('source coverage evidence changed')
    for cache in caches:
        validate_hash(cache.sha256)
        if cache.path.resolve().is_relative_to(store.path.resolve()):
            raise ValueError('candidate cache cannot live inside its publication output')
    check()
    if recover_from_publisher is None:
        store.write('source-inventory.json', plan)
        store.write('publication-input.json', {'source_inventory': str(inventory.resolve()), 'source_inventory_sha256': inventory_sha256, 'publisher_sha256': file_hash(Path(__file__))})
    else:
        validate_hash(recover_from_publisher)
        if (store.path / 'manifest.json').exists() or orjson.loads((store.path / 'source-inventory.json').read_bytes()) != plan:
            raise ValueError('recovery requires the same unsealed source inventory')
        prior = orjson.loads((store.path / 'publication-input.json').read_bytes())
        if prior != {'source_inventory': str(inventory.resolve()), 'source_inventory_sha256': inventory_sha256, 'publisher_sha256': recover_from_publisher}:
            raise ValueError('recovery publisher or inventory identity changed')
    coverage: list[dict[str, object]] = []

    def sequences() -> Iterator[ActorSequence]:
        for cache in caches:
            counts = {'actor_sequences': 0, 'requests': 0, 'source_eligible': 0, 'withheld_incomplete_history': 0, 'unavailable_state_receipts': 0}
            shapes: dict[str, dict[str, int]] = {}
            for sequence in cached_sequences(cache.path, cache.sha256, cache.kind, check):
                counts['actor_sequences'] += 1
                counts['requests'] += len(sequence.predictions)
                counts['source_eligible'] += sum(row.loss_eligible for row in sequence.predictions)
                counts['unavailable_state_receipts'] += len(sequence.unavailable_post_states)
                shape = shapes.setdefault(sequence.split, {'sequences': 0, 'events': 0, 'leaves': 0, 'max_events': 0, 'max_leaves': 0, 'max_records': 0})
                leaves = sum(map(len, sequence.records))
                shape['sequences'] += 1
                shape['events'] += len(sequence.events)
                shape['leaves'] += leaves
                shape['max_events'] = max(shape['max_events'], len(sequence.events))
                shape['max_leaves'] = max(shape['max_leaves'], leaves)
                shape['max_records'] = max(shape['max_records'], len(sequence.records))
                if cache.kind in ('recorded', 'recorded_group'):
                    counts['withheld_incomplete_history'] += sum(row.loss_eligible for row in sequence.predictions)
                    sequence = replace(sequence, predictions=tuple(replace(row, loss_eligible=False) for row in sequence.predictions))
                yield sequence
            coverage.append({**supplied[len(coverage)], **counts, 'shapes_by_original_split': shapes, 'fitting': cache.kind not in ('recorded', 'recorded_group'), 'history_scope': 'incomplete recorded actor selections' if cache.kind in ('recorded', 'recorded_group') else 'complete recovered recorded request history; unrecorded engine microsteps not inferred'})

    connection = sqlite3.connect(store.path / 'membership.sqlite3', timeout=1)
    try:
        if recover_from_publisher is None:
            membership = prepare_membership(connection, sequences(), check, lambda: store.write('ingestion.json', {'source_inventory_sha256': inventory_sha256, 'sources': coverage, 'status': 'all source caches exhausted and verified; global admission pending'}))
        else:
            from deepmew_ex.datasets.v2.learning.prepared_recovery import audit_ingested

            def recovery_progress(fields: dict[str, object]) -> None:
                encoded = orjson.dumps({'event': 'ingestion_recovery_audit', 'epoch_seconds': time.time(), **fields})
                with (store.path / 'recovery-progress.jsonl').open('ab') as handle:
                    handle.write(encoded + b'\n')
                print(encoded.decode(), flush=True)

            # A live SQL writer holds this lock. The caller must also verify that
            # the original publisher has terminated before requesting recovery.
            with guarded_sql(connection, check):
                connection.execute('BEGIN EXCLUSIVE')
                coverage = audit_ingested(connection, caches, check, recovery_progress)
                connection.commit()
            check()
            receipt_name = 'ingestion-recovery.json' if not (store.path / 'ingestion-recovery.json').exists() else f'ingestion-recovery-{time.time_ns()}.json'
            store.write(receipt_name, {'source_inventory_sha256': inventory_sha256, 'prior_publisher_sha256': recover_from_publisher, 'recovery_publisher_sha256': file_hash(Path(__file__)), 'audit_code_sha256': file_hash(Path(__file__).with_name('prepared_recovery.py')), 'verified_database_sha256': file_hash(store.path / 'membership.sqlite3'), 'sources': coverage, 'scope': 'every source payload, coordinate, original mask, split and causal signature rechecked; no target or history changed'})
            membership = finalize_membership(connection, check)
        if not connection.execute("SELECT 1 FROM predictions p JOIN sequences s ON s.ordinal=p.sequence_id WHERE s.split='train' AND p.eligible=1 LIMIT 1").fetchone():
            raise ValueError('prepared corpus has no eligible training targets')
    finally:
        connection.close()
    check()
    if file_hash(inventory) != inventory_sha256 or file_hash(original_path) != original['sha256'] or file_hash(report_path) != report_sha or any(file_hash(Path(row['path'])) != row['sha256'] for row in plan['coverage_evidence']):
        raise ValueError('prepared source evidence changed during publication')
    for cache in caches:
        check()
        wal = Path(str(cache.path) + '-wal')
        if (wal.exists() and wal.stat().st_size) or file_hash(cache.path) != cache.sha256:
            raise ValueError('candidate cache changed before publication')
    source_coverage = {'catalog_rows': catalog_rows, 'sources': coverage, 'membership': membership, 'counts_overlap_before_global_membership': True, 'training_consumption_proven': False, 'original_port_report_sha256': report_sha}
    store.write('coverage.json', source_coverage)
    return store.seal({'kind': 'v2_prepared_temporal_corpus', 'training_admitted': True, 'input_contract': QUERY_INPUT_CONTRACT, 'parser_hashes': binding_hashes(), 'original_corpus_manifest_sha256': original['sha256'], 'source_inventory_sha256': inventory_sha256, 'source_coverage': source_coverage, 'membership': membership, 'optimizer_updates': 0})


def validate_input_contract(sequence: ActorSequence, input_contract: str) -> None:
    if input_contract not in INPUT_CONTRACTS:
        raise ValueError('unsupported prepared input contract')
    query_mode = input_contract == QUERY_INPUT_CONTRACT
    for position, prediction in enumerate(sequence.predictions):
        record = sequence.records[sequence.events[prediction.event_index]]
        fields = {path.split('/')[2] for path, _, _ in record if path.startswith('/actor_queries/')}
        if (query_mode and fields != {'deck', 'looking'}) or (not query_mode and fields):
            raise ValueError('prepared features do not match the declared input contract')
        if query_mode and (len(sequence.events) != 2 * len(sequence.predictions) or prediction.event_index != 2 * position):
            raise ValueError('query-aware input contract requires exact request/action event pairs')


def cached_sequences(path: Path, sha256: str, kind: CandidateKind, check: Callable[[], None]) -> Iterator[ActorSequence]:
    """Stream completed source candidates into global membership, never train them.

    Source-universe completeness, generation split authority and partial-history
    objective eligibility remain the caller's admission obligations. A cache
    hash or a successfully streamed subset does not prove full-corpus coverage.
    """
    contracts = {'reference': (4, 'learning/recovery_cache.py'), 'factory': (2, 'learning/factory_cache.py'), 'factory_members': (1, 'learning/factory_member_cache.py'), 'recorded': (2, 'learning/recorded_cache.py'), 'recorded_group': (1, 'learning/recorded_generation.py'), 'raw': (2, 'learning/raw_cache.py')}
    if kind not in contracts:
        raise ValueError('unknown temporal candidate cache kind')
    wal = Path(str(path) + '-wal')

    def verify_identity() -> None:
        check()
        if (wal.exists() and wal.stat().st_size) or file_hash(path) != sha256:
            raise ValueError('candidate cache is live or its identity changed')

    verify_identity()
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        identities = connection.execute('SELECT identity FROM metadata').fetchall()
        completions = connection.execute('SELECT receipt FROM completion').fetchall()
        if len(identities) != 1 or len(completions) != 1:
            raise ValueError('candidate cache requires exact completed membership')
        identity, receipt = orjson.loads(identities[0][0]), orjson.loads(completions[0][0])
        expected_parser = recorded_legacy_contract().hashes if kind == 'raw' else binding_hashes()
        version, producer = contracts[kind]
        if receipt['identity'] != identity or identity['version'] != version or producer not in identity['code'] or identity['parser'] != expected_parser or receipt['training_admitted'] is not False:
            raise ValueError('candidate cache contract or completion identity mismatch')
        root = Path(__file__).parents[1].resolve()
        for name, expected in identity['code'].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or file_hash(source) != expected:
                raise ValueError('candidate producer code changed before admission')
        if kind == 'raw' and receipt.get('historical_masks', {}).get('status') != 'applied':
            raise ValueError('raw candidate historical masks remain unreconciled')
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('candidate cache integrity check failed')
        decoder = zstandard.ZstdDecompressor()
        requests, sequences = 0, 0
        if kind == 'raw':
            actual_counts = dict(connection.execute('SELECT status,COUNT(*) FROM sources GROUP BY status'))
            if actual_counts != receipt['source_counts'] or set(actual_counts) - {'candidate', 'unsupported', 'rejected'}:
                raise ValueError('raw candidate source classifications changed')
            for game, split, expected_requests, encoded, mask_json in connection.execute("SELECT game,split,requests,payload,mask_receipt FROM sources WHERE status='candidate' ORDER BY sha"):
                check()
                masks = orjson.loads(mask_json)
                if masks.get('status') != 'applied' or masks.get('effective_split') != split:
                    raise ValueError('raw candidate mask receipt is missing or inconsistent')
                members = [sequence_from_payload(value) for value in orjson.loads(decoder.decompress(encoded))]
                if len({sequence.actor for sequence in members}) != len(members) or sum(len(sequence.predictions) for sequence in members) != expected_requests or any(sequence.game_id != game or sequence.split != split for sequence in members):
                    raise ValueError('raw candidate sequence membership changed')
                for sequence in members:
                    validate_input_contract(sequence, QUERY_INPUT_CONTRACT)
                    requests += len(sequence.predictions)
                    sequences += 1
                    yield sequence
            if requests != receipt['candidate_requests']:
                raise ValueError('raw candidate request count changed')
        else:
            if connection.execute('SELECT COUNT(*) FROM games').fetchone()[0] != receipt['games']:
                raise ValueError('candidate game population changed')
            query = 'SELECT s.game,s.actor,s.events,s.records,s.eligible,s.payload,g.split FROM sequences s JOIN games g USING(game) ORDER BY s.game,s.actor'
            for game, actor, events, records, eligible, encoded, split in connection.execute(query):
                check()
                sequence = sequence_from_payload(orjson.loads(decoder.decompress(encoded)))
                if (sequence.game_id, sequence.actor, len(sequence.events), len(sequence.records), sum(row.loss_eligible for row in sequence.predictions), sequence.split) != (game, actor, events, records, eligible, split):
                    raise ValueError('candidate sequence membership changed')
                validate_input_contract(sequence, QUERY_INPUT_CONTRACT)
                requests += len(sequence.predictions)
                sequences += 1
                yield sequence
            if requests != receipt['requests'] or sequences != receipt['sequences'] or sequences != connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0]:
                raise ValueError('candidate request or sequence population changed')
        verify_identity()
    finally:
        connection.close()


def prefix_signatures(sequence: ActorSequence) -> Iterator[tuple[str, str]]:
    """Hash every causal event and exact legal contract, excluding the target."""
    prefix = hashlib.sha256(b'full-actor-prefix-v1')
    cursor = 0
    for prediction in sequence.predictions:
        while cursor <= prediction.event_index:
            encoded = orjson.dumps(sequence.records[sequence.events[cursor]])
            prefix.update(len(encoded).to_bytes(8, 'big'))
            prefix.update(encoded)
            cursor += 1
        legal = [{'options': [sequence.records[index] for index in stage.options], 'minimum': stage.minimum, 'maximum': stage.maximum, 'ordered': stage.ordered, 'repeated': stage.repeated} for stage in prediction.stages]
        signature = prefix.copy()
        signature.update(orjson.dumps(legal))
        yield prediction.row_id, signature.hexdigest()


def prepare_membership(connection: sqlite3.Connection, sequences: Iterable[ActorSequence], check_resources: Callable[[], None], after_ingestion: Callable[[], object] | None = None) -> dict[str, object]:
    """Caller supplies verified candidates; this gate preserves masks and splits.

    No input cap, window truncation or state-only deduplication. Whole identical
    actor histories are deduplicated; repeated prefixes in distinct games remain
    examples unless reserved-split protection applies.
    """
    connection.executescript('CREATE TABLE sequences (ordinal INTEGER PRIMARY KEY,game TEXT NOT NULL,actor INTEGER NOT NULL,family TEXT NOT NULL,split TEXT NOT NULL,rank INTEGER NOT NULL,content TEXT NOT NULL,payload BLOB NOT NULL); CREATE TABLE predictions (sequence_id INTEGER NOT NULL,row_id TEXT NOT NULL,position INTEGER NOT NULL,signature TEXT NOT NULL,source_eligible INTEGER NOT NULL,eligible INTEGER NOT NULL DEFAULT 0,reason TEXT NOT NULL DEFAULT \'pending\',PRIMARY KEY(sequence_id,row_id),UNIQUE(sequence_id,position)); CREATE TABLE reservations (signature TEXT PRIMARY KEY,rank INTEGER NOT NULL) WITHOUT ROWID;')
    encoder = zstandard.ZstdCompressor(level=3)
    count = 0
    # Publication is admitted only after the entire database is validated and
    # sealed. Bound transactions without fsyncing each actor; interrupted batches
    # roll back and the partial attempt remains unadmitted, never resumable.
    with connection:
        for ordinal, sequence in enumerate(sequences):
            check_resources()
            if not sequence.history_available or sequence.split not in SPLIT_PRIORITY:
                raise ValueError('prepared membership requires causal history and a known original split')
            # Native dataclass serialization preserves the exact JSON bytes
            # without recursively copying every semantic leaf through asdict.
            serialized = orjson.dumps(sequence)
            sequence_from_payload(orjson.loads(serialized))
            rank = SPLIT_PRIORITY[sequence.split]
            semantic = {'records': sequence.records, 'events': sequence.events, 'targets': [{'event': row.event_index, 'stages': row.stages} for row in sequence.predictions]}
            content = digest(orjson.dumps(semantic))
            signatures = dict(prefix_signatures(sequence))
            connection.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?,?,?)', (ordinal, sequence.game_id, sequence.actor, sequence.source_family, sequence.split, rank, content, encoder.compress(serialized)))
            for position, prediction in enumerate(sequence.predictions):
                signature = signatures[prediction.row_id]
                connection.execute('INSERT INTO predictions(sequence_id,row_id,position,signature,source_eligible) VALUES (?,?,?,?,?)', (ordinal, prediction.row_id, position, signature, int(prediction.loss_eligible)))
            count += 1
            if count % 64 == 0:
                connection.commit()
    if not count:
        raise ValueError('empty prepared corpus')
    if after_ingestion is not None:
        after_ingestion()
    return finalize_membership(connection, check_resources)


def finalize_membership(connection: sqlite3.Connection, check_resources: Callable[[], None]) -> dict[str, object]:
    """Apply unchanged admission rules to a complete, validated ingestion stage."""
    with guarded_sql(connection, check_resources):
        return _finalize_membership(connection, check_resources)


@contextmanager
def guarded_sql(connection: sqlite3.Connection, check_resources: Callable[[], None]) -> Iterator[None]:
    """Check owner/resource limits inside long SQL, retaining the original error."""
    interrupted: BaseException | None = None
    last_checked = -float('inf')

    def progress() -> int:
        nonlocal interrupted, last_checked
        now = time.monotonic()
        if now - last_checked < 1:
            return 0
        last_checked = now
        try:
            check_resources()
        except BaseException as error:
            interrupted = error
            return 1
        return 0

    connection.set_progress_handler(progress, 10000)
    try:
        yield
    except sqlite3.OperationalError:
        if interrupted is not None:
            raise interrupted
        raise
    finally:
        connection.set_progress_handler(None, 0)


def _finalize_membership(connection: sqlite3.Connection, check_resources: Callable[[], None]) -> dict[str, object]:
    count = connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0]
    if not count or connection.execute('SELECT 1 FROM reservations LIMIT 1').fetchone():
        raise ValueError('global admission requires nonempty ingestion and empty reservations')
    # Keep the rank beside its signature: one B-tree probe replaces a separate
    # random rowid-table fetch for every eligible historical decision.
    schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='reservations'").fetchone()[0]
    if 'WITHOUT ROWID' not in schema.upper():
        connection.execute('BEGIN')
        connection.execute('DROP TABLE reservations')  # Verified empty above.
        connection.execute('CREATE TABLE reservations (signature TEXT PRIMARY KEY,rank INTEGER NOT NULL) WITHOUT ROWID')
        connection.commit()
    connection.execute('PRAGMA cache_size=-131072')  # Bounded 128 MiB page cache.
    check_resources()
    connection.executescript('CREATE INDEX IF NOT EXISTS sequence_contents ON sequences(content,rank DESC,ordinal); CREATE INDEX IF NOT EXISTS sequence_games ON sequences(game,rank); CREATE INDEX IF NOT EXISTS prediction_signatures ON predictions(signature,sequence_id);')
    if connection.execute('SELECT game,actor FROM sequences GROUP BY game,actor HAVING COUNT(DISTINCT content)>1 LIMIT 1').fetchone():
        raise ValueError('one native actor has conflicting recovered histories')
    # Whole source games cannot straddle original split assignments, including
    # the case where their two actor histories differ in information content.
    with connection:
        connection.execute('CREATE INDEX IF NOT EXISTS prediction_coordinates ON predictions(row_id,source_eligible)')
        # The same physical source target may not bypass its historical mask by
        # arriving through another representation. Distinct verified coordinates
        # in identical histories may still contribute their eligible targets.
        connection.execute('UPDATE predictions SET source_eligible=0 WHERE source_eligible=1 AND row_id IN (SELECT row_id FROM predictions WHERE source_eligible=0)')
        connection.execute('UPDATE sequences SET rank=(SELECT MAX(other.rank) FROM sequences other WHERE other.game=sequences.game) WHERE rank<(SELECT MAX(other.rank) FROM sequences other WHERE other.game=sequences.game)')
        for name, rank in SPLIT_PRIORITY.items():
            connection.execute('UPDATE sequences SET split=? WHERE rank=? AND split<>?', (name, rank, name))
        # Keep per-prediction metadata probes out of the large payload table.
        connection.execute('CREATE INDEX sequence_metadata ON sequences(ordinal,content,rank)')
        connection.execute('CREATE INDEX prediction_eligible_positions ON predictions(sequence_id,position) WHERE source_eligible=1')
        # Whole-game propagation never lowers a source rank. Build the final
        # reservations once from every prediction, including masked histories,
        # instead of random per-row upserts followed by the same global maximum.
        connection.execute('INSERT INTO reservations SELECT p.signature,MAX(s.rank) FROM predictions p JOIN sequences s INDEXED BY sequence_metadata ON s.ordinal=p.sequence_id GROUP BY p.signature')
        connection.execute("UPDATE predictions SET reason=CASE WHEN source_eligible=0 THEN 'source_mask' WHEN (SELECT rank FROM reservations WHERE signature=predictions.signature)>(SELECT rank FROM sequences INDEXED BY sequence_metadata WHERE ordinal=sequence_id) THEN 'reserved_causal_prefix' WHEN sequence_id!=(SELECT s2.ordinal FROM sequences s2 INDEXED BY sequence_contents WHERE s2.content=(SELECT content FROM sequences INDEXED BY sequence_metadata WHERE ordinal=predictions.sequence_id) AND EXISTS(SELECT 1 FROM predictions p2 INDEXED BY prediction_eligible_positions WHERE p2.sequence_id=s2.ordinal AND p2.position=predictions.position AND p2.source_eligible=1) ORDER BY s2.rank DESC,s2.ordinal LIMIT 1) THEN 'duplicate_actor_history' ELSE 'admitted' END")
        connection.execute("UPDATE predictions SET eligible=(reason='admitted') WHERE eligible!=(reason='admitted')")
    # The source coordinate is never allowed to acquire competing trainable
    # meanings merely because it appeared in more than one candidate source.
    if connection.execute('SELECT row_id FROM predictions WHERE eligible=1 GROUP BY row_id HAVING COUNT(*)>1 LIMIT 1').fetchone():
        raise ValueError('conflicting trainable source coordinates')
    if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise ValueError('prepared membership database failed integrity check')
    return {'actor_sequences': count, 'reasons': dict(connection.execute('SELECT reason,COUNT(*) FROM predictions GROUP BY reason')), 'splits': connection.execute('SELECT s.split,COUNT(*),SUM(p.eligible) FROM predictions p JOIN sequences s ON s.ordinal=p.sequence_id GROUP BY s.split').fetchall(), 'deduplication': 'complete identical actor histories; different causal histories at the same board state are retained', 'reserved_protection': 'original source masks, whole native game split, exact causal prefix', 'arbitrary_decision_cap': None}


LOADER_WINDOW_PER_WORKER = 8


def admitted_sequence(connection: sqlite3.Connection, decoder: zstandard.ZstdDecompressor, input_contract: str, ordinal: int, stored_split: str, content: str, encoded: bytes) -> ActorSequence:
    """Decode one stored sequence and re-derive its admitted targets from frozen membership."""
    sequence = sequence_from_payload(orjson.loads(decoder.decompress(encoded)))
    validate_input_contract(sequence, input_contract)
    winners = connection.execute('SELECT p.position,p.row_id,p.sequence_id FROM predictions p JOIN sequences s ON s.ordinal=p.sequence_id WHERE s.content=? AND s.split=? AND p.eligible=1 ORDER BY p.position', (content, stored_split)).fetchall()
    admitted: dict[int, str] = {}
    sources = {ordinal: sequence}
    for position, row_id, owner in winners:
        if position in admitted:
            raise ValueError('duplicate admitted temporal target')
        if owner not in sources:
            source = connection.execute('SELECT payload FROM sequences WHERE ordinal=?', (owner,)).fetchone()[0]
            sources[owner] = sequence_from_payload(orjson.loads(decoder.decompress(source)))
        original = sources[owner].predictions[position]
        if original.row_id != row_id or not original.loss_eligible:
            raise ValueError('prepared membership re-enabled a source-masked target')
        admitted[position] = row_id
    return replace(sequence, split=stored_split, predictions=tuple(replace(row, row_id=admitted.get(position, row.row_id), loss_eligible=position in admitted) for position, row in enumerate(sequence.predictions)))


_loader_state: tuple[sqlite3.Connection, zstandard.ZstdDecompressor, str] | None = None


def _loader_worker_start(uri: str, input_contract: str) -> None:
    global _loader_state
    _loader_state = (sqlite3.connect(uri, uri=True), zstandard.ZstdDecompressor(), input_contract)


def _loader_worker_admit(task: int | tuple[int, str, str, bytes]) -> ActorSequence:
    if _loader_state is None:
        raise RuntimeError('loader worker was not initialized')
    connection, decoder, input_contract = _loader_state
    row = connection.execute('SELECT ordinal,split,content,payload FROM sequences WHERE ordinal=?', (task,)).fetchone() if isinstance(task, int) else task
    if row is None:
        raise ValueError('admitted ordinal disappeared from frozen membership')
    return admitted_sequence(connection, decoder, input_contract, *row)


class PreparedCorpus:
    def __init__(self, root: Path) -> None:
        self.root = root
        manifest = verify_run(root)
        if manifest.get('status') != 'completed' or manifest.get('kind') != 'v2_prepared_temporal_corpus' or manifest.get('training_admitted') is not True:
            raise ValueError('training requires a sealed admitted temporal corpus')
        self.parser_hashes = binding_hashes()
        if manifest.get('parser_hashes') != self.parser_hashes:
            raise ValueError('prepared corpus parser identity changed')
        if manifest.get('input_contract') not in INPUT_CONTRACTS:
            raise ValueError('prepared corpus requires an explicit supported input contract')
        self.input_contract = manifest['input_contract']
        self.manifest = manifest
        self.connection = sqlite3.connect((root / 'membership.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        self.decoder = zstandard.ZstdDecompressor()

    def sequences(self, split: str, order_seed: int | None = None, workers: int = 0) -> Iterator[ActorSequence]:
        """Yield every admitted sequence of a split exactly once.

        ``order_seed=None`` keeps the canonical ordinal order. An integer seed
        yields the same admitted population in a deterministic permutation, so
        source-blocked ordinals do not become source-blocked optimizer phases
        and an exact resume can replay the identical order from its cursor.
        ``workers`` > 0 decodes in that many processes with identical output.
        """
        if split not in SPLIT_PRIORITY:
            raise ValueError('unknown prepared split')
        if type(workers) is not int or not 0 <= workers <= 32:
            raise ValueError('sequence loader workers must be an integer in [0, 32]')
        admitted = 'FROM sequences s WHERE s.split=? AND s.ordinal=(SELECT MIN(other.ordinal) FROM sequences other WHERE other.content=s.content AND other.rank=s.rank) AND EXISTS (SELECT 1 FROM predictions p JOIN sequences owner ON owner.ordinal=p.sequence_id WHERE owner.content=s.content AND owner.rank=s.rank AND p.eligible=1)'
        if order_seed is None:
            rows: Iterable[tuple[int, str, str, bytes]] = self.connection.execute(f'SELECT s.ordinal,s.split,s.content,s.payload {admitted} ORDER BY s.ordinal', (split,))
        else:
            if type(order_seed) is not int:
                raise ValueError('sequence order seed must be an integer')
            ordinals = [row[0] for row in self.connection.execute(f'SELECT s.ordinal {admitted} ORDER BY s.ordinal', (split,))]
            random.Random(order_seed).shuffle(ordinals)
            rows = (self.connection.execute('SELECT ordinal,split,content,payload FROM sequences WHERE ordinal=?', (ordinal,)).fetchone() for ordinal in ordinals)
        if workers == 0:
            for row in rows:
                yield admitted_sequence(self.connection, self.decoder, self.input_contract, *row)
            return
        # Workers fetch shuffled rows themselves so random reads overlap; the
        # canonical cursor streams sequentially and hands rows over. Results are
        # consumed strictly in submission order through a bounded window, so the
        # yielded order and content equal the serial path exactly.
        tasks: Iterable[int | tuple[int, str, str, bytes]] = ordinals if order_seed is not None else rows
        context = multiprocessing.get_context('spawn')
        pool = context.Pool(workers, initializer=_loader_worker_start, initargs=((self.root / 'membership.sqlite3').resolve().as_uri() + '?mode=ro', self.input_contract))
        try:
            window: deque[multiprocessing.pool.AsyncResult] = deque()
            for task in tasks:
                window.append(pool.apply_async(_loader_worker_admit, (task,)))
                if len(window) >= workers * LOADER_WINDOW_PER_WORKER:
                    yield window.popleft().get()
            while window:
                yield window.popleft().get()
            pool.close()
            pool.join()
        finally:
            pool.terminate()

    def expected_rows(self, split: str, check: Callable[[], None] = lambda: None) -> Iterator[tuple[str, str, str, int]]:
        """(decision id, game, source family, sequence ordinal) for every admitted target of ``split``.

        The same admitted-sequence predicate and winner join as ``sequences``,
        evaluated in frozen membership SQL without decoding any payload.
        """
        if split not in SPLIT_PRIORITY:
            raise ValueError('unknown prepared split')
        query = ('WITH admitted AS (SELECT s.ordinal, s.content, s.split, s.game, s.family FROM sequences s WHERE s.split=? '
                 'AND s.ordinal=(SELECT MIN(other.ordinal) FROM sequences other WHERE other.content=s.content AND other.rank=s.rank) '
                 'AND EXISTS (SELECT 1 FROM predictions p JOIN sequences owner ON owner.ordinal=p.sequence_id WHERE owner.content=s.content AND owner.rank=s.rank AND p.eligible=1)) '
                 'SELECT p.row_id, a.game, a.family, a.ordinal, p.position FROM admitted a JOIN sequences owner ON owner.content=a.content AND owner.split=a.split '
                 'JOIN predictions p ON p.sequence_id=owner.ordinal AND p.eligible=1')
        positions: set[tuple[int, int]] = set()
        for count, (row_id, game, family, ordinal, position) in enumerate(self.connection.execute(query, (split,))):
            if count % 65536 == 0:
                check()
            if (ordinal, position) in positions:
                raise ValueError('duplicate admitted temporal target')
            positions.add((ordinal, position))
            yield row_id, game, family, ordinal

    def close(self) -> None:
        self.connection.close()
