"""Recheck interrupted ingestion against its frozen, previously verified caches."""

from __future__ import annotations

import sqlite3
import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypedDict, cast

import orjson
import zstandard

from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.prepared import CandidateCache
from deepmew_ex.registry.store import digest, file_hash


class StagePayload(TypedDict):
    options: list[int]
    minimum: int
    maximum: int
    ordered: bool
    repeated: bool


class PredictionPayload(TypedDict):
    row_id: str
    event_index: int
    loss_eligible: bool
    stages: list[StagePayload]


class ActorPayload(TypedDict):
    game_id: str
    actor: int
    source_family: str
    split: str
    history_available: bool
    records: list[list[object]]
    events: list[int]
    predictions: list[PredictionPayload]
    unavailable_post_states: list[str]


def source_payloads(connection: sqlite3.Connection, raw: bool, decoder: zstandard.ZstdDecompressor) -> Iterator[bytes]:
    if raw:
        for (encoded,) in connection.execute("SELECT payload FROM sources WHERE status='candidate' ORDER BY sha"):
            for sequence in orjson.loads(decoder.decompress(encoded)):
                yield orjson.dumps(sequence)
        return
    for (encoded,) in connection.execute('SELECT s.payload FROM sequences s JOIN games g USING(game) ORDER BY s.game,s.actor'):
        yield decoder.decompress(encoded)


def payload_signatures(data: ActorPayload) -> Iterator[str]:
    """The canonical prefix_signatures byte contract without rebuilding dataclasses."""
    prefix = hashlib.sha256(b'full-actor-prefix-v1')
    cursor = 0
    for prediction in data['predictions']:
        while cursor <= prediction['event_index']:
            encoded = orjson.dumps(data['records'][data['events'][cursor]])
            prefix.update(len(encoded).to_bytes(8, 'big'))
            prefix.update(encoded)
            cursor += 1
        legal = [{'options': [data['records'][index] for index in stage['options']], 'minimum': stage['minimum'], 'maximum': stage['maximum'], 'ordered': stage['ordered'], 'repeated': stage['repeated']} for stage in prediction['stages']]
        signature = prefix.copy()
        signature.update(orjson.dumps(legal))
        yield signature.hexdigest()


def audit_ingested(connection: sqlite3.Connection, caches: tuple[CandidateCache, ...], check: Callable[[], None], progress: Callable[[dict[str, object]], None] | None = None) -> list[dict[str, object]]:
    """Require every committed payload, mask and coordinate to match the sources.

    This is recovery of a prior verified publisher, not a replacement candidate
    validator. The frozen hashes bind its completed cache validation. Global SQL
    never modifies payloads or coordinates. Content and causal signatures are
    independently recomputed; all mutable admission fields must be rolled back.
    The caller holds an exclusive transaction and has confirmed writer exit.
    """
    if not connection.in_transaction:
        raise ValueError('ingestion recovery requires an exclusive transaction')
    if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise ValueError('interrupted ingestion integrity check failed')
    if connection.execute('SELECT 1 FROM reservations LIMIT 1').fetchone():
        raise ValueError('admission transaction was already committed; ingestion recovery is inapplicable')
    actual = connection.execute('SELECT ordinal,game,actor,family,split,rank,content,payload FROM sequences ORDER BY ordinal')
    decoder = zstandard.ZstdDecompressor()
    coverage: list[dict[str, object]] = []
    ordinal, requests = 0, 0
    for cache in caches:
        check()
        wal = Path(str(cache.path) + '-wal')
        if (wal.exists() and wal.stat().st_size) or file_hash(cache.path) != cache.sha256:
            raise ValueError('recovery candidate cache is live or changed')
        source = sqlite3.connect(cache.path.resolve().as_uri() + '?mode=ro', uri=True)
        counts = {'actor_sequences': 0, 'requests': 0, 'source_eligible': 0, 'withheld_incomplete_history': 0, 'unavailable_state_receipts': 0}
        shapes: dict[str, dict[str, int]] = {}
        incomplete = cache.kind in ('recorded', 'recorded_group')
        try:
            completion = source.execute('SELECT receipt FROM completion').fetchall()
            identities = source.execute('SELECT identity FROM metadata').fetchall()
            if len(completion) != 1 or len(identities) != 1:
                raise ValueError('recovery candidate lacks one completed identity')
            receipt = orjson.loads(completion[0][0])
            if receipt.get('identity') != orjson.loads(identities[0][0]):
                raise ValueError('recovery candidate completion identity changed')
            for encoded in source_payloads(source, cache.kind == 'raw', decoder):
                check()
                data = cast(ActorPayload, orjson.loads(encoded))
                if data['history_available'] is not True or data['split'] not in SPLIT_PRIORITY or any(type(row['loss_eligible']) is not bool for row in data['predictions']):
                    raise ValueError('invalid recovered source history, split or mask')
                eligible = sum(row['loss_eligible'] for row in data['predictions'])
                counts['source_eligible'] += eligible
                if incomplete:
                    counts['withheld_incomplete_history'] += eligible
                    for prediction in data['predictions']:
                        prediction['loss_eligible'] = False
                    encoded = orjson.dumps(data)
                stored = actual.fetchone()
                expected = (ordinal, data['game_id'], data['actor'], data['source_family'], data['split'], SPLIT_PRIORITY[data['split']])
                if stored is None or stored[:6] != expected or decoder.decompress(stored[7]) != encoded:
                    raise ValueError('committed ingestion payload, order or original split differs from frozen cache')
                semantic = {'records': data['records'], 'events': data['events'], 'targets': [{'event': row['event_index'], 'stages': row['stages']} for row in data['predictions']]}
                if stored[6] != digest(orjson.dumps(semantic)):
                    raise ValueError('committed actor content hash changed')
                rows = connection.execute('SELECT row_id,position,source_eligible,eligible,reason,signature FROM predictions WHERE sequence_id=? ORDER BY position', (ordinal,)).fetchall()
                expected_rows = [(row['row_id'], position, int(row['loss_eligible']), 0, 'pending', signature) for position, (row, signature) in enumerate(zip(data['predictions'], payload_signatures(data), strict=True))]
                if rows != expected_rows:
                    raise ValueError('committed target coordinates or pre-admission masks changed')
                counts['actor_sequences'] += 1
                counts['requests'] += len(rows)
                counts['unavailable_state_receipts'] += len(data['unavailable_post_states'])
                shape = shapes.setdefault(data['split'], {'sequences': 0, 'events': 0, 'leaves': 0, 'max_events': 0, 'max_leaves': 0, 'max_records': 0})
                leaves = sum(map(len, data['records']))
                shape['sequences'] += 1
                shape['events'] += len(data['events'])
                shape['leaves'] += leaves
                shape['max_events'] = max(shape['max_events'], len(data['events']))
                shape['max_leaves'] = max(shape['max_leaves'], leaves)
                shape['max_records'] = max(shape['max_records'], len(data['records']))
                ordinal += 1
                requests += len(rows)
                if progress is not None and ordinal % 512 == 0:
                    progress({'actor_sequences': ordinal, 'requests': requests, 'completed_caches': len(coverage), 'current_cache': str(cache.path)})
            expected_requests = receipt['candidate_requests'] if cache.kind == 'raw' else receipt['requests']
            if counts['requests'] != expected_requests or cache.kind != 'raw' and counts['actor_sequences'] != receipt['sequences']:
                raise ValueError('recovery candidate population differs from completed receipt')
        finally:
            source.close()
        check()
        if (wal.exists() and wal.stat().st_size) or file_hash(cache.path) != cache.sha256:
            raise ValueError('recovery candidate changed during audit')
        coverage.append({'path': str(cache.path.resolve()), 'sha256': cache.sha256, 'kind': cache.kind, **counts, 'shapes_by_original_split': shapes, 'fitting': not incomplete, 'history_scope': 'incomplete recorded actor selections' if incomplete else 'complete recovered recorded request history; unrecorded engine microsteps not inferred'})
        if progress is not None:
            progress({'actor_sequences': ordinal, 'requests': requests, 'completed_caches': len(coverage), 'completed_cache': str(cache.path)})
    if not ordinal or actual.fetchone() is not None or connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] != requests:
        raise ValueError('recovery ingestion omitted or added source members')
    return coverage
