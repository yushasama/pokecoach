"""Recover source factory histories absent from the older canonical catalog."""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import orjson

from deepmew_ex.datasets.v2.corpus.records import REFERENCE_SCHEMA, SEMANTIC_FIELDS, action_envelope
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.recovery import RecoveredGame, recover_reference_game
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.records import validate_hash
from deepmew_ex.registry.store import canonical, digest


@dataclass
class SourceStreamAudit:
    compressed_bytes: int = 0
    decoded_bytes: int = 0
    complete_json_rows: int = 0
    container_complete: bool = False
    incomplete_tail_bytes: int = 0
    incomplete_tail_sha256: str | None = None


def factory_rows(path: Path, audit: SourceStreamAudit, check: Callable[[], None], allow_truncated: bool = False) -> Iterator[tuple[int, Payload]]:
    """Retain complete JSON objects from a frozen interrupted gzip stream.

    Missing trailers are explicit evidence gaps. Invalid compression/checksums,
    malformed interior JSON and non-object rows still fail. The caller verifies
    original compressed bytes and complete per-game terminal/action contracts.
    """
    decoder = zlib.decompressobj(31) if path.suffix == '.gz' else None
    pending = b''
    with path.open('rb') as handle:
        while encoded := handle.read(65536):
            check()
            audit.compressed_bytes += len(encoded)
            while encoded:
                if decoder is None:
                    decoded, encoded = encoded, b''
                else:
                    if decoder.eof:
                        decoder = zlib.decompressobj(31)
                    decoded = decoder.decompress(encoded)
                    encoded = decoder.unused_data
                audit.decoded_bytes += len(decoded)
                pending += decoded
                lines = pending.split(b'\n')
                pending = lines.pop()
                for line in lines:
                    row = orjson.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('factory JSON row is not an object')
                    audit.complete_json_rows += 1
                    yield audit.complete_json_rows, row
    audit.container_complete = decoder is None or decoder.eof
    if pending:
        try:
            row = orjson.loads(pending)
        except orjson.JSONDecodeError:
            if audit.container_complete or not allow_truncated:
                raise
            audit.incomplete_tail_bytes = len(pending)
            audit.incomplete_tail_sha256 = digest(pending)
        else:
            if not isinstance(row, dict):
                raise ValueError('factory JSON tail is not an object')
            audit.complete_json_rows += 1
            yield audit.complete_json_rows, row
    if not audit.container_complete and not allow_truncated:
        raise EOFError('factory compressed stream lacks a complete trailer')


def factory_games(path: Path, audit: SourceStreamAudit, check: Callable[[], None], allow_truncated: bool = False) -> Iterator[tuple[int, list[Payload]]]:
    rows: list[Payload] = []
    first = 1
    for line, row in factory_rows(path, audit, check, allow_truncated):
        if not isinstance(row.get('game_id'), str):
            raise ValueError('factory source has no exact native game identity')
        if rows and row['game_id'] != rows[0]['game_id']:
            yield first, rows
            rows, first = [], line
        rows.append(row)
    if rows:
        yield first, rows


def recover_factory_source_game(rows: list[Payload], summary: Payload, source: Path, source_sha256: str, first_line: int, split: str, target_eligible: tuple[bool, ...]) -> RecoveredGame:
    """Caller verifies the whole archive, summary and historical split authority.

    These references are constructed from the source itself, not represented as
    previously canonicalized evidence. Existing semantic legal/action checks and
    full global continuity validation still apply. Native labels are scoped by
    immutable source identity because interrupted generations reused those labels.
    """
    validate_hash(source_sha256)
    if split not in SPLIT_PRIORITY or type(first_line) is not int or first_line < 1 or len(target_eligible) != len(rows) or any(type(value) is not bool for value in target_eligible):
        raise ValueError('factory recovery requires exact source coordinates, masks and explicit split')
    orders = [0, 0]
    frames: list[Payload] = []
    for index, row in enumerate(rows):
        seat = row.get('seat')
        legal, observation = row.get('legal_set'), row.get('raw_observation')
        if type(seat) is not int or seat not in (0, 1) or not isinstance(observation, dict) or not isinstance(legal, list) or not legal or any(not isinstance(option, dict) or set(option) - SEMANTIC_FIELDS for option in legal):
            raise ValueError('factory source lacks exact actor observation/legal semantics')
        chosen = row.get('chosen_action')
        if not isinstance(chosen, dict):
            raise ValueError('factory source lacks exact selected action')
        native = row.get('game_id')
        game = 'factory:' + digest(canonical([source_sha256, native, row.get('game_index')]))
        coordinate = [source_sha256, first_line + index, seat, index]
        frame = {'schema': REFERENCE_SCHEMA, 'source_schema': row.get('schema_version'), 'frame_id': 'factory-frame:' + digest(canonical(coordinate)), 'episode_id': game, 'source_family': 'unified_r2_source', 'effective_split': split, 'native_game_binding': {'raw_game_id': native, 'source_game_index': row.get('game_index')}, 'global_order': index, 'actor_seat': seat, 'actor_order': orders[seat], 'state_before': {'source_path': str(source), 'source_sha256': source_sha256, 'source_line': first_line + index, 'source_field': 'raw_observation', 'content_sha256': hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()}, 'legal_actions': legal, 'chosen_action': action_envelope(legal, chosen), 'target_mask': {'source_target_eligible': target_eligible[index]}, 'terminal_target': .5 if summary.get('result') == 2 else float(summary.get('result') == seat), 'action_result': {'state_after_reference': None}}
        frames.append(frame)
        orders[seat] += 1
    recovered = recover_reference_game(frames, rows, summary)
    game = frames[0]['episode_id']
    return replace(recovered, game_id=game, sequences=tuple(replace(sequence, game_id=game) for sequence in recovered.sequences))
