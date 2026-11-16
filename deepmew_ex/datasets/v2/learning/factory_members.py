"""Verify independent gzip members without concealing rejected source byte ranges."""

from __future__ import annotations

import hashlib
import tempfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class GzipMember:
    start: int
    end: int
    sha256: str
    decoded_bytes: int
    stream: BinaryIO


def member_at(path: Path, start: int, spool_root: Path, check: Callable[[], None]) -> GzipMember:
    """No decoded data escapes until the gzip trailer and CRC are verified."""
    stream = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode='w+b', dir=spool_root)
    decoder = zlib.decompressobj(31)
    sha = hashlib.sha256()
    consumed, decoded_bytes = 0, 0
    try:
        with path.open('rb') as handle:
            handle.seek(start)
            encoded = b''
            while not decoder.eof:
                check()
                if not encoded:
                    encoded = handle.read(65536)
                if not encoded:
                    raise EOFError('gzip member lacks a complete trailer')
                decoded = decoder.decompress(encoded, 1024 * 1024)
                remainder = decoder.unused_data if decoder.eof else decoder.unconsumed_tail
                size = len(encoded) - len(remainder)
                sha.update(encoded[:size])
                consumed += size
                encoded = remainder
                stream.write(decoded)
                decoded_bytes += len(decoded)
        stream.seek(0)
        return GzipMember(start, start + consumed, sha.hexdigest(), decoded_bytes, stream)
    except BaseException:
        stream.close()
        raise


def header_offsets(path: Path, start: int, check: Callable[[], None]) -> Iterator[int]:
    """Candidates alone confer no validity; each is independently CRC checked."""
    with path.open('rb') as handle:
        handle.seek(start)
        prefix = b''
        offset = start
        while chunk := handle.read(1024 * 1024):
            check()
            block = prefix + chunk
            base = offset - len(prefix)
            position = 0
            while (position := block.find(b'\x1f\x8b\x08', position)) >= 0:
                yield base + position
                position += 3
            prefix = block[-2:]
            offset += len(chunk)


def span_hash(path: Path, start: int, end: int, check: Callable[[], None]) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            check()
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError('source shortened while hashing rejected span')
            sha.update(block)
            remaining -= len(block)
    return sha.hexdigest()


def verified_members(path: Path, spool_root: Path, check: Callable[[], None], report: Callable[[dict[str, object]], None], start: int = 0) -> Iterator[GzipMember]:
    """Partition source bytes into verified members and explicit rejected spans.

    A verified member still needs JSON, complete-game, source identity, action,
    observer, terminal and split checks before it can become a candidate.
    The consumer must finish reading a yielded member before advancing.
    """
    size = path.stat().st_size
    if type(start) is not int or not 0 <= start <= size:
        raise ValueError('invalid member start offset')
    cursor = start
    while cursor < size:
        check()
        member = None
        try:
            member = member_at(path, cursor, spool_root, check)
        except (zlib.error, EOFError) as error:
            first_error = {'type': type(error).__name__, 'message': str(error)}
            failed_candidates = 0
            for offset in header_offsets(path, cursor + 1, check):
                try:
                    member = member_at(path, offset, spool_root, check)
                    break
                except (zlib.error, EOFError):
                    failed_candidates += 1
            end = size if member is None else member.start
            report({'status': 'rejected_gzip_span', 'start': cursor, 'end': end, 'sha256': span_hash(path, cursor, end, check), 'error': first_error, 'failed_header_candidates': failed_candidates})
            if member is None:
                break
        try:
            yield member
            cursor = member.end
        finally:
            member.stream.close()
    if path.stat().st_size != size:
        raise ValueError('source size changed during gzip member traversal')
