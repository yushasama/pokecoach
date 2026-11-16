from __future__ import annotations

import gzip
import hashlib
import zlib
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.learning.factory_members import member_at, verified_members


def interrupted(data: bytes) -> bytes:
    encoder = zlib.compressobj(wbits=31)
    return encoder.compress(data) + encoder.flush(zlib.Z_SYNC_FLUSH)


def test_complete_members_have_exact_original_ranges_and_crc(tmp_path: Path) -> None:
    blocks = [gzip.compress(b'{"game":1}\n'), gzip.compress(b'{"game":2}\n')]
    path = tmp_path / 'source.gz'
    path.write_bytes(b''.join(blocks))
    audit: list[dict[str, object]] = []
    members = [(m.start, m.end, m.sha256, m.stream.read()) for m in verified_members(path, tmp_path, lambda: None, audit.append)]
    assert members == [(0, len(blocks[0]), hashlib.sha256(blocks[0]).hexdigest(), b'{"game":1}\n'), (len(blocks[0]), sum(map(len, blocks)), hashlib.sha256(blocks[1]).hexdigest(), b'{"game":2}\n')]
    assert not audit


@pytest.mark.parametrize('defect', ['interrupted', 'checksum', 'deflate', 'false_header', 'padding'])
def test_invalid_member_never_leaks_rows_and_later_complete_game_survives(tmp_path: Path, defect: str) -> None:
    first = gzip.compress(b'{"game":1}\n')
    damaged = bytearray(gzip.compress(b'{"game":2,"partial":true}\n'))
    if defect == 'interrupted':
        bad = interrupted(b'{"game":2,"partial":true}\n')
    elif defect == 'checksum':
        damaged[-8] ^= 1
        bad = bytes(damaged)
    elif defect == 'deflate':
        damaged[10] = 7
        bad = bytes(damaged)
    elif defect == 'false_header':
        bad = b'garbage\x1f\x8b\x08\x00invalid fake member'
    else:
        bad = bytes(19)
    last = gzip.compress(b'{"game":2,"complete":true}\n')
    path = tmp_path / 'source.gz'
    path.write_bytes(first + bad + last)
    audit: list[dict[str, object]] = []
    members = [(m.start, m.end, m.stream.read()) for m in verified_members(path, tmp_path, lambda: None, audit.append)]
    assert [row[2] for row in members] == [b'{"game":1}\n', b'{"game":2,"complete":true}\n']
    assert members[1][:2] == (len(first + bad), len(first + bad + last))
    assert len(audit) == 1 and audit[0]['start'] == len(first) and audit[0]['end'] == len(first + bad)
    assert audit[0]['sha256'] == hashlib.sha256(bad).hexdigest()


def test_truncated_last_member_retained_only_as_audited_bytes(tmp_path: Path) -> None:
    path = tmp_path / 'source.gz'
    bad = interrupted(b'{"game":1}\n')
    path.write_bytes(bad)
    audit: list[dict[str, object]] = []
    assert list(verified_members(path, tmp_path, lambda: None, audit.append)) == []
    assert audit[0]['start'] == 0 and audit[0]['end'] == len(bad)


def test_large_decoded_member_spills_without_truncating_context(tmp_path: Path) -> None:
    payload = b'{"large":"' + b'x' * (10 * 1024 * 1024) + b'"}\n'
    path = tmp_path / 'source.gz'
    encoded = gzip.compress(payload)
    path.write_bytes(encoded)
    member = member_at(path, 0, tmp_path, lambda: None)
    try:
        assert member.decoded_bytes == len(payload) and member.end == len(encoded)
        assert hashlib.sha256(member.stream.read()).digest() == hashlib.sha256(payload).digest()
    finally:
        member.stream.close()


def test_resource_stop_is_not_misclassified_as_bad_source(tmp_path: Path) -> None:
    path = tmp_path / 'source.gz'
    path.write_bytes(gzip.compress(b'payload'))
    def stop() -> None:
        raise InterruptedError('approved boundary')
    with pytest.raises(InterruptedError, match='approved boundary'):
        list(verified_members(path, tmp_path, stop, lambda row: None))


def test_resume_from_verified_byte_boundary_keeps_original_coordinates(tmp_path: Path) -> None:
    first, second = gzip.compress(b'first'), gzip.compress(b'second')
    path = tmp_path / 'source.gz'
    path.write_bytes(first + second)
    result = [(m.start, m.end, m.stream.read()) for m in verified_members(path, tmp_path, lambda: None, lambda row: None, len(first))]
    assert result == [(len(first), len(first + second), b'second')]
