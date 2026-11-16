"""Small atomic-ish writers for generated evidence files."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


ATOMIC_REPLACE_ATTEMPTS = 8  # Bounded retry for transient Windows sharing violations.
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05
ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.4


def _replace_with_retry(source: str, target: Path) -> None:
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)

            return
        except PermissionError:
            if attempt + 1 == ATOMIC_REPLACE_ATTEMPTS:
                raise
            delay = min(
                ATOMIC_REPLACE_INITIAL_DELAY_SECONDS * (2 ** attempt),
                ATOMIC_REPLACE_MAX_DELAY_SECONDS,
            )
            time.sleep(delay)


def _write_text_atomic(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_name = handle.name
    try:
        with handle:
            handle.write(text)
        _replace_with_retry(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def write_json(path: Path, data: Any) -> Path:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    return _write_text_atomic(Path(path), text)


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    text = "\n".join(lines)
    if text:
        text += "\n"
    return _write_text_atomic(Path(path), text)


def write_text(path: Path, text: str) -> Path:
    return _write_text_atomic(Path(path), text)


def write_markdown(path: Path, text: str) -> Path:
    if text and not text.endswith("\n"):
        text += "\n"
    return _write_text_atomic(Path(path), text)
