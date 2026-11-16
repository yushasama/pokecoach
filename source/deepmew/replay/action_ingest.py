"""Flatten `deepmew.decision_frame.v1` trace frames into raw replay-action rows.

Reads existing `frames.jsonl` (no game replay). `state_json` is the frame's raw
observation; `legal_options_json` is that observation's raw `select.option` list.
When a frame has no raw observation, those fields are left null and counted, never
fabricated. `deck_id`/`deck_hash` reuse the per-game deck metadata helper.
`source_rating` is null for arena/trace-origin rows (no ladder rating).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from deepmew.replay.action_schema import (
    SCHEMA_VERSION,
    ReplayActionSchemaError,
    validate_action_row,
)
from deepmew.replay.decision_ingest import (
    FRAMES_FILENAME,
    deck_metadata_from_path,
    resolve_frames_path,
)
from deepmew.trace.schema import DECISION_FRAME_SCHEMA_VERSION

DEFAULT_SOURCE_KIND = "trace"

DeckResolver = Callable[[str | None], "tuple[str | None, str | None]"]


class ReplayActionIngestError(ValueError):
    """Raised when a trace artifact cannot be ingested into action rows."""


@dataclass(frozen=True)
class ActionRowResult:
    row: dict[str, Any] | None
    malformed: bool
    missing_state: bool
    missing_legal: bool
    error: str | None = None


@dataclass(frozen=True)
class ActionIngestResult:
    rows: list[dict[str, Any]]
    source_kind: str
    source_run_id: str
    source_path: str
    frame_count: int
    malformed_count: int
    missing_state_count: int
    missing_legal_options_count: int

    @property
    def row_count(self) -> int:
        return len(self.rows)


def derive_source(frames_path: Path) -> tuple[str, str]:
    """Return (source_kind, source_run_id) from `.../<kind>/<date>/<run_id>/frames.jsonl`."""
    resolved = frames_path.resolve()
    source_run_id = resolved.parent.name
    parts = resolved.parts
    source_kind = DEFAULT_SOURCE_KIND
    if "runs" in parts:
        runs_index = parts.index("runs")
        if runs_index + 1 < len(parts):
            source_kind = parts[runs_index + 1]
    return source_kind, source_run_id


def iter_raw_frames(frames_path: Path) -> Iterator[dict[str, Any]]:
    """Stream-read decision frames, validating the frame schema version."""
    if not frames_path.is_file():
        raise ReplayActionIngestError(f"{FRAMES_FILENAME} not found: {frames_path}")
    with frames_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayActionIngestError(f"{frames_path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ReplayActionIngestError(f"{frames_path}:{line_number}: frame must be a JSON object")
            schema = payload.get("schema_version")
            if schema != DECISION_FRAME_SCHEMA_VERSION:
                raise ReplayActionIngestError(
                    f"{frames_path}:{line_number}: unsupported frame schema_version {schema!r}; "
                    f"expected {DECISION_FRAME_SCHEMA_VERSION!r}"
                )
            yield payload


def _raw_legal_options(state_json: Any) -> list[Any] | None:
    if not isinstance(state_json, dict):
        return None
    select = state_json.get("select")
    if not isinstance(select, dict):
        return None
    options = select.get("option")
    return options if isinstance(options, list) else None


def build_action_row(
    frame: dict[str, Any],
    *,
    source_kind: str,
    source_run_id: str,
    source_path: str,
    deck_id: str | None,
    deck_hash: str | None,
    index: int | None = None,
) -> ActionRowResult:
    """Build and validate one raw action row from a decision frame.

    Returns an ActionRowResult; on schema violation the row is None and
    `malformed` is True (the caller skips and counts it) rather than raising.
    """
    raw_observation = frame.get("raw_observation")
    state_json = raw_observation if isinstance(raw_observation, dict) else None
    legal_options_json = _raw_legal_options(state_json)

    row = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_run_id": source_run_id,
        "source_path": source_path,
        "game_id": frame.get("game_id"),
        "player_id": frame.get("seat"),
        "turn": frame.get("turn"),
        "step": frame.get("decision_index"),
        "frame_id": frame.get("frame_id"),
        "seed": frame.get("seed"),
        "state_json": state_json,
        "legal_options_json": legal_options_json,
        "chosen_indices": list(frame.get("chosen_indices") or []),
        "result": frame.get("final_result"),
        "result_reason": frame.get("final_result_reason"),
        "deck_id": deck_id,
        "deck_hash": deck_hash,
        "source_rating": None,
    }
    try:
        validate_action_row(row, index=index)
    except ReplayActionSchemaError as exc:
        return ActionRowResult(row=None, malformed=True, missing_state=False, missing_legal=False, error=str(exc))
    return ActionRowResult(
        row=row,
        malformed=False,
        missing_state=state_json is None,
        missing_legal=legal_options_json is None,
    )


def iter_action_rows(
    frames_path: str | Path,
    *,
    deck_resolver: DeckResolver = deck_metadata_from_path,
    deck_cache: dict[str | None, tuple[str | None, str | None]] | None = None,
) -> Iterator[ActionRowResult]:
    """Stream ActionRowResults for one trace run (frames.jsonl file or run dir)."""
    resolved = resolve_frames_path(frames_path)
    source_kind, source_run_id = derive_source(resolved)
    source_path = str(resolved)
    cache: dict[str | None, tuple[str | None, str | None]] = deck_cache if deck_cache is not None else {}
    index = 0
    for frame in iter_raw_frames(resolved):
        deck_path = frame.get("deck_path")
        if deck_path not in cache:
            cache[deck_path] = deck_resolver(deck_path)
        deck_id, deck_hash = cache[deck_path]
        yield build_action_row(
            frame,
            source_kind=source_kind,
            source_run_id=source_run_id,
            source_path=source_path,
            deck_id=deck_id,
            deck_hash=deck_hash,
            index=index,
        )
        index += 1


def ingest_frames(
    frames_path: str | Path,
    *,
    deck_resolver: DeckResolver = deck_metadata_from_path,
) -> ActionIngestResult:
    """Collect action rows for one trace run (small/test use; the big build streams)."""
    resolved = resolve_frames_path(frames_path)
    source_kind, source_run_id = derive_source(resolved)
    rows: list[dict[str, Any]] = []
    frame_count = malformed = missing_state = missing_legal = 0
    for result in iter_action_rows(resolved, deck_resolver=deck_resolver):
        frame_count += 1
        if result.malformed:
            malformed += 1
            continue
        rows.append(result.row)  # type: ignore[arg-type]
        if result.missing_state:
            missing_state += 1
        if result.missing_legal:
            missing_legal += 1
    return ActionIngestResult(
        rows=rows,
        source_kind=source_kind,
        source_run_id=source_run_id,
        source_path=str(resolved),
        frame_count=frame_count,
        malformed_count=malformed,
        missing_state_count=missing_state,
        missing_legal_options_count=missing_legal,
    )
