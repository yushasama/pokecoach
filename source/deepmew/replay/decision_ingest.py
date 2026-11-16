"""Flatten `deepmew.decision_frame.v1` trace frames into per-decision replay rows.

Reads existing `frames.jsonl` artifacts (no game replay, no engine
instrumentation). Derives `opponent_label` from the other seat in the same game,
and `deck_id`/`deck_hash` from the frame's `deck_path` using the shared deck
reader. Old pre-TD.0.10 frames that lack option-identity enrichment are ingested
with `chosen_resolved` marked conservatively, or skipped under `require_resolved`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from deepmew.deck.io import read_deck
from deepmew.replay.decision_schema import (
    CARD_BEARING_OPTION_TYPES,
    SCHEMA_VERSION,
    ReplayDecisionSchemaError,
    validate_decision_row,
)
from deepmew.trace.schema import DECISION_FRAME_SCHEMA_VERSION

FRAMES_FILENAME = "frames.jsonl"
DEFAULT_SOURCE_KIND = "trace"

# Deck metadata resolver signature: deck_path -> (deck_id, deck_hash).
DeckResolver = Callable[[str | None], "tuple[str | None, str | None]"]


class ReplayDecisionIngestError(ValueError):
    """Raised when a trace artifact cannot be ingested."""


@dataclass(frozen=True)
class DecisionIngestResult:
    rows: list[dict[str, Any]]
    source_kind: str
    source_run_id: str
    source_path: str
    frame_count: int
    skipped_unresolved: int
    pre_enrichment_frames: int

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class _DeckCache:
    resolver: DeckResolver
    _cache: dict[str | None, tuple[str | None, str | None]] = field(default_factory=dict)

    def get(self, deck_path: str | None) -> tuple[str | None, str | None]:
        if deck_path not in self._cache:
            self._cache[deck_path] = self.resolver(deck_path)
        return self._cache[deck_path]


def deck_metadata_from_path(deck_path: str | None) -> tuple[str | None, str | None]:
    """Return (deck_id, deck_hash) for a deck file, mirroring the mixture reporter.

    deck_hash is the first 16 hex chars of the sha256 over newline-joined card ids,
    matching `deepmew.opponents.reporting.deck_metadata`. Missing/unreadable files
    yield (deck_id-from-stem-or-None, None) rather than raising.
    """
    if not deck_path:
        return None, None
    path = Path(deck_path)
    deck_id = path.stem or None
    try:
        card_ids = read_deck(path)
    except (FileNotFoundError, OSError, ValueError):
        return deck_id, None
    normalized = "\n".join(str(card_id) for card_id in card_ids) + "\n"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return deck_id, digest[:16]


def resolve_frames_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        return p / FRAMES_FILENAME
    return p


def _derive_source(frames_path: Path) -> tuple[str, str]:
    """Return (source_kind, source_run_id) from a `.../<kind>/<date>/<run_id>/frames.jsonl` path."""
    resolved = frames_path.resolve()
    run_dir = resolved.parent
    source_run_id = run_dir.name
    parts = resolved.parts
    source_kind = DEFAULT_SOURCE_KIND
    if "runs" in parts:
        runs_index = parts.index("runs")
        if runs_index + 1 < len(parts):
            source_kind = parts[runs_index + 1]
    return source_kind, source_run_id


def _read_frames(frames_path: Path) -> list[dict[str, Any]]:
    if not frames_path.is_file():
        raise ReplayDecisionIngestError(f"{FRAMES_FILENAME} not found: {frames_path}")
    frames: list[dict[str, Any]] = []
    with frames_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayDecisionIngestError(f"{frames_path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ReplayDecisionIngestError(f"{frames_path}:{line_number}: frame must be a JSON object")
            schema = payload.get("schema_version")
            if schema != DECISION_FRAME_SCHEMA_VERSION:
                raise ReplayDecisionIngestError(
                    f"{frames_path}:{line_number}: unsupported frame schema_version {schema!r}; "
                    f"expected {DECISION_FRAME_SCHEMA_VERSION!r}"
                )
            frames.append(payload)
    return frames


def _seat_label_map(frames: list[dict[str, Any]]) -> dict[str, dict[int, str]]:
    """Map game_id -> {seat: agent_label}, used to derive opponent labels."""
    mapping: dict[str, dict[int, str]] = {}
    for frame in frames:
        game_id = frame.get("game_id")
        seat = frame.get("seat")
        label = frame.get("agent_label")
        if game_id is None or seat is None or label is None:
            continue
        mapping.setdefault(str(game_id), {}).setdefault(int(seat), str(label))
    return mapping


def _opponent_label(frame: dict[str, Any], seat_labels: dict[str, dict[int, str]]) -> str | None:
    game_id = frame.get("game_id")
    seat = frame.get("seat")
    if game_id is None or seat is None:
        return None
    by_seat = seat_labels.get(str(game_id), {})
    return by_seat.get(1 - int(seat))


def _frame_is_enriched(frame: dict[str, Any]) -> bool:
    """A post-TD.0.10 frame carries `resolved` on its option summaries."""
    options = frame.get("legal_options") or frame.get("chosen_option_summaries") or []
    if not options:
        return True
    first = options[0]
    return isinstance(first, dict) and "resolved" in first


def _chosen_resolved(chosen_summaries: list[dict[str, Any]]) -> bool:
    """Conservative: unresolved if any chosen summary lacks the enrichment field,
    or if any card-bearing chosen option is not resolved."""
    if not chosen_summaries:
        return False
    for option in chosen_summaries:
        if not isinstance(option, dict) or "resolved" not in option:
            return False
        if option.get("option_type") in CARD_BEARING_OPTION_TYPES and not option.get("resolved"):
            return False
    return True


def _option_types(options: list[dict[str, Any]]) -> list[str]:
    return [str(option.get("option_type")) for option in options if isinstance(option, dict)]


def _decision_row(
    frame: dict[str, Any],
    *,
    source_kind: str,
    source_run_id: str,
    source_path: str,
    opponent_label: str | None,
    deck_id: str | None,
    deck_hash: str | None,
    index: int | None,
) -> dict[str, Any]:
    legal_options = frame.get("legal_options") or []
    chosen_summaries = frame.get("chosen_option_summaries") or []
    final_result = frame.get("final_result")
    reason = frame.get("final_result_reason")
    seat = frame.get("seat")

    crashed = bool(isinstance(reason, str) and reason.startswith("crash:"))
    timed_out = bool(isinstance(reason, str) and reason.startswith("timeout"))
    draw = final_result == 2
    agent_win = (
        not draw
        and not crashed
        and not timed_out
        and final_result is not None
        and seat is not None
        and final_result == seat
    )

    row = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_run_id": source_run_id,
        "source_path": source_path,
        "trace_run_id": frame.get("run_id"),
        "game_id": frame.get("game_id"),
        "frame_id": frame.get("frame_id"),
        "decision_index": frame.get("decision_index"),
        "seed": frame.get("seed"),
        "seat": seat,
        "turn": frame.get("turn"),
        "agent_spec": frame.get("agent_spec"),
        "agent_label": frame.get("agent_label"),
        "policy_version": frame.get("policy_version"),
        "deck_path": frame.get("deck_path"),
        "deck_id": deck_id,
        "deck_hash": deck_hash,
        "opponent_label": opponent_label,
        "select_type": frame.get("select_type"),
        "select_context": frame.get("select_context"),
        "min_count": frame.get("min_count"),
        "max_count": frame.get("max_count"),
        "legal_count": len(legal_options),
        "legal_option_types": sorted(set(_option_types(legal_options))),
        "chosen_indices": list(frame.get("chosen_indices") or []),
        "chosen_option_types": _option_types(chosen_summaries),
        "chosen_card_ids": [option.get("card_id") for option in chosen_summaries if isinstance(option, dict)],
        "chosen_card_names": [option.get("card_name") for option in chosen_summaries if isinstance(option, dict)],
        "chosen_attack_ids": [option.get("attack_id") for option in chosen_summaries if isinstance(option, dict)],
        "chosen_resolved": _chosen_resolved(chosen_summaries),
        "final_result": final_result,
        "final_result_reason": reason,
        "agent_win": agent_win,
        "draw": draw,
        "crashed": crashed,
        "timed_out": timed_out,
    }
    return validate_decision_row(row, index=index)


def ingest_frames(
    path: str | Path,
    *,
    require_resolved: bool = False,
    deck_resolver: DeckResolver = deck_metadata_from_path,
) -> DecisionIngestResult:
    """Ingest one trace run (a `frames.jsonl` file or its run directory)."""
    frames_path = resolve_frames_path(path)
    frames = _read_frames(frames_path)
    source_kind, source_run_id = _derive_source(frames_path)
    source_path = str(frames_path)
    seat_labels = _seat_label_map(frames)
    deck_cache = _DeckCache(deck_resolver)

    rows: list[dict[str, Any]] = []
    skipped_unresolved = 0
    pre_enrichment_frames = 0
    for index, frame in enumerate(frames):
        enriched = _frame_is_enriched(frame)
        if not enriched:
            pre_enrichment_frames += 1
            if require_resolved:
                skipped_unresolved += 1
                continue
        deck_id, deck_hash = deck_cache.get(frame.get("deck_path"))
        opponent_label = _opponent_label(frame, seat_labels)
        try:
            rows.append(
                _decision_row(
                    frame,
                    source_kind=source_kind,
                    source_run_id=source_run_id,
                    source_path=source_path,
                    opponent_label=opponent_label,
                    deck_id=deck_id,
                    deck_hash=deck_hash,
                    index=index,
                )
            )
        except ReplayDecisionSchemaError as exc:
            raise ReplayDecisionIngestError(f"{source_path}: frame {index}: {exc}") from exc

    return DecisionIngestResult(
        rows=rows,
        source_kind=source_kind,
        source_run_id=source_run_id,
        source_path=source_path,
        frame_count=len(frames),
        skipped_unresolved=skipped_unresolved,
        pre_enrichment_frames=pre_enrichment_frames,
    )
