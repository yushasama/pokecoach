"""CSV + JSONL writers for the strategic-action-scoring dataset (machine-readable)."""

from __future__ import annotations

import csv
import io as _io
from pathlib import Path
from typing import Any

from deepmew.artifacts.writers import write_jsonl, write_text

CSV_COLUMNS: tuple[str, ...] = (
    "subject_deck_id",
    "opponent_deck_id",
    "subject_label",
    "opponent_label",
    "candidate_id",
    "card_id",
    "card_name",
    "action_kind",
    "provides_coverage",
    "is_core",
    "score",
    "confidence",
    "reason_tags",
    "components",
)


def _components_text(components: list[dict[str, Any]]) -> str:
    return "; ".join(f"{c['reason']}={c['delta']}" for c in components)


def row_to_csv(row: dict[str, Any]) -> dict[str, str]:
    return {
        "subject_deck_id": row["subject_deck_id"],
        "opponent_deck_id": row["opponent_deck_id"],
        "subject_label": row.get("subject_label", ""),
        "opponent_label": row.get("opponent_label", ""),
        "candidate_id": row["candidate_id"],
        "card_id": row["card_id"],
        "card_name": row["card_name"],
        "action_kind": row["action_kind"],
        "provides_coverage": ";".join(row.get("provides_coverage", [])),
        "is_core": str(row.get("is_core", "")),
        "score": f"{row['score']:.4f}",
        "confidence": row["confidence"],
        "reason_tags": ";".join(row["reason_tags"]),
        "components": _components_text(row["components"]),
    }


def rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = _io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row_to_csv(row))
    return buffer.getvalue()


def write_scores_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    return write_text(Path(path), rows_to_csv_text(rows))


def write_scores_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    return write_jsonl(Path(path), rows)
