"""CSV + JSONL writers for matchup threat surfaces and pairings (machine-readable)."""

from __future__ import annotations

import csv
import io as _io
from pathlib import Path

from deepmew.artifacts.writers import write_jsonl, write_text
from deepmew.matchup_threat_inference.schemas import DeckThreatProfile, MatchupPairing

CSV_COLUMNS: tuple[str, ...] = (
    "deck_id",
    "deck_hash",
    "archetype_label",
    "split",
    "profile_confidence",
    "threat_confidence",
    "presents",
    "exposes",
    "coverage",
    "reason_tags",
    "manual_review_reasons",
)


def _threats_text(items) -> str:
    return "; ".join(f"{t.category}(x{t.strength}{'' if t.is_primary else ',tech'},{t.confidence})" for t in items)


def _coverage_text(items) -> str:
    return "; ".join(
        f"{c.category}(x{c.strength}{'' if c.reliable else ',tech'},{c.source})" for c in items
    )


def profile_to_csv_row(profile: DeckThreatProfile) -> dict[str, str]:
    return {
        "deck_id": profile.deck_id,
        "deck_hash": profile.deck_hash,
        "archetype_label": profile.archetype_label,
        "split": profile.split,
        "profile_confidence": profile.profile_confidence,
        "threat_confidence": profile.threat_confidence,
        "presents": _threats_text(profile.presents),
        "exposes": _threats_text(profile.exposes),
        "coverage": _coverage_text(profile.coverage),
        "reason_tags": ";".join(profile.reason_tags),
        "manual_review_reasons": " | ".join(profile.manual_review_reasons),
    }


def profiles_to_csv_text(profiles: list[DeckThreatProfile]) -> str:
    buffer = _io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for profile in profiles:
        writer.writerow(profile_to_csv_row(profile))
    return buffer.getvalue()


def write_profiles_csv(path: str | Path, profiles: list[DeckThreatProfile]) -> Path:
    return write_text(Path(path), profiles_to_csv_text(profiles))


def write_profiles_jsonl(path: str | Path, profiles: list[DeckThreatProfile]) -> Path:
    return write_jsonl(Path(path), [p.to_json_dict() for p in profiles])


def write_pairings_jsonl(path: str | Path, pairings: list[MatchupPairing]) -> Path:
    return write_jsonl(Path(path), [p.to_json_dict() for p in pairings])
