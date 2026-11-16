"""Load the CardFunctionClassifier v2 CSV into a ``card_id -> V2Card`` map."""

from __future__ import annotations

import csv
from pathlib import Path

from deepmew.deck_profile_inference.schemas import V2Card

DEFAULT_V2_CSV = ".research/card_function_classifier_v2/all_cards_function_tags_v2.csv"


def _to_int(value: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _max_damage(damage_values: str) -> int:
    best = 0
    for chunk in (damage_values or "").split("|"):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            best = max(best, int(digits))
    return best


def _rule_ids(parse_events: str) -> frozenset[str]:
    ids: set[str] = set()
    for event in (parse_events or "").split(";"):
        parts = event.split("|")
        if len(parts) >= 2 and parts[1]:
            ids.add(parts[1])
    return frozenset(ids)


def _v2_card_from_row(row: dict[str, str]) -> V2Card:
    tags = frozenset(t for t in (row.get("tags") or "").split(";") if t)
    subtype = (row.get("subtype") or "").strip() or None
    rule = (row.get("rule") or "").strip() or None
    return V2Card(
        card_id=str(row.get("card_id", "")).strip(),
        name=row.get("name", ""),
        card_type=row.get("card_type", ""),
        stage_type=row.get("stage", ""),
        subtype=subtype,
        rule=rule,
        hp=_to_int(row.get("hp", "")),
        provided_energy=row.get("provided_energy", ""),
        tags=tags,
        rule_ids=_rule_ids(row.get("parse_events", "")),
        confidence=(row.get("confidence") or "").strip() or "low",
        needs_manual_review=(row.get("needs_manual_review") or "").strip().lower() == "true",
        max_damage=_max_damage(row.get("damage_values", "")),
    )


def load_v2_cards(csv_path: str | Path = DEFAULT_V2_CSV) -> dict[str, V2Card]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CardFunctionClassifier v2 CSV not found: {path}")
    out: dict[str, V2Card] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            card = _v2_card_from_row(row)
            if card.card_id:
                out[card.card_id] = card
    return out
