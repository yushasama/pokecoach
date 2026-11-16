"""CSV export for classified cards (machine-readable first)."""

from __future__ import annotations

import csv
import io as _io
from pathlib import Path

from deepmew.artifacts.writers import write_text
from deepmew.card_functions.schemas import CSV_COLUMNS, ClassifiedCard


def classified_to_row(card: ClassifiedCard) -> dict[str, str]:
    return {
        "card_id": card.card_id,
        "name": card.name,
        "card_type": card.card_type,
        "subtype": card.subtype or "",
        "stage": card.stage,
        "hp": "" if card.hp is None else str(card.hp),
        "rule": card.rule or "",
        "provided_energy": card.provided_energy,
        "tags": ";".join(card.tags),
        "confidence": card.confidence,
        "needs_manual_review": "true" if card.needs_manual_review else "false",
        "review_reasons": ";".join(card.review_reasons),
        "manual_audit_priority": str(card.manual_audit_priority),
        "parse_events": ";".join(e.as_compact() for e in card.parse_events),
        "ability_text": card.ability_text,
        "attack_text": card.attack_text,
        "tera_text": card.tera_text,
        "trainer_text": card.trainer_text,
        "energy_text": card.energy_text,
        "move_names": card.move_names,
        "damage_values": card.damage_values,
        "full_text": card.full_text,
    }


def cards_to_csv_text(cards: list[ClassifiedCard]) -> str:
    buffer = _io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for card in cards:
        writer.writerow(classified_to_row(card))
    return buffer.getvalue()


def write_cards_csv(path: str | Path, cards: list[ClassifiedCard]) -> Path:
    return write_text(Path(path), cards_to_csv_text(cards))
