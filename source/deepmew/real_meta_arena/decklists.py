"""Load K3 Deck Zoo decklists and validate them as engine-instantiable decks.

A K3 decklist is a `{deck_hash, archetype, split, is_canonical, card_counts}`
record (card_counts sums to 60). Validation reuses the same path as
`deepmew.deck.validation`: 60 cards, known card IDs, <=1 ACE SPEC, a Basic
present, and `battle_start(deck, deck)` engine acceptance. Provenance
(deck_hash/archetype/variant/bucket) is preserved end to end.

Eval-only: nothing here is imported by the submission runtime; root `deck.csv`
is never mutated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cg.game import battle_finish, battle_start
from deepmew.deck.tables import card_table

DECKLISTS_FILENAME = "deck_zoo_decklists.jsonl"
DECK_ZOO_FILENAME = "deck_zoo.csv"


@dataclass(frozen=True)
class ZooDeck:
    deck_hash: str
    archetype: str
    split: str
    is_canonical: bool
    card_ids: tuple[int, ...]
    card_count: int
    variant_label: str = ""

    @property
    def is_standard_size(self) -> bool:
        return self.card_count == 60


@dataclass
class DecklistValidation:
    deck_hash: str
    archetype: str
    variant_label: str
    split: str
    card_count: int
    is_standard_size: bool
    known_card_ids: bool
    ace_spec_ok: bool
    has_basic: bool
    engine_accepted: bool | None
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def expand_card_counts(card_counts: dict[str, int]) -> tuple[int, ...]:
    """Expand {card_id: count} into a deterministic flat list of card IDs."""
    cards: list[int] = []
    for card_id in sorted(card_counts, key=lambda c: int(c)):
        cards.extend([int(card_id)] * int(card_counts[card_id]))
    return tuple(cards)


def load_zoo_decklists(deck_zoo_dir: str | Path) -> list[ZooDeck]:
    """Load canonical-deck decklists from a K3 deck-zoo run directory."""
    root = Path(deck_zoo_dir)
    path = root / DECKLISTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"deck zoo decklists not found: {path}")
    variant_by_hash = _variant_labels(root / DECK_ZOO_FILENAME)
    decks: list[ZooDeck] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        card_ids = expand_card_counts(row.get("card_counts") or {})
        decks.append(
            ZooDeck(
                deck_hash=str(row.get("deck_hash")),
                archetype=str(row.get("archetype") or "Unknown"),
                split=str(row.get("split") or "train"),
                is_canonical=bool(row.get("is_canonical")),
                card_ids=card_ids,
                card_count=len(card_ids),
                variant_label=variant_by_hash.get(str(row.get("deck_hash")), str(row.get("archetype") or "")),
            )
        )
    return decks


def _variant_labels(deck_zoo_csv: Path) -> dict[str, str]:
    if not deck_zoo_csv.exists():
        return {}
    import csv

    labels: dict[str, str] = {}
    with deck_zoo_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["deck_hash"]] = row.get("variant_label", "")
    return labels


def validate_decklist(deck: ZooDeck, *, check_engine: bool = True) -> DecklistValidation:
    """List-based legality check mirroring deepmew.deck.validation.validate_deck."""
    cards_by_id = card_table()
    card_ids = list(deck.card_ids)
    errors: list[str] = []
    warnings: list[str] = []

    is_standard = len(card_ids) == 60
    if not is_standard:
        errors.append(f"deck must contain exactly 60 cards; found {len(card_ids)}")

    missing = sorted({cid for cid in card_ids if cid not in cards_by_id})
    known_ok = not missing
    if missing:
        errors.append(f"unknown card id(s): {', '.join(str(c) for c in missing)}")

    ace_count = sum(1 for cid in card_ids if cid in cards_by_id and cards_by_id[cid].aceSpec)
    ace_ok = ace_count <= 1
    if not ace_ok:
        errors.append(f"deck may contain at most 1 ACE SPEC; found {ace_count}")

    has_basic = any(cards_by_id[cid].basic for cid in card_ids if cid in cards_by_id)
    if not has_basic:
        warnings.append("deck has no Basic Pokemon")

    engine_accepted: bool | None = None
    if check_engine and not errors:
        obs = None
        try:
            obs, start_data = battle_start(card_ids, card_ids)
            engine_accepted = obs is not None
            if obs is None:
                errors.append(
                    f"BattleStart failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
                )
        except Exception as exc:  # never let a bad deck crash the validator
            engine_accepted = False
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if obs is not None:
                battle_finish()

    return DecklistValidation(
        deck_hash=deck.deck_hash,
        archetype=deck.archetype,
        variant_label=deck.variant_label,
        split=deck.split,
        card_count=len(card_ids),
        is_standard_size=is_standard,
        known_card_ids=known_ok,
        ace_spec_ok=ace_ok,
        has_basic=has_basic,
        engine_accepted=engine_accepted,
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def validation_to_row(v: DecklistValidation) -> dict[str, Any]:
    return {
        "deck_hash": v.deck_hash,
        "archetype": v.archetype,
        "variant_label": v.variant_label,
        "split": v.split,
        "card_count": v.card_count,
        "is_standard_size": v.is_standard_size,
        "known_card_ids": v.known_card_ids,
        "ace_spec_ok": v.ace_spec_ok,
        "has_basic": v.has_basic,
        "engine_accepted": v.engine_accepted,
        "valid": v.valid,
        "errors": "; ".join(v.errors),
        "warnings": "; ".join(v.warnings),
    }
