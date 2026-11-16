"""Load ``EN_Card_Data.csv`` and assemble card-type-aware :class:`CardInput`s.

The CSV is *one row per move*: a Pokémon with two attacks (or an Ability + an
attack, or a Tera box + attacks) occupies several rows sharing one ``Card ID``.
So the 2,022 rows collapse to 1,267 unique card IDs. We group rows by ``Card ID``
(preserving file order) and split each Pokémon's rows into Ability / attack /
Tera sections using the ``[Ability]`` and ``[Tera]`` move-name prefixes. Trainers
and Energy carry their effect in a single row.
"""

from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path

from deepmew.card_functions.schemas import (
    CARD_TYPE_ENERGY,
    CARD_TYPE_POKEMON,
    CARD_TYPE_TRAINER,
    CardInput,
    MoveText,
    SECTION_ABILITY,
    SECTION_ATTACK,
    SECTION_TERA,
    SUBTYPE_BASIC_ENERGY,
    SUBTYPE_ITEM,
    SUBTYPE_SPECIAL_ENERGY,
    SUBTYPE_STADIUM,
    SUBTYPE_SUPPORTER,
    SUBTYPE_TOOL,
)

DEFAULT_CSV_PATH = "data/pokemon-tcg-ai-battle/EN_Card_Data.csv"

_ABILITY_PREFIX = re.compile(r"^\s*\[ability\]\s*", re.IGNORECASE)
_TERA_PREFIX = re.compile(r"^\s*\[tera\]\s*", re.IGNORECASE)
_LEADING_INT = re.compile(r"(\d+)")

# Stage/Type -> (card_type, subtype) resolution.
_STAGE_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "basic pokémon": (CARD_TYPE_POKEMON, None),
    "stage 1 pokémon": (CARD_TYPE_POKEMON, None),
    "stage 2 pokémon": (CARD_TYPE_POKEMON, None),
    "item": (CARD_TYPE_TRAINER, SUBTYPE_ITEM),
    "supporter": (CARD_TYPE_TRAINER, SUBTYPE_SUPPORTER),
    "stadium": (CARD_TYPE_TRAINER, SUBTYPE_STADIUM),
    "pokémon tool": (CARD_TYPE_TRAINER, SUBTYPE_TOOL),
    "basic energy": (CARD_TYPE_ENERGY, SUBTYPE_BASIC_ENERGY),
    "special energy": (CARD_TYPE_ENERGY, SUBTYPE_SPECIAL_ENERGY),
}


def _na(value: str | None) -> str:
    """Normalise the CSV's ``n/a`` / empty sentinels to an empty string."""
    if value is None:
        return ""
    text = value.strip()
    return "" if text.lower() in ("", "n/a") else text


def _to_int(value: str | None) -> int | None:
    text = _na(value)
    if not text:
        return None
    match = _LEADING_INT.search(text)
    return int(match.group(1)) if match else None


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map logical column keys to the CSV's (sometimes awkward) header names."""

    def find(*candidates: str, exact: bool = False) -> str:
        for cand in candidates:
            for name in fieldnames:
                if name == cand:
                    return name
        if not exact:
            for cand in candidates:
                for name in fieldnames:
                    if name.lower().startswith(cand.lower()):
                        return name
        raise KeyError(f"could not resolve column for {candidates!r} in {fieldnames!r}")

    return {
        "card_id": find("Card ID"),
        "name": find("Card Name"),
        "stage_type": find("Stage (Pok", "Stage"),
        "rule": find("Rule", exact=True),
        "category": find("Category", exact=True),
        "previous_stage": find("Previous stage"),
        "hp": find("HP", exact=True),
        "type": find("Type", exact=True),
        "move_name": find("Move Name"),
        "cost": find("Cost", exact=True),
        "damage": find("Damage", exact=True),
        "effect": find("Effect Explanation"),
    }


def _classify_move(move_name_raw: str, cost: str, damage: str, effect: str) -> MoveText | None:
    """Turn one Pokémon row into an Ability / attack / Tera :class:`MoveText`."""
    name = _na(move_name_raw)
    effect_text = _na(effect)
    if _ABILITY_PREFIX.match(name):
        return MoveText(
            kind=SECTION_ABILITY,
            name=_ABILITY_PREFIX.sub("", name).strip(),
            cost="",
            damage=None,
            effect=effect_text,
        )
    if _TERA_PREFIX.match(name):
        return MoveText(
            kind=SECTION_TERA,
            name="Tera",
            cost="",
            damage=None,
            effect=effect_text,
        )
    if not name and not effect_text:
        return None
    return MoveText(
        kind=SECTION_ATTACK,
        name=name,
        cost=_na(cost),
        damage=_to_int(damage),
        effect=effect_text,
    )


def _build_card(rows: list[dict[str, str]], cols: dict[str, str]) -> CardInput:
    first = rows[0]
    stage_type = _na(first[cols["stage_type"]])
    card_type, subtype = _STAGE_TYPE_MAP.get(stage_type.lower(), (CARD_TYPE_TRAINER, None))

    rule_value = _na(first[cols["rule"]]) or None
    hp = _to_int(first[cols["hp"]])
    provided_energy = _na(first[cols["type"]]) if card_type == CARD_TYPE_ENERGY else ""
    previous_stage = _na(first[cols["previous_stage"]])

    moves: list[MoveText] = []
    trainer_text = ""
    energy_text = ""

    if card_type == CARD_TYPE_POKEMON:
        for row in rows:
            move = _classify_move(
                row[cols["move_name"]], row[cols["cost"]], row[cols["damage"]], row[cols["effect"]]
            )
            if move is not None:
                moves.append(move)
    elif card_type == CARD_TYPE_ENERGY:
        energy_text = " ".join(_na(row[cols["effect"]]) for row in rows if _na(row[cols["effect"]]))
    else:  # trainer
        trainer_text = " ".join(
            _na(row[cols["effect"]]) for row in rows if _na(row[cols["effect"]])
        )

    return CardInput(
        card_id=str(first[cols["card_id"]]).strip(),
        name=_na(first[cols["name"]]),
        card_type=card_type,
        stage_type=stage_type,
        subtype=subtype,
        rule=rule_value,
        hp=hp,
        provided_energy=provided_energy,
        previous_stage=previous_stage,
        moves=tuple(moves),
        trainer_text=trainer_text.strip(),
        energy_text=energy_text.strip(),
    )


def load_card_inputs(csv_path: str | Path = DEFAULT_CSV_PATH) -> list[CardInput]:
    """Load every unique card (by ``Card ID``) from ``EN_Card_Data.csv``."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"EN_Card_Data.csv not found: {path}")
    grouped: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        cols = _resolve_columns(list(reader.fieldnames))
        for row in reader:
            card_id = str(row[cols["card_id"]]).strip()
            if not card_id:
                continue
            grouped.setdefault(card_id, []).append(row)
    return [_build_card(rows, cols) for rows in grouped.values()]
