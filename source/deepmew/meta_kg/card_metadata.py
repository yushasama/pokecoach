"""Optional card metadata enrichment (Pokemon/Trainer/Energy, stage, evolution).

When the battle engine (`cg`) is importable in the offline/training environment,
its card data gives reliable card categories and evolution lines. This is the
"EN card metadata enrichment" the K2A plan anticipated: it improves type/stage
guesses and anchor terminal detection. It is strictly optional -- if `cg` cannot
be imported the whole pipeline still runs from payload names alone
(``metadata_status = payload_only``).

This module is only ever loaded by the offline meta_kg builder; it never reaches
the submission/eval path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepmew.meta_kg.schemas import CardId

# cg.api CardData.cardType values.
CARD_TYPE_POKEMON = 0
CARD_TYPE_ENERGY = 5
TRAINER_CARD_TYPES = {1, 2, 3, 4}  # item, tool, supporter, stadium


@dataclass(frozen=True)
class CardMeta:
    card_id: CardId
    type_guess: str  # pokemon | trainer | energy | unknown
    stage_guess: str  # basic | stage1 | stage2 | none | unknown
    is_terminal: bool  # nothing in the pool evolves from this Pokemon
    is_ex: bool
    is_mega: bool
    evolves_from: str | None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _type_guess(card_type: Any) -> str:
    if card_type == CARD_TYPE_POKEMON:
        return "pokemon"
    if card_type == CARD_TYPE_ENERGY:
        return "energy"
    if card_type in TRAINER_CARD_TYPES:
        return "trainer"
    return "unknown"


def _stage_guess(card: Any, type_guess: str) -> str:
    if type_guess != "pokemon":
        return "none"
    if _get(card, "stage2"):
        return "stage2"
    if _get(card, "stage1"):
        return "stage1"
    if _get(card, "basic"):
        return "basic"
    return "unknown"


def load_card_metadata(enabled: bool = True) -> dict[CardId, CardMeta]:
    """Return card metadata keyed by string card id, or {} if unavailable."""
    if not enabled:
        return {}
    try:
        from cg.api import all_card_data  # local import: optional dependency
    except Exception:
        return {}
    try:
        data = all_card_data()
    except Exception:
        return {}

    evolved_from_names: set[str] = set()
    for card in data:
        src = _get(card, "evolvesFrom")
        if src:
            evolved_from_names.add(str(src))

    meta: dict[CardId, CardMeta] = {}
    for card in data:
        raw_id = _get(card, "cardId", _get(card, "id"))
        if raw_id is None:
            continue
        cid = str(raw_id)
        type_guess = _type_guess(_get(card, "cardType"))
        stage_guess = _stage_guess(card, type_guess)
        name = _get(card, "name")
        is_terminal = type_guess == "pokemon" and (not name or str(name) not in evolved_from_names)
        meta[cid] = CardMeta(
            card_id=cid,
            type_guess=type_guess,
            stage_guess=stage_guess,
            is_terminal=is_terminal,
            is_ex=bool(_get(card, "ex")) or bool(_get(card, "megaEx")),
            is_mega=bool(_get(card, "megaEx")),
            evolves_from=str(name) if name and str(name) in evolved_from_names else None,
        )
    # Re-key evolves_from to the card's own pre-evolution name for line building.
    by_name_chain: dict[CardId, str | None] = {}
    for card in data:
        raw_id = _get(card, "cardId", _get(card, "id"))
        if raw_id is None:
            continue
        by_name_chain[str(raw_id)] = (str(_get(card, "evolvesFrom")) if _get(card, "evolvesFrom") else None)
    # Replace evolves_from with the actual pre-evolution name (not self).
    for cid, pre in by_name_chain.items():
        if cid in meta:
            m = meta[cid]
            meta[cid] = CardMeta(
                card_id=m.card_id,
                type_guess=m.type_guess,
                stage_guess=m.stage_guess,
                is_terminal=m.is_terminal,
                is_ex=m.is_ex,
                is_mega=m.is_mega,
                evolves_from=pre,
            )
    return meta


def evolution_line_ids(
    terminal_id: CardId,
    meta: dict[CardId, CardMeta],
    name_to_ids: dict[str, list[CardId]],
    present_ids: set[CardId],
) -> list[CardId]:
    """Walk pre-evolutions of a terminal card, restricted to ``present_ids``."""
    line: list[CardId] = []
    seen: set[CardId] = {terminal_id}
    cursor = meta.get(terminal_id)
    guard = 0
    while cursor is not None and cursor.evolves_from and guard < 4:
        guard += 1
        pre_name = cursor.evolves_from.strip().lower()
        next_cursor = None
        for cid in name_to_ids.get(pre_name, []):
            if cid in present_ids and cid not in seen:
                line.append(cid)
                seen.add(cid)
                next_cursor = meta.get(cid)
        cursor = next_cursor
    return line
