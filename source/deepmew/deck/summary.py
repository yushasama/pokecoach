from __future__ import annotations

from collections import Counter
from pathlib import Path

from cg.api import CardType
from deepmew.deck.io import read_deck
from deepmew.deck.models import DeckCard, DeckSummary
from deepmew.deck.tables import ENERGY_TYPES, TRAINER_TYPES, attack_table, card_table, energy_name, type_name


def summarize_deck(path: str | Path) -> DeckSummary:
    deck = read_deck(path)
    cards_by_id = card_table()
    attacks_by_id = attack_table()
    counts = Counter(deck)
    rows: list[DeckCard] = []

    for card_id, count in sorted(counts.items(), key=lambda item: (str(cards_by_id.get(item[0], item[0])), item[0])):
        card = cards_by_id.get(card_id)
        if card is None:
            rows.append(
                DeckCard(card_id, count, "UNKNOWN", "UNKNOWN", False, False, False, False, False, False, 0, None, ())
            )
            continue
        attacks = []
        for attack_id in card.attacks or []:
            attack = attacks_by_id.get(int(attack_id))
            if attack is None:
                continue
            attacks.append(
                {
                    "attack_id": attack.attackId,
                    "name": attack.name,
                    "damage": attack.damage,
                    "energies": [energy_name(energy) for energy in attack.energies or []],
                }
            )
        rows.append(
            DeckCard(
                card_id=card_id,
                count=count,
                name=card.name,
                card_type=type_name(card),
                basic=bool(card.basic),
                stage1=bool(card.stage1),
                stage2=bool(card.stage2),
                ex=bool(card.ex),
                mega_ex=bool(card.megaEx),
                ace_spec=bool(card.aceSpec),
                hp=int(card.hp or 0),
                evolves_from=card.evolvesFrom,
                attacks=tuple(attacks),
            )
        )

    def card_type_count(types: set[CardType]) -> int:
        total = 0
        for card_id, count in counts.items():
            card = cards_by_id.get(card_id)
            if card is not None and CardType(card.cardType) in types:
                total += count
        return total

    pokemon_count = card_type_count({CardType.POKEMON})
    trainer_count = card_type_count(TRAINER_TYPES)
    energy_count = card_type_count(ENERGY_TYPES)
    main_attackers = tuple(
        sorted(
            (row for row in rows if row.card_type == "POKEMON" and row.attacks),
            key=lambda row: (
                -max((int(attack["damage"]) for attack in row.attacks), default=0),
                -row.hp,
                row.name,
            ),
        )[:5]
    )
    return DeckSummary(
        path=str(path),
        total_cards=len(deck),
        unique_cards=len(counts),
        pokemon_count=pokemon_count,
        trainer_count=trainer_count,
        energy_count=energy_count,
        basic_count=sum(row.count for row in rows if row.card_type == "POKEMON" and row.basic),
        evolution_count=sum(row.count for row in rows if row.card_type == "POKEMON" and (row.stage1 or row.stage2)),
        ex_count=sum(row.count for row in rows if row.ex),
        mega_ex_count=sum(row.count for row in rows if row.mega_ex),
        ace_spec_count=sum(row.count for row in rows if row.ace_spec),
        cards=tuple(rows),
        main_attackers=main_attackers,
    )
