"""Deck building: the engine's card pool, standard deck rules, and saved user decks."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from cg.api import CardType, EnergyType

from deepmew_ex.coach.labels import card_kind, card_name, cards, deck_label, enum_text

DECK_SIZE = 60
MAX_COPIES = 4
TYPE_NAMES = {CardType.POKEMON: 'Pokémon', CardType.ITEM: 'Item', CardType.TOOL: 'Tool', CardType.SUPPORTER: 'Supporter', CardType.STADIUM: 'Stadium', CardType.BASIC_ENERGY: 'Basic Energy', CardType.SPECIAL_ENERGY: 'Special Energy'}


def catalog(images: set[int]) -> list[dict[str, object]]:
    """Every card the engine can play, with the fields a deck builder filters on."""
    rows = []
    for card in sorted(cards().values(), key=lambda c: (c.cardType != CardType.POKEMON, c.name, c.cardId)):
        stage = 'Basic' if card.basic else 'Stage 1' if card.stage1 else 'Stage 2' if card.stage2 else ''
        rows.append({'id': card.cardId, 'name': card_name(card.cardId), 'kind': card_kind(card.cardId), 'type': TYPE_NAMES.get(card.cardType, 'Card'),
                     'stage': stage, 'hp': card.hp or None, 'ex': bool(card.ex), 'mega': bool(card.megaEx), 'aceSpec': bool(card.aceSpec),
                     'energy': enum_text(EnergyType, card.energyType) if card.cardType in (CardType.POKEMON, CardType.BASIC_ENERGY) else '',
                     'evolvesFrom': card.evolvesFrom, 'image': card.cardId in images})
    return rows


def problems(deck: list[int]) -> list[str]:
    """Standard construction rules: 60 cards, ≤4 copies by name (Basic Energy exempt), ≥1 Basic Pokémon, ≤1 ACE SPEC."""
    pool = cards()
    issues = []
    unknown = sorted({card for card in deck if card not in pool})
    if unknown:
        issues.append(f'{len(unknown)} card(s) are not in the game\'s card pool')
    if len(deck) != DECK_SIZE:
        issues.append(f'Deck has {len(deck)} cards — it must have exactly {DECK_SIZE}')
    names = Counter(pool[card].name for card in deck if card in pool and pool[card].cardType != CardType.BASIC_ENERGY)
    over = sorted(name for name, count in names.items() if count > MAX_COPIES)
    if over:
        issues.append(f'More than {MAX_COPIES} copies of: {", ".join(over)}')
    if not any(card in pool and pool[card].cardType == CardType.POKEMON and pool[card].basic for card in deck):
        issues.append('Needs at least one Basic Pokémon')
    aces = sum(1 for card in deck if card in pool and pool[card].aceSpec)
    if aces > 1:
        issues.append(f'Only 1 ACE SPEC card is allowed (has {aces})')
    return issues


def slug(name: str) -> str:
    value = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not value:
        raise ValueError('deck name is required')
    return value[:60]


class DeckStore:
    def __init__(self, root: Path, samples: dict[str, list[int]]) -> None:
        self.root, self.samples = root, samples
        root.mkdir(parents=True, exist_ok=True)

    def saved(self) -> dict[str, list[int]]:
        decks = {}
        for path in sorted(self.root.glob('*.json')):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                decks[str(data['name'])] = [int(card) for card in data['cards']]
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return decks

    def all(self) -> dict[str, dict[str, object]]:
        """Saved decks first, then built-in sample decks (usable as templates)."""
        rows: dict[str, dict[str, object]] = {}
        for name, deck in self.saved().items():
            rows[name] = {'cards': deck, 'saved': True, 'label': deck_label(deck)}
        for name, deck in self.samples.items():
            rows.setdefault(name, {'cards': deck, 'saved': False, 'label': name})
        return rows

    def save(self, name: str, deck: list[int]) -> None:
        issues = problems(deck)
        if issues:
            raise ValueError('; '.join(issues))
        name = name.strip()
        if name in self.samples:
            raise ValueError('that name belongs to a sample deck — pick another name')
        (self.root / f'{slug(name)}.json').write_text(json.dumps({'name': name, 'cards': deck}), encoding='utf-8')

    def delete(self, name: str) -> None:
        path = self.root / f'{slug(name)}.json'
        if path.is_file():
            path.unlink()
