"""Human-readable text for native CABT observations: cards, moves, prompts, board, log."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from cg.api import AreaType, CardType, EnergyType, LogType, OptionType, SelectContext, SpecialConditionType, all_attack, all_card_data

Payload = dict[str, Any]

PROMPTS = {
    SelectContext.MAIN: 'Your turn: choose an action',
    SelectContext.SETUP_ACTIVE_POKEMON: 'Setup: choose your Active Pokémon',
    SelectContext.SETUP_BENCH_POKEMON: 'Setup: choose Pokémon for your Bench',
    SelectContext.SWITCH: 'Choose a Pokémon to switch into the Active Spot',
    SelectContext.TO_ACTIVE: 'Choose a Pokémon for the Active Spot',
    SelectContext.TO_BENCH: 'Choose Pokémon to put on your Bench',
    SelectContext.TO_HAND: 'Choose card(s) to put into your hand',
    SelectContext.DISCARD: 'Choose card(s) to discard',
    SelectContext.TO_DECK: 'Choose card(s) to return to your deck',
    SelectContext.ATTACK: 'Choose an attack',
    SelectContext.EVOLVE: 'Choose an evolution',
    SelectContext.IS_FIRST: 'Do you want to go first?',
    SelectContext.MULLIGAN: 'Mulligan: redraw your hand?',
    SelectContext.ACTIVATE: 'Activate this effect?',
    SelectContext.COIN_HEAD: 'Call the coin flip: heads?',
    SelectContext.ATTACH_FROM: 'Choose the Pokémon to attach to',
    SelectContext.DAMAGE: 'Choose a Pokémon to damage',
    SelectContext.HEAL: 'Choose a Pokémon to heal',
    SelectContext.DRAW_COUNT: 'How many cards to draw?',
}


@lru_cache(maxsize=1)
def cards() -> dict[int, Any]:
    return {card.cardId: card for card in all_card_data()}


@lru_cache(maxsize=1)
def attacks() -> dict[int, Any]:
    return {attack.attackId: attack for attack in all_attack()}


ENERGY_SYMBOLS = {'{G}': 'Grass', '{R}': 'Fire', '{W}': 'Water', '{L}': 'Lightning', '{P}': 'Psychic', '{F}': 'Fighting', '{D}': 'Darkness', '{M}': 'Metal', '{N}': 'Dragon', '{C}': 'Colorless'}


def pretty(text: str) -> str:
    for symbol, word in ENERGY_SYMBOLS.items():
        text = text.replace(symbol, word)
    return text


def card_name(card_id: int | None) -> str:
    if card_id is None:
        return 'face-down card'
    card = cards().get(card_id)
    return pretty(card.name) if card is not None else f'card #{card_id}'


def card_kind(card_id: int | None) -> str:
    card = cards().get(card_id)
    if card is None:
        return 'unknown'
    if card.cardType == CardType.POKEMON:
        return 'pokemon'
    return 'energy' if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY) else 'trainer'


def enum_text(kind: type, value: object) -> str:
    try:
        return kind(value).name.replace('_', ' ').title()
    except (ValueError, TypeError):
        return str(value)


def deck_label(deck: list[int]) -> str:
    """Name a deck by its most-played ex / evolved Pokémon, then its most-played Pokémon."""
    counts = Counter(deck)
    pokemon = [(count, card_id) for card_id, count in counts.items() if card_id in cards() and cards()[card_id].cardType == CardType.POKEMON]
    stars = sorted(((count, card_id) for count, card_id in pokemon if cards()[card_id].ex or cards()[card_id].stage2 or cards()[card_id].stage1), reverse=True)
    lead = stars or sorted(pokemon, reverse=True)
    names: list[str] = []
    for _, card_id in lead:
        name = card_name(card_id)
        if name not in names:
            names.append(name)
        if len(names) == 2:
            break
    return ' / '.join(names) if names else 'Unknown deck'


def read_deck(path: Path) -> list[int]:
    values = [int(line) for line in path.read_text(encoding='utf-8').split() if line.strip()]
    if len(values) < 60:
        raise ValueError(f'deck has {len(values)} cards: {path}')
    return values[:60]


def player(observation: Payload, index: int) -> Payload:
    return observation['current']['players'][index]


def pokemon_text(pokemon: Payload | None) -> str:
    if not pokemon:
        return 'face-down'
    energy = len(pokemon.get('energies') or [])
    return f"{card_name(pokemon.get('id'))} ({pokemon.get('hp')}/{pokemon.get('maxHp')} HP, {energy} energy)"


def area_card(observation: Payload, area: int | None, index: int | None, owner: int | None) -> str:
    """Name the card an option points at, from the actor-visible observation."""
    if area is None or index is None:
        return ''
    current, select = observation['current'], observation.get('select') or {}
    seat = current['yourIndex'] if owner is None else owner
    side = current['players'][seat]
    try:
        if area == AreaType.HAND:
            return card_name((side.get('hand') or [])[index]['id'])
        if area == AreaType.ACTIVE:
            return pokemon_text((side.get('active') or [None])[index])
        if area == AreaType.BENCH:
            return pokemon_text(side['bench'][index])
        if area == AreaType.DISCARD:
            return card_name(side['discard'][index]['id'])
        if area == AreaType.DECK:
            return card_name((select.get('deck') or [])[index]['id'])
        if area == AreaType.PRIZE:
            prize = side['prize'][index]
            return card_name(prize['id']) if prize else 'face-down prize'
        if area == AreaType.LOOKING:
            seen = (current.get('looking') or [])[index]
            return card_name(seen['id']) if seen else 'face-down card'
        if area == AreaType.STADIUM:
            return card_name(current['stadium'][index]['id'])
    except (IndexError, KeyError, TypeError):
        pass
    return f'{enum_text(AreaType, area)} #{index}'


def where(area: int | None, index: int | None) -> str:
    if area == AreaType.ACTIVE:
        return 'Active'
    if area == AreaType.BENCH:
        return f'Bench {index + 1}' if index is not None else 'Bench'
    return enum_text(AreaType, area)


def option_label(observation: Payload, option: Payload) -> str:
    kind = option.get('type')
    owner = option.get('playerIndex')
    mine = owner is None or owner == observation['current']['yourIndex']
    side = '' if mine else "opponent's "
    target = area_card(observation, option.get('area'), option.get('index'), owner)
    if kind == OptionType.PLAY:
        # PLAY options carry only a hand index (no area field).
        return f"Play {area_card(observation, AreaType.HAND, option.get('index'), None)}"
    if kind == OptionType.ATTACH:
        return f"Attach {target} → {area_card(observation, option.get('inPlayArea'), option.get('inPlayIndex'), None)} ({where(option.get('inPlayArea'), option.get('inPlayIndex'))})"
    if kind == OptionType.EVOLVE:
        return f"Evolve {area_card(observation, option.get('inPlayArea'), option.get('inPlayIndex'), None)} into {target}"
    if kind == OptionType.ABILITY:
        return f"Use ability of {target}"
    if kind == OptionType.ATTACK:
        attack = attacks().get(option.get('attackId'))
        return f"Attack: {attack.name} ({attack.damage} dmg)" if attack else f"Attack #{option.get('attackId')}"
    if kind == OptionType.RETREAT:
        return 'Retreat Active Pokémon'
    if kind == OptionType.END:
        return 'End turn'
    if kind == OptionType.YES:
        return 'Yes'
    if kind == OptionType.NO:
        return 'No'
    if kind == OptionType.NUMBER:
        return str(option.get('number'))
    if kind == OptionType.DISCARD:
        return f"Discard {target} from play"
    if kind == OptionType.CARD:
        return f"{side}{target}" + (f" ({where(option.get('area'), option.get('index'))})" if option.get('area') in (AreaType.ACTIVE, AreaType.BENCH) else '')
    if kind in (OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
        return f"{enum_text(OptionType, kind)} on {side}{target}"
    if kind == OptionType.ENERGY:
        return f"{enum_text(EnergyType, option.get('energyIndex'))} energy ×{option.get('count') or 1} on {target}"
    if kind == OptionType.SKILL:
        return f"Resolve {card_name(option.get('cardId'))}" if option.get('cardId') else 'Resolve special condition'
    if kind == OptionType.SPECIAL_CONDITION:
        return enum_text(SpecialConditionType, option.get('specialConditionType'))
    return enum_text(OptionType, kind)


ZONES = {AreaType.HAND: 'hand', AreaType.ACTIVE: 'active', AreaType.BENCH: 'bench', AreaType.DISCARD: 'discard', AreaType.DECK: 'deck', AreaType.PRIZE: 'prize', AreaType.LOOKING: 'looking', AreaType.STADIUM: 'stadium'}


def option_card_id(observation: Payload, option: Payload) -> int | None:
    """Card id an option is about (for showing the card in pickers), if visible."""
    area, index, owner = option.get('area'), option.get('index'), option.get('playerIndex')
    if option.get('type') == OptionType.PLAY:
        area = AreaType.HAND
    if option.get('type') == OptionType.SKILL:
        return option.get('cardId') or None
    if area is None or index is None:
        return None
    current, select = observation['current'], observation.get('select') or {}
    side = current['players'][current['yourIndex'] if owner is None else owner]
    try:
        if area == AreaType.HAND:
            return (side.get('hand') or [])[index]['id']
        if area == AreaType.DECK:
            return (select.get('deck') or [])[index]['id']
        if area == AreaType.DISCARD:
            return side['discard'][index]['id']
        if area == AreaType.LOOKING:
            seen = (current.get('looking') or [])[index]
            return seen['id'] if seen else None
        if area == AreaType.PRIZE:
            prize = side['prize'][index]
            return prize['id'] if prize else None
        if area == AreaType.ACTIVE:
            mon = (side.get('active') or [None])[index]
            return mon['id'] if mon else None
        if area == AreaType.BENCH:
            return side['bench'][index]['id']
    except (IndexError, KeyError, TypeError):
        return None
    return None


def option_detail(option: Payload) -> Payload | None:
    """Attack cost, damage and rules text for attack options."""
    if option.get('type') != OptionType.ATTACK:
        return None
    attack = attacks().get(option.get('attackId'))
    if attack is None:
        return None
    return {'name': attack.name, 'damage': attack.damage, 'text': attack.text, 'cost': [enum_text(EnergyType, value) for value in attack.energies]}


def option_targets(observation: Payload, option: Payload, seat: int) -> list[Payload]:
    """Board cards an option touches, as {side, zone, index}; the first is the card to click."""
    def ref(area: object, index: object, owner: object = None) -> Payload | None:
        if area not in ZONES or index is None:
            return None
        who = observation['current']['yourIndex'] if owner is None else owner
        return {'side': 'you' if who == seat else 'opponent', 'zone': ZONES[area], 'index': index}
    kind = option.get('type')
    refs: list[Payload | None]
    if kind in (OptionType.ATTACH, OptionType.EVOLVE):
        refs = [ref(option.get('area'), option.get('index')), ref(option.get('inPlayArea'), option.get('inPlayIndex'))]
    elif kind == OptionType.PLAY:
        refs = [ref(AreaType.HAND, option.get('index'))]
    elif kind in (OptionType.ATTACK, OptionType.RETREAT):
        refs = [ref(AreaType.ACTIVE, 0, seat)]
    else:
        refs = [ref(option.get('area'), option.get('index'), option.get('playerIndex'))]
    return [value for value in refs if value is not None]


def prompt(observation: Payload) -> str:
    select = observation.get('select') or {}
    context = select.get('context')
    text = PROMPTS.get(context) if context in PROMPTS else enum_text(SelectContext, context)
    effect = select.get('effect') or select.get('contextCard')
    if effect and effect.get('id'):
        text += f" — {card_name(effect['id'])}"
    low, high = select.get('minCount', 1), select.get('maxCount', 1)
    if (low, high) != (1, 1):
        text += f" (pick {low}–{high})" if low != high else f" (pick {low})"
    return text


def board(observation: Payload, seat: int) -> Payload:
    """Board from the human seat's perspective using only that seat's visible information."""
    current = observation['current']
    view: Payload = {'turn': current.get('turn'), 'yourTurn': current.get('yourIndex') == seat, 'stadium': [{'id': card['id'], 'name': card_name(card['id'])} for card in current.get('stadium') or []]}
    for label, index in (('you', seat), ('opponent', 1 - seat)):
        side = current['players'][index]
        view[label] = {
            'active': [pokemon_view(pokemon) for pokemon in side.get('active') or []],
            'bench': [pokemon_view(pokemon) for pokemon in side.get('bench') or []],
            'hand': [{'id': card['id'], 'name': card_name(card['id']), 'kind': card_kind(card['id'])} for card in side.get('hand') or []] if side.get('hand') is not None else None,
            'discardTop': ({'id': side['discard'][-1]['id'], 'name': card_name(side['discard'][-1]['id'])} if side.get('discard') else None),
            'discard': [{'id': card['id'], 'name': card_name(card['id'])} for card in side.get('discard') or []],
            'handCount': side.get('handCount'),
            'deckCount': side.get('deckCount'),
            'prizesLeft': len(side.get('prize') or []),
            'discardCount': len(side.get('discard') or []),
            'conditions': [name for name in ('poisoned', 'burned', 'asleep', 'paralyzed', 'confused') if side.get(name)],
        }
    return view


def pokemon_view(pokemon: Payload | None) -> Payload | None:
    if not pokemon:
        return None
    card = cards().get(pokemon.get('id'))
    return {
        'id': pokemon.get('id'),
        'name': card_name(pokemon.get('id')),
        'hp': pokemon.get('hp'), 'maxHp': pokemon.get('maxHp'),
        'energy': [enum_text(EnergyType, value) for value in pokemon.get('energies') or []],
        'tools': [card_name(tool['id']) for tool in pokemon.get('tools') or []],
        'toolIds': [tool['id'] for tool in pokemon.get('tools') or []],
        'stack': [card_name(card['id']) for card in pokemon.get('preEvolution') or []],
        'ex': bool(card and card.ex),
        'attacks': [f"{attacks()[a].name} ({attacks()[a].damage})" for a in (card.attacks if card else []) if a in attacks()],
    }


def log_lines(observation: Payload, seat: int) -> list[str]:
    lines = []
    for event in observation.get('logs') or []:
        kind = event.get('type')
        who = 'You' if event.get('playerIndex') == seat else 'Opponent'
        name = card_name(event.get('cardId')) if event.get('cardId') else ''
        if kind == LogType.TURN_START:
            lines.append(f'— {who}: turn start —')
        elif kind == LogType.PLAY:
            lines.append(f'{who} played {name}')
        elif kind == LogType.ATTACH:
            lines.append(f"{who} attached {name} to {card_name(event.get('cardIdTarget'))}")
        elif kind == LogType.EVOLVE:
            lines.append(f"{who} evolved {card_name(event.get('cardIdBefore'))} into {card_name(event.get('cardIdAfter'))}")
        elif kind == LogType.ATTACK:
            attack = attacks().get(event.get('attackId'))
            lines.append(f"{'Your' if who == 'You' else who + chr(39) + 's'} {name} used {attack.name if attack else 'an attack'}")
        elif kind == LogType.HP_CHANGE and event.get('value'):
            lines.append(f"{name} {'healed' if event.get('isRecover') else 'took'} {abs(event['value'])} {'HP' if event.get('isRecover') else 'damage'}")
        elif kind == LogType.SWITCH:
            lines.append(f"{who} switched {card_name(event.get('cardIdActive'))} ↔ {card_name(event.get('cardIdBench'))}")
        elif kind == LogType.COIN:
            lines.append(f"Coin flip: {'heads' if event.get('head') else 'tails'}")
        elif kind in (LogType.POISONED, LogType.BURNED, LogType.ASLEEP, LogType.PARALYZED, LogType.CONFUSED):
            lines.append(f"{name} is {enum_text(LogType, kind).lower()}")
        elif kind == LogType.RESULT:
            lines.append('Game over')
    return lines

