"""Plain-language reasons for a recommended move, computed from the actor-visible game state.

Every statement comes from engine data: card HP, attack costs and damage,
weakness/resistance, attached energy, prize counts and card text. Damage is an
estimate (base damage ×2 for weakness, −30 for resistance); card effects can
change it, so it is worded as "about".
"""

from __future__ import annotations

from typing import Any

from cg.api import AreaType, CardType, EnergyType, OptionType, SelectContext

from deepmew_ex.coach.labels import attacks, card_name, cards, pretty

Payload = dict[str, Any]


def energy_short(cost: list[int], attached: list[int]) -> int:
    """How many more energy an attack needs: typed costs first, then Colorless from anything left."""
    pool = list(attached)
    missing = 0
    for need in [value for value in cost if value != EnergyType.COLORLESS]:
        for index, have in enumerate(pool):
            if have == need or have == EnergyType.RAINBOW:
                pool.pop(index)
                break
        else:
            missing += 1
    colorless = sum(1 for value in cost if value == EnergyType.COLORLESS)
    return missing + max(0, colorless - len(pool))


def first_sentence(text: str, limit: int = 160) -> str:
    text = pretty(' '.join((text or '').split()))
    cut = text.split('. ')[0]
    cut = cut if cut.endswith('.') else cut + '.'
    return cut if len(cut) <= limit else cut[:limit - 1] + '…'


def prize_value(card_id: int | None) -> int:
    card = cards().get(card_id)
    if card is None:
        return 1
    return 3 if card.megaEx else 2 if card.ex else 1


def damage_estimate(base: int, attacker_id: int | None, defender_id: int | None) -> int:
    attacker, defender = cards().get(attacker_id), cards().get(defender_id)
    damage = base
    if attacker is not None and defender is not None and base > 0:
        if defender.weakness is not None and defender.weakness == attacker.energyType:
            damage *= 2
        if defender.resistance is not None and defender.resistance == attacker.energyType:
            damage = max(0, damage - 30)
    return damage


def explain(observation: Payload, positions: list[int]) -> list[str]:
    try:
        return [line for line in _explain(observation, positions) if line]
    except Exception:
        return []


def _explain(observation: Payload, positions: list[int]) -> list[str]:
    current, select = observation['current'], observation.get('select') or {}
    me = current['yourIndex']
    mine, theirs = current['players'][me], current['players'][1 - me]
    options = select.get('option') or []
    chosen = [options[position] for position in positions if 0 <= position < len(options)]
    my_active = (mine.get('active') or [None])[0]
    their_active = (theirs.get('active') or [None])[0]
    lines: list[str] = []
    for option in chosen:
        kind = option.get('type')
        if kind == OptionType.ATTACK:
            attack = attacks().get(option.get('attackId'))
            if attack is None or not my_active or not their_active:
                continue
            damage = damage_estimate(attack.damage, my_active['id'], their_active['id'])
            defender = card_name(their_active['id'])
            if damage and damage >= their_active['hp']:
                prizes = prize_value(their_active['id'])
                lines.append(f"{attack.name} should Knock Out {defender} (about {damage} damage vs {their_active['hp']} HP left).")
                lines.append('That takes your last Prize cards — it wins the game.' if prizes >= len(mine.get('prize') or []) else f"Takes {prizes} Prize card{'s' if prizes > 1 else ''} ({len(mine.get('prize') or []) - prizes} left to win).")
            elif damage:
                lines.append(f"{attack.name} deals about {damage} to {defender} ({their_active['hp']} HP left), setting up a KO next turn.")
            if attack.text:
                lines.append(f"Effect: {first_sentence(attack.text)}")
            if damage and damage != attack.damage:
                lines.append('Weakness doubles this damage.' if damage > attack.damage else 'Resistance reduces this damage.')
        elif kind == OptionType.ATTACH:
            card = cards().get(_hand_card(mine, option))
            target = _in_play(mine, option.get('inPlayArea'), option.get('inPlayIndex'))
            if card is None or target is None:
                continue
            target_card = cards().get(target['id'])
            if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY) and target_card is not None:
                attached = list(target.get('energies') or []) + [card.energyType]
                plans = []
                for attack_id in target_card.attacks:
                    attack = attacks().get(attack_id)
                    if attack is None:
                        continue
                    short = energy_short(attack.energies, attached)
                    plans.append((short, attack))
                ready = [attack for short, attack in plans if short == 0]
                if ready:
                    best = max(ready, key=lambda a: a.damage)
                    lines.append(f"Powers up {card_name(target['id'])}: {best.name} ({best.damage}) becomes usable.")
                elif plans:
                    short, attack = min(plans, key=lambda row: (row[0], -row[1].damage))
                    lines.append(f"Builds toward {attack.name} ({attack.damage}) on {card_name(target['id'])}: {short} more energy needed.")
                if option.get('inPlayArea') == AreaType.BENCH:
                    lines.append('Charging a Benched Pokémon prepares your next attacker.')
            else:
                skill = card.skills[0].text if card.skills else ''
                lines.append(f"{card.name} on {card_name(target['id'])}: {first_sentence(skill)}" if skill else f"Attaches {card.name} to {card_name(target['id'])}.")
        elif kind == OptionType.EVOLVE:
            card = cards().get(_hand_card(mine, option))
            target = _in_play(mine, option.get('inPlayArea'), option.get('inPlayIndex'))
            if card is None or target is None:
                continue
            gain = card.hp - (target.get('maxHp') or 0)
            new = [attacks()[a].name for a in card.attacks if a in attacks()]
            lines.append(f"Evolves {card_name(target['id'])} into {card.name}: {'+' if gain >= 0 else ''}{gain} HP" + (f", unlocks {', '.join(new)}." if new else '.'))
            if card.skills:
                lines.append(f"Ability: {first_sentence(card.skills[0].text)}")
        elif kind == OptionType.PLAY:
            card = cards().get(_hand_card(mine, option))
            if card is None:
                continue
            if card.cardType == CardType.POKEMON:
                bench = len(mine.get('bench') or [])
                lines.append(f"Puts {card.name} on your Bench ({bench + 1}/{mine.get('benchMax') or 5}): more attackers and protection against a Knock Out ending the game.")
            else:
                text = card.skills[0].text if card.skills else ''
                role = {CardType.SUPPORTER: 'Supporter (one per turn)', CardType.ITEM: 'Item', CardType.STADIUM: 'Stadium', CardType.TOOL: 'Tool'}.get(card.cardType, 'Trainer')
                lines.append(f"{card.name} — {role}: {first_sentence(text)}" if text else f"Plays {card.name}.")
        elif kind == OptionType.ABILITY:
            pokemon = _in_play(mine, option.get('area'), option.get('index'))
            card = cards().get(pokemon['id']) if pokemon else None
            if card is not None and card.skills:
                lines.append(f"{card.name}'s ability: {first_sentence(card.skills[0].text)}")
        elif kind == OptionType.RETREAT:
            bench = mine.get('bench') or []
            if bench:
                best = max(bench, key=lambda mon: max([attacks()[a].damage for a in (cards()[mon['id']].attacks if mon['id'] in cards() else []) if a in attacks()] or [0]))
                lines.append(f"Retreating gets a stronger attacker into the Active Spot (e.g. {card_name(best['id'])}).")
            if my_active and my_active['hp'] <= my_active['maxHp'] // 3:
                lines.append(f"{card_name(my_active['id'])} is low on HP — saving it denies your opponent Prize cards.")
        elif kind == OptionType.END:
            lines.append('Nothing else here improves your position this turn; keep your remaining cards for later.')
        elif kind == OptionType.CARD:
            context = select.get('context')
            name = card_name(_option_card(observation, option))
            if context in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
                lines.append(f"{name} goes Active: it can start attacking or soak early damage.")
            elif context in (SelectContext.TO_HAND,):
                lines.append(f"Takes {name}: the card that best advances your plan right now.")
            elif context in (SelectContext.DISCARD,):
                lines.append(f"Discards {name}: the card you need least.")
            elif context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                lines.append(f"Benches {name}: a backup attacker, so one Knock Out can't end the game.")
            elif context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER):
                lines.append(f"Targets {name}.")
        elif kind in (OptionType.YES, OptionType.NO) and select.get('context') == SelectContext.IS_FIRST:
            lines.append('Going first: you set up before your opponent can attack.' if kind == OptionType.YES else 'Going second: you get to attack on your first turn and draw an extra card.')
    if not lines and chosen:
        lines.append("The coach's model rates this the strongest option in this position.")
    return lines


def _hand_card(side: Payload, option: Payload) -> int | None:
    if option.get('area') != AreaType.HAND and option.get('type') != OptionType.PLAY:
        return None
    hand = side.get('hand') or []
    index = option.get('index')
    return hand[index]['id'] if index is not None and 0 <= index < len(hand) else None


def _in_play(side: Payload, area: int | None, index: int | None) -> Payload | None:
    if area == AreaType.ACTIVE:
        return (side.get('active') or [None])[0]
    if area == AreaType.BENCH and index is not None and 0 <= index < len(side.get('bench') or []):
        return side['bench'][index]
    return None


def _option_card(observation: Payload, option: Payload) -> int | None:
    from deepmew_ex.coach.labels import option_card_id
    return option_card_id(observation, option)
