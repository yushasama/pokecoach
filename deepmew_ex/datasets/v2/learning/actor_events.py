"""Actor-visible event identities, with explicit private-zone and source-seat gates."""

from __future__ import annotations

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.schema import Payload, payload

# Identity-bearing CABT effects. Cross-seat visibility still requires a public
# state witness; a log kind or observation endpoint turn is not that witness.
PUBLIC_IDENTITIES = {
    'SWITCH': ('cardIdActive', 'serialActive', 'cardIdBench', 'serialBench'),
    'CHANGE': ('cardIdBefore', 'serialBefore', 'cardIdAfter', 'serialAfter'),
    'PLAY': ('cardId', 'serial'),
    'ATTACH': ('cardId', 'serial', 'cardIdTarget', 'serialTarget'),
    'EVOLVE': ('cardId', 'serial', 'cardIdTarget', 'serialTarget'),
    'DEVOLVE': ('cardId', 'serial', 'cardIdTarget', 'serialTarget'),
    'MOVE_ATTACHED': ('cardId', 'serial', 'cardIdBefore', 'serialBefore', 'cardIdAfter', 'serialAfter'),
    'ATTACK': ('cardId', 'serial'),
    'HP_CHANGE': ('cardId', 'serial'),
    'POISONED': ('cardId', 'serial'),
    'BURNED': ('cardId', 'serial'),
    'ASLEEP': ('cardId', 'serial'),
    'PARALYZED': ('cardId', 'serial'),
    'CONFUSED': ('cardId', 'serial'),
}
# AreaType values in the hash-bound source/cg/api.py. Prize, deck, hand,
# looking, player and every unrecognized area deliberately remain excluded.
PUBLIC_AREAS = frozenset({3, 4, 5, 7, 8, 9, 10})
KNOWN_AREAS = PUBLIC_AREAS | {1, 2, 6, 11, 12}
CARD_IDENTITY = ('cardId', 'serial')


def observer_card_identities(current: Payload, observer: int) -> frozenset[tuple[int, int]]:
    """Witness identities in this observer's current view, never another seat's."""
    if type(observer) is not int or observer not in (0, 1) or type(current.get('yourIndex')) is not int or current['yourIndex'] != observer:
        raise ValueError('card witness belongs to a different observer')
    result: set[tuple[int, int]] = set()

    def card(value: object) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise ValueError('malformed public card witness')
        identity = value.get('id'), value.get('serial')
        if any(type(item) is not int or item < 0 for item in identity):
            raise ValueError('invalid public card witness')
        result.add(identity)
        for zone in ('energyCards', 'tools', 'preEvolution'):
            for attached in value.get(zone, []):
                card(attached)

    players = current.get('players')
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError('public witness requires two players')
    for player in players:
        if not isinstance(player, dict):
            raise ValueError('malformed public player witness')
        for zone in ('active', 'bench', 'discard'):
            for value in player.get(zone, []):
                card(value)
    for value in current.get('stadium', []):
        card(value)
    return frozenset(result)


def actor_events(logs: object, observer: int, source_actor: int, visible_identities: frozenset[tuple[int, int]] = frozenset()) -> list[Payload]:
    if type(observer) is not int or type(source_actor) is not int or observer not in (0, 1) or source_actor not in (0, 1):
        raise ValueError('event visibility requires exact observer and source seats')
    if not isinstance(logs, list) or any(not isinstance(event, dict) for event in logs):
        raise ValueError('malformed actor event log')
    contract = load_contract()
    result = []
    for event in logs:
        kind = event.get('type')
        if type(kind) is not int:
            raise ValueError('event type must be an integer')
        projected = contract.public_logs([event])
        if len(projected) != 1:
            raise ValueError('event has no conservative public projection')
        value = payload(projected[0])
        name = contract.log_names.get(kind, 'unknown').upper()
        fields = PUBLIC_IDENTITIES.get(name, ())
        source_visible = source_actor == observer
        own_observed = source_actor == observer and type(event.get('playerIndex')) is int and event['playerIndex'] == observer
        if name == 'DRAW':
            fields = CARD_IDENTITY if own_observed else ()
        elif name == 'MOVE_CARD':
            areas = (event.get('fromArea'), event.get('toArea'))
            known_movement = all(type(area) is int and area in KNOWN_AREAS for area in areas)
            public_movement = known_movement and any(area in PUBLIC_AREAS for area in areas)
            fields = CARD_IDENTITY if known_movement and (source_visible or public_movement) else ()
        for offset in range(0, len(fields), 2):
            pair = fields[offset:offset + 2]
            identity = tuple(event.get(field) for field in pair)
            if all(item is None for item in identity):
                continue
            if any(type(item) is not int or item < 0 for item in identity):
                raise ValueError('invalid visible event card identity')
            if source_visible or identity in visible_identities:
                value.update(zip(pair, identity, strict=True))
        result.append(value)
    return result
