"""Native observer-only card query views required to interpret indexed choices."""

from __future__ import annotations

from dataclasses import replace

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.learning.semantic import leaves, relative_players
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence
from deepmew_ex.datasets.v2.schema import Payload


def actor_queries(observation: Payload, observer: int) -> Payload:
    current, selection = observation.get('current'), observation.get('select')
    if type(observer) is not int or observer not in (0, 1) or not isinstance(current, dict) or type(current.get('yourIndex')) is not int or current['yourIndex'] != observer or not isinstance(selection, dict):
        raise ValueError('card queries require the active observer request')
    result: Payload = {}
    fields = load_contract().observation_fields['Card']
    for name, value in (('deck', selection.get('deck')), ('looking', current.get('looking'))):
        if value is None:
            result[name] = None
            continue
        if not isinstance(value, list):
            raise ValueError('malformed native actor card query')
        copied = []
        for card in value:
            if card is None and name == 'looking':
                copied.append(None)
                continue
            if not isinstance(card, dict) or set(card) - fields or any(type(card.get(key)) is not int or card[key] < 0 for key in ('id', 'serial')) or type(card.get('playerIndex')) is not int or card['playerIndex'] not in (0, 1):
                raise ValueError('invalid native query card identity')
            copied.append(dict(card))
        result[name] = copied
    return result


def enrich_query_views(sequence: ActorSequence, observations: list[Payload]) -> ActorSequence:
    """Upgrade verified v3 records using the same source coordinates, loss unchanged."""
    if len(sequence.predictions) != len(observations):
        raise ValueError('query enrichment request membership mismatch')
    records = list(sequence.records)
    for prediction, observation in zip(sequence.predictions, observations, strict=True):
        index = sequence.events[prediction.event_index]
        if any(path.startswith('/actor_queries/') for path, _, _ in records[index]):
            raise ValueError('query views already present; refuse double enrichment')
        queries = relative_players(actor_queries(observation, sequence.actor), sequence.actor)
        records[index] = list(leaves({'actor_queries': queries})) + records[index]
    return replace(sequence, records=tuple(records))
