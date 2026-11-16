from __future__ import annotations

import pytest

from deepmew_ex.datasets.v2.learning.actor_events import actor_events, observer_card_identities


def test_private_draw_is_visible_only_to_its_original_actor() -> None:
    event = {'type': 4, 'playerIndex': 0, 'cardId': 123, 'serial': 456}
    assert actor_events([event], 0, 0) == [event]
    assert actor_events([event], 1, 0) == [{'type': 4, 'playerIndex': 0}]
    assert actor_events([event], 0, 1) == [{'type': 4, 'playerIndex': 0}]


@pytest.mark.parametrize('area', [1, 2, 6, 11, 12, 999])
def test_hidden_movement_and_future_unknown_areas_do_not_reveal_opponent_cards(area: int) -> None:
    event = {'type': 6, 'playerIndex': 1, 'cardId': 123, 'serial': 456, 'fromArea': area, 'toArea': 2}
    assert 'cardId' not in actor_events([event], 0, 1)[0]
    if area == 999:
        assert 'cardId' not in actor_events([event], 1, 1)[0]
    else:
        assert actor_events([event], 1, 1)[0]['cardId'] == 123


@pytest.mark.parametrize('kind', [5, 7, 999])
def test_reverse_and_unknown_events_never_reveal_card_identity(kind: int) -> None:
    event = {'type': kind, 'playerIndex': 0, 'cardId': 123, 'serial': 456, 'fromArea': 4, 'toArea': 5}
    assert 'cardId' not in actor_events([event], 0, 0)[0]
    assert 'serial' not in actor_events([event], 1, 0)[0]


def test_public_effect_preserves_source_target_and_movement_identities() -> None:
    attach = {'type': 11, 'playerIndex': 1, 'cardId': 101, 'serial': 201, 'cardIdTarget': 102, 'serialTarget': 202, 'private_extra': 'hidden'}
    expected = {key: value for key, value in attach.items() if key != 'private_extra'}
    assert actor_events([attach], 0, 1, frozenset({(101, 201), (102, 202)})) == [expected]
    assert actor_events([attach], 1, 1) == [expected]
    move = {'type': 6, 'playerIndex': 1, 'cardId': 101, 'serial': 201, 'fromArea': 2, 'toArea': 3}
    assert actor_events([move], 0, 1, frozenset({(101, 201)})) == [move]


def test_malformed_visible_identity_fails_without_coercion() -> None:
    with pytest.raises(ValueError, match='visible event card identity'):
        actor_events([{'type': 15, 'playerIndex': 0, 'cardId': True, 'serial': 1}], 0, 0)


def test_setup_facedown_movement_does_not_reveal_another_actors_choice() -> None:
    event = {'type': 6, 'playerIndex': 1, 'cardId': 123, 'serial': 456, 'fromArea': 2, 'toArea': 4}
    assert 'cardId' not in actor_events([event], 0, 1)[0]
    assert actor_events([event], 1, 1)[0]['cardId'] == 123
    assert 'cardId' not in actor_events([event, {'type': 3, 'playerIndex': 0}], 0, 1)[0]


def test_witness_cannot_use_another_actors_view_or_hidden_zones() -> None:
    card = {'id': 123, 'serial': 456}
    current = {'yourIndex': 0, 'players': [{'active': [None], 'hand': [card], 'prize': [card]}, {'bench': [card]}], 'looking': [card]}
    assert observer_card_identities(current, 0) == frozenset({(123, 456)})
    current['players'][1]['bench'] = []
    assert not observer_card_identities(current, 0)
    with pytest.raises(ValueError, match='different observer'):
        observer_card_identities(current, 1)


def test_cross_seat_effect_requires_exact_pair_not_card_type_or_serial_alone() -> None:
    event = {'type': 11, 'playerIndex': 1, 'cardId': 101, 'serial': 201, 'cardIdTarget': 102, 'serialTarget': 202}
    partial = actor_events([event], 0, 1, frozenset({(101, 201), (102, 999), (999, 202)}))[0]
    assert partial['cardId'] == 101
    assert 'cardIdTarget' not in partial and 'serialTarget' not in partial
    assert 'cardId' not in actor_events([event], 0, 1)[0]


def test_private_draw_remains_hidden_even_if_identity_has_a_witness() -> None:
    event = {'type': 4, 'playerIndex': 1, 'cardId': 123, 'serial': 456}
    assert 'cardId' not in actor_events([event], 0, 1, frozenset({(123, 456)}))[0]


def test_public_card_witness_preserves_attached_and_nested_identities() -> None:
    current = {'yourIndex': 0, 'players': [{'active': [{'id': 1, 'serial': 2, 'energyCards': [{'id': 3, 'serial': 4}], 'tools': [{'id': 5, 'serial': 6}], 'preEvolution': [{'id': 7, 'serial': 8}]}]}, {}], 'stadium': [{'id': 9, 'serial': 10}]}
    assert observer_card_identities(current, 0) == frozenset({(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)})
