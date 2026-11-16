import sys

import pytest

from deepmew_ex.experiments.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from cg.api import CardType, EnergyType  # noqa: E402

from deepmew_ex.coach.decks import problems  # noqa: E402
from deepmew_ex.coach.explain import damage_estimate, energy_short  # noqa: E402
from deepmew_ex.coach.labels import cards, read_deck  # noqa: E402


def test_sample_deck_is_legal_and_rule_breaks_are_reported() -> None:
    deck = read_deck(PROJECT_ROOT / 'source/deck.csv')
    assert problems(deck) == []
    assert any('exactly 60' in issue for issue in problems(deck[:59]))
    trainer = next(card.cardId for card in cards().values() if card.cardType == CardType.ITEM and not card.aceSpec)
    assert any('More than 4' in issue for issue in problems(deck[:55] + [trainer] * 5))
    energy = next(card.cardId for card in cards().values() if card.cardType == CardType.BASIC_ENERGY)
    assert any('Basic Pokémon' in issue for issue in problems([energy] * 60))
    aces = [card.cardId for card in cards().values() if card.aceSpec][:2]
    if len(aces) == 2:
        assert any('ACE SPEC' in issue for issue in problems(deck[:58] + aces))


@pytest.mark.parametrize('cost, attached, missing', [
    ([EnergyType.FIRE, EnergyType.COLORLESS], [EnergyType.FIRE], 1),
    ([EnergyType.FIRE, EnergyType.COLORLESS], [EnergyType.WATER, EnergyType.FIRE], 0),
    ([EnergyType.FIRE, EnergyType.FIRE], [EnergyType.WATER], 2),
    ([EnergyType.COLORLESS] * 3, [EnergyType.GRASS], 2),
    ([EnergyType.PSYCHIC], [EnergyType.RAINBOW], 0),
])
def test_energy_short_matches_typed_then_colorless(cost: list[int], attached: list[int], missing: int) -> None:
    assert energy_short(cost, attached) == missing


def test_damage_estimate_applies_weakness_and_resistance() -> None:
    attacker = next(card for card in cards().values() if card.cardType == CardType.POKEMON and card.energyType == EnergyType.FIRE)
    weak = next((card for card in cards().values() if card.weakness == EnergyType.FIRE), None)
    resists = next((card for card in cards().values() if card.resistance == EnergyType.FIRE), None)
    if weak:
        assert damage_estimate(60, attacker.cardId, weak.cardId) == 120
    if resists:
        assert damage_estimate(60, attacker.cardId, resists.cardId) == 30
    assert damage_estimate(0, attacker.cardId, attacker.cardId) == 0
