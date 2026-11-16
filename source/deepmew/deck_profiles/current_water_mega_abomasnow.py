"""Profile constants for the shipped Water / Mega Abomasnow deck."""

from __future__ import annotations

from dataclasses import dataclass


BASIC_WATER_ENERGY_ID = 3

KYOGRE_CARD_ID = 721
SNOVER_CARD_ID = 722
MEGA_ABOMASNOW_CARD_ID = 723

MEGA_SIGNAL_CARD_ID = 1145
MAXIMUM_BELT_CARD_ID = 1158
CYRANO_CARD_ID = 1205
LILLIES_DETERMINATION_CARD_ID = 1227
WAITRESS_CARD_ID = 1235

HAMMER_LANCHE_ATTACK_ID = 1046
RIPTIDE_ATTACK_ID = 1042

MEGA_LINE_IDS = frozenset({SNOVER_CARD_ID, MEGA_ABOMASNOW_CARD_ID})
VISIBLE_SETUP_PLAY_IDS = frozenset({MEGA_SIGNAL_CARD_ID, WAITRESS_CARD_ID})
NAMED_PLAY_CARD_IDS = frozenset(
    {
        MEGA_SIGNAL_CARD_ID,
        WAITRESS_CARD_ID,
        LILLIES_DETERMINATION_CARD_ID,
        CYRANO_CARD_ID,
        MAXIMUM_BELT_CARD_ID,
    }
)


@dataclass(frozen=True)
class CurrentDeckProfile:
    profile_id: str
    basic_energy_ids: frozenset[int]
    mega_line_ids: frozenset[int]
    fallback_attacker_ids: frozenset[int]
    setup_card_ids: frozenset[int]
    named_play_card_ids: frozenset[int]
    burst_attack_ids: frozenset[int]
    recycle_attack_ids: frozenset[int]

    def is_basic_energy(self, card_id: int | None) -> bool:
        return card_id in self.basic_energy_ids

    def is_mega_line(self, card_id: int | None) -> bool:
        return card_id in self.mega_line_ids

    def is_fallback_attacker(self, card_id: int | None) -> bool:
        return card_id in self.fallback_attacker_ids

    def is_setup_card(self, card_id: int | None) -> bool:
        return card_id in self.setup_card_ids

    def is_named_play_card(self, card_id: int | None) -> bool:
        return card_id in self.named_play_card_ids

    def is_burst_attack(self, attack_id: int | None) -> bool:
        return attack_id in self.burst_attack_ids

    def is_recycle_attack(self, attack_id: int | None) -> bool:
        return attack_id in self.recycle_attack_ids


CURRENT_WATER_MEGA_ABOMASNOW = CurrentDeckProfile(
    profile_id="current_water_mega_abomasnow",
    basic_energy_ids=frozenset({BASIC_WATER_ENERGY_ID}),
    mega_line_ids=MEGA_LINE_IDS,
    fallback_attacker_ids=frozenset({KYOGRE_CARD_ID}),
    setup_card_ids=VISIBLE_SETUP_PLAY_IDS,
    named_play_card_ids=NAMED_PLAY_CARD_IDS,
    burst_attack_ids=frozenset({HAMMER_LANCHE_ATTACK_ID}),
    recycle_attack_ids=frozenset({RIPTIDE_ATTACK_ID}),
)
