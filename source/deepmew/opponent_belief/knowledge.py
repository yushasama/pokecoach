"""Hypothesis knowledge base for OpponentBeliefState v0.

The default is a deterministic, self-contained *fixture* knowledge base built from
canonical card IDs (sourced from ``EN_Card_Data.csv``) and the support-aware meta
deck maps. It mirrors the K2A/DSU/deck-zoo *concepts* - archetype anchors, terminal
vs line vs support vs staple vs splash roles, per-archetype lift, support-chain /
package components - without depending on a live (gitignored) K2A run, so the
artifacts are byte-reproducible from a clean checkout.

``KnowledgeBase.from_meta_kg`` is provided for wrapping a real K2A run directory
when one exists; it is best-effort and never required by the default build.

Card IDs are canonical; names are display-only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmew.opponent_belief.schemas import CardId

# Card roles - identity weight decreases left to right; staple/splash never anchor.
ROLE_TERMINAL = "terminal"
ROLE_LINE = "line"
ROLE_SUPPORT = "support"
ROLE_ENERGY = "energy"
ROLE_TECH = "tech"
ROLE_STAPLE = "staple"
ROLE_SPLASH = "splash"


@dataclass(frozen=True)
class CardKnowledge:
    card_id: CardId
    name: str
    role: str
    # archetype -> lift (>= 0; 1.0 == neutral, > 1 == over-represented in archetype).
    lift_by_archetype: dict[str, float]
    global_prevalence: float  # fraction of decks containing the card (0, 1]
    staple_score: float = 0.0  # 0 == identity-bearing, 1 == pure staple
    reliable: bool = True  # a one-of tech is recorded but never a full anchor

    def archetypes(self) -> list[str]:
        return sorted(self.lift_by_archetype)


@dataclass(frozen=True)
class SupportComponent:
    """A correlation-safe evidence cluster: an evolution line or a support package.

    Members of one component are *correlated* (they co-occur by construction), so the
    belief engine scores them sub-modularly within the component rather than as
    independent observations. ``prereq``/``support``/``payoff`` mark the
    prerequisite -> support -> payoff completion relation.
    """

    component_id: str
    archetype: str
    members: frozenset[CardId]
    prereq: CardId | None = None
    support: CardId | None = None
    payoff: CardId | None = None
    variant_specific: str | None = None  # variant label this component is specific to


@dataclass(frozen=True)
class KnownDeck:
    deck_hash: str
    archetype: str
    variant_label: str
    card_counts: dict[CardId, int]

    def contains(self, card_id: CardId) -> bool:
        return card_id in self.card_counts


@dataclass(frozen=True)
class Archetype:
    name: str
    structural_playstyle: str


@dataclass
class KnowledgeBase:
    source: str  # "fixture" or "meta_kg:<dir>"
    archetypes: dict[str, Archetype]
    cards: dict[CardId, CardKnowledge]
    decks: list[KnownDeck]
    components_by_archetype: dict[str, list[SupportComponent]]

    def card(self, card_id: CardId) -> CardKnowledge | None:
        return self.cards.get(card_id)

    def role(self, card_id: CardId) -> str:
        card = self.cards.get(card_id)
        return card.role if card else ROLE_STAPLE

    def lift(self, card_id: CardId, archetype: str) -> float:
        card = self.cards.get(card_id)
        if card is None:
            return 1.0
        return card.lift_by_archetype.get(archetype, 1.0)

    def staple_score(self, card_id: CardId) -> float:
        card = self.cards.get(card_id)
        return card.staple_score if card else 1.0

    def archetype_names(self) -> list[str]:
        return sorted(self.archetypes)

    def playstyle(self, archetype: str) -> str:
        arch = self.archetypes.get(archetype)
        return arch.structural_playstyle if arch else "unknown"

    def components(self, archetype: str) -> list[SupportComponent]:
        return self.components_by_archetype.get(archetype, [])

    def to_summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "archetype_count": len(self.archetypes),
            "card_count": len(self.cards),
            "known_deck_count": len(self.decks),
            "component_count": sum(len(v) for v in self.components_by_archetype.values()),
            "staple_card_count": sum(1 for c in self.cards.values() if c.role == ROLE_STAPLE),
            "splash_card_count": sum(1 for c in self.cards.values() if c.role == ROLE_SPLASH),
        }

    # -- optional real-K2A wrapper (best-effort; not used by the default build) --
    @classmethod
    def from_meta_kg(cls, run_dir: str | Path) -> "KnowledgeBase":
        """Wrap a live K2A meta-kg run directory if available.

        Reads ``live_belief.json``, ``staples.csv``, ``card_lift_by_archetype.csv``,
        and ``deck_archetypes.csv``/``support_packages.csv`` when present. This path
        exists so the belief engine *extends* K2A rather than reimplementing it; the
        default fixture build does not require it.
        """
        from deepmew.opponent_belief.meta_kg_adapter import knowledge_base_from_meta_kg

        return knowledge_base_from_meta_kg(run_dir)


# --- Fixture meta definition ---------------------------------------------------
# Each archetype lists (card_id, name, role, lift, reliable). Lift is the
# in-archetype over-representation; cards absent from a list have neutral lift 1.0
# there. Prevalence and staple scores are derived below from how widely a card
# appears across the fixture meta - exactly the K2A inverse-prevalence logic.

_ARCHETYPES: dict[str, str] = {
    "Dragapult ex": "stage2_spread_tempo",
    "Mega Starmie ex": "evolution_bench_pressure",
    "Alakazam": "hand_size_scaling_control",
    "Mega Lucario ex": "fighting_acceleration_aggro",
    "Hop's Trevenant": "single_prize_tempo",
    "Jellicent Control": "item_lock_control",
}

# card_id: (name, role, {archetype: lift}, reliable)
_CARDS: dict[CardId, tuple[str, str, dict[str, float], bool]] = {
    # Dragapult ex
    "119": ("Dreepy", ROLE_LINE, {"Dragapult ex": 6.0}, True),
    "120": ("Drakloak", ROLE_LINE, {"Dragapult ex": 8.0}, True),
    "121": ("Dragapult ex", ROLE_TERMINAL, {"Dragapult ex": 32.0}, True),
    "1198": ("Crispin", ROLE_SUPPORT, {"Dragapult ex": 9.0}, True),
    "112": ("Munkidori", ROLE_TECH, {"Dragapult ex": 5.0}, False),
    # A Dragapult-associated meta tech that is in NONE of our canonical lists - used to
    # exercise "unseen tech inside a known archetype raises UNKNOWN-within-archetype".
    "1080": ("Unfair Stamp", ROLE_TECH, {"Dragapult ex": 3.0}, True),
    # Mega Starmie ex
    "1030": ("Staryu", ROLE_LINE, {"Mega Starmie ex": 7.0}, True),
    "1031": ("Mega Starmie ex", ROLE_TERMINAL, {"Mega Starmie ex": 34.0}, True),
    "1145": ("Mega Signal", ROLE_SUPPORT, {"Mega Starmie ex": 12.0}, True),
    "1189": ("Salvatore", ROLE_SUPPORT, {"Mega Starmie ex": 6.0, "Alakazam": 2.0}, True),
    # Alakazam
    "109": ("Abra", ROLE_LINE, {"Alakazam": 7.0}, True),
    "742": ("Kadabra", ROLE_LINE, {"Alakazam": 9.0}, True),
    "245": ("Alakazam", ROLE_TERMINAL, {"Alakazam": 30.0}, True),
    "1225": ("Hilda", ROLE_SUPPORT, {"Alakazam": 5.0, "Mega Starmie ex": 2.0}, True),
    "1231": ("Dawn", ROLE_SUPPORT, {"Alakazam": 6.0}, True),
    # shared cross-archetype accelerant (Dragapult + Alakazam) - ambiguous opener
    "1079": ("Rare Candy", ROLE_SUPPORT, {"Dragapult ex": 3.0, "Alakazam": 3.2}, True),
    # Mega Lucario ex
    "333": ("Riolu", ROLE_LINE, {"Mega Lucario ex": 8.0}, True),
    "678": ("Mega Lucario ex", ROLE_TERMINAL, {"Mega Lucario ex": 33.0}, True),
    "1142": ("Fighting Gong", ROLE_SUPPORT, {"Mega Lucario ex": 10.0}, True),
    "1141": ("Premium Power Pro", ROLE_SUPPORT, {"Mega Lucario ex": 7.0}, True),
    # Hop's Trevenant
    "878": ("Hop's Phantump", ROLE_LINE, {"Hop's Trevenant": 9.0}, True),
    "879": ("Hop's Trevenant", ROLE_TERMINAL, {"Hop's Trevenant": 31.0}, True),
    "1115": ("Hop's Bag", ROLE_SUPPORT, {"Hop's Trevenant": 11.0}, True),
    "1171": ("Hop's Choice Band", ROLE_SUPPORT, {"Hop's Trevenant": 9.0}, True),
    "304": ("Hop's Snorlax", ROLE_SUPPORT, {"Hop's Trevenant": 8.0}, True),
    "1255": ("Postwick", ROLE_SUPPORT, {"Hop's Trevenant": 7.0}, True),
    # Jellicent Control
    "597": ("Frillish", ROLE_LINE, {"Jellicent Control": 8.0}, True),
    "598": ("Jellicent ex", ROLE_TERMINAL, {"Jellicent Control": 30.0}, True),
    "73": ("Rellor", ROLE_LINE, {"Jellicent Control": 7.0}, True),
    "74": ("Rabsca", ROLE_TECH, {"Jellicent Control": 9.0}, True),
    "1264": ("Battle Cage", ROLE_TECH, {"Jellicent Control": 6.0}, False),
    "858": ("Psyduck", ROLE_TECH, {"Jellicent Control": 6.0}, False),
    "1158": ("Maximum Belt", ROLE_TECH, {"Jellicent Control": 5.0}, False),
    "751": ("Xerneas", ROLE_SUPPORT, {"Jellicent Control": 5.0}, True),
    # Energy (package evidence, never identity anchors)
    "2": ("Basic Fire Energy", ROLE_ENERGY, {"Dragapult ex": 2.2}, True),
    "5": ("Basic Psychic Energy", ROLE_ENERGY, {"Dragapult ex": 1.8, "Alakazam": 2.0, "Jellicent Control": 1.8}, True),
    "7": ("Basic Darkness Energy", ROLE_ENERGY, {"Dragapult ex": 2.5}, True),
    "3": ("Basic Water Energy", ROLE_ENERGY, {"Mega Starmie ex": 2.2, "Jellicent Control": 2.0}, True),
    "6": ("Basic Fighting Energy", ROLE_ENERGY, {"Mega Lucario ex": 2.4}, True),
    # Cross-deck splash engine (never anchors an archetype)
    "140": ("Fezandipiti ex", ROLE_SPLASH, {}, True),
    "56": ("Flutter Mane", ROLE_SPLASH, {"Jellicent Control": 1.6}, False),
    # Staples (broad, low identity)
    "1182": ("Boss's Orders", ROLE_STAPLE, {}, True),
    "1121": ("Ultra Ball", ROLE_STAPLE, {}, True),
    "1086": ("Buddy-Buddy Poffin", ROLE_STAPLE, {}, True),
    "1097": ("Night Stretcher", ROLE_STAPLE, {}, True),
    "1152": ("Poke Pad", ROLE_STAPLE, {}, True),
    "1227": ("Lillie's Determination", ROLE_STAPLE, {}, True),
    "1122": ("Pokegear 3.0", ROLE_STAPLE, {}, True),
}

# Support-chain / package components (DSU/entity relationships). Members of one
# component are correlated; the engine scores them sub-modularly.
_COMPONENTS: list[tuple[str, str, list[CardId], CardId | None, CardId | None, CardId | None, str | None]] = [
    # (component_id, archetype, members, prereq, support, payoff, variant_specific)
    ("dragapult_core_line", "Dragapult ex", ["119", "120", "121"], "119", "1079", "121", None),
    ("dragapult_split_energy", "Dragapult ex", ["1198", "2", "5"], "1198", "1198", "2", None),
    ("dragapult_darkness_splash", "Dragapult ex", ["112", "7"], None, "112", None, "Dragapult ex / Munkidori"),
    ("starmie_core_line", "Mega Starmie ex", ["1030", "1031"], "1030", "1145", "1031", None),
    ("starmie_signal_package", "Mega Starmie ex", ["1145", "1189", "3"], "1145", "1189", "1031", None),
    ("alakazam_core_line", "Alakazam", ["109", "742", "245"], "109", "1079", "245", None),
    ("alakazam_candy_package", "Alakazam", ["1079", "1225", "1231"], "1079", "1225", "245", None),
    ("lucario_core_line", "Mega Lucario ex", ["333", "678"], "333", "1142", "678", None),
    ("lucario_fighting_package", "Mega Lucario ex", ["1142", "1141", "6"], "1142", "1141", "678", None),
    ("trevenant_core_line", "Hop's Trevenant", ["878", "879"], "878", "1115", "879", None),
    ("trevenant_hop_package", "Hop's Trevenant", ["1115", "1171", "304", "1255"], "1115", "1171", "879", None),
    ("jellicent_jelly_line", "Jellicent Control", ["597", "598"], "597", None, "598", None),
    ("jellicent_rabsca_line", "Jellicent Control", ["73", "74"], "73", None, "74", None),
    ("jellicent_tech_package", "Jellicent Control", ["1264", "858", "1158"], None, None, None, None),
]

# Known decks (deck_hash, archetype, variant, card_counts). Two variants per
# archetype so list-variant resolution and UNKNOWN-within-archetype are meaningful.
# Counts are the *identifiable* subset (canonical staples + signature cards); the
# multiset-compatibility logic uses coverage + over-copy contradiction on these.
_DECKS: list[tuple[str, str, str, dict[CardId, int]]] = [
    ("dgp_munkidori_01", "Dragapult ex", "Dragapult ex / Munkidori",
     {"119": 4, "120": 2, "121": 3, "1198": 3, "112": 2, "7": 4, "2": 4, "5": 4,
      "1079": 4, "1121": 4, "1086": 4, "1182": 2, "1097": 2}),
    ("dgp_budew_01", "Dragapult ex", "Dragapult ex / Budew",
     {"119": 4, "120": 2, "121": 3, "1198": 4, "2": 5, "5": 5, "235": 1,
      "1079": 4, "1121": 4, "1086": 4, "1182": 2, "1097": 2, "1152": 2}),
    ("starmie_signal_01", "Mega Starmie ex", "Mega Starmie ex / Mega Signal",
     {"1030": 4, "1031": 3, "1145": 4, "1189": 3, "3": 9,
      "1121": 4, "1086": 4, "1182": 2, "1097": 2, "1225": 2}),
    ("starmie_salvatore_01", "Mega Starmie ex", "Mega Starmie ex / Salvatore tempo",
     {"1030": 4, "1031": 2, "1145": 2, "1189": 4, "3": 10,
      "1121": 4, "1086": 4, "1182": 3, "1152": 2}),
    ("alakazam_candy_01", "Alakazam", "Alakazam / Rare Candy",
     {"109": 4, "742": 1, "245": 3, "1079": 4, "1225": 3, "1231": 3, "5": 8,
      "1121": 3, "1086": 4, "1182": 2, "1152": 3}),
    ("alakazam_kadabra_01", "Alakazam", "Alakazam / Kadabra line",
     {"109": 4, "742": 3, "245": 3, "1225": 2, "1231": 3, "5": 8,
      "1121": 3, "1086": 4, "1182": 2, "1097": 2}),
    ("lucario_gong_01", "Mega Lucario ex", "Mega Lucario ex / Fighting Gong",
     {"333": 4, "678": 3, "1142": 4, "1141": 3, "6": 11,
      "1121": 4, "1182": 3, "1097": 2}),
    ("lucario_power_01", "Mega Lucario ex", "Mega Lucario ex / Power Pro aggro",
     {"333": 4, "678": 3, "1142": 2, "1141": 4, "6": 12, "1121": 4, "1182": 2}),
    ("trevenant_bag_01", "Hop's Trevenant", "Hop's Trevenant / Hop's Bag",
     {"878": 4, "879": 3, "1115": 4, "1171": 3, "304": 2, "1255": 2, "5": 7,
      "1152": 3, "1227": 3, "1182": 2, "1097": 2}),
    ("trevenant_postwick_01", "Hop's Trevenant", "Hop's Trevenant / Postwick stack",
     {"878": 4, "879": 3, "1115": 3, "1171": 4, "304": 2, "1255": 3, "5": 7,
      "1152": 2, "1227": 3, "1182": 2}),
    ("jellicent_rabsca_01", "Jellicent Control", "Jellicent / Rabsca tech",
     {"597": 3, "598": 3, "73": 2, "74": 2, "1264": 1, "858": 1, "1158": 2, "751": 2,
      "3": 6, "5": 4, "1121": 4, "1086": 4, "1182": 2, "1152": 2}),
    ("jellicent_judge_01", "Jellicent Control", "Jellicent / hand disruption",
     {"597": 3, "598": 3, "74": 1, "858": 2, "1158": 2, "751": 3, "56": 2,
      "3": 6, "5": 5, "1121": 4, "1086": 3, "1182": 3}),
]


def _build_cards() -> dict[CardId, CardKnowledge]:
    # Prevalence = fraction of known decks containing the card; staple score from a
    # K2A-style blend of breadth (deck frequency) and lack of archetype lift.
    deck_count = len(_DECKS)
    decks_with: dict[CardId, int] = {}
    archetypes_of: dict[CardId, set[str]] = {}
    for _, archetype, _v, counts in _DECKS:
        for cid in counts:
            decks_with[cid] = decks_with.get(cid, 0) + 1
            archetypes_of.setdefault(cid, set()).add(archetype)

    cards: dict[CardId, CardKnowledge] = {}
    for cid, (name, role, lift_by_arch, reliable) in _CARDS.items():
        present = decks_with.get(cid, 0)
        prevalence = max(present / deck_count, 1.0 / (2 * deck_count))
        max_lift = max(lift_by_arch.values(), default=1.0)
        spread = len(archetypes_of.get(cid, set()))
        if role == ROLE_STAPLE:
            staple = round(min(1.0, 0.55 + 0.1 * spread), 4)
        elif role == ROLE_SPLASH:
            staple = round(min(1.0, 0.4 + 0.1 * spread), 4)
        else:
            # Identity cards: staple score falls as max archetype lift rises.
            staple = round(max(0.0, min(0.5, 1.0 - (max_lift - 1.0) / 4.0)), 4)
        cards[cid] = CardKnowledge(
            card_id=cid,
            name=name,
            role=role,
            lift_by_archetype=dict(lift_by_arch),
            global_prevalence=round(prevalence, 4),
            staple_score=staple,
            reliable=reliable,
        )
    return cards


def build_fixture_knowledge_base() -> KnowledgeBase:
    """Deterministic, self-contained fixture knowledge base (the default)."""
    archetypes = {name: Archetype(name=name, structural_playstyle=play) for name, play in _ARCHETYPES.items()}
    cards = _build_cards()
    decks = [
        KnownDeck(deck_hash=h, archetype=a, variant_label=v, card_counts=dict(c))
        for h, a, v, c in _DECKS
    ]
    components_by_archetype: dict[str, list[SupportComponent]] = {}
    for comp_id, archetype, members, prereq, support, payoff, variant in _COMPONENTS:
        components_by_archetype.setdefault(archetype, []).append(
            SupportComponent(
                component_id=comp_id,
                archetype=archetype,
                members=frozenset(members),
                prereq=prereq,
                support=support,
                payoff=payoff,
                variant_specific=variant,
            )
        )
    return KnowledgeBase(
        source="fixture",
        archetypes=archetypes,
        cards=cards,
        decks=decks,
        components_by_archetype=components_by_archetype,
    )


def inverse_prevalence_weight(prevalence: float) -> float:
    """K2A-style inverse-prevalence: rarer cards are more diagnostic (capped)."""
    return min(3.0, math.log(1.0 / max(prevalence, 1e-3)))
