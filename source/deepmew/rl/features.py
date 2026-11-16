"""Optional feature modules (all disabled in RL0).

A `FeatureModule` is the ONLY sanctioned way for a later experiment to inject
support-chain, DSU-package, opponent-belief, or robust-posterior signal into the
observation. Each module declares its name, inputs, forbidden inputs, output
schema, leakage boundary, and the experiment ids that may enable it. Adding a
module is a config + registry change; it extends the `ObservationEncoder` schema
but never touches the `EnvAdapter`, `RewardFn`, `OpponentSampler`, or trainer.

The concrete signal wiring (to the offline belief/prior packages) is deferred:
these scaffold stubs emit a declared-shape zero vector and import nothing at
module load time. The real, leakage-safe adapter is future work gated behind an
ablation that proves the module improves measured play strength.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from deepmew.rl.safety import assert_no_forbidden_inputs


@dataclass(frozen=True)
class FeatureModuleSpec:
    name: str
    inputs: tuple[str, ...]
    forbidden_inputs: tuple[str, ...]
    output_schema: tuple[str, ...]
    leakage_boundary: str
    enabled_experiment_ids: tuple[str, ...]
    requires_offline_proof: bool = True

    @property
    def output_dim(self) -> int:
        return len(self.output_schema)

    def __post_init__(self) -> None:
        # A module may never declare a future/private opponent field as an input.
        assert_no_forbidden_inputs(self.name, self.inputs)


class FeatureModule:
    """Base class. Subclasses keep any heavy import lazy inside ``compute``."""

    spec: FeatureModuleSpec

    def compute(self, obs: dict[str, Any], context: dict[str, Any]) -> list[float]:
        raise NotImplementedError


class _ZeroStubModule(FeatureModule):
    """Declared-shape placeholder until the leakage-safe adapter is wired."""

    def __init__(self, spec: FeatureModuleSpec):
        self.spec = spec

    def compute(self, obs: dict[str, Any], context: dict[str, Any]) -> list[float]:
        return [0.0] * self.spec.output_dim


_BELIEF_KB: Any | None = None


def _visible_card_id(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    card_id = value.get("id", value.get("cardId"))
    if card_id is None:
        return None
    try:
        return str(int(card_id))
    except (TypeError, ValueError):
        return str(card_id)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _opponent_public_card_events(obs: dict[str, Any]) -> list[tuple[str, str]]:
    """Return `(event_type, card_id)` pairs from currently public opponent zones."""
    from deepmew.opponent_belief.schemas import (
        EVENT_CARD_DISCARDED_PUBLIC,
        EVENT_CARD_REVEALED,
        EVENT_ENERGY_ATTACHED,
        EVENT_POKEMON_BENCHED,
        EVENT_STADIUM_PLAYED,
        EVENT_TOOL_ATTACHED,
    )

    current = obs.get("current") if isinstance(obs, dict) else {}
    current = current if isinstance(current, dict) else {}
    players = _as_list(current.get("players"))
    your_index = int(current.get("yourIndex") or 0)
    if len(players) < 2 or your_index not in (0, 1):
        return []
    opponent = players[1 - your_index] if isinstance(players[1 - your_index], dict) else {}
    events: list[tuple[str, str]] = []

    for area_name in ("active", "bench"):
        for pokemon in _as_list(opponent.get(area_name)):
            if not isinstance(pokemon, dict):
                continue
            card_id = _visible_card_id(pokemon)
            if card_id is not None:
                events.append((EVENT_POKEMON_BENCHED if area_name == "bench" else EVENT_CARD_REVEALED, card_id))
            for energy in _as_list(pokemon.get("energyCards")) + _as_list(pokemon.get("energies")):
                energy_id = _visible_card_id(energy)
                if energy_id is not None:
                    events.append((EVENT_ENERGY_ATTACHED, energy_id))
            for tool in _as_list(pokemon.get("tools")):
                tool_id = _visible_card_id(tool)
                if tool_id is not None:
                    events.append((EVENT_TOOL_ATTACHED, tool_id))

    for card in _as_list(opponent.get("discard")):
        card_id = _visible_card_id(card)
        if card_id is not None:
            events.append((EVENT_CARD_DISCARDED_PUBLIC, card_id))

    stadium = current.get("stadium")
    stadium_id = _visible_card_id(stadium)
    if stadium_id is not None:
        events.append((EVENT_STADIUM_PLAYED, stadium_id))
    return events


class _DsuBeliefFeatureModule(FeatureModule):
    """Public-only DEC-038 belief features for RL5.

    This recomputes a belief snapshot from currently visible public cards. It is
    deliberately conservative: no hidden hand, deck, prize, ground-truth
    archetype, or future replay row can enter because the only source is the
    current observation's public opponent board/discard/stadium fields.
    """

    def __init__(self, spec: FeatureModuleSpec):
        self.spec = spec

    def _kb(self) -> Any:
        global _BELIEF_KB
        if _BELIEF_KB is None:
            from deepmew.opponent_belief.knowledge import build_fixture_knowledge_base

            _BELIEF_KB = build_fixture_knowledge_base()
        return _BELIEF_KB

    def compute(self, obs: dict[str, Any], context: dict[str, Any]) -> list[float]:
        from deepmew.opponent_belief.belief import OpponentBeliefState
        from deepmew.opponent_belief.schemas import BeliefConfig, EvidenceEvent

        state = OpponentBeliefState(self._kb(), BeliefConfig.for_mode("B5"), modeled_actor="opponent")
        events = _opponent_public_card_events(obs)
        counts: Counter[str] = Counter()
        turn = int(((obs.get("current") or {}) if isinstance(obs, dict) else {}).get("turn") or 0)
        for sequence_index, (event_type, card_id) in enumerate(events):
            counts[card_id] += 1
            state.update(
                EvidenceEvent(
                    event_id=f"rl5_public_{sequence_index}_{event_type}_{card_id}",
                    sequence_index=sequence_index,
                    turn=turn,
                    actor="opponent",
                    event_type=event_type,
                    card_id=card_id,
                    count=counts[card_id],
                )
            )
        snapshot = state.snapshot()
        credible_size = min(len(snapshot.credible_set), 10) / 10.0
        return [
            float(snapshot.top_probability),
            float(snapshot.unknown_mass),
            float(snapshot.normalized_entropy),
            float(credible_size),
        ]


def _belief_snapshot_features(obs: dict[str, Any], kb: Any) -> list[float]:
    """4 public-only belief-posterior features from currently visible reveals."""
    from deepmew.opponent_belief.belief import OpponentBeliefState
    from deepmew.opponent_belief.schemas import BeliefConfig, EvidenceEvent

    state = OpponentBeliefState(kb, BeliefConfig.for_mode("B5"), modeled_actor="opponent")
    events = _opponent_public_card_events(obs)
    counts: Counter[str] = Counter()
    turn = int(((obs.get("current") or {}) if isinstance(obs, dict) else {}).get("turn") or 0)
    for sequence_index, (event_type, card_id) in enumerate(events):
        counts[card_id] += 1
        state.update(
            EvidenceEvent(
                event_id=f"pb_public_{sequence_index}_{event_type}_{card_id}",
                sequence_index=sequence_index,
                turn=turn,
                actor="opponent",
                event_type=event_type,
                card_id=card_id,
                count=counts[card_id],
            )
        )
    snapshot = state.snapshot()
    credible_size = min(len(snapshot.credible_set), 10) / 10.0
    return [float(snapshot.top_probability), float(snapshot.unknown_mass),
            float(snapshot.normalized_entropy), float(credible_size)]


def _public_board_threat_features(obs: dict[str, Any]) -> list[float]:
    """9 public board / prize / threat features from OUR own state + the
    opponent's PUBLICLY VISIBLE board. No hidden hand/deck/prize contents, no
    future info, no ground-truth decklist."""
    from deepmew.opponent_belief.schemas import (
        EVENT_CARD_REVEALED, EVENT_ENERGY_ATTACHED, EVENT_POKEMON_BENCHED,
        EVENT_STADIUM_PLAYED, EVENT_TOOL_ATTACHED,
    )

    current = obs.get("current") if isinstance(obs, dict) else {}
    current = current if isinstance(current, dict) else {}
    players = _as_list(current.get("players"))
    your_index = int(current.get("yourIndex") or 0)
    if len(players) < 2 or your_index not in (0, 1):
        return [0.0] * 9
    me = players[your_index] if isinstance(players[your_index], dict) else {}
    opp = players[1 - your_index] if isinstance(players[1 - your_index], dict) else {}

    def active_hp_ratio(player):
        active = _as_list(player.get("active"))
        p = active[0] if active and isinstance(active[0], dict) else {}
        max_hp = int(p.get("maxHp") or 0)
        if max_hp <= 0:
            return None
        cur = int(p.get("hp")) if p.get("hp") is not None else max(0, max_hp - int(p.get("damage") or 0))
        return max(0.0, min(1.0, cur / max_hp))

    my_bench = len(_as_list(me.get("bench")))
    opp_bench = len(_as_list(opp.get("bench")))
    my_has_active = bool(_as_list(me.get("active")))
    my_prizes = len(_as_list(me.get("prize")))
    opp_prizes = len(_as_list(opp.get("prize")))
    my_hp = active_hp_ratio(me)
    opp_hp = active_hp_ratio(opp)

    events = _opponent_public_card_events(obs)
    opp_energy = sum(1 for et, _ in events if et == EVENT_ENERGY_ATTACHED)
    opp_tool = 1.0 if any(et == EVENT_TOOL_ATTACHED for et, _ in events) else 0.0
    opp_stadium = 1.0 if any(et == EVENT_STADIUM_PLAYED for et, _ in events) else 0.0
    opp_pokemon_ids = {cid for et, cid in events if et in (EVENT_POKEMON_BENCHED, EVENT_CARD_REVEALED)}

    return [
        (opp_prizes - my_prizes) / 6.0,                              # prize_lead (>0 = we're ahead)
        1.0 if my_bench == 0 else 0.0,                               # my_no_bench
        1.0 if opp_bench == 0 else 0.0,                              # opp_no_bench
        1.0 if (my_hp is not None and my_hp < 0.35) else 0.0,        # my_active_ko_risk
        1.0 if (opp_hp is not None and 0.0 < opp_hp < 0.35) else 0.0,  # opp_active_ko_opportunity
        min(opp_energy, 12) / 12.0,                                  # opp_revealed_energy
        opp_tool,                                                    # opp_revealed_tool
        opp_stadium,                                                 # opp_stadium
        min(len(opp_pokemon_ids), 8) / 8.0,                          # opp_revealed_distinct_pokemon
    ]


class _PublicBeliefFeatureModule(FeatureModule):
    """RL4B public-belief features (DEC-052): DEC-038 public-only archetype
    posterior confidence + richer public board / prize / threat signal, all
    derived from the CURRENT observation's public fields. Goes into actor AND
    critic (the features are runtime-computable, so the actor may condition on
    them). No hidden hand/deck/prize contents, no future info, no ground-truth
    decklist, no oracle archetype.
    """

    def __init__(self, spec: FeatureModuleSpec):
        self.spec = spec

    def _kb(self) -> Any:
        global _BELIEF_KB
        if _BELIEF_KB is None:
            from deepmew.opponent_belief.knowledge import build_fixture_knowledge_base

            _BELIEF_KB = build_fixture_knowledge_base()
        return _BELIEF_KB

    def compute(self, obs: dict[str, Any], context: dict[str, Any]) -> list[float]:
        return _belief_snapshot_features(obs, self._kb()) + _public_board_threat_features(obs)


SUPPORT_CHAIN_SPEC = FeatureModuleSpec(
    name="support_chain",
    inputs=("public_revealed_cards", "deck_function_tags"),
    forbidden_inputs=("opponent_hand_cards", "future_draw"),
    output_schema=(
        "support_chain_completion",
        "support_chain_distinct_members",
        "support_chain_payoff_present",
    ),
    leakage_boundary="public revealed cards + static card-function tags only; no hidden hand",
    enabled_experiment_ids=("RL3_support_features", "RL4_opponent_mixture",
                            "RL5_dsu_belief", "RL6_robust_posterior"),
)

DSU_BELIEF_SPEC = FeatureModuleSpec(
    name="dsu_belief",
    inputs=("public_revealed_cards", "k2a_dsu_packages", "archetype_priors"),
    forbidden_inputs=("opponent_deck_order", "ground_truth_archetype", "future_cards"),
    output_schema=(
        "belief_top_archetype_mass",
        "belief_unknown_mass",
        "belief_entropy",
        "belief_credible_set_size",
    ),
    leakage_boundary="DEC-038 public-only posterior; ground truth is eval-label only",
    enabled_experiment_ids=("RL5_dsu_belief", "RL6_robust_posterior"),
)

ROBUST_POSTERIOR_SPEC = FeatureModuleSpec(
    name="robust_posterior",
    inputs=("dsu_belief_posterior", "matchup_card_priors_index"),
    forbidden_inputs=("opponent_hand_identities", "final_result_lookahead"),
    output_schema=(
        "posterior_coverage_mass",
        "posterior_lower_tail_proxy",
        "common_choke_bonus",
    ),
    leakage_boundary="DEC-039 inspection-only priors as FEATURES, never hard action overrides",
    enabled_experiment_ids=("RL6_robust_posterior",),
)


PUBLIC_BELIEF_SPEC = FeatureModuleSpec(
    name="public_belief",
    inputs=("public_revealed_cards", "public_board_state", "public_prize_counts", "archetype_priors"),
    forbidden_inputs=("opponent_hand_cards", "opponent_deck_order", "opponent_prize_cards",
                      "ground_truth_archetype", "future_cards"),
    output_schema=(
        "belief_top_probability", "belief_unknown_mass", "belief_normalized_entropy", "belief_credible_set_norm",
        "prize_lead_norm", "my_no_bench_flag", "opp_no_bench_flag", "my_active_ko_risk_flag",
        "opp_active_ko_opportunity_flag", "opp_revealed_energy_norm", "opp_revealed_tool_flag",
        "opp_stadium_flag", "opp_revealed_distinct_pokemon_norm",
    ),
    leakage_boundary=("DEC-038 public-only posterior + current-observation public board/prize/threat; "
                      "no hidden hand/deck/prize contents, no future info, no ground-truth decklist/archetype"),
    enabled_experiment_ids=("RL4B_public_belief",),
)


FEATURE_MODULE_SPECS: dict[str, FeatureModuleSpec] = {
    SUPPORT_CHAIN_SPEC.name: SUPPORT_CHAIN_SPEC,
    DSU_BELIEF_SPEC.name: DSU_BELIEF_SPEC,
    ROBUST_POSTERIOR_SPEC.name: ROBUST_POSTERIOR_SPEC,
    PUBLIC_BELIEF_SPEC.name: PUBLIC_BELIEF_SPEC,
}


def make_feature_module(name: str) -> FeatureModule:
    if name not in FEATURE_MODULE_SPECS:
        raise ValueError(
            f"unknown feature module {name!r}; known: {sorted(FEATURE_MODULE_SPECS)}"
        )
    spec = FEATURE_MODULE_SPECS[name]
    if name == "dsu_belief":
        return _DsuBeliefFeatureModule(spec)
    if name == "public_belief":
        return _PublicBeliefFeatureModule(spec)
    return _ZeroStubModule(spec)


def build_feature_modules(names: tuple[str, ...]) -> tuple[FeatureModule, ...]:
    return tuple(make_feature_module(name) for name in names)
