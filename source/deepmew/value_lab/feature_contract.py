"""Exact per-column contract for the accepted 144-float value cache.

The cache stores bytes, coarse block widths, and source hashes.  This module
turns the live feature-producing symbols into an explicit inspectable column
contract without changing any cached value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from cg.api import OptionType, SelectContext

from deepmew.replay.canonical_actor_dataset import FEATURE_DIM, S1_SLICE
from deepmew.second_order.reader import SECOND_ORDER_STATE_SCHEMA


FEATURE_CONTRACT_SCHEMA_VERSION = "deepmew.value_feature_contract.v1"


@dataclass(frozen=True)
class FeatureSpec:
    index: int
    name: str
    formula: str
    source_symbol: str
    semantic_block: str
    dtype: str
    declared_range: str
    missing_default: str
    standardized_in_cache: bool
    directly_actor_observable: bool
    deck_specificity: str
    policy_specificity: str
    source_specificity: str
    turn_specificity: str
    phase_specificity: str
    duplicate_candidates: tuple[int, ...] = ()
    caveat: str = ""


def _side_specs(start: int, side: str) -> list[FeatureSpec]:
    actor = "own" if side == "own" else "opponent"
    return [
        FeatureSpec(start + 0, f"{actor}_prizes_remaining_norm_side", "len(prize)/6", "scripts.run_unified_r3_value._side_features", "score_prizes", "float32", "nominal [0,1], not clipped", "non-list prize -> 6", False, True, "general", "none_direct", "missingness_proxy", "state", "all", (39 if side == "own" else 40, 119 if side == "own" else 120)),
        FeatureSpec(start + 1, f"{actor}_hand_count_norm_side", "(len(hand) if hand is list else int(handCount))/12", "scripts.run_unified_r3_value._side_features", "hand_resources", "float32", "nominal >=0, not clipped", "missing handCount -> 0", False, True, "deck_proxy", "policy_proxy", "missingness_proxy", "state", "all", (126, 143) if side == "own" else ()),
        FeatureSpec(start + 2, f"{actor}_deck_count_norm_side", "int(deckCount)/60", "scripts.run_unified_r3_value._side_features", "deck_discard_resources", "float32", "nominal [0,1], not clipped", "missing deckCount -> 0", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all", (29, 127) if side == "own" else ()),
        FeatureSpec(start + 3, f"{actor}_discard_count_norm60", "len(discard)/60", "scripts.run_unified_r3_value._side_features", "deck_discard_resources", "float32", "nominal [0,1], not clipped", "non-list discard -> 0", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all", (28,) if side == "own" else ()),
        FeatureSpec(start + 4, f"{actor}_bench_count_norm5", "len(nonempty bench)/5", "scripts.run_unified_r3_value._side_features", "board_occupancy", "float32", "nominal [0,1], not clipped", "non-list bench -> 0", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all", (124 if side == "own" else 125, 44 if side == "own" else 48)),
        FeatureSpec(start + 5, f"{actor}_active_present_flag", "1 if active[0] exists else 0", "scripts.run_unified_r3_value._side_features", "board_occupancy", "float32", "{0,1}", "missing/non-list active -> 0", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all"),
        FeatureSpec(start + 6, f"{actor}_active_hp_fraction", "clamp(int(hp)/int(maxHp),0,1)", "scripts.run_unified_r3_value._side_features", "hp_damage", "float32", "[0,1]", "maxHp<=0 -> 0; missing hp -> 0", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all", (121 if side == "own" else 122,)),
        FeatureSpec(start + 7, f"{actor}_active_energy_norm4", "min(1,(len(energies)+len(energyCards))/4)", "scripts.run_unified_r3_value._side_features", "energy_readiness", "float32", "[0,1]", "missing/non-list energy arrays -> 0", False, True, "deck_proxy", "policy_proxy", "representation_proxy", "state", "all", (123,) if side == "own" else (), "Adds both arrays when both representations are present."),
        FeatureSpec(start + 8, f"{actor}_active_tool_present_flag", "1 if len(active.tools)>0 else 0", "scripts.run_unified_r3_value._side_features", "status_tools", "float32", "{0,1}", "missing/non-list tools -> 0", False, True, "deck_proxy", "policy_proxy", "missingness_proxy", "state", "all"),
        FeatureSpec(start + 9, f"{actor}_bench_hp_fraction_sum_norm5", "sum(clamp(bench_i.hp/bench_i.maxHp,0,1))/5", "scripts.run_unified_r3_value._side_features", "hp_damage", "float32", "nominal [0,1], final sum not clipped", "missing bench/maxHp -> 0 contribution", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all"),
        FeatureSpec(start + 10, f"{actor}_status_count_norm5", "sum(player flags asleep,burned,confused,paralyzed,poisoned)/5", "scripts.run_unified_r3_value._side_features", "status_tools", "float32", "[0,1]", "missing flags -> false", False, True, "general", "policy_proxy", "missingness_proxy", "state", "all"),
    ]


def _base_specs() -> list[FeatureSpec]:
    specs = _side_specs(0, "own") + _side_specs(11, "opponent")
    specs.extend([
        FeatureSpec(22, "turn_norm30", "min(1,int(current.turn)/30)", "scripts.run_unified_r3_value._features", "turn_action_flags", "float32", "<=1; negative not lower-clipped", "missing turn -> 0", False, True, "general", "policy_proxy", "missingness_proxy", "direct", "all", (118,)),
        FeatureSpec(23, "actor_is_first_player_flag", "1 if int(firstPlayer)==actor seat else 0", "scripts.run_unified_r3_value._features", "turn_action_flags", "float32", "{0,1}", "missing firstPlayer -> false", False, True, "general", "none_direct", "none_direct", "game_role", "all", (132,)),
        FeatureSpec(24, "energy_attached_this_turn_flag", "1 if bool(current.energyAttached) else 0", "scripts.run_unified_r3_value._features", "turn_action_flags", "float32", "{0,1}", "missing -> false", False, True, "general", "policy_proxy", "missingness_proxy", "direct", "main_turn", (130,)),
        FeatureSpec(25, "supporter_played_this_turn_flag", "1 if bool(current.supporterPlayed) else 0", "scripts.run_unified_r3_value._features", "turn_action_flags", "float32", "{0,1}", "missing -> false", False, True, "general", "policy_proxy", "missingness_proxy", "direct", "main_turn", (128,)),
        FeatureSpec(26, "stadium_present_list_flag", "1 if current.stadium is a nonempty list else 0", "scripts.run_unified_r3_value._features", "turn_action_flags", "float32", "{0,1}", "dict/missing/empty stadium -> 0", False, True, "deck_proxy", "policy_proxy", "representation_proxy", "state", "all", (31,), "A stadium encoded as a dict is false here."),
    ])
    second_formulas = (
        "min(count own discard card id 3,16)/16",
        "min(len(own discard),40)/40",
        "min(int(own deckCount),60)/60",
        "1 if int(own deckCount)<=12 else 0",
        "1 if parsed stadium card id is not None else 0",
        "1 if parsed stadium card id==1262 else 0",
        "1 if Surfing Beach is in play and a legal ABILITY/STADIUM option exists else 0",
        "1 if surfing_beach_available and own active is Water and own Water bench count>0 else 0",
        "1 if own active is detected Water else 0",
        "min(own detected Water bench count,5)/5",
        "1 if card id 723 is active or benched else 0",
        "1 if active card id==723 else 0",
        "min(len(own prize),6)/6",
        "min(len(opponent prize),6)/6",
    )
    duplicates = ((3,), (3,), (2, 127), (), (26,), (), (), (), (), (), (), (), (0, 119), (11, 120))
    for offset, (name, formula, duplicate) in enumerate(zip(SECOND_ORDER_STATE_SCHEMA, second_formulas, duplicates)):
        named = 5 <= offset <= 11
        specs.append(FeatureSpec(
            27 + offset,
            name,
            formula,
            "deepmew.second_order.reader.SecondOrderState.vector",
            "named_card_proxies" if named else ("score_prizes" if offset >= 12 else "deck_discard_resources"),
            "float32",
            "[0,1]",
            "reader-specific; see formula",
            False,
            True,
            "named_card_specific" if named else "general",
            "policy_proxy",
            "source_and_deck_proxy" if named else "missingness_proxy",
            "state",
            "legal_context" if offset in (6, 7) else "all",
            duplicate,
        ))
    dev_names = ("bench_energy_norm6", "total_in_play_energy_norm8", "max_in_play_energy_norm4", "board_width_norm6")
    dev_formulas = (
        "min(sum(len(bench_i.energies)),6)/6",
        "min(len(active.energies)+sum(len(bench_i.energies)),8)/8",
        "min(max energy-unit count over active+bench,4)/4",
        "(active-present + nonempty-bench-count)/6",
    )
    for side, start in (("own", 41), ("opponent", 45)):
        for offset, (name, formula) in enumerate(zip(dev_names, dev_formulas)):
            specs.append(FeatureSpec(start + offset, f"{side}_{name}", formula, "scripts.run_unified_r3_value._dev_features", "energy_readiness" if offset < 3 else "board_occupancy", "float32", "[0,1]", "missing arrays -> 0", False, True, "deck_proxy", "policy_proxy", "representation_proxy", "state", "all", ((7, 123) if side == "own" and offset in (1, 2) else ((4, 124) if side == "own" and offset == 3 else ((15, 125) if side == "opponent" and offset == 3 else ())))))

    return specs


def _encode_base_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for offset, context in enumerate(SelectContext):
        specs.append(FeatureSpec(
            49 + offset,
            f"select_context_{context.name.lower()}_onehot",
            f"1 if select.context==SelectContext.{context.name}.value ({context.value}) else 0",
            "scripts.run_unified_r4_ablation.encode_base",
            "legal_context",
            "float32",
            "{0,1}",
            "missing/type-mismatched context -> all context columns 0",
            False,
            True,
            "none_direct",
            "policy_proxy",
            "schema_and_phase_proxy",
            "decision",
            context.name.lower(),
        ))
    for offset, option in enumerate(OptionType):
        specs.append(FeatureSpec(
            98 + offset,
            f"legal_option_{option.name.lower()}_fraction",
            f"count recognized OptionType.{option.name}/max(1,total legal options)",
            "scripts.run_unified_r4_ablation.encode_base",
            "legal_option_mix",
            "float32",
            "[0,1]",
            "empty options -> 0; unknown types remain only in denominator",
            False,
            True,
            "deck_proxy",
            "policy_and_state_proxy",
            "schema_proxy",
            "decision",
            "legal_selection",
        ))
    scalar_specs = (
        ("legal_option_count_norm30", "min(len(options),30)/30", "[0,1]", "missing options -> empty", "selection_counts", ()),
        ("select_min_count_norm10", "min(int(select.minCount),10)/10", "<=1; negative not lower-clipped", "missing -> 0", "selection_counts", ()),
        ("select_max_count_norm10", "min(int(select.maxCount),10)/10", "<=1; negative not lower-clipped", "missing -> 0", "selection_counts", ()),
        ("turn_norm40", "min(int(current.turn),40)/40", "<=1; negative not lower-clipped", "missing -> 0", "turn_action_flags", (22,)),
        ("own_prizes_remaining_norm_encode", "len(own.prize)/6", "nominal [0,1]", "valid player missing prize -> 0; invalid players -> 1", "score_prizes", (0, 39)),
        ("opponent_prizes_remaining_norm_encode", "len(opponent.prize)/6", "nominal [0,1]", "valid player missing prize -> 0; invalid players -> 1", "score_prizes", (11, 40)),
        ("own_active_hp_fraction_encode", "clamp(hp/maxHp,0,1)", "[0,1]", "maxHp<=0 -> 0; missing hp with maxHp -> full", "hp_damage", (6,)),
        ("opponent_active_hp_fraction_encode", "clamp(hp/maxHp,0,1)", "[0,1]", "maxHp<=0 -> 0; missing hp with maxHp -> full", "hp_damage", (17,)),
        ("own_active_energy_norm6_encode", "min(active energy count,6)/6", "[0,1]", "energies list, else energyCards list, else scalar/missing 0", "energy_readiness", (7, 42, 43)),
        ("own_bench_count_norm5_encode", "min(len(own.bench),5)/5", "[0,1]", "missing bench -> 0", "board_occupancy", (4, 44)),
        ("opponent_bench_count_norm5_encode", "min(len(opponent.bench),5)/5", "[0,1]", "missing bench -> 0", "board_occupancy", (15, 48)),
        ("own_hand_count_norm12_encode", "min(handCount fallback visible hand length,12)/12", "[0,1]", "missing -> 0", "hand_resources", (1, 143)),
        ("own_deck_count_norm60_encode", "min(deckCount,60)/60", "<=1; negative not lower-clipped", "missing -> 60 -> 1", "deck_discard_resources", (2, 29)),
        ("supporter_played_this_turn_flag_encode", "1 if bool(current.supporterPlayed) else 0", "{0,1}", "missing -> false", "turn_action_flags", (25,)),
        ("stadium_played_this_turn_flag", "1 if bool(current.stadiumPlayed) else 0", "{0,1}", "missing -> false", "turn_action_flags", ()),
        ("energy_attached_this_turn_flag_encode", "1 if bool(current.energyAttached) else 0", "{0,1}", "missing -> false", "turn_action_flags", (24,)),
        ("retreated_this_turn_flag", "1 if bool(current.retreated) else 0", "{0,1}", "missing -> false", "turn_action_flags", ()),
        ("actor_is_first_player_flag_encode", "1 if int(firstPlayer)==actor seat else 0", "{0,1}", "missing -> false", "turn_action_flags", (23,)),
    )
    for offset, (name, formula, value_range, missing, block, duplicates) in enumerate(scalar_specs):
        specs.append(FeatureSpec(115 + offset, name, formula, "scripts.run_unified_r4_ablation.encode_base", block, "float32", value_range, missing, False, True, "deck_proxy" if block in {"hand_resources", "deck_discard_resources", "energy_readiness"} else "general", "policy_proxy", "missingness_proxy", "direct" if block == "turn_action_flags" else "state", "selection_or_turn", duplicates))

    return specs


def _hand_specs() -> list[FeatureSpec]:
    categories = ("pokemon", "item", "tool", "supporter", "stadium", "basic_energy", "special_energy")
    specs = [
        FeatureSpec(133 + offset, f"own_hand_{name}_count_norm10", f"min(count parsed own-hand cards with cardType category {offset},10)/10", "scripts.build_value_q_canonical._hand11", "hand_composition", "float32", "[0,1]", "unparsed/missing hand IDs -> no contribution", False, True, "strong_deck_proxy", "policy_proxy", "source_and_provenance_proxy", "state", "all", (), "Top-level row own_hand is consulted before projected observation hand.")
        for offset, name in enumerate(categories)
    ]
    extras = (
        ("own_hand_basic_pokemon_count_norm10", "min(count cards with basic and cardType==POKEMON,10)/10"),
        ("own_hand_evolution_pokemon_count_norm10", "min(count cards with stage1 or stage2 or megaEx,10)/10"),
        ("own_hand_ex_pokemon_count_norm10", "min(count cards with ex or megaEx,10)/10"),
        ("own_hand_parsed_size_norm12", "min(number of parsed own-hand IDs,12)/12"),
    )
    for offset, (name, formula) in enumerate(extras):
        specs.append(FeatureSpec(140 + offset, name, formula, "scripts.build_value_q_canonical._hand11", "hand_composition", "float32", "[0,1]", "unparsed/missing hand IDs -> 0", False, True, "strong_deck_proxy", "policy_proxy", "source_and_provenance_proxy", "state", "all", (1, 126) if offset == 3 else (), "Top-level row own_hand is consulted before projected observation hand."))

    return specs


def feature_specs() -> tuple[FeatureSpec, ...]:
    specs = tuple(sorted(_base_specs() + _encode_base_specs() + _hand_specs(), key=lambda spec: spec.index))
    if len(specs) != FEATURE_DIM or tuple(spec.index for spec in specs) != tuple(range(FEATURE_DIM)):
        raise AssertionError("feature contract does not exactly cover canonical columns")
    if len({spec.name for spec in specs}) != FEATURE_DIM:
        raise AssertionError("feature contract names are not unique")
    if len(tuple(SelectContext)) != 49 or len(tuple(OptionType)) != 17 or S1_SLICE != (0, 49):
        raise AssertionError("live enum or S1 boundary drifted from the accepted cache")

    return specs


def representation_manifest(name: str, indices: Iterable[int]) -> dict[str, Any]:
    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected) or any(index < 0 or index >= FEATURE_DIM for index in selected):
        raise ValueError("representation indices must be unique canonical feature indices")
    by_index = {spec.index: spec for spec in feature_specs()}
    payload = {
        "schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "name": str(name),
        "indices": list(selected),
        "features": [asdict(by_index[index]) for index in selected],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()

    return payload


def full_feature_manifest() -> dict[str, Any]:
    return representation_manifest("canonical_s2_0_144", range(FEATURE_DIM))
