"""Rule engine: card text -> structured :class:`ParseEvent`s -> function tags.

Design notes
------------
* The engine is *card-type-aware*: it scans each text unit (Ability, attack,
  Tera/rule box, Item/Supporter/Stadium/Tool text, Energy text) with the rules
  appropriate to that section. ``ability_damage_counters`` can only come from an
  Ability; ``bench_attack_damage`` only from an attack; etc.
* It emits structured events *before* final tags so every tag is inspectable.
* Negative regex guards are first-class. The known v1 failures (Boss's Orders as
  spread, Rare Candy as lock, Scoop Up Cyclone as search, Energy Switch as
  acceleration, Battle Cage as ordinary-damage prevention, Tera as wall) are all
  prevented here by explicit guards, not by card-name special cases.
"""

from __future__ import annotations

import re

from deepmew.card_functions.schemas import (
    CARD_TYPE_POKEMON,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CardInput,
    ParseEvent,
    SECTION_ABILITY,
    SECTION_ATTACK,
    SECTION_ENERGY,
    SECTION_STADIUM,
    SECTION_TOOL,
    SUBTYPE_SPECIAL_ENERGY,
    SUBTYPE_STADIUM,
    SUBTYPE_TOOL,
)

_COUNTER_TAGS = {"bench_damage_counters", "active_damage_counters", "ability_damage_counters"}


def normalize(text: str) -> str:
    """Lowercase, fold smart punctuation, collapse whitespace (keep ``{X}``)."""
    if not text:
        return ""
    folded = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace(" ", " ")
        .replace("×", "x")
    )
    return " ".join(folded.lower().split())


def _snippet(text: str, match: re.Match | None, width: int = 90) -> str:
    if match is None:
        return text[:width]
    start = max(0, match.start() - 10)
    return text[start : start + width].strip()


def scan_card(card: CardInput) -> list[ParseEvent]:
    """Return all parse events for ``card`` (pre-suppression, ordered by scan)."""
    events: list[ParseEvent] = []
    units = _text_units(card)
    for section, text in units:
        _scan_unit(section, text, events)

    _scan_card_level(card, events)
    return events


def _text_units(card: CardInput) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for move in card.moves:
        if move.effect:
            units.append((move.kind, move.effect))
    if card.trainer_text:
        section = card.subtype or "item"
        units.append((section, card.trainer_text))
    if card.energy_text:
        units.append((SECTION_ENERGY, card.energy_text))
    return units


def _scan_unit(section: str, raw: str, events: list[ParseEvent]) -> None:
    t = normalize(raw)
    if not t:
        return

    def emit(tag: str, conf: str, rule_id: str, m: re.Match | None = None) -> None:
        events.append(ParseEvent(section, rule_id, tag, conf, _snippet(t, m)))

    use_restriction = bool(
        re.search(r"you can'?t (use|play) this card", t)
        or re.search(r"you cannot (use|play) this card", t)
    )
    damage_still_taken = "damage from attacks is still taken" in t or "damage is not an effect" in t

    _rule_damage_counters(section, t, emit)
    _rule_bench_attack_damage(section, t, emit)
    _rule_damage_math(section, t, emit, damage_still_taken)
    _rule_status(section, t, emit)
    _rule_switch_gust(section, t, emit)
    _rule_locks(section, t, emit, use_restriction)
    _rule_prevention(section, t, emit, damage_still_taken)
    _rule_draw_search_evolution(section, t, emit)
    _rule_energy(section, t, emit)
    _rule_recovery_pickup_heal(section, t, emit)
    _rule_disruption_scaling(section, t, emit)
    _rule_self_and_variance(section, t, emit)
    _rule_dependencies_and_engine(section, t, emit)


# --- damage counters (bench / active / ability) -------------------------------


def _rule_damage_counters(section: str, t: str, emit) -> None:
    place = re.search(r"\b(put|place)s?\b[^.]*?\bdamage counters?\b", t)
    move = re.search(r"\bmove[s]?\b[^.]*?\bdamage counters?\b", t)
    if place:
        target_match = re.search(r"damage counters?\s+on\s+([^.]*)", t)
        target = target_match.group(1) if target_match else ""
        opponent = "opponent" in target
        if section == SECTION_ABILITY and opponent:
            emit("ability_damage_counters", CONFIDENCE_HIGH, "ability_places_counters", place)
        elif not opponent and ("this pok" in target or "your" in target):
            emit("self_damage", CONFIDENCE_MEDIUM, "places_counters_on_self", place)
        elif opponent:
            if "bench" in target:
                emit("bench_damage_counters", CONFIDENCE_HIGH, "attack_counters_bench", place)
            elif "active" in target:
                emit("active_damage_counters", CONFIDENCE_HIGH, "attack_counters_active", place)
            else:
                emit("active_damage_counters", CONFIDENCE_MEDIUM, "attack_counters_flex", place)
    if move and "opponent" in t and not place:
        # Moving the opponent's *existing* counters around is damage manipulation,
        # not placement (e.g. Alakazam "Strange Hacking").
        emit("damage_modifier", CONFIDENCE_MEDIUM, "moves_opponent_counters", move)


# --- ordinary bench attack damage ---------------------------------------------


def _rule_bench_attack_damage(section: str, t: str, emit) -> None:
    if section != SECTION_ATTACK:
        return
    m = re.search(r"does\s+\d+\s+damage to\s+([^.]*)", t)
    if m and "bench" in m.group(1) and "opponent" in m.group(1) and "damage counter" not in m.group(1):
        emit("bench_attack_damage", CONFIDENCE_HIGH, "attack_damage_to_bench", m)
        return
    spread = re.search(r"does\s+\d+\s+damage to each of your opponent", t)
    if spread:
        emit("bench_attack_damage", CONFIDENCE_HIGH, "attack_damage_spread_all", spread)


# --- damage math (modifier / reduction) ---------------------------------------


def _rule_damage_math(section: str, t: str, emit, damage_still_taken: bool) -> None:
    if re.search(r"\d+\s+more damage", t):
        emit("damage_modifier", CONFIDENCE_HIGH, "conditional_more_damage", re.search(r"\d+\s+more damage", t))
    if re.search(r"isn'?t affected by weakness or resistance", t) or re.search(
        r"not affected by weakness", t
    ):
        emit("damage_modifier", CONFIDENCE_MEDIUM, "ignores_weakness_resistance", None)
    red = re.search(r"(take[s]?\s+\d+\s+less damage|reduce[sd]?\b[^.]*damage|\d+\s+less damage from attacks)", t)
    if red and not damage_still_taken:
        emit("damage_reduction", CONFIDENCE_MEDIUM, "reduces_damage_taken", red)


# --- status conditions --------------------------------------------------------

_STATUS_PATTERNS = {
    "poison": r"\bnow poisoned\b",
    "burn": r"\bnow burned\b",
    "paralysis": r"\bnow paralyzed\b",
    "confusion": r"\bnow confused\b",
    "sleep": r"\bnow asleep\b",
}


def _rule_status(section: str, t: str, emit) -> None:
    inflicted = False
    for tag, pattern in _STATUS_PATTERNS.items():
        m = re.search(pattern, t)
        if m:
            emit(tag, CONFIDENCE_HIGH, f"inflicts_{tag}", m)
            inflicted = True
    if inflicted:
        emit("status_condition", CONFIDENCE_HIGH, "inflicts_status", None)


# --- switching / gust ---------------------------------------------------------


def _rule_switch_gust(section: str, t: str, emit) -> None:
    gust = (
        re.search(r"switch\s+in\s+1\s+of\s+your\s+opponent", t)
        or re.search(r"switch\s+1\s+of\s+your\s+opponent'?s\s+benched[^.]*with[^.]*active", t)
        or re.search(r"switch[^.]*opponent'?s benched[^.]*to the active", t)
    )
    if gust:
        emit("gust_opponent", CONFIDENCE_HIGH, "gust_opponent_bench_to_active", gust)
    sw_opp = re.search(r"your opponent switches their active", t) or re.search(
        r"switch the defending pok[^.]*with 1 of (your opponent'?s|their) benched", t
    )
    if sw_opp:
        emit("switch_opponent", CONFIDENCE_MEDIUM, "forces_opponent_switch", sw_opp)
    sw_self = re.search(r"switch (your active pok[^.]*|this pok[^.]*) with 1 of your benched", t) or re.search(
        r"switch your active pok[^.]*with[^.]*your bench", t
    )
    if sw_self:
        emit("switch_self", CONFIDENCE_MEDIUM, "switches_own_active", sw_self)


# --- locks --------------------------------------------------------------------


def _rule_locks(section: str, t: str, emit, use_restriction: bool) -> None:
    m = re.search(r"can'?t retreat", t)
    if m:
        emit("retreat_lock", CONFIDENCE_HIGH, "defender_cant_retreat", m)
    m = re.search(r"can'?t play item cards", t)
    if m:
        emit("item_lock", CONFIDENCE_HIGH, "item_lock", m)
    m = re.search(r"can'?t .{0,30}pok[^.]*tool", t) or re.search(r"pok[^.]*tools[^.]*have no effect", t)
    if m:
        emit("tool_lock", CONFIDENCE_MEDIUM, "tool_lock", m)
    m = (
        re.search(r"abilities[^.]*have no effect", t)
        or re.search(r"can'?t use[^.]*abilities", t)
        or re.search(r"have no abilities", t)
    )
    if m:
        emit("ability_lock", CONFIDENCE_HIGH, "ability_lock", m)


# --- prevention / protection / wall -------------------------------------------


def _rule_prevention(section: str, t: str, emit, damage_still_taken: bool) -> None:
    prevent_counters = re.search(r"prevent all damage counters from being placed", t)
    prevent_dmg_eff = re.search(r"prevent all damage from and effects of attacks", t)
    prevent_damage = re.search(r"prevent all (of the )?damage\b(?!\s*counters)", t)
    prevent_effects = re.search(r"prevent all effects of attacks", t)
    bench_scope = "benched pok" in t
    self_bench = re.search(r"(as long as|while) this pok[^.]*is on your bench", t)

    if prevent_counters:
        emit("effect_prevention", CONFIDENCE_HIGH, "prevents_counter_placement", prevent_counters)
        if "bench" in t:
            emit("bench_protection", CONFIDENCE_HIGH, "protects_bench_from_counters", prevent_counters)

    if prevent_dmg_eff or prevent_damage or prevent_effects:
        anchor = prevent_dmg_eff or prevent_damage or prevent_effects
        if self_bench:
            emit("self_bench_protection", CONFIDENCE_HIGH, "tera_bench_immunity", self_bench)
            if prevent_effects or prevent_dmg_eff:
                emit("effect_prevention", CONFIDENCE_MEDIUM, "self_bench_effect_prevention", anchor)
        elif bench_scope:
            emit("bench_protection", CONFIDENCE_HIGH, "protects_own_bench", anchor)
            if prevent_effects or prevent_dmg_eff:
                emit("effect_prevention", CONFIDENCE_MEDIUM, "bench_effect_prevention", anchor)
        else:
            if prevent_effects or prevent_dmg_eff:
                emit("effect_prevention", CONFIDENCE_HIGH, "prevents_attack_effects", anchor)
            if (prevent_damage or prevent_dmg_eff) and not damage_still_taken:
                emit("wall_tank", CONFIDENCE_MEDIUM, "prevents_attack_damage_self", anchor)
                emit("attack_prevention", CONFIDENCE_MEDIUM, "prevents_attack_damage", anchor)

    m = re.search(r"can'?t (attack|use attacks)", t)
    if m:
        emit("attack_prevention", CONFIDENCE_HIGH, "cant_attack", m)


# --- draw / search / evolution ------------------------------------------------


def _rule_draw_search_evolution(section: str, t: str, emit) -> None:
    m = re.search(r"draw \d+ cards?", t) or re.search(r"draw a card", t)
    if m:
        emit("draw", CONFIDENCE_HIGH, "draws_cards", m)

    search_deck = "search your deck" in t
    if search_deck:
        if "evolves from" in t or "that evolves" in t or re.search(r"stage [12][^.]*evolv", t):
            emit("evolution_search", CONFIDENCE_HIGH, "searches_for_evolution", None)
        else:
            emit("search", CONFIDENCE_HIGH, "searches_deck", None)

    m = re.search(r"skipping the stage", t)
    if m:
        emit("evolution_skip", CONFIDENCE_HIGH, "skips_evolution_stage", m)
    enabler = re.search(r"put that card onto[^.]*evolve", t)
    if enabler:
        emit("evolution_enabler", CONFIDENCE_HIGH, "puts_evolution_onto_pokemon", enabler)
    elif (
        "evolve" in t
        and "onto" in t
        and "when you play this pok" not in t
        and section in {"item", "supporter", SECTION_ABILITY}
    ):
        emit("evolution_enabler", CONFIDENCE_MEDIUM, "enables_evolution", None)


# --- energy -------------------------------------------------------------------


def _rule_energy(section: str, t: str, emit) -> None:
    reposition = re.search(r"move\b[^.]*\benergy\b[^.]*\bfrom\b[^.]*\bto\b[^.]*your", t) or re.search(
        r"move (a|an|1|\d+)[^.]*energy from 1 of your pok", t
    )
    if reposition:
        emit("energy_reposition", CONFIDENCE_HIGH, "moves_energy_own_to_own", reposition)

    if not reposition:
        accel = re.search(r"attach[^.]*energy[^.]*from your (hand|deck|discard pile)", t) or re.search(
            r"attach \d+[^.]*energy[^.]*to (your|1 of your)[^.]*pok", t
        )
        if accel and "attach this card" not in t:
            emit("energy_acceleration", CONFIDENCE_HIGH, "accelerates_energy", accel)

    rec = re.search(r"energy[^.]*from your discard pile[^.]*into your hand", t)
    if rec:
        emit("energy_recovery", CONFIDENCE_HIGH, "recovers_energy_to_hand", rec)

    denial = re.search(r"discard\b[^.]*\benergy\b[^.]*\bopponent", t) or re.search(
        r"opponent'?s pok[^.]*discard[^.]*energy", t
    )
    if denial:
        emit("energy_denial", CONFIDENCE_HIGH, "discards_opponent_energy", denial)

    if "special energy" in t or section == SECTION_ENERGY and "this card provides" in t and "every type" in t:
        emit("special_energy_dependency", CONFIDENCE_MEDIUM, "references_special_energy", None)


# --- recovery / pickup / healing ----------------------------------------------


def _rule_recovery_pickup_heal(section: str, t: str, emit) -> None:
    m = (
        re.search(r"heal \d+ damage", t)
        or re.search(r"remove \d+ damage counters?", t)
        or re.search(r"remove all special conditions", t)
    )
    if m:
        emit("healing", CONFIDENCE_HIGH, "heals_or_removes_counters", m)

    pickup = re.search(r"put (1 of your pok[^.]*|this pok[^.]*)[^.]*into your hand", t)
    if pickup and "discard pile" not in pickup.group(0):
        emit("pokemon_pickup", CONFIDENCE_HIGH, "picks_up_own_pokemon", pickup)
        emit("pokemon_recovery", CONFIDENCE_MEDIUM, "returns_board_pokemon_to_hand", pickup)

    rec = re.search(r"pok[^.]*from your discard pile[^.]*(into your hand|onto your bench)", t)
    if rec:
        emit("pokemon_recovery", CONFIDENCE_HIGH, "recovers_pokemon_from_discard", rec)

    disc = re.search(r"from your discard pile[^.]*into your hand", t)
    if disc and not pickup:
        emit("discard_recovery", CONFIDENCE_MEDIUM, "recovers_cards_from_discard", disc)


# --- disruption / scaling -----------------------------------------------------


def _rule_disruption_scaling(section: str, t: str, emit) -> None:
    hand = (
        re.search(r"each player shuffles their hand", t)
        or re.search(r"your opponent shuffles their hand", t)
        or re.search(r"discard[^.]*your opponent'?s hand", t)
        or re.search(r"opponent reveals their hand", t)
    )
    if hand:
        emit("hand_disruption", CONFIDENCE_MEDIUM, "disrupts_opponent_hand", hand)

    deck = re.search(r"discard[^.]*from the top of your opponent'?s deck", t)
    if deck:
        emit("deck_disruption", CONFIDENCE_MEDIUM, "mills_opponent_deck", deck)

    m = re.search(r"for each card in your hand", t) or re.search(r"number of cards in your hand", t)
    if m:
        emit("hand_size_scaling", CONFIDENCE_HIGH, "scales_with_hand_size", m)

    m = (
        re.search(r"for each[^.]*prize card", t)
        or re.search(r"number of prize cards", t)
        or re.search(r"prize cards? you (have )?taken", t)
    )
    if m:
        emit("prize_count_scaling", CONFIDENCE_HIGH, "scales_with_prizes", m)

    m = re.search(r"for each[^.]*in your discard pile", t)
    if m:
        emit("discard_scaling", CONFIDENCE_HIGH, "scales_with_discard", m)


# --- self-cost / variance -----------------------------------------------------


def _rule_self_and_variance(section: str, t: str, emit) -> None:
    if "flip a coin" in t:
        emit("coin_flip_variance", CONFIDENCE_HIGH, "coin_flip", re.search(r"flip a coin", t))
    m = re.search(r"this pok[^.]*is knocked out", t)
    if m:
        emit("self_ko", CONFIDENCE_HIGH, "self_knocks_out", m)
    m = re.search(r"put \d+ damage counters? on this pok", t) or re.search(r"\d+ damage to itself", t)
    if m:
        emit("self_damage", CONFIDENCE_MEDIUM, "damages_self", m)
    m = (
        re.search(r"discard your hand", t)
        or re.search(r"discard[^.]*energy[^.]*from this pok", t)
        or re.search(r"discard \d+ cards? from your hand", t)
    )
    if m:
        emit("self_discard", CONFIDENCE_MEDIUM, "discards_own_cards", m)


# --- dependencies / ability engine --------------------------------------------


def _rule_dependencies_and_engine(section: str, t: str, emit) -> None:
    if section == SECTION_STADIUM:
        emit("stadium_dependency", CONFIDENCE_MEDIUM, "is_stadium", None)
    elif "stadium" in t:
        emit("stadium_dependency", CONFIDENCE_LOW, "references_stadium", None)
    if section == SECTION_TOOL:
        emit("tool_dependency", CONFIDENCE_MEDIUM, "is_pokemon_tool", None)
    elif "pok" in t and "tool" in t and re.search(r"pok[^.]*tool", t):
        emit("tool_dependency", CONFIDENCE_LOW, "references_tool", None)
    if section == SECTION_ABILITY and any(
        k in t for k in ("draw", "search your deck", "attach", "into your hand", "look at")
    ):
        emit("ability_engine", CONFIDENCE_MEDIUM, "ability_provides_engine", None)


# --- card-level rules (attacker / finisher) -----------------------------------


def _scan_card_level(card: CardInput, events: list[ParseEvent], finisher_min_damage: int = 180) -> None:
    if card.card_type != CARD_TYPE_POKEMON:
        return
    deals_counters = any(e.tag in _COUNTER_TAGS for e in events)
    has_bench_dmg = any(e.tag == "bench_attack_damage" for e in events)
    max_damage = card.max_printed_damage
    if max_damage > 0 or deals_counters or has_bench_dmg:
        conf = CONFIDENCE_HIGH if max_damage > 0 else CONFIDENCE_MEDIUM
        events.append(ParseEvent("card", "has_damaging_action", "attacker", conf, f"max_damage={max_damage}"))
    if max_damage >= finisher_min_damage:
        events.append(
            ParseEvent("card", "high_printed_damage", "finisher", CONFIDENCE_MEDIUM, f"max_damage={max_damage}")
        )
    if card.attack_texts:
        for atk in card.attack_texts:
            t = normalize(atk.effect)
            if t and any(k in t for k in ("search your deck", "draw", "attach", "evolve", "switch")):
                events.append(
                    ParseEvent(SECTION_ATTACK, "attack_has_setup", "setup_attacker", CONFIDENCE_MEDIUM, t[:90])
                )
                break
