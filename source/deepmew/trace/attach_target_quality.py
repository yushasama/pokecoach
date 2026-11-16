"""Attach target quality audit built on deterministic trace probes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from cg.api import AreaType, OptionType, all_card_data

from deepmew.arena import PairSummary, run_pair
from deepmew.artifacts.run_dir import safe_slug
from deepmew.deck_profiles import KYOGRE_CARD_ID, MEGA_ABOMASNOW_CARD_ID, MEGA_LINE_IDS, SNOVER_CARD_ID
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board, attachment_effect
from deepmew.trace.diff import diff_probe_sets
from deepmew.trace.probe import probe_frames, read_frames, resolve_frames_path
from deepmew.trace.record import TraceRecorder
from deepmew.trace.schema import PolicyDiff, PolicyProbe


TARGET_QUALITY_SCHEMA_VERSION = "deepmew.deck_attach_target_quality.v1"
TARGET_EVENT_SCHEMA_VERSION = "deepmew.attach_target_quality_event.v1"
RUN_KIND = "deck_attach_target_quality"

DEFAULT_POLICIES = [
    "v2.4",
    "first-legal",
    "v2.4[attach_target=active]",
    "v2.4[attach_target=bench]",
    "v2.4[attach_target=first]",
    "v2.4[semantics=off]",
]
DEFAULT_MATCHUPS = [
    ("v2.4", "first-legal"),
    ("v2.4", "v2.4[attach_target=active]"),
    ("v2.4", "v2.4[attach_target=bench]"),
    ("v2.4", "v2.4[attach_target=first]"),
]
SEMANTICS_CONTROL_MATCHUP = ("v2.4", "v2.4[semantics=off]")
ATTACH_TARGET_POLICIES = [
    "v2.4[attach_target=active]",
    "v2.4[attach_target=bench]",
    "v2.4[attach_target=first]",
]
ATTACH_TARGET_BUCKETS = [
    "best_visible_target_chosen",
    "active_target_chosen",
    "bench_target_chosen",
    "mega_line_target_chosen",
    "kyogre_target_chosen",
    "unknown_target_chosen",
    "better_visible_target_available",
    "mega_line_underfed_possible",
    "kyogre_overfed_possible",
    "already_ready_overattached_possible",
    "near_ready_target_missed",
    "active_bias_likely_helpful",
    "bench_bias_likely_helpful",
    "first_target_order_likely_helpful",
    "timing_more_important_than_target",
    "target_more_important_than_timing",
    "option_order_more_important_than_target",
    "unknown",
]
BETTER_TARGET_MARGIN = 12
REQUIRED_EXAMPLE_KEYS = [
    "v2.4_skipped_attach_while_first_legal_attached",
    "v2.4_attached_to_kyogre_while_mega_line_looked_underfed",
    "v2.4_attached_to_already_ready_target",
    "active_or_first_ablation_chose_best_visible_target",
    "target_quality_uncertain_frame",
]

_CARD_NAME_BY_ID: dict[int, str] | None = None


@dataclass(frozen=True)
class TargetCandidate:
    option_index: int
    area: str | None
    in_play_index: int | None
    card_id: int | None
    card_name: str | None
    energy_count: int | None
    before_shortfall: int | None
    after_shortfall: int | None
    best_damage: int | None
    already_ready: bool
    near_ready: bool
    far_from_ready: bool
    known_target: bool
    improves_readiness: bool
    enables_attack: bool
    target_is_closest: bool
    roles: tuple[str, ...]
    primary_role: str
    audit_score: int
    score_reasons: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return dict(value)


def _payloads(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [_payload(value) for value in values]


def _probe_key(probe: dict[str, Any]) -> tuple[str, str]:
    return str(probe.get("frame_id") or ""), str(probe.get("policy_spec") or "")


def _chosen_options(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
    if probe is None:
        return []
    return list(probe.get("chosen_option_summaries") or [])


def _chosen_indices(probe: dict[str, Any] | None) -> list[int]:
    if probe is None:
        return []
    indices: list[int] = []
    for value in probe.get("chosen_indices") or []:
        if isinstance(value, int):
            indices.append(value)
    return indices


def _chosen_types(probe: dict[str, Any] | None) -> set[str]:
    return {str(option.get("option_type")) for option in _chosen_options(probe)}


def _first_chosen_type(probe: dict[str, Any] | None) -> str:
    options = _chosen_options(probe)
    if not options:
        return "none"
    return str(options[0].get("option_type") or "unknown")


def _chooses_attach(probe: dict[str, Any] | None) -> bool:
    return "ATTACH" in _chosen_types(probe)


def _same_choice(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return False
    return _chosen_indices(left) == _chosen_indices(right) and not left.get("error") and not right.get("error")


def _raw_observation(frame: dict[str, Any]) -> dict[str, Any] | None:
    raw = frame.get("raw_observation")
    return raw if isinstance(raw, dict) else None


def _raw_select(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame)
    if raw is None:
        return {}
    select = raw.get("select")
    return select if isinstance(select, dict) else {}


def _raw_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    options = _raw_select(frame).get("option") or []
    return [option for option in options if isinstance(option, dict)]


def _legal_options(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return list(frame.get("legal_options") or [])


def _legal_attach_indices(frame: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for option in _legal_options(frame):
        if str(option.get("option_type")) != "ATTACH":
            continue
        value = option.get("index")
        if isinstance(value, int):
            indices.append(value)
    if indices:
        return indices
    raw_indices: list[int] = []
    for index, option in enumerate(_raw_options(frame)):
        if option.get("type") == OptionType.ATTACH.value:
            raw_indices.append(index)
    return raw_indices


def _attach_legal(frame: dict[str, Any]) -> bool:
    board = frame.get("board_summary") or {}
    return bool(board.get("attach_legal")) or bool(_legal_attach_indices(frame))


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return AreaType(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _card_names() -> dict[int, str]:
    global _CARD_NAME_BY_ID
    if _CARD_NAME_BY_ID is None:
        names: dict[int, str] = {}
        for card in all_card_data():
            card_id = _int_or_none(getattr(card, "cardId", getattr(card, "id", None)))
            if card_id is None:
                continue
            name = getattr(card, "name", getattr(card, "cardName", None))
            if name:
                names[card_id] = str(name)
        _CARD_NAME_BY_ID = names
    return _CARD_NAME_BY_ID


def _card_name(card_id: int | None) -> str | None:
    if card_id is None:
        return None
    return _card_names().get(card_id)


def _own_player(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame) or {}
    current = raw.get("current") or {}
    players = current.get("players") or []
    your_index = _int_or_none(current.get("yourIndex"))
    if your_index is None:
        your_index = _int_or_none(frame.get("seat")) or 0
    if your_index < 0 or your_index >= len(players):
        return {}
    player = players[your_index]
    return player if isinstance(player, dict) else {}


def _energy_count(pokemon: dict[str, Any]) -> int:
    energy_cards = pokemon.get("energyCards")
    if isinstance(energy_cards, list):
        return len(energy_cards)
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)
    return int(pokemon.get("energy_count") or 0)


def _pokemon_card_id(pokemon: dict[str, Any] | None) -> int | None:
    if not pokemon:
        return None
    return _int_or_none(pokemon.get("id", pokemon.get("card_id")))


def _own_pokemon(frame: dict[str, Any]) -> list[dict[str, Any]]:
    player = _own_player(frame)
    pokemon: list[dict[str, Any]] = []
    active = player.get("active") or []
    if active and isinstance(active[0], dict):
        pokemon.append({"area": "ACTIVE", "index": 0, **active[0]})
    for index, item in enumerate(player.get("bench") or []):
        if isinstance(item, dict):
            pokemon.append({"area": "BENCH", "index": index, **item})
    if pokemon:
        return pokemon

    board = frame.get("board_summary") or {}
    own_active = board.get("own_active")
    if isinstance(own_active, dict) and own_active.get("card_id") is not None:
        pokemon.append(
            {
                "area": "ACTIVE",
                "index": 0,
                "id": own_active.get("card_id"),
                "energy_count": own_active.get("energy_count") or 0,
            }
        )
    for index, item in enumerate(board.get("own_bench") or []):
        if isinstance(item, dict):
            pokemon.append(
                {
                    "area": "BENCH",
                    "index": index,
                    "id": item.get("card_id"),
                    "energy_count": item.get("energy_count") or 0,
                }
            )
    return pokemon


def _pokemon_at(frame: dict[str, Any], area: str | None, index: int | None) -> dict[str, Any] | None:
    if area is None or index is None:
        return None
    for pokemon in _own_pokemon(frame):
        if str(pokemon.get("area")) == area and int(pokemon.get("index") or 0) == index:
            return pokemon
    return None


def _attach_target_from_option(frame: dict[str, Any], option_index: int) -> tuple[str | None, int | None, dict[str, Any] | None]:
    options = _raw_options(frame)
    if not (0 <= option_index < len(options)):
        return None, None, None
    option = options[option_index]
    area = _area_name(option.get("inPlayArea"))
    index = _int_or_none(option.get("inPlayIndex"))
    if index is None:
        index = 0
    return area, index, _pokemon_at(frame, area, index)


def _readiness_by_target(frame: dict[str, Any]) -> dict[tuple[str, int], Any]:
    raw = _raw_observation(frame)
    if raw is None:
        return {}
    try:
        plan = analyze_board(raw)
    except Exception:
        return {}
    return {(item.area, item.index): item for item in plan.readiness}


def _attach_effects(frame: dict[str, Any]) -> dict[int, Any]:
    raw = _raw_observation(frame)
    if raw is None:
        return {}
    try:
        plan = analyze_board(raw)
    except Exception:
        return {}
    options = _raw_options(frame)
    effects: dict[int, Any] = {}
    for index in _legal_attach_indices(frame):
        if 0 <= index < len(options):
            effects[index] = attachment_effect(plan, options[index])
    return effects


def _is_mega_line(card_id: int | None) -> bool:
    return card_id in MEGA_LINE_IDS


def _roles_for_candidate(
    *,
    area: str | None,
    card_id: int | None,
    before_shortfall: int | None,
    known_target: bool,
) -> tuple[str, ...]:
    roles: list[str] = []
    if card_id == MEGA_ABOMASNOW_CARD_ID:
        roles.append("mega_abomasnow")
    if card_id == SNOVER_CARD_ID:
        roles.append("snover")
    if _is_mega_line(card_id):
        roles.append("mega_line")
    if card_id == KYOGRE_CARD_ID:
        roles.append("kyogre")
    if area == "ACTIVE":
        roles.append("active_attacker")
    elif area == "BENCH":
        roles.append("bench_attacker")

    if before_shortfall == 0:
        roles.append("already_ready")
    elif before_shortfall == 1:
        roles.append("near_ready")
    elif before_shortfall is not None:
        roles.append("far_from_ready")

    if not known_target or not roles:
        roles.append("unknown")
    return tuple(dict.fromkeys(roles))


def _primary_role(candidate: TargetCandidate | dict[str, Any]) -> str:
    roles = tuple(candidate.roles if isinstance(candidate, TargetCandidate) else candidate.get("roles") or ())
    for role in ("mega_abomasnow", "snover", "mega_line", "kyogre", "active_attacker", "bench_attacker", "unknown"):
        if role in roles:
            return role
    return roles[0] if roles else "unknown"


def _base_candidate_score(
    *,
    area: str | None,
    card_id: int | None,
    energy_count: int | None,
    before_shortfall: int | None,
    after_shortfall: int | None,
    best_damage: int | None,
    known_target: bool,
    improves_readiness: bool,
    enables_attack: bool,
    target_is_closest: bool,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if not known_target:
        return -20, ["unknown-target"]

    if area == "ACTIVE":
        score += 8
        reasons.append("active-target")
    elif area == "BENCH":
        score += 4
        reasons.append("bench-target")

    if card_id == MEGA_ABOMASNOW_CARD_ID:
        score += 36
        reasons.append("mega-abomasnow-plan")
    elif card_id == SNOVER_CARD_ID:
        score += 30
        reasons.append("snover-mega-line-plan")
    elif card_id == KYOGRE_CARD_ID:
        score += 16
        reasons.append("kyogre-backup-plan")
    elif card_id is not None:
        score += 2
        reasons.append("visible-pokemon-target")

    if before_shortfall == 0:
        score -= 30
        reasons.append("already-attack-ready")
    elif before_shortfall == 1:
        score += 38
        reasons.append("one-energy-from-attack")
    elif before_shortfall is not None and after_shortfall is not None and after_shortfall < before_shortfall:
        score += 14
        reasons.append("reduces-visible-shortfall")

    if improves_readiness:
        score += 10
        reasons.append("improves-readiness")
    if enables_attack:
        score += 18
        reasons.append("enables-attack")
    if target_is_closest:
        score += 10
        reasons.append("closest-visible-attacker")
    if best_damage is not None and best_damage >= 100:
        score += 8
        reasons.append("meaningful-visible-damage")
    if _is_mega_line(card_id) and (energy_count or 0) < 2:
        score += 14
        reasons.append("mega-line-under-two-energy")
    return score, reasons


def build_target_candidates(frame: dict[str, Any]) -> list[TargetCandidate]:
    """Build visible attach targets and audit-only scores for one frame."""

    effects = _attach_effects(frame)
    readiness = _readiness_by_target(frame)
    candidates: list[TargetCandidate] = []
    for option_index in _legal_attach_indices(frame):
        area, in_play_index, pokemon = _attach_target_from_option(frame, option_index)
        target_readiness = readiness.get((area or "", in_play_index or 0))
        effect = effects.get(option_index)
        known_target = pokemon is not None
        card_id = _pokemon_card_id(pokemon)
        energy_count = _energy_count(pokemon) if pokemon is not None else None
        before_shortfall = getattr(effect, "before_shortfall", None)
        after_shortfall = getattr(effect, "after_shortfall", None)
        best_damage = getattr(target_readiness, "best_damage", None)
        improves_readiness = bool(getattr(effect, "improves_readiness", False))
        enables_attack = bool(getattr(effect, "enables_attack", False))
        target_is_closest = bool(getattr(effect, "target_is_closest", False))
        if before_shortfall is None and target_readiness is not None:
            before_shortfall = target_readiness.best_shortfall
            after_shortfall = max(0, before_shortfall - 1) if before_shortfall is not None else None
        already_ready = before_shortfall == 0
        near_ready = before_shortfall == 1
        far_from_ready = before_shortfall is not None and before_shortfall > 1
        score, reasons = _base_candidate_score(
            area=area,
            card_id=card_id,
            energy_count=energy_count,
            before_shortfall=before_shortfall,
            after_shortfall=after_shortfall,
            best_damage=best_damage,
            known_target=known_target,
            improves_readiness=improves_readiness,
            enables_attack=enables_attack,
            target_is_closest=target_is_closest,
        )
        roles = _roles_for_candidate(
            area=area,
            card_id=card_id,
            before_shortfall=before_shortfall,
            known_target=known_target,
        )
        candidate = TargetCandidate(
            option_index=option_index,
            area=area,
            in_play_index=in_play_index,
            card_id=card_id,
            card_name=_card_name(card_id),
            energy_count=energy_count,
            before_shortfall=before_shortfall,
            after_shortfall=after_shortfall,
            best_damage=best_damage,
            already_ready=already_ready,
            near_ready=near_ready,
            far_from_ready=far_from_ready,
            known_target=known_target,
            improves_readiness=improves_readiness,
            enables_attack=enables_attack,
            target_is_closest=target_is_closest,
            roles=roles,
            primary_role="unknown",
            audit_score=score,
            score_reasons=tuple(reasons),
        )
        candidates.append(candidate)

    adjusted: list[TargetCandidate] = []
    mega_underfed = any(candidate.card_id in MEGA_LINE_IDS and not candidate.already_ready for candidate in candidates)
    non_ready_available = any(candidate.known_target and not candidate.already_ready for candidate in candidates)
    for candidate in candidates:
        score = candidate.audit_score
        reasons = list(candidate.score_reasons)
        if mega_underfed and candidate.card_id == KYOGRE_CARD_ID:
            score -= 18
            reasons.append("mega-line-visible-need-over-kyogre")
        if non_ready_available and candidate.already_ready:
            score -= 12
            reasons.append("underdeveloped-target-available")
        adjusted_candidate = TargetCandidate(
            **{
                **candidate.to_jsonable(),
                "primary_role": _primary_role(candidate),
                "audit_score": score,
                "score_reasons": tuple(reasons),
                "roles": tuple(candidate.roles),
            }
        )
        adjusted.append(adjusted_candidate)
    return adjusted


def _candidate_by_index(candidates: list[TargetCandidate]) -> dict[int, TargetCandidate]:
    return {candidate.option_index: candidate for candidate in candidates}


def _best_candidates(candidates: list[TargetCandidate]) -> list[TargetCandidate]:
    if not candidates:
        return []
    best_score = max(candidate.audit_score for candidate in candidates)
    return [candidate for candidate in candidates if candidate.audit_score == best_score]


def _clearly_better_candidates(selected: TargetCandidate | None, candidates: list[TargetCandidate]) -> list[TargetCandidate]:
    if selected is None:
        return []
    return [candidate for candidate in candidates if candidate.audit_score >= selected.audit_score + BETTER_TARGET_MARGIN]


def _mega_line_underfed(candidates: list[TargetCandidate]) -> bool:
    return any(
        candidate.card_id in MEGA_LINE_IDS
        and candidate.known_target
        and not candidate.already_ready
        and (candidate.energy_count or 0) < 2
        for candidate in candidates
    )


def _near_ready_candidates(candidates: list[TargetCandidate]) -> list[TargetCandidate]:
    return [candidate for candidate in candidates if candidate.near_ready]


def _candidate_summary(candidate: TargetCandidate | None) -> dict[str, Any] | None:
    return None if candidate is None else candidate.to_jsonable()


def classify_attach_target_choice(
    frame: dict[str, Any],
    probe: dict[str, Any] | None,
    *,
    candidates: list[TargetCandidate] | None = None,
    all_probes: dict[str, dict[str, Any]] | None = None,
    default_probe: dict[str, Any] | None = None,
    first_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one policy's attach target choice on a single frame."""

    candidates = candidates if candidates is not None else build_target_candidates(frame)
    all_probes = all_probes or {}
    by_index = _candidate_by_index(candidates)
    chosen_indices = _chosen_indices(probe)
    selected_indices = [index for index in chosen_indices if index in by_index]
    selected_candidate = by_index.get(selected_indices[0]) if selected_indices else None
    best_candidates = _best_candidates(candidates)
    best_indices = {candidate.option_index for candidate in best_candidates}
    better_candidates = _clearly_better_candidates(selected_candidate, candidates)
    buckets: set[str] = set()
    policy = str((probe or {}).get("policy_spec") or "unknown")
    attached = selected_candidate is not None
    attach_legal = _attach_legal(frame)

    if attached:
        if selected_candidate.option_index in best_indices:
            buckets.add("best_visible_target_chosen")
        if selected_candidate.area == "ACTIVE":
            buckets.add("active_target_chosen")
        elif selected_candidate.area == "BENCH":
            buckets.add("bench_target_chosen")
        if selected_candidate.card_id in MEGA_LINE_IDS:
            buckets.add("mega_line_target_chosen")
        if selected_candidate.card_id == KYOGRE_CARD_ID:
            buckets.add("kyogre_target_chosen")
        if not selected_candidate.known_target or "unknown" in selected_candidate.roles:
            buckets.add("unknown_target_chosen")
        if better_candidates:
            buckets.add("better_visible_target_available")
            buckets.add("target_more_important_than_timing")
        if _mega_line_underfed(candidates) and selected_candidate.card_id not in MEGA_LINE_IDS:
            buckets.add("mega_line_underfed_possible")
        if _mega_line_underfed(candidates) and selected_candidate.card_id == KYOGRE_CARD_ID:
            buckets.add("kyogre_overfed_possible")
        if selected_candidate.already_ready and any(
            candidate.known_target and not candidate.already_ready for candidate in candidates
        ):
            buckets.add("already_ready_overattached_possible")
        near_ready = _near_ready_candidates(candidates)
        if near_ready and selected_candidate.option_index not in {candidate.option_index for candidate in near_ready}:
            buckets.add("near_ready_target_missed")
    elif attach_legal and policy == "v2.4":
        buckets.add("timing_more_important_than_target")
        if _mega_line_underfed(candidates):
            buckets.add("mega_line_underfed_possible")
        if _near_ready_candidates(candidates):
            buckets.add("near_ready_target_missed")

    if policy == "v2.4[attach_target=active]" and attached:
        if selected_candidate.option_index in best_indices:
            buckets.add("active_bias_likely_helpful")
        elif first_probe is not None and _same_choice(probe, first_probe) and not _same_choice(default_probe, first_probe):
            buckets.add("active_bias_likely_helpful")
    if policy == "v2.4[attach_target=bench]" and attached:
        if selected_candidate.option_index in best_indices or selected_candidate.card_id in MEGA_LINE_IDS:
            buckets.add("bench_bias_likely_helpful")
    if policy == "v2.4[attach_target=first]" and attached:
        if first_probe is not None and _same_choice(probe, first_probe) and not _same_choice(default_probe, first_probe):
            buckets.add("first_target_order_likely_helpful")
        elif selected_candidate.option_index == min(_legal_attach_indices(frame) or [selected_candidate.option_index]):
            buckets.add("first_target_order_likely_helpful")
        if selected_candidate.option_index not in best_indices:
            buckets.add("option_order_more_important_than_target")

    if policy == "first-legal" and attached:
        first_target_probe = all_probes.get("v2.4[attach_target=first]")
        if first_target_probe is not None and _same_choice(probe, first_target_probe):
            if selected_candidate.option_index not in best_indices:
                buckets.add("option_order_more_important_than_target")

    buckets.discard("unknown")
    if not buckets:
        buckets.add("unknown")

    return {
        "schema_version": TARGET_EVENT_SCHEMA_VERSION,
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "seat": frame.get("seat"),
        "select_context": frame.get("select_context"),
        "policy_spec": policy,
        "chosen_indices": chosen_indices,
        "chosen_type": _first_chosen_type(probe),
        "attach_legal": attach_legal,
        "attached": attached,
        "missed_attach": attach_legal and policy == "v2.4" and not attached,
        "selected_attach_indices": selected_indices,
        "selected_candidate": _candidate_summary(selected_candidate),
        "best_candidate_indices": sorted(best_indices),
        "best_candidates": [candidate.to_jsonable() for candidate in best_candidates],
        "better_candidates": [candidate.to_jsonable() for candidate in better_candidates],
        "candidates": [candidate.to_jsonable() for candidate in candidates],
        "buckets": sorted(buckets),
    }


def run_trace_matchups(
    *,
    run_id: str,
    matchups: list[tuple[str, str]],
    games: int,
    seed: int,
    max_steps: int,
    limit_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[PairSummary], dict[str, Any]]:
    recorder = TraceRecorder(run_id=run_id, sample_limit=limit_frames)
    pair_summaries: list[PairSummary] = []
    for index, (agent_a, agent_b) in enumerate(matchups):
        pair_seed = seed + index * 10007
        prefix = f"decktarget-{index:02d}-{safe_slug(agent_a)}-vs-{safe_slug(agent_b)}"
        pair_summaries.append(
            run_pair(
                agent_a,
                agent_b,
                games=games,
                max_steps=max_steps,
                seed=pair_seed,
                trace_recorder=recorder,
                game_id_prefix=prefix,
            )
        )
    return [frame.to_jsonable() for frame in recorder.frames], pair_summaries, recorder.summary()


def read_probe_run(path: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_path = Path(path)
    probes_path = run_path / "probes.jsonl" if run_path.is_dir() else run_path
    diffs_path = run_path / "diffs.jsonl" if run_path.is_dir() else run_path.with_name("diffs.jsonl")
    return load_jsonl(probes_path), load_jsonl(diffs_path)


def build_probe_outputs(
    frames: list[dict[str, Any]],
    policies: list[str],
    *,
    base_policy: str = "v2.4",
) -> tuple[list[PolicyProbe], list[PolicyDiff]]:
    probes = probe_frames(frames, policies)
    diffs = diff_probe_sets(probes, base_policy=base_policy)
    return probes, diffs


def _summarize_pair(pair: PairSummary) -> dict[str, Any]:
    return {
        "agent_a": pair.agent_a,
        "agent_b": pair.agent_b,
        "games": pair.games,
        "completed": pair.completed,
        "a_wins": pair.a_wins,
        "b_wins": pair.b_wins,
        "draws": pair.draws,
        "crashes": pair.crashes,
        "timeouts": pair.timeouts,
        "avg_turns": pair.avg_turns,
        "elapsed_seconds": pair.elapsed_seconds,
    }


def _role_distribution(events: list[dict[str, Any]], policy: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for event in events:
        if event["policy_spec"] != policy or not event["attached"]:
            continue
        selected = event.get("selected_candidate") or {}
        counter[str(selected.get("primary_role") or "unknown")] += 1
    return dict(counter)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _counter_rows(counter_dict: dict[str, Any], *, limit: int = 12) -> list[list[Any]]:
    counter = Counter({str(key): int(value) for key, value in counter_dict.items()})
    rows = [[key, value] for key, value in counter.most_common(limit)]
    return rows or [["none", 0]]


def _event_has(event: dict[str, Any], bucket: str) -> bool:
    return bucket in set(event.get("buckets") or [])


def _add_example(
    examples: dict[str, list[dict[str, Any]]],
    key: str,
    event: dict[str, Any],
    *,
    limit: int = 5,
) -> None:
    if len(examples[key]) >= limit:
        return
    examples[key].append(
        {
            "frame_id": event.get("frame_id"),
            "game_id": event.get("game_id"),
            "turn": event.get("turn"),
            "seat": event.get("seat"),
            "policy_spec": event.get("policy_spec"),
            "chosen_type": event.get("chosen_type"),
            "selected_candidate": event.get("selected_candidate"),
            "best_candidate_indices": event.get("best_candidate_indices"),
            "buckets": event.get("buckets"),
        }
    )


def _conclusion(*, missed_attach: int, default_attaches: int, default_better: int, first_order: int) -> str:
    if missed_attach >= max(10, default_better * 2):
        return "attach timing appears larger than target quality, possible and not proven causal"
    if default_better >= max(5, missed_attach // 2):
        return "attach target quality appears competitive with timing, possible and not proven causal"
    if first_order >= max(5, default_better + missed_attach // 4):
        return "option order may be carrying value, possible and not proven causal"
    if missed_attach or default_better or first_order:
        return "mixed timing, target, and option-order evidence, not proven causal"
    return "unknown, insufficient visible-state evidence"


def analyze_target_quality(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    probes: Iterable[Any],
    diffs: Iterable[Any],
    policies: list[str],
    pair_summaries: Iterable[PairSummary] | None = None,
    trace_source: str | None = None,
    probe_source: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probe_payloads = _payloads(probes)
    diff_payloads = _payloads(diffs)
    pair_payloads = [_summarize_pair(pair) for pair in pair_summaries or []]
    probes_by_frame: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for probe in probe_payloads:
        frame_id, policy = _probe_key(probe)
        probes_by_frame[frame_id][policy] = probe

    attach_legal_frames = 0
    events: list[dict[str, Any]] = []
    bucket_counter: Counter[str] = Counter()
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_required: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame in frames:
        if not _attach_legal(frame):
            continue
        attach_legal_frames += 1
        frame_id = str(frame.get("frame_id") or "")
        frame_probes = probes_by_frame.get(frame_id, {})
        candidates = build_target_candidates(frame)
        default_probe = frame_probes.get("v2.4")
        first_probe = frame_probes.get("first-legal")
        for policy in policies:
            probe = frame_probes.get(policy)
            if probe is None or probe.get("error"):
                continue
            event = classify_attach_target_choice(
                frame,
                probe,
                candidates=candidates,
                all_probes=frame_probes,
                default_probe=default_probe,
                first_probe=first_probe,
            )
            # Keep the event stream focused on actual ATTACH choices and default missed attach frames.
            if not event["attached"] and not event["missed_attach"]:
                continue
            events.append(event)
            bucket_counter.update(event["buckets"])
            for bucket in event["buckets"]:
                _add_example(examples_by_bucket, bucket, event)

            if event["policy_spec"] == "v2.4" and event["missed_attach"] and _chooses_attach(first_probe):
                _add_example(examples_required, "v2.4_skipped_attach_while_first_legal_attached", event)
            if event["policy_spec"] == "v2.4" and _event_has(event, "kyogre_overfed_possible"):
                _add_example(examples_required, "v2.4_attached_to_kyogre_while_mega_line_looked_underfed", event)
            if event["policy_spec"] == "v2.4" and _event_has(event, "already_ready_overattached_possible"):
                _add_example(examples_required, "v2.4_attached_to_already_ready_target", event)
            if event["policy_spec"] in {"v2.4[attach_target=active]", "v2.4[attach_target=first]"} and _event_has(
                event, "best_visible_target_chosen"
            ):
                _add_example(examples_required, "active_or_first_ablation_chose_best_visible_target", event)
            if _event_has(event, "unknown") or _event_has(event, "unknown_target_chosen"):
                _add_example(examples_required, "target_quality_uncertain_frame", event)

    for key in REQUIRED_EXAMPLE_KEYS:
        examples_required.setdefault(key, [])

    actual_attach_frames_by_policy: dict[str, int] = {}
    best_visible_agreement_by_policy: dict[str, dict[str, Any]] = {}
    better_visible_by_policy: Counter[str] = Counter()
    mega_line_target_by_policy: Counter[str] = Counter()
    kyogre_target_by_policy: Counter[str] = Counter()
    already_ready_overattach_by_policy: Counter[str] = Counter()
    active_target_by_policy: Counter[str] = Counter()
    bench_target_by_policy: Counter[str] = Counter()
    role_distributions: dict[str, dict[str, int]] = {}

    for policy in policies:
        attached_events = [event for event in events if event["policy_spec"] == policy and event["attached"]]
        actual_attach_frames_by_policy[policy] = len(attached_events)
        best_count = sum(1 for event in attached_events if _event_has(event, "best_visible_target_chosen"))
        best_visible_agreement_by_policy[policy] = {
            "best_visible_target_chosen": best_count,
            "actual_attach_frames": len(attached_events),
            "rate": _rate(best_count, len(attached_events)),
        }
        role_distributions[policy] = _role_distribution(events, policy)
        for event in attached_events:
            if _event_has(event, "better_visible_target_available"):
                better_visible_by_policy[policy] += 1
            if _event_has(event, "mega_line_target_chosen"):
                mega_line_target_by_policy[policy] += 1
            if _event_has(event, "kyogre_target_chosen"):
                kyogre_target_by_policy[policy] += 1
            if _event_has(event, "already_ready_overattached_possible"):
                already_ready_overattach_by_policy[policy] += 1
            if _event_has(event, "active_target_chosen"):
                active_target_by_policy[policy] += 1
            if _event_has(event, "bench_target_chosen"):
                bench_target_by_policy[policy] += 1

    missed_attach_frames = sum(1 for event in events if event["policy_spec"] == "v2.4" and event["missed_attach"])
    default_attaches = actual_attach_frames_by_policy.get("v2.4", 0)
    default_better = better_visible_by_policy.get("v2.4", 0)
    first_order = bucket_counter.get("first_target_order_likely_helpful", 0)
    conclusion = _conclusion(
        missed_attach=missed_attach_frames,
        default_attaches=default_attaches,
        default_better=default_better,
        first_order=first_order,
    )

    target_quality = {
        "schema_version": TARGET_QUALITY_SCHEMA_VERSION,
        "run_id": run_id,
        "trace_source": trace_source,
        "probe_source": probe_source,
        "policies": policies,
        "frames": len(frames),
        "games": len({frame.get("game_id") for frame in frames}),
        "probes": len(probe_payloads),
        "diffs": len(diff_payloads),
        "probe_errors": sum(1 for probe in probe_payloads if probe.get("error")),
        "pair_summaries": pair_payloads,
        "metrics": {
            "total_frames_analyzed": len(frames),
            "attach_legal_frames": attach_legal_frames,
            "target_events": len(events),
            "actual_attach_frames_by_policy": actual_attach_frames_by_policy,
            "missed_attach_frames_by_default_v24": missed_attach_frames,
            "default_v24_attach_target_role_distribution": role_distributions.get("v2.4", {}),
            "first_legal_attach_target_role_distribution": role_distributions.get("first-legal", {}),
            "attach_target_ablation_role_distributions": {
                policy: role_distributions.get(policy, {}) for policy in ATTACH_TARGET_POLICIES
            },
            "all_policy_role_distributions": role_distributions,
            "best_visible_target_agreement_rate_by_policy": best_visible_agreement_by_policy,
            "better_visible_target_available_count_by_policy": dict(better_visible_by_policy),
            "mega_line_target_count_by_policy": dict(mega_line_target_by_policy),
            "kyogre_target_count_by_policy": dict(kyogre_target_by_policy),
            "already_ready_overattach_possible_count_by_policy": dict(already_ready_overattach_by_policy),
            "near_ready_target_missed_count": bucket_counter.get("near_ready_target_missed", 0),
            "active_target_count_by_policy": dict(active_target_by_policy),
            "bench_target_count_by_policy": dict(bench_target_by_policy),
            "active_vs_bench_target_rates": {
                policy: {
                    "active_rate": _rate(active_target_by_policy.get(policy, 0), actual_attach_frames_by_policy.get(policy, 0)),
                    "bench_rate": _rate(bench_target_by_policy.get(policy, 0), actual_attach_frames_by_policy.get(policy, 0)),
                }
                for policy in policies
            },
            "timing_vs_target_vs_option_order_conclusion": conclusion,
        },
        "bucket_counts": dict(bucket_counter),
        "examples_by_bucket": dict(examples_by_bucket),
        "required_examples": dict(examples_required),
        "do_not_implement_yet": [
            "Do not change gameplay policy from this audit-only evidence pass.",
            "Do not tune scores from the audit-only target score.",
            "Do not treat visible-state evidence as hidden hand, deck, or future-game truth.",
            "Do not implement trainer semantics here.",
            "Do not alter Hammer-lanche or Riptide semantics here.",
            "Do not change deck.csv or add deck mutation.",
        ],
    }
    return target_quality, events


def _pair_rows(target_quality: dict[str, Any]) -> list[list[Any]]:
    rows = [
        [
            pair["agent_a"],
            pair["agent_b"],
            f"{pair['a_wins']}-{pair['b_wins']}-{pair['draws']}",
            f"{pair['completed']}/{pair['games']}",
            pair["crashes"],
            pair["timeouts"],
            f"{pair['avg_turns']:.2f}",
        ]
        for pair in target_quality.get("pair_summaries", [])
    ]
    return rows or [["unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"]]


def _rate_rows(target_quality: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    metrics = target_quality["metrics"]
    for policy, values in metrics["best_visible_target_agreement_rate_by_policy"].items():
        rate = values["rate"]
        rows.append(
            [
                policy,
                values["best_visible_target_chosen"],
                values["actual_attach_frames"],
                "n/a" if rate is None else f"{rate:.3f}",
                metrics["better_visible_target_available_count_by_policy"].get(policy, 0),
                metrics["mega_line_target_count_by_policy"].get(policy, 0),
                metrics["kyogre_target_count_by_policy"].get(policy, 0),
            ]
        )
    return rows or [["none", 0, 0, "n/a", 0, 0, 0]]


def _role_rows(distributions: dict[str, dict[str, int]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for policy, distribution in distributions.items():
        if not distribution:
            rows.append([policy, "none", 0])
        else:
            for role, count in Counter(distribution).most_common():
                rows.append([policy, role, count])
    return rows or [["none", "none", 0]]


def _examples_table(examples: dict[str, list[dict[str, Any]]]) -> str:
    rows: list[list[Any]] = []
    ordered_keys = [key for key in REQUIRED_EXAMPLE_KEYS if key in examples]
    ordered_keys.extend(key for key in examples if key not in ordered_keys)
    for key in ordered_keys:
        values = examples.get(key) or []
        if not values:
            rows.append([key, "none observed", "none", "none", "none", "none", "none"])
            continue
        for example in values[:3]:
            selected = example.get("selected_candidate") or {}
            rows.append(
                [
                    key,
                    example.get("frame_id"),
                    example.get("policy_spec"),
                    example.get("chosen_type"),
                    selected.get("primary_role", "none"),
                    selected.get("audit_score", "none"),
                    ",".join(example.get("buckets") or []),
                ]
            )
    return table(["example", "frame_id", "policy", "choice", "target_role", "audit_score", "buckets"], rows or [["none", "none", "none", "none", "none", "none", "none"]])


def build_target_quality_report(target_quality: dict[str, Any]) -> str:
    metrics = target_quality["metrics"]
    recommendation = metrics["timing_vs_target_vs_option_order_conclusion"]
    lines = [
        title("Deck Track V3.3 Attach Target Quality Audit"),
        f"- run_id: `{target_quality['run_id']}`\n",
        f"- frames: {target_quality['frames']}\n",
        f"- games_with_frames: {target_quality['games']}\n",
        f"- probes: {target_quality['probes']}\n",
        f"- probe_errors: {target_quality['probe_errors']}\n",
        f"- trace_source: `{target_quality.get('trace_source')}`\n",
        f"- probe_source: `{target_quality.get('probe_source')}`\n",
        "\n",
        section("Evidence Boundary"),
        "This report uses an audit-only target score, visible-state evidence, and conservative possible tags.\n\n",
        "The score is not a policy score and is not the true best move. Hidden hand, hidden deck order, future draws, and full-game counterfactuals are not observed.\n\n",
        "Downstream wins and losses after a divergent action are not proven causal. This audit separates attach timing from attach target quality.\n\n",
        section("Matchups"),
        table(["agent_a", "agent_b", "record", "completed", "crashes", "timeouts", "avg_turns"], _pair_rows(target_quality)),
        "\n",
        section("Core Metrics"),
        table(
            ["metric", "value"],
            [
                ["total_frames_analyzed", metrics["total_frames_analyzed"]],
                ["attach_legal_frames", metrics["attach_legal_frames"]],
                ["target_events", metrics["target_events"]],
                ["missed_attach_frames_by_default_v24", metrics["missed_attach_frames_by_default_v24"]],
                ["near_ready_target_missed_count", metrics["near_ready_target_missed_count"]],
                ["timing_vs_target_vs_option_order_conclusion", recommendation],
            ],
        ),
        "\n",
        section("Audit-Best Agreement"),
        table(
            ["policy", "best_visible", "attach_frames", "rate", "better_available", "mega_line_targets", "kyogre_targets"],
            _rate_rows(target_quality),
        ),
        "\n",
        section("Role Distributions"),
        "Default V2.4:\n\n",
        table(["policy", "role", "count"], _role_rows({"v2.4": metrics["default_v24_attach_target_role_distribution"]})),
        "\n",
        "First-legal:\n\n",
        table(["policy", "role", "count"], _role_rows({"first-legal": metrics["first_legal_attach_target_role_distribution"]})),
        "\n",
        "Attach-target ablations:\n\n",
        table(["policy", "role", "count"], _role_rows(metrics["attach_target_ablation_role_distributions"])),
        "\n",
        section("Active vs Bench Rates"),
        table(
            ["policy", "active_rate", "bench_rate"],
            [
                [
                    policy,
                    "n/a" if rates["active_rate"] is None else f"{rates['active_rate']:.3f}",
                    "n/a" if rates["bench_rate"] is None else f"{rates['bench_rate']:.3f}",
                ]
                for policy, rates in metrics["active_vs_bench_target_rates"].items()
            ],
        ),
        "\n",
        section("Top Buckets"),
        table(["bucket", "count"], _counter_rows(target_quality["bucket_counts"])),
        "\n",
        section("Required Example Frames"),
        _examples_table(target_quality.get("required_examples", {})),
        "\n",
        section("Interpretation"),
        bullets(
            [
                f"Main conclusion, {recommendation}.",
                "Mega Abomasnow or Snover underfeed is marked possible only when the line is visible and below useful readiness while an attachment is skipped or goes elsewhere.",
                "Kyogre overfeed is marked possible only when Kyogre receives the visible attachment while the Mega line looks underfed.",
                "Already-ready overattach is marked possible only when the chosen target appears ready and another visible target remains underdeveloped.",
                "Option-order value is possible when first-target or first-legal agreement exceeds what the audit-only target score explains.",
            ]
        ),
        "\n",
        section("Answers"),
        bullets(
            [
                "When V2.4 attaches, see the v2.4 row in Audit-Best Agreement for audit-best visible target rate.",
                "When first-legal attaches, see the first-legal row in Audit-Best Agreement for audit-best visible target rate.",
                "Active, bench, and first-target alignment are compared in role distributions, active-vs-bench rates, and top buckets.",
                "The main failure is summarized by the timing vs target vs option-order conclusion, with missed attach frames separated from better target frames.",
                "Mega line underfeed, Kyogre overfeed, and already-ready overattach remain visible-state possible findings, not proven causal.",
            ]
        ),
        "\n",
        section("Do Not Implement Yet"),
        bullets(target_quality.get("do_not_implement_yet", [])),
    ]
    return "".join(lines)


def build_target_quality_from_existing(
    *,
    run_id: str,
    trace_run: Path | str,
    probe_run: Path | str | None,
    policies: list[str],
    limit_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    frames_path = resolve_frames_path(trace_run)
    frames = read_frames(frames_path, limit=limit_frames)
    if probe_run is None:
        probes, diffs = build_probe_outputs(frames, policies)
        probe_source = None
    else:
        probes, diffs = read_probe_run(probe_run)
        probe_source = str(probe_run)
    target_quality, events = analyze_target_quality(
        run_id=run_id,
        frames=frames,
        probes=probes,
        diffs=diffs,
        policies=policies,
        trace_source=str(frames_path),
        probe_source=probe_source,
    )
    return frames, _payloads(probes), _payloads(diffs), target_quality, events
