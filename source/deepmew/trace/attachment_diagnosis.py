"""Attachment timing and target diagnosis built on deterministic trace probes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from cg.api import AreaType

from deepmew.arena import PairSummary, run_pair
from deepmew.artifacts.run_dir import safe_slug
from deepmew.deck_profiles import KYOGRE_CARD_ID, MEGA_ABOMASNOW_CARD_ID, MEGA_LINE_IDS, SNOVER_CARD_ID, WAITRESS_CARD_ID
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.state_machine import analyze_board, attachment_effect
from deepmew.trace.diff import diff_probe_sets
from deepmew.trace.probe import probe_frames, read_frames, resolve_frames_path
from deepmew.trace.record import TraceRecorder
from deepmew.trace.schema import PolicyDiff, PolicyProbe


DIAGNOSIS_SCHEMA_VERSION = "deepmew.deck_attachment_diagnosis.v1"
RUN_KIND = "deck_attachment_diagnosis"

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
SETUP_TYPES = {"PLAY", "EVOLVE", "ABILITY", "SKILL"}
ATTACH_TARGET_POLICIES = [
    "v2.4[attach_target=active]",
    "v2.4[attach_target=bench]",
    "v2.4[attach_target=first]",
]


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


def _chooses_attach(probe: dict[str, Any] | None) -> bool:
    return "ATTACH" in _chosen_types(probe)


def _first_chosen_type(probe: dict[str, Any] | None) -> str:
    options = _chosen_options(probe)
    if not options:
        return "none"
    return str(options[0].get("option_type") or "unknown")


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
    return indices


def _attach_legal(frame: dict[str, Any]) -> bool:
    board = frame.get("board_summary") or {}
    return bool(board.get("attach_legal")) or bool(_legal_attach_indices(frame))


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


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return AreaType(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _own_player(frame: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_observation(frame) or {}
    current = raw.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex") or frame.get("seat") or 0)
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
        pokemon.append({"area": "ACTIVE", "index": 0, "id": own_active.get("card_id"), "energy_count": own_active.get("energy_count") or 0})
    for index, item in enumerate(board.get("own_bench") or []):
        if isinstance(item, dict):
            pokemon.append({"area": "BENCH", "index": index, "id": item.get("card_id"), "energy_count": item.get("energy_count") or 0})
    return pokemon


def _card_id(pokemon: dict[str, Any] | None) -> int | None:
    if not pokemon:
        return None
    value = pokemon.get("id", pokemon.get("card_id"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pokemon_at(frame: dict[str, Any], area: str | None, index: int | None) -> dict[str, Any] | None:
    if area is None or index is None:
        return None
    for pokemon in _own_pokemon(frame):
        if str(pokemon.get("area")) == area and int(pokemon.get("index") or 0) == index:
            return pokemon
    return None


def _attach_target_from_option(frame: dict[str, Any], option_index: int) -> dict[str, Any] | None:
    options = _raw_options(frame)
    if not (0 <= option_index < len(options)):
        return None
    option = options[option_index]
    area = _area_name(option.get("inPlayArea"))
    try:
        index = int(option.get("inPlayIndex") or 0)
    except (TypeError, ValueError):
        index = 0
    return _pokemon_at(frame, area, index)


def _selected_attach_targets(frame: dict[str, Any], probe: dict[str, Any] | None) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for index in _chosen_indices(probe):
        option_summary = next((item for item in _legal_options(frame) if item.get("index") == index), {})
        if option_summary.get("option_type") != "ATTACH":
            continue
        target = _attach_target_from_option(frame, index)
        if target is not None:
            targets.append(target)
    return targets


def _selected_attach_indices(frame: dict[str, Any], probe: dict[str, Any] | None) -> list[int]:
    attach_indices = set(_legal_attach_indices(frame))
    return [index for index in _chosen_indices(probe) if index in attach_indices]


def _mega_line_energy(frame: dict[str, Any]) -> int:
    values = [_energy_count(pokemon) for pokemon in _own_pokemon(frame) if _card_id(pokemon) in MEGA_LINE_IDS]
    return max(values) if values else 0


def _kyogre_energy(frame: dict[str, Any]) -> int:
    values = [_energy_count(pokemon) for pokemon in _own_pokemon(frame) if _card_id(pokemon) == KYOGRE_CARD_ID]
    return max(values) if values else 0


def _mega_line_exists(frame: dict[str, Any]) -> bool:
    return any(_card_id(pokemon) in MEGA_LINE_IDS for pokemon in _own_pokemon(frame))


def _mega_line_underfed_possible(frame: dict[str, Any], probe: dict[str, Any] | None = None) -> bool:
    if not _mega_line_exists(frame):
        return False
    if _mega_line_energy(frame) >= 2:
        return False
    if probe is None:
        return True
    targets = _selected_attach_targets(frame, probe)
    if not targets:
        return True
    return all(_card_id(target) not in MEGA_LINE_IDS for target in targets)


def _kyogre_overfed_possible(frame: dict[str, Any], probe: dict[str, Any] | None) -> bool:
    if not _mega_line_underfed_possible(frame):
        return False
    for target in _selected_attach_targets(frame, probe):
        if _card_id(target) == KYOGRE_CARD_ID:
            return True
    return False


def _recent_waitress_context(frame: dict[str, Any]) -> bool:
    raw = _raw_observation(frame) or {}
    text = json.dumps(raw.get("logs") or [], sort_keys=True)
    text += " " + " ".join(str(note) for note in frame.get("notes") or [])
    board = frame.get("board_summary") or {}
    return bool(board.get("waitress_recent") or "Waitress" in text or str(WAITRESS_CARD_ID) in text)


def _attach_effects(frame: dict[str, Any]) -> dict[int, Any]:
    raw = _raw_observation(frame)
    if raw is None:
        return {}
    try:
        plan = analyze_board(raw)
    except Exception:
        return {}
    effects: dict[int, Any] = {}
    for index in _legal_attach_indices(frame):
        options = _raw_options(frame)
        if 0 <= index < len(options):
            effects[index] = attachment_effect(plan, options[index])
    return effects


def _target_quality_buckets(frame: dict[str, Any], probe: dict[str, Any] | None) -> set[str]:
    buckets: set[str] = set()
    selected_attach_indices = _selected_attach_indices(frame, probe)
    if not selected_attach_indices:
        return buckets

    effects = _attach_effects(frame)
    known_effects = {index: effect for index, effect in effects.items() if getattr(effect, "known_target", False)}
    if known_effects:
        best_after = min((effect.after_shortfall for effect in known_effects.values() if effect.after_shortfall is not None), default=None)
        for index in selected_attach_indices:
            effect = known_effects.get(index)
            if effect is None:
                continue
            if effect.before_shortfall == 0:
                buckets.add("already_ready_overattached")
            if best_after is not None and effect.after_shortfall is not None and effect.after_shortfall > best_after:
                buckets.add("wrong_attach_target_possible")

    if _mega_line_underfed_possible(frame, probe):
        buckets.add("mega_line_underfed")
    if _kyogre_overfed_possible(frame, probe):
        buckets.add("kyogre_overfed")
    if _recent_waitress_context(frame) and ({"wrong_attach_target_possible", "kyogre_overfed", "already_ready_overattached"} & buckets):
        buckets.add("waitress_attach_misuse_possible")
    return buckets


def _first_legal_order_attach(frame: dict[str, Any], first_probe: dict[str, Any] | None) -> bool:
    attach_indices = _legal_attach_indices(frame)
    chosen = _chosen_indices(first_probe)
    if not attach_indices or not chosen:
        return False
    return min(attach_indices) in chosen and min(chosen) == min(attach_indices)


def _same_choice(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return False
    return _chosen_indices(left) == _chosen_indices(right) and not left.get("error") and not right.get("error")


def _same_attachment_relevant_choice(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if _same_choice(left, right):
        return True
    return _chooses_attach(left) == _chooses_attach(right)


def classify_attachment_symptoms(
    frame: dict[str, Any],
    base_probe: dict[str, Any] | None,
    compare_probe: dict[str, Any] | None,
    all_probes: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    all_probes = all_probes or {}
    buckets: set[str] = set()
    legal_attach = _attach_legal(frame)
    base_attaches = _chooses_attach(base_probe)
    compare_attaches = _chooses_attach(compare_probe)
    base_type = _first_chosen_type(base_probe)
    compare_policy = str((compare_probe or {}).get("policy_spec") or "")

    if legal_attach and not base_attaches and compare_attaches:
        buckets.add("missed_attach_window")
        if base_type == "ATTACK":
            buckets.add("delayed_attach_for_attack")
        elif base_type == "PLAY":
            buckets.add("delayed_attach_for_play")
        elif base_type == "EVOLVE":
            buckets.add("delayed_attach_for_evolve")
        if base_type in SETUP_TYPES:
            buckets.add("delayed_attach_for_setup")

    if base_attaches:
        buckets.update(_target_quality_buckets(frame, base_probe))
    elif legal_attach and _mega_line_underfed_possible(frame, base_probe):
        buckets.add("mega_line_underfed")

    first_probe = all_probes.get("first-legal")
    if legal_attach and first_probe is not None and _chooses_attach(first_probe) and not _same_choice(base_probe, first_probe):
        if _first_legal_order_attach(frame, first_probe):
            buckets.add("first_legal_order_effect_possible")
        for policy, bucket in (
            ("v2.4[attach_target=active]", "active_attach_bias_helped"),
            ("v2.4[attach_target=bench]", "bench_attach_bias_helped"),
            ("v2.4[attach_target=first]", "first_attach_target_helped"),
        ):
            probe = all_probes.get(policy)
            if probe is not None and _same_choice(probe, first_probe):
                buckets.add(bucket)

    if compare_policy == "v2.4[semantics=off]" and legal_attach and _same_attachment_relevant_choice(base_probe, compare_probe):
        buckets.add("semantics_control_no_effect")

    if frame.get("select_context") == "IS_FIRST":
        buckets.add("first_second_choice_interaction")

    buckets.discard("unknown")
    if not buckets:
        buckets.add("unknown")
    return sorted(buckets)


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
        prefix = f"deckattach-{index:02d}-{safe_slug(agent_a)}-vs-{safe_slug(agent_b)}"
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


def _ablation_result_summary(pair_payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for pair in pair_payloads:
        agent_b = str(pair.get("agent_b"))
        if "attach_target=" not in agent_b:
            continue
        results[agent_b] = {
            "record_vs_default": f"{pair.get('a_wins')}-{pair.get('b_wins')}-{pair.get('draws')}",
            "ablation_wins": pair.get("b_wins", 0),
            "default_wins": pair.get("a_wins", 0),
            "looked_better_in_sample": int(pair.get("b_wins", 0)) > int(pair.get("a_wins", 0)),
        }
    return results


def _example_row(
    frame: dict[str, Any],
    probes_by_policy: dict[str, dict[str, Any]],
    buckets: Iterable[str],
) -> dict[str, Any]:
    return {
        "frame_id": frame.get("frame_id"),
        "game_id": frame.get("game_id"),
        "turn": frame.get("turn"),
        "seat": frame.get("seat"),
        "select_context": frame.get("select_context"),
        "v2.4_choice": (probes_by_policy.get("v2.4") or {}).get("chosen_option_summaries", []),
        "first_legal_choice": (probes_by_policy.get("first-legal") or {}).get("chosen_option_summaries", []),
        "active_ablation_choice": (probes_by_policy.get("v2.4[attach_target=active]") or {}).get("chosen_option_summaries", []),
        "bench_ablation_choice": (probes_by_policy.get("v2.4[attach_target=bench]") or {}).get("chosen_option_summaries", []),
        "first_ablation_choice": (probes_by_policy.get("v2.4[attach_target=first]") or {}).get("chosen_option_summaries", []),
        "buckets": sorted(buckets),
    }


def _top_examples(examples_by_bucket: dict[str, list[dict[str, Any]]], *, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    return {bucket: examples[:limit] for bucket, examples in sorted(examples_by_bucket.items())}


def _is_recorded_loss_for_policy(frame: dict[str, Any], policy_label: str) -> bool:
    if frame.get("agent_label") != policy_label:
        return False
    result = frame.get("final_result")
    seat = frame.get("seat")
    if result not in (0, 1) or seat not in (0, 1):
        return False
    return int(result) != int(seat)


def _attach_context_kind(frame: dict[str, Any]) -> str:
    if frame.get("select_context") == "MAIN":
        return "manual_main"
    if _recent_waitress_context(frame):
        return "waitress_possible_effect"
    return str(frame.get("select_context") or "unknown")


def analyze_diagnosis(
    *,
    run_id: str,
    frames: list[dict[str, Any]],
    probes: Iterable[Any],
    diffs: Iterable[Any],
    policies: list[str],
    pair_summaries: Iterable[PairSummary] | None = None,
    trace_source: str | None = None,
    probe_source: str | None = None,
) -> dict[str, Any]:
    probe_payloads = _payloads(probes)
    diff_payloads = _payloads(diffs)
    pair_payloads = [_summarize_pair(pair) for pair in pair_summaries or []]
    probes_by_frame: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for probe in probe_payloads:
        frame_id, policy = _probe_key(probe)
        probes_by_frame[frame_id][policy] = probe

    attach_legal_frames = 0
    first_comparable_attach_frames = 0
    first_different_attach_frames = 0
    first_attaches_v24_not = 0
    v24_instead_counter: Counter[str] = Counter()
    attach_context_counter: Counter[str] = Counter()
    attach_target_agreement: Counter[str] = Counter()
    symptom_counter: Counter[str] = Counter()
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    symptom_by_seat: Counter[str] = Counter()
    first_second_counter: Counter[str] = Counter()
    v24_attach_target_counter: Counter[str] = Counter()
    v24_already_ready_attachments = 0
    v24_kyogre_overfed_frames = 0
    v24_mega_underfed_loss_frames = 0
    v24_mega_underfed_loss_games: set[str] = set()
    semantics_control_comparable = 0
    semantics_control_no_effect = 0

    for frame in frames:
        if not _attach_legal(frame):
            continue
        attach_legal_frames += 1
        attach_context_counter[_attach_context_kind(frame)] += 1
        frame_id = str(frame.get("frame_id") or "")
        policy_probes = probes_by_frame.get(frame_id, {})
        base = policy_probes.get("v2.4")
        first = policy_probes.get("first-legal")
        if base is None or base.get("error"):
            continue

        if first is not None and not first.get("error"):
            first_comparable_attach_frames += 1
            if not _same_choice(base, first):
                first_different_attach_frames += 1
            if _chooses_attach(first) and not _chooses_attach(base):
                first_attaches_v24_not += 1
                v24_instead_counter[_first_chosen_type(base)] += 1
            for policy in ATTACH_TARGET_POLICIES:
                probe = policy_probes.get(policy)
                if probe is not None and _chooses_attach(first) and _same_choice(probe, first):
                    attach_target_agreement[policy] += 1

        for target in _selected_attach_targets(frame, base):
            card_id = _card_id(target)
            if card_id == KYOGRE_CARD_ID:
                v24_attach_target_counter["Kyogre"] += 1
            elif card_id in MEGA_LINE_IDS:
                v24_attach_target_counter["Mega line"] += 1
            elif card_id is None:
                v24_attach_target_counter["unknown"] += 1
            else:
                v24_attach_target_counter[str(card_id)] += 1

        frame_buckets: set[str] = set()
        for compare_policy in ["first-legal", *ATTACH_TARGET_POLICIES, "v2.4[semantics=off]"]:
            compare = policy_probes.get(compare_policy)
            if compare is None or compare.get("error"):
                continue
            buckets = classify_attachment_symptoms(frame, base, compare, policy_probes)
            frame_buckets.update(bucket for bucket in buckets if bucket != "unknown")

        if not frame_buckets:
            frame_buckets.add("unknown")
        symptom_counter.update(frame_buckets)
        seat = frame.get("seat")
        if {"missed_attach_window", "delayed_attach_for_attack", "delayed_attach_for_play", "delayed_attach_for_evolve", "delayed_attach_for_setup"} & frame_buckets:
            symptom_by_seat[f"seat_{seat}"] += 1
        if frame.get("select_context") == "IS_FIRST" or "first_second_choice_interaction" in frame_buckets:
            first_second_counter[f"seat_{seat}"] += 1
        if "already_ready_overattached" in frame_buckets:
            v24_already_ready_attachments += 1
        if "kyogre_overfed" in frame_buckets:
            v24_kyogre_overfed_frames += 1
        if _is_recorded_loss_for_policy(frame, "v2.4") and "mega_line_underfed" in frame_buckets:
            v24_mega_underfed_loss_frames += 1
            v24_mega_underfed_loss_games.add(str(frame.get("game_id")))
        if "semantics_control_no_effect" in frame_buckets:
            semantics_control_no_effect += 1

        sem = policy_probes.get("v2.4[semantics=off]")
        if sem is not None and not sem.get("error"):
            semantics_control_comparable += 1

        example = _example_row(frame, policy_probes, frame_buckets)
        for bucket in frame_buckets:
            examples_by_bucket[bucket].append(example)

    probe_errors = sum(1 for probe in probe_payloads if probe.get("error"))
    ablation_results = _ablation_result_summary(pair_payloads)
    recommendations = recommend_next_targets(
        symptom_counter=symptom_counter,
        first_attaches_v24_not=first_attaches_v24_not,
        ablation_agreement=attach_target_agreement,
        ablation_results=ablation_results,
    )
    likely_cause = infer_likely_cause(
        first_attaches_v24_not=first_attaches_v24_not,
        first_different_attach_frames=first_different_attach_frames,
        symptom_counter=symptom_counter,
        attach_target_agreement=attach_target_agreement,
    )

    return {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "run_id": run_id,
        "trace_source": trace_source,
        "probe_source": probe_source,
        "policies": policies,
        "frames": len(frames),
        "games": len({frame.get("game_id") for frame in frames}),
        "probes": len(probe_payloads),
        "diffs": len(diff_payloads),
        "probe_errors": probe_errors,
        "pair_summaries": pair_payloads,
        "attach_legal_summary": {
            "attach_legal_frames": attach_legal_frames,
            "first_comparable_attach_frames": first_comparable_attach_frames,
            "v24_differs_from_first_legal_on_attach_legal_frames": first_different_attach_frames,
            "first_legal_attaches_while_v24_does_not": first_attaches_v24_not,
            "v24_choice_instead_of_attach_when_first_legal_attaches": dict(v24_instead_counter),
            "attach_contexts": dict(attach_context_counter),
        },
        "target_ablation_summary": {
            "agreement_with_first_legal_when_first_attaches": dict(attach_target_agreement),
            "sampled_pair_results": ablation_results,
            "v24_attach_targets": dict(v24_attach_target_counter),
        },
        "downstream_outcome_signals": {
            "v24_mega_underfed_loss_frames": v24_mega_underfed_loss_frames,
            "v24_mega_underfed_loss_games": len(v24_mega_underfed_loss_games),
            "v24_kyogre_overfed_frames": v24_kyogre_overfed_frames,
            "v24_already_ready_attachment_frames": v24_already_ready_attachments,
        },
        "first_second_summary": {
            "attach_timing_symptoms_by_seat": dict(symptom_by_seat),
            "first_second_choice_interaction_by_seat": dict(first_second_counter),
        },
        "semantics_control": {
            "comparable_attach_frames": semantics_control_comparable,
            "same_attachment_relevant_choice_frames": semantics_control_no_effect,
        },
        "symptom_counts": dict(symptom_counter),
        "examples_by_bucket": _top_examples(examples_by_bucket),
        "likely_cause": likely_cause,
        "recommendations": recommendations,
        "do_not_implement_yet": [
            "Do not change gameplay policy from this diagnosis pass.",
            "Do not tune scores based on this run alone.",
            "Do not add trainer semantics until a separate narrow task is approved.",
            "Do not alter Hammer-lanche or Riptide semantics here.",
            "Do not change deck.csv or add deck mutation.",
            "Do not build a broad card parser.",
        ],
    }


def infer_likely_cause(
    *,
    first_attaches_v24_not: int,
    first_different_attach_frames: int,
    symptom_counter: Counter[str],
    attach_target_agreement: Counter[str],
) -> str:
    if first_attaches_v24_not >= max(10, first_different_attach_frames // 3):
        return "observed symptom points most toward attach timing, not proven causal"
    if symptom_counter.get("wrong_attach_target_possible", 0) or attach_target_agreement:
        return "observed symptom points toward attach target quality or option order, not proven causal"
    if symptom_counter.get("first_legal_order_effect_possible", 0):
        return "observed symptom points toward first legal option ordering, not proven causal"
    return "unknown or insufficient evidence"


def recommend_next_targets(
    *,
    symptom_counter: Counter[str],
    first_attaches_v24_not: int,
    ablation_agreement: Counter[str],
    ablation_results: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    candidates: list[tuple[int, dict[str, str]]] = []
    candidates.append(
        (
            first_attaches_v24_not + symptom_counter.get("missed_attach_window", 0),
            {
                "target": "Attachment timing audit",
                "classification": "basic Pokemon sequencing diagnosis",
                "why": "First-legal or target ablations attach on frames where default V2.4 does not.",
            },
        )
    )
    candidates.append(
        (
            symptom_counter.get("wrong_attach_target_possible", 0)
            + symptom_counter.get("already_ready_overattached", 0)
            + sum(ablation_agreement.values()),
            {
                "target": "Attach target quality audit",
                "classification": "basic Pokemon target diagnosis",
                "why": "Target ablations or readiness checks suggest default V2.4 may attach to weaker targets.",
            },
        )
    )
    candidates.append(
        (
            symptom_counter.get("mega_line_underfed", 0),
            {
                "target": "Mega line feeding audit",
                "classification": "current-deck setup diagnosis",
                "why": "Snover or Mega Abomasnow line appears underfed on attachment-relevant frames.",
            },
        )
    )
    candidates.append(
        (
            symptom_counter.get("kyogre_overfed", 0),
            {
                "target": "Kyogre attachment audit",
                "classification": "current-deck target diagnosis",
                "why": "Kyogre may be receiving energy while the Mega line is still behind.",
            },
        )
    )
    candidates.append(
        (
            max((int(item.get("ablation_wins", 0)) - int(item.get("default_wins", 0)) for item in ablation_results.values()), default=0),
            {
                "target": "Promote no policy yet, rerun ablations with larger sample",
                "classification": "measurement",
                "why": "A target ablation looked better in the sampled run, but arena outcomes are not causal proof.",
            },
        )
    )
    ranked = [payload for score, payload in sorted(candidates, key=lambda item: (-item[0], item[1]["target"])) if score > 0]
    if not ranked:
        return [
            {
                "target": "Collect a larger attachment trace sample",
                "classification": "needs more data",
                "why": "This sample did not expose enough attachment-specific evidence for a safe next task.",
            }
        ]
    return ranked[:2]


def _counter_rows(counter_dict: dict[str, Any], *, limit: int = 12) -> list[list[Any]]:
    counter = Counter({str(key): int(value) for key, value in counter_dict.items()})
    rows = [[key, value] for key, value in counter.most_common(limit)]
    return rows or [["none", 0]]


def _pair_rows(diagnosis: dict[str, Any]) -> list[list[Any]]:
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
        for pair in diagnosis.get("pair_summaries", [])
    ]
    return rows or [["unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"]]


def _examples_table(diagnosis: dict[str, Any], *, limit: int = 8) -> str:
    rows: list[list[Any]] = []
    for bucket, examples in diagnosis.get("examples_by_bucket", {}).items():
        for example in examples[:2]:
            rows.append(
                [
                    bucket,
                    example.get("frame_id"),
                    example.get("turn"),
                    example.get("seat"),
                    example.get("select_context"),
                    example.get("v2.4_choice"),
                    example.get("first_legal_choice"),
                ]
            )
            if len(rows) >= limit:
                return table(["bucket", "frame_id", "turn", "seat", "context", "v2.4", "first-legal"], rows)
    if not rows:
        return "No example frames.\n"
    return table(["bucket", "frame_id", "turn", "seat", "context", "v2.4", "first-legal"], rows)


def build_diagnosis_report(diagnosis: dict[str, Any]) -> str:
    attach = diagnosis["attach_legal_summary"]
    target = diagnosis["target_ablation_summary"]
    downstream = diagnosis["downstream_outcome_signals"]
    first_second = diagnosis["first_second_summary"]
    semantics = diagnosis["semantics_control"]
    recommendation_lines = [
        f"{item['target']}, {item['classification']}, {item['why']}"
        for item in diagnosis.get("recommendations", [])
    ]

    ablation_rows = []
    for policy, item in target["sampled_pair_results"].items():
        ablation_rows.append(
            [
                policy,
                item["record_vs_default"],
                item["ablation_wins"],
                item["default_wins"],
                item["looked_better_in_sample"],
            ]
        )
    if not ablation_rows:
        ablation_rows = [["none", "not sampled", 0, 0, False]]

    lines = [
        title("Deck Track V3.2 Attachment Diagnosis"),
        f"- run_id: `{diagnosis['run_id']}`\n",
        f"- frames: {diagnosis['frames']}\n",
        f"- games_with_frames: {diagnosis['games']}\n",
        f"- probes: {diagnosis['probes']}\n",
        f"- diffs: {diagnosis['diffs']}\n",
        f"- probe_errors: {diagnosis['probe_errors']}\n",
        f"- trace_source: `{diagnosis.get('trace_source')}`\n",
        f"- probe_source: `{diagnosis.get('probe_source')}`\n",
        "\n",
        section("Evidence Boundary"),
        "Same-frame evidence means the recorded observation was replayed through each policy and compared at that exact decision point.\n\n",
        "Downstream game result is separated below because later decisions may already have diverged. Treat every bucket as an observed symptom and at most a likely contributor, not proven causal.\n\n",
        "Manual attachment is inferred when ATTACH is legal in `MAIN`. Non-MAIN or log-linked attachment is reported separately when visible; exact effect-driven attachment attribution is weak unless logs expose the source.\n\n",
        section("Matchups"),
        table(["agent_a", "agent_b", "record", "completed", "crashes", "timeouts", "avg_turns"], _pair_rows(diagnosis)),
        "\n",
        section("Attach-Legal Same-Frame Evidence"),
        table(
            ["metric", "count"],
            [
                ["attach_legal_frames", attach["attach_legal_frames"]],
                ["first_comparable_attach_frames", attach["first_comparable_attach_frames"]],
                ["v24_differs_from_first_legal_on_attach_legal_frames", attach["v24_differs_from_first_legal_on_attach_legal_frames"]],
                ["first_legal_attaches_while_v24_does_not", attach["first_legal_attaches_while_v24_does_not"]],
            ],
        ),
        "\n",
        "What V2.4 did instead when first-legal attached:\n\n",
        table(["v2.4_choice_type", "count"], _counter_rows(attach["v24_choice_instead_of_attach_when_first_legal_attaches"])),
        "\n",
        "Attach contexts:\n\n",
        table(["context", "count"], _counter_rows(attach["attach_contexts"])),
        "\n",
        section("Attach Target Ablations"),
        "Agreement with first-legal when first-legal attached:\n\n",
        table(["policy", "count"], _counter_rows(target["agreement_with_first_legal_when_first_attaches"])),
        "\n",
        "Sampled pair results, default V2.4 is agent_a and the ablation is agent_b:\n\n",
        table(["policy", "record_vs_default", "ablation_wins", "default_wins", "looked_better_in_sample"], ablation_rows),
        "\n",
        "V2.4 attachment targets when identifiable:\n\n",
        table(["target", "count"], _counter_rows(target["v24_attach_targets"])),
        "\n",
        section("Downstream Outcome Signals"),
        table(
            ["signal", "count"],
            [
                ["v24_mega_underfed_loss_frames", downstream["v24_mega_underfed_loss_frames"]],
                ["v24_mega_underfed_loss_games", downstream["v24_mega_underfed_loss_games"]],
                ["v24_kyogre_overfed_frames", downstream["v24_kyogre_overfed_frames"]],
                ["v24_already_ready_attachment_frames", downstream["v24_already_ready_attachment_frames"]],
            ],
        ),
        "\n",
        section("Seat And First-Second Signals"),
        "Attach timing symptoms by seat:\n\n",
        table(["seat", "count"], _counter_rows(first_second["attach_timing_symptoms_by_seat"])),
        "\n",
        "First/second interaction frames by seat:\n\n",
        table(["seat", "count"], _counter_rows(first_second["first_second_choice_interaction_by_seat"])),
        "\n",
        section("Semantics Control"),
        table(
            ["metric", "count"],
            [
                ["comparable_attach_frames", semantics["comparable_attach_frames"]],
                ["same_attachment_relevant_choice_frames", semantics["same_attachment_relevant_choice_frames"]],
            ],
        ),
        "\n",
        section("Top Symptom Buckets"),
        table(["bucket", "count"], _counter_rows(diagnosis["symptom_counts"])),
        "\n",
        section("Example Frames"),
        _examples_table(diagnosis),
        "\n",
        section("Interpretation"),
        bullets(
            [
                f"Observed symptom, likely contributor, not proven causal: {diagnosis['likely_cause']}.",
                "Target-quality tags are possible tags when exact target value is weakly inferred from visible energy and attack-readiness shortfall.",
                "First-legal option-order tags mean ATTACH appeared early enough for first-legal to take it; they do not prove ordering is the reason for the win.",
                "Downstream loss frames are reported separately from same-frame evidence because later decisions are polluted after divergence.",
            ]
        ),
        "\n",
        section("Limitations"),
        bullets(
            [
                "Manual attachment is more reliable than effect-driven attachment because MAIN context exposes it directly.",
                "Waitress-linked attachment is tagged as possible unless logs or context expose the source.",
                "Attach target quality is inferred from visible board state and simple readiness, not from hidden cards or counterfactual rollout.",
            ]
        ),
        "\n",
        section("Recommended Next Tasks"),
        bullets(recommendation_lines),
        "\n",
        section("Do Not Implement Yet"),
        bullets(diagnosis.get("do_not_implement_yet", [])),
    ]
    return "".join(lines)


def build_diagnosis_from_existing(
    *,
    run_id: str,
    trace_run: Path | str,
    probe_run: Path | str | None,
    policies: list[str],
    limit_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frames_path = resolve_frames_path(trace_run)
    frames = read_frames(frames_path, limit=limit_frames)
    if probe_run is None:
        probes, diffs = build_probe_outputs(frames, policies)
        probe_source = None
    else:
        probes, diffs = read_probe_run(probe_run)
        probe_source = str(probe_run)
    diagnosis = analyze_diagnosis(
        run_id=run_id,
        frames=frames,
        probes=probes,
        diffs=diffs,
        policies=policies,
        trace_source=str(frames_path),
        probe_source=probe_source,
    )
    return frames, _payloads(probes), _payloads(diffs), diagnosis
