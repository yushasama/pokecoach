"""Deck semantics diagnosis built on deterministic trace probes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from deepmew.arena import PairSummary, run_pair
from deepmew.artifacts.run_dir import safe_slug
from deepmew.card_semantics import HAMMER_LANCHE_ATTACK_ID, RIPTIDE_ATTACK_ID
from deepmew.reports.markdown import bullets, section, table, title
from deepmew.trace.classify import classify_frame_diff
from deepmew.trace.diff import diff_probe_sets, diff_probes
from deepmew.trace.probe import probe_frames, read_frames, resolve_frames_path
from deepmew.trace.record import MEGA_SIGNAL_CARD_ID, WAITRESS_CARD_ID, TraceRecorder
from deepmew.trace.schema import PolicyDiff, PolicyProbe


DIAGNOSIS_SCHEMA_VERSION = "deepmew.deck_semantics_diagnosis.v1"
DEFAULT_POLICIES = ["v2.4", "v2.4[semantics=off]", "first-legal"]
DEFAULT_MATCHUPS = [("v2.4", "v2.4[semantics=off]"), ("v2.4", "first-legal")]
FIRST_CHOICE_MATCHUP = ("v2.4[first=yes]", "v2.4")
SETUP_TYPES = {"PLAY", "EVOLVE", "ABILITY", "SKILL", "ATTACH"}


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


def _chosen_types(probe: dict[str, Any] | None) -> set[str]:
    return {str(option.get("option_type")) for option in _chosen_options(probe)}


def _chosen_attack_ids(probe: dict[str, Any] | None) -> set[int]:
    attack_ids: set[int] = set()
    for option in _chosen_options(probe):
        value = option.get("attack_id")
        try:
            if value is not None:
                attack_ids.add(int(value))
        except (TypeError, ValueError):
            pass
    return attack_ids


def _chosen_card_ids(probe: dict[str, Any] | None) -> set[int]:
    card_ids: set[int] = set()
    for option in _chosen_options(probe):
        value = option.get("card_id")
        try:
            if value is not None:
                card_ids.add(int(value))
        except (TypeError, ValueError):
            pass
    return card_ids


def _legal_attack(frame: dict[str, Any], attack_id: int) -> bool:
    for option in frame.get("legal_options") or []:
        try:
            if option.get("attack_id") is not None and int(option.get("attack_id")) == attack_id:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _legal_type(frame: dict[str, Any], option_type: str) -> bool:
    return any(str(option.get("option_type")) == option_type for option in frame.get("legal_options") or [])


def _legal_card(frame: dict[str, Any], card_id: int) -> bool:
    for option in frame.get("legal_options") or []:
        try:
            if option.get("card_id") is not None and int(option.get("card_id")) == card_id:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _board(frame: dict[str, Any]) -> dict[str, Any]:
    return dict(frame.get("board_summary") or {})


def _deck_count(frame: dict[str, Any]) -> int | None:
    value = _board(frame).get("own_deck_count")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_recorded_loss_for_policy(frame: dict[str, Any], policy_label: str) -> bool:
    if frame.get("agent_label") != policy_label:
        return False
    result = frame.get("final_result")
    seat = frame.get("seat")
    if result not in (0, 1) or seat not in (0, 1):
        return False
    return int(result) != int(seat)


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
        prefix = f"decksem-{index:02d}-{safe_slug(agent_a)}-vs-{safe_slug(agent_b)}"
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


def classify_semantics_symptoms(
    frame: dict[str, Any],
    diff: dict[str, Any],
    base_probe: dict[str, Any] | None,
    compare_probe: dict[str, Any] | None,
) -> list[str]:
    buckets = set(classify_frame_diff(frame, diff))
    board = _board(frame)
    base_attacks = _chosen_attack_ids(base_probe)
    compare_attacks = _chosen_attack_ids(compare_probe)
    base_types = _chosen_types(base_probe)
    compare_types = _chosen_types(compare_probe)
    same_choice = bool(diff.get("same_choice"))

    if not same_choice and HAMMER_LANCHE_ATTACK_ID in base_attacks:
        turn = frame.get("turn") or board.get("turn")
        try:
            if turn is not None and int(turn) <= 3:
                buckets.add("early_hammer_lanche")
        except (TypeError, ValueError):
            pass
        deck_count = _deck_count(frame)
        if deck_count is not None and deck_count <= 10:
            buckets.add("self_mill_pressure")
    if not same_choice and (RIPTIDE_ATTACK_ID in base_attacks or RIPTIDE_ATTACK_ID in compare_attacks):
        buckets.add("riptide_timing")
    if not same_choice and board.get("waitress_legal") and "ATTACK" in base_types and WAITRESS_CARD_ID not in _chosen_card_ids(base_probe):
        buckets.add("missing_waitress_setup")
    if not same_choice and board.get("mega_signal_legal") and "ATTACK" in base_types and MEGA_SIGNAL_CARD_ID not in _chosen_card_ids(base_probe):
        buckets.add("missing_mega_signal_setup")
    if not same_choice and "ATTACK" in base_types and ("ATTACH" in compare_types or _legal_type(frame, "ATTACH")):
        buckets.add("bad_attach_target")
    if not same_choice and "ATTACK" in base_types and ("EVOLVE" in compare_types or _legal_type(frame, "EVOLVE")):
        buckets.add("bad_evolution_timing")
    if frame.get("select_context") == "IS_FIRST":
        buckets.add("first_second_choice")
    if diff.get("same_choice") and "semantics" in f"{diff.get('base_policy')} {diff.get('compare_policy')}":
        buckets.add("semantics_no_effect")

    buckets.discard("unknown")
    if not buckets:
        buckets.add("unknown")
    return sorted(buckets)


def _diff_for_pair(
    frame_id: str,
    base_probe: dict[str, Any] | None,
    compare_probe: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if base_probe is None or compare_probe is None:
        return None
    return diff_probes(base_probe, compare_probe).to_jsonable()


def _top_examples(
    examples_by_bucket: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    return {bucket: examples[:limit] for bucket, examples in sorted(examples_by_bucket.items())}


def _example_row(
    frame: dict[str, Any],
    base_probe: dict[str, Any] | None,
    compare_probe: dict[str, Any] | None,
    diff: dict[str, Any],
) -> dict[str, Any]:
    board = _board(frame)
    return {
        "frame_id": frame.get("frame_id"),
        "turn": frame.get("turn"),
        "select_context": frame.get("select_context"),
        "own_deck_count": board.get("own_deck_count"),
        "base_policy": diff.get("base_policy"),
        "compare_policy": diff.get("compare_policy"),
        "base_choice": (base_probe or {}).get("chosen_option_summaries", []),
        "compare_choice": (compare_probe or {}).get("chosen_option_summaries", []),
        "diff_tags": diff.get("diff_tags", []),
    }


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
    frame_by_id = {str(frame.get("frame_id")): frame for frame in frames}
    probes_by_frame: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for probe in probe_payloads:
        frame_id, policy = _probe_key(probe)
        probes_by_frame[frame_id][policy] = probe

    comparable_semantic_frames = 0
    semantic_choice_diffs = 0
    semantic_same_choices = 0
    semantic_on_choice_types: Counter[str] = Counter()
    hammer_legal = 0
    hammer_chosen_on = 0
    hammer_chosen_off = 0
    hammer_off_avoided = 0
    riptide_legal = 0
    riptide_chosen_on = 0
    suspicious: Counter[str] = Counter()
    symptom_counter: Counter[str] = Counter()
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame in frames:
        frame_id = str(frame.get("frame_id") or "")
        on_probe = probes_by_frame.get(frame_id, {}).get("v2.4")
        off_probe = probes_by_frame.get(frame_id, {}).get("v2.4[semantics=off]")
        if not on_probe or not off_probe or on_probe.get("error") or off_probe.get("error"):
            continue
        comparable_semantic_frames += 1
        on_attacks = _chosen_attack_ids(on_probe)
        off_attacks = _chosen_attack_ids(off_probe)
        on_types = _chosen_types(on_probe)
        if (on_probe.get("chosen_indices") or []) == (off_probe.get("chosen_indices") or []):
            semantic_same_choices += 1
        else:
            semantic_choice_diffs += 1
            semantic_on_choice_types.update(on_types or {"none"})
        if _legal_attack(frame, HAMMER_LANCHE_ATTACK_ID) or _board(frame).get("hammer_lanche_legal"):
            hammer_legal += 1
        if HAMMER_LANCHE_ATTACK_ID in on_attacks:
            hammer_chosen_on += 1
            if HAMMER_LANCHE_ATTACK_ID not in off_attacks:
                hammer_off_avoided += 1
        if HAMMER_LANCHE_ATTACK_ID in off_attacks:
            hammer_chosen_off += 1
        if _legal_attack(frame, RIPTIDE_ATTACK_ID) or _board(frame).get("riptide_legal"):
            riptide_legal += 1
        if RIPTIDE_ATTACK_ID in on_attacks:
            riptide_chosen_on += 1
        if "ATTACK" in on_types:
            if _legal_card(frame, WAITRESS_CARD_ID) or _board(frame).get("waitress_legal"):
                suspicious["attack_chosen_while_waitress_legal"] += 1
            if _legal_card(frame, MEGA_SIGNAL_CARD_ID) or _board(frame).get("mega_signal_legal"):
                suspicious["attack_chosen_while_mega_signal_legal"] += 1
            if _legal_type(frame, "EVOLVE") or _board(frame).get("evolve_legal"):
                suspicious["attack_chosen_while_evolve_legal"] += 1
            if _legal_type(frame, "ATTACH") or _board(frame).get("attach_legal"):
                suspicious["attack_chosen_while_attach_legal"] += 1

        diff = _diff_for_pair(frame_id, on_probe, off_probe)
        if diff is None:
            continue
        buckets = classify_semantics_symptoms(frame, diff, on_probe, off_probe)
        symptom_counter.update(buckets)
        example = _example_row(frame, on_probe, off_probe, diff)
        for bucket in buckets:
            examples_by_bucket[bucket].append(example)

    first_comparable_frames = 0
    first_choice_diffs = 0
    first_same_choices = 0
    first_legal_choice_types: Counter[str] = Counter()
    first_legal_attach_when_v24_not = 0
    first_legal_attack_when_v24_not = 0
    for frame in frames:
        frame_id = str(frame.get("frame_id") or "")
        on_probe = probes_by_frame.get(frame_id, {}).get("v2.4")
        first_probe = probes_by_frame.get(frame_id, {}).get("first-legal")
        if not on_probe or not first_probe or on_probe.get("error") or first_probe.get("error"):
            continue
        first_comparable_frames += 1
        first_types = _chosen_types(first_probe)
        on_types = _chosen_types(on_probe)
        if (on_probe.get("chosen_indices") or []) == (first_probe.get("chosen_indices") or []):
            first_same_choices += 1
        else:
            first_choice_diffs += 1
            first_legal_choice_types.update(first_types or {"none"})
            if "ATTACH" in first_types and "ATTACH" not in on_types:
                first_legal_attach_when_v24_not += 1
            if "ATTACK" in first_types and "ATTACK" not in on_types:
                first_legal_attack_when_v24_not += 1
        diff = _diff_for_pair(frame_id, on_probe, first_probe)
        if diff is None:
            continue
        buckets = classify_semantics_symptoms(frame, diff, on_probe, first_probe)
        symptom_counter.update(buckets)
        example = _example_row(frame, on_probe, first_probe, diff)
        for bucket in buckets:
            examples_by_bucket[bucket].append(example)

    semantic_on_loss_frames = 0
    semantic_on_loss_low_deck_hammer = 0
    semantic_on_loss_deck_reason = 0
    for frame in frames:
        frame_id = str(frame.get("frame_id") or "")
        on_probe = probes_by_frame.get(frame_id, {}).get("v2.4")
        if not _is_recorded_loss_for_policy(frame, "v2.4"):
            continue
        semantic_on_loss_frames += 1
        deck_count = _deck_count(frame)
        if on_probe and HAMMER_LANCHE_ATTACK_ID in _chosen_attack_ids(on_probe) and deck_count is not None and deck_count <= 10:
            semantic_on_loss_low_deck_hammer += 1
        reason = str(frame.get("final_result_reason") or "").lower()
        if "deck" in reason or "draw" in reason:
            semantic_on_loss_deck_reason += 1

    probe_errors = sum(1 for probe in probe_payloads if probe.get("error"))
    recommendations = recommend_next_targets(
        symptom_counter=symptom_counter,
        semantics_diffs=semantic_choice_diffs,
        first_attach_diffs=first_legal_attach_when_v24_not,
        hammer_off_avoided=hammer_off_avoided,
        suspicious=suspicious,
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
        "semantics_on_vs_off": {
            "comparable_frames": comparable_semantic_frames,
            "different_choice": semantic_choice_diffs,
            "same_choice": semantic_same_choices,
            "semantics_on_choice_types_when_different": dict(semantic_on_choice_types),
            "hammer_lanche_legal": hammer_legal,
            "hammer_lanche_chosen_by_semantics_on": hammer_chosen_on,
            "hammer_lanche_chosen_by_semantics_off": hammer_chosen_off,
            "semantics_off_avoided_hammer_lanche_on_same_frame": hammer_off_avoided,
            "riptide_legal": riptide_legal,
            "riptide_chosen_by_semantics_on": riptide_chosen_on,
        },
        "first_legal_comparison": {
            "comparable_frames": first_comparable_frames,
            "different_choice": first_choice_diffs,
            "same_choice": first_same_choices,
            "first_legal_choice_types_when_different": dict(first_legal_choice_types),
            "first_legal_attach_when_v24_not": first_legal_attach_when_v24_not,
            "first_legal_attack_when_v24_not": first_legal_attack_when_v24_not,
        },
        "downstream_outcome_signals": {
            "semantic_on_recorded_loss_frames": semantic_on_loss_frames,
            "semantic_on_loss_low_deck_hammer_frames": semantic_on_loss_low_deck_hammer,
            "semantic_on_loss_deck_or_draw_reason_frames": semantic_on_loss_deck_reason,
        },
        "suspicious_same_frame_counts": dict(suspicious),
        "symptom_counts": dict(symptom_counter),
        "examples_by_bucket": _top_examples(examples_by_bucket),
        "recommendations": recommendations,
        "do_not_implement_yet": [
            "Do not implement a broad card parser from official card files.",
            "Do not change Hammer-lanche or Riptide semantics without stronger evidence.",
            "Do not add trainer semantics until same-frame trace counts identify a narrow card target.",
            "Do not tune scores from aggregate wins without explaining same-frame symptoms.",
            "Do not change deck.csv during this diagnosis track.",
        ],
    }


def recommend_next_targets(
    *,
    symptom_counter: Counter[str],
    semantics_diffs: int,
    first_attach_diffs: int,
    hammer_off_avoided: int,
    suspicious: Counter[str],
) -> list[dict[str, str]]:
    candidates: list[tuple[int, dict[str, str]]] = []
    candidates.append(
        (
            symptom_counter.get("missing_mega_signal_setup", 0),
            {
                "target": "Mega Signal setup semantics diagnosis",
                "classification": "current-deck card semantics",
                "why": "Mega Signal is legal in attack/setup divergences and may represent deck-specific evolution setup value.",
            },
        )
    )
    candidates.append(
        (
            symptom_counter.get("missing_waitress_setup", 0),
            {
                "target": "Waitress draw/setup semantics diagnosis",
                "classification": "current-deck card semantics",
                "why": "Waitress is legal in attack/setup divergences and may convert weak turns into future pressure.",
            },
        )
    )
    candidates.append(
        (
            first_attach_diffs + symptom_counter.get("bad_attach_target", 0),
            {
                "target": "Attach target and attachment timing diagnosis",
                "classification": "basic Pokemon sequencing",
                "why": "First-legal often differs on ATTACH choices, but this must be separated from engine ordering.",
            },
        )
    )
    candidates.append(
        (
            hammer_off_avoided + symptom_counter.get("early_hammer_lanche", 0),
            {
                "target": "Hammer-lanche timing audit",
                "classification": "current-deck attack semantics",
                "why": "Semantics can cause Hammer-lanche selection, but timing and setup context remain diagnostic only.",
            },
        )
    )
    candidates.append(
        (
            suspicious.get("attack_chosen_while_evolve_legal", 0) + symptom_counter.get("bad_evolution_timing", 0),
            {
                "target": "Evolution timing diagnosis",
                "classification": "basic Pokemon sequencing",
                "why": "EVOLVE legal while attacking can signal missed development timing, especially before Mega Abomasnow pressure.",
            },
        )
    )

    ranked = [payload for count, payload in sorted(candidates, key=lambda item: (-item[0], item[1]["target"])) if count > 0]
    if not ranked and semantics_diffs == 0:
        return [
            {
                "target": "Stop adding card semantics, use V3 arena for broader ablations",
                "classification": "measurement",
                "why": "Same-frame semantics-on and semantics-off choices did not diverge in this sample.",
            }
        ]
    if not ranked:
        return [
            {
                "target": "Collect larger deterministic trace sample",
                "classification": "needs more data",
                "why": "This run did not expose enough classified symptom frames for a safe implementation target.",
            }
        ]
    return ranked[:2]


def _counter_rows(counter_dict: dict[str, Any], *, limit: int = 10) -> list[list[Any]]:
    counter = Counter({str(key): int(value) for key, value in counter_dict.items()})
    rows = [[key, value] for key, value in counter.most_common(limit)]
    return rows or [["none", 0]]


def build_diagnosis_report(diagnosis: dict[str, Any]) -> str:
    sem = diagnosis["semantics_on_vs_off"]
    first = diagnosis["first_legal_comparison"]
    downstream = diagnosis["downstream_outcome_signals"]
    suspicious = diagnosis["suspicious_same_frame_counts"]
    pair_rows = [
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
    if not pair_rows:
        pair_rows = [["unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"]]

    recommendation_lines = [
        f"{item['target']}, {item['classification']}, {item['why']}"
        for item in diagnosis.get("recommendations", [])
    ]

    lines = [
        title("Deck Track V3.1 Semantics Diagnosis"),
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
        "Downstream outcome signals come from the original traced game after later decisions may already have diverged, so they are an observed symptom, a likely contributor at most, and not proven causal.\n\n",
        section("Matchups"),
        table(["agent_a", "agent_b", "record", "completed", "crashes", "timeouts", "avg_turns"], pair_rows),
        "\n",
        section("Semantics On vs Off"),
        table(
            ["metric", "count"],
            [
                ["comparable_frames", sem["comparable_frames"]],
                ["different_choice", sem["different_choice"]],
                ["same_choice", sem["same_choice"]],
                ["hammer_lanche_legal", sem["hammer_lanche_legal"]],
                ["hammer_lanche_chosen_by_semantics_on", sem["hammer_lanche_chosen_by_semantics_on"]],
                ["semantics_off_avoided_hammer_lanche_on_same_frame", sem["semantics_off_avoided_hammer_lanche_on_same_frame"]],
                ["riptide_legal", sem["riptide_legal"]],
                ["riptide_chosen_by_semantics_on", sem["riptide_chosen_by_semantics_on"]],
            ],
        ),
        "\n",
        "When semantics-on differed, it chose these option categories:\n\n",
        table(["option_type", "count"], _counter_rows(sem["semantics_on_choice_types_when_different"])),
        "\n",
        section("First Legal Comparison"),
        table(
            ["metric", "count"],
            [
                ["comparable_frames", first["comparable_frames"]],
                ["different_choice", first["different_choice"]],
                ["same_choice", first["same_choice"]],
                ["first_legal_attach_when_v24_not", first["first_legal_attach_when_v24_not"]],
                ["first_legal_attack_when_v24_not", first["first_legal_attack_when_v24_not"]],
            ],
        ),
        "\n",
        "First-legal chose these option categories when it differed from V2.4:\n\n",
        table(["option_type", "count"], _counter_rows(first["first_legal_choice_types_when_different"])),
        "\n",
        section("Suspicious Same-Frame Counts"),
        table(["symptom", "count"], _counter_rows(suspicious)),
        "\n",
        section("Downstream Outcome Signals"),
        table(
            ["signal", "count"],
            [
                ["semantic_on_recorded_loss_frames", downstream["semantic_on_recorded_loss_frames"]],
                ["semantic_on_loss_low_deck_hammer_frames", downstream["semantic_on_loss_low_deck_hammer_frames"]],
                ["semantic_on_loss_deck_or_draw_reason_frames", downstream["semantic_on_loss_deck_or_draw_reason_frames"]],
            ],
        ),
        "\n",
        section("Top Symptom Buckets"),
        table(["bucket", "count"], _counter_rows(diagnosis["symptom_counts"])),
        "\n",
        section("Interpretation"),
        bullets(
            [
                "Observed symptom, semantics-on and semantics-off differences identify where the current-deck attack overlay changes choices, not whether that choice wins the game.",
                "Likely contributor, first-legal differences should be read as attach, setup, or first-choice pressure only when same-frame evidence supports those labels.",
                "Not proven causal, any full-game loss after a divergent frame includes downstream policy pollution.",
            ]
        ),
        "\n",
        section("Recommended Next Targets"),
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
