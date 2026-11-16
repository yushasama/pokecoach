"""Policy probe diffing for same-frame comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from deepmew.deck_profiles import HAMMER_LANCHE_ATTACK_ID, RIPTIDE_ATTACK_ID
from deepmew.trace.schema import POLICY_DIFF_SCHEMA_VERSION, PolicyDiff


SETUP_TYPES = {"PLAY", "EVOLVE", "ABILITY", "SKILL"}


def _choice_types(probe: dict[str, Any]) -> set[str]:
    return {str(item.get("option_type")) for item in probe.get("chosen_option_summaries") or []}


def _attack_ids(probe: dict[str, Any]) -> set[int]:
    attack_ids = set()
    for item in probe.get("chosen_option_summaries") or []:
        value = item.get("attack_id")
        if value is not None:
            try:
                attack_ids.add(int(value))
            except (TypeError, ValueError):
                pass
    return attack_ids


def _as_payload(probe: Any) -> dict[str, Any]:
    if hasattr(probe, "to_jsonable"):
        return probe.to_jsonable()
    return dict(probe)


def diff_probes(base_probe: Any, compare_probe: Any) -> PolicyDiff:
    base = _as_payload(base_probe)
    compare = _as_payload(compare_probe)
    base_choice = list(base.get("chosen_indices") or [])
    compare_choice = list(compare.get("chosen_indices") or [])
    same_choice = base_choice == compare_choice and not base.get("error") and not compare.get("error")
    tags = ["same_choice" if same_choice else "different_choice"]

    base_policy = str(base.get("policy_spec") or "")
    compare_policy = str(compare.get("policy_spec") or "")
    if not same_choice and "semantics" in f"{base_policy} {compare_policy}":
        tags.append("semantics_changed_choice")
    if not same_choice and ("first-legal" in base_policy or "first-legal" in compare_policy):
        tags.append("first_legal_order_effect_possible")

    attack_ids = _attack_ids(base) | _attack_ids(compare)
    if HAMMER_LANCHE_ATTACK_ID in attack_ids:
        tags.append("hammer_lanche_chosen")
    if RIPTIDE_ATTACK_ID in attack_ids:
        tags.append("riptide_chosen")

    base_types = _choice_types(base)
    compare_types = _choice_types(compare)
    all_types = base_types | compare_types
    if not same_choice:
        if "ATTACK" in all_types and all_types & SETUP_TYPES:
            tags.append("attack_vs_setup")
        if "ATTACK" in all_types and "ATTACH" in all_types:
            tags.append("attack_vs_attach")
        if "ATTACK" in all_types and "EVOLVE" in all_types:
            tags.append("attack_vs_evolve")
        if "ATTACH" in all_types and all_types & SETUP_TYPES:
            tags.append("setup_vs_attach")

    if len(tags) == 1 and not same_choice:
        tags.append("unknown")

    notes = []
    if base.get("error"):
        notes.append(f"base error: {base['error']}")
    if compare.get("error"):
        notes.append(f"compare error: {compare['error']}")

    return PolicyDiff(
        schema_version=POLICY_DIFF_SCHEMA_VERSION,
        frame_id=str(base.get("frame_id") or compare.get("frame_id") or ""),
        base_policy=base_policy,
        compare_policy=compare_policy,
        same_choice=same_choice,
        base_choice=base_choice,
        compare_choice=compare_choice,
        diff_tags=tags,
        notes=notes,
    )


def diff_probe_sets(probes: Iterable[Any], *, base_policy: str | None = None) -> list[PolicyDiff]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        payload = _as_payload(probe)
        by_frame[str(payload.get("frame_id") or "")].append(payload)

    diffs: list[PolicyDiff] = []
    for frame_probes in by_frame.values():
        if len(frame_probes) < 2:
            continue
        if base_policy is None:
            base = frame_probes[0]
        else:
            base = next((probe for probe in frame_probes if probe.get("policy_spec") == base_policy), frame_probes[0])
        for compare in frame_probes:
            if compare is base:
                continue
            diffs.append(diff_probes(base, compare))
    return diffs
