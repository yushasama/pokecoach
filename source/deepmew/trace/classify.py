"""Conservative symptom classifiers for trace diffs."""

from __future__ import annotations

from typing import Any


SYMPTOM_BUCKETS = {
    "early_hammer_lanche",
    "riptide_timing",
    "self_mill_pressure",
    "missing_waitress_setup",
    "missing_mega_signal_setup",
    "bad_attach_target",
    "bad_evolution_timing",
    "first_second_choice",
    "semantics_no_effect",
    "unknown",
}


def _turn(frame: dict[str, Any]) -> int | None:
    value = frame.get("turn")
    if value is None:
        value = (frame.get("board_summary") or {}).get("turn")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_frame_diff(frame: dict[str, Any], diff: dict[str, Any]) -> list[str]:
    tags = set(diff.get("diff_tags") or [])
    board = frame.get("board_summary") or {}
    buckets: list[str] = []
    turn = _turn(frame)

    if "hammer_lanche_chosen" in tags:
        if turn is not None and turn <= 3:
            buckets.append("early_hammer_lanche")
        if (board.get("own_deck_count") or 999) <= 10:
            buckets.append("self_mill_pressure")
    if "riptide_chosen" in tags:
        buckets.append("riptide_timing")
    if board.get("waitress_legal") and ("attack_vs_setup" in tags or "attack_vs_attach" in tags):
        buckets.append("missing_waitress_setup")
    if board.get("mega_signal_legal") and "attack_vs_setup" in tags:
        buckets.append("missing_mega_signal_setup")
    if "attack_vs_attach" in tags or "setup_vs_attach" in tags:
        buckets.append("bad_attach_target")
    if "attack_vs_evolve" in tags:
        buckets.append("bad_evolution_timing")
    if frame.get("select_context") == "IS_FIRST":
        buckets.append("first_second_choice")
    if "same_choice" in tags and "semantics" in f"{diff.get('base_policy')} {diff.get('compare_policy')}":
        buckets.append("semantics_no_effect")

    if not buckets:
        buckets.append("unknown")
    return [bucket for bucket in buckets if bucket in SYMPTOM_BUCKETS]
