"""CSV + JSONL writers for inferred deck profiles (machine-readable first)."""

from __future__ import annotations

import csv
import io as _io
from pathlib import Path

from deepmew.artifacts.writers import write_jsonl, write_text
from deepmew.deck_profile_inference.schemas import CardRef, DeckProfile

CSV_COLUMNS: tuple[str, ...] = (
    "deck_id",
    "deck_hash",
    "archetype_label",
    "split",
    "card_count",
    "profile_confidence",
    "resolved_card_share",
    "flagged_review_share",
    "reason_tags",
    "main_attackers",
    "setup_attackers",
    "finishers",
    "draw_engine",
    "search_engine",
    "energy_plan",
    "evolution_plan",
    "bench_pressure",
    "damage_counter_plan",
    "ability_damage_counter_plan",
    "switch_gust_plan",
    "lock_control_plan",
    "protection_plan",
    "healing_recovery_plan",
    "disruption_plan",
    "status_plan",
    "dependencies",
    "choke_points",
    "manual_review_reasons",
)


def _refs(refs: list[CardRef]) -> str:
    return "; ".join(r.as_text() for r in refs)


def _ref_dict(plan: dict) -> str:
    parts = []
    for key, value in plan.items():
        if isinstance(value, list):
            if value and isinstance(value[0], CardRef):
                rendered = _refs(value)
            else:
                rendered = ", ".join(str(v) for v in value)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " | ".join(parts)


def _deps(deps: dict[str, int]) -> str:
    return "; ".join(f"{k}:{v}" for k, v in deps.items())


def profile_to_csv_row(profile: DeckProfile) -> dict[str, str]:
    return {
        "deck_id": profile.deck_id,
        "deck_hash": profile.deck_hash,
        "archetype_label": profile.archetype_label,
        "split": profile.split,
        "card_count": str(profile.card_count),
        "profile_confidence": profile.profile_confidence,
        "resolved_card_share": f"{profile.resolved_card_share:.4f}",
        "flagged_review_share": f"{profile.flagged_review_share:.4f}",
        "reason_tags": ";".join(profile.reason_tags),
        "main_attackers": _refs(profile.main_attackers),
        "setup_attackers": _refs(profile.setup_attackers),
        "finishers": _refs(profile.finishers),
        "draw_engine": _refs(profile.draw_engine),
        "search_engine": _refs(profile.search_engine),
        "energy_plan": _ref_dict(profile.energy_plan),
        "evolution_plan": _ref_dict(profile.evolution_plan),
        "bench_pressure": _ref_dict(profile.bench_pressure),
        "damage_counter_plan": _refs(profile.damage_counter_plan),
        "ability_damage_counter_plan": _refs(profile.ability_damage_counter_plan),
        "switch_gust_plan": _refs(profile.switch_gust_plan),
        "lock_control_plan": _refs(profile.lock_control_plan),
        "protection_plan": _ref_dict(profile.protection_plan),
        "healing_recovery_plan": _refs(profile.healing_recovery_plan),
        "disruption_plan": _refs(profile.disruption_plan),
        "status_plan": _refs(profile.status_plan),
        "dependencies": _deps(profile.dependencies),
        "choke_points": ";".join(profile.choke_points),
        "manual_review_reasons": " | ".join(profile.manual_review_reasons),
    }


def profiles_to_csv_text(profiles: list[DeckProfile]) -> str:
    buffer = _io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for profile in profiles:
        writer.writerow(profile_to_csv_row(profile))
    return buffer.getvalue()


def write_profiles_csv(path: str | Path, profiles: list[DeckProfile]) -> Path:
    return write_text(Path(path), profiles_to_csv_text(profiles))


def write_profiles_jsonl(path: str | Path, profiles: list[DeckProfile]) -> Path:
    return write_jsonl(Path(path), [p.to_json_dict() for p in profiles])
