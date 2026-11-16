"""Debug rendering for board plans."""

from __future__ import annotations

from deepmew.state_machine.state import BoardPlan


def describe_plan(plan: BoardPlan) -> str:
    parts = [
        f"state={plan.state.value}",
        f"energy_attached={plan.energy_attached}",
        f"legal_attack={plan.legal_attack_available}",
        f"legal_attach={plan.legal_attach_available}",
        f"legal_play={plan.legal_play_available}",
        f"bench={plan.bench_count}",
        f"active_shortfall={plan.active_shortfall}",
        f"best_shortfall={plan.best_shortfall}",
    ]
    if plan.closest_area is not None:
        parts.append(f"closest={plan.closest_area}[{plan.closest_index}]")
    return " ".join(parts)
