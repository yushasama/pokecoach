from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationConfig:
    first_choice: str = "default"
    semantics: str = "default"
    attach_timing: str = "default"
    evolve_priority: str = "default"
    attach_bonus: int = 0
    attack_bonus: int = 0
    setup_bonus: int = 0
    attach_target: str = "default"


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    options: tuple[tuple[str, str | int], ...] = ()


@dataclass(frozen=True)
class DeckSpec:
    deck_id: str | None = None
    deck_hash: str | None = None
    path: str | None = None


def ablation_items(ablation: AblationConfig) -> tuple[tuple[str, str | int], ...]:
    items: list[tuple[str, str | int]] = []
    if ablation.first_choice != "default":
        items.append(("first", ablation.first_choice))
    if ablation.semantics != "default":
        items.append(("semantics", ablation.semantics))
    if ablation.attach_timing != "default":
        items.append(("attach_timing", ablation.attach_timing))
    if ablation.evolve_priority != "default":
        items.append(("evolve_priority", ablation.evolve_priority))
    if ablation.attach_bonus:
        items.append(("attach_bonus", ablation.attach_bonus))
    if ablation.attack_bonus:
        items.append(("attack_bonus", ablation.attack_bonus))
    if ablation.setup_bonus:
        items.append(("setup_bonus", ablation.setup_bonus))
    if ablation.attach_target != "default":
        items.append(("attach_target", ablation.attach_target))
    return tuple(items)


@dataclass(frozen=True)
class AgentSpec:
    raw: str
    base: str
    ablation: AblationConfig = field(default_factory=AblationConfig)
    policy: PolicySpec | None = None
    deck: DeckSpec | None = None

    @property
    def label(self) -> str:
        items = ablation_items(self.ablation)
        if not items:
            return self.base
        body = ",".join(f"{key}={value}" for key, value in items)
        return f"{self.base}[{body}]"

    @property
    def policy_spec(self) -> PolicySpec:
        return self.policy or PolicySpec(policy_id=self.base, options=ablation_items(self.ablation))
