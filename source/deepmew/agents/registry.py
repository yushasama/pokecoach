from __future__ import annotations

import re
from dataclasses import dataclass

from deepmew.agents.specs import AblationConfig, AgentSpec, PolicySpec, ablation_items


BASE_AGENTS = ("random", "first-legal", "greedy", "legacy-v2", "v2.4", "v4")
AGENT_ALIASES = {
    "first": "first-legal",
    "first-legal": "first-legal",
    "greedy": "greedy",
    "dumb": "greedy",
    "legacy": "legacy-v2",
    "legacy-v2": "legacy-v2",
    "random": "random",
    "v2": "v2.4",
    "v2.4": "v2.4",
    "v4": "v4",
}

_AGENT_RE = re.compile(r"\s*([^\[\]]+)(?:\[(.*)\])?\s*")


@dataclass(frozen=True)
class AgentRegistry:
    aliases: dict[str, str]
    base_agents: tuple[str, ...]

    def parse(self, text: str) -> AgentSpec:
        return parse_agent_spec(text, registry=self)

    def resolve_base(self, base_raw: str) -> str:
        base = self.aliases.get(base_raw)
        if base is None:
            known = ", ".join(self.base_agents)
            raise ValueError(f"unknown agent: {base_raw}; known agents: {known}")
        return base


DEFAULT_REGISTRY = AgentRegistry(aliases=AGENT_ALIASES, base_agents=BASE_AGENTS)


def _parse_items(raw: str, body: str | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if not body:
        return values
    for item in body.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"invalid ablation item in {raw}: {item}")
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"duplicate ablation key for {raw}: {key}")
        values[key] = value.strip()
    return values


def _pop_choice(values: dict[str, str], *keys: str, default: str) -> str:
    for key in keys:
        if key in values:
            return values.pop(key).lower()
    return default


def _validate_choice(key: str, value: str, valid: set[str]) -> None:
    if value not in valid:
        raise ValueError(f"{key} must be one of {sorted(valid)}")


def _pop_int(values: dict[str, str], key: str) -> int:
    raw = values.pop(key, "0")
    try:
        return int(raw or 0)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def parse_agent_spec(raw: str, *, registry: AgentRegistry = DEFAULT_REGISTRY) -> AgentSpec:
    match = _AGENT_RE.fullmatch(raw)
    if match is None:
        raise ValueError(f"invalid agent spec: {raw}")
    base_raw = match.group(1).strip()
    base = registry.resolve_base(base_raw)
    values = _parse_items(raw, match.group(2))

    first_choice = _pop_choice(values, "first", "first_choice", default="default")
    semantics = _pop_choice(values, "semantics", default="default")
    attach_timing = _pop_choice(values, "attach_timing", default="default")
    evolve_priority = _pop_choice(values, "evolve_priority", default="default")
    attach_target = _pop_choice(values, "attach_target", default="default")

    _validate_choice("first", first_choice, {"default", "yes", "no"})
    _validate_choice("semantics", semantics, {"default", "on", "off"})
    _validate_choice("attach_timing", attach_timing, {"default", "new", "old"})
    _validate_choice("evolve_priority", evolve_priority, {"default", "new", "old"})
    _validate_choice("attach_target", attach_target, {"default", "active", "bench", "first"})

    config = AblationConfig(
        first_choice=first_choice,
        semantics=semantics,
        attach_timing=attach_timing,
        evolve_priority=evolve_priority,
        attach_bonus=_pop_int(values, "attach_bonus"),
        attack_bonus=_pop_int(values, "attack_bonus"),
        setup_bonus=_pop_int(values, "setup_bonus"),
        attach_target=attach_target,
    )
    if values:
        raise ValueError(f"unknown ablation keys for {raw}: {', '.join(sorted(values))}")
    return AgentSpec(
        raw=raw,
        base=base,
        ablation=config,
        policy=PolicySpec(policy_id=base, options=ablation_items(config)),
    )
