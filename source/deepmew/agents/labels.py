from __future__ import annotations

import re

from deepmew.agents.specs import AblationConfig, AgentSpec, ablation_items


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def filesystem_safe_label(value: str) -> str:
    safe = _SAFE_CHARS.sub("_", value.strip()).strip("._-")
    return safe or "agent"


def canonical_agent_label(spec: AgentSpec) -> str:
    parts = [spec.base]
    if spec.base == "v4":
        parts.append("arena_only")
    for key, value in ablation_items(spec.ablation):
        parts.append(f"{key}-{value}")
    return filesystem_safe_label("__".join(str(part) for part in parts))


def is_experimental(spec: AgentSpec) -> bool:
    return spec.base == "v4" or spec.ablation != AblationConfig()


def agent_family(spec: AgentSpec) -> str:
    if spec.base == "v4":
        return "v4_arena_only"
    return filesystem_safe_label(spec.base.replace(".", "_"))
