from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    index: int
    option: dict[str, Any]
    option_type: str
    select_type: str
    context: str
    score: int
    reasons: tuple[str, ...]
