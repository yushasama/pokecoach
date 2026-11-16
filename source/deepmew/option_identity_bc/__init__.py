from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EncodedDecision:
    features: list[list[float]]
    target_index: int
    legal_count: int
    metadata: dict[str, Any]
