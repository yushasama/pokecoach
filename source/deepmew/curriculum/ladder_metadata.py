"""Training-only ladder metadata helpers for AWR refgame rows.

These labels are coarse curriculum coordinates, not Kaggle ladder truth. They
make generated rows groupable without changing policy/runtime behavior.
"""

from __future__ import annotations

import re
from typing import Any

UNKNOWN = "unknown"

SHELL_LABELS = (
    ("archaludon", ("archaludon",)),
    ("mega_lucario", ("lucario", "riolu", "solrock")),
    ("alakazam", ("alakazam", "kadabra", "abra")),
    ("grimmsnarl", ("grimmsnarl", "morgrem", "impidimp")),
    ("mega_starmie", ("starmie", "staryu")),
    ("garchomp", ("garchomp", "gabite", "gible")),
    ("dragapult", ("dragapult", "drakloak", "dreepy")),
    ("crustle_ogerpon", ("crustle", "dwebble", "ogerpon")),
    ("fire_probe", ("cinderace", "typhlosion", "charizard", "fire")),
)


def _clean(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("mega ", "mega_")

    return text


def normalize_opponent_shell(value: Any) -> str:
    text = _clean(value)
    if not text or text in {"?", "none", "null", "unknown"}:
        return UNKNOWN
    for label, needles in SHELL_LABELS:
        if any(needle in text for needle in needles):
            return label

    return UNKNOWN


def normalize_opponent_bucket(value: Any) -> str:
    text = _clean(value).replace("-", "_")
    if not text or text in {"?", "none", "null", "unknown"}:
        return UNKNOWN
    if text in {"600_800", "800_900", "900_1000", "1000_plus"}:
        return text
    match = re.match(r"^(\d+)[_-](\d+)$", text)
    if not match:
        return UNKNOWN
    low = int(match.group(1))
    high = int(match.group(2))
    if low < 800 and high <= 800:
        return "600_800"
    if low < 900 and high <= 900:
        return "800_900"
    if low < 1000 and high <= 1000:
        return "900_1000"
    if low >= 1000 or high > 1000:
        return "1000_plus"

    return UNKNOWN


def normalize_opponent_pilot(value: Any, *, source_label: Any = "") -> str:
    text = _clean(value)
    source = _clean(source_label)
    joined = f"{text} {source}".strip()
    if not joined or joined in {"?", "none", "null", "unknown"}:
        return UNKNOWN
    if "exploiter" in joined:
        return "exploiter"
    if "replay" in joined:
        return "replay"
    if "greedy" in joined:
        return "greedy"
    if "mirror" in joined:
        return "self_snapshot"
    if "checkpoint:" in joined:
        return "self_snapshot"
    if "self" in joined:
        return "self"
    if "high_band" in joined or "bd27" in joined:
        return "high_band"
    if "reference" in joined or "baseline" in joined or "v2.4" in joined or "legacy-v2" in joined:
        return "baseline"

    return UNKNOWN


def derive_training_phase(*, opponent_bucket: str, opponent_pilot: str,
                          opponent_shell: str, source_label: Any = "") -> str:
    source = _clean(source_label)
    if opponent_pilot == "exploiter":
        return "exploiter"
    if "mirror" in source or opponent_pilot in {"self", "self_snapshot"}:
        return "mirror"
    if opponent_pilot == "high_band":
        return "high_band"
    if opponent_bucket == "600_800":
        return "placement"
    if opponent_bucket == "800_900":
        return "low_climb"
    if opponent_bucket == "900_1000":
        return "wall_800_1000"
    if opponent_bucket == "1000_plus":
        return "high_band"
    if opponent_shell == "mega_starmie":
        return "mirror" if "starmie" in source else UNKNOWN

    return UNKNOWN


def build_matchup_cell_key(row: dict[str, Any]) -> str:
    values = [
        str(row.get("training_phase") or UNKNOWN),
        str(row.get("opponent_bucket") or UNKNOWN),
        str(row.get("opponent_pilot") or UNKNOWN),
        str(row.get("opponent_shell") or UNKNOWN),
        str(row.get("opponent_deck_hash") or UNKNOWN),
    ]

    return "|".join(values)


def ladder_metadata(
    *,
    opponent_shell: Any = "",
    opponent_bucket: Any = "",
    opponent_pilot: Any = "",
    opponent_policy_id: Any = "",
    opponent_label: Any = "",
    bucket: Any = "",
    opponent_deck_hash: Any = "",
    training_phase: Any = "",
) -> dict[str, str]:
    shell = normalize_opponent_shell(opponent_shell or opponent_label or opponent_policy_id)
    bucket_label = normalize_opponent_bucket(opponent_bucket or bucket)
    pilot = normalize_opponent_pilot(opponent_pilot or opponent_policy_id, source_label=opponent_label)
    phase = _clean(training_phase)
    if not phase or phase == UNKNOWN:
        phase = derive_training_phase(
            opponent_bucket=bucket_label,
            opponent_pilot=pilot,
            opponent_shell=shell,
            source_label=opponent_label or opponent_policy_id,
        )
    row = {
        "opponent_shell": shell,
        "opponent_bucket": bucket_label,
        "opponent_pilot": pilot,
        "training_phase": phase or UNKNOWN,
        "opponent_deck_hash": str(opponent_deck_hash or UNKNOWN),
    }
    row["matchup_cell_key"] = build_matchup_cell_key(row)
    row.pop("opponent_deck_hash")

    return row


def enrich_refgame_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = ladder_metadata(
        opponent_shell=row.get("opponent_shell"),
        opponent_bucket=row.get("opponent_bucket"),
        opponent_pilot=row.get("opponent_pilot"),
        opponent_policy_id=row.get("opponent_policy_id"),
        opponent_label=row.get("opponent_policy_id") or row.get("opponent_deck_id"),
        bucket=row.get("bucket"),
        opponent_deck_hash=row.get("opponent_deck_hash"),
        training_phase=row.get("training_phase"),
    )
    for key, value in metadata.items():
        row.setdefault(key, value)
    row["matchup_cell_key"] = build_matchup_cell_key(row)

    return row
