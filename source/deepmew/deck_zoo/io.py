"""Load K2A meta-kg outputs (and full decklists) for the Deck Zoo build."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from deepmew.meta_kg import io as meta_io

DECK_ARCHETYPES = "deck_archetypes.csv"
ARCHETYPE_SUMMARY = "archetype_summary.csv"
DECK_SIMILARITY = "deck_similarity.csv"
CARD_CATALOG = "card_catalog.csv"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(value: str, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def find_latest_meta_kg(runs_root: str | Path = ".research/runs/meta_kg") -> Path:
    root = Path(runs_root)
    candidates = sorted(root.glob("*/*"))
    runs = [c for c in candidates if (c / DECK_ARCHETYPES).exists()]
    if not runs:
        raise FileNotFoundError(f"no meta_kg run with {DECK_ARCHETYPES} under {root}")
    return runs[-1]


def load_meta_kg(meta_kg_dir: str | Path) -> dict[str, Any]:
    """Load the K2A artifacts needed by the Deck Zoo, parsing JSON list cells."""
    root = Path(meta_kg_dir)
    decks_raw = _read_csv(root / DECK_ARCHETYPES)
    decks: list[dict[str, Any]] = []
    for row in decks_raw:
        decks.append(
            {
                "deck_hash": row["deck_hash"],
                "primary_archetype": row["primary_archetype"],
                "secondary_package": row.get("secondary_package", ""),
                "variant_label": row["variant_label"],
                "classification_status": row["classification_status"],
                "confidence": float(row.get("confidence") or 0.0),
                "artifact_count": int(row.get("artifact_count") or 0),
                "episode_count": int(row.get("episode_count") or 0),
                "core_cards": _json(row.get("core_cards"), []),
                "package_cards": _json(row.get("package_cards"), []),
                "staple_cards": _json(row.get("staple_cards"), []),
                "tech_cards": _json(row.get("tech_cards"), []),
                "representative_player_ids": _json(row.get("representative_player_ids"), []),
                "representative_episode_ids": _json(row.get("representative_episode_ids"), []),
            }
        )

    summary = _read_csv(root / ARCHETYPE_SUMMARY)
    archetype_summary = []
    for row in summary:
        archetype_summary.append(
            {
                "archetype": row["archetype"],
                "deck_count": int(row.get("deck_count") or 0),
                "artifact_count": int(row.get("artifact_count") or 0),
                "episode_count": int(row.get("episode_count") or 0),
                "meta_share_by_artifact": float(row.get("meta_share_by_artifact") or 0.0),
                "meta_share_by_deck": float(row.get("meta_share_by_deck") or 0.0),
                "avg_confidence": float(row.get("avg_confidence") or 0.0),
            }
        )

    similarity: dict[str, list[tuple[str, float, bool]]] = defaultdict(list)
    sim_path = root / DECK_SIMILARITY
    if sim_path.exists():
        for row in _read_csv(sim_path):
            similarity[row["deck_hash"]].append(
                (
                    row["neighbor_deck_hash"],
                    float(row.get("weighted_similarity") or 0.0),
                    str(row.get("same_primary_archetype")).lower() == "true",
                )
            )

    card_names: dict[str, str] = {}
    catalog_path = root / CARD_CATALOG
    if catalog_path.exists():
        for row in _read_csv(catalog_path):
            card_names[row["card_id"]] = row.get("card_name", "")

    return {
        "decks": decks,
        "archetype_summary": archetype_summary,
        "similarity": dict(similarity),
        "card_names": card_names,
    }


def load_full_decklists(input_dir: str | Path) -> dict[str, dict[str, int]]:
    """Map deck_hash -> {card_id: count} by re-aggregating mined deck artifacts."""
    artifacts = meta_io.load_deck_artifacts(input_dir)
    unique_decks, _conflicts = meta_io.aggregate_unique_decks(artifacts)
    return {deck_hash: dict(deck.cards) for deck_hash, deck in unique_decks.items()}
