"""Run, deck, backend, and schema metadata for RL experiments.

This module is deliberately observational: it hashes the current deck and
schema surfaces, but does not affect observations, masks, rewards, or actions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from deepmew.deck.io import read_deck
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.deck_identity import agent_deck_identity_from_config
from deepmew.rl.decks import resolve_repo_path
from deepmew.rl.logging import sha256_json, sha256_text
from deepmew.rl.opponents import OPPONENT_SETS


DEFAULT_DECK_PATH = Path("deck.csv")
DEFAULT_CARD_POOL_VERSION = "cg_api_current"
DEFAULT_FORMAT_BUCKET = "pokemon_tcg_ai_battle_2026"


def _safe_path(path: Path | str) -> Path:
    return resolve_repo_path(path)


def normalized_deck_payload(
    deck: Sequence[int],
    *,
    card_pool_version: str = DEFAULT_CARD_POOL_VERSION,
) -> dict[str, Any]:
    counts = Counter(int(card_id) for card_id in deck)
    return {
        "card_pool_version": card_pool_version,
        "cards": [
            {"card_id": card_id, "count": counts[card_id]}
            for card_id in sorted(counts)
        ],
    }


def normalized_deck_hash(
    deck: Sequence[int],
    *,
    card_pool_version: str = DEFAULT_CARD_POOL_VERSION,
) -> str:
    payload = normalized_deck_payload(deck, card_pool_version=card_pool_version)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def deck_metadata(
    path: Path | str = DEFAULT_DECK_PATH,
    *,
    deck_id: str | None = None,
    deck_name: str | None = None,
    source: str | None = None,
    card_pool_version: str = DEFAULT_CARD_POOL_VERSION,
) -> dict[str, Any]:
    deck_path = _safe_path(path)
    deck = read_deck(deck_path)
    deck_hash = normalized_deck_hash(deck, card_pool_version=card_pool_version)
    resolved_id = deck_id or f"{deck_path.stem}:{deck_hash.split(':', 1)[1][:12]}"
    return {
        "deck_id": resolved_id,
        "deck_name": deck_name or deck_path.stem,
        "deck_source_path": str(deck_path),
        "deck_source": source or str(deck_path),
        "deck_hash": deck_hash,
        "card_count": len(deck),
        "unique_card_count": len(set(deck)),
        "card_pool_version": card_pool_version,
    }


def obs_schema_hash(schema_names: Sequence[str]) -> str:
    return sha256_text("\n".join(schema_names))


def action_schema_hash(*, max_options: int, action_mask: str = "legal_only") -> str:
    return sha256_json(
        {
            "schema": "deepmew.rl_action_schema.v0",
            "action_mask": action_mask,
            "max_options": int(max_options),
            "selection_semantics": "complete_ordered_option_indices",
        }
    )


def opponent_policy_ids(config: RLExperimentConfig) -> list[str]:
    entries = OPPONENT_SETS.get(config.opponent_sampler, [])
    if entries:
        return [raw for raw, _weight in entries]
    return ["first-legal", "v2.4"]


def opponent_deck_identities(policies: Iterable[str]) -> list[dict[str, Any]]:
    from deepmew.rl.opponents import OpponentAgent

    rows: list[dict[str, Any]] = []
    for policy_id in policies:
        try:
            rows.append(OpponentAgent.from_str(str(policy_id)).deck_identity.to_dict())
        except Exception as exc:
            rows.append(
                {
                    "policy_id": str(policy_id),
                    "policy_role": "unknown",
                    "deck_id": "unavailable",
                    "deck_role": "unavailable",
                    "deck_path": "",
                    "deck_source_path": "",
                    "deck_hash": "",
                    "deck_status": "unavailable",
                    "pairing_role": "unavailable",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def build_run_metadata(
    *,
    config: RLExperimentConfig,
    schema_names: Sequence[str],
    max_options: int,
    trainer_backend: str,
    advantage_backend: str = "unknown",
    vector_backend: str = "unknown",
    puffer_vectorized_status: str = "not_attempted",
    opponent_policies: Iterable[str] | None = None,
    agent_deck_path: Path | str = DEFAULT_DECK_PATH,
    opponent_deck_path: Path | str = DEFAULT_DECK_PATH,
    seat_policy: str = "alternating",
    format_bucket: str = DEFAULT_FORMAT_BUCKET,
    engine_version: str = "cg_local",
    card_pool_version: str = DEFAULT_CARD_POOL_VERSION,
) -> dict[str, Any]:
    if agent_deck_path == DEFAULT_DECK_PATH:
        agent_deck_path = config.agent_deck_path
    if opponent_deck_path == DEFAULT_DECK_PATH:
        opponent_deck_path = config.opponent_deck_path
    agent_identity = agent_deck_identity_from_config(config)
    policies = list(opponent_policies) if opponent_policies is not None else opponent_policy_ids(config)
    opponent_rows = opponent_deck_identities(policies)
    available_hashes = sorted({row.get("deck_hash", "") for row in opponent_rows if row.get("deck_hash")})
    top_opponent_deck_id = (
        opponent_rows[0].get("deck_id", "")
        if len({row.get("deck_id", "") for row in opponent_rows}) == 1 and opponent_rows
        else "per_opponent_explicit"
    )
    top_opponent_deck_hash = available_hashes[0] if len(available_hashes) == 1 else "per_opponent_explicit"
    return {
        "schema_version": "deepmew.rl_run_metadata.v0",
        "experiment": config.experiment_id,
        "trainer_backend": trainer_backend,
        "advantage_backend": advantage_backend,
        "vector_backend": vector_backend,
        "puffer_vectorized_status": puffer_vectorized_status,
        "obs_schema_hash": obs_schema_hash(schema_names),
        "action_schema_hash": action_schema_hash(max_options=max_options, action_mask=config.action_mask),
        "agent_policy_id": config.experiment_id,
        "agent_deck_id": agent_identity.deck_id,
        "agent_deck_name": agent_identity.deck_id,
        "agent_deck_source_path": agent_identity.deck_source_path,
        "agent_deck_source": agent_identity.deck_source_id or agent_identity.deck_source_path,
        "agent_deck_hash": agent_identity.deck_hash,
        "agent_deck_status": agent_identity.deck_status,
        "opponent_deck_id": top_opponent_deck_id,
        "opponent_deck_name": top_opponent_deck_id,
        "opponent_deck_source_path": "per_opponent_explicit",
        "opponent_deck_source": "per_opponent_explicit",
        "opponent_deck_hash": top_opponent_deck_hash,
        "opponents": opponent_rows,
        "seat_policy": seat_policy,
        "format_bucket": format_bucket,
        "card_pool_version": card_pool_version,
        "engine_version": engine_version,
        "deck_config_locked": True,
    }


def checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    extra = payload.get("extra")
    if isinstance(extra, dict):
        metadata = extra.get("run_metadata")
        if isinstance(metadata, dict):
            return metadata
    return None


def append_eval_summary_csv(
    path: Path | str,
    *,
    checkpoint: Path | str | None,
    eval_payload: dict[str, Any],
    run_metadata: dict[str, Any],
    seed: int,
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "opponent_policy",
        "agent_deck_id",
        "agent_deck_hash",
        "opponent_deck_id",
        "opponent_deck_hash",
        "seat",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "seed",
        "agent_policy_id",
        "opponent_policy_id",
        "opponent_pairing_role",
        "opponent_deck_status",
        "deck_source_path",
        "obs_schema_hash",
        "action_schema_hash",
    ]
    exists = csv_path.exists()
    per_opponent = (eval_payload.get("result") or eval_payload).get("per_opponent", {})
    opponent_rows = {
        str(row.get("policy_id") or ""): row
        for row in (run_metadata.get("opponents") or [])
        if isinstance(row, dict)
    }
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for opponent_policy, bucket in per_opponent.items():
            row_identity = opponent_rows.get(str(opponent_policy), {})
            if isinstance(bucket, dict) and bucket.get("opponent_deck_id"):
                row_identity = {
                    "policy_id": bucket.get("opponent_policy_id", opponent_policy),
                    "deck_id": bucket.get("opponent_deck_id", ""),
                    "deck_hash": bucket.get("opponent_deck_hash", ""),
                    "pairing_role": bucket.get("opponent_pairing_role", ""),
                    "deck_status": bucket.get("opponent_deck_status", ""),
                    "deck_source_path": bucket.get("deck_source_path", ""),
                }
            games = int(bucket.get("games") or bucket.get("episodes") or 0)
            wins = int(bucket.get("wins") or 0)
            losses = int(bucket.get("losses") or 0)
            draws = int(bucket.get("draws") or 0)
            completed = wins + losses + draws
            writer.writerow(
                {
                    "checkpoint": str(checkpoint) if checkpoint is not None else "",
                    "opponent_policy": opponent_policy,
                    "agent_deck_id": run_metadata.get("agent_deck_id", ""),
                    "agent_deck_hash": run_metadata.get("agent_deck_hash", ""),
                    "opponent_deck_id": row_identity.get("deck_id", run_metadata.get("opponent_deck_id", "")),
                    "opponent_deck_hash": row_identity.get("deck_hash", run_metadata.get("opponent_deck_hash", "")),
                    "seat": run_metadata.get("seat_policy", "alternating"),
                    "games": games,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": round(wins / completed, 6) if completed else 0.0,
                    "seed": seed,
                    "agent_policy_id": run_metadata.get("agent_policy_id", run_metadata.get("experiment", "")),
                    "opponent_policy_id": row_identity.get("policy_id", opponent_policy),
                    "opponent_pairing_role": row_identity.get("pairing_role", ""),
                    "opponent_deck_status": row_identity.get("deck_status", ""),
                    "deck_source_path": row_identity.get("deck_source_path", ""),
                    "obs_schema_hash": run_metadata.get("obs_schema_hash", ""),
                    "action_schema_hash": run_metadata.get("action_schema_hash", ""),
                }
            )
