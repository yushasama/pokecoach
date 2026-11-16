"""Metadata-only attribution audit for the frozen V0 corpus.

This module never loads target labels, model predictions, or feature rows. It
uses only the accepted cache eligibility mask, block metadata, and canonical
game sidecars to answer which acting decks and policies produced V-eligible
rows. Exact deck hashes remain provenance; they are not model inputs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from deepmew.artifacts.writers import write_json, write_text
from deepmew.rl.metadata import normalized_deck_hash


SCHEMA_VERSION = "deepmew.v0_corpus_attribution.v1"
MANIFEST_VERSION = "deepmew.run_manifest.v1"

IMPLICIT_STARMIE_STRATEGY_SPECIALIST = "IMPLICIT_STARMIE_STRATEGY_SPECIALIST"
STARMIE_HEAVY_MIXED_STRATEGY = "STARMIE_HEAVY_MIXED_STRATEGY"
MIXED_STRATEGY_UNIVERSAL = "MIXED_STRATEGY_UNIVERSAL"
UNRESOLVED_ATTRIBUTION = "UNRESOLVED_ATTRIBUTION"

DEFAULT_MIN_IDENTITY_COVERAGE = 0.99
DEFAULT_SPECIALIST_SHARE = 0.50
DEFAULT_HEAVY_SHARE = 0.20
DEFAULT_MIXED_MIN_DECK_SHARE = 0.05
DEFAULT_MIXED_MIN_DECKS = 3

# Formal K2A core-card evidence for Mega Starmie ex in the pinned R16 labels.
STARMIE_CORE_CARD_ID = 1031

POLICY_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("public_replay_unknown", re.compile(r"^kaggle:", re.IGNORECASE)),
    ("checkpoint", re.compile(r"checkpoint:|\.pt(?:$|[^a-z])", re.IGNORECASE)),
    ("reference", re.compile(r"reference[-_]", re.IGNORECASE)),
    ("exploiter", re.compile(r"exploiter", re.IGNORECASE)),
    ("replay", re.compile(r"replay", re.IGNORECASE)),
    ("greedy", re.compile(r"greedy", re.IGNORECASE)),
    ("v2_4", re.compile(r"v2\.4", re.IGNORECASE)),
)


@dataclass(frozen=True)
class BlockAttribution:
    block_key: str
    game_id: str
    source_id: str
    unit_id: str
    seat: int
    eligible_rows: int
    deck_hash: str | None
    opponent_deck_hash: str | None
    opponent_deck_source: str
    deck_id: str | None
    opponent_deck_id: str | None
    policy_id: str
    opponent_policy_id: str
    information_mode: str
    source_rating: float | None
    opponent_difficulty_bucket: str | None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")

    return payload


def _read_deck(path: str | Path) -> list[int]:
    cards = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                cards.append(int(value))
    if len(cards) != 60:
        raise ValueError(f"expected 60 cards in {path}, found {len(cards)}")

    return cards


def policy_family(policy_id: str) -> str:
    for family, pattern in POLICY_FAMILY_PATTERNS:
        if pattern.search(policy_id):
            return family

    return "named_policy"


def strength_bucket(source_rating: float | None) -> str:
    if source_rating is None or not math.isfinite(source_rating):
        return "not_instrumented"
    lower = math.floor(source_rating / 100.0) * 100

    return f"{lower}-{lower + 99}"


def classify_v0_attribution(
    *,
    identity_coverage: float,
    exact_starmie_share: float,
    any_starmie_share: float,
    acting_deck_shares: Sequence[float],
    min_identity_coverage: float = DEFAULT_MIN_IDENTITY_COVERAGE,
    specialist_share: float = DEFAULT_SPECIALIST_SHARE,
    heavy_share: float = DEFAULT_HEAVY_SHARE,
    mixed_min_deck_share: float = DEFAULT_MIXED_MIN_DECK_SHARE,
    mixed_min_decks: int = DEFAULT_MIXED_MIN_DECKS,
) -> str:
    """Apply a transparent interpretation rubric to exact corpus shares."""
    if identity_coverage < min_identity_coverage:
        return UNRESOLVED_ATTRIBUTION
    if exact_starmie_share >= specialist_share:
        return IMPLICIT_STARMIE_STRATEGY_SPECIALIST
    if any_starmie_share >= heavy_share:
        return STARMIE_HEAVY_MIXED_STRATEGY
    significant_decks = sum(share >= mixed_min_deck_share for share in acting_deck_shares)
    largest_share = max(acting_deck_shares, default=1.0)
    if significant_decks >= mixed_min_decks and largest_share < specialist_share:
        return MIXED_STRATEGY_UNIVERSAL

    return UNRESOLVED_ATTRIBUTION


def _load_archetype_labels(path: str | Path) -> dict[str, dict[str, str]]:
    labels = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            deck_hash = str(row.get("deck_hash") or "")
            if deck_hash:
                labels[deck_hash] = {
                    "primary_archetype": str(row.get("primary_archetype") or ""),
                    "secondary_package": str(row.get("secondary_package") or ""),
                    "variant_label": str(row.get("variant_label") or ""),
                    "classification_status": str(row.get("classification_status") or ""),
                }

    return labels


def _load_pool_starmie_hashes(path: str | Path) -> set[str]:
    payload = _read_json(path)
    pools = payload.get("pools")
    if not isinstance(pools, dict):
        raise ValueError("deck pool artifact is missing pools")
    cards_by_hash: dict[str, tuple[int, ...]] = {}
    for rows in pools.values():
        if not isinstance(rows, list):
            raise ValueError("deck pool bucket must contain a list")
        for row in rows:
            deck_hash = str(row.get("deck_hash") or "")
            cards = tuple(sorted(int(card_id) for card_id in row.get("card_ids") or []))
            if not deck_hash or len(cards) != 60:
                raise ValueError("deck pool row requires a hash and 60 cards")
            if normalized_deck_hash(list(cards)) != deck_hash:
                raise ValueError(f"deck pool hash mismatch for {deck_hash}")
            prior = cards_by_hash.setdefault(deck_hash, cards)
            if prior != cards:
                raise ValueError(f"deck pool hash collision for {deck_hash}")

    return {
        deck_hash
        for deck_hash, cards in cards_by_hash.items()
        if STARMIE_CORE_CARD_ID in cards
    }


def is_starmie_archetype_label(label: Mapping[str, str]) -> bool:
    """Return whether formal K2A semantics identify Starmie anywhere in the list."""
    semantic_fields = (
        label.get("primary_archetype", ""),
        label.get("secondary_package", ""),
        label.get("variant_label", ""),
    )

    return any("starmie" in value.casefold() for value in semantic_fields)


def resolve_starmie_hashes(
    *,
    observed_hashes: Iterable[str],
    archetype_labels: Mapping[str, Mapping[str, str]],
    explicit_hashes: Iterable[str],
    pool_starmie_hashes: Iterable[str] = (),
) -> tuple[set[str], list[dict[str, Any]]]:
    """Resolve formal Starmie semantics across short K2A and full pool hashes."""
    support: dict[str, set[str]] = defaultdict(set)
    explicit = set(explicit_hashes)
    for deck_hash in explicit:
        support[deck_hash].add("explicit_full_list")
    observed = set(observed_hashes)
    for deck_hash in set(pool_starmie_hashes) & observed:
        support[deck_hash].add("pool_full_list_core_card_1031")

    labeled = {
        key: label
        for key, label in archetype_labels.items()
        if is_starmie_archetype_label(label)
    }
    for observed_hash in observed:
        observed_token = observed_hash.removeprefix("sha256:")
        matching_keys = []
        for label_hash in labeled:
            label_token = label_hash.removeprefix("sha256:")
            is_exact = observed_token == label_token
            is_short_to_full_bridge = (
                len(label_token) == 16
                and len(observed_token) == 64
                and observed_token.startswith(label_token)
            )
            if is_exact or is_short_to_full_bridge:
                matching_keys.append(label_hash)
        if len(matching_keys) > 1:
            raise ValueError(
                f"ambiguous K2A hash bridge for {observed_hash}: {matching_keys}"
            )
        if not matching_keys:
            continue
        label_hash = matching_keys[0]
        label = labeled[label_hash]
        if "starmie" in label.get("primary_archetype", "").casefold():
            support[observed_hash].add("primary_archetype")
        if "starmie" in label.get("secondary_package", "").casefold():
            support[observed_hash].add("secondary_package")
        if observed_hash != label_hash:
            support[observed_hash].add("normalized_pool_hash_bridge")

    provenance = [
        {
            "deck_hash": deck_hash,
            "sources": sorted(sources),
        }
        for deck_hash, sources in sorted(support.items())
    ]

    return set(support), provenance


def _load_game_records(
    canonical_root: str | Path,
    cache_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    root = Path(canonical_root)
    records = {}
    for item in cache_manifest.get("inputs") or []:
        source_id = str(item["source_id"])
        unit_id = str(item["unit_id"])
        part = str(item["part"])
        path = root / source_id / unit_id / f"{part}.games.jsonl.gz"
        if not path.is_file():
            raise ValueError(f"missing canonical game sidecar: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = f"{record['canonical_game_id']}:s{record['seat']}"
                if key in records:
                    raise ValueError(f"duplicate canonical block sidecar {key}")
                records[key] = record

    return records


def _eligible_rows_by_block(
    cache_dir: str | Path,
    cache_manifest: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    cache = Path(cache_dir)
    with gzip.open(cache / "blocks_keys.json.gz", "rt", encoding="utf-8") as handle:
        keys = json.load(handle)
    with np.load(cache / "blocks.npz", allow_pickle=False) as payload:
        blocks = {name: payload[name] for name in payload.files if name != "z"}
    with np.load(cache / "meta.npz", allow_pickle=False) as payload:
        eligible = payload["is_target_eligible"]
    if eligible.dtype != np.bool_ or eligible.shape != (int(cache_manifest["rows"]),):
        raise ValueError("cache eligibility mask shape or dtype mismatch")
    starts = blocks["row_start"].astype(np.int64, copy=False)
    lengths = blocks["row_len"].astype(np.int64, copy=False)
    if len(starts) != int(cache_manifest["blocks"]):
        raise ValueError("cache block count mismatch")
    prefix = np.empty(len(eligible) + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(eligible, dtype=np.int64, out=prefix[1:])
    counts = prefix[starts + lengths] - prefix[starts]
    if int(counts.sum()) != int(cache_manifest["eligible_target_rows"]):
        raise ValueError("block attribution does not cover every eligible target")

    return blocks, keys, counts


def _build_blocks(
    *,
    cache_dir: str | Path,
    canonical_root: str | Path,
    cache_manifest: Mapping[str, Any],
) -> list[BlockAttribution]:
    arrays, keys, eligible_counts = _eligible_rows_by_block(cache_dir, cache_manifest)
    game_records = _load_game_records(canonical_root, cache_manifest)
    block_keys = keys["block_keys"]
    if set(block_keys) != set(game_records):
        raise ValueError("canonical game sidecars do not exactly match cache blocks")

    blocks = []
    for index, block_key in enumerate(block_keys):
        record = game_records[block_key]
        source_id = keys["source_ids"][int(arrays["source_idx"][index])]
        deck_hash = keys["deck_vocab"][int(arrays["deck_gidx"][index])]
        opponent_hash = keys["deck_vocab"][int(arrays["opp_deck_gidx"][index])]
        policy_id = keys["policy_vocab"][int(arrays["policy_gidx"][index])]
        opponent_policy_id = keys["policy_vocab"][int(arrays["opp_policy_gidx"][index])]
        if str(record.get("source_id")) != source_id:
            raise ValueError(f"source mismatch for {block_key}")
        if str(record.get("deck_hash")) != deck_hash:
            raise ValueError(f"acting deck mismatch for {block_key}")
        if str(record.get("policy_id")) != policy_id:
            raise ValueError(f"acting policy mismatch for {block_key}")
        if int(record.get("seat", -1)) != int(arrays["seat"][index]):
            raise ValueError(f"acting seat mismatch for {block_key}")
        if str(record.get("opponent_policy_id")) != opponent_policy_id:
            raise ValueError(f"opponent policy mismatch for {block_key}")
        if record.get("information_mode") != "actor_visible":
            raise ValueError(f"information mode mismatch for {block_key}")
        blocks.append(BlockAttribution(
            block_key=block_key,
            game_id=str(record["canonical_game_id"]),
            source_id=source_id,
            unit_id=str(record["unit_id"]),
            seat=int(record["seat"]),
            eligible_rows=int(eligible_counts[index]),
            deck_hash=None if deck_hash in {"", "None"} else deck_hash,
            opponent_deck_hash=None if opponent_hash in {"", "None"} else opponent_hash,
            opponent_deck_source=(
                "unresolved" if opponent_hash in {"", "None"} else "recorded"
            ),
            deck_id=str(record["deck_id"]) if record.get("deck_id") is not None else None,
            opponent_deck_id=(
                str(record["opponent_deck_id"])
                if record.get("opponent_deck_id") is not None
                else None
            ),
            policy_id=policy_id,
            opponent_policy_id=opponent_policy_id,
            information_mode=str(record.get("information_mode") or ""),
            source_rating=(
                float(record["source_rating"])
                if isinstance(record.get("source_rating"), (int, float))
                and not isinstance(record.get("source_rating"), bool)
                else None
            ),
            opponent_difficulty_bucket=(
                str(record["bucket"]) if record.get("bucket") is not None else None
            ),
        ))

    return resolve_missing_opponent_decks(blocks)


def resolve_missing_opponent_decks(
    blocks: Sequence[BlockAttribution],
) -> list[BlockAttribution]:
    decks_by_game_seat: dict[str, dict[int, str]] = defaultdict(dict)
    for block in blocks:
        if block.deck_hash:
            decks_by_game_seat[block.game_id][block.seat] = block.deck_hash
    resolved = []
    for block in blocks:
        opponent = block.opponent_deck_hash
        if opponent is None:
            opponent = decks_by_game_seat[block.game_id].get(1 - block.seat)
        source = block.opponent_deck_source
        if opponent is not None and source == "unresolved":
            source = "paired_actor_block"
        resolved.append(replace(
            block,
            opponent_deck_hash=opponent,
            opponent_deck_source=source,
        ))

    return resolved


def _ranked_rows(
    counter: Mapping[Any, int],
    *,
    total: int,
    key_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        values = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(key_names, values)}
        row["eligible_rows"] = int(count)
        row["eligible_row_share"] = round(count / total, 9) if total else 0.0
        rows.append(row)

    return rows


def _top_display(values: Counter[str]) -> str | None:
    if not values:
        return None

    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][0]


def summarize_blocks(
    blocks: Sequence[BlockAttribution],
    *,
    exact_starmie_hash: str,
    starmie_hashes: set[str],
    starmie_hash_provenance: Sequence[Mapping[str, Any]],
    archetype_labels: Mapping[str, Mapping[str, str]],
    uncontrolled_sources: set[str],
) -> dict[str, Any]:
    total_rows = sum(block.eligible_rows for block in blocks)
    if total_rows <= 0:
        raise ValueError("attribution requires eligible rows")
    eligible_blocks = [block for block in blocks if block.eligible_rows > 0]
    game_blocks: dict[str, list[BlockAttribution]] = defaultdict(list)
    for block in blocks:
        game_blocks[block.game_id].append(block)
    eligible_games = {
        game_id
        for game_id, members in game_blocks.items()
        if sum(member.eligible_rows for member in members) > 0
    }

    counters: dict[str, Counter[Any]] = {
        "source": Counter(),
        "acting_deck": Counter(),
        "matchup": Counter(),
        "acting_seat": Counter(),
        "controlled_agent_seat": Counter(),
        "policy_id": Counter(),
        "policy_family": Counter(),
        "generation": Counter(),
        "checkpoint": Counter(),
        "strength_bucket": Counter(),
        "opponent_difficulty_bucket": Counter(),
        "information_mode": Counter(),
        "opponent_deck_source": Counter(),
    }
    deck_displays: dict[str, Counter[str]] = defaultdict(Counter)
    opponent_displays: dict[str, Counter[str]] = defaultdict(Counter)
    source_games: dict[str, set[str]] = defaultdict(set)
    deck_games: dict[str, set[str]] = defaultdict(set)
    matchup_games: dict[tuple[str, str], set[str]] = defaultdict(set)
    semantic_rows = 0
    confident_semantic_rows = 0
    for block in eligible_blocks:
        rows = block.eligible_rows
        acting_deck = block.deck_hash or "unattributed"
        opponent_deck = block.opponent_deck_hash or "unattributed"
        matchup = (acting_deck, opponent_deck)
        controlled_seat = (
            "not_applicable_public_replay"
            if block.source_id in uncontrolled_sources
            else str(block.seat)
        )
        checkpoint = block.policy_id if policy_family(block.policy_id) == "checkpoint" else "not_applicable"
        counters["source"][block.source_id] += rows
        counters["acting_deck"][acting_deck] += rows
        counters["matchup"][matchup] += rows
        counters["acting_seat"][str(block.seat)] += rows
        counters["controlled_agent_seat"][controlled_seat] += rows
        counters["policy_id"][block.policy_id] += rows
        counters["policy_family"][policy_family(block.policy_id)] += rows
        counters["generation"][f"{block.source_id}/{block.unit_id}"] += rows
        counters["checkpoint"][checkpoint] += rows
        counters["strength_bucket"][strength_bucket(block.source_rating)] += rows
        counters["opponent_difficulty_bucket"][block.opponent_difficulty_bucket or "not_instrumented"] += rows
        counters["information_mode"][block.information_mode or "unattributed"] += rows
        counters["opponent_deck_source"][block.opponent_deck_source] += rows
        source_games[block.source_id].add(block.game_id)
        deck_games[acting_deck].add(block.game_id)
        matchup_games[matchup].add(block.game_id)
        if block.deck_id:
            deck_displays[acting_deck][block.deck_id] += rows
        if block.opponent_deck_id:
            opponent_displays[opponent_deck][block.opponent_deck_id] += rows
        label = archetype_labels.get(acting_deck, {})
        if acting_deck in starmie_hashes or label.get("primary_archetype"):
            semantic_rows += rows
        if acting_deck in starmie_hashes or label.get("classification_status") == "confident":
            confident_semantic_rows += rows

    exact_rows = counters["acting_deck"][exact_starmie_hash]
    exact_games = len(deck_games[exact_starmie_hash])
    any_starmie_rows = sum(counters["acting_deck"][deck_hash] for deck_hash in starmie_hashes)
    starmie_games = {
        game_id
        for game_id in eligible_games
        if any(
            block.deck_hash in starmie_hashes
            or block.opponent_deck_hash in starmie_hashes
            for block in game_blocks[game_id]
        )
    }
    starmie_acting_games = {
        game_id
        for game_id in eligible_games
        if any(block.deck_hash in starmie_hashes for block in game_blocks[game_id])
    }
    starmie_actor_rows = sum(
        block.eligible_rows
        for game_id in starmie_games
        for block in game_blocks[game_id]
        if block.deck_hash in starmie_hashes
    )
    non_starmie_actor_rows = sum(
        block.eligible_rows
        for game_id in starmie_games
        for block in game_blocks[game_id]
        if block.deck_hash not in starmie_hashes
    )
    starmie_game_rows = starmie_actor_rows + non_starmie_actor_rows
    controlled_starmie_game_rows = sum(
        block.eligible_rows
        for game_id in starmie_games
        for block in game_blocks[game_id]
        if block.source_id not in uncontrolled_sources
    )
    public_replay_starmie_game_rows = starmie_game_rows - controlled_starmie_game_rows
    both_sides_starmie_games = sum(
        1
        for game_id in starmie_games
        if all(
            block.deck_hash in starmie_hashes
            and block.opponent_deck_hash in starmie_hashes
            for block in game_blocks[game_id]
        )
    )
    exact_variant_mirror_games = sum(
        1
        for game_id in starmie_games
        if any(
            block.deck_hash in starmie_hashes
            and block.deck_hash == block.opponent_deck_hash
            for block in game_blocks[game_id]
        )
    )
    attributed_rows = total_rows - counters["acting_deck"]["unattributed"]
    identity_coverage = attributed_rows / total_rows
    deck_shares = [count / total_rows for count in counters["acting_deck"].values()]
    classification = classify_v0_attribution(
        identity_coverage=identity_coverage,
        exact_starmie_share=exact_rows / total_rows,
        any_starmie_share=any_starmie_rows / total_rows,
        acting_deck_shares=deck_shares,
    )

    acting_decks = _ranked_rows(
        counters["acting_deck"], total=total_rows, key_names=("deck_hash",)
    )
    for row in acting_decks:
        deck_hash = row["deck_hash"]
        label = archetype_labels.get(deck_hash, {})
        row["deck_id"] = _top_display(deck_displays[deck_hash])
        row["primary_archetype"] = label.get("primary_archetype") or None
        row["classification_status"] = label.get("classification_status") or None
        row["eligible_games"] = len(deck_games[deck_hash])
        row["is_exact_submitted_starmie"] = deck_hash == exact_starmie_hash
        row["is_supported_starmie_variant"] = deck_hash in starmie_hashes

    matchups = _ranked_rows(
        counters["matchup"],
        total=total_rows,
        key_names=("acting_deck_hash", "opponent_deck_hash"),
    )
    for row in matchups:
        acting_hash = row["acting_deck_hash"]
        opponent_hash = row["opponent_deck_hash"]
        row["acting_deck_id"] = _top_display(deck_displays[acting_hash])
        row["opponent_deck_id"] = _top_display(opponent_displays[opponent_hash])
        row["eligible_games"] = len(matchup_games[(acting_hash, opponent_hash)])

    by_source = _ranked_rows(counters["source"], total=total_rows, key_names=("source_id",))
    for row in by_source:
        row["eligible_games"] = len(source_games[row["source_id"]])

    return {
        "classification": {
            "verdict": classification,
            "rubric": {
                "minimum_identity_coverage": DEFAULT_MIN_IDENTITY_COVERAGE,
                "exact_starmie_specialist_share": DEFAULT_SPECIALIST_SHARE,
                "any_starmie_heavy_share": DEFAULT_HEAVY_SHARE,
                "mixed_minimum_deck_share": DEFAULT_MIXED_MIN_DECK_SHARE,
                "mixed_minimum_decks": DEFAULT_MIXED_MIN_DECKS,
            },
            "interpretation": (
                "corpus-mixture classification only; it does not establish out-of-distribution value generalization"
            ),
        },
        "totals": {
            "eligible_rows": total_rows,
            "complete_games": len(game_blocks),
            "games_with_eligible_rows": len(eligible_games),
            "blocks": len(blocks),
            "blocks_with_eligible_rows": len(eligible_blocks),
        },
        "identity_coverage": {
            "acting_deck_rows_attributed": attributed_rows,
            "acting_deck_row_coverage": round(identity_coverage, 9),
            "opponent_deck_rows_unattributed_after_paired_seat_recovery": counters["matchup"][
                ("unattributed", "unattributed")
            ] + sum(
                count
                for (acting, opponent), count in counters["matchup"].items()
                if acting != "unattributed" and opponent == "unattributed"
            ),
            "formal_semantic_archetype_rows": semantic_rows,
            "formal_semantic_archetype_row_coverage": round(semantic_rows / total_rows, 9),
            "confident_semantic_archetype_rows": confident_semantic_rows,
            "confident_semantic_archetype_row_coverage": round(
                confident_semantic_rows / total_rows, 9
            ),
            "opponent_deck_rows_recorded": counters["opponent_deck_source"]["recorded"],
            "opponent_deck_rows_recovered_from_paired_actor_block": counters[
                "opponent_deck_source"
            ]["paired_actor_block"],
            "strength_status": "not_instrumented unless a numeric source_rating is present",
        },
        "starmie": {
            "exact_submitted_acting_rows": exact_rows,
            "exact_submitted_acting_row_share": round(exact_rows / total_rows, 9),
            "exact_submitted_acting_games": exact_games,
            "any_supported_variant_acting_rows": any_starmie_rows,
            "any_supported_variant_acting_row_share": round(any_starmie_rows / total_rows, 9),
            "games_containing_supported_starmie": len(starmie_games),
            "game_share": round(len(starmie_games) / len(eligible_games), 9),
            "games_with_starmie_acting": len(starmie_acting_games),
            "starmie_actor_rows_in_starmie_games": starmie_actor_rows,
            "non_starmie_actor_rows_in_starmie_games": non_starmie_actor_rows,
            "non_starmie_actor_share_within_starmie_games": round(
                non_starmie_actor_rows / starmie_game_rows, 9
            ) if starmie_game_rows else 0.0,
            "known_controlled_opponent_seat_rows": 0,
            "known_controlled_opponent_seat_share": 0.0,
            "controlled_source_rows_in_starmie_games": controlled_starmie_game_rows,
            "public_replay_rows_without_controlled_seat_reference": public_replay_starmie_game_rows,
            "opponent_seat_status": (
                "not_observed_in_controlled_sources; not_applicable_to_public_replays"
            ),
            "both_sides_starmie_bearing_games": both_sides_starmie_games,
            "exact_variant_mirror_games": exact_variant_mirror_games,
            "supported_hash_provenance": list(starmie_hash_provenance),
        },
        "by_source": by_source,
        "by_acting_deck": acting_decks,
        "by_matchup": matchups,
        "by_policy_id": _ranked_rows(
            counters["policy_id"], total=total_rows, key_names=("policy_id",)
        ),
        "by_policy_family": _ranked_rows(
            counters["policy_family"], total=total_rows, key_names=("policy_family",)
        ),
        "by_acting_seat": _ranked_rows(
            counters["acting_seat"], total=total_rows, key_names=("acting_seat",)
        ),
        "by_controlled_agent_seat": _ranked_rows(
            counters["controlled_agent_seat"],
            total=total_rows,
            key_names=("controlled_agent_seat",),
        ),
        "by_generation": _ranked_rows(
            counters["generation"], total=total_rows, key_names=("generation",)
        ),
        "by_checkpoint": _ranked_rows(
            counters["checkpoint"], total=total_rows, key_names=("checkpoint",)
        ),
        "by_empirical_strength_bucket": _ranked_rows(
            counters["strength_bucket"], total=total_rows, key_names=("strength_bucket",)
        ),
        "by_opponent_difficulty_bucket": _ranked_rows(
            counters["opponent_difficulty_bucket"],
            total=total_rows,
            key_names=("opponent_difficulty_bucket",),
        ),
        "by_information_mode": _ranked_rows(
            counters["information_mode"], total=total_rows, key_names=("information_mode",)
        ),
        "by_opponent_deck_source": _ranked_rows(
            counters["opponent_deck_source"],
            total=total_rows,
            key_names=("opponent_deck_source",),
        ),
    }


def render_report(result: Mapping[str, Any]) -> str:
    audit = result["audit"]
    starmie = audit["starmie"]
    totals = audit["totals"]
    identity = audit["identity_coverage"]
    lines = [
        "# Frozen V0 corpus attribution",
        "",
        f"Verdict: **{audit['classification']['verdict']}**.",
        "",
        "This receipt is metadata-only. It did not load target labels, predictions, feature rows, or sealed-test outcomes, and it did not retrain V0.",
        "",
        "## Coverage",
        "",
        f"- Eligible rows: {totals['eligible_rows']:,} across {totals['games_with_eligible_rows']:,} games with at least one eligible target ({totals['complete_games']:,} complete games total).",
        f"- Acting-deck identity coverage: {identity['acting_deck_row_coverage']:.3%}; opponent deck rows unattributed after paired-seat recovery: {identity['opponent_deck_rows_unattributed_after_paired_seat_recovery']:,}.",
        f"- Formal semantic-archetype coverage: {identity['formal_semantic_archetype_row_coverage']:.3%} ({identity['confident_semantic_archetype_row_coverage']:.3%} confident); missing semantic labels remain explicit rather than inferred from names.",
        f"- Opponent deck identity was recorded for {identity['opponent_deck_rows_recorded']:,} rows and recovered from the other public actor block for {identity['opponent_deck_rows_recovered_from_paired_actor_block']:,}; unresolved after recovery: {identity['opponent_deck_rows_unattributed_after_paired_seat_recovery']:,}.",
        f"- Information mode: `{audit['by_information_mode'][0]['information_mode']}` for {audit['by_information_mode'][0]['eligible_row_share']:.3%} of eligible rows.",
        "",
        "## Starmie attribution",
        "",
        f"- Exact submitted Starmie acting rows: {starmie['exact_submitted_acting_rows']:,} ({starmie['exact_submitted_acting_row_share']:.3%}) across {starmie['exact_submitted_acting_games']:,} games.",
        f"- Any formally labeled Starmie-bearing variant acting rows: {starmie['any_supported_variant_acting_rows']:,} ({starmie['any_supported_variant_acting_row_share']:.3%}).",
        f"- Games containing a formally labeled Starmie-bearing variant: {starmie['games_containing_supported_starmie']:,} ({starmie['game_share']:.3%}); Starmie is the acting deck in {starmie['games_with_starmie_acting']:,} of them.",
        f"- Known controlled-opponent-seat rows: {starmie['known_controlled_opponent_seat_rows']:,} ({starmie['known_controlled_opponent_seat_share']:.3%}); controlled producers do not serialize opponent turns, and public replays have no controlled-seat reference.",
        f"- Non-Starmie acting rows inside Starmie-containing games: {starmie['non_starmie_actor_rows_in_starmie_games']:,} ({starmie['non_starmie_actor_share_within_starmie_games']:.3%} of eligible rows in those games); this is a deck-role slice, not an opponent-seat estimate.",
        f"- Exact-variant Starmie mirrors: {starmie['exact_variant_mirror_games']:,}; both sides carry a Starmie-bearing list in {starmie['both_sides_starmie_bearing_games']:,} games.",
        "",
        "The corpus is not Starmie-heavy, although Starmie games are strongly biased toward the Starmie actor because the inspected producer contract records only the controlled actor's turns for controlled refgame sources.",
        "",
        "## Top acting decks",
        "",
        "| Deck hash | Deck ID | Rows | Share | Games |",
        "|---|---|---:|---:|---:|",
    ]
    for row in audit["by_acting_deck"][:12]:
        lines.append(
            f"| `{row['deck_hash']}` | {row['deck_id'] or 'unattributed'} | "
            f"{row['eligible_rows']:,} | {row['eligible_row_share']:.3%} | {row['eligible_games']:,} |"
        )
    lines.extend([
        "",
        "## Top matchup pairs",
        "",
        "| Acting deck | Opponent deck | Rows | Share | Games |",
        "|---|---|---:|---:|---:|",
    ])
    for row in audit["by_matchup"][:12]:
        lines.append(
            f"| `{row['acting_deck_hash']}` | `{row['opponent_deck_hash']}` | "
            f"{row['eligible_rows']:,} | {row['eligible_row_share']:.3%} | {row['eligible_games']:,} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        f"`{audit['classification']['verdict']}` describes the observed training mixture, not proven universal competence. V0 remains frozen. The next dependency is a versioned strategic ontology/classifier; no exact deck ID is authorized as a predictive feature.",
        "",
    ])

    return "\n".join(lines)


def run_v0_corpus_attribution(
    *,
    cache_dir: str | Path,
    canonical_root: str | Path,
    archetype_labels_path: str | Path,
    deck_pool_path: str | Path,
    exact_starmie_deck_path: str | Path,
    additional_starmie_deck_paths: Iterable[str | Path],
    output_dir: str | Path,
    uncontrolled_sources: Iterable[str] = ("kaggle_ingest",),
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    canonical_root = Path(canonical_root)
    cache_manifest_path = cache_dir / "cache_manifest.json"
    cache_manifest = _read_json(cache_manifest_path)
    if (
        cache_manifest.get("schema_version") != "deepmew.value_lab_cache.v2"
        or cache_manifest.get("status") != "completed"
    ):
        raise ValueError("V0 attribution requires the accepted completed cache v2")
    if Path(cache_manifest.get("canonical_root", "")).resolve() != canonical_root.resolve():
        raise ValueError("canonical root does not match the cache binding")

    exact_cards = _read_deck(exact_starmie_deck_path)
    exact_hash = normalized_deck_hash(exact_cards)
    explicit_paths = [Path(exact_starmie_deck_path), *map(Path, additional_starmie_deck_paths)]
    explicit_hashes = {normalized_deck_hash(_read_deck(path)) for path in explicit_paths}
    labels = _load_archetype_labels(archetype_labels_path)
    pool_starmie_hashes = _load_pool_starmie_hashes(deck_pool_path)
    blocks = _build_blocks(
        cache_dir=cache_dir,
        canonical_root=canonical_root,
        cache_manifest=cache_manifest,
    )
    observed_hashes = {
        deck_hash
        for block in blocks
        for deck_hash in (block.deck_hash, block.opponent_deck_hash)
        if deck_hash is not None
    }
    starmie_hashes, starmie_hash_provenance = resolve_starmie_hashes(
        observed_hashes=observed_hashes,
        archetype_labels=labels,
        explicit_hashes=explicit_hashes,
        pool_starmie_hashes=pool_starmie_hashes,
    )
    audit = summarize_blocks(
        blocks,
        exact_starmie_hash=exact_hash,
        starmie_hashes=starmie_hashes,
        starmie_hash_provenance=starmie_hash_provenance,
        archetype_labels=labels,
        uncontrolled_sources=set(uncontrolled_sources),
    )
    expected_totals = {
        "eligible_rows": int(cache_manifest["eligible_target_rows"]),
        "complete_games": int(cache_manifest["games"]),
        "blocks": int(cache_manifest["blocks"]),
    }
    observed_totals = {key: audit["totals"][key] for key in expected_totals}
    if observed_totals != expected_totals:
        raise ValueError(
            f"attribution totals do not match cache manifest: "
            f"observed={observed_totals}, expected={expected_totals}"
        )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": created_at_utc,
        "information_mode": "actor_visible_metadata_only",
        "protected_split_use": {
            "target_labels_loaded": False,
            "predictions_loaded": False,
            "feature_rows_loaded": False,
            "sealed_test_outcomes_opened": False,
            "eligibility_metadata_counted_across_all_folds": True,
        },
        "input_identity": {
            "cache_dir": str(cache_dir),
            "cache_manifest_sha256": _sha256(cache_manifest_path),
            "catalog_sha256": cache_manifest.get("catalog_sha256"),
            "split_manifest_sha256": cache_manifest.get("split_manifest_sha256"),
            "row_exclusion_manifest_sha256": cache_manifest.get("row_exclusion_manifest_sha256"),
            "input_bindings_sha256": cache_manifest.get("input_bindings_sha256"),
            "archetype_labels_path": str(Path(archetype_labels_path)),
            "archetype_labels_sha256": _sha256(archetype_labels_path),
            "deck_pool_path": str(Path(deck_pool_path)),
            "deck_pool_sha256": _sha256(deck_pool_path),
            "exact_starmie_deck": {
                "path": str(Path(exact_starmie_deck_path)),
                "file_sha256": _sha256(exact_starmie_deck_path),
                "normalized_deck_sha256": exact_hash,
            },
            "additional_starmie_decks": [
                {
                    "path": str(path),
                    "file_sha256": _sha256(path),
                    "normalized_deck_sha256": normalized_deck_hash(_read_deck(path)),
                }
                for path in explicit_paths[1:]
            ],
            "supported_starmie_hash_count": len(starmie_hashes),
            "implementation_sha256": _sha256(Path(__file__)),
            "entrypoint_sha256": _sha256(
                Path(__file__).resolve().parents[2] / "scripts" / "run_v0_corpus_attribution.py"
            ),
        },
        "artifact_identity": {
            "ontology_version": "not_defined_at_attribution_gate",
            "classifier_hash": "not_applicable_at_attribution_gate",
            "deck_hash_role": "provenance_and_slice_only",
            "policy_family": "stratified_from_recorded_policy_id",
            "generation": "stratified_by_source_and_unit",
            "empirical_strength": "numeric_source_rating_only_else_not_instrumented",
            "league_snapshot": "mixed_historical_corpus_not_one_snapshot",
            "calibration": "not_applicable_metadata_receipt",
            "promotion_status": "diagnostic_receipt_no_control_authority",
            "known_limitations": [
                "formal strategic semantics cover only confident K2A labels and explicit Starmie lists",
                "controlled refgame sources record controlled-actor turns rather than both actors",
                "source bucket is opponent difficulty and is not treated as actor strength",
                "corpus-mixture classification does not prove unseen-strategy generalization",
            ],
        },
        "audit": audit,
        "protected_surfaces_changed": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    report_path = output_dir / "REPORT.md"
    write_json(result_path, result)
    write_text(report_path, render_report(result))
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "run_id": "v0-corpus-attribution-20260711",
        "created_at_utc": created_at_utc,
        "kind": "value_corpus_attribution",
        "status": "completed",
        "inputs": result["input_identity"],
        "outputs": {
            "result": {
                "path": str(result_path),
                "bytes": result_path.stat().st_size,
                "sha256": _sha256(result_path),
            },
            "report": {
                "path": str(report_path),
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            },
        },
        "summary": {
            "verdict": audit["classification"]["verdict"],
            "eligible_rows": audit["totals"]["eligible_rows"],
            "eligible_games": audit["totals"]["games_with_eligible_rows"],
            "exact_starmie_row_share": audit["starmie"]["exact_submitted_acting_row_share"],
            "any_starmie_row_share": audit["starmie"]["any_supported_variant_acting_row_share"],
            "sealed_test_outcomes_opened": False,
        },
        "protected_surfaces_changed": [],
    }
    write_json(output_dir / "manifest.json", manifest)

    return result
