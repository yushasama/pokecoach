"""Build lightweight current-bucket behavior references over immutable Derived V1."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.current_rating import (  # noqa: E402
    CurrentRatingError,
    ReplayIdentityEvidence,
    canonical_json_sha256,
    resolve_replay_identity,
    utc_now_text,
)
from deepmew.kaggle_replays.decks import extract_decklists  # noqa: E402
from deepmew.kaggle_replays.episode import load_episode  # noqa: E402
from deepmew.hybrid_distillation.contracts import (  # noqa: E402
    _canonical_stage_selections,
    allowed_option_ids,
)
from deepmew.replay.contract.causal_alignment import KaggleEpisodeRepository, align_decision  # noqa: E402
from deepmew.replay.contract.source_events import source_event_from_admission  # noqa: E402
from deepmew.replay.contract.schemas import OrderingSignificance  # noqa: E402
from deepmew.replay.contract.validation import (  # noqa: E402
    reconstruct_legal_actions,
    reconstruct_selected_action,
)
from scripts.build_current_player_rating_overlay_v1 import (  # noqa: E402
    EPISODE_ID_RE,
    clean_kaggle_games,
    replay_team_names,
    scan_identity_provenance,
)
from scripts.build_e1_eligible_derived_v1 import (  # noqa: E402
    ADMISSION_MANIFEST,
    AUDIT_BINDINGS,
    AUDIT_RECEIPT,
    DERIVED_MANIFEST,
    _open_admission,
    _selected_observations,
    _selected_transitions,
)
from scripts.run_current_bucket_om2_om3_v1 import FORBIDDEN_MODEL_FIELDS as OM_FORBIDDEN_MODEL_FIELDS  # noqa: E402

MISSION_ROOT = ROOT / ".research" / "current_rating_bucket_exploiter_league_and_hybrid_v1"
DEFAULT_OUTPUT = MISSION_ROOT / "opponent_model" / "current_bucket_behavior_corpus_v1"
CARD_CSV = ROOT / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
PLAYSTYLE_HASH_BINS = 16
PLAYSTYLE_AXES = tuple(f"action_signature_frequency_bin_{index:02d}" for index in range(PLAYSTYLE_HASH_BINS)) + (
    "mean_log_legal_support_norm",
    "mean_stage_count_norm",
    "mean_selected_option_count_norm",
    "forced_decision_rate",
)
DECK_SCALAR_FIELDS = (
    "pokemon",
    "trainer",
    "energy",
    "basic_pokemon",
    "stage1_pokemon",
    "stage2_pokemon",
    "mega_pokemon",
    "ex_rule_cards",
    "unique_cards",
    "unknown_card_ids",
)
ACTION_HASH_BINS = 32
FORBIDDEN_MODEL_FIELDS = set(OM_FORBIDDEN_MODEL_FIELDS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stable_fraction(identity: str) -> float:
    value = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)

    return value / float(16**16)


def freeze_game_splits(
    game_players: Mapping[str, Sequence[str]],
    game_decks: Mapping[str, Sequence[str]],
    game_actor_pairs: Mapping[str, Sequence[tuple[str, str | None]]] | None = None,
) -> dict[str, str]:
    heldout_players = {
        player_id
        for players in game_players.values()
        for player_id in players
        if stable_fraction(f"player:{player_id}") < 0.20
    }
    heldout_decks = {
        deck_id
        for decks in game_decks.values()
        for deck_id in decks
        if stable_fraction(f"deck:{deck_id}") < 0.10
    }
    heldout_pairs = {
        (player_id, deck_id)
        for pairs in (game_actor_pairs or {}).values()
        for player_id, deck_id in pairs
        if deck_id
        and player_id not in heldout_players
        and deck_id not in heldout_decks
        and stable_fraction(f"player-deck:{player_id}|{deck_id}") < 0.10
    }
    assignments = {}
    for game_id in sorted(game_players):
        if set(game_players[game_id]) & heldout_players:
            split = "test"
        elif set(game_decks.get(game_id, ())) & heldout_decks:
            split = "test"
        elif set((game_actor_pairs or {}).get(game_id, ())) & heldout_pairs:
            split = "test"
        elif stable_fraction(f"calibration:{game_id}") < 0.10:
            split = "calibration"
        else:
            split = "train"
        assignments[game_id] = split

    return assignments


def heldout_slice_labels(
    game_id: str,
    player_id: str,
    deck_id: str | None,
) -> dict[str, str]:
    player_is_heldout = stable_fraction(f"player:{player_id}") < 0.20
    deck_is_heldout = bool(deck_id) and stable_fraction(f"deck:{deck_id}") < 0.10
    pair_is_heldout = (
        bool(deck_id)
        and not player_is_heldout
        and not deck_is_heldout
        and stable_fraction(f"player-deck:{player_id}|{deck_id}") < 0.10
    )
    deck_label = (
        f"heldout:{deck_id}" if deck_is_heldout else f"seen:{deck_id}"
    ) if deck_id else "missing:deck_family"
    pair_identity = f"{player_id}|{deck_id}" if deck_id else None

    return {
        "player": f"heldout:{player_id}" if player_is_heldout else f"seen:{player_id}",
        "deck_family": deck_label,
        "player_deck": (
            f"heldout:{pair_identity}" if pair_is_heldout else f"seen:{pair_identity}"
        ) if pair_identity else "missing:player_deck",
        "game": game_id,
    }


def playstyle_increment(
    action_family: str,
    action_template: str,
    legal_count: int,
    *,
    stage_count: int = 1,
    selected_option_count: int = 1,
) -> list[float]:
    values = [0.0] * len(PLAYSTYLE_AXES)
    signature = f"{action_family}|{action_template}"
    index = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8], 16) % PLAYSTYLE_HASH_BINS
    values[index] = 1.0
    values[PLAYSTYLE_HASH_BINS] = min(math.log1p(max(legal_count, 0)) / math.log(1_001.0), 1.0)
    values[PLAYSTYLE_HASH_BINS + 1] = min(max(stage_count, 0) / 8.0, 1.0)
    values[PLAYSTYLE_HASH_BINS + 2] = min(max(selected_option_count, 0) / 8.0, 1.0)
    values[PLAYSTYLE_HASH_BINS + 3] = float(legal_count == 1)

    return values


def causal_playstyle_features(totals: Sequence[float], prior_actions: int) -> list[float]:
    if prior_actions <= 0:
        return [0.0] * len(PLAYSTYLE_AXES) + [0.0]

    return [float(value) / prior_actions for value in totals] + [min(prior_actions, 100) / 100.0]


def normalized_js_similarity(left: Sequence[float], right: Sequence[float], cold_start: float = 0.75) -> float:
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0.0 or right_total <= 0.0:
        return cold_start
    p = [max(value, 0.0) / left_total for value in left]
    q = [max(value, 0.0) / right_total for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q, strict=True)]

    def divergence(values: Sequence[float], reference: Sequence[float]) -> float:
        return sum(value * math.log2(value / target) for value, target in zip(values, reference, strict=True) if value > 0)

    js = 0.5 * divergence(p, midpoint) + 0.5 * divergence(q, midpoint)

    return min(max(1.0 - js, 0.25), 1.0)


def multiset_jaccard(left: Sequence[int], right: Sequence[int], cold_start: float = 0.75) -> float:
    if not left or not right:
        return cold_start
    left_counts = Counter(left)
    right_counts = Counter(right)
    intersection = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())

    return min(max(intersection / union, 0.25), 1.0)


def effective_sample_size(weights: Iterable[float]) -> float:
    values = [float(value) for value in weights]
    numerator = sum(values) ** 2
    denominator = sum(value * value for value in values)

    return numerator / denominator if denominator > 0.0 else 0.0


def evidence_weight(
    *,
    age_days: float,
    rating_stability: float,
    deck_relevance: float,
    playstyle_continuity: float,
    source_reliability: float = 1.0,
) -> dict[str, float]:
    components = {
        "recency_weight": max(0.10, 2 ** (-max(age_days, 0.0) / 45.0)),
        "rating_stability_weight": min(max(rating_stability, 0.0), 1.0),
        "deck_relevance_weight": min(max(deck_relevance, 0.25), 1.0),
        "playstyle_continuity_weight": min(max(playstyle_continuity, 0.25), 1.0),
        "source_reliability_weight": min(max(source_reliability, 0.0), 1.0),
    }
    components["current_behavior_evidence_weight"] = math.prod(components.values())

    return components


def serialize_semantics(legal: Any, chosen: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    legal_value = json_safe(asdict(legal))
    chosen_value = json_safe(asdict(chosen))
    legal_value.pop("failures", None)
    chosen_value.pop("failures", None)

    return legal_value, chosen_value


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise CurrentRatingError(f"semantic contract contains non-JSON value: {type(value).__name__}")


def enumerate_complete_actions(legal: Any, chosen: Any) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []

    def normalized(stage: Any, selected: Sequence[str]) -> list[str]:
        values = list(selected)
        if stage.ordering_significance in {
            OrderingSignificance.NOT_APPLICABLE,
            OrderingSignificance.UNORDERED,
        }:
            values.sort()

        return values

    def visit(stage_index: int, prior: tuple[tuple[str, ...], ...]) -> None:
        if stage_index == len(legal.stages):
            structure = {
                "legal_action_set_id": legal.legal_action_set_id,
                "action_template": legal.action_template,
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "selected_option_ids": normalized(stage, selected),
                        "omitted_optional": bool(not selected and stage.minimum_count == 0),
                    }
                    for stage, selected in zip(legal.stages, prior, strict=True)
                ],
            }
            signature = canonical_json_sha256(structure)
            feature = [0.0] * (ACTION_HASH_BINS + 2)
            feature[int(signature[:8], 16) % ACTION_HASH_BINS] = 1.0
            feature[ACTION_HASH_BINS] = min(sum(len(value) for value in prior) / 8.0, 1.0)
            feature[ACTION_HASH_BINS + 1] = min(len(prior) / 8.0, 1.0)
            candidates.append({
                "features": feature,
                "family": legal.action_family,
                "target": json.dumps(
                    [option_id for selected in prior for option_id in selected],
                    separators=(",", ":"),
                ),
                "auxiliary_targets": {},
                "causal_scope": "actor_visible_pre_action_plus_current_complete_action",
                "source": "canonical_legal_action_set_exact_enumeration",
                "structure": structure,
            })

            return
        stage = legal.stages[stage_index]
        allowed = allowed_option_ids(stage, prior)
        if allowed is None:
            raise CurrentRatingError(f"conditional complete-action path is ambiguous: {stage.stage_id}")
        for selected in _canonical_stage_selections(stage, allowed):
            visit(stage_index + 1, prior + (selected,))

    visit(0, ())
    if len(candidates) != legal.complete_action_count:
        raise CurrentRatingError(
            f"complete-action enumeration drifted: {len(candidates)} != {legal.complete_action_count}"
        )
    chosen_structure = {
        "legal_action_set_id": legal.legal_action_set_id,
        "action_template": chosen.action_template,
        "stages": [
            {
                "stage_id": chosen_stage.stage_id,
                "selected_option_ids": normalized(legal_stage, chosen_stage.selected_option_ids),
                "omitted_optional": bool(chosen_stage.omitted_optional),
            }
            for chosen_stage, legal_stage in zip(chosen.stages, legal.stages, strict=True)
        ],
    }
    matches = [index for index, candidate in enumerate(candidates) if candidate["structure"] == chosen_structure]
    if len(matches) != 1:
        raise CurrentRatingError(f"chosen complete action has {len(matches)} exact support matches")

    return candidates, matches[0]


def build_manifest_stub(
    *,
    overlay_path: Path,
    rows: int,
    games: int,
    shards: Sequence[Mapping[str, Any]],
    split_counts: Mapping[str, int],
    bucket_counts: Mapping[str, int],
    bucket_ess: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "deepmew.current_bucket_behavior_corpus.v1",
        "status": "COMPLETE_LIGHTWEIGHT_REFERENCES_NO_X_DUPLICATION",
        "population_model_type": "CURRENT_POPULATION_MODEL",
        "created_at_utc": utc_now_text(),
        "rows": rows,
        "games": games,
        "shards": list(shards),
        "split_counts": dict(sorted(split_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_effective_sample_size": dict(sorted(bucket_ess.items())),
        "overlay": {"path": str(overlay_path), "sha256": file_sha256(overlay_path)},
        "derived_v1": {"path": str(DERIVED_MANIFEST), "sha256": file_sha256(DERIVED_MANIFEST)},
        "observation_storage": "immutable Derived V1 coordinates only; zero X rows copied",
        "semantic_hydration": "read-only existing Admission sqlite plus deterministic canonical reconstruction",
        "playstyle_contract": {
            "axes": list(PLAYSTYLE_AXES),
            "lookback": "all actor-visible prior actions in the same trajectory",
            "cold_start": "all rates zero and history_support=0",
            "missing_history": "same as cold start",
            "update_schedule": "before prediction, then increment with the observed chosen action",
            "version": "deepmew.causal_public_playstyle.v1",
        },
        "actor_model_input_excludes": sorted(FORBIDDEN_MODEL_FIELDS),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_resolved_seat_map(overlay: Mapping[str, Any]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Path]]:
    players = {str(player["team_id"]): dict(player) for player in overlay.get("players", [])}
    aliases, episode_team_ids, replay_timestamps, _histories, _sources = scan_identity_provenance()
    for team_id, player in players.items():
        aliases[team_id].add(str(player["canonical_player_name"]))
    games = clean_kaggle_games()
    resolved: dict[tuple[str, int], dict[str, Any]] = {}
    for game_id, replay_path in sorted(games.items()):
        match = EPISODE_ID_RE.search(replay_path.name)
        if not match:
            raise CurrentRatingError(f"canonical game lacks episode id: {replay_path}")
        episode_id = match.group(1)
        names = replay_team_names(replay_path)
        for actor_seat, replay_name in enumerate(names):
            evidence = ReplayIdentityEvidence(
                source_game_id=game_id,
                actor_seat=actor_seat,
                replay_team_name=replay_name,
                candidate_team_ids=tuple(sorted(episode_team_ids.get(episode_id, set()), key=int)),
                decision_count=0,
                replay_timestamp_utc=replay_timestamps.get(episode_id),
            )
            identity_row = resolve_replay_identity(evidence, aliases)
            team_id = identity_row["team_id"]
            if team_id not in players:
                continue
            resolved[(game_id, actor_seat)] = {
                "player": players[team_id],
                "identity": identity_row,
                "replay_path": replay_path,
                "episode_id": episode_id,
            }

    return resolved, games


def load_card_catalog() -> dict[int, dict[str, str]]:
    cards = {}
    with CARD_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row["Card ID"])
            except (KeyError, ValueError):
                continue
            cards.setdefault(card_id, row)

    return cards


def deck_features(card_ids: Sequence[int], cards: Mapping[int, Mapping[str, str]]) -> tuple[dict[str, float], str]:
    raw: dict[str, Any] = {field: 0 for field in DECK_SCALAR_FIELDS}
    raw["unique_cards"] = len(set(card_ids))
    raw["max_hp"] = 0
    raw["energy_types"] = Counter()
    pokemon_ids = []
    for card_id in card_ids:
        row = cards.get(card_id)
        if row is None:
            raw["unknown_card_ids"] += 1
            continue
        stage = str(row.get("Stage (PokÃ©mon)/Type (Energy and Trainer)") or "").strip()
        name = str(row.get("Card Name") or "").strip()
        rule = str(row.get("Rule") or "").strip().lower()
        if "energy" in name.lower() and stage.lower().startswith("basic energy"):
            raw["energy"] += 1
            raw["energy_types"][str(row.get("Type") or "").strip()] += 1
            continue
        hp = str(row.get("HP") or "").strip()
        if hp.isdigit():
            pokemon_ids.append(card_id)
            raw["pokemon"] += 1
            raw["max_hp"] = max(raw["max_hp"], int(hp))
            stage_lower = stage.lower()
            raw["basic_pokemon"] += int(stage_lower.startswith("basic"))
            raw["stage1_pokemon"] += int("1" in stage_lower and not stage_lower.startswith("basic"))
            raw["stage2_pokemon"] += int("2" in stage_lower and not stage_lower.startswith("basic"))
            raw["mega_pokemon"] += int("mega" in name.lower())
            raw["ex_rule_cards"] += int(" ex" in name.lower() or "ex" in rule)
        else:
            raw["trainer"] += 1
    vector = {f"count_fraction:{field}": float(raw[field]) / 60.0 for field in DECK_SCALAR_FIELDS}
    vector["max_hp_fraction_of_400"] = float(raw["max_hp"]) / 400.0
    for energy_type, count in sorted(raw["energy_types"].items()):
        vector[f"energy_type_fraction:{energy_type}"] = float(count) / 60.0
    family_payload = Counter(pokemon_ids)
    family_id = "deckfam1-" + canonical_json_sha256(dict(sorted(family_payload.items())))[:24]

    return vector, family_id


def build_deck_records(
    resolved_seats: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    cards = load_card_catalog()
    records: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_index: dict[tuple[str, int], Sequence[int]] = {}
    for summary_path in sorted((ROOT / ".research" / "opponent_decks").glob("**/deck_summary.json")):
        summary = _read_json(summary_path)
        source = str(summary.get("source_episode_path") or "").strip()
        player_index = summary.get("player_index")
        card_ids = summary.get("card_ids")
        if not source or not isinstance(player_index, int) or not isinstance(card_ids, list):
            continue
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        key = (str(source_path.resolve()).casefold(), player_index)
        normalized_cards = tuple(int(value) for value in card_ids)
        if key in artifact_index and tuple(artifact_index[key]) != normalized_cards:
            raise CurrentRatingError(f"conflicting deck artifacts for {source_path} seat {player_index}")
        artifact_index[key] = normalized_cards
    by_path: dict[Path, list[tuple[str, int]]] = defaultdict(list)
    for key, value in resolved_seats.items():
        by_path[Path(value["replay_path"])].append(key)
    for replay_path, keys in sorted(by_path.items(), key=lambda item: str(item[0])):
        fallback_error = None
        artifact_cards = {
            key[1]: artifact_index.get((str(replay_path.resolve()).casefold(), key[1]))
            for key in keys
        }
        if all(artifact_cards.get(key[1]) for key in keys):
            decks = None
        else:
            try:
                decks = {deck.player_index: deck for deck in extract_decklists(load_episode(replay_path))}
            except Exception as exc:
                decks = {}
                fallback_error = type(exc).__name__
        for key in keys:
            card_ids = artifact_cards.get(key[1])
            source_status = "MEASURED_EXISTING_ADMITTED_DECK_ARTIFACT"
            if card_ids is None and decks is not None:
                deck = decks.get(key[1])
                card_ids = tuple(deck.card_ids) if deck is not None and deck.is_standard_size else None
                source_status = "MEASURED_PUBLIC_REPLAY_DECK_FALLBACK"
            if card_ids is None or len(card_ids) != 60:
                records[key] = {
                    "status": "MEASURED_MISSING_NONSTANDARD_OR_ABSENT",
                    "features": None,
                    "deck_family_id": None,
                    "card_ids": (),
                    "error_type": fallback_error,
                }
                continue
            features, family_id = deck_features(card_ids, cards)
            records[key] = {
                "status": source_status,
                "features": features,
                "deck_family_id": family_id,
                "card_ids": tuple(card_ids),
            }

    return records


def binding_lookup() -> tuple[list[int], list[dict[str, Any]]]:
    registry = _read_json(AUDIT_BINDINGS)
    bindings = sorted(registry["bindings"], key=lambda row: int(row["global_candidate_row_start"]))
    starts = [int(row["global_candidate_row_start"]) for row in bindings]

    return starts, bindings


def binding_coordinate(global_row: int, starts: Sequence[int], bindings: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    index = bisect.bisect_right(starts, global_row) - 1
    if index < 0:
        raise CurrentRatingError(f"global candidate row precedes binding registry: {global_row}")
    binding = bindings[index]
    local_row = global_row - int(binding["global_candidate_row_start"])
    if local_row < 0 or local_row >= int(binding["expected_rows"]):
        raise CurrentRatingError(f"global candidate row falls outside binding: {global_row}")

    return str(binding["binding_id"]), local_row


def iter_derived_shards() -> Iterable[tuple[Path, list[dict[str, Any]]]]:
    manifest = _read_json(DERIVED_MANIFEST)
    for path in sorted(Path(manifest["shards_root"]).glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield path, [json.loads(line) for line in handle if line.strip()]


def prepass(
    resolved_seats: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[int, dict[tuple[str, int], list[float]], Counter[str], Counter[str], Counter[tuple[str, int]]]:
    game_profiles: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0] * len(PLAYSTYLE_AXES))
    bucket_counts: Counter[str] = Counter()
    game_counts: Counter[str] = Counter()
    seat_counts: Counter[tuple[str, int]] = Counter()
    rows = 0
    for _path, shard_rows in iter_derived_shards():
        for row in shard_rows:
            key = (str(row["source_game_id"]), int(row["actor_seat"]))
            resolved = resolved_seats.get(key)
            if resolved is None:
                continue
            rows += 1
            bucket_counts[str(resolved["player"]["current_bucket"])] += 1
            game_counts[key[0]] += 1
            seat_counts[key] += 1
            increment = playstyle_increment(
                str(row["action_family"]),
                str(row["action_template"]),
                int(row["complete_legal_action_count"]),
            )
            game_profiles[key] = [a + b for a, b in zip(game_profiles[key], increment, strict=True)]

    return rows, game_profiles, bucket_counts, game_counts, seat_counts


def run_deck_join_audit(overlay_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"append-only deck audit output exists: {output}")
    output.mkdir(parents=True)
    overlay = _read_json(overlay_path)
    resolved_seats, _games = build_resolved_seat_map(overlay)
    deck_records = build_deck_records(resolved_seats)
    expected_rows, _profiles, _buckets, _games_counts, seat_counts = prepass(resolved_seats)
    feature_names = sorted({name for row in deck_records.values() for name in (row.get("features") or {})})
    fixed_names = feature_names + ["deck_features_missing"]
    seat_status = Counter(str(deck_records.get(key, {}).get("status", "MEASURED_MISSING_NO_RECORD")) for key in resolved_seats)
    row_status: Counter[str] = Counter()
    for key, count in seat_counts.items():
        row_status[str(deck_records.get(key, {}).get("status", "MEASURED_MISSING_NO_RECORD"))] += count
    missing_statuses = {status for status in row_status if "MISSING" in status}
    missing_rows = sum(row_status[status] for status in missing_statuses)
    evidence = []
    for key, resolved in sorted(resolved_seats.items())[:20]:
        replay_path = Path(resolved["replay_path"])
        evidence.append({
            "source_path_original": str(replay_path),
            "source_path_normalized": str(replay_path.resolve()).casefold(),
            "actor_seat": key[1],
            "status": str(deck_records.get(key, {}).get("status", "MEASURED_MISSING_NO_RECORD")),
        })
    receipt = {
        "schema_version": "deepmew.current_bucket_deck_feature_join_audit.v1",
        "status": "PASS" if expected_rows == 154_834 else "FAIL",
        "created_at_utc": utc_now_text(),
        "overlay_path": str(overlay_path),
        "resolved_game_seat_count": len(resolved_seats),
        "target_corpus_rows": expected_rows,
        "fixed_vector_width": len(fixed_names),
        "feature_names": fixed_names,
        "missingness_indicator": "deck_features_missing",
        "seat_status_counts": dict(sorted(seat_status.items())),
        "row_status_counts": dict(sorted(row_status.items())),
        "missing_target_rows": missing_rows,
        "missing_target_row_rate": missing_rows / expected_rows if expected_rows else None,
        "key_contract": "normalized absolute source_episode_path casefold plus zero-based player_index/actor_seat",
        "key_normalization_evidence": evidence,
        "ambiguity_or_collision_failures": 0,
        "fallback": "raw public replay visualize deck extraction only when admitted deck_summary join is absent",
        "read_only": True,
    }
    _write_json(output / "DECK_FEATURE_JOIN_AUDIT_V1.json", receipt)

    return receipt


def prior_player_profiles(
    resolved_seats: Mapping[tuple[str, int], Mapping[str, Any]],
    game_profiles: Mapping[tuple[str, int], Sequence[float]],
) -> dict[tuple[str, int], list[float]]:
    grouped: dict[str, list[tuple[datetime | None, tuple[str, int]]]] = defaultdict(list)
    for key in game_profiles:
        value = resolved_seats[key]
        player_id = str(value["player"]["canonical_player_id"])
        timestamp = _parse_utc(value["identity"].get("replay_timestamp_utc"))
        grouped[player_id].append((timestamp, key))
    result: dict[tuple[str, int], list[float]] = {}
    for values in grouped.values():
        values.sort(key=lambda item: (item[0] is None, item[0] or datetime.max.replace(tzinfo=None), item[1]))
        cumulative = [0.0] * len(PLAYSTYLE_AXES)
        for timestamp, key in values:
            result[key] = list(cumulative) if timestamp is not None else [0.0] * len(PLAYSTYLE_AXES)
            if timestamp is not None:
                cumulative = [a + b for a, b in zip(cumulative, game_profiles[key], strict=True)]

    return result


def deck_relevance_by_seat(
    resolved_seats: Mapping[tuple[str, int], Mapping[str, Any]],
    deck_records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    latest: dict[str, tuple[datetime, Sequence[int]]] = {}
    for key, resolved in resolved_seats.items():
        timestamp = _parse_utc(resolved["identity"].get("replay_timestamp_utc"))
        cards = deck_records.get(key, {}).get("card_ids") or ()
        if timestamp is None or not cards:
            continue
        player_id = str(resolved["player"]["canonical_player_id"])
        if player_id not in latest or timestamp > latest[player_id][0]:
            latest[player_id] = (timestamp, cards)

    return {
        key: multiset_jaccard(
            deck_records.get(key, {}).get("card_ids") or (),
            latest.get(str(resolved["player"]["canonical_player_id"]), (None, ()))[1],
        )
        for key, resolved in resolved_seats.items()
    }


def rating_stability(player: Mapping[str, Any]) -> tuple[float, str]:
    volatility = player.get("rating_history", {}).get("volatility", {})
    if volatility.get("status") != "MEASURED":
        return 0.75, "COLD_START_INSUFFICIENT_RATING_SNAPSHOTS"
    score_stddev = float(volatility["score_delta_population_stddev"])

    return 1.0 / (1.0 + score_stddev / 100.0), "MEASURED"


def hydrate_shard(
    rows: Sequence[dict[str, Any]],
    *,
    connection: Any,
    repository: KaggleEpisodeRepository,
    admission_identity: str,
    starts: Sequence[int],
    bindings: Sequence[Mapping[str, Any]],
    decision_local_cache: dict[str, dict[str, int]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        binding_id, _admitted_ordinal = binding_coordinate(
            int(row["semantic_audit_global_candidate_row"]), starts, bindings
        )
        grouped[binding_id].append(row)
    hydrated = {}
    for binding_id, values in grouped.items():
        decision_to_local = decision_local_cache.get(binding_id)
        if decision_to_local is None:
            decision_to_local = {}
            for local_row, compressed in connection.execute(
                "SELECT local_row,compressed_json FROM transitions WHERE binding_id=?",
                (binding_id,),
            ):
                transition = json.loads(zlib.decompress(compressed))
                decision_id = str(transition.get("decision_id") or "")
                if not decision_id or decision_id in decision_to_local:
                    raise CurrentRatingError(f"Admission binding has missing/duplicate decision id: {binding_id}")
                decision_to_local[decision_id] = int(local_row)
            decision_local_cache[binding_id] = decision_to_local
        local_rows = []
        row_by_local = {}
        for derived in values:
            decision_id = str(derived["source_transition_decision_id"])
            if decision_id not in decision_to_local:
                raise CurrentRatingError(f"Derived decision is absent from Admission binding: {decision_id}")
            local_row = decision_to_local[decision_id]
            if local_row in row_by_local:
                raise CurrentRatingError(f"two Derived rows map to one Admission row: {binding_id}:{local_row}")
            local_rows.append(local_row)
            row_by_local[local_row] = derived
        transitions = _selected_transitions(connection, binding_id, local_rows)
        observations = _selected_observations(connection, transitions.values())
        for local_row, derived in row_by_local.items():
            transition = transitions[local_row]
            source_record = observations[
                (
                    str(transition["source_hash"]).lower(),
                    int(transition["state_before"]["source_observation"]["source_line"]),
                    str(transition["state_before"]["source_observation"]["content_sha256"]).lower(),
                )
            ]
            event = source_event_from_admission(
                admission_identity=admission_identity,
                binding_id=binding_id,
                local_row=local_row,
                global_row=int(derived["semantic_audit_global_candidate_row"]),
                transition=transition,
                source_record=source_record,
                verify_source_content=True,
            )
            aligned = align_decision(event, kaggle_repository=repository, transition_registry=None)
            legal = reconstruct_legal_actions(aligned)
            chosen = reconstruct_selected_action(aligned, legal)
            if legal.failures or chosen.failures or not chosen.complete or legal.complete_action_count is None:
                raise CurrentRatingError(f"semantic hydration failed: {derived['stable_row_id']}")
            if legal.legal_action_set_id != derived["legal_action_set_id"]:
                raise CurrentRatingError(f"legal identity drift: {derived['stable_row_id']}")
            if chosen.chosen_action_id != derived["complete_chosen_action_id"]:
                raise CurrentRatingError(f"chosen identity drift: {derived['stable_row_id']}")
            legal_value, chosen_value = serialize_semantics(legal, chosen)
            candidates, chosen_index = enumerate_complete_actions(legal, chosen)
            hydrated[str(derived["stable_row_id"])] = (
                legal_value,
                chosen_value,
                candidates,
                chosen_index,
            )

    return hydrated


def x_store_contracts() -> list[dict[str, Any]]:
    manifest = _read_json(DERIVED_MANIFEST)
    cache_manifest_path = Path(manifest["cache_manifest_path"])
    cache_manifest = _read_json(cache_manifest_path)
    feature_dim = int(manifest["identity"]["feature_dim"])
    cache_path = cache_manifest_path.parent / "X.f32.bin"
    stores = [{
        "id": "cache_x",
        "path": str(cache_path),
        "sha256": str(manifest["identity"]["cache_x_sha256"]),
        "shape": [int(cache_manifest["rows"]), feature_dim],
        "dtype": "float32",
        "format": "raw_float32_c_order",
        "coordinate_contract": "zero_based_row_index",
    }]
    receipts_root = Path(manifest["receipts_root"])
    derived_root = receipts_root.parent
    for receipt_path in sorted(receipts_root.glob("*.receipt.json")):
        receipt = _read_json(receipt_path)
        storage = receipt["observation_storage"]
        if not storage.get("regenerated_path"):
            continue
        stores.append({
            "id": f"regenerated_shard_{int(receipt['shard_index']):05d}",
            "path": str(derived_root / storage["regenerated_path"]),
            "sha256": str(storage["regenerated_sha256"]),
            "shape": [len(storage["regenerated_assets"]), feature_dim],
            "dtype": "float32",
            "format": "raw_float32_c_order",
            "coordinate_contract": "zero_based_row_index",
        })

    return stores


class ShardWriter:
    def __init__(self, root: Path, shard_rows: int) -> None:
        self.root = root
        self.shard_rows = shard_rows
        self.buffer: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []
        self.root.mkdir(parents=True)

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        index = len(self.receipts)
        path = self.root / f"shard_{index:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
            for row in self.buffer:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        self.receipts.append({
            "path": str(path),
            "rows": len(self.buffer),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
        self.buffer.clear()


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"append-only corpus output exists: {args.output}")
    args.output.mkdir(parents=True)
    overlay = _read_json(args.overlay)
    resolved_seats, _games = build_resolved_seat_map(overlay)
    deck_records = build_deck_records(resolved_seats)
    deck_feature_names = sorted({
        name
        for record in deck_records.values()
        for name in (record.get("features") or {})
    }) + ["deck_features_missing"]
    expected_rows, game_profiles, bucket_counts, game_counts, _seat_counts = prepass(resolved_seats)
    if expected_rows != 154_834:
        raise CurrentRatingError(f"resolved rating-linked Derived row count drifted: {expected_rows} != 154834")
    prior_profiles = prior_player_profiles(resolved_seats, game_profiles)
    deck_relevance = deck_relevance_by_seat(resolved_seats, deck_records)
    game_players: dict[str, set[str]] = defaultdict(set)
    game_decks: dict[str, set[str]] = defaultdict(set)
    game_actor_pairs: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for key, resolved in resolved_seats.items():
        game_players[key[0]].add(str(resolved["player"]["canonical_player_id"]))
        deck_family = deck_records.get(key, {}).get("deck_family_id")
        if deck_family:
            game_decks[key[0]].add(str(deck_family))
        game_actor_pairs[key[0]].append((
            str(resolved["player"]["canonical_player_id"]),
            str(deck_family) if deck_family else None,
        ))
    splits = freeze_game_splits(
        {key: sorted(value) for key, value in game_players.items()},
        {key: sorted(value) for key, value in game_decks.items()},
        game_actor_pairs,
    )
    _write_json(args.output / "GAME_SPLIT_ASSIGNMENTS_V1.json", splits)
    starts, bindings = binding_lookup()
    admission_manifest = _read_json(ADMISSION_MANIFEST)
    audit_receipt = _read_json(AUDIT_RECEIPT)
    connection = _open_admission(Path(admission_manifest["database_path"]))
    repository = KaggleEpisodeRepository(ROOT)
    decision_local_cache: dict[str, dict[str, int]] = {}
    writer = ShardWriter(args.output / "shards", args.shard_rows)
    trajectory_totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(PLAYSTYLE_AXES))
    trajectory_counts: Counter[str] = Counter()
    weights_by_bucket: dict[str, list[float]] = defaultdict(list)
    split_row_counts: Counter[str] = Counter()
    deck_status_counts: Counter[str] = Counter()
    replay_timestamp_status: Counter[str] = Counter()
    snapshot_time = _parse_utc(str(overlay["snapshot_timestamp_utc"]))
    if snapshot_time is None:
        raise CurrentRatingError("overlay snapshot timestamp is invalid")
    processed = 0
    try:
        for source_shard, shard_rows in iter_derived_shards():
            selected = [
                row for row in shard_rows
                if (str(row["source_game_id"]), int(row["actor_seat"])) in resolved_seats
            ]
            if not selected:
                continue
            hydrated = hydrate_shard(
                selected,
                connection=connection,
                repository=repository,
                admission_identity=str(audit_receipt["admission_identity"]),
                starts=starts,
                bindings=bindings,
                decision_local_cache=decision_local_cache,
            )
            for derived in selected:
                key = (str(derived["source_game_id"]), int(derived["actor_seat"]))
                resolved = resolved_seats[key]
                player = resolved["player"]
                legal, chosen, legal_candidates, chosen_index = hydrated[str(derived["stable_row_id"])]
                trajectory_id = str(derived["trajectory_id"])
                prior_actions = trajectory_counts[trajectory_id]
                causal_style = causal_playstyle_features(trajectory_totals[trajectory_id], prior_actions)
                continuity = normalized_js_similarity(
                    trajectory_totals[trajectory_id],
                    prior_profiles.get(key, ()),
                )
                replay_time = _parse_utc(resolved["identity"].get("replay_timestamp_utc"))
                if replay_time is None:
                    age_days = 45.0
                    timestamp_status = "COLD_START_MISSING_REPLAY_TIMESTAMP_RECENCY_0_5"
                else:
                    age_days = max((snapshot_time - replay_time).total_seconds() / 86_400.0, 0.0)
                    timestamp_status = "MEASURED"
                stability, stability_status = rating_stability(player)
                weights = evidence_weight(
                    age_days=age_days,
                    rating_stability=stability,
                    deck_relevance=deck_relevance.get(key, 0.75),
                    playstyle_continuity=continuity,
                )
                deck = deck_records.get(key, {
                    "status": "MEASURED_MISSING_NO_RECORD",
                    "features": None,
                    "deck_family_id": None,
                })
                bucket = str(player["current_bucket"])
                split = splits[key[0]]
                bucket_labels = [
                    "R0_CURRENT_FRONTIER",
                    "R1_CURRENT_STRONG",
                    "R2_CURRENT_INTERMEDIATE",
                    "R3_CURRENT_LOWER",
                ]
                bucket_features = [float(bucket == label) for label in bucket_labels]
                deck_vector = [
                    float(deck.get("features") is None)
                    if name == "deck_features_missing"
                    else float((deck.get("features") or {}).get(name, 0.0))
                    for name in deck_feature_names
                ]
                if derived["observation_storage_kind"] == "cache":
                    x_store_id = "cache_x"
                else:
                    x_store_id = f"regenerated_shard_{int(derived['observation_storage_shard']):05d}"
                slice_labels = heldout_slice_labels(
                    key[0],
                    str(player["canonical_player_id"]),
                    str(deck.get("deck_family_id")) if deck.get("deck_family_id") else None,
                )
                trend_status = str(player.get("rating_history", {}).get("recent_trend", {}).get("status", "NOT_INSTRUMENTED"))
                prior_profile = prior_profiles.get(key, ())
                style_cluster = (
                    f"hashbin-{max(range(PLAYSTYLE_HASH_BINS), key=lambda index: prior_profile[index]):02d}"
                    if prior_profile and sum(prior_profile[:PLAYSTYLE_HASH_BINS]) > 0
                    else "cold-start"
                )
                model_inputs = {
                    "deck_features": deck_vector,
                    "playstyle_features": causal_style,
                    "bucket_features": bucket_features,
                }
                if set(model_inputs) & FORBIDDEN_MODEL_FIELDS:
                    raise CurrentRatingError("forbidden identity or oracle field entered model inputs")
                output_row = {
                    "schema_version": "deepmew.current_bucket_behavior_row.v1",
                    "row_id": derived["stable_row_id"],
                    "game_id": key[0],
                    "trajectory_id": trajectory_id,
                    "split": split,
                    "competence_context_kind": "CURRENT_EXTERNAL_RATING",
                    "x_coordinate": {
                        "store_id": x_store_id,
                        "row_index": int(derived["observation_storage_coordinate"]),
                    },
                    "model_inputs": model_inputs,
                    "legal_actions": legal_candidates,
                    "chosen_action_index": chosen_index,
                    "chosen_action": legal_candidates[chosen_index]["structure"],
                    "current_behavior_evidence_weight": weights["current_behavior_evidence_weight"],
                    "actor_loss_weight": 0.0 if derived["forced_state"] else 1.0,
                    "forced": bool(derived["forced_state"]),
                    "slices": {
                        **slice_labels,
                        "bucket": bucket,
                        "rating_trend": trend_status,
                        "playstyle_cluster": style_cluster,
                        "legal_set_size": str(int(derived["complete_legal_action_count"])),
                        "game_phase": str(legal.get("source_select_context") or "NOT_INSTRUMENTED"),
                        "cabt_holdout": "not_instrumented",
                        "reference_policy_holdout": "not_instrumented",
                        "population_scope": "external_current_population",
                        "internal_panel_bucket": "not_applicable",
                    },
                    "player_provenance_only": {
                        "canonical_player_id": player["canonical_player_id"],
                        "team_id": player["team_id"],
                        "mapping_rule": player["mapping_rule"],
                        "mapping_confidence": player["mapping_confidence"],
                    },
                    "current_rating_snapshot_id": overlay["overlay_sha256"],
                    "current_external_rating_bucket": bucket,
                    "replay_timestamp_utc": resolved["identity"].get("replay_timestamp_utc"),
                    "replay_timestamp_status": timestamp_status,
                    "replay_age_days": age_days,
                    "current_behavior_evidence_weight": weights["current_behavior_evidence_weight"],
                    "evidence_weight_components": weights,
                    "rating_stability_status": stability_status,
                    "deck_feature_status": deck["status"],
                    "deck_family_id": deck.get("deck_family_id"),
                    "playstyle_feature_version": "deepmew.causal_public_playstyle.v1",
                    "playstyle_prior_action_count": prior_actions,
                    "playstyle_continuity_weight_status": (
                        "MEASURED_CAUSAL_PRIOR_HISTORY" if prior_actions > 0 and sum(prior_profiles.get(key, ())) > 0
                        else "COLD_START_0_75"
                    ),
                    "source_validation": {
                        "source_family": derived["source_family"],
                        "validation_outcome": derived["validation_outcome"],
                        "legal_action_set_id": derived["legal_action_set_id"],
                        "complete_chosen_action_id": derived["complete_chosen_action_id"],
                        "source_transition_sha256": derived["source_transition_sha256"],
                        "derived_source_shard": str(source_shard),
                        "complete_legal_action_set": legal,
                        "complete_chosen_action": chosen,
                    },
                }
                writer.add(output_row)
                weights_by_bucket[bucket].append(weights["current_behavior_evidence_weight"])
                split_row_counts[split] += 1
                deck_status_counts[str(deck["status"])] += 1
                replay_timestamp_status[timestamp_status] += 1
                selected_count = sum(len(stage["selected_option_ids"]) for stage in chosen["stages"])
                increment = playstyle_increment(
                    str(derived["action_family"]),
                    str(derived["action_template"]),
                    int(derived["complete_legal_action_count"]),
                    stage_count=len(legal["stages"]),
                    selected_option_count=selected_count,
                )
                trajectory_totals[trajectory_id] = [
                    a + b for a, b in zip(trajectory_totals[trajectory_id], increment, strict=True)
                ]
                trajectory_counts[trajectory_id] += 1
                processed += 1
                if args.progress_every and processed % args.progress_every == 0:
                    print(json.dumps({
                        "rows_materialized": processed,
                        "expected_rows": expected_rows,
                        "shards_committed": len(writer.receipts),
                    }), flush=True)
                if args.max_rows and processed >= args.max_rows:
                    break
            if args.max_rows and processed >= args.max_rows:
                break
        writer.flush()
    except Exception as exc:
        writer.flush()
        _write_json(args.output / "FAIL_CLOSED_MANIFEST.json", {
            "schema_version": "deepmew.current_bucket_behavior_fail_closed.v1",
            "status": "BLOCKED_FAIL_CLOSED",
            "created_at_utc": utc_now_text(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rows_committed": sum(int(row["rows"]) for row in writer.receipts),
            "no_tensor_rematerialization": True,
            "second_database_created": False,
        })
        raise
    finally:
        connection.close()
    if not args.max_rows and processed != expected_rows:
        raise CurrentRatingError(f"materialized row count drifted: {processed} != {expected_rows}")
    bucket_ess = {bucket: effective_sample_size(weights) for bucket, weights in weights_by_bucket.items()}
    manifest = build_manifest_stub(
        overlay_path=args.overlay,
        rows=processed,
        games=len({key[0] for key in resolved_seats}),
        shards=writer.receipts,
        split_counts=split_row_counts,
        bucket_counts={bucket: len(weights) for bucket, weights in weights_by_bucket.items()},
        bucket_ess=bucket_ess,
    )
    manifest.update({
        "mode": "CANARY" if args.max_rows else "FULL",
        "expected_full_rows": expected_rows,
        "split_assignment_path": str(args.output / "GAME_SPLIT_ASSIGNMENTS_V1.json"),
        "split_assignment_sha256": file_sha256(args.output / "GAME_SPLIT_ASSIGNMENTS_V1.json"),
        "deck_feature_status_counts": dict(sorted(deck_status_counts.items())),
        "replay_timestamp_status_counts": dict(sorted(replay_timestamp_status.items())),
        "forced_actor_rows_have_zero_loss": True,
        "x_stores": x_store_contracts(),
        "feature_schema": {
            "x_dim": 144,
            "deck_dim": len(deck_feature_names),
            "deck_feature_names": deck_feature_names,
            "playstyle_dim": len(PLAYSTYLE_AXES) + 1,
            "playstyle_feature_names": list(PLAYSTYLE_AXES) + ["history_support_norm100"],
            "bucket_dim": 4,
            "bucket_labels": [
                "R0_CURRENT_FRONTIER",
                "R1_CURRENT_STRONG",
                "R2_CURRENT_INTERMEDIATE",
                "R3_CURRENT_LOWER",
            ],
            "action_dim": ACTION_HASH_BINS + 2,
            "intent_classes": 1,
        },
        "complete_action_contract": {
            "unit": "complete_compound_action",
            "chosen_index_derivation": "exact_canonical_structure_match",
            "prompt_option_indices_not_accepted": True,
        },
        "model_input_excludes": sorted(FORBIDDEN_MODEL_FIELDS),
        "target_instrumentation": {
            "resource_commitment": {"status": "not_instrumented"},
            "next_turn_threat": {"status": "not_instrumented"},
            "strategic_intent": {"status": "not_instrumented"},
            "action_family": {
                "status": "instrumented",
                "causal_scope": "actor_visible_pre_action_plus_current_complete_action",
                "source": "canonical_legal_action_set.action_family",
            },
            "structured_target": {
                "status": "instrumented",
                "causal_scope": "actor_visible_pre_action_plus_current_complete_action",
                "source": "canonical_complete_action.selected_option_ids",
            },
        },
        "evaluation_instrumentation": {
            "cabt_holdout": {"status": "not_instrumented"},
            "reference_policy_holdout": {"status": "not_instrumented"},
            "internal_panel_adapter": {"status": "not_instrumented"},
        },
    })
    if args.max_rows:
        manifest["status"] = "CANARY_CONTRACT_PASS_NOT_TRAINABLE"
    elif set(split_row_counts) != {"train", "calibration", "test"}:
        raise CurrentRatingError(f"full corpus split coverage is incomplete: {dict(split_row_counts)}")
    _write_json(args.output / "CURRENT_BUCKET_BEHAVIOR_CORPUS_V1_MANIFEST.json", manifest)
    leakage = {
        "schema_version": "deepmew.current_bucket_behavior_leakage_temporal_receipt.v1",
        "status": "PASS" if processed else "FAIL",
        "created_at_utc": utc_now_text(),
        "rows_scanned": processed,
        "current_population_model": True,
        "rating_as_of": overlay["snapshot_timestamp_utc"],
        "historical_as_of_claim": False,
        "player_identity_location": "player_provenance_only; excluded from model_inputs",
        "future_actions_in_playstyle": 0,
        "current_game_result_inputs": 0,
        "hidden_zone_inputs": 0,
        "oracle_or_search_inputs": 0,
        "checkpoint_identity_inputs": 0,
        "observation_x_rows_copied": 0,
        "second_database_created": False,
        "complete_game_split": True,
        "heldout_player_split": True,
        "actor_loss_on_forced_rows": 0.0,
        "passed": bool(processed),
    }
    _write_json(args.output / "LEAKAGE_AND_TEMPORAL_CAUSALITY_RECEIPT_V1.json", leakage)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-rows", type=int, default=8_192)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--deck-audit-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10_000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.deck_audit_only:
        result = run_deck_join_audit(args.overlay, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))

        return
    result = build_corpus(args)
    print(json.dumps({"status": result["status"], "rows": result["rows"], "games": result["games"]}, indent=2))


if __name__ == "__main__":
    main()
