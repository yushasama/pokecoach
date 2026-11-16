"""Build a research-only sparse ordered-pair league payoff matrix.

Every distinct ordered pair ``i -> j`` receives the same even game budget.
Pairs are interleaved one game at a time, seats alternate within each ordered
pair, and progress is atomically persisted after every game.  The CABT engine
RNG is not instrumented, so games and reverse ordered pairs are independent,
not paired.  Outputs are diagnostic only: this runner computes no Elo, assumes
no transitivity, selects no member, and mutates no runtime/submission surface.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for import_path in (ROOT, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_r17_bucket_arena as r17  # noqa: E402
from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown  # noqa: E402
from deepmew.curriculum.league_registry import LeagueRole, sha256_path  # noqa: E402
from deepmew.options.identity import resolve_option_identity  # noqa: E402
from deepmew.policy.selection import selection_bounds  # noqa: E402
from deepmew.real_meta_arena.runner import RealMetaGameRecord, run_real_meta_game  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402


ROSTER_SCHEMA_VERSION = "deepmew.league_payoff_roster.v1"
SCHEMA_VERSION = "deepmew.league_payoff_matrix.v2"
GAME_SCHEMA_VERSION = "deepmew.league_payoff_game.v2"
PROGRESS_SCHEMA_VERSION = "deepmew.league_payoff_progress.v2"
PAIR_SCHEMA_VERSION = "deepmew.league_payoff_pair.v2"
PUBLIC_TELEMETRY_SCHEMA_VERSION = "deepmew.league_payoff_public_telemetry.v1"
MEGA_STARMIE_EX_CARD_ID = 1031
ENGINE_RNG_CONTRACT = (
    "not_instrumented; cg.game.battle_start exposes no seed; ordered-pair "
    "samples and reverse pairs are independent, not paired"
)
IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__", ".pytest_cache", ".git"})
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


def _integer_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prizes_remaining(player: Any) -> int | None:
    if not isinstance(player, dict):
        return None
    prizes = player.get("prize")
    if isinstance(prizes, list):
        return len(prizes)
    for key in ("prizeCount", "prizesRemaining"):
        value = _integer_or_none(player.get(key))
        if value is not None:
            return max(0, value)

    return None


def _selection_combination_count(option_count: int, minimum: int, maximum: int) -> int:
    return sum(math.comb(option_count, count) for count in range(minimum, maximum + 1))


class PublicGameTelemetry:
    """Aggregate actor-visible/public game metrics without retaining observations."""

    def __init__(self, label_by_seat: dict[int, str]) -> None:
        if set(label_by_seat) != {0, 1} or len(set(label_by_seat.values())) != 2:
            raise ValueError("public telemetry requires two distinct seat labels")
        self.label_by_seat = dict(label_by_seat)
        self.seat_by_label = {label: seat for seat, label in label_by_seat.items()}
        self.rows = {
            label: {
                "decision_count": 0,
                "forced_decision_count": 0,
                "action_category_counts": Counter(),
                "unforced_action_identity_counts": Counter(),
                "unforced_action_categories_by_legal_cardinality": {},
                "unforced_action_identities_by_legal_cardinality": {},
                "first_attack_turn": None,
                "first_prize_turn": None,
                "mega_starmie_evolution_turn": None,
                "starting_prizes": None,
                "minimum_prizes_remaining": None,
                "maximum_prize_lead": None,
                "positive_prize_lead_by_turn_5": False,
            }
            for label in label_by_seat.values()
        }

    def observe(self, observation: dict[str, Any]) -> None:
        current = observation.get("current")
        if not isinstance(current, dict):
            return
        players = current.get("players")
        if not isinstance(players, list) or len(players) < 2:
            return
        turn = _integer_or_none(current.get("turn"))
        for seat, label in self.label_by_seat.items():
            own = _prizes_remaining(players[seat])
            opponent = _prizes_remaining(players[1 - seat])
            if own is None:
                continue
            row = self.rows[label]
            starting = row["starting_prizes"]
            row["starting_prizes"] = own if starting is None else max(int(starting), own)
            minimum = row["minimum_prizes_remaining"]
            row["minimum_prizes_remaining"] = own if minimum is None else min(int(minimum), own)
            if (
                row["first_prize_turn"] is None
                and turn is not None
                and own < int(row["starting_prizes"])
            ):
                row["first_prize_turn"] = turn
            if opponent is None:
                continue
            prize_lead = opponent - own
            maximum_lead = row["maximum_prize_lead"]
            row["maximum_prize_lead"] = (
                prize_lead if maximum_lead is None else max(int(maximum_lead), prize_lead)
            )
            if turn is not None and turn <= 5 and prize_lead > 0:
                row["positive_prize_lead_by_turn_5"] = True

    def observe_action(
        self,
        label: str,
        observation: dict[str, Any],
        selected: list[int],
    ) -> None:
        self.observe(observation)
        if label not in self.rows:
            raise ValueError(f"unknown telemetry label: {label!r}")
        select = observation.get("select")
        if not isinstance(select, dict):
            return
        options = select.get("option")
        if not isinstance(options, list):
            return
        minimum, maximum = selection_bounds(select, len(options))
        forced = _selection_combination_count(len(options), minimum, maximum) == 1
        row = self.rows[label]
        row["decision_count"] += 1
        if forced:
            row["forced_decision_count"] += 1
        current = observation.get("current")
        turn = _integer_or_none(current.get("turn")) if isinstance(current, dict) else None
        identities = [
            resolve_option_identity(observation, options[index])
            for index in selected
            if isinstance(index, int) and 0 <= index < len(options) and isinstance(options[index], dict)
        ]
        categories = [identity.option_type for identity in identities] or ["STOP"]
        row["action_category_counts"].update(categories)
        if not forced:
            cardinality = str(len(options))
            cardinality_counts = row["unforced_action_categories_by_legal_cardinality"].setdefault(
                cardinality,
                Counter(),
            )
            cardinality_counts.update(categories)
            identity_counts = row["unforced_action_identities_by_legal_cardinality"].setdefault(
                cardinality,
                Counter(),
            )
            semantic_identities = [
                "|".join(
                    str(value)
                    for value in (
                        identity.option_type,
                        identity.card_id,
                        identity.attack_id,
                        identity.target_area,
                        identity.target_card_id,
                    )
                )
                for identity in identities
            ] or ["STOP|None|None|None|None"]
            row["unforced_action_identity_counts"].update(semantic_identities)
            identity_counts.update(semantic_identities)
        if turn is not None and "ATTACK" in categories and row["first_attack_turn"] is None:
            row["first_attack_turn"] = turn
        if turn is not None and row["mega_starmie_evolution_turn"] is None:
            if any(
                identity.option_type == "EVOLVE"
                and identity.card_id == MEGA_STARMIE_EX_CARD_ID
                for identity in identities
            ):
                row["mega_starmie_evolution_turn"] = turn

    def receipt(self) -> dict[str, Any]:
        members: dict[str, Any] = {}
        for label, source in self.rows.items():
            decision_count = int(source["decision_count"])
            forced_count = int(source["forced_decision_count"])
            start = source["starting_prizes"]
            minimum = source["minimum_prizes_remaining"]
            prizes_taken = int(start) - int(minimum) if start is not None and minimum is not None else None
            first_attack = source["first_attack_turn"]
            first_prize = source["first_prize_turn"]
            evolution = source["mega_starmie_evolution_turn"]
            members[label] = {
                "seat": self.seat_by_label[label],
                "decision_count": decision_count,
                "forced_decision_count": forced_count,
                "forced_decision_rate": forced_count / decision_count if decision_count else None,
                "action_category_counts": dict(sorted(source["action_category_counts"].items())),
                "unforced_action_identity_counts": dict(
                    sorted(source["unforced_action_identity_counts"].items())
                ),
                "unforced_action_categories_by_legal_cardinality": {
                    cardinality: dict(sorted(counts.items()))
                    for cardinality, counts in sorted(
                        source["unforced_action_categories_by_legal_cardinality"].items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "unforced_action_identities_by_legal_cardinality": {
                    cardinality: dict(sorted(counts.items()))
                    for cardinality, counts in sorted(
                        source["unforced_action_identities_by_legal_cardinality"].items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "first_attack_turn": first_attack,
                "first_attack_turn_capped_11": min(int(first_attack), 11) if first_attack is not None else 11,
                "attack_by_turn_3": first_attack is not None and int(first_attack) <= 3,
                "first_prize_turn": first_prize,
                "first_prize_turn_capped_11": min(int(first_prize), 11) if first_prize is not None else 11,
                "prize_by_turn_4": first_prize is not None and int(first_prize) <= 4,
                "prizes_taken": prizes_taken,
                "zero_prize_game": prizes_taken == 0 if prizes_taken is not None else None,
                "maximum_prize_lead": source["maximum_prize_lead"],
                "positive_prize_lead_by_turn_5": source["positive_prize_lead_by_turn_5"],
                "mega_starmie_evolution_turn": evolution,
                "mega_starmie_evolution_by_turn_4": evolution is not None and int(evolution) <= 4,
            }

        return {
            "schema_version": PUBLIC_TELEMETRY_SCHEMA_VERSION,
            "provenance": "actor_visible_selected_actions_and_public_player_counts_only",
            "raw_observation_persisted": False,
            "legal_option_payload_persisted": False,
            "members": members,
        }


class PublicTelemetryAgent:
    def __init__(self, inner: Any, label: str, telemetry: PublicGameTelemetry) -> None:
        self.inner = inner
        self.label = label
        self.telemetry = telemetry

    def select(self, observation: dict[str, Any], rng: random.Random) -> list[int]:
        selected = self.inner.select(observation, rng)
        self.telemetry.observe_action(self.label, observation, selected)

        return selected


class PayoffSafetyError(RuntimeError):
    """Raised after a game safety failure has been durably recorded."""


class PayoffRunLock:
    """OS-released non-blocking lock preventing concurrent run/resume writers."""

    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / ".payoff-run.lock"
        self.handle: Any = None

    def __enter__(self) -> "PayoffRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another payoff run/resume holds {self.path}") from exc

        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def exclusive_payoff_run(function):
    @wraps(function)
    def locked(args, *function_args, **function_kwargs):
        with PayoffRunLock(Path(args.out_dir).resolve()):
            return function(args, *function_args, **function_kwargs)

    return locked


@dataclass(frozen=True)
class RosterMember:
    league_member_id: str
    label: str
    spec: str
    deck_binding: str
    feature_policy_id: str
    role: str

    def receipt(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OrderedPair:
    pair_id: str
    row_member_id: str
    column_member_id: str
    row_label: str
    column_label: str

    def receipt(self) -> dict[str, str]:
        return {
            "schema_version": PAIR_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "row_member_id": self.row_member_id,
            "column_member_id": self.column_member_id,
            "row_label": self.row_label,
            "column_label": self.column_label,
        }


@dataclass(frozen=True)
class PayoffGameJob:
    job_key: str
    pair: OrderedPair
    game_index: int
    row_seat: int
    policy_rng_seed: int
    execution_ordinal: int = 0


@dataclass(frozen=True)
class BuiltMember:
    member: RosterMember
    agent: Any
    deck: tuple[int, ...]
    deck_hash: str
    live_feature_policy_id: str
    live_feature_deck_hash: str
    feature_engine_deck_match: bool
    feature_policy_binding: str

    def receipt(self) -> dict[str, Any]:
        return {
            **self.member.receipt(),
            "deck_hash": self.deck_hash,
            "deck_size": len(self.deck),
            "live_feature_policy_id": self.live_feature_policy_id,
            "live_feature_deck_hash": self.live_feature_deck_hash,
            "feature_engine_deck_match": self.feature_engine_deck_match,
            "feature_policy_binding": self.feature_policy_binding,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(payload: Any, *, length: int = 24) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _directory_files(path: Path) -> list[Path]:
    files: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directories, names in os.walk(path, onerror=raise_walk_error):
        directories[:] = sorted(name for name in directories if name not in IGNORED_DIRECTORY_NAMES)
        current_path = Path(current)
        for name in sorted(names):
            candidate = current_path / name
            if candidate.suffix.lower() in IGNORED_SUFFIXES:
                continue
            files.append(candidate)

    return files


def file_receipt(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"hashed input does not exist: {resolved}")
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }

    rows = []
    digest = hashlib.sha256()
    for candidate in _directory_files(resolved):
        relative = candidate.relative_to(resolved).as_posix()
        size = candidate.stat().st_size
        sha256 = _sha256_file(candidate)
        rows.append({"relative_path": relative, "size_bytes": size, "sha256": sha256})
        digest.update(f"{relative}\0{size}\0{sha256}\n".encode("utf-8"))

    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(rows),
        "size_bytes": sum(row["size_bytes"] for row in rows),
        "sha256": sha256_path(resolved, purpose="payoff input directory"),
        "hash_contract": "deepmew.directory.v1",
        "files": rows,
    }


def _require_string(row: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    if key not in row or not isinstance(row[key], str):
        raise ValueError(f"roster member {key!r} must be a string")
    value = row[key].strip()
    if not value and not allow_empty:
        raise ValueError(f"roster member {key!r} must not be empty")

    return value


def _base_spec_kind(spec: str) -> str:
    kind, separator, target = spec.partition(":")
    if not separator or not target:
        raise ValueError(f"member spec needs kind:target, got {spec!r}")

    return kind.removesuffix("+guard")


def load_roster(path: str | Path) -> list[RosterMember]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != ROSTER_SCHEMA_VERSION:
        raise ValueError(f"roster schema_version must be {ROSTER_SCHEMA_VERSION!r}")
    rows = payload.get("members")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("payoff roster requires at least two members")

    members = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("every roster member must be an object")
        member = RosterMember(
            league_member_id=_require_string(raw, "league_member_id"),
            label=_require_string(raw, "label"),
            spec=_require_string(raw, "spec"),
            deck_binding=_require_string(raw, "deck_binding", allow_empty=True),
            feature_policy_id=_require_string(raw, "feature_policy_id"),
            role=_require_string(raw, "role"),
        )
        LeagueRole(member.role)
        kind = _base_spec_kind(member.spec)
        if kind == "checkpoint" and not member.deck_binding:
            raise ValueError(f"checkpoint member {member.label!r} requires an explicit deck_binding")
        if kind == "spec" and member.deck_binding:
            raise ValueError(
                f"spec member {member.label!r} must use an empty deck_binding; "
                "R17 obtains that member's deck from the named policy"
            )
        members.append(member)

    labels = [member.label for member in members]
    if len(labels) != len(set(labels)):
        raise ValueError("roster member labels must be unique")

    return members


def build_ordered_pairs(members: list[RosterMember]) -> list[OrderedPair]:
    pairs = []
    keys: set[str] = set()
    for row_member in members:
        for column_member in members:
            if row_member.label == column_member.label:
                continue
            identity = {
                "row_member_id": row_member.league_member_id,
                "column_member_id": column_member.league_member_id,
            }
            pair_id = _stable_id(identity)
            if pair_id in keys:
                raise ValueError(f"duplicate ordered-pair identity: {identity}")
            keys.add(pair_id)
            pairs.append(OrderedPair(
                pair_id,
                row_member.league_member_id,
                column_member.league_member_id,
                row_member.label,
                column_member.label,
            ))

    return pairs


def derive_policy_rng_seed(base_seed: int, pair_id: str, game_index: int) -> int:
    digest = hashlib.sha256(
        _canonical_json({"base_seed": base_seed, "pair_id": pair_id, "game_index": game_index}).encode("utf-8")
    ).digest()

    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_interleaved_jobs(
    pairs: list[OrderedPair],
    *,
    games: int,
    base_seed: int,
) -> list[PayoffGameJob]:
    jobs = []
    keys: set[str] = set()
    for game_index in range(games):
        for pair in pairs:
            identity = {"pair_id": pair.pair_id, "game_index": game_index}
            job_key = _stable_id(identity)
            if job_key in keys:
                raise ValueError(f"duplicate payoff game identity: {identity}")
            keys.add(job_key)
            jobs.append(
                PayoffGameJob(
                    job_key=job_key,
                    pair=pair,
                    game_index=game_index,
                    row_seat=game_index % 2,
                    policy_rng_seed=derive_policy_rng_seed(base_seed, pair.pair_id, game_index),
                    execution_ordinal=len(jobs),
                )
            )

    return jobs


def _unwrap_agent(agent: Any) -> Any:
    current = agent
    visited: set[int] = set()
    while hasattr(current, "inner") and id(current) not in visited:
        visited.add(id(current))
        current = current.inner

    return current


def _bind_and_verify_feature_identity(
    agent: Any,
    feature_policy_id: str,
    engine_deck_hash: str,
) -> tuple[str, str, bool, str]:
    base = _unwrap_agent(agent)
    policy = getattr(base, "_policy", None)
    if policy is not None and hasattr(policy, "policy_id"):
        policy.policy_id = feature_policy_id
        actual = str(policy.policy_id)
        if actual != feature_policy_id:
            raise ValueError(f"failed to bind feature_policy_id {feature_policy_id!r}; got {actual!r}")
        feature_deck_hash = str(getattr(policy, "deck_hash", ""))
        if not feature_deck_hash:
            raise ValueError(f"member policy {type(policy).__name__} does not expose feature deck identity")
        manifest = getattr(policy, "manifest", None)
        manifest_feature_deck_hash = (
            str(manifest.get("feature_deck_hash", "")) if isinstance(manifest, dict) else ""
        )
        if manifest_feature_deck_hash and manifest_feature_deck_hash != feature_deck_hash:
            raise ValueError(
                "live feature deck identity does not match the loaded artifact manifest: "
                f"runtime={feature_deck_hash!r}, manifest={manifest_feature_deck_hash!r}"
            )
        identity_source = "artifact_manifest_verified" if manifest_feature_deck_hash else "live_binding_verified"

        return (
            actual,
            feature_deck_hash,
            feature_deck_hash == engine_deck_hash,
            f"explicit_runtime_policy_id;{identity_source}",
        )

    if isinstance(base, (r17.SpecCandidateAgent, r17.ReferenceTeacherCandidateAgent)):
        return feature_policy_id, engine_deck_hash, True, "explicit_not_applicable_to_heuristic"

    raise ValueError(
        f"member agent {type(base).__name__} does not expose an auditable feature_policy_id binding"
    )


def _agent_deck(agent: Any) -> tuple[int, ...]:
    base = _unwrap_agent(agent)
    deck = tuple(int(card_id) for card_id in (getattr(base, "deck", None) or []))
    if len(deck) != 60:
        raise ValueError(f"member agent {getattr(agent, 'label', '?')!r} did not expose a 60-card deck")

    return deck


def _apply_engine_deck_binding(agent: Any, deck_binding: str) -> None:
    if not deck_binding:
        return
    binding = r17._checkpoint_deck_binding(deck_binding)
    if binding is None:
        raise ValueError(f"failed to resolve engine deck binding {deck_binding!r}")
    deck = [int(card_id) for card_id in binding["deck"]]
    if len(deck) != 60:
        raise ValueError(f"engine deck binding {deck_binding!r} did not resolve to 60 cards")
    base = _unwrap_agent(agent)
    base.deck = deck


def build_member_agents(
    members: list[RosterMember],
    *,
    deck_policy: str,
) -> dict[str, BuiltMember]:
    built: dict[str, BuiltMember] = {}
    build_parameters = inspect.signature(r17.build_candidate).parameters
    supports_feature_policy_id = "feature_policy_id" in build_parameters
    for member in members:
        if _base_spec_kind(member.spec) == "bundle":
            bundle_path = str(Path(member.spec.partition(":")[2]).resolve())
            sys.path[:] = [entry for entry in sys.path if entry != bundle_path]
            sys.path.insert(0, bundle_path)
            sys.modules.pop("r5_distilled_policy", None)
        kwargs = {}
        if supports_feature_policy_id:
            kwargs["feature_policy_id"] = member.feature_policy_id
        agent = r17.build_candidate(member.spec, deck_policy, member.deck_binding, **kwargs)
        _apply_engine_deck_binding(agent, member.deck_binding)
        deck = _agent_deck(agent)
        deck_hash = normalized_deck_hash(list(deck))
        live_policy_id, live_feature_deck_hash, feature_engine_match, binding = _bind_and_verify_feature_identity(
            agent,
            member.feature_policy_id,
            deck_hash,
        )
        built[member.label] = BuiltMember(
            member=member,
            agent=agent,
            deck=deck,
            deck_hash=deck_hash,
            live_feature_policy_id=live_policy_id,
            live_feature_deck_hash=live_feature_deck_hash,
            feature_engine_deck_match=feature_engine_match,
            feature_policy_binding=binding,
        )

    return built


def _reference_path(reference: str, *, reference_kind: str) -> Path | None:
    if not reference:
        return None
    if reference_kind == "spec":
        kind, separator, target = reference.partition(":")
        kind = kind.removesuffix("+guard")
        if separator and kind in {"checkpoint", "bundle"} and target:
            return Path(target)

        return None
    kind, separator, target = reference.partition(":")
    if separator and kind == "deck" and target:
        return Path(target)
    if separator and kind == "bundle" and target:
        return Path(target)

    return Path(reference)


def _source_paths() -> dict[str, Path]:
    paths = {
        "payoff_runner": Path(__file__),
        "r17_builder": Path(r17.__file__),
        "deepmew_behavior_tree": ROOT / "deepmew",
        "cg_vendor_tree": ROOT / "cg",
        "root_main": ROOT / "main.py",
        "root_deck": ROOT / "deck.csv",
    }
    for label, value in (
        ("r4f_live_policy", getattr(r17, "r4f", None)),
        ("r4_feature_builder", getattr(getattr(r17, "r4f", None), "r4", None)),
    ):
        source = getattr(value, "__file__", None)
        if source:
            paths[label] = Path(source)
    adapter_source = inspect.getsourcefile(r17.PilotPolicyAdapter)
    if adapter_source:
        paths["pilot_policy_adapter"] = Path(adapter_source)

    return paths


def build_input_receipts(
    roster_path: str | Path,
    members: Iterable[RosterMember],
) -> dict[str, Any]:
    references = {"roster": Path(roster_path), **_source_paths()}
    for member in members:
        artifact = _reference_path(member.spec, reference_kind="spec")
        if artifact is not None:
            references[f"member_artifact:{member.label}"] = artifact
        binding = _reference_path(member.deck_binding, reference_kind="deck_binding")
        if binding is not None:
            references[f"member_deck_binding:{member.label}"] = binding

    return {label: file_receipt(path) for label, path in sorted(references.items())}


def _command_receipt(argv: list[str]) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), *argv]

    return {
        "argv": command,
        "display": subprocess.list2cmdline(command),
        "cwd": str(Path.cwd().resolve()),
    }


def _environment_receipt() -> dict[str, Any]:
    torch_module = getattr(getattr(r17, "r4f", None), "torch", None)
    numpy_module = getattr(getattr(r17, "r4f", None), "np", None)

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "torch": str(getattr(torch_module, "__version__", "not_imported")),
        "numpy": str(getattr(numpy_module, "__version__", "not_imported")),
    }


def build_config_receipt(
    args: argparse.Namespace,
    *,
    members: list[RosterMember],
    built_members: dict[str, BuiltMember],
    pairs: list[OrderedPair],
    input_receipts: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "schema_migration": (
            "v2 supersedes v1 by adding immutable league-member pair IDs, execution ordinals, "
            "strict resume safety validation, and separate engine-deck/feature-deck receipts"
        ),
        "run_id": args.run_id,
        "generation_id": getattr(args, "generation_id", ""),
        "snapshot_id": getattr(args, "snapshot_id", ""),
        "out_dir": str(Path(args.out_dir).resolve()),
        "roster_path": str(Path(args.roster).resolve()),
        "members": [built_members[member.label].receipt() for member in members],
        "ordered_pairs": [pair.receipt() for pair in pairs],
        "ordered_pair_count": len(pairs),
        "games_per_ordered_pair": args.games,
        "total_games": len(pairs) * args.games,
        "execution_order": "game_index outer, roster-ordered distinct pairs inner",
        "execution_ordinal": "zero-based index in the interleaved game schedule",
        "seat_schedule": "row member seat = zero_based_game_index % 2",
        "base_seed": args.seed,
        "policy_rng_seed_formula": "sha256(base_seed, pair_id, game_index) truncated to signed-63-bit",
        "engine_rng_control": ENGINE_RNG_CONTRACT,
        "sample_relationship": "independent; not paired",
        "capture_public_telemetry": bool(getattr(args, "capture_public_telemetry", False)),
        "deck_policy": args.deck_policy,
        "max_steps": args.max_steps,
        "wall_clock_timeout": "not_instrumented; max_steps is the only timeout guard",
        "min_decided": args.min_decided,
        "max_wilson_width": args.max_wilson_width,
        "redundancy_threshold": args.redundancy_threshold,
        "payoff_definition": "W_ij = (wins + 0.5 * draws) / completed games",
        "wilson_definition": "Wilson 95% interval on wins / (wins + losses); draws excluded and reported",
        "decision_role": "diagnostic only",
        "elo_computed": False,
        "transitivity_assumed": False,
        "selection_performed": False,
        "runtime_mutation": False,
        "behavior_source_hash_scope": (
            "full deepmew and cg trees plus R17/R4F/R4/adapter/root entry sources and every member artifact/deck"
        ),
        "input_receipts": input_receipts,
        "environment": _environment_receipt(),
    }
    fingerprint = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at_utc": _utc_now(),
        "config_fingerprint": fingerprint,
        "config": config,
        "launch_command": command,
    }


def _validate_progress_jobs(progress: dict[str, Any], jobs: list[PayoffGameJob]) -> None:
    expected = {job.job_key: job for job in jobs}
    persisted = progress.get("jobs")
    if not isinstance(persisted, dict):
        raise ValueError("progress jobs must be an object")
    unexpected = sorted(set(persisted) - set(expected))
    if unexpected:
        raise ValueError(f"progress contains jobs outside this payoff schedule: {unexpected[:3]}")
    for job_key, row in persisted.items():
        job = expected[job_key]
        required = {
            "schema_version": GAME_SCHEMA_VERSION,
            "job_key": job_key,
            "pair_id": job.pair.pair_id,
            "row_member_id": job.pair.row_member_id,
            "column_member_id": job.pair.column_member_id,
            "row_label": job.pair.row_label,
            "column_label": job.pair.column_label,
            "game_index": job.game_index,
            "row_seat": job.row_seat,
            "policy_rng_seed": job.policy_rng_seed,
            "execution_ordinal": job.execution_ordinal,
        }
        if not isinstance(row, dict) or any(row.get(key) != value for key, value in required.items()):
            raise ValueError(f"persisted payoff game identity mismatch for {job_key}")
        outcome = row.get("outcome")
        if outcome not in {"win", "loss", "draw", "crash", "timeout", "illegal", "incomplete"}:
            raise ValueError(f"persisted payoff game has unknown outcome for {job_key}")
        flags = {key: row.get(key) for key in ("crashed", "timed_out", "illegal", "incomplete")}
        if any(not isinstance(value, bool) for value in flags.values()):
            raise ValueError(f"persisted payoff game lacks boolean safety flags for {job_key}")
        expected_failure = {
            "crash": "crashed",
            "timeout": "timed_out",
            "illegal": "illegal",
            "incomplete": "incomplete",
        }.get(outcome)
        active = [key for key, value in flags.items() if value]
        if expected_failure is None and active:
            raise ValueError(f"successful payoff game has safety flags for {job_key}: {active}")
        if expected_failure is not None and active != [expected_failure]:
            raise ValueError(
                f"payoff game outcome/safety mismatch for {job_key}: outcome={outcome}, flags={active}"
            )


def prepare_progress(
    out_dir: Path,
    config_receipt: dict[str, Any],
    *,
    jobs: list[PayoffGameJob],
    resume: bool,
    command: dict[str, Any],
) -> dict[str, Any]:
    config_path = out_dir / "config.json"
    progress_path = out_dir / "progress.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"payoff run already exists; use --resume: {out_dir}")
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != config_receipt["config_fingerprint"]:
            raise ValueError("resume config/input hash mismatch; refusing to mix payoff samples")
        if not progress_path.exists():
            writer_temps = [
                path.name
                for path in out_dir.iterdir()
                if path.is_file()
                and path.name.startswith(".progress.json.")
                and path.name.endswith(".tmp")
            ]
            unexpected_files = [
                path.name
                for path in out_dir.iterdir()
                if path.name not in {"config.json", ".payoff-run.lock"}
                and path.name not in writer_temps
            ]
            if unexpected_files:
                raise ValueError(
                    "resume progress.json is missing but other run artifacts exist: "
                    f"{sorted(unexpected_files)}"
                )
            progress = {
                "schema_version": PROGRESS_SCHEMA_VERSION,
                "config_fingerprint": config_receipt["config_fingerprint"],
                "status": "running",
                "created_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_wall_seconds": 0.0,
                "commands": [command],
                "recovered_uncommitted_writer_temps": sorted(writer_temps),
                "jobs": {},
            }
            write_json(progress_path, progress)

            return progress
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config_fingerprint") != config_receipt["config_fingerprint"]:
            raise ValueError("progress/config fingerprint mismatch")
        _validate_progress_jobs(progress, jobs)
        persisted_failure = next(
            (row for row in progress["jobs"].values() if _is_safety_failure(row)),
            None,
        )
        if persisted_failure is not None:
            progress["status"] = "safety_failed"
        commands = progress.setdefault("commands", [])
        if command not in commands:
            commands.append(command)

        return progress

    non_lock_entries = (
        [path for path in out_dir.iterdir() if path.name != ".payoff-run.lock"]
        if out_dir.exists() else []
    )
    if non_lock_entries:
        raise FileExistsError(f"output directory is non-empty and has no payoff config: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "config_fingerprint": config_receipt["config_fingerprint"],
        "status": "running",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "elapsed_wall_seconds": 0.0,
        "commands": [command],
        "recovered_uncommitted_writer_temps": [],
        "jobs": {},
    }
    write_json(config_path, config_receipt)
    write_json(progress_path, progress)

    return progress


def _reset_agent(agent: Any, seed: int) -> None:
    queue = [agent]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        reset = getattr(current, "reset", None)
        if callable(reset):
            reset(seed)
        for attribute in ("inner", "_policy"):
            child = getattr(current, attribute, None)
            if child is not None:
                queue.append(child)


def _crash_record(job: PayoffGameJob, exc: Exception, elapsed_seconds: float) -> RealMetaGameRecord:
    return RealMetaGameRecord(
        game_index=job.game_index,
        seat_our=job.row_seat,
        rng_seed=job.policy_rng_seed,
        result=None,
        turns=0,
        crashed=True,
        timed_out=False,
        illegal=False,
        elapsed_seconds=elapsed_seconds,
        error=f"{type(exc).__name__}: {exc}",
    )


def _game_row(job: PayoffGameJob, record: RealMetaGameRecord) -> dict[str, Any]:
    outcome = record.outcome
    incomplete = outcome not in {"win", "loss", "draw", "crash", "timeout", "illegal"}

    return {
        "schema_version": GAME_SCHEMA_VERSION,
        "job_key": job.job_key,
        "pair_id": job.pair.pair_id,
        "row_member_id": job.pair.row_member_id,
        "column_member_id": job.pair.column_member_id,
        "row_label": job.pair.row_label,
        "column_label": job.pair.column_label,
        "game_index": job.game_index,
        "execution_ordinal": job.execution_ordinal,
        "row_seat": job.row_seat,
        "policy_rng_seed": job.policy_rng_seed,
        "engine_rng_control": "not_instrumented",
        "sample_relationship": "independent; not paired",
        "result_seat": record.result,
        "outcome": "incomplete" if incomplete else outcome,
        "turns": record.turns,
        "crashed": record.crashed,
        "timed_out": record.timed_out,
        "illegal": record.illegal,
        "incomplete": incomplete,
        "illegal_agent": record.illegal_agent,
        "elapsed_seconds": record.elapsed_seconds,
        "error": record.error,
    }


def _is_safety_failure(row: dict[str, Any]) -> bool:
    return any(bool(row.get(key)) for key in ("crashed", "timed_out", "illegal", "incomplete"))


def wilson_interval(wins: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = wins / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total)) / denominator

    return max(0.0, center - half), min(1.0, center + half)


def pair_statistics(
    pair: OrderedPair,
    rows: list[dict[str, Any]],
    *,
    planned_games: int,
    min_decided: int,
    max_wilson_width: float,
) -> dict[str, Any]:
    wins = sum(row["outcome"] == "win" for row in rows)
    losses = sum(row["outcome"] == "loss" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    completed = wins + losses + draws
    decided = wins + losses
    win_rate = wins / decided if decided else None
    point_rate = (wins + 0.5 * draws) / completed if completed else None
    low, high = wilson_interval(wins, decided)
    width = high - low
    safety = {
        "crashes": sum(bool(row.get("crashed")) for row in rows),
        "timeouts": sum(bool(row.get("timed_out")) for row in rows),
        "illegal": sum(bool(row.get("illegal")) for row in rows),
        "incomplete": sum(bool(row.get("incomplete")) for row in rows),
    }
    budget_complete = len(rows) == planned_games and completed == planned_games
    low_sample = decided < min_decided or not budget_complete

    return {
        **pair.receipt(),
        "planned_games": planned_games,
        "attempted": len(rows),
        "completed": completed,
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "W_ij": round(point_rate, 6) if point_rate is not None else None,
        "decided_win_rate": round(win_rate, 6) if win_rate is not None else None,
        "wilson95_decided": [round(low, 6), round(high, 6)],
        "wilson95_width": round(width, 6),
        "low_sample": low_sample,
        "wide_uncertainty": width > max_wilson_width,
        "uncertainty_flag": low_sample or width > max_wilson_width,
        "budget_complete": budget_complete,
        "safety": safety,
        "safety_clean": not any(safety.values()),
        "elapsed_game_seconds": round(sum(float(row.get("elapsed_seconds") or 0.0) for row in rows), 6),
    }


def _asymmetry_diagnostics(
    labels: list[str],
    stats_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = []
    for left, right in itertools.combinations(labels, 2):
        forward = stats_by_pair[(left, right)]
        reverse = stats_by_pair[(right, left)]
        forward_w = forward["W_ij"]
        reverse_w = reverse["W_ij"]
        diagnostics.append({
            "left": left,
            "right": right,
            "W_left_right": forward_w,
            "W_right_left": reverse_w,
            "directional_difference": round(forward_w - reverse_w, 6)
            if forward_w is not None and reverse_w is not None else None,
            "zero_sum_consistency_gap": round(forward_w + reverse_w - 1.0, 6)
            if forward_w is not None and reverse_w is not None else None,
            "independent_reverse_samples": True,
            "uncertainty_flag": forward["uncertainty_flag"] or reverse["uncertainty_flag"],
        })

    return diagnostics


def _observed_cycles(
    labels: list[str],
    stats_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for key, stats in stats_by_pair.items():
        lower = stats["wilson95_decided"][0]
        if (
            stats["budget_complete"]
            and stats["safety_clean"]
            and not stats["low_sample"]
            and stats["W_ij"] is not None
            and stats["W_ij"] > 0.5
            and lower > 0.5
        ):
            edges[key] = {
                "winner": key[0],
                "loser": key[1],
                "W_ij": stats["W_ij"],
                "wilson95_decided": stats["wilson95_decided"],
            }

    cycles: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ordered in itertools.permutations(labels, 3):
        a, b, c = ordered
        if (a, b) not in edges or (b, c) not in edges or (c, a) not in edges:
            continue
        rotations = (ordered, (b, c, a), (c, a, b))
        key = min(rotations)
        cycles[key] = {
            "members": list(key),
            "edges": [edges[(key[0], key[1])], edges[(key[1], key[2])], edges[(key[2], key[0])]],
            "status": "observed_confident_three_cycle",
        }

    return list(edges.values()), [cycles[key] for key in sorted(cycles)]


def _profile_diagnostics(
    labels: list[str],
    stats_by_pair: dict[tuple[str, str], dict[str, Any]],
    *,
    redundancy_threshold: float,
) -> list[dict[str, Any]]:
    diagnostics = []
    for left, right in itertools.combinations(labels, 2):
        common = []
        uncertain = False
        for opponent in labels:
            if opponent in {left, right}:
                continue
            left_stats = stats_by_pair[(left, opponent)]
            right_stats = stats_by_pair[(right, opponent)]
            if left_stats["W_ij"] is None or right_stats["W_ij"] is None:
                continue
            common.append({
                "opponent": opponent,
                "left_W": left_stats["W_ij"],
                "right_W": right_stats["W_ij"],
                "difference": round(left_stats["W_ij"] - right_stats["W_ij"], 6),
            })
            uncertain = uncertain or left_stats["uncertainty_flag"] or right_stats["uncertainty_flag"]
        differences = [row["difference"] for row in common]
        l2 = math.sqrt(statistics.fmean(value * value for value in differences)) if differences else None
        mean_absolute = statistics.fmean(abs(value) for value in differences) if differences else None
        diagnostics.append({
            "left": left,
            "right": right,
            "profile": "outgoing W_i,k over common third-party opponents",
            "common_opponents": common,
            "common_opponent_count": len(common),
            "root_mean_square_distance": round(l2, 6) if l2 is not None else None,
            "mean_absolute_distance": round(mean_absolute, 6) if mean_absolute is not None else None,
            "uncertainty_flag": uncertain,
            "redundancy_threshold": redundancy_threshold,
            "redundancy_diagnostic": (
                len(common) >= 2 and not uncertain and l2 is not None and l2 <= redundancy_threshold
            ),
            "decision_role": "diagnostic only",
        })

    return diagnostics


def build_results(
    config_receipt: dict[str, Any],
    progress: dict[str, Any],
    *,
    members: list[RosterMember],
    pairs: list[OrderedPair],
    jobs: list[PayoffGameJob],
    games: int,
    min_decided: int,
    max_wilson_width: float,
    redundancy_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_progress_jobs(progress, jobs)
    persisted = progress.get("jobs") or {}
    game_rows = [persisted[job.job_key] for job in jobs if job.job_key in persisted]
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in game_rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pair_rows = [
        pair_statistics(
            pair,
            by_pair.get(pair.pair_id, []),
            planned_games=games,
            min_decided=min_decided,
            max_wilson_width=max_wilson_width,
        )
        for pair in pairs
    ]
    stats_by_pair = {(row["row_label"], row["column_label"]): row for row in pair_rows}
    labels = [member.label for member in members]
    asymmetry = _asymmetry_diagnostics(labels, stats_by_pair)
    edges, cycles = _observed_cycles(labels, stats_by_pair)
    profiles = _profile_diagnostics(
        labels,
        stats_by_pair,
        redundancy_threshold=redundancy_threshold,
    )
    expected_keys = {job.job_key for job in jobs}
    complete = set(persisted) == expected_keys
    safety_clean = all(row["safety_clean"] for row in pair_rows)
    pair_budgets_complete = all(row["budget_complete"] for row in pair_rows)
    status = "safety_failed" if progress.get("status") == "safety_failed" else "completed" if complete else "partial"
    champion = next((member for member in members if member.role == LeagueRole.CHAMPION.value), None)
    candidate = next((member for member in members if member.role == LeagueRole.CANDIDATE.value), None)
    receipts = config_receipt["config"]["input_receipts"]
    member_identities = {
        "generation_id": config_receipt["config"].get("generation_id") or None,
        "snapshot_id": config_receipt["config"].get("snapshot_id") or None,
        "champion_member_id": champion.league_member_id if champion is not None else None,
        "candidate_member_id": candidate.league_member_id if candidate is not None else None,
        "champion_checkpoint_sha256": (
            receipts.get(f"member_artifact:{champion.label}", {}).get("sha256")
            if champion is not None else None
        ),
        "candidate_checkpoint_sha256": (
            receipts.get(f"member_artifact:{candidate.label}", {}).get("sha256")
            if candidate is not None else None
        ),
        "champion_deck_sha256": (
            receipts.get(f"member_deck_binding:{champion.label}", {}).get("sha256")
            if champion is not None else None
        ),
        "candidate_deck_sha256": (
            receipts.get(f"member_deck_binding:{candidate.label}", {}).get("sha256")
            if candidate is not None else None
        ),
        "feature_policy_id": (
            champion.feature_policy_id
            if champion is not None and candidate is not None
            and champion.feature_policy_id == candidate.feature_policy_id else None
        ),
    }

    results = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config_receipt["run_id"],
        "created_at_utc": config_receipt["created_at_utc"],
        "updated_at_utc": _utc_now(),
        "status": status,
        "config_fingerprint": config_receipt["config_fingerprint"],
        "artifact_path": config_receipt["config"]["out_dir"],
        "commands": progress.get("commands") or [config_receipt["launch_command"]],
        "engine_rng_control": ENGINE_RNG_CONTRACT,
        "sample_relationship": "independent; not paired",
        "schedule": {
            "member_count": len(members),
            "ordered_pair_count": len(pairs),
            "games_per_ordered_pair": games,
            "games_expected": len(jobs),
            "games_attempted": len(game_rows),
            "distinct_pairs_only": True,
            "self_pairs_excluded": True,
            "pair_execution_interleaved": True,
            "seat_alternation": True,
            "complete": complete,
            "all_pair_budgets_complete": pair_budgets_complete,
        },
        "safety_clean": safety_clean,
        "analysis_ready": complete and safety_clean and pair_budgets_complete,
        "members": config_receipt["config"]["members"],
        "member_identities": member_identities,
        "matrix": {
            "storage": "sparse ordered rows",
            "definition": "W_ij = (wins + 0.5 * draws) / completed games",
            "rows": [row for row in pair_rows if row["attempted"] > 0],
            "unobserved_pair_count": sum(row["attempted"] == 0 for row in pair_rows),
            "low_sample_pair_count": sum(row["low_sample"] for row in pair_rows),
            "uncertainty_flag_pair_count": sum(row["uncertainty_flag"] for row in pair_rows),
        },
        "asymmetric_pairs": asymmetry,
        "confident_dominance_edges": edges,
        "observed_cycles": cycles,
        "cycle_claim": "observed only" if cycles else "none observed; no acyclicity or transitivity claim",
        "profile_distance_redundancy": profiles,
        "elapsed_wall_seconds": progress.get("elapsed_wall_seconds", 0.0),
        "decision": "diagnostic only",
        "elo_computed": False,
        "transitivity_assumed": False,
        "selection_performed": False,
        "runtime_mutated": False,
    }

    return results, game_rows


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# League ordered-pair payoff matrix",
        "",
        f"- status: **{results['status']}**",
        f"- analysis_ready: `{results['analysis_ready']}`; safety_clean: `{results['safety_clean']}`",
        "- decision: **diagnostic only**; no Elo, transitivity assumption, selection, packaging, or runtime mutation",
        f"- engine RNG: `{results['engine_rng_control']}`",
        "- reverse ordered pairs are independently sampled, not paired",
        f"- config fingerprint: `{results['config_fingerprint']}`",
        "",
        "## Exact command",
        "",
        "```text",
        results["commands"][-1]["display"],
        "```",
        "",
        "## Sparse W_ij",
        "",
        "| row i | column j | attempted/planned | W_ij | W-L-D | Wilson 95% decided | low sample | uncertainty |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in results["matrix"]["rows"]:
        lines.append(
            f"| `{row['row_label']}` | `{row['column_label']}` | {row['attempted']}/{row['planned_games']} "
            f"| {row['W_ij']} | {row['wins']}-{row['losses']}-{row['draws']} "
            f"| {row['wilson95_decided']} | {row['low_sample']} | {row['uncertainty_flag']} |"
        )
    lines.extend([
        "",
        "## Observed cycles",
        "",
        f"- claim: `{results['cycle_claim']}`",
        f"- confident three-cycles: `{len(results['observed_cycles'])}`",
        "",
        "## Profile redundancy diagnostics",
        "",
        "| left | right | common opponents | RMS distance | uncertain | redundant diagnostic |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in results["profile_distance_redundancy"]:
        lines.append(
            f"| `{row['left']}` | `{row['right']}` | {row['common_opponent_count']} "
            f"| {row['root_mean_square_distance']} | {row['uncertainty_flag']} "
            f"| {row['redundancy_diagnostic']} |"
        )
    lines.extend([
        "",
        "Profile distance and redundancy are diagnostics only; they do not remove or select members.",
    ])

    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    config_receipt: dict[str, Any],
    progress: dict[str, Any],
    results: dict[str, Any],
    game_rows: list[dict[str, Any]],
) -> None:
    write_jsonl(out_dir / "games.jsonl", game_rows)
    write_json(out_dir / "results.json", results)
    write_markdown(out_dir / "report.md", render_report(results))
    manifest = build_manifest(
        run_id=config_receipt["run_id"],
        kind="league_payoff_matrix",
        command=(progress.get("commands") or [config_receipt["launch_command"]])[-1]["argv"],
        status=results["status"],
        inputs={
            "config": "config.json",
            "config_fingerprint": config_receipt["config_fingerprint"],
            "receipts": config_receipt["config"]["input_receipts"],
        },
        outputs={
            "writer_lock": ".payoff-run.lock",
            "progress": "progress.json",
            "games": "games.jsonl",
            "results": "results.json",
            "report": "report.md",
        },
        summary={
            "status": results["status"],
            "analysis_ready": results["analysis_ready"],
            "safety_clean": results["safety_clean"],
            "games_attempted": results["schedule"]["games_attempted"],
            "games_expected": results["schedule"]["games_expected"],
            "observed_cycle_count": len(results["observed_cycles"]),
            "decision": "diagnostic only",
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)


def _validate_args(args: argparse.Namespace) -> None:
    if args.games <= 0 or args.games % 2:
        raise ValueError("--games must be a positive even number for equal alternating-seat exposure")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.min_decided < 0 or args.min_decided > args.games:
        raise ValueError("--min-decided must be between zero and --games")
    if not 0.0 < args.max_wilson_width <= 1.0:
        raise ValueError("--max-wilson-width must be in (0, 1]")
    if args.redundancy_threshold < 0.0:
        raise ValueError("--redundancy-threshold must be non-negative")


@exclusive_payoff_run
def run_payoff_matrix(args: argparse.Namespace, *, command_argv: list[str] | None = None) -> dict[str, Any]:
    _validate_args(args)
    members = load_roster(args.roster)
    generation_id = getattr(args, "generation_id", "")
    snapshot_id = getattr(args, "snapshot_id", "")
    if bool(generation_id) != bool(snapshot_id):
        raise ValueError("generation-bound payoff requires both --generation-id and --snapshot-id")
    if generation_id:
        champions = [member for member in members if member.role == LeagueRole.CHAMPION.value]
        candidates = [member for member in members if member.role == LeagueRole.CANDIDATE.value]
        if len(champions) != 1 or len(candidates) != 1:
            raise ValueError("generation-bound payoff requires exactly one champion and one candidate")
        if champions[0].feature_policy_id != candidates[0].feature_policy_id:
            raise ValueError("generation champion/candidate feature-policy identities differ")
    pairs = build_ordered_pairs(members)
    jobs = build_interleaved_jobs(pairs, games=args.games, base_seed=args.seed)
    built_members = build_member_agents(members, deck_policy=args.deck_policy)
    input_receipts = build_input_receipts(args.roster, members)
    command = _command_receipt(command_argv if command_argv is not None else sys.argv[1:])
    config_receipt = build_config_receipt(
        args,
        members=members,
        built_members=built_members,
        pairs=pairs,
        input_receipts=input_receipts,
        command=command,
    )
    out_dir = Path(args.out_dir).resolve()
    progress = prepare_progress(
        out_dir,
        config_receipt,
        jobs=jobs,
        resume=args.resume,
        command=command,
    )
    config_receipt = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    if progress.get("status") == "safety_failed":
        progress["updated_at_utc"] = _utc_now()
        write_json(out_dir / "progress.json", progress)
        results, game_rows = build_results(
            config_receipt,
            progress,
            members=members,
            pairs=pairs,
            jobs=jobs,
            games=args.games,
            min_decided=args.min_decided,
            max_wilson_width=args.max_wilson_width,
            redundancy_threshold=args.redundancy_threshold,
        )
        write_outputs(out_dir, config_receipt, progress, results, game_rows)
        raise PayoffSafetyError("prior payoff run safety failure remains a hard stop; artifacts refreshed")
    base_elapsed = float(progress.get("elapsed_wall_seconds") or 0.0)
    started = time.perf_counter()
    for job in jobs:
        if job.job_key in progress["jobs"]:
            continue
        row_member = built_members[job.pair.row_label]
        column_member = built_members[job.pair.column_label]
        telemetry = None
        row_agent = row_member.agent
        column_agent = column_member.agent
        if bool(getattr(args, "capture_public_telemetry", False)):
            telemetry = PublicGameTelemetry({
                job.row_seat: job.pair.row_label,
                1 - job.row_seat: job.pair.column_label,
            })
            row_agent = PublicTelemetryAgent(row_member.agent, job.pair.row_label, telemetry)
            column_agent = PublicTelemetryAgent(
                column_member.agent,
                job.pair.column_label,
                telemetry,
            )
        game_started = time.perf_counter()
        try:
            _reset_agent(row_member.agent, job.policy_rng_seed)
            _reset_agent(column_member.agent, job.policy_rng_seed ^ 0x5DEECE66D)
            record = run_real_meta_game(
                row_agent,
                column_agent,
                list(row_member.deck),
                list(column_member.deck),
                seat_our=job.row_seat,
                max_steps=args.max_steps,
                rng=random.Random(job.policy_rng_seed),
                game_index=job.game_index,
                rng_seed=job.policy_rng_seed,
                on_observation=telemetry.observe if telemetry is not None else None,
            )
        except Exception as exc:
            record = _crash_record(job, exc, time.perf_counter() - game_started)
        row = _game_row(job, record)
        if telemetry is not None:
            row["public_telemetry"] = telemetry.receipt()
        progress["jobs"][job.job_key] = row
        progress["elapsed_wall_seconds"] = round(base_elapsed + time.perf_counter() - started, 6)
        progress["updated_at_utc"] = _utc_now()
        if _is_safety_failure(row):
            progress["status"] = "safety_failed"
        write_json(out_dir / "progress.json", progress)
        print(
            f"[{job.game_index + 1}/{args.games}] {job.pair.row_label}->{job.pair.column_label} "
            f"seat={job.row_seat} outcome={row['outcome']}",
            flush=True,
        )
        if _is_safety_failure(row):
            results, game_rows = build_results(
                config_receipt,
                progress,
                members=members,
                pairs=pairs,
                jobs=jobs,
                games=args.games,
                min_decided=args.min_decided,
                max_wilson_width=args.max_wilson_width,
                redundancy_threshold=args.redundancy_threshold,
            )
            write_outputs(out_dir, config_receipt, progress, results, game_rows)
            raise PayoffSafetyError(
                f"payoff matrix hard-stopped on {job.pair.row_label}->{job.pair.column_label} "
                f"game {job.game_index}: {row['outcome']}"
            )

    progress["status"] = "completed"
    progress["elapsed_wall_seconds"] = round(base_elapsed + time.perf_counter() - started, 6)
    progress["updated_at_utc"] = _utc_now()
    write_json(out_dir / "progress.json", progress)
    results, game_rows = build_results(
        config_receipt,
        progress,
        members=members,
        pairs=pairs,
        jobs=jobs,
        games=args.games,
        min_decided=args.min_decided,
        max_wilson_width=args.max_wilson_width,
        redundancy_threshold=args.redundancy_threshold,
    )
    write_outputs(out_dir, config_receipt, progress, results, game_rows)
    print(json.dumps({
        "status": results["status"],
        "analysis_ready": results["analysis_ready"],
        "safety_clean": results["safety_clean"],
        "observed_cycles": len(results["observed_cycles"]),
        "artifacts": str(out_dir),
    }, indent=2, sort_keys=True))

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        required=True,
        help="JSON roster with league_member_id/label/spec/deck_binding/feature_policy_id/role",
    )
    parser.add_argument("--games", type=int, default=64, help="positive even games per distinct ordered pair")
    parser.add_argument("--seed", type=int, default=271010, help="base seed for policy RNG receipts only")
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--min-decided", type=int, default=32)
    parser.add_argument("--max-wilson-width", type=float, default=0.35)
    parser.add_argument("--redundancy-threshold", type=float, default=0.05)
    parser.add_argument(
        "--capture-public-telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "persist aggregate actor-visible/public action, tempo, prize, and focal-evolution metrics "
            "for the same payoff games; never persist raw observations or unselected options"
        ),
    )
    parser.add_argument("--out-dir", required=True, help="stable durable research output directory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-id", default="", help="league generation receipt identity")
    parser.add_argument("--snapshot-id", default="", help="frozen league snapshot identity")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    final_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(final_argv)
    try:
        run_payoff_matrix(args, command_argv=final_argv)
    except PayoffSafetyError as exc:
        print(f"SAFETY HARD STOP: {exc}", file=sys.stderr)

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
