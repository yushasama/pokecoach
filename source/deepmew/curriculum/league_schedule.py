"""Pure league schedule identities and fixed-budget PFSP allocation.

This module owns no engine or training behavior.  It turns an explicit league
cell into durable identities, derives worker-count-independent game seeds, and
allocates an exact game budget from empirical candidate win rates.  PFSP is an
opt-in scheduler only; callers preserve uniform scheduling when it is disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar
from urllib.parse import quote

from deepmew.curriculum.league_registry import sha256_path


CELL_ID_VERSION = "deepmew.league_cell.v2"
GAME_ID_VERSION = "deepmew.league_game.v2"
SCHEDULE_VERSION = "deepmew.league_schedule.v2"
UNKNOWN_WIN_RATE = 0.5
MAX_ENGINE_SEED = (1 << 31) - 1


def _encoded(value: Any) -> str:
    return quote(str(value), safe="")


@dataclass(frozen=True)
class LeagueCell:
    """One unique opponent/deck/stage scheduling cell."""

    opponent_source: str
    opponent_member: str
    target_stage: str
    source_bucket: str
    deck_hash: str
    pilot: str
    fixed_deck_ref: str = ""
    opponent_policy_id: str = ""
    opponent_league_member_id: str = ""

    def __post_init__(self) -> None:
        required = {
            "opponent_source": self.opponent_source,
            "opponent_member": self.opponent_member,
            "target_stage": self.target_stage,
            "source_bucket": self.source_bucket,
            "deck_hash": self.deck_hash,
            "pilot": self.pilot,
        }
        missing = [name for name, value in required.items() if not str(value)]
        if missing:
            raise ValueError(f"league cell fields must be non-empty: {missing}")

    @property
    def cell_id(self) -> str:
        fields = (
            ("opponent_source", self.opponent_source),
            ("opponent_member", self.opponent_member),
            ("target_stage", self.target_stage),
            ("source_bucket", self.source_bucket),
            ("deck_hash", self.deck_hash),
            ("pilot", self.pilot),
            ("fixed_deck_ref", self.fixed_deck_ref),
            ("opponent_policy_id", self.opponent_policy_id),
            ("opponent_league_member_id", self.opponent_league_member_id),
        )

        return CELL_ID_VERSION + "|" + "|".join(
            f"{name}={_encoded(value)}" for name, value in fields
        )

    @property
    def legacy_key(self) -> str:
        """Pre-v1 lookup key, accepted only as explicitly labelled evidence fallback."""

        return f"{self.deck_hash}:{self.pilot}"

    def to_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_id_version": CELL_ID_VERSION,
            "opponent_source": self.opponent_source,
            "opponent_member": self.opponent_member,
            "target_stage": self.target_stage,
            "source_bucket": self.source_bucket,
            "deck_hash": self.deck_hash,
            "pilot": self.pilot,
            "fixed_deck_ref": self.fixed_deck_ref or None,
            "opponent_policy_id": self.opponent_policy_id or None,
            "opponent_league_member_id": self.opponent_league_member_id or None,
        }


def canonical_game_id(
    *, run_id: str, round_id: str, cell_id: str, game_ordinal: int,
) -> str:
    """Return a versioned game ID containing run, round, cell, and cell-local ordinal."""

    if not run_id or not round_id or not cell_id:
        raise ValueError("run_id, round_id, and cell_id must be non-empty")
    if game_ordinal < 0:
        raise ValueError("game_ordinal must be non-negative")

    return (
        f"{GAME_ID_VERSION}|run_id={_encoded(run_id)}|round_id={_encoded(round_id)}"
        f"|cell_id={_encoded(cell_id)}|game_ordinal={game_ordinal:06d}"
    )


def deterministic_game_seed(run_seed: int, game_id: str) -> int:
    """Derive a policy-RNG seed from stable game identity, never worker placement."""

    payload = f"{SCHEDULE_VERSION}\0{int(run_seed)}\0{game_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    return value % MAX_ENGINE_SEED


@dataclass(frozen=True)
class ScheduledGame:
    cell: LeagueCell
    cell_ordinal: int
    game_ordinal: int
    schedule_ordinal: int
    game_id: str
    seed: int


def build_game_schedule(
    cells_and_counts: Sequence[tuple[LeagueCell, int]],
    *,
    run_id: str,
    round_id: str,
    run_seed: int,
) -> tuple[ScheduledGame, ...]:
    """Expand unique cells into a deterministic global game schedule."""

    cell_ids = [cell.cell_id for cell, _count in cells_and_counts]
    if len(cell_ids) != len(set(cell_ids)):
        duplicates = sorted({cell_id for cell_id in cell_ids if cell_ids.count(cell_id) > 1})
        raise ValueError(f"duplicate league cells are not schedulable: {duplicates}")

    games: list[ScheduledGame] = []
    for cell_ordinal, (cell, count) in enumerate(cells_and_counts):
        if int(count) < 0:
            raise ValueError(f"game count must be non-negative for {cell.cell_id}")
        for game_ordinal in range(int(count)):
            game_id = canonical_game_id(
                run_id=run_id,
                round_id=round_id,
                cell_id=cell.cell_id,
                game_ordinal=game_ordinal,
            )
            games.append(ScheduledGame(
                cell=cell,
                cell_ordinal=cell_ordinal,
                game_ordinal=game_ordinal,
                schedule_ordinal=len(games),
                game_id=game_id,
                seed=deterministic_game_seed(run_seed, game_id),
            ))

    game_ids = [game.game_id for game in games]
    if len(game_ids) != len(set(game_ids)):
        raise AssertionError("canonical league game IDs collided")

    return tuple(games)


def games_for_worker(
    schedule: Sequence[ScheduledGame], *, worker_count: int, worker_index: int,
) -> tuple[ScheduledGame, ...]:
    """Partition whole cells without changing any game identity or seed."""

    if worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must be within worker_count")

    return tuple(game for game in schedule if game.cell_ordinal % worker_count == worker_index)


T = TypeVar("T")


def select_stage_opponents(
    opponents: Sequence[T], *, pilot: str, target_stage: str, exploiter_top: int,
) -> list[T]:
    """Keep exploiters on the target stage's own top decks, not carried low bands."""

    if pilot != "exploiter":
        return list(opponents)
    if exploiter_top <= 0:
        return []
    target_opponents = [
        opponent for opponent in opponents
        if str(getattr(opponent, "bucket", "")) == target_stage
    ]

    return target_opponents[:exploiter_top]


@dataclass(frozen=True)
class EmpiricalWinRate:
    win_rate: float | None = None
    status: str = "unknown"
    games: int | None = None
    source_key: str = ""

    @property
    def is_usable(self) -> bool:
        return self.status in {"observed", "legacy_observed"} and self.win_rate is not None


@dataclass(frozen=True)
class PfspEvidence:
    exact: Mapping[str, EmpiricalWinRate]
    legacy: Mapping[str, EmpiricalWinRate]
    source_status: str = "available"
    source_path: str = ""
    ignored_direct_weights: bool = False

    @classmethod
    def empty(cls, status: str = "not_configured") -> "PfspEvidence":
        return cls(exact={}, legacy={}, source_status=status)

    def estimate_for(self, cell: LeagueCell) -> EmpiricalWinRate:
        exact = self.exact.get(cell.cell_id)
        if exact is not None:
            return exact
        legacy = self.legacy.get(cell.legacy_key)
        if legacy is not None:
            status = "legacy_observed" if legacy.status == "observed" else legacy.status

            return replace(legacy, status=status, source_key=cell.legacy_key)

        return EmpiricalWinRate(status="unknown", source_key=cell.cell_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "source_path": self.source_path or None,
            "exact_estimates": len(self.exact),
            "legacy_estimates": len(self.legacy),
            "ignored_direct_weights": self.ignored_direct_weights,
        }


def _parsed_estimate(value: Any, *, source_key: str, inherited_stale: bool = False) -> EmpiricalWinRate:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        win_rate = float(value)
        status = "stale" if inherited_stale else "observed"
        games = None
    elif isinstance(value, Mapping):
        raw_win_rate = value.get("win_rate", value.get("winrate"))
        win_rate = float(raw_win_rate) if raw_win_rate is not None else None
        is_stale = inherited_stale or bool(value.get("stale", False))
        status = "stale" if is_stale else str(value.get("status") or ("observed" if win_rate is not None else "unknown"))
        raw_games = value.get("games", value.get("n"))
        games = int(raw_games) if raw_games is not None else None
    else:
        return EmpiricalWinRate(status="unknown", source_key=source_key)
    if win_rate is not None and not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"empirical win rate outside [0,1] for {source_key}: {win_rate}")

    return EmpiricalWinRate(win_rate=win_rate, status=status, games=games, source_key=source_key)


def parse_pfsp_evidence(data: Mapping[str, Any], *, source_path: str = "") -> PfspEvidence:
    """Parse empirical cell rates while labelling legacy and stale evidence explicitly."""

    exact: dict[str, EmpiricalWinRate] = {}
    legacy: dict[str, EmpiricalWinRate] = {}
    inherited_stale = bool(data.get("stale", False))

    def add(key: str, value: Any) -> None:
        estimate = _parsed_estimate(value, source_key=key, inherited_stale=inherited_stale)
        destination = exact if key.startswith(f"{CELL_ID_VERSION}|") else legacy
        destination[key] = estimate

    for field_name in ("cell_win_rates", "empirical_win_rates", "win_rates"):
        values = data.get(field_name)
        if isinstance(values, Mapping):
            for key, value in values.items():
                add(str(key), value)

    cells = data.get("cells")
    if isinstance(cells, Mapping):
        for key, value in cells.items():
            add(str(key), value)
    elif isinstance(cells, list):
        for row in cells:
            if isinstance(row, Mapping) and row.get("cell_id"):
                add(str(row["cell_id"]), row)

    for row in data.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("cell_id") or "")
        if not key and row.get("deck_hash") and row.get("pilot"):
            key = f"{row['deck_hash']}:{row['pilot']}"
        if not key:
            continue
        if row.get("win_rate") is not None or row.get("winrate") is not None:
            add(key, row)
            continue
        wins = int(row.get("wins") or 0)
        losses = int(row.get("losses") or 0)
        decided = wins + losses
        value = dict(row)
        value["win_rate"] = wins / decided if decided else None
        value["games"] = int(row.get("games") or decided)
        add(key, value)

    ignored_direct_weights = isinstance(data.get("pfsp_weights"), Mapping)
    source_status = "available" if exact or legacy else "empirical_rates_missing"

    return PfspEvidence(
        exact=exact,
        legacy=legacy,
        source_status=source_status,
        source_path=source_path,
        ignored_direct_weights=ignored_direct_weights,
    )


def load_pfsp_evidence(path: str | Path) -> PfspEvidence:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("PFSP evidence JSON must contain an object")

    return parse_pfsp_evidence(data, source_path=str(source.resolve()))


@dataclass(frozen=True)
class PfspCellAllocation:
    cell_id: str
    games: int
    empirical_win_rate: float | None
    effective_win_rate: float | None
    estimate_status: str
    score: float | None
    source_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "games": self.games,
            "empirical_win_rate": self.empirical_win_rate,
            "effective_win_rate": self.effective_win_rate,
            "estimate_status": self.estimate_status,
            "pfsp_score": self.score,
            "source_key": self.source_key or None,
        }


@dataclass(frozen=True)
class PfspAllocation:
    enabled: bool
    budget: int
    min_games: int
    floor: float
    power: float
    cells: tuple[PfspCellAllocation, ...]

    @property
    def games_by_cell(self) -> dict[str, int]:
        return {cell.cell_id: cell.games for cell in self.cells}

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "budget": self.budget,
            "allocated_games": sum(cell.games for cell in self.cells),
            "min_games": self.min_games,
            "floor": self.floor,
            "power": self.power,
            "cells": [cell.to_json() for cell in self.cells],
        }


def _mapping_estimate(values: Mapping[str, Any], cell: LeagueCell) -> EmpiricalWinRate:
    if cell.cell_id not in values:
        return EmpiricalWinRate(status="unknown", source_key=cell.cell_id)

    return _parsed_estimate(values[cell.cell_id], source_key=cell.cell_id)


def allocate_pfsp_games(
    cells: Sequence[LeagueCell],
    evidence: PfspEvidence | Mapping[str, Any] | None,
    *,
    enabled: bool,
    base_games: int,
    min_games: int,
    floor: float,
    power: float,
    unknown_win_rate: float = UNKNOWN_WIN_RATE,
) -> PfspAllocation:
    """Allocate exactly ``base_games * len(cells)`` games with integer floors.

    The PFSP score is ``max(floor, (1 - effective_win_rate) ** power)``.
    Unknown and stale estimates use the explicit neutral fallback.  Largest
    remainder rounding makes the integer allocation deterministic and exact.
    """

    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        duplicates = sorted({cell_id for cell_id in cell_ids if cell_ids.count(cell_id) > 1})
        raise ValueError(f"duplicate PFSP cells: {duplicates}")
    if base_games < 0:
        raise ValueError("base_games must be non-negative")
    budget = int(base_games) * len(cells)
    if not enabled:
        allocations = tuple(PfspCellAllocation(
            cell_id=cell.cell_id,
            games=int(base_games),
            empirical_win_rate=None,
            effective_win_rate=None,
            estimate_status="disabled",
            score=None,
            source_key="",
        ) for cell in cells)

        return PfspAllocation(
            enabled=False,
            budget=budget,
            min_games=int(min_games),
            floor=float(floor),
            power=float(power),
            cells=allocations,
        )
    if min_games < 0:
        raise ValueError("min_games must be non-negative")
    if int(min_games) * len(cells) > budget:
        raise ValueError(
            f"PFSP minimum floors require {int(min_games) * len(cells)} games, budget is {budget}"
        )
    if floor < 0.0:
        raise ValueError("PFSP floor must be non-negative")
    if power <= 0.0:
        raise ValueError("PFSP power must be positive")
    if not 0.0 <= unknown_win_rate <= 1.0:
        raise ValueError("unknown_win_rate must be within [0,1]")
    if not cells:
        return PfspAllocation(
            enabled=True,
            budget=0,
            min_games=int(min_games),
            floor=float(floor),
            power=float(power),
            cells=(),
        )

    estimates: list[EmpiricalWinRate] = []
    scores: list[float] = []
    effective_rates: list[float] = []
    for cell in cells:
        if isinstance(evidence, PfspEvidence):
            estimate = evidence.estimate_for(cell)
        elif isinstance(evidence, Mapping):
            estimate = _mapping_estimate(evidence, cell)
        else:
            estimate = EmpiricalWinRate(status="unknown", source_key=cell.cell_id)
        effective_rate = float(estimate.win_rate) if estimate.is_usable else float(unknown_win_rate)
        score = max(float(floor), (1.0 - effective_rate) ** float(power))
        estimates.append(estimate)
        effective_rates.append(effective_rate)
        scores.append(score)

    if sum(scores) <= 0.0:
        scores = [1.0 for _cell in cells]
    remaining = budget - int(min_games) * len(cells)
    score_total = sum(scores)
    exact_extras = [remaining * score / score_total for score in scores]
    integer_extras = [math.floor(value) for value in exact_extras]
    leftover = remaining - sum(integer_extras)
    remainder_order = sorted(
        range(len(cells)),
        key=lambda index: (-(exact_extras[index] - integer_extras[index]), cells[index].cell_id),
    )
    for index in remainder_order[:leftover]:
        integer_extras[index] += 1

    allocations = tuple(PfspCellAllocation(
        cell_id=cell.cell_id,
        games=int(min_games) + integer_extras[index],
        empirical_win_rate=estimates[index].win_rate,
        effective_win_rate=effective_rates[index],
        estimate_status=estimates[index].status,
        score=scores[index],
        source_key=estimates[index].source_key,
    ) for index, cell in enumerate(cells))
    if sum(allocation.games for allocation in allocations) != budget:
        raise AssertionError("PFSP allocation did not preserve the fixed budget")

    return PfspAllocation(
        enabled=True,
        budget=budget,
        min_games=int(min_games),
        floor=float(floor),
        power=float(power),
        cells=allocations,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _artifact_identity(path: Path, *, bundle: bool) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "status": "not_found", "artifact_sha256": None}
    if resolved.is_file():
        return {
            "path": str(resolved),
            "status": "exact",
            "artifact_sha256": _sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    if not bundle:
        return {"path": str(resolved), "status": "no_policy_artifacts", "artifact_sha256": None}

    return {
        "path": str(resolved),
        "status": "exact",
        "artifact_sha256": sha256_path(resolved, purpose="policy bundle"),
        "hash_contract": "deepmew.directory.v1",
    }


def resolve_feature_policy_id(
    spec: str,
    *,
    explicit: str = "",
    checkpoint_default: str = "bucket_exploiter",
) -> str:
    """Return the exact categorical policy ID used by a model feature encoder."""

    if explicit:
        return explicit
    kind, separator, target = str(spec).partition(":")
    if kind == "bundle" and separator and target:
        manifest_path = Path(target) / "r5_policy_weights.manifest.json"
        if not manifest_path.is_file():
            return "not_instrumented"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        return str(manifest.get("policy_id") or "not_instrumented")
    if (kind == "checkpoint" and separator and target) or spec == "exploiter":
        return checkpoint_default

    return "not_applicable"


def policy_identity(
    spec: str,
    *,
    champion: str | Path = "",
    replay_tables: str | Path = "",
    feature_policy_id: str = "",
    checkpoint_feature_policy_id: str = "bucket_exploiter",
    named_policy_source: str | Path = "",
) -> dict[str, Any]:
    """Describe the exact live checkpoint/bundle/table identity when one exists."""

    kind, separator, target = str(spec).partition(":")
    if kind == "bundle" and separator and target:
        artifact = _artifact_identity(Path(target), bundle=True)
        policy_kind = "bundle"
    elif kind == "checkpoint" and separator and target:
        artifact = _artifact_identity(Path(target), bundle=False)
        policy_kind = "checkpoint"
    elif spec == "exploiter":
        artifact = _artifact_identity(Path(champion), bundle=False) if str(champion) else {
            "path": None, "status": "champion_not_configured", "artifact_sha256": None,
        }
        policy_kind = "checkpoint"
    elif spec == "replay":
        artifact = _artifact_identity(Path(replay_tables), bundle=False) if str(replay_tables) else {
            "path": None, "status": "replay_tables_not_configured", "artifact_sha256": None,
        }
        policy_kind = "replay_table"
    else:
        artifact = (
            _artifact_identity(Path(named_policy_source), bundle=True)
            if str(named_policy_source)
            else {"path": None, "status": "named_policy", "artifact_sha256": None}
        )
        policy_kind = "named_policy"
    suffix = artifact.get("artifact_sha256") or artifact.get("status")

    return {
        "spec": str(spec),
        "kind": policy_kind,
        "policy_id": f"{spec}#{suffix}",
        "feature_policy_id": resolve_feature_policy_id(
            spec,
            explicit=feature_policy_id,
            checkpoint_default=checkpoint_feature_policy_id,
        ),
        **artifact,
    }
