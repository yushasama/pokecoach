"""League data generation from a YAML roster (curriculum + fixed-deck sparring).

Reads a league config (see .research/curriculum/league.example.yaml) instead of a
long CLI: declares the policy being trained (our side), and a roster of opponents,
each with a brain and the deck(s) it pilots. Generates outcome-stamped refgame
shards (reusing run_r18_curriculum_datagen.play_recorded_game) for the AWR trainer.

Opponent shapes:
  - BAND (`decks: band`): the brain pilots the curriculum's per-ELO-band decks; PFSP
    can weight games toward the band decks that beat us.
  - FIXED (`decks: [bundle:<dir>, ...]`): the brain pilots its own deck -- the ~600
    ELO reference agents, bd27, our current Starmie policy on its coherent deck.

INVARIANT #2 (fixed frozen population), #5 (filter/target, not oracle), #8 (offline).

Example:
  python scripts/run_league_datagen.py --config .research/curriculum/league.example.yaml \
      --out-dir D:/DeepMew_artifacts/datasets/r19/league
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main  # noqa: E402
from deepmew.bucket_arena import build_stage, entry_from_json  # noqa: E402
from deepmew.curriculum.league_config import (  # noqa: E402
    LeagueConfig, OurSide, load_league_config,
)
from deepmew.curriculum.league_schedule import (  # noqa: E402
    CELL_ID_VERSION,
    GAME_ID_VERSION,
    SCHEDULE_VERSION,
    LeagueCell,
    PfspEvidence,
    PfspCellAllocation,
    allocate_pfsp_games,
    build_game_schedule,
    games_for_worker,
    load_pfsp_evidence,
    policy_identity,
    select_stage_opponents,
)
from deepmew.curriculum.league_registry import (  # noqa: E402
    LeagueRole,
    load_registry,
    sha256_path,
)
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl.observation import ObservationEncoder  # noqa: E402
import run_r17_bucket_arena as arena  # noqa: E402
import run_r18_curriculum_datagen as dg  # noqa: E402

SCHEMA_VERSION = "deepmew.r19.league_datagen.v2"
FIXED_STAGE = "fixed"
FIXED_SOURCE_BUCKET = "not_applicable"


@dataclass(frozen=True)
class LeagueTarget:
    cards: tuple[int, ...]
    archetype: str
    engine_bucket: str
    cell: LeagueCell


@dataclass(frozen=True)
class CellPlan:
    opponent: Any
    target: LeagueTarget
    allocation: PfspCellAllocation
    opponent_policy_identity: dict[str, Any]


def _read_deck_csv(path: Path) -> list[int]:
    return [int(v) for v in path.read_text(encoding="utf-8").split()][:60]


def resolve_deck_ref(ref: str) -> list[int]:
    """Deck ref -> 60 card ids. `root`|`bundle:<dir>`|`deck:<path>`."""
    if ref == "root":
        return [int(v) for v in main.read_deck_csv()][:60]
    kind, _, target = ref.partition(":")
    if kind == "bundle" and target:
        return _read_deck_csv(Path(target) / "deck.csv")
    if kind == "deck" and target:
        return _read_deck_csv(Path(target))
    raise ValueError(f"unknown deck ref {ref!r} (use root | bundle:<dir> | deck:<path>)")


def _policy_artifact_path(reference: str) -> Path:
    kind, separator, target = reference.partition(":")
    if separator and kind in {"checkpoint", "bundle"} and target:
        return Path(target).resolve(strict=True)
    raise ValueError("generation-bound learner pilot must be checkpoint:<path> or bundle:<dir>")


def _deck_artifact_path(reference: str) -> Path:
    if reference == "root":
        return (ROOT / "deck.csv").resolve(strict=True)
    kind, separator, target = reference.partition(":")
    if separator and kind == "deck" and target:
        return Path(target).resolve(strict=True)
    if separator and kind == "bundle" and target:
        return (Path(target) / "deck.csv").resolve(strict=True)
    raise ValueError("generation-bound learner deck must be root, deck:<path>, or bundle:<dir>")


def _generation_binding(
    cfg: LeagueConfig,
    *,
    generation_id: str,
    snapshot_id: str,
    registry_path: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    requested = any((generation_id, snapshot_id, registry_path))
    if not requested:
        return None, {}, {}
    if not all((generation_id, snapshot_id, registry_path, cfg.our.league_member_id)):
        raise ValueError(
            "generation-bound datagen requires generation_id, snapshot_id, registry_path, "
            "and our.league_member_id"
        )
    registry_file = Path(registry_path).resolve(strict=True)
    registry = load_registry(registry_file)
    champion = registry.active_champion()
    if champion is None or champion.league_member_id != cfg.our.league_member_id:
        raise ValueError("generation datagen learner is not the active registry champion")
    if champion.role is not LeagueRole.CHAMPION:
        raise ValueError("generation datagen learner does not have champion role")
    if champion.feature_policy_id != cfg.our.effective_feature_policy_id:
        raise ValueError("generation datagen feature-policy identity differs from registry")
    snapshot = next(
        (item for item in registry.snapshots if item.snapshot_id == snapshot_id),
        None,
    )
    if snapshot is None:
        raise ValueError(f"generation datagen snapshot is not registered: {snapshot_id}")
    frozen = next(
        (item for item in snapshot.members if item.league_member_id == champion.league_member_id),
        None,
    )
    if frozen is None or frozen.role is not LeagueRole.CHAMPION:
        raise ValueError("generation snapshot does not freeze the learner as champion")
    policy_path = _policy_artifact_path(cfg.our.pilot)
    deck_path = _deck_artifact_path(cfg.our.deck)
    policy_sha = sha256_path(policy_path, purpose="generation learner artifact")
    deck_sha = sha256_path(deck_path, purpose="generation learner deck")
    if policy_sha != champion.artifact_sha256 or deck_sha != champion.deck_sha256:
        raise ValueError("generation datagen learner bytes differ from the registered champion")
    policy_receipt = {
        "path": str(policy_path),
        "sha256": policy_sha,
        "league_member_id": champion.league_member_id,
    }
    deck_receipt = {"path": str(deck_path), "sha256": deck_sha}
    binding = {
        "generation_id": generation_id,
        "snapshot_id": snapshot_id,
        "registry_path": str(registry_file),
        "registry_sha256_at_start": sha256_path(registry_file, purpose="league registry"),
        "champion_member_id": champion.league_member_id,
        "champion_artifact_sha256": policy_sha,
        "champion_deck_sha256": deck_sha,
        "feature_policy_id": champion.feature_policy_id,
        "verified": True,
    }

    return binding, policy_receipt, deck_receipt


def _schedule_fingerprint(
    *,
    run_id: str,
    round_id: str,
    cells: list[dict[str, Any]],
) -> str:
    payload = {
        "schedule_version": SCHEDULE_VERSION,
        "run_id": run_id,
        "round_id": round_id,
        "cells": sorted(cells, key=lambda item: str(item.get("cell_id") or "")),
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _shard_receipts(shard_dir: Path) -> list[dict[str, Any]]:
    receipts = []
    for path in sorted(shard_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        if rows <= 0:
            raise ValueError(f"generation datagen shard is empty: {path}")
        receipts.append({
            "path": str(path.resolve()),
            "sha256": sha256_path(path, purpose="league datagen shard"),
            "rows": rows,
        })
    if not receipts:
        raise ValueError("league datagen produced no shard receipts")

    return receipts


def build_pilot(pilot: str, deck: list[int], deck_hash: str, *, cfg: LeagueConfig,
                band_tables: Any, bucket: str = "", archetype: str = "Unknown",
                feature_policy_id: str = "") -> Any:
    """Dispatch a brain onto a deck. bundle:<dir> -> DistilledPolicy runtime; everything
    else -> the shared datagen pilot builder (heuristic/replay/exploiter/checkpoint)."""
    kind, _, target = pilot.partition(":")
    if kind == "bundle" and target:
        return arena.BundleAgent(
            Path(target).resolve(),
            Path(target).name,
            feature_policy_id=feature_policy_id,
            engine_deck=list(deck),
        )
    return dg._build_pilot(pilot, deck, deck_hash, champion=cfg.champion or dg.DEFAULT_CHAMPION,
                           band_tables=band_tables, bucket=bucket, archetype=archetype,
                           feature_policy_id=feature_policy_id)


def live_feature_binding(agent: Any) -> dict[str, str | None]:
    current = getattr(agent, "inner", agent)
    policy = getattr(current, "_policy", None) or getattr(current, "policy", None)
    if policy is None:
        return {
            "status": "not_applicable",
            "feature_policy_id": None,
            "feature_deck_hash": None,
        }

    return {
        "status": "instrumented",
        "feature_policy_id": str(getattr(policy, "policy_id", "")) or None,
        "feature_deck_hash": str(getattr(policy, "deck_hash", "")) or None,
    }


def _targets_for(
    opp,
    pools,
    shells,
    cfg,
    *,
    opponent_policy_id: str = "",
    opponent_league_member_id: str = "",
):
    """Yield identity-complete targets for one league member."""
    if opp.is_band:
        top_n = opp.top_n if opp.top_n is not None else cfg.top_n
        for stage in cfg.stages:
            etop = opp.exploiter_top if opp.exploiter_top is not None else cfg.exploiter_top
            if opp.pilot == "exploiter":
                opponents = build_stage(
                    pools,
                    stage,
                    top_n=top_n,
                    carry_n=0,
                    include_previous=False,
                )
            else:
                opponents = build_stage(pools, stage, top_n=top_n, carry_n=cfg.carry_n)
            opponents = select_stage_opponents(
                opponents, pilot=opp.pilot, target_stage=stage, exploiter_top=etop,
            )
            for s in opponents:
                shell = shells.get(s.deck_hash, "Unknown")
                yield LeagueTarget(
                    cards=tuple(s.card_ids),
                    archetype=shell,
                    engine_bucket=stage,
                    cell=LeagueCell(
                        opponent_source="band",
                        opponent_member=opp.name,
                        target_stage=stage,
                        source_bucket=s.bucket,
                        deck_hash=s.deck_hash,
                        pilot=opp.pilot,
                        opponent_policy_id=opponent_policy_id,
                        opponent_league_member_id=opponent_league_member_id,
                    ),
                )
    else:
        for ref in opp.deck_refs:
            cards = resolve_deck_ref(ref)
            deck_hash = normalized_deck_hash(cards)
            yield LeagueTarget(
                cards=tuple(cards),
                archetype=opp.name,
                engine_bucket=opp.name,
                cell=LeagueCell(
                    opponent_source="fixed_deck",
                    opponent_member=opp.name,
                    target_stage=FIXED_STAGE,
                    source_bucket=FIXED_SOURCE_BUCKET,
                    deck_hash=deck_hash,
                    pilot=opp.pilot,
                    fixed_deck_ref=ref,
                    opponent_policy_id=opponent_policy_id,
                    opponent_league_member_id=opponent_league_member_id,
                ),
            )


def apply_our_overrides(
    cfg: LeagueConfig,
    *,
    pilot: str = "",
    deck: str = "",
    label: str = "",
    feature_policy_id: str = "",
    league_member_id: str = "",
) -> LeagueConfig:
    """Return cfg with optional CLI overrides for the training side only."""
    if not (pilot or deck or label or feature_policy_id or league_member_id):
        return cfg
    our = OurSide(
        pilot=pilot or cfg.our.pilot,
        deck=deck or cfg.our.deck,
        label=label or cfg.our.label,
        feature_policy_id=feature_policy_id or cfg.our.feature_policy_id,
        league_member_id=league_member_id or cfg.our.league_member_id,
    )

    return replace(cfg, our=our)


def _inferred_schedule_identity(out_dir: Path) -> tuple[str, str]:
    name = out_dir.name
    if name.startswith("r") and name.endswith("_data") and name[1:-5].isdigit():
        return out_dir.parent.name, name[:-5]

    return name, "round-0000"


def _validate_identity_shards(out_dir: Path) -> dict[str, int]:
    game_cells: dict[str, str] = {}
    decision_ids: set[str] = set()
    rows = 0
    for path in sorted((out_dir / "shards").glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                game_id = str(row.get("game_id") or "")
                decision_id = str(row.get("decision_id") or "")
                cell_id = str(row.get("cell_id") or "")
                if not game_id or not decision_id or not cell_id:
                    raise ValueError(f"{path}:{line_number}: missing league identity fields")
                if decision_id in decision_ids:
                    raise ValueError(f"duplicate league decision_id: {decision_id}")
                previous_cell = game_cells.setdefault(game_id, cell_id)
                if previous_cell != cell_id:
                    raise ValueError(f"game_id {game_id} spans multiple league cells")
                decision_ids.add(decision_id)
                rows += 1

    return {
        "rows": rows,
        "unique_game_ids": len(game_cells),
        "unique_decision_ids": len(decision_ids),
        "duplicate_decision_ids": 0,
    }


def run(cfg: LeagueConfig, out_dir: Path, max_steps: int = 5000,
        *, shard_count: int = 1, shard_index: int = 0,
        run_id: str = "", round_id: str = "",
        generation_id: str = "", snapshot_id: str = "",
        registry_path: str = "") -> dict[str, Any]:
    started = time.perf_counter()
    inferred_run_id, inferred_round_id = _inferred_schedule_identity(out_dir)
    run_id = run_id or inferred_run_id
    round_id = round_id or inferred_round_id
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    encoder = ObservationEncoder()
    generation_binding, our_artifact_binding, our_deck_binding = _generation_binding(
        cfg,
        generation_id=generation_id,
        snapshot_id=snapshot_id,
        registry_path=registry_path,
    )

    band_tables = None
    if cfg.needs_replay():
        if not cfg.replay_tables:
            raise SystemExit("a `replay` opponent is enabled but config has no `replay_tables`")
        from deepmew.curriculum.replay_pilots import BandReplayTables  # noqa: PLC0415
        band_tables = BandReplayTables.from_json(json.loads(Path(cfg.replay_tables).read_text(encoding="utf-8")))

    our_deck = resolve_deck_ref(cfg.our.deck)
    our_hash = normalized_deck_hash(our_deck)
    our_feature_policy_id = cfg.our.effective_feature_policy_id
    our_pilot = build_pilot(
        cfg.our.pilot,
        our_deck,
        our_hash,
        cfg=cfg,
        band_tables=band_tables,
        feature_policy_id=our_feature_policy_id,
    )
    our_policy_identity = policy_identity(
        cfg.our.pilot,
        champion=cfg.champion,
        replay_tables=cfg.replay_tables,
        feature_policy_id=our_feature_policy_id,
        named_policy_source=ROOT / "deepmew" / "arena",
    )
    if our_artifact_binding:
        our_policy_identity = {**our_policy_identity, **our_artifact_binding}
    our_feature_binding = live_feature_binding(our_pilot)
    live_our_feature_policy_id = our_feature_binding["feature_policy_id"]
    if live_our_feature_policy_id is not None and live_our_feature_policy_id != our_feature_policy_id:
        raise ValueError(
            f"learner feature-policy mismatch: configured {our_feature_policy_id!r}, "
            f"live {live_our_feature_policy_id!r}"
        )

    pools_payload = json.loads(Path(cfg.pools).read_text(encoding="utf-8")) if getattr(cfg, "pools", "") else None
    # pools path is injected onto cfg by the CLI (kept off the frozen dataclass).
    pools = {b: [entry_from_json(r) for r in rows] for b, rows in pools_payload["pools"].items()}
    shells = {r["deck_hash"]: str(r.get("shell") or "Unknown")
              for rows in pools_payload["pools"].values() for r in rows}

    if cfg.pfsp.enabled and not cfg.pfsp.weights_path:
        raise ValueError("PFSP is enabled but pfsp.weights_path is not configured")
    if cfg.pfsp.enabled:
        pfsp_evidence = load_pfsp_evidence(cfg.pfsp.weights_path)
    else:
        pfsp_evidence = PfspEvidence.empty("disabled")

    cell_plans: list[CellPlan] = []
    pfsp_allocations: dict[str, dict[str, Any]] = {}
    for opp in cfg.enabled_opponents():
        live_identity = policy_identity(
            opp.pilot,
            champion=cfg.champion,
            replay_tables=cfg.replay_tables,
            feature_policy_id=opp.feature_policy_id,
            named_policy_source=ROOT / "deepmew" / "arena",
        )
        targets = list(_targets_for(
            opp,
            pools,
            shells,
            cfg,
            opponent_policy_id=live_identity["policy_id"],
            opponent_league_member_id=opp.league_member_id,
        ))
        base_games = opp.games if opp.games is not None else cfg.games
        allocation = allocate_pfsp_games(
            [target.cell for target in targets],
            pfsp_evidence,
            enabled=cfg.pfsp.enabled,
            base_games=int(base_games),
            min_games=cfg.pfsp.min_games,
            floor=cfg.pfsp.floor,
            power=cfg.pfsp.power,
        )
        pfsp_allocations[opp.name] = allocation.to_json()
        for target, cell_allocation in zip(targets, allocation.cells, strict=True):
            cell_plans.append(CellPlan(
                opponent=opp,
                target=target,
                allocation=cell_allocation,
                opponent_policy_identity=live_identity,
            ))

    schedule = build_game_schedule(
        [(plan.target.cell, plan.allocation.games) for plan in cell_plans],
        run_id=run_id,
        round_id=round_id,
        run_seed=cfg.seed,
    )

    shard_rows: list[dict[str, Any]] = []
    shard_idx = 0
    totals = {
        "games": 0,
        "rows": 0,
        "our_wins": 0,
        "completed": 0,
        "crashes": 0,
        "timeouts": 0,
        "illegal": 0,
        "incomplete": 0,
    }
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    assigned_schedule = games_for_worker(
        schedule, worker_count=shard_count, worker_index=shard_index,
    )
    games_by_cell: dict[str, list[Any]] = {}
    for game in assigned_schedule:
        games_by_cell.setdefault(game.cell.cell_id, []).append(game)
    per_opp: dict[str, dict[str, int]] = {}
    emitted_decision_ids: set[str] = set()
    opponent_feature_bindings: dict[str, dict[str, str | None]] = {}

    def flush() -> None:
        nonlocal shard_rows, shard_idx
        if not shard_rows:
            return
        path = out_dir / "shards" / f"decisions_shard_{shard_idx:03d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as h:
            for r in shard_rows:
                h.write(json.dumps(r) + "\n")
        shard_idx += 1
        shard_rows = []

    for plan in cell_plans:
        target = plan.target
        cell = target.cell
        cell_games = games_by_cell.get(cell.cell_id, [])
        stat = per_opp.setdefault(plan.opponent.name, {"games": 0, "our_wins": 0, "completed": 0})
        if not cell_games:
            continue
        cards = list(target.cards)
        opp_pilot = build_pilot(
            plan.opponent.pilot,
            cards,
            cell.deck_hash,
            cfg=cfg,
            band_tables=band_tables,
            bucket=cell.source_bucket,
            archetype=target.archetype,
            feature_policy_id=plan.opponent_policy_identity["feature_policy_id"],
        )
        opp_label = f"{plan.opponent.name}:{target.archetype}"
        opponent_feature_binding = live_feature_binding(opp_pilot)
        expected_opponent_feature_id = plan.opponent_policy_identity["feature_policy_id"]
        live_opponent_feature_id = opponent_feature_binding["feature_policy_id"]
        if (
            live_opponent_feature_id is not None
            and live_opponent_feature_id != expected_opponent_feature_id
        ):
            raise ValueError(
                f"opponent feature-policy mismatch for {cell.cell_id}: "
                f"configured {expected_opponent_feature_id!r}, live {live_opponent_feature_id!r}"
            )
        opponent_feature_bindings[cell.cell_id] = opponent_feature_binding
        for game in cell_games:
            rows, safety = dg.play_recorded_game(
                our_pilot, opp_pilot, our_deck, cards,
                our_seat=game.game_ordinal % 2, our_label=cfg.our.label, opp_label=opp_label,
                opp_deck_hash=cell.deck_hash, bucket=cell.source_bucket,
                game_id=game.game_id,
                game_index=game.schedule_ordinal, seed=game.seed,
                encoder=encoder, max_steps=max_steps,
                opponent_shell=target.archetype, opponent_pilot=plan.opponent.pilot,
                feature_policy_id=our_feature_policy_id,
            )
            totals["games"] += 1
            stat["games"] += 1
            for k in ("completed", "crashes", "timeouts", "illegal"):
                totals[k] += safety.get(k, 0)
            totals["our_wins"] += safety.get("our_wins", 0)
            stat["our_wins"] += safety.get("our_wins", 0)
            stat["completed"] += safety.get("completed", 0)
            if any(safety.get(key, 0) for key in ("crashes", "timeouts", "illegal")):
                raise RuntimeError(f"league safety hard stop for {game.game_id}: {safety}")
            row_identity = {
                "league_schedule_version": SCHEDULE_VERSION,
                "cell_id": cell.cell_id,
                "cell_id_version": CELL_ID_VERSION,
                "game_id_version": GAME_ID_VERSION,
                "league_run_id": run_id,
                "league_round_id": round_id,
                "game_ordinal": game.game_ordinal,
                "target_stage": cell.target_stage,
                "source_bucket": cell.source_bucket,
                "opponent_source": cell.opponent_source,
                "opponent_member": cell.opponent_member,
                "fixed_deck_ref": cell.fixed_deck_ref or None,
                "our_live_policy_id": our_policy_identity["policy_id"],
                "our_feature_policy_id": our_feature_policy_id,
                "our_feature_deck_hash": our_feature_binding["feature_deck_hash"],
                "our_league_member_id": cfg.our.league_member_id or None,
                "our_policy_artifact_sha256": our_policy_identity.get("artifact_sha256"),
                "opponent_live_policy_id": plan.opponent_policy_identity["policy_id"],
                "opponent_feature_policy_id": plan.opponent_policy_identity["feature_policy_id"],
                "opponent_feature_deck_hash": opponent_feature_binding["feature_deck_hash"],
                "opponent_league_member_id": cell.opponent_league_member_id or None,
                "opponent_policy_artifact_sha256": plan.opponent_policy_identity.get("artifact_sha256"),
            }
            for fallback_index, row in enumerate(rows):
                decision_index = int(row.get("decision_index", fallback_index))
                decision_id = f"{game.game_id}:d{decision_index:04d}"
                existing_decision_id = str(row.get("decision_id") or decision_id)
                if existing_decision_id != decision_id:
                    raise ValueError(f"unexpected decision identity {existing_decision_id}")
                if decision_id in emitted_decision_ids:
                    raise ValueError(f"duplicate decision identity before shard write: {decision_id}")
                emitted_decision_ids.add(decision_id)
                row["decision_id"] = decision_id
                row.update(row_identity)
            totals["rows"] += len(rows)
            shard_rows.extend(rows)
            if len(shard_rows) >= 20000:
                flush()
        print(f"[{plan.opponent.name}] {cell.target_stage}/{cell.source_bucket} "
              f"games={stat['games']} wins={stat['our_wins']}/{stat['completed']}", flush=True)
    flush()
    identity_summary = _validate_identity_shards(out_dir)
    if identity_summary["rows"] != totals["rows"]:
        raise ValueError(
            f"league identity scan row mismatch: {identity_summary['rows']} != {totals['rows']}"
        )

    assigned_cells = [
        {
            **plan.target.cell.to_json(),
            "games_allocated": plan.allocation.games,
            "pfsp": plan.allocation.to_json(),
            "opponent_policy_identity": plan.opponent_policy_identity,
            "opponent_feature_binding": opponent_feature_bindings.get(plan.target.cell.cell_id),
        }
        for index, plan in enumerate(cell_plans)
        if index % shard_count == shard_index
    ]
    totals["incomplete"] = totals["games"] - totals["completed"]
    if totals["incomplete"] != 0:
        raise ValueError("league datagen completed-game budget is incomplete")
    shard_receipts = _shard_receipts(out_dir / "shards")

    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": dg._now(),
        "status": "completed",
        "schedule_version": SCHEDULE_VERSION,
        "schedule_fingerprint": _schedule_fingerprint(
            run_id=run_id,
            round_id=round_id,
            cells=assigned_cells,
        ),
        "run_id": run_id, "round_id": round_id,
        "generation_binding": generation_binding,
        "our": {
            "pilot": cfg.our.pilot, "deck": cfg.our.deck, "deck_hash": our_hash,
            "policy_identity": our_policy_identity,
            "deck_binding": our_deck_binding,
            "feature_binding": our_feature_binding,
            "league_member_id": cfg.our.league_member_id or None,
        },
        "opponents": [o.name for o in cfg.enabled_opponents()],
        "stages": list(cfg.stages), "pfsp_enabled": cfg.pfsp.enabled,
        "pfsp": {
            "enabled": cfg.pfsp.enabled,
            "floor": cfg.pfsp.floor,
            "power": cfg.pfsp.power,
            "min_games": cfg.pfsp.min_games,
            "evidence": pfsp_evidence.to_json(),
            "allocations_by_opponent": pfsp_allocations,
        },
        "cells": assigned_cells,
        "shard_count": shard_count, "shard_index": shard_index,
        "totals": totals, "per_opponent": per_opp, "shards": shard_idx,
        "shard_receipts": shard_receipts,
        "identity": identity_summary,
        "our_win_rate": round(totals["our_wins"] / max(1, totals["completed"]), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="league YAML (see .research/curriculum/league.example.yaml)")
    ap.add_argument("--pools", default="", help="bucket_pools.json (overrides config; required for band opponents)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--champion", default="", help="override config champion checkpoint path (e.g. Modal /repo/champion.pt)")
    ap.add_argument("--replay-tables", default="", help="override config band_replay_tables.json path")
    ap.add_argument("--our-pilot", default="", help="override config our.pilot, e.g. checkpoint:<round.pt>")
    ap.add_argument("--our-deck", default="", help="override config our.deck, e.g. deck:<deck.csv>")
    ap.add_argument("--our-label", default="", help="override config our.label")
    ap.add_argument(
        "--our-feature-policy-id",
        default="",
        help="override the categorical policy ID used by the live feature encoder",
    )
    ap.add_argument(
        "--our-league-member-id",
        default="",
        help="immutable registry member ID for the learner snapshot",
    )
    ap.add_argument("--seed", type=int, default=-1, help="override config schedule seed")
    ap.add_argument("--run-id", default="", help="stable run identity included in every game ID")
    ap.add_argument("--round-id", default="", help="stable round identity included in every game ID")
    ap.add_argument("--generation-id", default="", help="league generation receipt identity")
    ap.add_argument("--league-snapshot-id", default="", help="frozen league snapshot identity")
    ap.add_argument("--league-registry", default="", help="registry used to verify learner bytes")
    ap.add_argument("--workers", type=int, default=1, help="local process-parallel workers over league cells")
    ap.add_argument("--shard-index", type=int, default=0, help="internal: which worker slice this process runs")
    ap.add_argument("--shard-count", type=int, default=1, help="internal: total worker slices")
    ap.add_argument("--max-steps", type=int, default=5000)
    args = ap.parse_args(argv)

    cfg = load_league_config(args.config)
    cfg = apply_our_overrides(
        cfg,
        pilot=args.our_pilot,
        deck=args.our_deck,
        label=args.our_label,
        feature_policy_id=args.our_feature_policy_id,
        league_member_id=args.our_league_member_id,
    )
    if args.champion:
        object.__setattr__(cfg, "champion", args.champion)
    if args.replay_tables:
        object.__setattr__(cfg, "replay_tables", args.replay_tables)
    if args.seed >= 0:
        object.__setattr__(cfg, "seed", args.seed)
    pools = args.pools
    if not pools:
        hits = sorted((ROOT / ".research/runs/bucket_pool").rglob("bucket_pools.json"))
        if not hits:
            raise SystemExit("no --pools given and no bucket_pools.json found")
        pools = str(hits[-1])
    args.pools = pools
    object.__setattr__(cfg, "pools", pools)  # frozen dataclass: attach runtime pools path
    inferred_run_id, inferred_round_id = _inferred_schedule_identity(Path(args.out_dir))
    args.run_id = args.run_id or inferred_run_id
    args.round_id = args.round_id or inferred_round_id
    if int(args.workers) > 1 and int(args.shard_count) <= 1:
        return _run_workers(args)
    run(cfg, Path(args.out_dir), max_steps=args.max_steps,
        shard_count=int(args.shard_count), shard_index=int(args.shard_index),
        run_id=args.run_id, round_id=args.round_id,
        generation_id=args.generation_id,
        snapshot_id=args.league_snapshot_id,
        registry_path=args.league_registry)
    return 0


def _append_arg(parts: list[str], flag: str, value: str | int) -> None:
    if str(value):
        parts.extend([flag, str(value)])


def _run_workers(args: argparse.Namespace) -> int:
    """Fan out worker subprocesses over league target cells, then merge shards."""
    started = time.perf_counter()
    n = int(args.workers)
    base = Path(args.out_dir)
    (base / "shards").mkdir(parents=True, exist_ok=True)
    common = [
        "--config", args.config,
        "--pools", args.pools,
        "--max-steps", str(args.max_steps),
        "--run-id", args.run_id,
        "--round-id", args.round_id,
    ]
    for flag, value in (
        ("--generation-id", args.generation_id),
        ("--league-snapshot-id", args.league_snapshot_id),
        ("--league-registry", args.league_registry),
    ):
        _append_arg(common, flag, value)
    for flag, value in (
        ("--champion", args.champion),
        ("--replay-tables", args.replay_tables),
        ("--our-pilot", args.our_pilot),
        ("--our-deck", args.our_deck),
        ("--our-label", args.our_label),
        ("--our-feature-policy-id", args.our_feature_policy_id),
        ("--our-league-member-id", args.our_league_member_id),
    ):
        _append_arg(common, flag, value)
    if int(args.seed) >= 0:
        common.extend(["--seed", str(args.seed)])

    procs = []
    for i in range(n):
        wdir = base / f"w{i}"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *common,
               "--out-dir", str(wdir), "--shard-count", str(n), "--shard-index", str(i)]
        procs.append((i, wdir, subprocess.Popen(cmd)))
        print(f"[league-workers] launched worker {i}/{n} -> {wdir}", flush=True)
    failed: list[int] = []
    for i, _wdir, p in procs:
        rc = p.wait()
        print(f"[league-workers] worker {i} exit {rc}", flush=True)
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"league worker(s) failed: {failed}")

    merged = {
        "games": 0,
        "rows": 0,
        "our_wins": 0,
        "completed": 0,
        "crashes": 0,
        "timeouts": 0,
        "illegal": 0,
        "incomplete": 0,
    }
    per_opp: dict[str, dict[str, int]] = {}
    cells_by_id: dict[str, dict[str, Any]] = {}
    first_manifest: dict[str, Any] | None = None
    out_shard = 0
    for _i, wdir, _p in procs:
        for sh in sorted((wdir / "shards").glob("*.jsonl.gz")):
            shutil.copy2(sh, base / "shards" / f"decisions_shard_{out_shard:03d}.jsonl.gz")
            out_shard += 1
        mf = wdir / "manifest.json"
        if not mf.exists():
            continue
        data = json.loads(mf.read_text(encoding="utf-8"))
        if first_manifest is None:
            first_manifest = data
        elif (
            data.get("generation_binding") != first_manifest.get("generation_binding")
            or data.get("our") != first_manifest.get("our")
            or data.get("schedule_version") != first_manifest.get("schedule_version")
        ):
            raise ValueError(f"worker manifest generation/learner binding mismatch: {mf}")
        for cell in data.get("cells") or []:
            cell_id = str(cell.get("cell_id") or "")
            if not cell_id:
                raise ValueError(f"worker manifest has a cell without cell_id: {mf}")
            if cell_id in cells_by_id:
                raise ValueError(f"duplicate scheduled cell across workers: {cell_id}")
            cells_by_id[cell_id] = cell
        for k in merged:
            merged[k] += int((data.get("totals") or {}).get(k, 0))
        for name, stat in (data.get("per_opponent") or {}).items():
            cur = per_opp.setdefault(name, {"games": 0, "our_wins": 0, "completed": 0})
            for k in cur:
                cur[k] += int(stat.get(k, 0))
    if first_manifest is None:
        raise SystemExit("league workers produced no manifests")
    identity_summary = _validate_identity_shards(base)
    if identity_summary["rows"] != merged["rows"]:
        raise ValueError(
            f"merged league identity scan row mismatch: {identity_summary['rows']} != {merged['rows']}"
        )

    merged["incomplete"] = merged["games"] - merged["completed"]
    if merged["incomplete"] != 0:
        raise ValueError("merged league datagen completed-game budget is incomplete")
    merged_cells = [cells_by_id[cell_id] for cell_id in sorted(cells_by_id)]
    shard_receipts = _shard_receipts(base / "shards")
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": dg._now(),
        "status": "completed",
        "schedule_version": SCHEDULE_VERSION,
        "schedule_fingerprint": _schedule_fingerprint(
            run_id=args.run_id,
            round_id=args.round_id,
            cells=merged_cells,
        ),
        "run_id": args.run_id, "round_id": args.round_id,
        "generation_binding": first_manifest.get("generation_binding"),
        "our": first_manifest.get("our", {}),
        "opponents": first_manifest.get("opponents", []),
        "stages": first_manifest.get("stages", []),
        "pfsp_enabled": first_manifest.get("pfsp_enabled", False),
        "pfsp": first_manifest.get("pfsp", {}),
        "cells": merged_cells,
        "workers": n, "totals": merged, "per_opponent": per_opp, "shards": out_shard,
        "shard_receipts": shard_receipts,
        "identity": identity_summary,
        "our_win_rate": round(merged["our_wins"] / max(1, merged["completed"]), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
