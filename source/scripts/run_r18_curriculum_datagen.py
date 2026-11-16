"""R18: hard-curriculum data generation for Starmie policy training.

Generates training rows of a Starmie pilot playing the progressive ELO-bucket
curriculum (600-800 -> 900-1000 ...) against SMART opponents (v2.4 heuristic and
the champion `exploiter`), NOT first-legal. Rows are stamped with the game outcome
so a downstream trainer can advantage-weight (learn from Starmie's WINS). This is
the AlphaStar-league / PFSP data lane over a FIXED opponent snapshot -> INVARIANT #2
respected (no self-play-to-Nash / equilibrium signal).

Output layout matches the value/policy trainers' refgames-dir contract:
`<out>/shards/decisions_shard_XXX.jsonl.gz` (+ manifest). Runs locally on CPU
(libcg) and is import-clean for a Modal container (cg/libcg.so ships in the repo).

Example
-------
    python scripts/run_r18_curriculum_datagen.py \
        --pools .research/runs/bucket_pool/.../bucket_pools.json \
        --stages 600-800 800-900 900-1000 \
        --our-pilot v2.4 --opp-pilots v2.4 exploiter \
        --games 8 --top-n 8 --out-dir D:/DeepMew_artifacts/datasets/r18/starmie_curriculum
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
import main  # noqa: E402
from deepmew.bucket_arena import build_stage, entry_from_json  # noqa: E402
from deepmew.artifacts.run_dir import safe_slug  # noqa: E402
from deepmew.artifacts.writers import write_json  # noqa: E402
from deepmew.curriculum.ladder_metadata import ladder_metadata  # noqa: E402
from deepmew.replay.chosen_action_advantage import TRANSITION_SCHEMA_VERSION  # noqa: E402
from deepmew.replay.information_boundary import (  # noqa: E402
    InformationBoundary,
    SCHEMA_VERSION as INFORMATION_BOUNDARY_SCHEMA_VERSION,
    require_oracle_output_path,
)
from deepmew.rl.observation import ObservationEncoder  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent  # noqa: E402
import run_r17_bucket_arena as arena  # noqa: E402

SCHEMA_VERSION = "deepmew.r18.curriculum_datagen.v1"
ROW_SCHEMA_VERSION = "deepmew.refgame_decision_row.rebuilt.v2"
PROGRESS_SCHEMA_VERSION = "deepmew.r18.curriculum_datagen.progress.v1"
DEFAULT_STARMIE = ROOT / ".research" / "submissions" / "20260706-r4f-starmie-18ffc1e6"
DEFAULT_CHAMPION = arena.DEFAULT_CHAMPION


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        return {"git_head": "not_available", "git_dirty": "not_available"}

    return {"git_head": head, "git_dirty": dirty}


def _path_provenance(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.is_file():
        return {"path": str(resolved), "sha256": _sha256_file(resolved), "bytes": resolved.stat().st_size}
    if not resolved.is_dir():
        return {"path": str(resolved), "status": "not_found"}
    files = [
        candidate for candidate in sorted(resolved.rglob("*"))
        if candidate.is_file() and "__pycache__" not in candidate.parts and candidate.suffix != ".pyc"
    ]
    return {
        "path": str(resolved),
        "files": [
            {
                "relative_path": candidate.relative_to(resolved).as_posix(),
                "sha256": _sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
            for candidate in files
        ],
    }


def _policy_provenance(spec: str, *, champion: str | Path) -> dict[str, Any]:
    if spec == "exploiter":
        return {"spec": spec, "kind": "checkpoint", "checkpoint": _path_provenance(champion)}
    if spec.startswith("checkpoint:"):
        return {
            "spec": spec, "kind": "checkpoint",
            "checkpoint": _path_provenance(spec.split(":", 1)[1]),
        }

    return {"spec": spec, "kind": "named_policy", "checkpoint": "not_applicable"}


def _atomic_write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
                for row in rows:
                    compressed.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_durable_rows(out_dir: Path) -> tuple[set[str], set[str], int, int]:
    game_ids: set[str] = set()
    decision_ids: set[str] = set()
    winning_game_ids: set[str] = set()
    row_count = 0
    for path in sorted((out_dir / "shards").glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                game_id = str(row.get("game_id") or "")
                decision_id = str(row.get("decision_id") or "")
                if not game_id or not decision_id:
                    raise ValueError(f"{path}:{line_number}: missing stable game_id/decision_id")
                if decision_id in decision_ids:
                    raise ValueError(f"duplicate decision_id in durable shards: {decision_id}")
                game_ids.add(game_id)
                decision_ids.add(decision_id)
                if _as_dict(row.get("outcome")).get("won") is True:
                    winning_game_ids.add(game_id)
                row_count += 1

    return game_ids, decision_ids, row_count, len(winning_game_ids)


def _next_shard_index(out_dir: Path) -> int:
    indices: list[int] = []
    for path in (out_dir / "shards").glob("decisions_shard_*.jsonl.gz"):
        try:
            indices.append(int(path.stem.split("_")[-1].split(".")[0]))
        except ValueError:
            continue

    return max(indices, default=-1) + 1


def _generation_contract(args: argparse.Namespace, *, run_id: str, information_mode: str) -> dict[str, Any]:
    our_policy = _policy_provenance(args.our_pilot, champion=args.champion)
    opponent_policies = [_policy_provenance(spec, champion=args.champion) for spec in args.opp_pilots]
    contract = {
        "run_id": run_id,
        "information_mode": information_mode,
        "pools": str(Path(args.pools).resolve()),
        "pools_sha256": _sha256_file(Path(args.pools)),
        "script_sha256": _sha256_file(Path(__file__)),
        "starmie_bundle": str(Path(args.starmie_bundle).resolve()),
        "starmie_deck_sha256": _sha256_file(Path(args.starmie_bundle) / "deck.csv"),
        "stages": list(args.stages),
        "our_policy": our_policy,
        "opponent_policies": opponent_policies,
        "games": int(args.games),
        "top_n": int(args.top_n),
        "carry_n": int(args.carry_n),
        "exploiter_top": int(args.exploiter_top),
        "shard_rows": int(args.shard_rows),
        "progress_games": int(args.progress_games),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "shard_count": int(getattr(args, "shard_count", 1)),
        "shard_index": int(getattr(args, "shard_index", 0)),
    }
    replay_tables = str(getattr(args, "replay_tables", "") or "")
    contract["replay_tables"] = (
        _path_provenance(replay_tables) if replay_tables else "not_used"
    )
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract["contract_sha256"] = _sha256_bytes(encoded)

    return contract


def _load_or_initialize_progress(
    out_dir: Path, *, contract: dict[str, Any], resume: bool,
) -> tuple[dict[str, Any], set[str], int]:
    progress_path = out_dir / "progress.json"
    durable_game_ids, _decision_ids, durable_rows, durable_wins = _read_durable_rows(out_dir)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"output already has progress; pass --resume: {progress_path}")
        existing_contract = progress.get("generation_contract")
        if not isinstance(existing_contract, dict) or existing_contract.get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError("resume generation contract does not match existing progress.json")
        progress["resume_events"] = int(progress.get("resume_events") or 0) + 1
    else:
        if durable_rows:
            action = "resume" if resume else "start a new run in"
            raise ValueError(f"cannot {action} a directory with durable shards but no progress.json")
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "run_id": contract["run_id"],
            "created_at_utc": _now(),
            "generation_contract": contract,
            "resume_events": 0,
        }
    progress.update({
        "updated_at_utc": _now(),
        "durable_game_ids": sorted(durable_game_ids),
        "durable_games": len(durable_game_ids),
        "durable_rows": durable_rows,
        "durable_wins": durable_wins,
        "next_shard_index": _next_shard_index(out_dir),
        "status": "running",
    })
    write_json(progress_path, progress)

    return progress, durable_game_ids, durable_rows


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _own_hand_ids(obs: dict[str, Any], seat: int) -> list[int]:
    players = obs.get("current", {}).get("players", []) if isinstance(obs, dict) else []
    if not (0 <= seat < len(players)):
        return []
    hand = _as_dict(players[seat]).get("hand") or []
    return [c["id"] for c in hand if isinstance(c, dict) and isinstance(c.get("id"), int)]


def _build_pilot(spec: str, deck: list[int], deck_hash: str, *,
                 champion: str | Path = DEFAULT_CHAMPION, band_tables: Any = None,
                 bucket: str = "", archetype: str = "Unknown",
                 feature_policy_id: str = "") -> Any:
    """A SeatAgent (select(obs, rng)) piloting `deck`. Opponent tiers, ascending:
    {first-legal, v2.4, greedy, ...} heuristic; `replay` = faithful band-stratified
    R16 pilot (needs `band_tables` + `bucket`/`archetype`); `exploiter` = fixed
    champion checkpoint on the deck; `checkpoint:<path>` = an ARBITRARY R4-F checkpoint
    piloting the deck (frozen league member / self-play with the CURRENT policy)."""
    if spec == "replay":
        if band_tables is None:
            raise ValueError("replay pilot requires --replay-tables (band_replay_tables.json)")
        from deepmew.real_meta_arena.replay_policy import ReplayAgent  # noqa: PLC0415
        return ReplayAgent(archetype=archetype, table=band_tables.table_for(bucket))
    if spec == "exploiter":
        return arena.build_exploiter(
            Path(champion),
            list(deck),
            deck_hash,
            "reference_abomasnow",
            feature_policy_id=feature_policy_id or "bucket_exploiter",
        )
    if spec.startswith("checkpoint:"):
        return arena.build_exploiter(
            Path(spec.split(":", 1)[1]),
            list(deck),
            deck_hash,
            "reference_abomasnow",
            feature_policy_id=feature_policy_id or "bucket_exploiter",
        )
    return SpecAgent.from_str(spec)


def play_recorded_game(
    our_pilot: Any, opp_pilot: Any, our_deck: list[int], opp_deck: list[int],
    *, our_seat: int, our_label: str, opp_label: str, opp_deck_hash: str, bucket: str,
    game_id: str, game_index: int, seed: int, encoder: ObservationEncoder, max_steps: int,
    opponent_shell: str = "", opponent_pilot: str = "", training_phase: str = "",
    information_boundary: InformationBoundary | None = None,
    generation_run_id: str = "", generated_at_utc: str = "",
    generation_provenance: dict[str, Any] | None = None,
    feature_policy_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    boundary = information_boundary or InformationBoundary()
    safety = {"illegal": 0, "crashes": 0, "timeouts": 0, "completed": 0, "our_wins": 0}
    rng = random.Random(seed)
    our_hash = normalized_deck_hash(our_deck)
    deck0, deck1 = (our_deck, opp_deck) if our_seat == 0 else (opp_deck, our_deck)
    rows: list[dict[str, Any]] = []
    obs = None
    try:
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            raise RuntimeError(f"BattleStart failed: {start_data.errorType}")
        decision_index = 0
        result = -1
        for _ in range(max_steps):
            current = _as_dict(obs.get("current"))
            result = _as_int(current.get("result"), -1)
            if result != -1:
                break
            player_index = _as_int(current.get("yourIndex"))
            actor_is_ours = player_index == our_seat
            actor = our_pilot if actor_is_ours else opp_pilot
            selected = actor.select(obs, rng)
            select = _as_dict(obs.get("select"))
            legal_count = len(select.get("option") or [])
            if any(i < 0 or i >= legal_count for i in selected):
                safety["illegal"] += 1
                return [], safety
            pending_transition_row: dict[str, Any] | None = None
            if actor_is_ours and legal_count:
                legal_set = legal_action_table(obs, select)
                artifact_obs = boundary.artifact_observation(
                    obs,
                    visualizer_payload=visualize_data() if boundary.reveals_opponent_private_state else None,
                )
                ladder_fields = ladder_metadata(
                    opponent_shell=opponent_shell,
                    opponent_bucket=bucket,
                    opponent_pilot=opponent_pilot,
                    opponent_policy_id=opp_label,
                    opponent_label=opp_label,
                    bucket=bucket,
                    opponent_deck_hash=opp_deck_hash,
                    training_phase=training_phase,
                )
                pending_transition_row = {
                    "schema_version": ROW_SCHEMA_VERSION,
                    "transition_schema_version": TRANSITION_SCHEMA_VERSION,
                    "information_boundary_schema_version": INFORMATION_BOUNDARY_SCHEMA_VERSION,
                    "information_mode": boundary.mode.value,
                    "generation_run_id": generation_run_id,
                    "generated_at_utc": generated_at_utc,
                    "generation_provenance": dict(generation_provenance or {}),
                    "game_id": game_id, "game_index": game_index,
                    "decision_id": f"{game_id}:d{decision_index:04d}",
                    "decision_index": decision_index, "seed": seed, "seat": our_seat,
                    "turn": _as_int(current.get("turn")),
                    "policy_id": feature_policy_id or our_label,
                    "feature_policy_id": feature_policy_id or our_label,
                    "deck_id": our_label, "deck_hash": our_hash,
                    "opponent_policy_id": opp_label, "opponent_deck_id": opp_label,
                    "opponent_deck_hash": opp_deck_hash, "bucket": bucket,
                    **ladder_fields,
                    "raw_observation": artifact_obs, "base84": encoder.encode(artifact_obs),
                    "own_hand": _own_hand_ids(artifact_obs, our_seat), "legal_set": legal_set,
                    "action_mask": {"n_options": legal_count,
                                    "min_count": _as_int(select.get("minCount")),
                                    "max_count": _as_int(select.get("maxCount"), legal_count)},
                    "chosen_action": {"indices": list(selected),
                                      "semantics": [legal_set[i] for i in selected if 0 <= i < len(legal_set)]},
                    "post_observation": None,
                    "post_observation_missing": True,
                    "post_turn": None,
                    "post_actor_seat": None,
                    "post_result": None,
                    "outcome": {},
                }
                rows.append(pending_transition_row)
                decision_index += 1
            post_obs = battle_select(selected)
            if pending_transition_row is not None:
                post_current = _as_dict(post_obs.get("current")) if isinstance(post_obs, dict) else {}
                artifact_post_obs = boundary.artifact_observation(
                    post_obs,
                    visualizer_payload=visualize_data() if boundary.reveals_opponent_private_state else None,
                ) if isinstance(post_obs, dict) else None
                pending_transition_row["post_observation"] = artifact_post_obs
                pending_transition_row["post_observation_missing"] = not isinstance(post_obs, dict)
                pending_transition_row["post_turn"] = _as_int(post_current.get("turn"), None)
                pending_transition_row["post_actor_seat"] = _as_int(post_current.get("yourIndex"), None)
                pending_transition_row["post_result"] = _as_int(post_current.get("result"), None)
            obs = post_obs
        else:
            safety["timeouts"] += 1
            return [], safety
        won = result == our_seat
        draw = result not in (0, 1)
        safety["completed"] += 1
        safety["our_wins"] += int(won)
        outcome = {"won": won, "lost": (not won) and not draw, "draw": draw,
                   "result": result, "crashed": False, "timed_out": False}
        for row in rows:
            row["outcome"] = dict(outcome)
        return rows, safety
    except Exception:
        safety["crashes"] += 1
        return [], safety
    finally:
        if obs is not None:
            battle_finish()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    run_id = safe_slug(str(getattr(args, "run_id", "") or out_dir.name), fallback="curriculum_datagen")
    information_boundary = InformationBoundary.from_reveal_flag(bool(args.reveal_opponent_private_state))
    require_oracle_output_path(out_dir, information_boundary)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    contract = _generation_contract(args, run_id=run_id, information_mode=information_boundary.mode.value)
    progress, completed_game_ids, durable_rows = _load_or_initialize_progress(
        out_dir, contract=contract, resume=bool(getattr(args, "resume", False)),
    )
    cumulative_elapsed = float(progress.get("elapsed_seconds") or 0.0)
    generated_at_utc = str(progress["created_at_utc"])
    starmie_deck = [int(v) for v in (Path(args.starmie_bundle) / "deck.csv").read_text().split()][:60]
    our_hash = normalized_deck_hash(starmie_deck)
    generation_provenance = {
        **_git_provenance(),
        "script": _path_provenance(Path(__file__)),
        "pools": _path_provenance(Path(args.pools)),
        "our_deck_bundle": _path_provenance(Path(args.starmie_bundle)),
        "our_policy": _policy_provenance(args.our_pilot, champion=args.champion),
        "opponent_policies": [
            _policy_provenance(spec, champion=args.champion) for spec in args.opp_pilots
        ],
        "value_scale": "terminal_outcome_only; no value label in datagen row",
        "actor_perspective": "our_seat",
    }
    row_provenance_base = {
        "git_head": generation_provenance.get("git_head"),
        "git_dirty": generation_provenance.get("git_dirty"),
        "script_sha256": _as_dict(generation_provenance.get("script")).get("sha256"),
        "generation_contract_sha256": contract["contract_sha256"],
        "our_policy_spec": args.our_pilot,
        "our_policy": generation_provenance["our_policy"],
    }
    band_tables = None
    if getattr(args, "replay_tables", ""):
        from deepmew.curriculum.replay_pilots import BandReplayTables  # noqa: PLC0415
        band_tables = BandReplayTables.from_json(json.loads(Path(args.replay_tables).read_text(encoding="utf-8")))
    our_pilot = _build_pilot(args.our_pilot, starmie_deck, our_hash, champion=args.champion,
                             band_tables=band_tables)
    our_label = f"starmie:{args.our_pilot}"
    encoder = ObservationEncoder()

    pools_payload = json.loads(Path(args.pools).read_text(encoding="utf-8"))
    pools = {b: [entry_from_json(r) for r in rows] for b, rows in pools_payload["pools"].items()}
    shells = {r["deck_hash"]: str(r.get("shell") or "Unknown")
              for rows in pools_payload["pools"].values() for r in rows}

    shard_rows: list[dict[str, Any]] = []
    pending_game_ids: set[str] = set()
    shard_idx = int(progress["next_shard_index"])
    totals = {
        "games": len(completed_game_ids), "rows": durable_rows,
        "our_wins": int(progress.get("durable_wins") or 0), "completed": len(completed_game_ids),
        "crashes": 0, "timeouts": 0, "illegal": 0,
    }
    skipped_games = 0
    cell_index = 0  # (stage, opponent) cells, for --shard-count/--shard-index splitting
    shard_count = max(1, int(getattr(args, "shard_count", 1)))
    shard_pick = int(getattr(args, "shard_index", 0))
    game_index = shard_pick * 1_000_000  # disjoint seed/id space per worker

    def write_progress(status: str = "running") -> None:
        elapsed = cumulative_elapsed + time.perf_counter() - started
        progress.update({
            "updated_at_utc": _now(), "status": status,
            "durable_game_ids": sorted(completed_game_ids),
            "durable_games": len(completed_game_ids), "durable_rows": totals["rows"],
            "durable_wins": totals["our_wins"], "next_shard_index": shard_idx,
            "elapsed_seconds": round(elapsed, 3), "totals": dict(totals),
            "skipped_games_this_invocation": skipped_games,
        })
        write_json(out_dir / "progress.json", progress)

    def flush() -> None:
        nonlocal shard_rows, shard_idx, pending_game_ids
        if not shard_rows:
            return
        path = out_dir / "shards" / f"decisions_shard_{shard_idx:03d}.jsonl.gz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite durable shard: {path}")
        for row in shard_rows:
            row["source_dataset"] = str(out_dir.resolve())
            row["source_shard"] = str(path.resolve())
        _atomic_write_gzip_jsonl(path, shard_rows)
        completed_game_ids.update(pending_game_ids)
        shard_idx += 1
        shard_rows = []
        pending_game_ids = set()
        write_progress()

    for stage in args.stages:
        opponents = build_stage(pools, stage, top_n=int(args.top_n), carry_n=int(args.carry_n))
        for oi, opp in enumerate(opponents):
            this_cell = cell_index
            cell_index += 1
            if this_cell % shard_count != shard_pick:
                continue  # another worker owns this (stage, opponent) cell
            opp_hash = opp.deck_hash
            opp_shell = shells.get(opp_hash, "Unknown")
            for opp_spec in args.opp_pilots:
                if opp_spec == "exploiter" and oi >= int(args.exploiter_top):
                    continue
                opp_pilot = _build_pilot(opp_spec, list(opp.card_ids), opp_hash,
                                         champion=args.champion, band_tables=band_tables,
                                         bucket=stage, archetype=opp_shell)
                opp_label = f"{opp_shell}:{opp_spec}"
                for g in range(int(args.games)):
                    cell_signature = _sha256_bytes(f"{stage}|{opp_hash}|{opp_spec}".encode("utf-8"))[:12]
                    game_id = f"{run_id}:w{shard_pick}:c{this_cell:04d}:{cell_signature}:g{g:04d}"
                    scheduled_game_index = game_index
                    game_index += 1
                    if game_id in completed_game_ids:
                        skipped_games += 1
                        continue
                    rows, safety = play_recorded_game(
                        our_pilot, opp_pilot, starmie_deck, list(opp.card_ids),
                        our_seat=g % 2, our_label=our_label, opp_label=opp_label,
                        opp_deck_hash=opp_hash, bucket=stage,
                        game_id=game_id,
                        game_index=scheduled_game_index, seed=int(args.seed) + scheduled_game_index,
                        encoder=encoder, max_steps=int(args.max_steps),
                        opponent_shell=opp_shell, opponent_pilot=opp_spec,
                        information_boundary=information_boundary,
                        generation_run_id=run_id, generated_at_utc=generated_at_utc,
                        generation_provenance={
                            **row_provenance_base,
                            "opponent_policy": _policy_provenance(opp_spec, champion=args.champion),
                        },
                    )
                    totals["games"] += 1
                    for k in ("completed", "crashes", "timeouts", "illegal"):
                        totals[k] += safety.get(k, 0)
                    totals["our_wins"] += safety.get("our_wins", 0)
                    totals["rows"] += len(rows)
                    if safety.get("illegal") or safety.get("crashes") or safety.get("timeouts"):
                        write_progress(status="failed_safety")
                        raise RuntimeError(f"safety hard stop for {game_id}: {safety}")
                    if safety.get("completed") and not rows:
                        write_progress(status="failed_empty_game")
                        raise RuntimeError(f"completed game emitted no decision rows: {game_id}")
                    shard_rows.extend(rows)
                    pending_game_ids.add(game_id)
                    if (len(shard_rows) >= int(args.shard_rows)
                            or len(pending_game_ids) >= int(args.progress_games)):
                        flush()
            print(f"[{stage}] {opp_shell[:26]:26s} done | games={totals['games']} rows={totals['rows']} "
                  f"wins={totals['our_wins']}/{totals['completed']} crashes={totals['crashes']}", flush=True)
    flush()
    _durable_games, durable_decision_ids, verified_rows, _durable_wins = _read_durable_rows(out_dir)
    if verified_rows != totals["rows"]:
        raise ValueError(f"durable row-count mismatch: scanned={verified_rows} totals={totals['rows']}")

    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": _now(),
        "run_id": run_id, "generation_contract": contract,
        "generation_provenance": generation_provenance,
        "information_boundary": information_boundary.metadata(),
        "information_mode": information_boundary.mode.value,
        "our_pilot": args.our_pilot, "opp_pilots": list(args.opp_pilots),
        "stages": list(args.stages), "starmie_deck_hash": our_hash,
        "totals": totals, "shards": shard_idx, "unique_decision_ids": len(durable_decision_ids),
        "duplicate_decision_ids": 0, "resume_events": int(progress.get("resume_events") or 0),
        "skipped_games_this_invocation": skipped_games,
        "win_rate": round(totals["our_wins"] / max(1, totals["completed"]), 4),
        "elapsed_seconds": round(cumulative_elapsed + time.perf_counter() - started, 1),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_progress(status="completed")
    print(json.dumps(manifest, indent=2))
    return manifest


def main_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--starmie-bundle", default=str(DEFAULT_STARMIE))
    ap.add_argument("--stages", nargs="+", default=["600-800", "800-900", "900-1000"])
    ap.add_argument("--our-pilot", default="v2.4",
                    help="v2.4 / first-legal / greedy / exploiter / checkpoint:<path> (self-play with a specific policy)")
    # greedy is a STRONG aggressive tempo pilot (empirically beats a v2.4-piloted
    # Starmie ~0.56), not a floor -> it belongs in the hard curriculum tier.
    ap.add_argument("--opp-pilots", nargs="+", default=["greedy", "v2.4", "replay", "exploiter"],
                    help="league opponent brains: heuristic (greedy/v2.4/first-legal), replay "
                         "(faithful band pilots, needs --replay-tables), exploiter (champion), "
                         "checkpoint:<path> (frozen league member)")
    ap.add_argument("--champion", default=str(DEFAULT_CHAMPION), help="checkpoint for the `exploiter` pilot")
    ap.add_argument("--replay-tables", default="", help="band_replay_tables.json for the `replay` pilot tier")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--carry-n", type=int, default=4)
    ap.add_argument("--exploiter-top", type=int, default=4)
    ap.add_argument("--shard-rows", type=int, default=20000)
    ap.add_argument("--progress-games", type=int, default=8,
                    help="flush an atomic shard after this many completed games, even below --shard-rows")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=181818)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="",
                    help="stable generation identity; defaults to the output-directory name")
    ap.add_argument("--resume", action="store_true",
                    help="resume a matching run from durable shards and progress.json")
    ap.add_argument("--reveal-opponent-private-state", action="store_true",
                    help="research oracle only: persist CABT visualizer opponent hand/deck fields; default is public_belief")
    ap.add_argument("--workers", type=int, default=1, help="local process-parallel workers (each a fresh libcg -> also mitigates BUG-008)")
    ap.add_argument("--shard-index", type=int, default=0, help="internal: which worker slice this process runs")
    ap.add_argument("--shard-count", type=int, default=1, help="internal: total worker slices")
    args = ap.parse_args(argv)
    if args.resume and int(args.workers) > 1:
        ap.error("--resume currently requires --workers 1; worker merge is not a resumable contract")
    if int(args.workers) > 1 and int(args.shard_count) <= 1:
        return _run_workers(args)
    run(args)
    return 0


def _run_workers(args: argparse.Namespace) -> int:
    """Fan out --workers subprocesses over (stage, opponent) cells, then merge shards."""
    import shutil
    import subprocess

    n = int(args.workers)
    base = Path(args.out_dir)
    (base / "shards").mkdir(parents=True, exist_ok=True)
    passthrough = ["--pools", args.pools, "--starmie-bundle", args.starmie_bundle,
                   "--stages", *args.stages, "--our-pilot", args.our_pilot,
                   "--opp-pilots", *args.opp_pilots, "--games", str(args.games),
                   "--top-n", str(args.top_n), "--carry-n", str(args.carry_n),
                   "--exploiter-top", str(args.exploiter_top), "--shard-rows", str(args.shard_rows),
                   "--progress-games", str(args.progress_games),
                   "--max-steps", str(args.max_steps), "--seed", str(args.seed),
                   "--champion", str(args.champion)]
    if getattr(args, "run_id", ""):
        passthrough += ["--run-id", str(args.run_id)]
    if bool(args.reveal_opponent_private_state):
        passthrough.append("--reveal-opponent-private-state")
    if getattr(args, "replay_tables", ""):
        passthrough += ["--replay-tables", str(args.replay_tables)]
    procs = []
    for i in range(n):
        wdir = base / f"w{i}"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *passthrough,
               "--out-dir", str(wdir), "--shard-count", str(n), "--shard-index", str(i)]
        procs.append((i, wdir, subprocess.Popen(cmd)))
        print(f"[workers] launched worker {i}/{n} -> {wdir}", flush=True)
    for i, _wdir, p in procs:
        rc = p.wait()
        print(f"[workers] worker {i} exit {rc}", flush=True)

    merged = {"games": 0, "rows": 0, "our_wins": 0, "completed": 0, "crashes": 0, "timeouts": 0, "illegal": 0}
    out_shard = 0
    for i, wdir, _p in procs:
        for sh in sorted((wdir / "shards").glob("*.jsonl.gz")):
            shutil.copy2(sh, base / "shards" / f"decisions_shard_{out_shard:03d}.jsonl.gz")
            out_shard += 1
        mf = wdir / "manifest.json"
        if mf.exists():
            t = json.loads(mf.read_text(encoding="utf-8")).get("totals", {})
            for k in merged:
                merged[k] += int(t.get(k, 0))
    manifest = {"schema_version": SCHEMA_VERSION, "created_at_utc": _now(), "workers": n,
                "information_boundary": InformationBoundary.from_reveal_flag(bool(args.reveal_opponent_private_state)).metadata(),
                "information_mode": InformationBoundary.from_reveal_flag(bool(args.reveal_opponent_private_state)).mode.value,
                "our_pilot": args.our_pilot, "opp_pilots": list(args.opp_pilots),
                "stages": list(args.stages), "totals": merged, "shards": out_shard,
                "win_rate": round(merged["our_wins"] / max(1, merged["completed"]), 4)}
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
