"""Research-only equal-budget shared-panel evaluation over the R17 arena.

The current champion and every candidate run on the same R17 stage cells,
opponent pilots, batch seeds, per-game policy RNG seeds, seat schedule, and game count.
The command writes an atomic resumable receipt after each cell/seed matchup and
hard-stops the whole panel on the first illegal selection, crash, or timeout.

This is evaluation infrastructure only. It never adopts a checkpoint, packages
a submission, or changes the runtime/default policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from deepmew.bucket_arena import build_stage, entry_from_json  # noqa: E402
from deepmew.curriculum.league_schedule import select_stage_opponents  # noqa: E402
from deepmew.real_meta_arena.runner import MatchupSummary, SpecAgent, run_real_meta_matchup  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402


SCHEMA_VERSION = "deepmew.league_shared_panel.v2"
GAME_SCHEMA_VERSION = "deepmew.league_shared_panel_game.v2"
PROGRESS_SCHEMA_VERSION = "deepmew.league_shared_panel_progress.v2"
GAME_SEED_MULTIPLIER = 1_000_003
PILOT_CHOICES = ("first-legal", "random", "greedy", "legacy-v2", "v2.4", "v4", "exploiter")
SEVERE_REGRESSION_RULE = "candidate_wilson95_upper < champion_wilson95_lower"


class PanelSafetyError(RuntimeError):
    """Raised after partial artifacts are durably written for a safety failure."""


@dataclass(frozen=True)
class PanelParticipant:
    label: str
    spec: str
    role: str
    league_member_id: str = ""


@dataclass(frozen=True)
class PanelCell:
    cell_id: str
    bucket: str
    carried: bool
    deck_hash: str
    shell: str
    pilot: str
    card_ids: tuple[int, ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "bucket": self.bucket,
            "carried": self.carried,
            "deck_hash": self.deck_hash,
            "shell": self.shell,
            "pilot": self.pilot,
        }


@dataclass(frozen=True)
class PanelJob:
    job_key: str
    participant: PanelParticipant
    cell: PanelCell
    panel_seed: int


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

    files = [
        candidate
        for candidate in sorted(resolved.rglob("*"))
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix.lower() not in {".pyc", ".pyo"}
    ]
    rows = []
    digest = hashlib.sha256()
    for candidate in files:
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
        "sha256": digest.hexdigest(),
        "files": rows,
    }


def parse_participant(value: str, *, role: str, league_member_id: str = "") -> PanelParticipant:
    label, separator, spec = value.partition("=")
    label = label.strip()
    spec = spec.strip()
    if not separator or not label or not spec:
        raise ValueError(f"{role} must use LABEL=SPEC, got {value!r}")

    return PanelParticipant(
        label=label,
        spec=spec,
        role=role,
        league_member_id=league_member_id.strip(),
    )


def parse_candidate_member_ids(values: Iterable[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for value in values:
        label, separator, member_id = value.partition("=")
        label = label.strip()
        member_id = member_id.strip()
        if not separator or not label or not member_id:
            raise ValueError(f"candidate member binding must use LABEL=MEMBER_ID, got {value!r}")
        if label in bindings:
            raise ValueError(f"duplicate candidate member binding for {label!r}")
        bindings[label] = member_id

    return bindings


def parse_participant_feature_policy_ids(values: Iterable[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for value in values:
        label, separator, policy_id = value.partition("=")
        label = label.strip()
        policy_id = policy_id.strip()
        if not separator or not label or not policy_id:
            raise ValueError(
                f"participant feature-policy binding must use LABEL=POLICY_ID, got {value!r}"
            )
        if label in bindings:
            raise ValueError(f"duplicate participant feature-policy binding for {label!r}")
        bindings[label] = policy_id

    return bindings


def _reference_path(reference: str) -> Path | None:
    kind, separator, target = reference.partition(":")
    kind = kind.removesuffix("+guard")
    if separator and kind in {"bundle", "checkpoint", "deck"} and target:
        return Path(target)
    if reference:
        candidate = Path(reference)
        if candidate.exists():
            return candidate

    return None


def build_input_receipts(
    *,
    pools_path: str,
    participants: Iterable[PanelParticipant],
    deck_binding: str,
    exploiter_checkpoint: str,
    include_exploiter: bool,
) -> dict[str, Any]:
    references: list[tuple[str, Path]] = [
        ("pools", Path(pools_path)),
        ("root_deck", ROOT / "deck.csv"),
        ("root_main_source", ROOT / "main.py"),
        ("cabt_vendored_surface", ROOT / "cg"),
        ("arena_policy_surface", ROOT / "deepmew" / "arena"),
        ("second_order_feature_surface", ROOT / "deepmew" / "second_order"),
        ("evaluator_source", Path(__file__)),
        ("r17_builder_source", Path(r17.__file__)),
        ("r4f_live_policy_source", ROOT / "scripts" / "run_r9_r4f_retrain.py"),
        ("r4_feature_source", ROOT / "scripts" / "run_unified_r4_ablation.py"),
        ("pilot_adapter_source", ROOT / "scripts" / "run_r12_deck_pilot_matrix.py"),
        ("real_meta_runner_source", Path(run_real_meta_matchup.__code__.co_filename)),
        ("r17_stage_source", Path(build_stage.__code__.co_filename)),
    ]
    for participant in participants:
        path = _reference_path(participant.spec)
        if path is not None:
            references.append((f"participant:{participant.label}", path))
    binding_path = _reference_path(deck_binding)
    if binding_path is not None:
        references.append(("deck_binding", binding_path))
    if include_exploiter:
        references.append(("exploiter_checkpoint", Path(exploiter_checkpoint)))

    receipts: dict[str, Any] = {}
    for label, path in references:
        if label in receipts:
            raise ValueError(f"duplicate input receipt label: {label}")
        receipts[label] = file_receipt(path)

    return receipts


def wilson_interval(wins: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Project-consistent Wilson score interval used by the R9 canary."""
    if total <= 0:
        return 0.0, 1.0
    phat = wins / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total)) / denominator

    return max(0.0, center - half), min(1.0, center + half)


def load_panel_cells(
    pools_path: str | Path,
    *,
    stage_name: str,
    top_n: int,
    carry_n: int,
    pilots: list[str],
    exploiter_top: int,
) -> list[PanelCell]:
    payload = json.loads(Path(pools_path).read_text(encoding="utf-8"))
    pool_rows = payload.get("pools") or {}
    pools = {
        bucket: [entry_from_json(row) for row in rows]
        for bucket, rows in pool_rows.items()
    }
    shells = {
        str(row["deck_hash"]): str(row.get("shell") or "Unknown")
        for rows in pool_rows.values()
        for row in rows
    }
    stage = build_stage(pools, stage_name, top_n=top_n, carry_n=carry_n)
    if not stage:
        raise ValueError(f"R17 stage {stage_name!r} is empty")

    cells: list[PanelCell] = []
    for pilot in pilots:
        opponents = select_stage_opponents(
            stage,
            pilot=pilot,
            target_stage=stage_name,
            exploiter_top=exploiter_top,
        )
        for opponent in opponents:
            cell_id = f"{opponent.bucket}|{opponent.deck_hash}|{pilot}"
            cells.append(
                PanelCell(
                    cell_id=cell_id,
                    bucket=opponent.bucket,
                    carried=opponent.carried,
                    deck_hash=opponent.deck_hash,
                    shell=shells.get(opponent.deck_hash, "Unknown"),
                    pilot=pilot,
                    card_ids=tuple(opponent.card_ids),
                )
            )

    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("R17 cell construction produced duplicate cell ids")

    return cells


def build_jobs(
    participants: list[PanelParticipant],
    cells: list[PanelCell],
    seeds: list[int],
) -> list[PanelJob]:
    jobs: list[PanelJob] = []
    keys: set[str] = set()
    for cell in cells:
        for panel_seed in seeds:
            for participant in participants:
                identity = {
                    "participant": participant.label,
                    "cell_id": cell.cell_id,
                    "panel_seed": panel_seed,
                }
                job_key = _stable_id(identity)
                if job_key in keys:
                    raise ValueError(f"duplicate shared-panel job: {identity}")
                keys.add(job_key)
                jobs.append(PanelJob(job_key, participant, cell, panel_seed))

    return jobs


def validate_durable_jobs(
    progress: dict[str, Any],
    jobs: list[PanelJob],
    *,
    games: int,
) -> None:
    expected = {job.job_key: job for job in jobs}
    for job_key, payload in (progress.get("jobs") or {}).items():
        job = expected.get(job_key)
        if job is None:
            raise ValueError(f"progress contains job outside this schedule: {job_key}")
        expected_identity = {
            "participant": job.participant.label,
            "cell_id": job.cell.cell_id,
            "panel_seed": job.panel_seed,
        }
        for key, value in expected_identity.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"durable job {job_key} {key} mismatch: {payload.get(key)!r} != {value!r}"
                )
        if int(payload.get("planned_games", -1)) != games:
            raise ValueError(f"durable job {job_key} planned-game mismatch")
        rows = payload.get("games") or []
        if int(payload.get("attempted", -1)) != games or int(payload.get("completed", -1)) != games:
            raise ValueError(f"durable job {job_key} is not a complete clean budget")
        if len(rows) != games:
            raise ValueError(f"durable job {job_key} game-row count mismatch")
        indices = {int(row.get("game_index", -1)) for row in rows}
        if indices != set(range(games)):
            raise ValueError(f"durable job {job_key} game indices are incomplete or duplicated")
        for row in rows:
            for key, value in expected_identity.items():
                if row.get(key) != value:
                    raise ValueError(f"durable game in {job_key} has mismatched {key}")
            if row.get("outcome") not in {"win", "loss", "draw"}:
                raise ValueError(f"durable job {job_key} contains non-completed outcome")
            if any(bool(row.get(key)) for key in ("crashed", "timed_out", "illegal")):
                raise ValueError(f"durable job {job_key} contains a safety failure")


def derive_game_seeds(panel_seed: int, games: int) -> list[int]:
    return [panel_seed * GAME_SEED_MULTIPLIER + game_index for game_index in range(games)]


def _agent_deck(agent: Any) -> list[int]:
    deck = list(getattr(agent, "deck", None) or getattr(getattr(agent, "inner", None), "deck", None) or [])
    if len(deck) != 60:
        raise ValueError(f"participant {getattr(agent, 'label', '?')!r} did not expose a 60-card deck")

    return [int(card_id) for card_id in deck]


def _agent_feature_policy_id(agent: Any) -> str | None:
    inner = getattr(agent, "inner", agent)
    policy = getattr(inner, "_policy", None)
    value = getattr(policy, "policy_id", None)

    return str(value) if value is not None else None


def build_participant_agents(
    participants: list[PanelParticipant],
    *,
    deck_policy: str,
    deck_binding: str,
    feature_policy_id: str = "",
    participant_feature_policy_ids: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    agents: dict[str, Any] = {}
    decks: dict[str, list[int]] = {}
    feature_bindings = participant_feature_policy_ids or {}
    for participant in participants:
        expected_feature_policy_id = feature_bindings.get(
            participant.label,
            feature_policy_id,
        )
        agent = r17.build_candidate(
            participant.spec,
            deck_policy,
            deck_binding,
            feature_policy_id=expected_feature_policy_id,
        )
        agents[participant.label] = agent
        decks[participant.label] = _agent_deck(agent)
        live_feature_policy_id = _agent_feature_policy_id(agent)
        if (
            expected_feature_policy_id
            and live_feature_policy_id is not None
            and live_feature_policy_id != expected_feature_policy_id
        ):
            raise ValueError(
                f"participant {participant.label!r} feature-policy mismatch: "
                f"expected {expected_feature_policy_id!r}, live {live_feature_policy_id!r}"
            )

    common_deck = decks[participants[0].label]
    mismatches = [label for label, deck in decks.items() if deck != common_deck]
    if mismatches:
        raise ValueError(f"shared panel requires an identical candidate/champion deck; mismatched: {mismatches}")

    return agents, common_deck


def _command_receipt(argv: list[str]) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), *argv]

    return {
        "argv": command,
        "display": subprocess.list2cmdline(command),
        "cwd": str(Path.cwd().resolve()),
    }


def build_config_receipt(
    args: argparse.Namespace,
    *,
    participants: list[PanelParticipant],
    cells: list[PanelCell],
    evaluation_deck: list[int],
    input_receipts: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any]:
    panel_config = {
        "run_id": args.run_id,
        "generation_id": getattr(args, "generation_id", ""),
        "snapshot_id": getattr(args, "snapshot_id", ""),
        "out_dir": str(Path(args.out_dir).resolve()),
        "participants": [asdict(participant) for participant in participants],
        "pools": str(Path(args.pools).resolve()),
        "stage": args.stage,
        "top_n": args.top_n,
        "carry_n": args.carry_n,
        "pilots": list(args.pilots),
        "exploiter_top": args.exploiter_top,
        "exploiter_checkpoint": str(Path(args.exploiter_checkpoint).resolve()) if "exploiter" in args.pilots else None,
        "deck_policy": args.deck_policy,
        "deck_binding": args.deck_binding,
        "feature_policy_id": (
            "per_participant"
            if getattr(args, "participant_feature_policy_ids_resolved", {})
            else args.feature_policy_id or "artifact/default"
        ),
        "participant_feature_policy_ids": dict(sorted(
            getattr(args, "participant_feature_policy_ids_resolved", {}).items()
        )),
        "evaluation_deck": {
            "card_ids": evaluation_deck,
            "normalized_deck_hash": normalized_deck_hash(evaluation_deck),
        },
        "games_per_cell_seed": args.games,
        "panel_seeds": list(args.seeds),
        "policy_rng_seed_formula": f"panel_seed * {GAME_SEED_MULTIPLIER} + zero_based_game_index",
        "engine_seed_control": "not_instrumented; vendored cg.game.battle_start exposes no seed parameter",
        "randomness_comparison": "independent equal-budget; not paired engine randomness",
        "job_order": "interleaved by cell, nominal batch seed, then participant",
        "seat_schedule": "candidate seat = zero_based_game_index % 2",
        "max_steps": args.max_steps,
        "wall_clock_timeout": "not_instrumented; max_steps is the only timeout guard",
        "cells": [cell.receipt() for cell in cells],
        "win_rate_denominator": "wins + losses (R17 convention; draws reported separately)",
        "interval": "Wilson 95% on wins / (wins + losses)",
        "severe_regression_rule": SEVERE_REGRESSION_RULE,
        "decision_role": "diagnostic only",
        "runtime_mutation": False,
        "input_receipts": input_receipts,
    }
    fingerprint = hashlib.sha256(_canonical_json(panel_config).encode("utf-8")).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at_utc": _utc_now(),
        "config_fingerprint": fingerprint,
        "panel_config": panel_config,
        "launch_command": command,
    }


def prepare_progress(
    out_dir: Path,
    config_receipt: dict[str, Any],
    *,
    resume: bool,
    command: dict[str, Any],
) -> dict[str, Any]:
    config_path = out_dir / "config.json"
    progress_path = out_dir / "progress.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"shared-panel run already exists; use --resume: {out_dir}")
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != config_receipt["config_fingerprint"]:
            raise ValueError("resume config/input hash mismatch; refusing to mix panels")
        if not progress_path.exists():
            raise ValueError("resume requested but progress.json is missing")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config_fingerprint") != config_receipt["config_fingerprint"]:
            raise ValueError("progress/config fingerprint mismatch")
        if progress.get("status") == "safety_failed":
            raise PanelSafetyError("prior panel safety failure is a hard stop; inspect existing artifacts")
        commands = progress.setdefault("commands", [])
        if command not in commands:
            commands.append(command)

        return progress

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is non-empty and has no config receipt: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "config_fingerprint": config_receipt["config_fingerprint"],
        "status": "running",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "elapsed_wall_seconds": 0.0,
        "commands": [command],
        "jobs": {},
    }
    write_json(config_path, config_receipt)
    write_json(progress_path, progress)

    return progress


def _build_opponent(job: PanelJob, args: argparse.Namespace) -> Any:
    if job.cell.pilot == "exploiter":
        return r17.build_exploiter(
            Path(args.exploiter_checkpoint),
            list(job.cell.card_ids),
            job.cell.deck_hash,
            args.deck_policy,
        )

    return SpecAgent.from_str(job.cell.pilot)


def _game_row(job: PanelJob, record: Any) -> dict[str, Any]:
    game_key = _stable_id({"job_key": job.job_key, "game_index": record.game_index})

    return {
        "schema_version": GAME_SCHEMA_VERSION,
        "game_key": game_key,
        "job_key": job.job_key,
        "participant": job.participant.label,
        "participant_role": job.participant.role,
        "cell_id": job.cell.cell_id,
        "bucket": job.cell.bucket,
        "carried": job.cell.carried,
        "opponent_deck_hash": job.cell.deck_hash,
        "opponent_shell": job.cell.shell,
        "opponent_pilot": job.cell.pilot,
        "panel_seed": job.panel_seed,
        "game_index": record.game_index,
        "rng_seed": record.rng_seed,
        "candidate_seat": record.seat_our,
        "result_seat": record.result,
        "outcome": record.outcome,
        "turns": record.turns,
        "crashed": record.crashed,
        "timed_out": record.timed_out,
        "illegal": record.illegal,
        "illegal_agent": record.illegal_agent,
        "elapsed_seconds": record.elapsed_seconds,
        "error": record.error,
    }


def _job_payload(job: PanelJob, summary: MatchupSummary) -> dict[str, Any]:
    games = [_game_row(job, record) for record in summary.records]

    return {
        "job_key": job.job_key,
        "participant": job.participant.label,
        "cell_id": job.cell.cell_id,
        "panel_seed": job.panel_seed,
        "planned_games": summary.games,
        "attempted": summary.attempted,
        "completed": summary.completed,
        "wins": summary.our_wins,
        "losses": summary.opp_wins,
        "draws": summary.draws,
        "crashes": summary.crashes,
        "timeouts": summary.timeouts,
        "illegal": summary.illegal,
        "avg_turns": summary.avg_turns,
        "elapsed_seconds": summary.elapsed_seconds,
        "stopped_early": summary.stopped_early,
        "errors": summary.errors,
        "games": games,
    }


def _stats(rows: list[dict[str, Any]], *, planned_games: int) -> dict[str, Any]:
    wins = sum(row["outcome"] == "win" for row in rows)
    losses = sum(row["outcome"] == "loss" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    crashes = sum(bool(row["crashed"]) for row in rows)
    timeouts = sum(bool(row["timed_out"]) for row in rows)
    illegal = sum(bool(row["illegal"]) for row in rows)
    incomplete = sum(row["outcome"] == "incomplete" for row in rows)
    completed = wins + losses + draws
    decided = wins + losses
    win_rate = wins / decided if decided else None
    point_rate = (wins + 0.5 * draws) / completed if completed else None
    interval = wilson_interval(wins, decided)
    elapsed = sum(float(row.get("elapsed_seconds") or 0.0) for row in rows)

    return {
        "planned_games": planned_games,
        "attempted": len(rows),
        "completed": completed,
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "crashes": crashes,
        "timeouts": timeouts,
        "illegal": illegal,
        "incomplete": incomplete,
        "win_rate": round(win_rate, 6) if win_rate is not None else None,
        "point_rate": round(point_rate, 6) if point_rate is not None else None,
        "wilson95": [round(bound, 6) for bound in interval],
        "elapsed_game_seconds": round(elapsed, 6),
        "completed_games_per_second": round(completed / elapsed, 6) if elapsed > 0 else None,
        "safety_clean": crashes == 0 and timeouts == 0 and illegal == 0 and incomplete == 0,
        "budget_complete": len(rows) == completed == planned_games,
    }


def _rows_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    return groups


def _participant_results(
    participants: list[PanelParticipant],
    cells: list[PanelCell],
    seeds: list[int],
    games: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_participant = _rows_by(rows, "participant")
    expected_total = len(cells) * len(seeds) * games
    result: dict[str, Any] = {}
    for participant in participants:
        participant_rows = by_participant.get(participant.label, [])
        by_cell = _rows_by(participant_rows, "cell_id")
        cell_results = []
        for cell in cells:
            cell_stats = _stats(by_cell.get(cell.cell_id, []), planned_games=len(seeds) * games)
            cell_results.append({**cell.receipt(), **cell_stats})
        eligible_worst = [row for row in cell_results if row["win_rate"] is not None]
        worst_cell = min(eligible_worst, key=lambda row: (row["win_rate"], row["cell_id"])) if eligible_worst else None
        result[participant.label] = {
            "label": participant.label,
            "role": participant.role,
            "spec": participant.spec,
            "league_member_id": participant.league_member_id or None,
            "aggregate": _stats(participant_rows, planned_games=expected_total),
            "worst_cell": worst_cell,
            "cells": cell_results,
        }

    return result


def _mean_and_se(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None

    return round(mean, 6), round(standard_error, 6) if standard_error is not None else None


def _independent_rate_delta_se(
    candidate_wins: int,
    candidate_total: int,
    champion_wins: int,
    champion_total: int,
) -> float | None:
    if candidate_total <= 0 or champion_total <= 0:
        return None
    candidate_rate = candidate_wins / candidate_total
    champion_rate = champion_wins / champion_total
    variance = (
        candidate_rate * (1.0 - candidate_rate) / candidate_total
        + champion_rate * (1.0 - champion_rate) / champion_total
    )

    return round(math.sqrt(variance), 6)


def _comparison_results(
    champion: PanelParticipant,
    candidates: list[PanelParticipant],
    participant_results: dict[str, Any],
    rows: list[dict[str, Any]],
    cells: list[PanelCell],
    seeds: list[int],
    games: int,
) -> list[dict[str, Any]]:
    by_participant = _rows_by(rows, "participant")
    champion_result = participant_results[champion.label]
    champion_cells = {row["cell_id"]: row for row in champion_result["cells"]}
    comparisons = []
    for candidate in candidates:
        candidate_result = participant_results[candidate.label]
        candidate_cells = {row["cell_id"]: row for row in candidate_result["cells"]}
        cell_deltas = []
        severe_cells = []
        for cell in cells:
            champion_cell = champion_cells[cell.cell_id]
            candidate_cell = candidate_cells[cell.cell_id]
            champion_wr = champion_cell["win_rate"]
            candidate_wr = candidate_cell["win_rate"]
            delta = candidate_wr - champion_wr if champion_wr is not None and candidate_wr is not None else None
            severe = (
                champion_cell["decided"] > 0
                and candidate_cell["decided"] > 0
                and candidate_cell["wilson95"][1] < champion_cell["wilson95"][0]
            )
            row = {
                "cell_id": cell.cell_id,
                "candidate_win_rate": candidate_wr,
                "champion_win_rate": champion_wr,
                "delta": round(delta, 6) if delta is not None else None,
                "candidate_wilson95": candidate_cell["wilson95"],
                "champion_wilson95": champion_cell["wilson95"],
                "severe_regression": severe,
            }
            cell_deltas.append(row)
            if severe:
                severe_cells.append(row)

        independent_batch_deltas = []
        champion_rows = by_participant.get(champion.label, [])
        candidate_rows = by_participant.get(candidate.label, [])
        for cell in cells:
            for panel_seed in seeds:
                champion_batch = [
                    row for row in champion_rows
                    if row["cell_id"] == cell.cell_id and row["panel_seed"] == panel_seed
                ]
                candidate_batch = [
                    row for row in candidate_rows
                    if row["cell_id"] == cell.cell_id and row["panel_seed"] == panel_seed
                ]
                champion_stats = _stats(champion_batch, planned_games=games)
                candidate_stats = _stats(candidate_batch, planned_games=games)
                if champion_stats["win_rate"] is not None and candidate_stats["win_rate"] is not None:
                    independent_batch_deltas.append(
                        candidate_stats["win_rate"] - champion_stats["win_rate"]
                    )
        independent_batch_mean, independent_batch_se = _mean_and_se(independent_batch_deltas)

        champion_aggregate = champion_result["aggregate"]
        candidate_aggregate = candidate_result["aggregate"]
        aggregate_delta = None
        if champion_aggregate["win_rate"] is not None and candidate_aggregate["win_rate"] is not None:
            aggregate_delta = candidate_aggregate["win_rate"] - champion_aggregate["win_rate"]
        expected_games = len(cells) * len(seeds) * games
        equal_budget = (
            champion_aggregate["attempted"] == expected_games
            and candidate_aggregate["attempted"] == expected_games
            and champion_aggregate["completed"] == expected_games
            and candidate_aggregate["completed"] == expected_games
            and champion_aggregate["safety_clean"]
            and candidate_aggregate["safety_clean"]
        )
        independent_delta_se = _independent_rate_delta_se(
            candidate_aggregate["wins"],
            candidate_aggregate["decided"],
            champion_aggregate["wins"],
            champion_aggregate["decided"],
        )
        candidate_wins_per_second = (
            candidate_aggregate["wins"] / candidate_aggregate["elapsed_game_seconds"]
            if candidate_aggregate["elapsed_game_seconds"] > 0 else None
        )
        champion_wins_per_second = (
            champion_aggregate["wins"] / champion_aggregate["elapsed_game_seconds"]
            if champion_aggregate["elapsed_game_seconds"] > 0 else None
        )
        marginal_wins_per_second = (
            candidate_wins_per_second - champion_wins_per_second
            if candidate_wins_per_second is not None and champion_wins_per_second is not None else None
        )
        available_deltas = [row["delta"] for row in cell_deltas if row["delta"] is not None]
        comparisons.append({
            "candidate": candidate.label,
            "champion": champion.label,
            "candidate_member_id": candidate.league_member_id or None,
            "champion_member_id": champion.league_member_id or None,
            "equal_completed_budget": equal_budget,
            "aggregate_win_rate_delta": round(aggregate_delta, 6) if aggregate_delta is not None else None,
            "candidate_aggregate_wilson95": candidate_aggregate["wilson95"],
            "champion_aggregate_wilson95": champion_aggregate["wilson95"],
            "randomness_comparison": "independent equal-budget; engine RNG is not seedable",
            "independent_aggregate_delta_se": independent_delta_se,
            "independent_cell_seed_batch_delta_mean": independent_batch_mean,
            "independent_cell_seed_batch_delta_se": independent_batch_se,
            "independent_cell_seed_batches": len(independent_batch_deltas),
            "candidate_local_wins_per_second": round(candidate_wins_per_second, 6)
            if candidate_wins_per_second is not None else None,
            "champion_local_wins_per_second": round(champion_wins_per_second, 6)
            if champion_wins_per_second is not None else None,
            "marginal_local_wins_per_second": round(marginal_wins_per_second, 6)
            if marginal_wins_per_second is not None else None,
            "regression_cell_count": sum(delta < 0 for delta in available_deltas),
            "maximum_cell_regression": min(available_deltas) if available_deltas else None,
            "severe_regression_rule": SEVERE_REGRESSION_RULE,
            "severe_regression_count": len(severe_cells),
            "severe_regression_cells": severe_cells,
            "cell_deltas": cell_deltas,
            "decision": "diagnostic only",
        })

    return comparisons


def build_results(
    config_receipt: dict[str, Any],
    progress: dict[str, Any],
    *,
    champion: PanelParticipant,
    candidates: list[PanelParticipant],
    cells: list[PanelCell],
    seeds: list[int],
    games: int,
    jobs: list[PanelJob],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_keys = {job.job_key for job in jobs}
    persisted_keys = set(progress.get("jobs") or {})
    unexpected = sorted(persisted_keys - expected_keys)
    if unexpected:
        raise ValueError(f"progress contains jobs outside this schedule: {unexpected[:3]}")

    game_rows = [
        game
        for job_key in sorted(persisted_keys)
        for game in progress["jobs"][job_key].get("games", [])
    ]
    game_keys = [row["game_key"] for row in game_rows]
    if len(game_keys) != len(set(game_keys)):
        raise ValueError("duplicate game keys detected in durable progress")

    participants = [champion, *candidates]
    participant_results = _participant_results(participants, cells, seeds, games, game_rows)
    complete = persisted_keys == expected_keys
    safety_clean = all(result["aggregate"]["safety_clean"] for result in participant_results.values())
    status = "safety_failed" if progress.get("status") == "safety_failed" else "completed" if complete else "partial"
    comparisons = _comparison_results(
        champion, candidates, participant_results, game_rows, cells, seeds, games,
    )
    is_comparable = complete and safety_clean and all(
        comparison["equal_completed_budget"] for comparison in comparisons
    )
    if is_comparable:
        audit_benefit = (
            "bounded local marginal evidence is recorded in comparisons; "
            "true Kaggle benefit remains unmeasured"
        )
        audit_next_action = (
            "consume the completed comparison through an explicit promotion/rejection gate; "
            "do not rerun merely to seek a preferred point estimate"
        )
    elif status == "safety_failed":
        audit_benefit = "unmeasured because the bounded panel failed its safety gate"
        audit_next_action = "hard stop and diagnose the safety failure before any selection claim"
    else:
        audit_benefit = "unmeasured because the bounded panel is incomplete"
        audit_next_action = "resume the exact receipt-bound panel before any selection claim"
    receipts = config_receipt["panel_config"]["input_receipts"]
    deck_sha256 = (receipts.get("deck_binding") or {}).get("sha256")
    candidate_ids = {candidate.label: candidate.league_member_id or None for candidate in candidates}
    candidate_hashes = {
        candidate.label: (receipts.get(f"participant:{candidate.label}") or {}).get("sha256")
        for candidate in candidates
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config_receipt["run_id"],
        "created_at_utc": config_receipt["created_at_utc"],
        "updated_at_utc": _utc_now(),
        "status": status,
        "config_fingerprint": config_receipt["config_fingerprint"],
        "artifact_path": config_receipt["panel_config"]["out_dir"],
        "commands": progress.get("commands") or [config_receipt["launch_command"]],
        "schedule": {
            "participants": len(participants),
            "cells": len(cells),
            "panel_seeds": seeds,
            "games_per_cell_seed": games,
            "jobs_expected": len(jobs),
            "jobs_completed": len(persisted_keys),
            "games_expected_per_participant": len(cells) * len(seeds) * games,
            "identical_cells_seeds_games": True,
            "complete": complete,
        },
        "safety_clean": safety_clean,
        "comparable": is_comparable,
        "participants": participant_results,
        "comparisons": comparisons,
        "member_identities": {
            "generation_id": config_receipt["panel_config"].get("generation_id") or None,
            "snapshot_id": config_receipt["panel_config"].get("snapshot_id") or None,
            "champion_member_id": champion.league_member_id or None,
            "candidate_member_id": (
                candidates[0].league_member_id if len(candidates) == 1 else None
            ) or None,
            "candidate_member_ids": candidate_ids,
            "champion_checkpoint_sha256": (
                receipts.get(f"participant:{champion.label}") or {}
            ).get("sha256"),
            "candidate_checkpoint_sha256": (
                candidate_hashes[candidates[0].label] if len(candidates) == 1 else None
            ),
            "candidate_checkpoint_sha256_by_label": candidate_hashes,
            "champion_deck_sha256": deck_sha256,
            "candidate_deck_sha256": deck_sha256 if len(candidates) == 1 else None,
            "feature_policy_id": config_receipt["panel_config"].get("feature_policy_id"),
        },
        "elapsed_wall_seconds": progress.get("elapsed_wall_seconds", 0.0),
        "feature_contribution_audit": {
            "feature": "shared-panel evaluator",
            "purpose": "measure candidate marginal value against the champion under one equal evaluation budget",
            "baseline": champion.label,
            "benefit": audit_benefit,
            "cost": "reported as elapsed wall/game seconds and completed games per second",
            "damage": "safety counters and severe per-cell regressions",
            "uniqueness": "one interleaved equal-budget schedule removes the prior 32-game versus 64-game confound",
            "precision": "Wilson 95% intervals",
            "recall": "all configured R17 cells, pilots, and seeds",
            "mean_true_objective_delta": "unavailable; local arena is not Kaggle truth",
            "throughput_impact": "reported per participant",
            "marginal_value_per_second": "see comparisons[].marginal_local_wins_per_second; local arena only",
            "failure_clustering": "reported by R17 cell",
            "decision": "diagnostic only",
            "next_action": audit_next_action,
        },
        "runtime_mutated": False,
    }

    return results, game_rows


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# League shared-panel evaluation",
        "",
        f"- status: **{results['status']}**",
        f"- comparable: `{results['comparable']}`; safety_clean: `{results['safety_clean']}`",
        f"- decision role: **diagnostic only**; runtime/default policy unchanged",
        f"- config fingerprint: `{results['config_fingerprint']}`",
        f"- artifact path: `{results['artifact_path']}`",
        "- engine seed control: `not_instrumented` (the current cg.game.battle_start wrapper has no seed parameter)",
        "- randomness comparison: **independent equal-budget**, not paired engine randomness",
        "- wall-clock timeout: `not_instrumented`; max-step timeout only",
        "",
        "## Exact command",
        "",
        "```text",
        results["commands"][-1]["display"],
        "```",
        "",
        "## Participants",
        "",
        "| participant | role | attempted/planned | W-L-D | safety C/T/I/U | win rate | Wilson 95% | worst cell |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for label, participant in results["participants"].items():
        aggregate = participant["aggregate"]
        worst = participant["worst_cell"]
        worst_text = "unavailable" if worst is None else f"{worst['cell_id']} ({worst['win_rate']})"
        lines.append(
            f"| `{label}` | {participant['role']} | {aggregate['attempted']}/{aggregate['planned_games']} "
            f"| {aggregate['wins']}-{aggregate['losses']}-{aggregate['draws']} "
            f"| {aggregate['crashes']}/{aggregate['timeouts']}/{aggregate['illegal']}/"
            f"{aggregate['incomplete']} "
            f"| {aggregate['win_rate']} | {aggregate['wilson95']} | {worst_text} |"
        )
    lines.extend([
        "",
        "## Candidate versus champion",
        "",
        "| candidate | aggregate delta +/- independent SE | independent cell-seed batch mean +/- SE | severe regressions | equal budget |",
        "|---|---:|---:|---:|---:|",
    ])
    for comparison in results["comparisons"]:
        lines.append(
            f"| `{comparison['candidate']}` | {comparison['aggregate_win_rate_delta']} +/- "
            f"{comparison['independent_aggregate_delta_se']} "
            f"| {comparison['independent_cell_seed_batch_delta_mean']} +/- "
            f"{comparison['independent_cell_seed_batch_delta_se']} "
            f"| {comparison['severe_regression_count']} | {comparison['equal_completed_budget']} |"
        )
    lines.extend([
        "",
        "Severe regression rule: " + SEVERE_REGRESSION_RULE + ".",
        "",
        "No checkpoint selection, runtime packaging, or submission action is performed by this evaluator.",
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
        kind="league_shared_panel",
        command=(progress.get("commands") or [config_receipt["launch_command"]])[-1]["argv"],
        status=results["status"],
        inputs={
            "config": "config.json",
            "config_fingerprint": config_receipt["config_fingerprint"],
            "receipts": config_receipt["panel_config"]["input_receipts"],
        },
        outputs={
            "progress": "progress.json",
            "games": "games.jsonl",
            "results": "results.json",
            "report": "report.md",
        },
        summary={
            "status": results["status"],
            "comparable": results["comparable"],
            "safety_clean": results["safety_clean"],
            "jobs_completed": results["schedule"]["jobs_completed"],
            "jobs_expected": results["schedule"]["jobs_expected"],
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)


def _validate_args(args: argparse.Namespace, champion: PanelParticipant, candidates: list[PanelParticipant]) -> None:
    participants = [champion, *candidates]
    labels = [participant.label for participant in participants]
    if len(labels) != len(set(labels)):
        raise ValueError("champion/candidate labels must be unique")
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.top_n <= 0 or args.carry_n < 0 or args.exploiter_top < 0:
        raise ValueError("--top-n must be positive and --carry-n/--exploiter-top must be non-negative")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must not contain duplicates")
    if len(args.pilots) != len(set(args.pilots)):
        raise ValueError("--pilots must not contain duplicates")
    if "exploiter" in args.pilots and args.exploiter_top <= 0:
        raise ValueError("exploiter pilot requires --exploiter-top > 0")


def run_panel(args: argparse.Namespace, *, command_argv: list[str] | None = None) -> dict[str, Any]:
    candidate_bindings = parse_candidate_member_ids(
        getattr(args, "candidate_member_id", []) or []
    )
    champion = parse_participant(
        args.champion,
        role="champion",
        league_member_id=getattr(args, "champion_member_id", ""),
    )
    candidates = []
    for value in args.candidate:
        parsed = parse_participant(value, role="candidate")
        candidates.append(PanelParticipant(
            label=parsed.label,
            spec=parsed.spec,
            role=parsed.role,
            league_member_id=candidate_bindings.get(parsed.label, ""),
        ))
    unknown_bindings = set(candidate_bindings).difference(candidate.label for candidate in candidates)
    if unknown_bindings:
        raise ValueError(f"candidate member bindings have no participant: {sorted(unknown_bindings)}")
    generation_values = {
        "generation_id": getattr(args, "generation_id", ""),
        "snapshot_id": getattr(args, "snapshot_id", ""),
        "champion_member_id": champion.league_member_id,
    }
    generation_mode = any(generation_values.values()) or bool(candidate_bindings)
    if generation_mode:
        missing = [key for key, value in generation_values.items() if not value]
        missing.extend(
            f"candidate_member_id:{candidate.label}"
            for candidate in candidates
            if not candidate.league_member_id
        )
        if not getattr(args, "feature_policy_id", ""):
            missing.append("feature_policy_id")
        if not getattr(args, "deck_binding", ""):
            missing.append("deck_binding")
        if missing:
            raise ValueError(f"generation-bound shared panel is missing: {missing}")
    _validate_args(args, champion, candidates)
    participants = [champion, *candidates]
    participant_feature_policy_ids = parse_participant_feature_policy_ids(
        getattr(args, "participant_feature_policy_id", []) or []
    )
    participant_labels = {participant.label for participant in participants}
    unknown_feature_bindings = set(participant_feature_policy_ids).difference(participant_labels)
    if unknown_feature_bindings:
        raise ValueError(
            "participant feature-policy bindings have no participant: "
            f"{sorted(unknown_feature_bindings)}"
        )
    if args.feature_policy_id and participant_feature_policy_ids:
        raise ValueError(
            "use either --feature-policy-id or --participant-feature-policy-id, not both"
        )
    if participant_feature_policy_ids and set(participant_feature_policy_ids) != participant_labels:
        missing_labels = sorted(participant_labels.difference(participant_feature_policy_ids))
        raise ValueError(
            f"participant feature-policy bindings are incomplete: {missing_labels}"
        )
    args.participant_feature_policy_ids_resolved = participant_feature_policy_ids
    command = _command_receipt(command_argv if command_argv is not None else sys.argv[1:])
    cells = load_panel_cells(
        args.pools,
        stage_name=args.stage,
        top_n=args.top_n,
        carry_n=args.carry_n,
        pilots=list(args.pilots),
        exploiter_top=args.exploiter_top,
    )
    agents, evaluation_deck = build_participant_agents(
        participants,
        deck_policy=args.deck_policy,
        deck_binding=args.deck_binding,
        feature_policy_id=args.feature_policy_id,
        participant_feature_policy_ids=participant_feature_policy_ids,
    )
    input_receipts = build_input_receipts(
        pools_path=args.pools,
        participants=participants,
        deck_binding=args.deck_binding,
        exploiter_checkpoint=args.exploiter_checkpoint,
        include_exploiter="exploiter" in args.pilots,
    )
    config_receipt = build_config_receipt(
        args,
        participants=participants,
        cells=cells,
        evaluation_deck=evaluation_deck,
        input_receipts=input_receipts,
        command=command,
    )
    out_dir = Path(args.out_dir).resolve()
    progress = prepare_progress(out_dir, config_receipt, resume=args.resume, command=command)
    config_receipt = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    jobs = build_jobs(participants, cells, list(args.seeds))
    expected_keys = {job.job_key for job in jobs}
    unexpected = set(progress.get("jobs") or {}) - expected_keys
    if unexpected:
        raise ValueError(f"progress contains unexpected job keys: {sorted(unexpected)[:3]}")
    validate_durable_jobs(progress, jobs, games=args.games)

    base_elapsed = float(progress.get("elapsed_wall_seconds") or 0.0)
    started = time.perf_counter()
    for job in jobs:
        if job.job_key in progress["jobs"]:
            continue
        opponent = _build_opponent(job, args)
        summary = run_real_meta_matchup(
            agents[job.participant.label],
            opponent,
            evaluation_deck,
            list(job.cell.card_ids),
            games=args.games,
            bucket=job.cell.bucket,
            opp_deck_hash=job.cell.deck_hash,
            opp_archetype=job.cell.shell,
            seed=job.panel_seed,
            max_steps=args.max_steps,
            game_seeds=derive_game_seeds(job.panel_seed, args.games),
            stop_on_safety_failure=True,
        )
        payload = _job_payload(job, summary)
        if job.job_key in progress["jobs"]:
            raise ValueError(f"duplicate completed job refused: {job.job_key}")
        progress["jobs"][job.job_key] = payload
        progress["elapsed_wall_seconds"] = round(base_elapsed + time.perf_counter() - started, 6)
        progress["updated_at_utc"] = _utc_now()
        safety = {key: payload[key] for key in ("crashes", "timeouts", "illegal") if payload[key]}
        if payload["completed"] != payload["planned_games"]:
            safety["incomplete"] = payload["planned_games"] - payload["completed"]
        if safety:
            progress["status"] = "safety_failed"
        else:
            validate_durable_jobs(progress, jobs, games=args.games)
        write_json(out_dir / "progress.json", progress)
        print(
            f"[{job.participant.label}] {job.cell.cell_id} seed={job.panel_seed} "
            f"{payload['wins']}-{payload['losses']}-{payload['draws']} "
            f"safety={payload['crashes']}/{payload['timeouts']}/{payload['illegal']}",
            flush=True,
        )
        if safety:
            results, game_rows = build_results(
                config_receipt,
                progress,
                champion=champion,
                candidates=candidates,
                cells=cells,
                seeds=list(args.seeds),
                games=args.games,
                jobs=jobs,
            )
            write_outputs(out_dir, config_receipt, progress, results, game_rows)
            raise PanelSafetyError(f"shared panel hard-stopped on safety failure: {safety}")

    progress["status"] = "completed"
    progress["elapsed_wall_seconds"] = round(base_elapsed + time.perf_counter() - started, 6)
    progress["updated_at_utc"] = _utc_now()
    write_json(out_dir / "progress.json", progress)
    results, game_rows = build_results(
        config_receipt,
        progress,
        champion=champion,
        candidates=candidates,
        cells=cells,
        seeds=list(args.seeds),
        games=args.games,
        jobs=jobs,
    )
    write_outputs(out_dir, config_receipt, progress, results, game_rows)
    print(json.dumps({
        "status": results["status"],
        "comparable": results["comparable"],
        "safety_clean": results["safety_clean"],
        "comparisons": results["comparisons"],
        "artifacts": str(out_dir),
    }, indent=2, sort_keys=True))

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", required=True, help="R17 bucket_pools.json")
    parser.add_argument("--champion", required=True, help="LABEL=SPEC current research champion")
    parser.add_argument("--candidate", action="append", required=True, help="LABEL=SPEC; repeat for candidates")
    parser.add_argument(
        "--deck-binding",
        default="",
        help="shared R17 checkpoint deck binding (bundle dir or deck:<csv>)",
    )
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument(
        "--feature-policy-id",
        default="",
        help="shared categorical policy ID used by every participant feature encoder",
    )
    parser.add_argument(
        "--participant-feature-policy-id",
        action="append",
        default=[],
        help=(
            "LABEL=POLICY_ID exact per-participant categorical feature identity; repeat for every "
            "participant when their frozen identities differ"
        ),
    )
    parser.add_argument("--stage", default="600-800")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--carry-n", type=int, default=4)
    parser.add_argument("--pilots", nargs="+", choices=PILOT_CHOICES, default=["greedy", "v2.4"])
    parser.add_argument("--exploiter-top", type=int, default=0)
    parser.add_argument("--exploiter-checkpoint", default=str(r17.DEFAULT_CHAMPION))
    parser.add_argument("--games", type=int, default=64, help="games per cell per panel seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[181818], help="shared batch seeds")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out-dir", required=True, help="stable durable run directory")
    parser.add_argument("--run-id", required=True, help="stable run id recorded in receipts")
    parser.add_argument("--generation-id", default="", help="league generation receipt identity")
    parser.add_argument("--snapshot-id", default="", help="frozen league snapshot identity")
    parser.add_argument("--champion-member-id", default="", help="immutable champion registry ID")
    parser.add_argument(
        "--candidate-member-id",
        action="append",
        default=[],
        help="LABEL=MEMBER_ID immutable candidate binding; repeat for candidates",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    final_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(final_argv)
    try:
        run_panel(args, command_argv=final_argv)
    except PanelSafetyError as exc:
        print(f"SAFETY HARD STOP: {exc}", file=sys.stderr)

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
