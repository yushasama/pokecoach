"""Run the predeclared fixed-v2.4 D0/D1/D3 deck-substrate control.

The harness changes only the 60-card deck supplied to CABT. Every arm uses the
same named ``v2.4`` selector and the same opponent cells, panel seeds, and game
budget. It is eval-only and cannot package or select a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_league_shared_panel as shared  # noqa: E402
import run_r17_bucket_arena as r17  # noqa: E402
from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.writers import write_json, write_markdown  # noqa: E402
from deepmew.real_meta_arena.decklists import ZooDeck, validate_decklist  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402


SCHEMA_VERSION = "deepmew.fixed_policy_deck_control.v1"
PROGRESS_SCHEMA_VERSION = "deepmew.fixed_policy_deck_control_progress.v1"
EXPECTED_REGISTRY_SHA256 = "7bc9629b316c980afde4834de15996d5b3c370ca271387a740f238237deff308"
EXPECTED_POOL_SHA256 = "c37bcc7b99d5b4c967c6ac0fe78b1b08c6b35e13095eceb10731fb43dbecde71"
EXPECTED_EXPLOITER_SHA256 = "ffd5dcf3104b015a5e11ed68bc194dccc0bbac072055a12f672f1b9236fddb8a"
REGISTRY_PATH = ROOT / ".research" / "DECK_REGISTRY.json"
POOL_PATH = ROOT / ".research" / "runs" / "bucket_pool" / "20260707" / "20260707T141406Z-bucket_pool-1c29c3dd" / "bucket_pools.json"
EXPLOITER_PATH = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
OUTPUT_ROOT = ROOT / ".research" / "deck_control_v0"
SMOKE_OUT_DIR = OUTPUT_ROOT / "d0_d1_d3_v24_20260712_smoke"
PRODUCTION_OUT_DIR = OUTPUT_ROOT / "d0_d1_d3_v24_20260712"
SMOKE_SEEDS = (9_271_199,)
PRODUCTION_SEEDS = (9_271_201, 9_271_202, 9_271_203)
SMOKE_GAMES = 1
PRODUCTION_GAMES = 32
STAGE = "600-800"
TOP_N = 8
CARRY_N = 4
PILOTS = ("greedy", "v2.4", "exploiter")
EXPLOITER_TOP = 4
MAX_STEPS = 5_000
SEVERE_REGRESSION_DELTA = -0.10
NONINFERIORITY_MARGIN = -0.03
ARM_IDS = ("D0", "D1", "D3")
FORBIDDEN_PERSISTED_KEYS = frozenset({
    "observation",
    "observations",
    "hand",
    "deck",
    "prize",
    "search_begin_input",
    "private_state",
    "private_hash",
})
PROTECTED_PATHS = (
    ROOT / "deck.csv",
    ROOT / "main.py",
    ROOT / "cg",
    ROOT / "deepmew" / "submissions",
)


class DeckControlSafetyError(RuntimeError):
    """Hard stop after a safety, identity, or protected-surface failure."""


@dataclass(frozen=True)
class DeckArm:
    arm_id: str
    name: str
    card_ids: tuple[int, ...]
    source_path: str
    source_sha256: str
    deck_file_sha256: str
    normalized_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "name": self.name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "deck_file_sha256": self.deck_file_sha256,
            "normalized_sha256": self.normalized_sha256,
            "card_count": len(self.card_ids),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _require_hash(actual: str, expected: str, label: str) -> None:
    if actual.lower() != expected.lower():
        raise DeckControlSafetyError(f"{label} SHA-256 mismatch")


def _parse_deck_bytes(payload: bytes, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(line.strip()) for line in payload.decode("utf-8").splitlines() if line.strip())
    except (UnicodeDecodeError, ValueError) as error:
        raise DeckControlSafetyError(f"{label} deck.csv is not a valid integer list") from error
    if len(values) != 60:
        raise DeckControlSafetyError(f"{label} deck.csv contains {len(values)} cards, expected 60")

    return values


def _tar_deck_bytes(path: Path) -> bytes:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.name == "deck.csv" and member.isfile()]
            if len(members) != 1:
                raise DeckControlSafetyError("submission archive must contain exactly one root deck.csv")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise DeckControlSafetyError("submission archive deck.csv is unreadable")

            return stream.read()
    except (OSError, tarfile.TarError) as error:
        raise DeckControlSafetyError("submission archive could not be read") from error


def _validate_engine_deck(arm_id: str, cards: tuple[int, ...]) -> None:
    deck = ZooDeck(
        deck_hash=normalized_deck_hash(list(cards)),
        archetype=arm_id,
        split="eval",
        is_canonical=True,
        card_ids=cards,
        card_count=len(cards),
    )
    receipt = validate_decklist(deck, check_engine=True)
    if not receipt.valid or receipt.engine_accepted is not True:
        raise DeckControlSafetyError(f"{arm_id} failed engine deck validation: {receipt.errors}")


def load_deck_arms(
    registry_path: Path = REGISTRY_PATH,
    *,
    root: Path = ROOT,
    require_registry_hash: bool = True,
) -> dict[str, DeckArm]:
    registry_path = Path(registry_path).resolve()
    if require_registry_hash:
        _require_hash(_sha256_file(registry_path), EXPECTED_REGISTRY_SHA256, "deck registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "deepmew.deck_registry.v1":
        raise DeckControlSafetyError("deck registry schema mismatch")
    entries = {str(entry.get("id")): entry for entry in registry.get("entries") or []}
    if any(arm_id not in entries for arm_id in ARM_IDS):
        raise DeckControlSafetyError("deck registry is missing D0, D1, or D3")
    arms: dict[str, DeckArm] = {}
    for arm_id in ARM_IDS:
        entry = entries[arm_id]
        artifact = entry.get("artifact_path")
        deck_path = entry.get("deck_path")
        if bool(artifact) == bool(deck_path):
            raise DeckControlSafetyError(f"{arm_id} must bind exactly one deck source")
        if artifact:
            source = (root / str(artifact)).resolve()
            source_hash = _sha256_file(source)
            _require_hash(source_hash, str(entry["artifact_sha256"]), f"{arm_id} artifact")
            payload = _tar_deck_bytes(source)
            expected_deck_hash = str(entry.get("deck_file_sha256") or entry.get("deck_sha256") or "")
        else:
            source = (root / str(deck_path)).resolve()
            payload = source.read_bytes()
            source_hash = _sha256_bytes(payload)
            expected_deck_hash = str(entry.get("deck_file_sha256") or "")
        deck_file_hash = _sha256_bytes(payload)
        _require_hash(deck_file_hash, expected_deck_hash, f"{arm_id} deck.csv")
        cards = _parse_deck_bytes(payload, arm_id)
        normalized_hash = normalized_deck_hash(list(cards))
        expected_normalized = entry.get("normalized_deck_sha256")
        if expected_normalized:
            _require_hash(normalized_hash, str(expected_normalized), f"{arm_id} normalized deck")
        _validate_engine_deck(arm_id, cards)
        arms[arm_id] = DeckArm(
            arm_id=arm_id,
            name=str(entry.get("name") or arm_id),
            card_ids=cards,
            source_path=str(source),
            source_sha256=source_hash,
            deck_file_sha256=deck_file_hash,
            normalized_sha256=normalized_hash,
        )

    return arms


def _tree_receipt(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)}
    if not path.is_dir():
        raise DeckControlSafetyError(f"protected path is missing: {path}")
    receipt = {}
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        key = str(file_path.relative_to(ROOT)).replace("\\", "/")
        receipt[key] = _sha256_file(file_path)

    return receipt


def protected_surface_receipt() -> dict[str, str]:
    receipt: dict[str, str] = {}
    for path in PROTECTED_PATHS:
        path_receipt = _tree_receipt(path)
        overlap = set(receipt).intersection(path_receipt)
        if overlap:
            raise DeckControlSafetyError(f"duplicate protected receipt paths: {sorted(overlap)[:3]}")
        receipt.update(path_receipt)

    return receipt


def _assert_protected_unchanged(expected: dict[str, str]) -> None:
    if protected_surface_receipt() != expected:
        raise DeckControlSafetyError("protected runtime/submission surfaces changed during the panel")


def _config(mode: str, out_dir: Path, arms: dict[str, DeckArm], protected: dict[str, str]) -> dict[str, Any]:
    seeds = SMOKE_SEEDS if mode == "smoke" else PRODUCTION_SEEDS
    games = SMOKE_GAMES if mode == "smoke" else PRODUCTION_GAMES
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "classification": "diagnostic_only",
        "policy": "v2.4",
        "arms": [arms[arm_id].receipt() for arm_id in ARM_IDS],
        "pools": {"path": str(POOL_PATH), "sha256": EXPECTED_POOL_SHA256},
        "exploiter": {"path": str(EXPLOITER_PATH), "sha256": EXPECTED_EXPLOITER_SHA256},
        "stage": STAGE,
        "top_n": TOP_N,
        "carry_n": CARRY_N,
        "pilots": list(PILOTS),
        "exploiter_top": EXPLOITER_TOP,
        "panel_seeds": list(seeds),
        "games_per_cell_seed": games,
        "max_steps": MAX_STEPS,
        "severe_regression_delta": SEVERE_REGRESSION_DELTA,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "out_dir": str(out_dir),
        "protected_surface_receipt": protected,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    payload["config_fingerprint"] = _sha256_bytes(canonical)

    return payload


def _prepare_progress(out_dir: Path, config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    config_path = out_dir / "config.json"
    progress_path = out_dir / "progress.json"
    if resume:
        if not config_path.is_file() or not progress_path.is_file():
            raise ValueError("--resume requires complete config.json and progress.json")
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        if prior_config != config:
            raise ValueError("resume config does not match the frozen deck-control contract")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise ValueError("deck-control progress schema mismatch")
        if progress.get("status") == "safety_failed":
            raise DeckControlSafetyError("prior safety failure is a hard stop")

        return progress
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("new deck-control output directory must be absent or empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "running",
        "config_fingerprint": config["config_fingerprint"],
        "jobs": {},
        "elapsed_wall_seconds": 0.0,
    }
    write_json(config_path, config)
    write_json(progress_path, progress)

    return progress


def _build_opponent(job: shared.PanelJob) -> Any:
    if job.cell.pilot == "exploiter":
        return r17.build_exploiter(
            EXPLOITER_PATH,
            list(job.cell.card_ids),
            job.cell.deck_hash,
            "reference_abomasnow",
        )

    return SpecAgent.from_str(job.cell.pilot)


def _confidence_interval(mean: float | None, standard_error: float | None) -> list[float] | None:
    if mean is None or standard_error is None:
        return None

    return [round(mean - 1.959964 * standard_error, 6), round(mean + 1.959964 * standard_error, 6)]


def _enhance_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    aggregate_ci = _confidence_interval(
        comparison["aggregate_win_rate_delta"],
        comparison["independent_aggregate_delta_se"],
    )
    batch_ci = _confidence_interval(
        comparison["independent_cell_seed_batch_delta_mean"],
        comparison["independent_cell_seed_batch_delta_se"],
    )
    threshold_cells = [
        row for row in comparison["cell_deltas"]
        if row["delta"] is not None and row["delta"] < SEVERE_REGRESSION_DELTA
    ]

    return {
        **comparison,
        "independent_aggregate_delta_ci95": aggregate_ci,
        "independent_cell_seed_batch_delta_ci95": batch_ci,
        "threshold_severe_regression_delta": SEVERE_REGRESSION_DELTA,
        "threshold_severe_regression_count": len(threshold_cells),
        "threshold_severe_regression_cells": threshold_cells,
    }


def _decide(mode: str, d1_d0: dict[str, Any], d1_d3: dict[str, Any]) -> dict[str, Any]:
    if mode == "smoke":
        return {
            "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT",
            "d1_noninferior_to_d3": None,
        }
    aggregate_ci = d1_d0["independent_aggregate_delta_ci95"]
    batch_ci = d1_d0["independent_cell_seed_batch_delta_ci95"]
    if aggregate_ci is None or batch_ci is None:
        verdict = "AMBIGUOUS"
    elif (
        aggregate_ci[0] > 0.0
        and batch_ci[0] > 0.0
        and d1_d0["threshold_severe_regression_count"] == 0
    ):
        verdict = "D1_FIXED_POLICY_GAIN"
    elif aggregate_ci[1] < 0.0 and batch_ci[1] < 0.0:
        verdict = "D1_FIXED_POLICY_LOSS"
    else:
        verdict = "AMBIGUOUS"
    d1_d3_aggregate = d1_d3["independent_aggregate_delta_ci95"]
    d1_d3_batch = d1_d3["independent_cell_seed_batch_delta_ci95"]
    noninferior = (
        verdict == "D1_FIXED_POLICY_GAIN"
        and d1_d3_aggregate is not None
        and d1_d3_batch is not None
        and d1_d3_aggregate[0] > NONINFERIORITY_MARGIN
        and d1_d3_batch[0] > NONINFERIORITY_MARGIN
    )

    return {
        "verdict": verdict,
        "d1_noninferior_to_d3": noninferior if verdict == "D1_FIXED_POLICY_GAIN" else None,
    }


def _assert_no_forbidden_persisted_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_PERSISTED_KEYS:
                raise DeckControlSafetyError(f"forbidden persisted field at {path}.{key}")
            _assert_no_forbidden_persisted_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_persisted_keys(child, f"{path}[{index}]")


def _build_results(
    mode: str,
    config: dict[str, Any],
    progress: dict[str, Any],
    participants: list[shared.PanelParticipant],
    cells: list[shared.PanelCell],
    seeds: list[int],
    games: int,
    jobs: list[shared.PanelJob],
) -> dict[str, Any]:
    game_rows = [
        game
        for job_key in sorted(progress["jobs"])
        for game in progress["jobs"][job_key].get("games", [])
    ]
    participant_results = shared._participant_results(participants, cells, seeds, games, game_rows)
    d0 = participants[0]
    d1 = participants[1]
    d3 = participants[2]
    control_comparisons = shared._comparison_results(
        d0,
        [d1, d3],
        participant_results,
        game_rows,
        cells,
        seeds,
        games,
    )
    d1_d0 = _enhance_comparison(control_comparisons[0])
    d3_d0 = _enhance_comparison(control_comparisons[1])
    d1_d3 = _enhance_comparison(
        shared._comparison_results(
            d3,
            [d1],
            participant_results,
            game_rows,
            cells,
            seeds,
            games,
        )[0]
    )
    expected_jobs = len(jobs)
    complete = len(progress["jobs"]) == expected_jobs
    safety_clean = (
        progress.get("status") != "safety_failed"
        and all(result["aggregate"]["safety_clean"] for result in participant_results.values())
    )
    comparable = complete and safety_clean and all(
        comparison["equal_completed_budget"] for comparison in (d1_d0, d3_d0, d1_d3)
    )
    decision = _decide(mode, d1_d0, d1_d3) if comparable else {
        "verdict": "NO_VERDICT",
        "d1_noninferior_to_d3": None,
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "completed"
            if complete and safety_clean
            else "safety_failed"
            if progress.get("status") == "safety_failed"
            else "partial"
        ),
        "mode": mode,
        "classification": "diagnostic_only",
        "config_fingerprint": config["config_fingerprint"],
        "schedule": {
            "participants": len(participants),
            "cells": len(cells),
            "seeds": seeds,
            "games_per_cell_seed": games,
            "jobs_expected": expected_jobs,
            "jobs_completed": len(progress["jobs"]),
            "games_expected_per_participant": len(cells) * len(seeds) * games,
            "identical_cells_seeds_games": True,
        },
        "safety_clean": safety_clean,
        "comparable": comparable,
        "participants": participant_results,
        "comparisons": {
            "D1_minus_D0": d1_d0,
            "D3_minus_D0": d3_d0,
            "D1_minus_D3": d1_d3,
        },
        "decision": decision,
        "elapsed_wall_seconds": progress.get("elapsed_wall_seconds", 0.0),
        "feature_contribution_audit": {
            "feature": "coherent revised Starmie deck D1 under fixed v2.4",
            "purpose": "measure deck-substrate marginal gameplay value without changing policy",
            "baseline": "D0 exact incoherent submitted deck under fixed v2.4",
            "benefit": "D1-D0 equal-budget local win-rate and cell-seed delta distributions",
            "cost": "elapsed wall/game seconds and 5,760 production games",
            "damage": "D1 severe regression cells and D1-D3 noninferiority",
            "uniqueness": "deck-only comparison; bundle brains and r7 are excluded",
            "precision": "independent aggregate and cell-seed batch 95% intervals",
            "recall": "20 predeclared R17 cells across three scratch seeds",
            "mean_true_objective_delta": "unavailable; local arena is not Kaggle truth",
            "throughput_impact": "reported per participant",
            "marginal_value_per_second": d1_d0.get("marginal_local_wins_per_second"),
            "failure_clustering": "reported by opponent cell and pilot",
            "decision": "diagnostic only",
            "next_action": "use the frozen verdict once; do not rerun to chase a preferred estimate",
        },
        "runtime_mutated": False,
        "submission_mutated": False,
        "default_v2_4_mutated": False,
    }
    _assert_no_forbidden_persisted_keys(results)

    return results


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Fixed-policy deck control",
        "",
        f"- mode: `{results['mode']}`",
        f"- status: **{results['status']}**",
        f"- verdict: **{results['decision']['verdict']}**",
        f"- safety clean: `{results['safety_clean']}`; comparable: `{results['comparable']}`",
        "",
        "| arm | attempted/planned | W-L-D | win rate | Wilson 95% |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm_id in ARM_IDS:
        aggregate = results["participants"][arm_id]["aggregate"]
        lines.append(
            f"| {arm_id} | {aggregate['attempted']}/{aggregate['planned_games']} | "
            f"{aggregate['wins']}-{aggregate['losses']}-{aggregate['draws']} | "
            f"{aggregate['win_rate']} | {aggregate['wilson95']} |"
        )
    lines.extend(("", "## Comparisons", ""))
    for name, comparison in results["comparisons"].items():
        lines.append(
            f"- `{name}`: delta `{comparison['aggregate_win_rate_delta']}`, "
            f"aggregate CI `{comparison['independent_aggregate_delta_ci95']}`, "
            f"batch CI `{comparison['independent_cell_seed_batch_delta_ci95']}`, "
            f"severe cells `{comparison['threshold_severe_regression_count']}`."
        )
    lines.extend((
        "",
        "Smoke scores are non-scientific. Production is a local diagnostic, not ladder truth.",
    ))

    return "\n".join(lines) + "\n"


def _verify_fixed_inputs() -> None:
    _require_hash(_sha256_file(POOL_PATH), EXPECTED_POOL_SHA256, "bucket pool")
    _require_hash(_sha256_file(EXPLOITER_PATH), EXPECTED_EXPLOITER_SHA256, "exploiter checkpoint")


def _require_smoke_gate() -> None:
    path = SMOKE_OUT_DIR / "results.json"
    if not path.is_file():
        raise ValueError("production requires the completed predeclared smoke")
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = len(ARM_IDS) * 20 * len(SMOKE_SEEDS) * SMOKE_GAMES
    if (
        result.get("mode") != "smoke"
        or result.get("status") != "completed"
        or result.get("safety_clean") is not True
        or result.get("schedule", {}).get("jobs_completed") != len(ARM_IDS) * 20
        or sum(value["aggregate"]["completed"] for value in result.get("participants", {}).values()) != expected
    ):
        raise DeckControlSafetyError("predeclared smoke receipt did not pass its exact gate")


def run(
    mode: str,
    *,
    resume: bool = False,
    matchup_runner: Callable[..., Any] = run_real_meta_matchup,
) -> dict[str, Any]:
    if mode not in {"smoke", "production"}:
        raise ValueError("mode must be smoke or production")
    if mode == "production":
        _require_smoke_gate()
    _verify_fixed_inputs()
    arms = load_deck_arms()
    protected = protected_surface_receipt()
    out_dir = SMOKE_OUT_DIR if mode == "smoke" else PRODUCTION_OUT_DIR
    seeds = list(SMOKE_SEEDS if mode == "smoke" else PRODUCTION_SEEDS)
    games = SMOKE_GAMES if mode == "smoke" else PRODUCTION_GAMES
    config = _config(mode, out_dir, arms, protected)
    progress = _prepare_progress(out_dir, config, resume=resume)
    if progress.get("config_fingerprint") != config["config_fingerprint"]:
        raise ValueError("progress config fingerprint mismatch")
    participants = [
        shared.PanelParticipant(arm_id, "spec:v2.4", "control" if arm_id == "D0" else "candidate")
        for arm_id in ARM_IDS
    ]
    cells = shared.load_panel_cells(
        POOL_PATH,
        stage_name=STAGE,
        top_n=TOP_N,
        carry_n=CARRY_N,
        pilots=list(PILOTS),
        exploiter_top=EXPLOITER_TOP,
    )
    if len(cells) != 20:
        raise DeckControlSafetyError(f"predeclared schedule expected 20 cells, found {len(cells)}")
    jobs = shared.build_jobs(participants, cells, seeds)
    shared.validate_durable_jobs(progress, jobs, games=games)
    started = time.perf_counter()
    base_elapsed = float(progress.get("elapsed_wall_seconds") or 0.0)
    agents = {arm_id: SpecAgent.from_str("v2.4") for arm_id in ARM_IDS}
    for job in jobs:
        if job.job_key in progress["jobs"]:
            continue
        summary = matchup_runner(
            agents[job.participant.label],
            _build_opponent(job),
            list(arms[job.participant.label].card_ids),
            list(job.cell.card_ids),
            games=games,
            bucket=job.cell.bucket,
            opp_deck_hash=job.cell.deck_hash,
            opp_archetype=job.cell.shell,
            seed=job.panel_seed,
            max_steps=MAX_STEPS,
            game_seeds=shared.derive_game_seeds(job.panel_seed, games),
            stop_on_safety_failure=True,
        )
        payload = shared._job_payload(job, summary)
        safety = {key: payload[key] for key in ("crashes", "timeouts", "illegal") if payload[key]}
        if payload["completed"] != payload["planned_games"]:
            safety["incomplete"] = payload["planned_games"] - payload["completed"]
        progress["jobs"][job.job_key] = payload
        progress["elapsed_wall_seconds"] = round(base_elapsed + time.perf_counter() - started, 6)
        progress["status"] = "safety_failed" if safety else "running"
        _assert_no_forbidden_persisted_keys(progress)
        write_json(out_dir / "progress.json", progress)
        print(
            f"[{job.participant.label}] {job.cell.cell_id} seed={job.panel_seed} "
            f"{payload['wins']}-{payload['losses']}-{payload['draws']} "
            f"safety={payload['illegal']}/{payload['crashes']}/{payload['timeouts']}",
            flush=True,
        )
        if safety:
            _assert_protected_unchanged(protected)
            results = _build_results(mode, config, progress, participants, cells, seeds, games, jobs)
            write_json(out_dir / "failure_results.json", results)
            raise DeckControlSafetyError(f"fixed-policy deck control hard-stopped: {safety}")
        shared.validate_durable_jobs(progress, jobs, games=games)
    progress["status"] = "completed"
    progress["elapsed_wall_seconds"] = round(base_elapsed + time.perf_counter() - started, 6)
    write_json(out_dir / "progress.json", progress)
    _assert_protected_unchanged(protected)
    results = _build_results(mode, config, progress, participants, cells, seeds, games, jobs)
    if results["status"] != "completed" or not results["comparable"]:
        raise DeckControlSafetyError("completed deck-control schedule is not comparable")
    write_json(out_dir / "results.json", results)
    write_markdown(out_dir / "report.md", _render_report(results))
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="fixed_policy_deck_control",
        command=[sys.executable, *sys.argv],
        inputs={
            "deck_registry": str(REGISTRY_PATH),
            "bucket_pool": str(POOL_PATH),
            "exploiter_checkpoint": str(EXPLOITER_PATH),
            "config_fingerprint": config["config_fingerprint"],
        },
        outputs={"results": "results.json", "report": "report.md", "progress": "progress.json"},
        summary={
            "mode": mode,
            "verdict": results["decision"]["verdict"],
            "safety_clean": results["safety_clean"],
            "comparable": results["comparable"],
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "production"))
    parser.add_argument("--resume", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run(args.mode, resume=args.resume)
    print(json.dumps({
        "status": results["status"],
        "mode": results["mode"],
        "verdict": results["decision"]["verdict"],
        "safety_clean": results["safety_clean"],
    }, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
