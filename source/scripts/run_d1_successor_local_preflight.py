"""Validate the frozen D1-successor population without training or paid work."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.rl.algorithms.r4f_ppo import R4FNativePPO, R4FPPOConfig
from scripts import run_terminal_league_ppo as terminal


DEFAULT_ROOT = ROOT / ".research" / "d1_successor_mixed_league_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)

    return path.resolve(strict=True) if path.is_absolute() else (ROOT / path).resolve(strict=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def validate_population(population_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(population_path.read_text(encoding="utf-8"))
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("population manifest has no members")
    if any(str(member.get("id", "")).lower() == "x3" for member in members):
        raise ValueError("X3 must not appear in the active population")
    receipts = []
    for member in members:
        deck_path = resolve_path(str(member["deck_path"]))
        deck_hash = sha256_file(deck_path)
        if deck_hash != member["deck_sha256"]:
            raise ValueError(f"{member['id']}: deck hash drift")
        checkpoint_path = None
        checkpoint_hash = None
        if member.get("checkpoint_path"):
            checkpoint_path = resolve_path(str(member["checkpoint_path"]))
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != member["checkpoint_sha256"]:
                raise ValueError(f"{member['id']}: checkpoint hash drift")
        receipts.append({
            "member_id": member["id"],
            "deck_path": str(deck_path),
            "deck_sha256": deck_hash,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": checkpoint_hash,
            "hashes_match": True,
        })

    return payload, receipts


def validate_successor_initialization(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    actor_path = resolve_path(contract["parent_actor_path"])
    critic_path = resolve_path(contract["critic_checkpoint_path"])
    config_path = resolve_path(contract["critic_config_receipt_path"])
    for path, key in (
        (actor_path, "parent_actor_sha256"),
        (critic_path, "critic_checkpoint_sha256"),
        (config_path, "critic_config_receipt_sha256"),
    ):
        if sha256_file(path) != contract[key]:
            raise ValueError(f"successor initialization hash drift: {path}")
    source_payload = terminal._load_source_payload(actor_path)
    input_dim = int(source_payload["input_dim"])
    ppo_config = R4FPPOConfig(
        learning_rate=3e-5,
        weight_decay=0.0,
        gamma=1.0,
        gae_lambda=0.95,
        clip_coef=0.15,
        value_coef=0.5,
        entropy_coef=0.005,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=64,
        critic_hidden_dim=128,
        device="cpu",
    )
    learner = R4FNativePPO.from_r4f_checkpoint(
        actor_path,
        terminal._frozen_equivalence_panel(input_dim),
        ppo_config,
    )
    critic_receipt = terminal.initialize_frozen_critic(
        learner,
        source_payload,
        critic_path,
        config_path,
    )
    if not learner.proof.passed:
        raise ValueError("D1 actor equivalence proof failed")

    return {
        "schema_version": "deepmew.d1_successor_initialization_receipt.v1",
        "status": "PASS",
        "new_lineage_node": contract["new_lineage_node"],
        "actor_checkpoint": str(actor_path),
        "actor_checkpoint_sha256": sha256_file(actor_path),
        "actor_equivalence_passed": True,
        "critic": critic_receipt,
        "fresh_optimizer": True,
        "training_steps": 0,
    }


def roster_for_member(population: Mapping[str, Any], member: Mapping[str, Any]) -> dict[str, Any]:
    d1 = next(row for row in population["members"] if row["id"] == "d1")

    def roster_row(source: Mapping[str, Any], *, label: str, member_id: str, role: str) -> dict[str, Any]:
        spec = str(source["spec"])
        deck_binding = "" if spec.startswith("spec:") else f"deck:{resolve_path(str(source['deck_path']))}"

        return {
            "league_member_id": member_id,
            "label": label,
            "spec": spec,
            "deck_binding": deck_binding,
            "feature_policy_id": source["feature_policy_id"],
            "role": role,
        }

    opponent_label = "D1_MIRROR" if member["id"] == "d1" else str(member["label"])

    return {
        "schema_version": "deepmew.league_payoff_roster.v1",
        "members": [
            roster_row(d1, label="D1_ANCHOR", member_id="d1s-preflight-anchor", role="champion"),
            roster_row(member, label=opponent_label, member_id=f"d1s-preflight-{member['id']}", role="candidate"),
        ],
    }


def verify_game_output(output_dir: Path, labels: list[str]) -> dict[str, Any]:
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    games = [json.loads(line) for line in (output_dir / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(games) != 4 or not result.get("safety_clean"):
        raise ValueError(f"bounded readiness job incomplete or unsafe: {output_dir}")
    for game in games:
        if game.get("crashed") or game.get("illegal") or game.get("timed_out") or game.get("incomplete"):
            raise ValueError(f"unsafe game in {output_dir}")
    decision_counts = {label: 0 for label in labels}
    for game in games:
        telemetry = game.get("public_telemetry", {}).get("members", {})
        for label in labels:
            decision_counts[label] += int(telemetry.get(label, {}).get("decision_count", 0))
    if any(count <= 0 for count in decision_counts.values()):
        raise ValueError(f"actor-visible observation/action evidence missing: {output_dir}")

    return {
        "games": len(games),
        "decision_counts": decision_counts,
        "illegal_actions": 0,
        "crashes": 0,
        "timeouts": 0,
        "incomplete_games": 0,
        "raw_observations_persisted": False,
        "legal_option_payloads_persisted": False,
    }


def run_member_games(population: Mapping[str, Any], out_root: Path, *, seed: int) -> dict[str, Any]:
    receipts = []
    total_games = 0
    telemetry_qualified_games = 0
    preserved_pretelemetry_jobs = []
    preserved_failed_attempts = []
    for index, member in enumerate(population["members"]):
        job_id = f"member_{index:02d}_{member['id']}"
        roster_path = out_root / "rosters" / f"{job_id}.json"
        output_dir = out_root / "jobs" / job_id
        log_path = out_root / "logs" / f"{job_id}.log"
        if output_dir.exists():
            result_path = output_dir / "results.json"
            games_path = output_dir / "games.jsonl"
            if result_path.exists() and games_path.exists():
                existing_result = json.loads(result_path.read_text(encoding="utf-8"))
                existing_games = games_path.read_text(encoding="utf-8").splitlines()
                if not existing_result.get("safety_clean") or len(existing_games) != 4:
                    raise FileExistsError(f"existing readiness output is not safely reusable: {output_dir}")
                try:
                    scientific = verify_game_output(
                        output_dir,
                        ["D1_ANCHOR", "D1_MIRROR" if member["id"] == "d1" else str(member["label"])],
                    )
                except ValueError as error:
                    if "actor-visible observation/action evidence missing" not in str(error):
                        raise
                    preserved_pretelemetry_jobs.append({
                        "job_id": job_id,
                        "member_id": member["id"],
                        "games": len(existing_games),
                        "safety_clean": True,
                        "telemetry_qualified": False,
                        "disposition": "preserved_lifecycle_evidence_not_used_for_observation_action_gate",
                    })
                    total_games += len(existing_games)
                    job_id = f"{job_id}_telemetry"
                else:
                    total_games += scientific["games"]
                    telemetry_qualified_games += scientific["games"]
                    receipts.append({
                        "job_id": job_id,
                        "member_id": member["id"],
                        "pid": None,
                        "return_code": 0,
                        "fresh_process": True,
                        "durable_completed_job_reused": True,
                        "output_dir": str(output_dir),
                        "scientific": scientific,
                    })
                    continue
            else:
                preserved_failed_attempts.append({
                    "job_id": job_id,
                    "member_id": member["id"],
                    "games": 0,
                    "disposition": "preserved_pre_game_loader_failure_not_scientific_evidence",
                    "log": str(log_path),
                })
                job_id = f"{job_id}_loader_repair"
            roster_path = out_root / "rosters" / f"{job_id}.json"
            output_dir = out_root / "jobs" / job_id
            log_path = out_root / "logs" / f"{job_id}.log"
            if output_dir.exists():
                labels = ["D1_ANCHOR", "D1_MIRROR" if member["id"] == "d1" else str(member["label"])]
                scientific = verify_game_output(output_dir, labels)
                total_games += scientific["games"]
                telemetry_qualified_games += scientific["games"]
                receipts.append({
                    "job_id": job_id,
                    "member_id": member["id"],
                    "pid": None,
                    "return_code": 0,
                    "fresh_process": True,
                    "durable_completed_job_reused": True,
                    "output_dir": str(output_dir),
                    "scientific": scientific,
                })
                continue
        roster = roster_for_member(population, member)
        write_json(roster_path, roster)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-X", "faulthandler", str(ROOT / "scripts" / "run_league_payoff_matrix.py"),
            "--roster", str(roster_path), "--games", "2", "--seed", str(seed + index),
            "--min-decided", "0", "--max-wilson-width", "1.0", "--out-dir", str(output_dir),
            "--run-id", job_id, "--capture-public-telemetry", "--no-resume",
        ]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"readiness child failed: {job_id}, exit={return_code}")
        labels = [row["label"] for row in roster["members"]]
        scientific = verify_game_output(output_dir, labels)
        total_games += scientific["games"]
        telemetry_qualified_games += scientific["games"]
        receipts.append({
            "job_id": job_id,
            "member_id": member["id"],
            "pid": process.pid,
            "return_code": return_code,
            "fresh_process": True,
            "command": command,
            "log": str(log_path),
            "output_dir": str(output_dir),
            "scientific": scientific,
        })
    lifecycle = {
        "schema_version": "deepmew.d1_successor_cabt_lifecycle.v1",
        "status": "PASS",
        "strategy": "serialized_fresh_subprocess_per_bounded_member_job",
        "member_jobs": len(receipts),
        "games": total_games,
        "telemetry_qualified_games": telemetry_qualified_games,
        "preserved_pretelemetry_jobs": preserved_pretelemetry_jobs,
        "preserved_failed_attempts": preserved_failed_attempts,
        "child_failures": 0,
        "access_violations": 0,
        "participant_reuse_across_jobs": False,
        "unsafe_in_process_reuse": False,
        "private_engine_values_persisted": False,
        "receipts": receipts,
    }
    write_json(out_root / "CABT_ISOLATION_AND_LIFECYCLE_RECEIPT.json", lifecycle)

    return lifecycle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--run-games", action="store_true")
    parser.add_argument("--seed", type=int, default=211402741)
    args = parser.parse_args()
    mission_root = Path(args.mission_root).resolve()
    population_path = mission_root / "POPULATION_MANIFEST.json"
    population, member_hashes = validate_population(population_path)
    initialization = validate_successor_initialization(mission_root / "SUCCESSOR_INITIALIZATION_CONTRACT.json")
    write_json(mission_root / "SUCCESSOR_INITIALIZATION_RECEIPT.json", initialization)
    materialization = terminal.preflight_campaign_opponents(
        mission_root / "CAMPAIGN_CONFIG.json",
        seed=args.seed,
    )
    lifecycle = run_member_games(population, mission_root / "local_readiness", seed=args.seed) if args.run_games else None
    readiness = {
        "schema_version": "deepmew.d1_successor_member_readiness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if lifecycle is not None else "STRUCTURAL_PASS_GAMES_NOT_RUN",
        "member_count": len(population["members"]),
        "member_hash_receipts": member_hashes,
        "production_path_materialization": materialization,
        "successor_initialization_receipt": str(mission_root / "SUCCESSOR_INITIALIZATION_RECEIPT.json"),
        "lifecycle_receipt": str(mission_root / "local_readiness" / "CABT_ISOLATION_AND_LIFECYCLE_RECEIPT.json") if lifecycle else None,
        "bounded_games": lifecycle["games"] if lifecycle else 0,
        "training_steps": 0,
        "paid_work": False,
        "modal_launched": False,
        "x3_included": False,
    }
    write_json(mission_root / "EVERY_MEMBER_LOADER_READINESS_RECEIPT.json", readiness)
    print(json.dumps({"status": readiness["status"], "members": readiness["member_count"], "games": readiness["bounded_games"]}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
