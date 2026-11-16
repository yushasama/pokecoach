"""Continuous resumable RL training runner.

This runner is intentionally orchestration-only. The learning algorithm remains
behind `deepmew.rl.trainer` / `deepmew.rl.algorithms.*`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.manifest import utc_now_iso  # noqa: E402
from deepmew.rl import metrics as rl_metrics  # noqa: E402
from deepmew.rl.algorithms.masked_ppo import PPOPolicy, build_model_from_checkpoint  # noqa: E402
from deepmew.rl.eval import Evaluator  # noqa: E402
from deepmew.rl.manifests import write_rl_run  # noqa: E402
from deepmew.rl.metadata import append_eval_summary_csv, build_run_metadata, checkpoint_metadata  # noqa: E402
from deepmew.rl.registry import (  # noqa: E402
    build_encoder,
    build_masker,
    build_opponent_sampler,
    build_reward,
    resolve_experiment,
)
from deepmew.rl.trainer import RLComponents, make_trainer  # noqa: E402


STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[run_rl_training_v0] received signal {signum}; finishing current chunk and checkpointing")


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="RL0_pure_masked_ppo")
    parser.add_argument("--backend", default="masked_ppo")
    parser.add_argument("--advantage-backend", default="torch", choices=["torch", "puffer_cpu"])
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--max-env-steps", "--train-steps", dest="max_env_steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--env-vectorization", default="local", choices=["local", "subprocess"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="latest", choices=["none", "latest"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-interval-minutes", type=float, default=10.0)
    parser.add_argument("--eval-interval-minutes", type=float, default=20.0)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--output-root", default=".research/rl_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--hard-stop-on-safety", action="store_true")
    parser.add_argument("--hard-stop-on-reference-regression", action="store_true")
    parser.add_argument("--wandb-or-local-logging", default="local")
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=512)
    parser.add_argument("--max-steps-per-game", type=int, default=None)
    return parser.parse_args(argv)


def _canonical_backend(backend: str, advantage_backend: str) -> str:
    if backend in {"puffer", "puffer_advantage_cpu", "masked_ppo_puffer_advantage"}:
        return "masked_ppo_puffer_advantage"
    if backend == "masked_ppo" and advantage_backend == "puffer_cpu":
        return "masked_ppo_puffer_advantage"
    return backend


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _disk_guard(path: Path, min_free_gb: float = 2.0) -> None:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"disk free space guard tripped: {free_gb:.2f} GiB < {min_free_gb:.2f} GiB")


def _checkpoint_label(label: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in label)
    return safe[:80] or "opponent"


def _save_rolling_checkpoint(latest: Path, ckpt_dir: Path, total_steps: int, *, keep: int = 5) -> Path | None:
    if not latest.exists():
        return None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rolling = ckpt_dir / f"step_{total_steps:012d}.pt"
    shutil.copy2(latest, rolling)
    old = sorted(ckpt_dir.glob("step_*.pt"), key=lambda path: path.name)
    for stale in old[:-keep]:
        stale.unlink(missing_ok=True)
    return rolling


def _evaluate_checkpoint(config, checkpoint: Path, *, games: int, seed: int, device: str) -> dict[str, Any]:
    model, algo_config, payload = build_model_from_checkpoint(checkpoint, device=device)
    policy = PPOPolicy(model, algo_config, deterministic=True)
    evaluator = Evaluator(config, seed=seed)
    result = evaluator.evaluate(games_per_opponent=games, policy=policy)
    metadata = checkpoint_metadata(payload)
    if metadata is None:
        puffer_status = "not_attempted"
        if algo_config.advantage_backend == "puffer_cpu":
            puffer_status = "not_vectorized_puffer_cpu_advantage_only"
        metadata = build_run_metadata(
            config=config,
            schema_names=payload.get("schema_names") or evaluator.encoder.schema.names,
            max_options=algo_config.action_dim,
            trainer_backend="masked_ppo_torch",
            advantage_backend=algo_config.advantage_backend,
            vector_backend="unknown_from_checkpoint",
            puffer_vectorized_status=puffer_status,
        )
    completed = sum(
        bucket["wins"] + bucket["losses"] + bucket["draws"]
        for bucket in result.per_opponent.values()
    )
    wins = sum(bucket["wins"] for bucket in result.per_opponent.values())
    score = wins / completed if completed else 0.0
    return {"score": score, "result": result.to_dict(), "run_metadata": metadata}


def _eval_reference_score(eval_payload: dict[str, Any] | None) -> float | None:
    if not eval_payload:
        return None
    per_opponent = ((eval_payload.get("result") or {}).get("per_opponent") or {})
    wins = 0
    completed = 0
    for label, bucket in per_opponent.items():
        if not str(label).startswith("reference_"):
            continue
        wins += int(bucket.get("wins") or 0)
        completed += int(bucket.get("wins") or 0)
        completed += int(bucket.get("losses") or 0)
        completed += int(bucket.get("draws") or 0)
    return (wins / completed) if completed else None


def _eval_safety_counts(eval_payload: dict[str, Any] | None) -> dict[str, int]:
    if not eval_payload:
        return {"crashes": 0, "timeouts": 0}
    result = eval_payload.get("result") or {}
    return {
        "crashes": int(result.get("crashes") or 0),
        "timeouts": int(result.get("timeouts") or 0),
    }


def _vector_backend_label(env_vectorization: str) -> str:
    if env_vectorization == "subprocess":
        return "subprocess_vector_env"
    return "single_process_env"


def _metadata_for_run(args: argparse.Namespace, config, components: RLComponents, canonical_backend: str) -> dict[str, Any]:
    if canonical_backend == "puffer_vectorized":
        return build_run_metadata(
            config=config,
            schema_names=components.encoder.schema.names,
            max_options=components.masker.max_options,
            trainer_backend="puffer_vectorized",
            advantage_backend="none",
            vector_backend="blocked_no_deepmew_native_vec_binding",
            puffer_vectorized_status="blocked_no_deepmew_native_vec_binding",
        )
    advantage_backend = "puffer_cpu" if canonical_backend == "masked_ppo_puffer_advantage" else args.advantage_backend
    puffer_status = "not_attempted"
    if advantage_backend == "puffer_cpu":
        puffer_status = "not_vectorized_puffer_cpu_advantage_only"
    return build_run_metadata(
        config=config,
        schema_names=components.encoder.schema.names,
        max_options=components.masker.max_options,
        trainer_backend="masked_ppo_torch",
        advantage_backend=advantage_backend,
        vector_backend=_vector_backend_label(args.env_vectorization),
        puffer_vectorized_status=puffer_status,
    )


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)
    args = _build_args(argv)
    if args.precision != "fp32":
        raise SystemExit("only --precision fp32 is currently supported by the reference backend")

    canonical_backend = _canonical_backend(args.backend, args.advantage_backend)
    if args.backend == "puffer":
        print(
            "[run_rl_training_v0] WARNING: --backend puffer is a compatibility alias for "
            "--backend masked_ppo_puffer_advantage; use --backend puffer_vectorized for real "
            "Puffer/vectorized training."
        )
    base_config = resolve_experiment(args.experiment).validate()
    config = replace(base_config, trainer_backend=canonical_backend)
    if args.max_steps_per_game is not None:
        config = replace(config, max_steps_per_game=args.max_steps_per_game)

    vector_suffix = "_subprocess" if args.env_vectorization == "subprocess" else ""
    run_name = f"{config.experiment_id}_{canonical_backend}{vector_suffix}_seed{args.seed}"
    run_dir = Path(args.output_root) / run_name
    ckpt_dir = run_dir / "checkpoints"
    latest = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "latest.pt"
    best_overall = ckpt_dir / "best_overall.pt"
    heartbeat_path = run_dir / "heartbeat.jsonl"
    crash_path = run_dir / "crash_report.json"
    summary_path = run_dir / "run_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    components = RLComponents(
        config=config,
        encoder=build_encoder(config),
        masker=build_masker(config),
        reward_fn=build_reward(config),
        opponent_sampler=build_opponent_sampler(config, seed=args.seed),
    )
    run_metadata = _metadata_for_run(args, config, components, canonical_backend)
    config_path = run_dir / "config.json"
    eval_history_path = run_dir / "eval_history.csv"
    _write_json(
        config_path,
        {
            "schema_version": "deepmew.rl_continuous_config.v0",
            "created_at_utc": utc_now_iso(),
            "args": vars(args),
            "experiment_id": config.experiment_id,
            "requested_backend": args.backend,
            "canonical_backend": canonical_backend,
            "run_metadata": run_metadata,
        },
    )

    if canonical_backend not in {"masked_ppo", "masked_ppo_puffer_advantage", "puffer_vectorized"}:
        raise SystemExit(
            "continuous runner supports --backend masked_ppo, masked_ppo_puffer_advantage, "
            "puffer_advantage_cpu, puffer_vectorized, or compatibility alias puffer"
        )

    deadline = time.monotonic() + args.hours * 3600.0 if args.hours is not None else None
    target_steps = args.max_env_steps if args.max_env_steps is not None else float("inf")
    total_steps = 0
    best_score = -1.0
    best_by_opponent: dict[str, float] = {}
    best_by_opponent_checkpoints: dict[str, str] = {}
    best_eval_payload: dict[str, Any] | None = None
    latest_eval_payload: dict[str, Any] | None = None
    reference_floor_score: float | None = None
    hard_stop_reason: str | None = None
    last_eval = 0.0
    last_heartbeat = 0.0
    chunk_index = 0
    started_wall = time.perf_counter()
    resume = args.resume == "latest" and latest.exists()

    try:
        while total_steps < target_steps:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if STOP_REQUESTED:
                break
            _disk_guard(run_dir)
            remaining = int(target_steps - total_steps) if target_steps != float("inf") else args.chunk_steps
            chunk_steps = max(1, min(args.chunk_steps, remaining))
            trainer = make_trainer(
                canonical_backend,
                components,
                seed=args.seed,
                device=args.device,
                checkpoint_path=latest,
                resume=resume,
                rollout_steps=args.rollout_steps,
                max_steps_per_game=args.max_steps_per_game,
                num_envs=args.num_envs,
                advantage_backend=args.advantage_backend,
                env_vectorization=args.env_vectorization,
            )
            chunk_started = time.perf_counter()
            result = trainer.run(total_steps=chunk_steps, dry_run=False)
            elapsed = time.perf_counter() - chunk_started
            if result.decision_steps == 0:
                metrics = rl_metrics.summarize(result.to_dict(), elapsed_seconds=elapsed)
                payload = {
                    "schema_version": "deepmew.rl_continuous_run.v0",
                    "status": "blocked",
                    "args": vars(args),
                    "requested_backend": args.backend,
                    "experiment_id": config.experiment_id,
                    "backend": canonical_backend,
                    "run_metadata": run_metadata,
                    "total_steps": total_steps,
                    "result": result.to_dict(),
                    "metrics": metrics,
                    "latest_checkpoint": str(latest) if latest.exists() else None,
                    "heartbeat": str(heartbeat_path),
                    "config_path": str(config_path),
                    "eval_history": str(eval_history_path),
                    "elapsed_seconds": round(time.perf_counter() - started_wall, 3),
                    "finished_at_utc": utc_now_iso(),
                }
                _write_json(summary_path, payload)
                print(json.dumps(payload, indent=2, default=_json_default))
                return 2
            total_steps += result.decision_steps
            chunk_index += 1
            resume = True
            metrics = rl_metrics.summarize(result.to_dict(), elapsed_seconds=elapsed)
            rolling_checkpoint = _save_rolling_checkpoint(latest, ckpt_dir, total_steps)
            if args.hard_stop_on_safety:
                if result.illegal_attempts or result.crashes or result.timeouts:
                    hard_stop_reason = (
                        "training_safety_regression:"
                        f" illegal={result.illegal_attempts}"
                        f" crashes={result.crashes}"
                        f" timeouts={result.timeouts}"
                    )

            now = time.monotonic()
            eval_payload = None
            if latest.exists() and (last_eval == 0.0 or (now - last_eval) >= args.eval_interval_minutes * 60.0):
                eval_payload = _evaluate_checkpoint(config, latest, games=args.eval_games, seed=args.seed, device=args.device)
                latest_eval_payload = eval_payload
                last_eval = now
                append_eval_summary_csv(
                    eval_history_path,
                    checkpoint=latest,
                    eval_payload=eval_payload,
                    run_metadata=eval_payload["run_metadata"],
                    seed=args.seed,
                )
                if eval_payload["score"] > best_score:
                    best_score = eval_payload["score"]
                    best_eval_payload = eval_payload
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(latest, best_overall)
                for opponent, bucket in eval_payload["result"].get("per_opponent", {}).items():
                    completed = bucket["wins"] + bucket["losses"] + bucket["draws"]
                    opponent_score = bucket["wins"] / completed if completed else 0.0
                    if opponent_score > best_by_opponent.get(opponent, -1.0):
                        best_by_opponent[opponent] = opponent_score
                        best_path = ckpt_dir / f"best_vs_{_checkpoint_label(opponent)}.pt"
                        shutil.copy2(latest, best_path)
                        best_by_opponent_checkpoints[opponent] = str(best_path)
                if args.hard_stop_on_safety:
                    safety_counts = _eval_safety_counts(eval_payload)
                    if safety_counts["crashes"] or safety_counts["timeouts"]:
                        hard_stop_reason = (
                            "eval_safety_regression:"
                            f" crashes={safety_counts['crashes']}"
                            f" timeouts={safety_counts['timeouts']}"
                        )
                if args.hard_stop_on_reference_regression:
                    reference_score = _eval_reference_score(eval_payload)
                    if reference_score is not None:
                        if reference_floor_score is None:
                            reference_floor_score = reference_score
                        elif reference_score < reference_floor_score:
                            hard_stop_reason = (
                                "reference_floor_regressed:"
                                f" {reference_score:.6f} < {reference_floor_score:.6f}"
                            )

            heartbeat = {
                "schema_version": "deepmew.rl_heartbeat.v0",
                "time_utc": utc_now_iso(),
                "chunk_index": chunk_index,
                "total_steps": total_steps,
                "target_steps": None if target_steps == float("inf") else target_steps,
                "elapsed_seconds": round(time.perf_counter() - started_wall, 3),
                "checkpoint": str(latest),
                "rolling_checkpoint": str(rolling_checkpoint) if rolling_checkpoint else None,
                "best_overall_checkpoint": str(best_overall) if best_overall.exists() else None,
                "best_eval_score": best_score if best_score >= 0 else None,
                "best_by_opponent": best_by_opponent,
                "best_by_opponent_checkpoints": best_by_opponent_checkpoints,
                "result": result.to_dict(),
                "metrics": metrics,
                "run_metadata": run_metadata,
                "config_path": str(config_path),
                "eval_history": str(eval_history_path),
                "eval": eval_payload,
                "reference_floor_score": reference_floor_score,
                "hard_stop_reason": hard_stop_reason,
            }
            _append_jsonl(heartbeat_path, heartbeat)
            if last_heartbeat == 0.0 or (now - last_heartbeat) >= args.heartbeat_seconds:
                print(
                    f"[run_rl_training_v0] chunk={chunk_index} total_steps={total_steps} "
                    f"latest={latest} best={best_score if best_score >= 0 else 'n/a'}"
                )
                last_heartbeat = now
            if hard_stop_reason:
                break

        status = "hard_stopped" if hard_stop_reason else "stopped" if STOP_REQUESTED else "completed"
        payload = {
            "schema_version": "deepmew.rl_continuous_run.v0",
            "status": status,
            "args": vars(args),
            "requested_backend": args.backend,
            "experiment_id": config.experiment_id,
            "backend": canonical_backend,
            "run_metadata": run_metadata,
            "total_steps": total_steps,
            "latest_checkpoint": str(latest) if latest.exists() else None,
            "rolling_checkpoints_kept": len(list(ckpt_dir.glob("step_*.pt"))) if ckpt_dir.exists() else 0,
            "best_overall_checkpoint": str(best_overall) if best_overall.exists() else None,
            "best_eval_score": best_score if best_score >= 0 else None,
            "latest_eval": latest_eval_payload,
            "best_eval": best_eval_payload,
            "best_by_opponent": best_by_opponent,
            "best_by_opponent_checkpoints": best_by_opponent_checkpoints,
            "reference_floor_score": reference_floor_score,
            "hard_stop_reason": hard_stop_reason,
            "heartbeat": str(heartbeat_path),
            "config_path": str(config_path),
            "eval_history": str(eval_history_path),
            "elapsed_seconds": round(time.perf_counter() - started_wall, 3),
            "finished_at_utc": utc_now_iso(),
        }
        _write_json(summary_path, payload)
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    except BaseException as exc:
        payload = {
            "schema_version": "deepmew.rl_crash_report.v0",
            "status": "crashed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "latest_checkpoint": str(latest) if latest.exists() else None,
            "best_overall_checkpoint": str(best_overall) if best_overall.exists() else None,
            "run_metadata": run_metadata,
            "config_path": str(config_path),
            "eval_history": str(eval_history_path),
            "time_utc": utc_now_iso(),
        }
        _write_json(crash_path, payload)
        print(json.dumps(payload, indent=2, default=_json_default))
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
