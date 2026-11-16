"""Evaluate an RL experiment's policy against a fixed opponent suite (smoke).

For the v0 scaffold the policy under test is the untrained `MaskedRandomPolicy`
(no trained checkpoint exists yet); a real backend loads a checkpoint and passes
its policy through the same `Evaluator` interface. Reports per-opponent win
rates and emits a manifest + summary. No training happens here.

Examples
--------
    python scripts/eval_rl_baseline_v0.py --experiment RL0_pure_masked_ppo --smoke-steps 32
    python scripts/eval_rl_baseline_v0.py --experiment RL0 --games-per-opponent 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.rl import metrics as rl_metrics  # noqa: E402
from deepmew.rl.eval import Evaluator  # noqa: E402
from deepmew.rl.manifests import write_rl_run  # noqa: E402
from deepmew.rl.metadata import append_eval_summary_csv, build_run_metadata, checkpoint_metadata  # noqa: E402
from deepmew.rl.registry import resolve_experiment  # noqa: E402


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL ablation-ladder smoke evaluator")
    parser.add_argument("--experiment", default="RL0_pure_masked_ppo")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games-per-opponent", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None,
                        help="coarse smoke budget; games_per_opponent = max(1, smoke_steps // 16)")
    parser.add_argument("--max-steps-per-game", type=int, default=None)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def _games_per_opponent(args: argparse.Namespace) -> int:
    if args.games_per_opponent is not None:
        return max(1, args.games_per_opponent)
    if args.smoke_steps is not None:
        return max(1, args.smoke_steps // 16)
    return 2


def main(argv: list[str]) -> int:
    args = _build_args(argv)
    config = resolve_experiment(args.experiment).validate()
    if args.backend is not None:
        from dataclasses import replace
        config = replace(config, trainer_backend=args.backend)
    games = _games_per_opponent(args)

    evaluator = Evaluator(config, seed=args.seed, max_steps_per_game=args.max_steps_per_game)
    policy = None
    policy_name = None
    checkpoint_payload = None
    checkpoint_algo_config = None
    if args.checkpoint:
        from deepmew.rl.algorithms.masked_ppo import PPOPolicy, build_model_from_checkpoint

        model, algo_config, payload = build_model_from_checkpoint(args.checkpoint, device=args.device)
        checkpoint_payload = payload
        checkpoint_algo_config = algo_config
        if payload.get("experiment_config", {}).get("experiment_id") not in (None, config.experiment_id):
            print(
                "  warning: checkpoint experiment "
                f"{payload.get('experiment_config', {}).get('experiment_id')} != {config.experiment_id}"
            )
        policy = PPOPolicy(model, algo_config, deterministic=True)
        policy_name = "PPOPolicy"
    run_metadata = checkpoint_metadata(checkpoint_payload) if checkpoint_payload is not None else None
    if run_metadata is None:
        advantage_backend = (
            checkpoint_algo_config.advantage_backend
            if checkpoint_algo_config is not None
            else "unknown"
        )
        puffer_status = "not_attempted"
        if advantage_backend == "puffer_cpu":
            puffer_status = "not_vectorized_puffer_cpu_advantage_only"
        run_metadata = build_run_metadata(
            config=config,
            schema_names=(
                checkpoint_payload.get("schema_names")
                if checkpoint_payload is not None and checkpoint_payload.get("schema_names")
                else evaluator.encoder.schema.names
            ),
            max_options=(
                checkpoint_algo_config.action_dim
                if checkpoint_algo_config is not None
                else evaluator.masker.max_options
            ),
            trainer_backend="masked_ppo_torch" if checkpoint_algo_config is not None else "unknown",
            advantage_backend=advantage_backend,
            vector_backend="unknown_from_eval",
            puffer_vectorized_status=puffer_status,
        )
    started = time.perf_counter()
    eval_result = evaluator.evaluate(games_per_opponent=games, policy=policy)
    elapsed = time.perf_counter() - started

    eval_dict = eval_result.to_dict()
    eval_dict["run_metadata"] = run_metadata
    # Reuse the metrics summarizer by mapping eval buckets into the run schema.
    aggregate = {
        "experiment_id": config.experiment_id,
        "episodes": sum(b["games"] for b in eval_result.per_opponent.values()),
        "wins": sum(b["wins"] for b in eval_result.per_opponent.values()),
        "losses": sum(b["losses"] for b in eval_result.per_opponent.values()),
        "draws": sum(b["draws"] for b in eval_result.per_opponent.values()),
        "crashes": eval_result.crashes,
        "timeouts": eval_result.timeouts,
        "decision_steps": 0,
        "illegal_attempts": 0,
        "mean_reward": 0.0,
        "per_opponent": {
            label: {"wins": b["wins"], "losses": b["losses"], "draws": b["draws"]}
            for label, b in eval_result.per_opponent.items()
        },
    }
    metrics = rl_metrics.summarize(aggregate, elapsed_seconds=elapsed)

    print(f"[eval_rl_baseline_v0] experiment={config.experiment_id} policy={policy_name or eval_result.policy_name} "
          f"games_per_opponent={games} seed={args.seed}")
    for label, bucket in eval_result.per_opponent.items():
        print(f"  vs {label}: win_rate={bucket['win_rate']} "
              f"(W{bucket['wins']}/L{bucket['losses']}/D{bucket['draws']} "
              f"crashes={bucket['crashes']} timeouts={bucket['timeouts']})")
    print(f"  crashes={eval_result.crashes} timeouts={eval_result.timeouts} "
          f"elapsed={eval_result.elapsed_seconds:.3f}s")

    if not args.no_write:
        run_dir = write_rl_run(
            config=config,
            schema=evaluator.encoder.schema,
            seed=args.seed,
            command=["scripts/eval_rl_baseline_v0.py", *argv],
            result={"eval": eval_dict, "dry_run": False, "episodes": aggregate["episodes"],
                    "decision_steps": 0, "crashes": eval_result.crashes,
                    "timeouts": eval_result.timeouts, "illegal_attempts": 0},
            metrics=metrics,
            out_root=args.out_root,
            suffix=args.suffix or f"eval_{config.experiment_id.split('_')[0].lower()}",
            run_metadata=run_metadata,
        )
        eval_csv = run_dir.output_path("eval_summary.csv")
        append_eval_summary_csv(
            eval_csv,
            checkpoint=args.checkpoint,
            eval_payload=eval_dict,
            run_metadata=run_metadata,
            seed=args.seed,
        )
        print(f"  manifest={run_dir.manifest_path}")
        print(f"  summary={run_dir.output_path('summary.json')}")
        print(f"  eval_csv={eval_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
