"""Train (smoke) an RL ablation-ladder experiment.

Default backend is the dependency-free `SmokeTrainer`: it drives the real
env/encoder/mask/reward/opponent objects with an untrained masked policy for a
tiny number of decision steps and emits a reproducible manifest + metrics
summary. This starts NO long training and changes NO runtime/submission file.

Examples
--------
    python scripts/train_rl_baseline_v0.py --experiment RL0_pure_masked_ppo --dry-run
    python scripts/train_rl_baseline_v0.py --experiment RL0_pure_masked_ppo --smoke-steps 32
    python scripts/train_rl_baseline_v0.py --experiment RL0 --smoke-steps 64 --seed 7
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
from deepmew.rl.manifests import write_rl_run  # noqa: E402
from deepmew.rl.registry import (  # noqa: E402
    build_encoder,
    build_masker,
    build_opponent_sampler,
    build_reward,
    resolve_experiment,
)
from deepmew.rl.trainer import RLComponents, make_trainer  # noqa: E402


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL ablation-ladder smoke trainer")
    parser.add_argument("--experiment", default="RL0_pure_masked_ppo")
    parser.add_argument("--backend", default=None,
                        help="algorithm backend: smoke, masked_ppo, or puffer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=None,
                        help="total RL decision steps; preferred over --smoke-steps")
    parser.add_argument("--smoke-steps", type=int, default=16,
                        help="total RL decision steps to roll out in smoke mode")
    parser.add_argument("--max-steps-per-game", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true",
                        help="assemble components and emit a manifest without playing")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--no-write", action="store_true",
                        help="skip writing the manifest/summary to disk")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _build_args(argv)
    config = resolve_experiment(args.experiment).validate()
    backend = args.backend or config.trainer_backend
    from dataclasses import replace
    config = replace(config, trainer_backend=backend)
    if args.max_steps_per_game is not None:
        config = replace(config, max_steps_per_game=args.max_steps_per_game)

    components = RLComponents(
        config=config,
        encoder=build_encoder(config),
        masker=build_masker(config),
        reward_fn=build_reward(config),
        opponent_sampler=build_opponent_sampler(config, seed=args.seed),
    )
    trainer_kwargs = {}
    if backend == "masked_ppo":
        trainer_kwargs.update(
            {
                "device": args.device,
                "checkpoint_path": args.checkpoint,
                "resume": args.resume,
                "rollout_steps": args.rollout_steps,
                "max_steps_per_game": args.max_steps_per_game,
            }
        )
    trainer = make_trainer(config.trainer_backend, components, seed=args.seed, **trainer_kwargs)

    started = time.perf_counter()
    total_steps = args.train_steps if args.train_steps is not None else args.smoke_steps
    result = trainer.run(total_steps=max(0, total_steps), dry_run=args.dry_run)
    elapsed = time.perf_counter() - started

    result_dict = result.to_dict()
    metrics = rl_metrics.summarize(result_dict, elapsed_seconds=elapsed)

    command = ["scripts/train_rl_baseline_v0.py", *argv]
    print(f"[train_rl_baseline_v0] experiment={config.experiment_id} backend={config.trainer_backend} "
          f"dry_run={args.dry_run} seed={args.seed}")
    print(f"  obs_dim={result.observation_dim} feature_modules={list(config.feature_modules)} "
          f"reward={config.reward} opponents={components.opponent_sampler.labels}")
    print(f"  episodes={result.episodes} decision_steps={result.decision_steps} "
          f"wins={result.wins} losses={result.losses} draws={result.draws} "
          f"crashes={result.crashes} timeouts={result.timeouts} "
          f"illegal_attempts={result.illegal_attempts}")
    rt = metrics["runtime_metrics"]
    print(f"  win_rate={rt['win_rate']} mean_reward={rt['mean_reward']} "
          f"steps_per_sec={rt['steps_per_sec']} per_opponent={rt['per_opponent_win_rate']}")
    if result.training_metrics:
        print(f"  training_metrics={result.training_metrics}")
    if args.checkpoint:
        print(f"  checkpoint={args.checkpoint} resume={args.resume}")

    if not args.no_write:
        run_dir = write_rl_run(
            config=config,
            schema=components.encoder.schema,
            seed=args.seed,
            command=command,
            result=result_dict,
            metrics=metrics,
            out_root=args.out_root,
            suffix=args.suffix or f"train_{config.experiment_id.split('_')[0].lower()}",
        )
        print(f"  manifest={run_dir.manifest_path}")
        print(f"  summary={run_dir.output_path('summary.json')}")
    else:
        print("  (--no-write: manifest/summary not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
