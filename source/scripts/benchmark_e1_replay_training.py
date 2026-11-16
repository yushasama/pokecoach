"""Benchmark the production streaming replay optimizer on admitted derived rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import torch

import run_terminal_league_ppo as terminal
from deepmew.rl.enhanced_replay_derived import DerivedReplayIndex
from deepmew.rl.enhanced_trainer import EnhancedLeagueTrainer
from deepmew.rl.algorithms.r4f_ppo import R4FPPOConfig


DEFAULT_R7 = Path(r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20260714)
    derived = DerivedReplayIndex(args.derived_manifest)
    rows = list(derived.global_rows())
    if len(rows) != len(set(rows)):
        raise SystemExit("benchmark rows are duplicated")
    examples = list(derived.load(rows))
    source = torch.load(DEFAULT_R7, map_location="cpu", weights_only=False)
    input_dim = int(source["input_dim"])
    config = R4FPPOConfig(
        learning_rate=1.0e-5,
        weight_decay=0.0,
        gamma=1.0,
        gae_lambda=0.95,
        clip_coef=0.15,
        value_coef=0.5,
        entropy_coef=0.005,
        max_grad_norm=0.5,
        update_epochs=1,
        minibatch_size=64,
        critic_hidden_dim=128,
        device=args.device,
    )
    trainer = EnhancedLeagueTrainer.from_r7_checkpoint(
        DEFAULT_R7,
        terminal._frozen_equivalence_panel(input_dim),
        q_state_dim=2,
        q_action_dim=2,
        ppo_config=config,
        lineage_id="e1-replay-training-benchmark",
    )
    # Preserve the production stage order using a bounded real V0 distillation batch.
    trainer.distill_v0(examples[:min(args.batch_size, len(examples))])
    trainer.begin_replay_warm_start()
    latencies = []
    receipts = []
    started = time.perf_counter()
    for offset in range(0, len(examples), args.batch_size):
        batch_started = time.perf_counter()
        receipts.append(trainer.replay_warm_start_batch(examples[offset:offset + args.batch_size]))
        latencies.append(time.perf_counter() - batch_started)
    seconds = time.perf_counter() - started
    receipt = trainer.finish_replay_warm_start(expected_unique_rows=len(examples))
    output = {
        "schema_version": "deepmew.e1_replay_training_benchmark.v1",
        "device": args.device,
        "rows": len(examples),
        "batch_size": args.batch_size,
        "optimizer_steps": len(latencies),
        "seconds": seconds,
        "rows_per_second": len(examples) / seconds,
        "p50_batch_seconds": statistics.median(latencies),
        "p95_batch_seconds": sorted(latencies)[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
        "forced_rows": receipt.consumed["forced_rows"],
        "nonforced_rows": receipt.consumed["nonforced_rows"],
        "actor_weight_sum": receipt.consumed["actor_weight_sum"],
        "first_row_id": receipt.consumed["first_row_id"],
        "final_row_id": receipt.consumed["final_row_id"],
        "batch_receipts": receipts,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
