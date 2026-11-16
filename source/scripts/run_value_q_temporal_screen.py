"""Run the compact MC-only snapshot/window/LSTM value comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.mc_screen import DEFAULT_BATCH_SIZE, DEFAULT_SEED  # noqa: E402
from deepmew.value_lab.temporal_screen import (  # noqa: E402
    DEFAULT_PARAMETER_BUDGET,
    DEFAULT_SEQUENCE_BLOCKS_PER_BATCH,
    DEFAULT_WINDOW,
    RUNG_TARGET_ROWS,
    run_temporal_screen,
)


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative and unique")

    return seeds


def _model_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("models must be comma-separated unique names")

    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rung", choices=sorted(RUNG_TARGET_ROWS), required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seeds", type=_seeds, default=(DEFAULT_SEED,))
    parser.add_argument("--target-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--validation-blocks-per-batch",
        type=int,
        default=DEFAULT_SEQUENCE_BLOCKS_PER_BATCH,
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--parameter-budget", type=int, default=DEFAULT_PARAMETER_BUDGET)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--v0-reference-checkpoint")
    parser.add_argument("--v0-reference-predictions")
    parser.add_argument("--models", type=_model_names, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_temporal_screen(
        cache_dir=Path(args.cache),
        split_path=Path(args.split),
        out_dir=Path(args.out),
        rung=args.rung,
        presentations=args.rows,
        seeds=args.seeds,
        target_batch_size=args.target_batch_size,
        validation_blocks_per_batch=args.validation_blocks_per_batch,
        threads=args.threads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bootstrap_replicates=args.bootstrap_replicates,
        parameter_budget=args.parameter_budget,
        window=args.window,
        v0_reference_checkpoint=(
            Path(args.v0_reference_checkpoint) if args.v0_reference_checkpoint else None
        ),
        v0_reference_predictions=(
            Path(args.v0_reference_predictions) if args.v0_reference_predictions else None
        ),
        model_names=args.models,
        command=[sys.executable, *sys.argv],
    )
    manifest = receipt["manifest"]
    print(json.dumps({
        "action": receipt["action"],
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "summary": manifest["summary"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
