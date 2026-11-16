"""Equal-budget AWR ladder-weighting ablation runner.

This script answers one narrow question: given the same init checkpoint, same
refgame rows, and same training budget, does a ladder-aware weighting config beat
the old AWR weighting under the same adopt-eval?

It is training/eval only. It does not touch runtime submission files.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
POOLS_DEFAULTS = sorted((ROOT / ".research/runs/bucket_pool").rglob("bucket_pools.json"))
POOLS_DEFAULT = POOLS_DEFAULTS[-1] if POOLS_DEFAULTS else ROOT / "bucket_pools.json"
TOP_STARYU_DECK = ROOT / ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"
CHAMPION_DEFAULT = ROOT / ".research/r12/lane1_r4f_focal_reweight/selected_ar_bc_r4f_reweighted_candidate.pt"
DEFAULT_CONFIGS = [
    ROOT / "configs/awr_ladder/A0_current.yaml",
    ROOT / "configs/awr_ladder/A4_exploiter.yaml",
]
WR_RE = re.compile(r'"overall_winrate":\s*([0-9.]+)')


def _run(cmd: list[str]) -> str:
    print(f"  $ {' '.join(str(part) for part in cmd[:8])} ...", flush=True)
    proc = subprocess.run([str(part) for part in cmd], cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], flush=True)
        raise RuntimeError(f"subrun failed ({proc.returncode}): {cmd[1] if len(cmd) > 1 else cmd[0]}")

    return proc.stdout + proc.stderr


def _label_for_config(path_text: str) -> str:
    path = Path(path_text)
    if path.name:
        return path.stem

    return "no_config"


def _train_one(args: argparse.Namespace, config_path: str, out_dir: Path) -> dict:
    cmd = [
        PY,
        "scripts/run_r18_awr_trainer.py",
        "--refgames-dir",
        args.refgames_dir,
        "--init-checkpoint",
        args.init_checkpoint,
        "--out-dir",
        str(out_dir),
        "--temperature",
        str(args.temperature),
        "--expected-loss-penalty",
        str(args.expected_loss_penalty),
        "--kl-beta",
        str(args.kl_beta),
        "--max-seconds",
        str(args.max_seconds),
        "--seed",
        str(args.seed),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if config_path:
        cmd += ["--ladder-weighting-config", config_path]
    _run(cmd)
    manifest_path = out_dir / "awr_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return manifest


def _eval_one(args: argparse.Namespace, ckpt: Path) -> float | None:
    if args.skip_eval:
        return None
    output = _run([
        PY,
        "scripts/run_r17_bucket_arena.py",
        "--pools",
        args.pools,
        "--candidate",
        f"checkpoint:{ckpt}",
        "--candidate-deck-binding",
        args.eval_deck_binding,
        "--stage",
        args.eval_stage,
        "--pilots",
        *args.eval_pilots.split(","),
        "--games",
        str(args.eval_games),
        "--top-n",
        str(args.top_n),
        "--exploiter-top",
        str(args.exploiter_top),
        "--champion",
        args.champion,
        "--no-write",
    ])
    matches = WR_RE.findall(output)
    if not matches:
        raise RuntimeError("could not parse overall_winrate from bucket arena output")

    return float(matches[-1])


def _write_report(out_dir: Path, payload: dict) -> None:
    rows = payload["rows"]
    lines = [
        "# AWR Ladder Weighting Ablation",
        "",
        "Training/eval-only. No runtime submission files changed.",
        "",
        "## Result",
        "",
        "| label | config | train_seconds | train_batches | eval_wr | mean_weight | ess | warnings |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        warnings = ", ".join(row["warnings"]) if row["warnings"] else ""
        eval_wr = "" if row["eval_wr"] is None else f"{row['eval_wr']:.4f}"
        lines.append(
            f"| {row['label']} | `{row['config']}` | {row['train_elapsed_seconds']} | "
            f"{row['train_batches']} | {eval_wr} | {row['mean_weight']} | "
            f"{row['effective_sample_size']} | {warnings} |"
        )
    lines.extend([
        "",
        "## Inputs",
        "",
        f"- refgames_dir: `{payload['inputs']['refgames_dir']}`",
        f"- init_checkpoint: `{payload['inputs']['init_checkpoint']}`",
        f"- max_seconds_each: `{payload['inputs']['max_seconds_each']}`",
        f"- eval_games: `{payload['inputs']['eval_games']}`",
        f"- eval_pilots: `{payload['inputs']['eval_pilots']}`",
        "",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _summarize_row(label: str, config_path: str, manifest: dict, eval_wr: float | None, ckpt: Path) -> dict:
    ladder = ((manifest.get("data") or {}).get("ladder_weighting") or {})
    weights = ladder.get("weights") or {}
    train = manifest.get("train") or {}

    return {
        "label": label,
        "config": config_path,
        "checkpoint": str(ckpt),
        "eval_wr": eval_wr,
        "train_elapsed_seconds": train.get("elapsed_seconds"),
        "train_batches": train.get("batches"),
        "mean_weight": weights.get("w_total", {}).get("mean"),
        "effective_sample_size": weights.get("effective_sample_size"),
        "warnings": ladder.get("warnings") or [],
        "manifest": str(ckpt.parent / "awr_manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refgames-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--configs", nargs="+", default=[str(path) for path in DEFAULT_CONFIGS])
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--expected-loss-penalty", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.03)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=181818)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--pools", default=str(POOLS_DEFAULT))
    parser.add_argument("--eval-deck-binding", default=f"deck:{TOP_STARYU_DECK}")
    parser.add_argument("--eval-stage", default="600-800")
    parser.add_argument("--eval-pilots", default="greedy,v2.4")
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--exploiter-top", type=int, default=2)
    parser.add_argument("--champion", default=str(CHAMPION_DEFAULT))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for config_path in args.configs:
        label = _label_for_config(config_path)
        candidate_dir = out_dir / label
        candidate_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ablation] training {label}", flush=True)
        manifest = _train_one(args, config_path, candidate_dir)
        ckpt = candidate_dir / "selected_ar_bc_r4f_candidate.pt"
        eval_wr = _eval_one(args, ckpt)
        rows.append(_summarize_row(label, config_path, manifest, eval_wr, ckpt))

    payload = {
        "schema_version": "deepmew.awr_ladder_ablation.v0",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "refgames_dir": args.refgames_dir,
            "init_checkpoint": args.init_checkpoint,
            "max_seconds_each": args.max_seconds,
            "eval_games": args.eval_games,
            "eval_pilots": args.eval_pilots,
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out_dir, payload)
    print(json.dumps({"out_dir": str(out_dir), "rows": rows}, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
