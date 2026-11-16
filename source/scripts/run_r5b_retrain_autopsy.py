from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from checkpoint_preservation import preserve_checkpoint
import run_unified_r4_ablation as r4


SCHEMA_VERSION = "deepmew.r5b_retrain_autopsy.rebuilt.v0"
DEFAULT_REGEN3 = Path(r"D:\DeepMew_artifacts\datasets\unified_r2\refgames_dec061_regen3_clean")
DEFAULT_OUT = ROOT / ".research" / "r5b" / "retrain_autopsy"
DEFAULT_R4A = ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"
ORIGINAL_V1B_ROWS = 4_810_253


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_model(path: Path) -> tuple[r4.AutoregressiveOptionScorer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = r4.AutoregressiveOptionScorer(
        int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
        embed_dim=int(payload["embed_dim"]),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state"])

    return model, payload


def _new_model(args: argparse.Namespace, input_dim: int) -> r4.AutoregressiveOptionScorer:
    return r4.AutoregressiveOptionScorer(
        input_dim,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        embed_dim=int(args.embed_dim),
        dropout=float(args.dropout),
    )


def save_checkpoint(path: Path, model: r4.AutoregressiveOptionScorer, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_kind": "autoregressive_option_scorer",
            "model_state": model.state_dict(),
            "input_dim": int(payload["input_dim"]),
            "hidden_dim": int(payload["hidden_dim"]),
            "depth": int(payload["depth"]),
            "embed_dim": int(payload["embed_dim"]),
            "dropout": float(payload.get("dropout", 0.0)),
            "feature_mode": "base",
            "total_rows_seen": int(payload["train"]["rows_seen"]),
            "added_rows": int(payload["train"]["rows_seen"]),
            "train": payload["train"],
            "initialization": payload["initialization"],
        },
        path,
    )


def render_summary(payload: dict[str, Any]) -> str:
    metrics = payload["offline_common_split"]["v1b_rebuilt"]
    lines = [
        "# R5B Retrain Autopsy Rebuild",
        "",
        f"- status: `{payload['status']}`",
        f"- model: `{payload['model_path']}`",
        f"- rows_seen: `{payload['train']['rows_seen']}`",
        f"- initialized_from: `{payload['initialization'].get('source_checkpoint')}`",
        f"- top1: `{metrics.get('top1')}`",
        f"- top3: `{metrics.get('top3')}`",
        f"- seq_exact_proxy: `{metrics.get('sequence_exact_sampled')}`",
        "",
        "This source recipe is rebuilt after cleanup and is not claimed byte-identical to the deleted v1b `.pt`.",
        "",
    ]

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = r4.collect_encoded(Path(args.refgames_dir), split="val", limit=int(args.val_rows), feature_mode="base")
    if not val:
        raise RuntimeError("no validation rows found")
    input_dim = len(val[0].features[0])
    torch.manual_seed(int(args.seed))
    if args.init_checkpoint:
        model, init_payload = _load_model(Path(args.init_checkpoint))
        initialization = {
            "source_checkpoint": str(Path(args.init_checkpoint)),
            "source_schema_version": init_payload.get("schema_version"),
            "warm_start": True,
        }
    else:
        model = _new_model(args, input_dim)
        initialization = {"source_checkpoint": None, "warm_start": False}
    train_samples = r4.iter_encoded(Path(args.refgames_dir), split="train", limit=int(args.train_rows), feature_mode="base")
    train = r4.train_autoregressive(model, train_samples, args)
    model.eval()
    metrics = r4.evaluate_autoregressive(model, val, batch_size=int(args.batch_size), weighted=False)
    checkpoints_dir = out_dir / "checkpoints"
    model_path = checkpoints_dir / f"selected_ar_bc_v1b_rows_{train['rows_seen']}.pt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "status": "trained_rebuilt_source_not_original_v1b",
        "model_path": str(model_path),
        "input_dim": input_dim,
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "embed_dim": int(args.embed_dim),
        "dropout": float(args.dropout),
        "target_rows": int(args.train_rows),
        "original_v1b_rows": ORIGINAL_V1B_ROWS,
        "initialization": initialization,
        "train": train,
        "offline_common_split": {"v1b_rebuilt": metrics},
    }
    save_checkpoint(model_path, model, payload)
    preservation = preserve_checkpoint(
        model_path,
        mission=str(args.backup_mission),
        name="v1b-source",
        reason="v1b source checkpoint rebuilt from R5B source recipe",
    )
    payload["checkpoint_preservation"] = preservation
    _write_json(out_dir / "r5b_retrain_summary.json", payload)
    _write_text(out_dir / "r5b_retrain_summary.md", render_summary(payload))
    _write_json(
        out_dir / "training_budget_audit.json",
        {
            "schema_version": "deepmew.r5b.training_budget_audit.rebuilt.v0",
            "created_at_utc": payload["created_at_utc"],
            "regen3_corpus": {"train_rows": 6_287_375, "source": str(Path(args.refgames_dir))},
            "selected_checkpoint": str(model_path),
            "rows_seen": train["rows_seen"],
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str)[:6000])

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuilt R5B v1b source-checkpoint retrain/autopsy entrypoint.")
    parser.add_argument("--refgames-dir", default=str(DEFAULT_REGEN3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_R4A))
    parser.add_argument("--train-rows", type=int, default=ORIGINAL_V1B_ROWS)
    parser.add_argument("--val-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=550005)
    parser.add_argument("--backup-mission", default="r12")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
