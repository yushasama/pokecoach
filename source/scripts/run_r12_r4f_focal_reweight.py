from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
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
import run_r9_r4f_retrain as r4f


SCHEMA_VERSION = "deepmew.r12.r4f_focal_reweight.v0"
DEFAULT_OUT = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight"
DEFAULT_INIT = ROOT / ".research" / "r10" / "lane2_r4f_equal_epochs" / "selected_ar_bc_r4f_candidate.pt"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _slice_flags(sample: Any) -> set[str]:
    flags = {"other"}
    if sample.target >= sample.legal_count:
        return {"stop"}
    features = sample.features[sample.target]
    option_second_order = features[-len(r4f.SECOND_ORDER_OPTION_SCHEMA):]
    attack_discard_scaling = option_second_order[0] >= 0.5
    attack_self_mill = option_second_order[1] >= 0.5
    zero_whiff = option_second_order[6] >= 0.5
    low_deck_self_mill = option_second_order[9] >= 0.5
    flags = set()
    if attack_discard_scaling and zero_whiff:
        flags.add("P1_riptide_whiff")
    if attack_self_mill and low_deck_self_mill:
        flags.add("P3_hammer_low_deck")
    if not flags:
        flags.add("other")

    return flags


def _slice_weight(sample: Any, args: argparse.Namespace) -> float:
    weight = r4f._r4f_row_weight(sample)
    flags = _slice_flags(sample)
    if "P1_riptide_whiff" in flags:
        weight = min(weight, float(args.p1_weight_cap))
    if "P3_hammer_low_deck" in flags:
        weight *= float(args.p3_weight_multiplier)

    return max(0.05, min(float(args.max_row_weight), weight))


def batch_steps_r12(samples: list[Any], args: argparse.Namespace) -> Any:
    packed = r4f.batch_steps_r4f(samples)
    adjusted = torch.tensor([_slice_weight(sample, args) for sample in samples], dtype=torch.float32)

    return r4f.R4FStepBatch(
        features=packed.features,
        option_meta=packed.option_meta,
        target=packed.target,
        mask=packed.mask,
        weights=adjusted,
        intent_target=packed.intent_target,
    )


def diagnose_loss_contribution(model: r4f.R4FIntentAutoregressiveScorer, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for encoded in r4f.iter_refgame_encoded_r9(Path(args.refgames_dir), split="train", limit=int(args.diagnose_rows)):
        rows.extend(r4f.r4.make_step_samples(encoded, weighted=False))
    by_slice: dict[str, Counter[str]] = defaultdict(Counter)
    loss_by_slice: Counter[str] = Counter()
    with torch.no_grad():
        for start in range(0, len(rows), int(args.batch_size)):
            chunk = rows[start:start + int(args.batch_size)]
            packed = r4f.batch_steps_r4f(chunk)
            scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
            losses = r4f.focal_cross_entropy(scores, packed.target, gamma=float(args.focal_gamma))
            for sample, loss in zip(chunk, losses.tolist()):
                for flag in _slice_flags(sample):
                    by_slice[flag]["rows"] += 1
                    loss_by_slice[flag] += float(loss)
    total_loss = sum(loss_by_slice.values())

    return {
        "rows": len(rows),
        "slices": {
            key: {
                "rows": by_slice[key]["rows"],
                "loss_sum": round(loss_by_slice[key], 6),
                "loss_share": round(loss_by_slice[key] / total_loss, 6) if total_loss else 0.0,
            }
            for key in sorted(by_slice)
        },
    }


def train_reweighted(model: r4f.R4FIntentAutoregressiveScorer, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rows_seen = 0
    step_rows_seen = 0
    batches = 0
    action_loss_total = 0.0
    intent_loss_total = 0.0
    slice_counts: Counter[str] = Counter()
    batch = []
    for encoded in r4f.iter_refgame_encoded_r9(Path(args.refgames_dir), split="train", limit=int(args.train_rows)):
        rows_seen += 1
        for step in r4f.r4.make_step_samples(encoded, weighted=False):
            batch.append(step)
            for flag in _slice_flags(step):
                slice_counts[flag] += 1
        if len(batch) < int(args.batch_size):
            continue
        packed = batch_steps_r12(batch, args)
        scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
        action_loss_by_row = r4f.focal_cross_entropy(scores, packed.target, gamma=float(args.focal_gamma))
        action_loss = (action_loss_by_row * packed.weights).sum() / packed.weights.sum()
        intent_loss = torch.nn.functional.cross_entropy(model.intent_logits(packed.features), packed.intent_target)
        loss = action_loss + float(args.intent_loss_weight) * intent_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_rows_seen += len(batch)
        batches += 1
        action_loss_total += float(action_loss.item())
        intent_loss_total += float(intent_loss.item())
        batch = []
    if batch:
        packed = batch_steps_r12(batch, args)
        scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
        action_loss_by_row = r4f.focal_cross_entropy(scores, packed.target, gamma=float(args.focal_gamma))
        action_loss = (action_loss_by_row * packed.weights).sum() / packed.weights.sum()
        intent_loss = torch.nn.functional.cross_entropy(model.intent_logits(packed.features), packed.intent_target)
        loss = action_loss + float(args.intent_loss_weight) * intent_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_rows_seen += len(batch)
        batches += 1
        action_loss_total += float(action_loss.item())
        intent_loss_total += float(intent_loss.item())

    return {
        "rows_seen": rows_seen,
        "step_rows_seen": step_rows_seen,
        "batches": batches,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mean_action_loss": round(action_loss_total / max(1, batches), 6),
        "mean_intent_loss": round(intent_loss_total / max(1, batches), 6),
        "slice_step_counts": dict(slice_counts),
        "p1_weight_cap": float(args.p1_weight_cap),
        "p3_weight_multiplier": float(args.p3_weight_multiplier),
    }


def render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 R4-F Focal Reweight Summary",
        "",
        f"- status: `{payload['status']}`",
        f"- model: `{payload['model_path']}`",
        f"- initialized_from: `{payload['initialization'].get('source_checkpoint')}`",
        f"- rows_seen: `{payload['train']['rows_seen']}`",
        f"- p1_weight_cap: `{payload['train']['p1_weight_cap']}`",
        f"- p3_weight_multiplier: `{payload['train']['p3_weight_multiplier']}`",
        "",
        "## Loss Contribution Before Training",
        "",
        "| slice | rows | loss_sum | loss_share |",
        "|---|---:|---:|---:|",
    ]
    for key, row in payload["pretrain_loss_contribution"]["slices"].items():
        lines.append(f"| `{key}` | {row['rows']} | {row['loss_sum']} | {row['loss_share']} |")
    lines.extend(["", "This is a candidate training checkpoint only; panel and powered canary decide selection.", ""])

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, prior_payload = r4f.load_r4f_model(Path(args.init_checkpoint))
    pretrain = diagnose_loss_contribution(model, args)
    train = train_reweighted(model, args)
    model.eval()
    val = r4f.collect_r9_validation(Path(args.refgames_dir), limit=int(args.val_rows))
    metrics = r4f.evaluate_r4f(model, val, batch_size=int(args.batch_size))
    model_path = out_dir / "selected_ar_bc_r4f_reweighted_candidate.pt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "status": "trained_reweighted_not_panel_adjudicated",
        "model_path": str(model_path),
        "input_dim": int(prior_payload["input_dim"]),
        "hidden_dim": int(prior_payload["hidden_dim"]),
        "depth": int(prior_payload["depth"]),
        "embed_dim": int(prior_payload["embed_dim"]),
        "dropout": float(prior_payload.get("dropout", 0.0)),
        "initialization": {
            "source_checkpoint": str(Path(args.init_checkpoint)),
            "source_schema_version": prior_payload.get("schema_version"),
            "prior_train": prior_payload.get("train"),
        },
        "pretrain_loss_contribution": pretrain,
        "train": train,
        "offline_common_split": {"R4-F_reweighted": metrics},
        "feature_audit_decision": "diagnostic only until powered canary and panel pass under equal budget",
    }
    args.input_dim = int(prior_payload["input_dim"])
    args.hidden_dim = int(prior_payload["hidden_dim"])
    args.depth = int(prior_payload["depth"])
    args.embed_dim = int(prior_payload["embed_dim"])
    args.dropout = float(prior_payload.get("dropout", 0.0))
    r4f.save_r4f_checkpoint(model_path, model, args, train, payload["initialization"])
    preservation = preserve_checkpoint(
        model_path,
        mission=str(args.backup_mission),
        name="r4f-reweighted",
        reason="R12 R4-F focal-reweighted candidate checkpoint",
    )
    payload["checkpoint_preservation"] = preservation
    _write_json(out_dir / "r4f_focal_reweight_summary.json", payload)
    _write_text(out_dir / "r4f_focal_reweight_summary.md", render_summary(payload))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str)[:8000])

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R12 Lane 1: R4-F focal reweight for P3 Hammer low-deck regression.")
    parser.add_argument("--refgames-dir", default=str(r4f.DEFAULT_REGEN3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--train-rows", type=int, default=200000)
    parser.add_argument("--diagnose-rows", type=int, default=50000)
    parser.add_argument("--val-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--intent-loss-weight", type=float, default=0.15)
    parser.add_argument("--p1-weight-cap", type=float, default=1.2)
    parser.add_argument("--p3-weight-multiplier", type=float, default=2.5)
    parser.add_argument("--max-row-weight", type=float, default=5.0)
    parser.add_argument("--backup-mission", default="r12")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
