from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from checkpoint_preservation import preserve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r9_r4f_retrain as r4f


SCHEMA_VERSION = "deepmew.r10.r4f_equal_epochs.v0"
DEFAULT_OUT = ROOT / ".research" / "r10" / "lane2_r4f_equal_epochs"
DEFAULT_R9 = ROOT / ".research" / "r9" / "lane2_r4f_equal_epoch" / "selected_ar_bc_r4f_candidate.pt"
DEFAULT_V1B = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"
DEFAULT_R4A = ROOT / ".research" / "unified_r4" / "R4-A" / "selected_model.pt"
DEFAULT_REGEN3_COUNT = ROOT / ".research" / "r5b" / "retrain_autopsy" / "training_budget_audit.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _train_rows_from_budget(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    regen3 = payload.get("regen3_corpus") or {}
    value = regen3.get("train_rows")

    return int(value) if value is not None else None


def _source_counts(train: dict[str, Any]) -> Counter[str]:
    return Counter({str(key): int(value) for key, value in (train.get("source_counts") or {}).items()})


def _iter_refgame_with_skip(refgames_dir: Path, *, skip_rows: int) -> Iterable[Any]:
    skipped = 0
    while True:
        for row in r4f.r4._iter_rows(refgames_dir, split="train", max_game_index=None):
            if skipped < skip_rows:
                skipped += 1
                continue
            sample = r4f.encode_refgame_row_r9(row)
            if sample is None:
                continue
            yield sample
        skipped = 0


def _train_batch(
    model: r4f.R4FIntentAutoregressiveScorer,
    optimizer: torch.optim.Optimizer,
    batch: list[Any],
    args: argparse.Namespace,
) -> tuple[float, float, int]:
    packed = r4f.batch_steps_r4f(batch)
    scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
    action_loss_by_row = r4f.focal_cross_entropy(scores, packed.target, gamma=float(args.focal_gamma))
    action_loss = (action_loss_by_row * packed.weights).sum() / packed.weights.sum()
    intent_loss = torch.nn.functional.cross_entropy(model.intent_logits(packed.features), packed.intent_target)
    loss = action_loss + float(args.intent_loss_weight) * intent_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(action_loss.item()), float(intent_loss.item()), len(batch)


def train_to_regen3_parity(
    model: r4f.R4FIntentAutoregressiveScorer,
    prior_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    prior_regen3_rows: int,
    target_regen3_rows: int,
    corpus_train_rows: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    model.train()
    interval = max(1, int(args.top_player_interval))
    top_player = list(r4f.iter_top_player_encoded(Path(args.top_player_examples), limit=int(args.top_player_rows))) if int(args.top_player_rows) > 0 else []
    prior_sources = _source_counts(prior_payload.get("train") or {})
    prior_top_rows = int(prior_sources.get("top_player", 0))
    top_index = prior_top_rows % max(1, len(top_player)) if top_player else 0
    skip_mod = corpus_train_rows if corpus_train_rows else max(1, target_regen3_rows)
    skip_rows = int(args.skip_rows) if args.skip_rows is not None else prior_regen3_rows % skip_mod
    iterator = _iter_refgame_with_skip(Path(args.refgames_dir), skip_rows=skip_rows)
    additional_regen3_needed = max(0, target_regen3_rows - prior_regen3_rows)
    batch: list[Any] = []
    rows_added = 0
    regen3_added = 0
    step_rows_seen = 0
    batches = 0
    action_loss_total = 0.0
    intent_loss_total = 0.0
    source_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    progress_rows: list[dict[str, Any]] = []

    def consume(encoded: Any) -> None:
        nonlocal rows_added, step_rows_seen, batches, action_loss_total, intent_loss_total, batch
        rows_added += 1
        source = str(encoded.metadata.get("source") or "")
        source_counts[source] += 1
        intent_counts[str(encoded.metadata.get("goal_of_play") or r4f.UNKNOWN)] += 1
        batch.extend(r4f.r4.make_step_samples(encoded, weighted=False))
        if len(batch) < int(args.batch_size):
            return
        action_loss, intent_loss, step_count = _train_batch(model, optimizer, batch, args)
        step_rows_seen += step_count
        batches += 1
        action_loss_total += action_loss
        intent_loss_total += intent_loss
        batch = []
        if int(args.progress_every) and batches % int(args.progress_every) == 0:
            row = {
                "batches": batches,
                "rows_added": rows_added,
                "regen3_added": regen3_added,
                "cumulative_regen3_rows": prior_regen3_rows + regen3_added,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "last_action_loss": round(action_loss, 6),
                "last_intent_loss": round(intent_loss, 6),
            }
            progress_rows.append(row)
            _write_json(Path(args.out_dir) / "training_progress.json", {"schema_version": SCHEMA_VERSION, "progress": progress_rows})
            print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)

    while regen3_added < additional_regen3_needed:
        encoded = next(iterator)
        regen3_added += 1
        consume(encoded)
        if top_player and (prior_regen3_rows + regen3_added) % interval == 0:
            consume(top_player[top_index % len(top_player)])
            top_index += 1

    if batch:
        action_loss, intent_loss, step_count = _train_batch(model, optimizer, batch, args)
        step_rows_seen += step_count
        batches += 1
        action_loss_total += action_loss
        intent_loss_total += intent_loss

    cumulative_sources = prior_sources + source_counts
    elapsed = time.perf_counter() - started

    return {
        "prior_rows_seen": int((prior_payload.get("train") or {}).get("rows_seen") or 0),
        "prior_regen3_rows": prior_regen3_rows,
        "target_regen3_rows": target_regen3_rows,
        "added_rows_seen": rows_added,
        "added_regen3_rows": regen3_added,
        "added_step_rows_seen": step_rows_seen,
        "cumulative_rows_seen": int((prior_payload.get("train") or {}).get("rows_seen") or 0) + rows_added,
        "cumulative_regen3_rows": int(cumulative_sources.get("regen3", 0)),
        "batches": batches,
        "elapsed_seconds": round(elapsed, 3),
        "mean_action_loss": round(action_loss_total / max(1, batches), 6),
        "mean_intent_loss": round(intent_loss_total / max(1, batches), 6),
        "skip_rows": skip_rows,
        "source_counts_added": dict(source_counts),
        "source_counts_cumulative": dict(cumulative_sources),
        "top_intents_added": [{"intent": key, "rows": value} for key, value in intent_counts.most_common(12)],
    }


def save_checkpoint(path: Path, model: r4f.R4FIntentAutoregressiveScorer, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_kind": "r4f_intent_autoregressive",
            "model_state": model.state_dict(),
            "input_dim": int(payload["input_dim"]),
            "hidden_dim": int(payload["hidden_dim"]),
            "depth": int(payload["depth"]),
            "embed_dim": int(payload["embed_dim"]),
            "dropout": float(payload.get("dropout", 0.0)),
            "feature_mode": "second_order",
            "intent_vocab": r4f.INTENT_VOCAB,
            "state_schema": r4f.SECOND_ORDER_STATE_SCHEMA,
            "option_schema": r4f.SECOND_ORDER_OPTION_SCHEMA,
            "train": payload["train"],
            "initialization": payload["initialization"],
            "r10_budget": payload["r10_budget"],
        },
        path,
    )


def render_summary(payload: dict[str, Any]) -> str:
    budget = payload["r10_budget"]
    lines = [
        "# R10 R4-F Equal-Epochs Summary",
        "",
        f"- status: `{payload['status']}`",
        f"- model: `{payload['model_path']}`",
        f"- initialized_from: `{payload['initialization'].get('source_checkpoint')}`",
        f"- target_regen3_rows: `{budget['target_regen3_rows']}`",
        f"- cumulative_regen3_rows: `{payload['train']['cumulative_regen3_rows']}`",
        f"- cumulative_rows_seen: `{payload['train']['cumulative_rows_seen']}`",
        f"- elapsed_seconds: `{payload['train']['elapsed_seconds']}`",
        "",
        "## Offline Common Split",
        "",
        "| model | top1 | top3 | seq exact | intent top1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["offline_common_split"].items():
        lines.append(
            f"| `{name}` | {metrics.get('top1')} | {metrics.get('top3')} | {metrics.get('sequence_exact_sampled')} | {metrics.get('intent_top1', '')} |"
        )
    calibration = payload.get("top_player_calibration") or {}
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- rows_scored: `{calibration.get('rows_scored')}`",
            f"- action_ece: `{calibration.get('ece')}`",
            f"- confident_wrong_count: `{calibration.get('confident_wrong_count')}`",
            "",
        ]
    )

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, prior_payload = r4f.load_r4f_model(Path(args.r9_checkpoint))
    v1b_payload = _load_checkpoint(Path(args.v1b_checkpoint))
    prior_sources = _source_counts(prior_payload.get("train") or {})
    prior_regen3_rows = int(args.prior_regen3_rows) if args.prior_regen3_rows is not None else int(prior_sources.get("regen3", 0))
    target_regen3_rows = int(args.target_regen3_rows) if args.target_regen3_rows is not None else int(v1b_payload.get("total_rows_seen") or 0)
    corpus_train_rows = int(args.corpus_train_rows) if args.corpus_train_rows is not None else _train_rows_from_budget(Path(args.training_budget_audit))
    train = train_to_regen3_parity(
        model,
        prior_payload,
        args,
        prior_regen3_rows=prior_regen3_rows,
        target_regen3_rows=target_regen3_rows,
        corpus_train_rows=corpus_train_rows,
    )
    model.eval()
    val_r9 = r4f.collect_r9_validation(Path(args.refgames_dir), limit=int(args.val_rows))
    val_base = r4f.collect_base_validation(Path(args.refgames_dir), limit=int(args.val_rows))
    metrics_r4f = r4f.evaluate_r4f(model, val_r9, batch_size=int(args.batch_size))
    r4a_model = r4f.load_r4a_model(Path(args.r4a_checkpoint))
    v1b_model = r4f.load_r4a_model(Path(args.v1b_checkpoint))
    metrics_r4a = r4f.r4.evaluate_autoregressive(r4a_model, val_base, batch_size=int(args.batch_size), weighted=False)
    metrics_v1b = r4f.r4.evaluate_autoregressive(v1b_model, val_base, batch_size=int(args.batch_size), weighted=False)
    calibration = r4f.action_ece_on_top_player(model, Path(args.top_player_examples), limit=int(args.calibration_rows), batch_size=int(args.batch_size))
    model_path = out_dir / "selected_ar_bc_r4f_candidate.pt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "status": "trained_to_regen3_epoch_parity_not_panel_adjudicated",
        "model_path": str(model_path),
        "input_dim": int(prior_payload["input_dim"]),
        "hidden_dim": int(prior_payload["hidden_dim"]),
        "depth": int(prior_payload["depth"]),
        "embed_dim": int(prior_payload["embed_dim"]),
        "dropout": float(prior_payload.get("dropout", 0.0)),
        "added_second_order_dims": len(r4f.SECOND_ORDER_STATE_SCHEMA) + len(r4f.SECOND_ORDER_OPTION_SCHEMA),
        "feature_schema": {
            "state": r4f.SECOND_ORDER_STATE_SCHEMA,
            "option": r4f.SECOND_ORDER_OPTION_SCHEMA,
        },
        "initialization": {
            "source_checkpoint": str(Path(args.r9_checkpoint)),
            "source_schema_version": prior_payload.get("schema_version"),
            "warm_start_valid": True,
            "prior_train": prior_payload.get("train"),
        },
        "r10_budget": {
            "v1b_checkpoint": str(Path(args.v1b_checkpoint)),
            "v1b_total_rows_seen": int(v1b_payload.get("total_rows_seen") or 0),
            "v1b_added_rows": int(v1b_payload.get("added_rows") or 0),
            "corpus_train_rows": corpus_train_rows,
            "prior_r4f_rows_seen": int((prior_payload.get("train") or {}).get("rows_seen") or 0),
            "prior_r4f_regen3_rows": prior_regen3_rows,
            "target_regen3_rows": target_regen3_rows,
            "effective_epoch_basis": "regen3 source rows; top-player replay rows are retained as R4-F auxiliary mix and counted separately.",
        },
        "train": train,
        "offline_common_split": {
            "R4-A_current": metrics_r4a,
            "v1b_incumbent": metrics_v1b,
            "R4-F_equal_epochs": metrics_r4f,
        },
        "top_player_calibration": calibration,
        "decision_rule": "Proceed to leakage/canary/panel/probes; selection requires panel >= v1b, canary >= incumbent, safety 0/0/0, and P1 reduced.",
    }
    save_checkpoint(model_path, model, payload)
    preservation = preserve_checkpoint(
        model_path,
        mission=str(args.backup_mission),
        name="r4f-equal-epochs",
        reason="R4-F equal-epochs checkpoint produced by R10/R12 source recipe",
    )
    payload["checkpoint_preservation"] = preservation
    _write_json(out_dir / "r4f_equal_epochs_summary.json", payload)
    _write_text(out_dir / "r4f_equal_epochs_summary.md", render_summary(payload))
    print(json.dumps({k: payload[k] for k in ("status", "r10_budget", "train", "offline_common_split", "top_player_calibration")}, indent=2, sort_keys=True, default=str)[:8000])

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R10: continue R9 R4-F until regen3 exposure reaches v1b equal-epochs parity.")
    parser.add_argument("--refgames-dir", default=str(r4f.DEFAULT_REGEN3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--r9-checkpoint", default=str(DEFAULT_R9))
    parser.add_argument("--v1b-checkpoint", default=str(DEFAULT_V1B))
    parser.add_argument("--r4a-checkpoint", default=str(DEFAULT_R4A))
    parser.add_argument("--top-player-examples", default=str(r4f.DEFAULT_TOP_PLAYER))
    parser.add_argument("--training-budget-audit", default=str(DEFAULT_REGEN3_COUNT))
    parser.add_argument("--prior-regen3-rows", type=int, default=None)
    parser.add_argument("--target-regen3-rows", type=int, default=None)
    parser.add_argument("--corpus-train-rows", type=int, default=None)
    parser.add_argument("--skip-rows", type=int, default=None)
    parser.add_argument("--top-player-rows", type=int, default=5000)
    parser.add_argument("--top-player-interval", type=int, default=8)
    parser.add_argument("--val-rows", type=int, default=20000)
    parser.add_argument("--calibration-rows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--intent-loss-weight", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--backup-mission", default="r12")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
