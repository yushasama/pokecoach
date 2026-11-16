"""R14 Lane 3: fine-tune the R4-F reweighted champion on fresh top-player ladder rows.

Continues from the R12 focal-reweighted candidate with the same P1/P3 slice
weighting, mixing top-player decision rows (train split only; holdout episodes
stay out for the agreement verdict) into the regen3 stream every ``--top-player-interval``
reference rows. Produces a candidate checkpoint only - panel + powered canary +
safety decide selection under DEC-066.

Example
-------
    uv run python scripts/run_r14_top_player_finetune.py \
        --top-player-examples .research/r14/lane2_top_player_bc/replay_label_examples.jsonl \
        --out-dir .research/r14/lane3_top_player_finetune
"""

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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from checkpoint_preservation import preserve_checkpoint  # noqa: E402
import run_r9_r4f_retrain as r4f  # noqa: E402
import run_r12_r4f_focal_reweight as r12  # noqa: E402
from eval_r4f_top_player_agreement import episode_split  # noqa: E402


SCHEMA_VERSION = "deepmew.r14.top_player_finetune.v0"
DEFAULT_OUT = ROOT / ".research" / "r14" / "lane3_top_player_finetune"
DEFAULT_INIT = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
DEFAULT_EXAMPLES = ROOT / ".research" / "r14" / "lane2_top_player_bc" / "replay_label_examples.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_top_player_train_rows(
    path: Path,
    *,
    shell_contains: str = "",
    emphasize_shell_contains: str = "",
    emphasize_copies: int = 1,
) -> tuple[list[Any], dict[str, int]]:
    encoded_rows: list[Any] = []
    counts = {"train_rows": 0, "holdout_rows": 0, "encode_failures": 0, "shell_filtered_out": 0, "emphasized_rows": 0}
    shell_filter = shell_contains.strip().lower()
    emphasis_filter = emphasize_shell_contains.strip().lower()
    copies = max(1, int(emphasize_copies))
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if episode_split(str(row.get("episode_id"))) != "train":
                counts["holdout_rows"] += 1
                continue
            shell = str(row.get("shell") or "").lower()
            if shell_filter and shell_filter not in shell:
                counts["shell_filtered_out"] += 1
                continue
            encoded = r4f.encode_top_player_row_r9(row)
            if encoded is None:
                counts["encode_failures"] += 1
                continue
            repeats = copies if emphasis_filter and emphasis_filter in shell else 1
            if repeats > 1:
                counts["emphasized_rows"] += 1
            encoded_rows.extend([encoded] * repeats)
            counts["train_rows"] += 1

    return encoded_rows, counts


def _iter_refgames_filtered(args: argparse.Namespace) -> Iterable[Any]:
    winners_only = bool(getattr(args, "refgames_winners_only", False))
    emitted = 0
    limit = int(args.train_rows)
    for encoded in r4f.iter_refgame_encoded_r9(Path(args.refgames_dir), split="train", limit=None):
        if winners_only and not _as_dict(encoded.metadata.get("outcome")).get("won"):
            continue
        yield encoded
        emitted += 1
        if limit > 0 and emitted >= limit:
            break


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def iter_mixed(args: argparse.Namespace, top_rows: list[Any]) -> Iterable[Any]:
    top_index = 0
    interval = max(1, int(args.top_player_interval))
    for ref_index, encoded in enumerate(_iter_refgames_filtered(args), start=1):
        yield encoded
        if top_rows and ref_index % interval == 0:
            yield top_rows[top_index % len(top_rows)]
            top_index += 1


def train_mixed(model: r4f.R4FIntentAutoregressiveScorer, args: argparse.Namespace, top_rows: list[Any]) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rows_seen = 0
    top_rows_seen = 0
    step_rows_seen = 0
    batches = 0
    action_loss_total = 0.0
    intent_loss_total = 0.0
    slice_counts: Counter[str] = Counter()

    def _flush(batch: list[Any]) -> None:
        nonlocal step_rows_seen, batches, action_loss_total, intent_loss_total
        packed = r12.batch_steps_r12(batch, args)
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

    batch: list[Any] = []
    for encoded in iter_mixed(args, top_rows):
        rows_seen += 1
        top_rows_seen += int(str(encoded.metadata.get("source") or "") == "top_player")
        for step in r4f.r4.make_step_samples(encoded, weighted=False):
            batch.append(step)
            for flag in r12._slice_flags(step):
                slice_counts[flag] += 1
        if len(batch) >= int(args.batch_size):
            _flush(batch)
            batch = []
    if batch:
        _flush(batch)

    return {
        "rows_seen": rows_seen,
        "top_player_row_exposures": top_rows_seen,
        "unique_top_player_train_rows": len(top_rows),
        "step_rows_seen": step_rows_seen,
        "batches": batches,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mean_action_loss": round(action_loss_total / max(1, batches), 6),
        "mean_intent_loss": round(intent_loss_total / max(1, batches), 6),
        "slice_step_counts": dict(slice_counts),
        "p1_weight_cap": float(args.p1_weight_cap),
        "p3_weight_multiplier": float(args.p3_weight_multiplier),
        "top_player_interval": int(args.top_player_interval),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, prior_payload = r4f.load_r4f_model(Path(args.init_checkpoint))
    top_rows, top_counts = load_top_player_train_rows(
        Path(args.top_player_examples),
        shell_contains=str(args.shell_contains),
        emphasize_shell_contains=str(args.emphasize_shell_contains),
        emphasize_copies=int(args.emphasize_copies),
    )
    train = train_mixed(model, args, top_rows)
    model.eval()
    val = r4f.collect_r9_validation(Path(args.refgames_dir), limit=int(args.val_rows))
    metrics = r4f.evaluate_r4f(model, val, batch_size=int(args.batch_size))
    model_path = out_dir / str(args.candidate_name)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "status": "trained_top_player_finetune_not_panel_adjudicated",
        "model_path": str(model_path),
        "input_dim": int(prior_payload["input_dim"]),
        "hidden_dim": int(prior_payload["hidden_dim"]),
        "depth": int(prior_payload["depth"]),
        "embed_dim": int(prior_payload["embed_dim"]),
        "dropout": float(prior_payload.get("dropout", 0.0)),
        "initialization": {
            "source_checkpoint": str(Path(args.init_checkpoint)),
            "source_schema_version": prior_payload.get("schema_version"),
        },
        "top_player_examples": {"path": str(args.top_player_examples), **top_counts},
        "train": train,
        "offline_common_split": {"R4-F_topplayer": metrics},
        "selection_note": "candidate only; DEC-066 incumbent-relative panel + powered canary + safety decide",
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
        name=str(args.backup_name),
        reason="Top-player fine-tune candidate checkpoint",
    )
    payload["checkpoint_preservation"] = preservation
    (out_dir / "top_player_finetune_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("status", "train", "offline_common_split")}, indent=2, sort_keys=True, default=str)[:4000])

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R14 Lane 3: top-player fine-tune of the R4-F reweighted champion.")
    parser.add_argument("--refgames-dir", default=str(r4f.DEFAULT_REGEN3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--top-player-examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--train-rows", type=int, default=200000)
    parser.add_argument("--top-player-interval", type=int, default=4)
    parser.add_argument("--refgames-winners-only", action="store_true")
    parser.add_argument("--shell-contains", default="")
    parser.add_argument("--emphasize-shell-contains", default="")
    parser.add_argument("--emphasize-copies", type=int, default=1)
    parser.add_argument("--candidate-name", default="selected_ar_bc_r4f_topplayer_candidate.pt")
    parser.add_argument("--backup-name", default="r4f-topplayer-finetune")
    parser.add_argument("--val-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--intent-loss-weight", type=float, default=0.15)
    parser.add_argument("--p1-weight-cap", type=float, default=1.2)
    parser.add_argument("--p3-weight-multiplier", type=float, default=2.5)
    parser.add_argument("--max-row-weight", type=float, default=5.0)
    parser.add_argument("--backup-mission", default="r14")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
