"""Measure R4-F-class checkpoint agreement with top-player ladder decisions (R14 Lane 2).

For each single-select decision made by a top-ranked player, score the legal
options with the checkpoint and check whether its argmax matches the expert's
choice. Slices by select context, rank bucket, and expert game outcome, and
reports the first-legal (index 0) baseline for calibration.

Rows are deterministically split by episode id (sha256 % 5 == 0 -> holdout) so
a model fine-tuned on the train split can be judged on unseen episodes.

Example
-------
    uv run python scripts/eval_r4f_top_player_agreement.py \
        --checkpoint "R4-F_reweighted=.research/r12/lane1_r4f_focal_reweight/selected_ar_bc_r4f_reweighted_candidate.pt" \
        --examples .research/r14/lane2_top_player_bc/replay_label_examples.jsonl \
        --split all --out-dir .research/r14/lane2_top_player_bc/agreement_incumbent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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

import run_r9_r4f_retrain as r4f  # noqa: E402


SCHEMA_VERSION = "deepmew.r14.top_player_agreement.v0"
HOLDOUT_MODULUS = 5


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def episode_split(episode_id: str) -> str:
    digest = hashlib.sha256(str(episode_id).encode("utf-8")).digest()

    return "holdout" if digest[0] % HOLDOUT_MODULUS == 0 else "train"


def iter_examples(path: Path, *, split: str) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split != "all" and episode_split(str(row.get("episode_id"))) != split:
                continue
            yield row


def _single_select(row: dict[str, Any]) -> bool:
    mask = row.get("action_mask") or {}
    max_count = mask.get("max_count")

    return isinstance(max_count, int) and max_count == 1


def evaluate_checkpoint(label: str, checkpoint: Path, rows: list[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    model, _payload = r4f.load_r4f_model(checkpoint)
    samples = []
    kept_rows = []
    for row in rows:
        encoded = r4f.encode_top_player_row_r9(row)
        if encoded is None:
            continue
        samples.append(encoded)
        kept_rows.append(row)

    total = Counter()
    by_context: dict[str, Counter] = defaultdict(Counter)
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    by_outcome: dict[str, Counter] = defaultdict(Counter)
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        chunk_rows = kept_rows[start:start + batch_size]
        first_steps = [r4f.r4.make_step_samples(sample, weighted=False)[0] for sample in chunk]
        packed = r4f.batch_steps_r4f(first_steps)
        with torch.no_grad():
            scores = model(packed.features, packed.option_meta).masked_fill(~packed.mask, -1e9)
            probs = torch.softmax(scores[:, :-1], dim=1)
        for index, (sample, row) in enumerate(zip(chunk, chunk_rows)):
            legal = sample.legal_count
            row_probs = probs[index, :legal]
            pred = int(row_probs.argmax().item())
            target = int(sample.target_index)
            top3 = set(torch.topk(row_probs, k=min(3, legal)).indices.tolist())
            counters = [
                total,
                by_context[str(sample.metadata.get("select_context") or "")],
                by_bucket[str(row.get("rank_bucket") or "")],
                by_outcome["expert_won" if row.get("reward") == 1 else "expert_lost"],
            ]
            for counter in counters:
                counter["n"] += 1
                counter["agree"] += int(pred == target)
                counter["agree_top3"] += int(target in top3)
                counter["expert_first_legal"] += int(target == 0)
                counter["model_first_legal"] += int(pred == 0)

    def _rates(counter: Counter) -> dict[str, Any]:
        n = max(1, counter["n"])

        return {
            "n": counter["n"],
            "agreement": round(counter["agree"] / n, 6),
            "agreement_top3": round(counter["agree_top3"] / n, 6),
            "expert_first_legal_rate": round(counter["expert_first_legal"] / n, 6),
            "model_first_legal_rate": round(counter["model_first_legal"] / n, 6),
        }

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "rows_scored": total["n"],
        "overall": _rates(total),
        "by_context": {key: _rates(counter) for key, counter in sorted(by_context.items())},
        "by_rank_bucket": {key: _rates(counter) for key, counter in sorted(by_bucket.items())},
        "by_expert_outcome": {key: _rates(counter) for key, counter in sorted(by_outcome.items())},
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R14 Lane 2 - Top-Player Agreement",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- examples: `{payload['inputs']['examples']}`",
        f"- split: `{payload['inputs']['split']}`",
        f"- single-select rows evaluated: {payload['single_select_rows']}",
        "",
    ]
    for result in payload["checkpoints"]:
        overall = result["overall"]
        lines.extend(
            [
                f"## {result['label']}",
                "",
                f"- rows scored: {result['rows_scored']}",
                f"- top-1 agreement with expert: **{overall['agreement']}** (top-3: {overall['agreement_top3']})",
                f"- expert chose first-legal: {overall['expert_first_legal_rate']}; model chose first-legal: {overall['model_first_legal_rate']}",
                "",
                "| slice | key | n | agreement | top3 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for slice_name in ("by_context", "by_rank_bucket", "by_expert_outcome"):
            for key, rates in result[slice_name].items():
                lines.append(f"| {slice_name} | {key or '(none)'} | {rates['n']} | {rates['agreement']} | {rates['agreement_top3']} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint agreement with top-player ladder decisions.")
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path, repeatable.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--split", choices=("all", "train", "holdout"), default="all")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    rows = [row for row in iter_examples(Path(args.examples), split=args.split) if _single_select(row)]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {"examples": args.examples, "split": args.split, "checkpoints": list(args.checkpoint)},
        "single_select_rows": len(rows),
        "checkpoints": [],
    }
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        payload["checkpoints"].append(evaluate_checkpoint(label, Path(path), rows, batch_size=int(args.batch_size)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agreement_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "agreement_report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"single_select_rows": len(rows), "results": [{r['label']: r['overall']} for r in payload["checkpoints"]]}, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
