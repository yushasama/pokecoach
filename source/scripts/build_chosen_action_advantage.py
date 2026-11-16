"""Build Stage 1 chosen-action advantage rows from refgame transition rows.

This script only scores the logged action's pre/post observation. It does not
train a value model, rewrite AWR, or branch counterfactual legal actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_unified_r3_value as value_source  # noqa: E402
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown  # noqa: E402
from deepmew.replay.chosen_action_advantage import (  # noqa: E402
    VALUE_ACTIVATION_SIGMOID,
    VALUE_SCALE_WIN_PROBABILITY,
    build_advantage_rows,
    iter_jsonl_rows,
    iter_refgame_rows,
    render_report,
)


JSONL_FILENAME = "chosen_action_advantage.jsonl"
SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"


class CheckpointValueScorer:
    """Thin wrapper around scripts/run_unified_r3_value.py's model contract."""

    value_scale = VALUE_SCALE_WIN_PROBABILITY
    value_activation_used = VALUE_ACTIVATION_SIGMOID

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        payload = torch.load(self.checkpoint, map_location=self.device)
        self.feature_config = str(payload["feature_config"])
        self.model = value_source.ValueMLP(int(payload["input_dim"]), dict(payload["arch"]))
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, observation: dict[str, Any]) -> float:
        features = value_source._features({"raw_observation": observation}, self.feature_config)
        tensor = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logit = self.model(tensor)[0]
            return float(torch.sigmoid(logit).item())


def _iter_sources(args: argparse.Namespace) -> tuple[Iterable[dict[str, Any]], str]:
    if args.refgames_dir:
        source = str(Path(args.refgames_dir))
        rows = iter_refgame_rows(args.refgames_dir)
    else:
        source = ",".join(str(Path(path)) for path in args.rows)
        rows = iter_jsonl_rows(args.rows)
    if int(args.limit) > 0:
        rows = islice(rows, int(args.limit))
    return rows, source


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.refgames_dir and not args.rows:
        raise ValueError("provide --refgames-dir or at least one --rows path")
    if not args.value_checkpoint and not args.dry_run:
        raise ValueError("provide --value-checkpoint, or pass --dry-run to validate rows without values")

    scorer = None if args.dry_run else CheckpointValueScorer(args.value_checkpoint, device=args.device)
    rows_iter, source = _iter_sources(args)
    result = build_advantage_rows(rows_iter, source=source, scorer=scorer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = write_jsonl(out_dir / JSONL_FILENAME, result.rows)
    summary = dict(result.summary)
    summary["source"] = source
    summary["dry_run"] = bool(args.dry_run)
    summary["value_checkpoint"] = None if args.dry_run else str(Path(args.value_checkpoint))
    summary["outputs"] = {
        "rows": str(jsonl_path),
        "summary": str(out_dir / SUMMARY_FILENAME),
        "report": str(out_dir / REPORT_FILENAME),
    }
    write_json(out_dir / SUMMARY_FILENAME, summary)
    write_markdown(out_dir / REPORT_FILENAME, render_report(summary))

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refgames-dir", help="Directory with shards/decisions_shard_*.jsonl.gz rows.")
    parser.add_argument("--rows", action="append", default=[], help="JSONL or .jsonl.gz row file. Repeatable.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--value-checkpoint", help="Explicit run_unified_r3_value.py checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate/report without computing V(s).")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        parser.exit(2, f"error: {type(exc).__name__}: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
