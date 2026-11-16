from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r9_r4f_retrain as r4f


SCHEMA_VERSION = "deepmew.r10.live_gap_autopsy.v0"
DEFAULT_OUT = ROOT / ".research" / "r10" / "lane3_live_gap_autopsy"
BASE_DIM = 280


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _round(value: Any, digits: int = 6) -> float | None:
    if value is None or value == "":
        return None

    return round(float(value), digits)


def _policy_row(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    for row in rows:
        if row.get("policy") == policy:
            return row

    return {}


def _offline_metrics(summary: dict[str, Any], key: str) -> dict[str, Any]:
    row = (summary.get("offline_common_split") or {}).get(key) or {}

    return {
        "offline_top1": _round(row.get("top1")),
        "offline_top3": _round(row.get("top3")),
        "offline_seq_exact": _round(row.get("sequence_exact_sampled")),
        "offline_step_rows": row.get("step_rows"),
        "offline_intent_top1": _round(row.get("intent_top1")),
    }


def _canary_metrics(summary: dict[str, Any], policy: str, v1b_policy: str = "v1b_incumbent") -> dict[str, Any]:
    rows = summary.get("rows") or []
    row = _policy_row(rows, policy)
    v1b = _policy_row(rows, v1b_policy)

    return {
        "canary_winrate": _round(row.get("winrate")),
        "canary_delta_v1b": _round(float(row.get("winrate", 0.0)) - float(v1b.get("winrate", 0.0))) if row and v1b else None,
        "canary_illegal": int(row.get("illegal", 0) or 0),
        "canary_crashes": int(row.get("crashes", 0) or 0),
        "canary_timeouts": int(row.get("timeouts", 0) or 0),
    }


def _panel_metrics(path: Path, policy: str, v1b_policy: str = "v1b_incumbent") -> dict[str, Any]:
    rows = _read_csv(path)
    row = _policy_row(rows, policy)
    v1b = _policy_row(rows, v1b_policy)

    return {
        "panel_overall": _round(row.get("overall_pooled")),
        "panel_delta_v1b": _round(float(row.get("overall_pooled", 0.0)) - float(v1b.get("overall_pooled", 0.0))) if row and v1b else None,
        "panel_seed_se": _round(row.get("overall_seed_se")),
        "panel_illegal": int(float(row.get("illegal", 0) or 0)),
        "panel_crashes": int(float(row.get("crashes", 0) or 0)),
        "panel_timeouts": int(float(row.get("timeouts", 0) or 0)),
    }


def _probe_metrics(summary: dict[str, Any], policy: str, v1b_policy: str = "v1b") -> dict[str, Any]:
    rows = summary.get("policies") or {}
    row = ((rows.get(policy) or {}).get("probes")) or {}
    v1b = ((rows.get(v1b_policy) or {}).get("probes")) or {}
    p1 = row.get("P1_riptide_whiff_rate")
    p2 = row.get("P2_mega_latency_mean")
    p3 = row.get("P3_hammer_low_deck_rate")
    p4 = row.get("P4_surfing_activation_rate")

    return {
        "p1_riptide_whiff": _round(p1),
        "p1_delta_v1b": _round(float(p1) - float(v1b.get("P1_riptide_whiff_rate"))) if p1 is not None and v1b.get("P1_riptide_whiff_rate") is not None else None,
        "p2_mega_latency": _round(p2),
        "p2_delta_v1b": _round(float(p2) - float(v1b.get("P2_mega_latency_mean"))) if p2 is not None and v1b.get("P2_mega_latency_mean") is not None else None,
        "p3_hammer_low_deck": _round(p3),
        "p3_delta_v1b": _round(float(p3) - float(v1b.get("P3_hammer_low_deck_rate"))) if p3 is not None and v1b.get("P3_hammer_low_deck_rate") is not None else None,
        "p4_surfing_activation": _round(p4),
        "p4_delta_v1b": _round(float(p4) - float(v1b.get("P4_surfing_activation_rate"))) if p4 is not None and v1b.get("P4_surfing_activation_rate") is not None else None,
    }


def build_scatter(args: argparse.Namespace) -> list[dict[str, Any]]:
    r9_summary = _read_json(Path(args.r9_summary))
    r10_summary = _read_json(Path(args.r10_summary))
    r9_canary = _read_json(Path(args.r9_canary))
    r10_canary = _read_json(Path(args.r10_canary))
    r9_probes = _read_json(Path(args.r9_probes))
    r10_probes = _read_json(Path(args.r10_probes))
    rows = []
    specs = [
        ("R4-A", "R9/R10 baseline", r10_summary, "R4-A_current", Path(args.r10_panel), "R4-A_incumbent", r10_canary, "R4-A_incumbent", r10_probes, "R4-A"),
        ("v1b", "R9/R10 incumbent", r10_summary, "v1b_incumbent", Path(args.r10_panel), "v1b_incumbent", r10_canary, "v1b_incumbent", r10_probes, "v1b"),
        ("R4-F equal-time", "R9", r9_summary, "R4-F_candidate", Path(args.r9_panel), "R4-F_candidate", r9_canary, "R4-F_candidate", r9_probes, "R4-F"),
        ("R4-F equal-epochs", "R10", r10_summary, "R4-F_equal_epochs", Path(args.r10_panel), "R4-F_candidate", r10_canary, "R4-F_candidate", r10_probes, "R4-F"),
    ]
    v1b_seq = float(_offline_metrics(r10_summary, "v1b_incumbent")["offline_seq_exact"] or 0.0)
    for label, lane, summary, offline_key, panel_path, panel_policy, canary_summary, canary_policy, probe_summary, probe_policy in specs:
        row = {"candidate": label, "lane": lane}
        row.update(_offline_metrics(summary, offline_key))
        row["offline_seq_delta_v1b"] = _round(float(row.get("offline_seq_exact") or 0.0) - v1b_seq)
        row.update(_panel_metrics(panel_path, panel_policy))
        row.update(_canary_metrics(canary_summary, canary_policy))
        row.update(_probe_metrics(probe_summary, probe_policy))
        rows.append(row)

    return rows


def build_panel_slices(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(path)
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "games": 0.0})
    for row in rows:
        key = (row["opponent"], row["policy"])
        grouped[key]["wins"] += float(row["wins"])
        grouped[key]["games"] += float(row["games"])
    opponents = sorted({key[0] for key in grouped})
    opponent_rows = []
    for opponent in opponents:
        values = {}
        for policy in ("R4-A_incumbent", "v1b_incumbent", "R4-F_candidate"):
            cell = grouped[(opponent, policy)]
            values[policy] = cell["wins"] / max(1.0, cell["games"])
        opponent_rows.append(
            {
                "opponent": opponent,
                "r4a_winrate": round(values["R4-A_incumbent"], 6),
                "v1b_winrate": round(values["v1b_incumbent"], 6),
                "r4f_winrate": round(values["R4-F_candidate"], 6),
                "r4f_minus_v1b": round(values["R4-F_candidate"] - values["v1b_incumbent"], 6),
                "r4f_minus_r4a": round(values["R4-F_candidate"] - values["R4-A_incumbent"], 6),
            }
        )
    opponent_rows.sort(key=lambda row: row["r4f_minus_v1b"])
    aggregate = _read_csv(path.parent / "aggregate.csv")
    v1b = _policy_row(aggregate, "v1b_incumbent")
    r4f_row = _policy_row(aggregate, "R4-F_candidate")
    group_rows = []
    for key in ("overall_pooled", "trivial_pooled", "v2_4_legacy_pooled", "references_pooled", "iono_pooled"):
        group_rows.append(
            {
                "slice": key,
                "v1b": _round(v1b.get(key)),
                "r4f": _round(r4f_row.get(key)),
                "r4f_minus_v1b": _round(float(r4f_row.get(key, 0.0)) - float(v1b.get(key, 0.0))),
            }
        )
    group_rows.sort(key=lambda row: row["r4f_minus_v1b"])

    return opponent_rows, group_rows


def _ablate_second_order(samples: list[Any]) -> list[Any]:
    ablated = []
    for sample in samples:
        features = []
        for option_features in sample.features:
            base = list(option_features[:BASE_DIM])
            features.append(base + [0.0] * (len(option_features) - BASE_DIM))
        ablated.append(replace(sample, features=features))

    return ablated


def run_conditioning_probe(args: argparse.Namespace) -> dict[str, Any]:
    model, checkpoint = r4f.load_r4f_model(Path(args.r4f_checkpoint))
    samples = r4f.collect_r9_validation(Path(args.refgames_dir), limit=int(args.val_rows))
    full = r4f.evaluate_r4f(model, samples, batch_size=int(args.batch_size))
    ablated = r4f.evaluate_r4f(model, _ablate_second_order(samples), batch_size=int(args.batch_size))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "checkpoint": str(Path(args.r4f_checkpoint)),
        "feature_mode": checkpoint.get("feature_mode"),
        "val_rows": len(samples),
        "base_dim_preserved": BASE_DIM,
        "second_order_dims_zeroed": int(checkpoint.get("input_dim", BASE_DIM)) - BASE_DIM,
        "full": {
            "top1": full.get("top1"),
            "top3": full.get("top3"),
            "sequence_exact_sampled": full.get("sequence_exact_sampled"),
            "intent_top1": full.get("intent_top1"),
        },
        "zero_second_order": {
            "top1": ablated.get("top1"),
            "top3": ablated.get("top3"),
            "sequence_exact_sampled": ablated.get("sequence_exact_sampled"),
            "intent_top1": ablated.get("intent_top1"),
        },
        "delta_zero_minus_full": {
            "top1": _round(float(ablated.get("top1", 0.0)) - float(full.get("top1", 0.0))),
            "top3": _round(float(ablated.get("top3", 0.0)) - float(full.get("top3", 0.0))),
            "sequence_exact_sampled": _round(float(ablated.get("sequence_exact_sampled", 0.0)) - float(full.get("sequence_exact_sampled", 0.0))),
            "intent_top1": _round(float(ablated.get("intent_top1", 0.0)) - float(full.get("intent_top1", 0.0))),
        },
    }

    return payload


def render_autopsy(scatter: list[dict[str, Any]], opponent_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], conditioning: dict[str, Any]) -> str:
    equal_epochs = next(row for row in scatter if row["candidate"] == "R4-F equal-epochs")
    equal_time = next(row for row in scatter if row["candidate"] == "R4-F equal-time")
    top_losses = opponent_rows[:4]
    top_wins = list(reversed(opponent_rows[-4:]))
    zero_delta = conditioning["delta_zero_minus_full"]
    lines = [
        "# R10 Offline-Live Autopsy",
        "",
        "## Verdict",
        "",
        "R4-F is partially resurrected at equal regen3 exposure, but not selected.",
        "",
        f"- Offline seq-exact delta vs v1b: `{equal_epochs['offline_seq_delta_v1b']}`",
        f"- Fresh panel delta vs v1b: `{equal_epochs['panel_delta_v1b']}`",
        f"- Canary delta vs v1b: `{equal_epochs['canary_delta_v1b']}`",
        f"- P1 delta vs v1b: `{equal_epochs['p1_delta_v1b']}`",
        f"- P3 delta vs v1b: `{equal_epochs['p3_delta_v1b']}`",
        "",
        "## Ranked Cause List",
        "",
        "1. Equal-time was undertrained, not feature-dead: equal-epochs moved offline seq-exact from "
        f"`{equal_time['offline_seq_exact']}` to `{equal_epochs['offline_seq_exact']}` and panel from `{equal_time['panel_overall']}` to `{equal_epochs['panel_overall']}`.",
        "2. Selection is blocked by canary fragility: equal-epochs R4-F beat v1b on the fresh panel but missed the random canary by one game.",
        "3. Offline-live coupling is weak: the offline seq-exact gain is much larger than the live panel gain, and the canary is negative.",
        "4. The reader signal helps the named Riptide failure: P1 improved versus v1b, but Hammer low-deck behavior regressed versus v1b.",
        "5. The panel lift is slice-dependent rather than universal, with v2.4-legacy regression offsetting gains elsewhere.",
        "",
        "## Panel Slice Losses",
        "",
        "| opponent | R4-F - v1b | R4-F | v1b |",
        "|---|---:|---:|---:|",
    ]
    for row in top_losses:
        lines.append(f"| `{row['opponent']}` | {row['r4f_minus_v1b']} | {row['r4f_winrate']} | {row['v1b_winrate']} |")
    lines.extend(["", "## Panel Slice Gains", "", "| opponent | R4-F - v1b | R4-F | v1b |", "|---|---:|---:|---:|"])
    for row in top_wins:
        lines.append(f"| `{row['opponent']}` | {row['r4f_minus_v1b']} | {row['r4f_winrate']} | {row['v1b_winrate']} |")
    lines.extend(["", "## Aggregate Slices", "", "| slice | R4-F - v1b | R4-F | v1b |", "|---|---:|---:|---:|"])
    for row in group_rows:
        lines.append(f"| `{row['slice']}` | {row['r4f_minus_v1b']} | {row['r4f']} | {row['v1b']} |")
    lines.extend(
        [
            "",
            "## Conditioning Probe",
            "",
            f"- zeroed second-order dims: `{conditioning['second_order_dims_zeroed']}`",
            f"- seq-exact delta zero-minus-full: `{zero_delta['sequence_exact_sampled']}`",
            f"- top1 delta zero-minus-full: `{zero_delta['top1']}`",
            f"- intent top1 delta zero-minus-full: `{zero_delta['intent_top1']}`",
            "",
            "## R11 Lever",
            "",
            "Do not select R4-F yet. The next lever should be a diagnostic/ranker-level fix for P3 low-deck Hammer risk and canary-stable random-opponent behavior before any new training arm.",
            "",
        ]
    )

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scatter = build_scatter(args)
    opponent_rows, group_rows = build_panel_slices(Path(args.r10_panel_by_opponent))
    conditioning = run_conditioning_probe(args)
    _write_csv(
        out_dir / "offline_live_scatter.csv",
        scatter,
        [
            "candidate",
            "lane",
            "offline_seq_exact",
            "offline_top1",
            "offline_top3",
            "offline_seq_delta_v1b",
            "panel_overall",
            "panel_delta_v1b",
            "canary_winrate",
            "canary_delta_v1b",
            "p1_riptide_whiff",
            "p1_delta_v1b",
            "p2_mega_latency",
            "p2_delta_v1b",
            "p3_hammer_low_deck",
            "p3_delta_v1b",
            "p4_surfing_activation",
            "p4_delta_v1b",
        ],
    )
    _write_csv(out_dir / "panel_opponent_slices.csv", opponent_rows, ["opponent", "r4a_winrate", "v1b_winrate", "r4f_winrate", "r4f_minus_v1b", "r4f_minus_r4a"])
    _write_csv(out_dir / "panel_aggregate_slices.csv", group_rows, ["slice", "v1b", "r4f", "r4f_minus_v1b"])
    _write_json(out_dir / "conditioning_probe.json", conditioning)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "scatter": scatter,
        "panel_opponent_slices": opponent_rows,
        "panel_aggregate_slices": group_rows,
        "conditioning_probe": conditioning,
    }
    _write_json(out_dir / "live_gap_autopsy.json", payload)
    _write_json(out_dir / "autopsy_summary.json", payload)
    _write_text(out_dir / "autopsy_summary.md", render_autopsy(scatter, opponent_rows, group_rows, conditioning))
    print(json.dumps({"scatter": scatter, "conditioning": conditioning["delta_zero_minus_full"], "worst_slices": opponent_rows[:4]}, indent=2, sort_keys=True))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R10 offline-live autopsy for R4-F equal-epochs.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--r9-summary", default=str(ROOT / ".research" / "r9" / "lane2_r4f_equal_epoch" / "r4f_retrain_summary.json"))
    parser.add_argument("--r10-summary", default=str(ROOT / ".research" / "r10" / "lane2_r4f_equal_epochs" / "r4f_equal_epochs_summary.json"))
    parser.add_argument("--r9-panel", default=str(ROOT / ".research" / "r9" / "lane3_panel_equal_epoch" / "aggregate.csv"))
    parser.add_argument("--r10-panel", default=str(ROOT / ".research" / "r10" / "lane2_panel_equal_epochs" / "aggregate.csv"))
    parser.add_argument("--r10-panel-by-opponent", default=str(ROOT / ".research" / "r10" / "lane2_panel_equal_epochs" / "by_opponent.csv"))
    parser.add_argument("--r9-canary", default=str(ROOT / ".research" / "r9" / "lane3_canary_equal_epoch" / "canary_summary.json"))
    parser.add_argument("--r10-canary", default=str(ROOT / ".research" / "r10" / "lane2_canary_equal_epochs" / "canary_summary.json"))
    parser.add_argument("--r9-probes", default=str(ROOT / ".research" / "r9" / "lane3_probes_equal_epoch" / "r4f_probes.json"))
    parser.add_argument("--r10-probes", default=str(ROOT / ".research" / "r10" / "lane2_probes_equal_epochs" / "r4f_probes.json"))
    parser.add_argument("--r4f-checkpoint", default=str(ROOT / ".research" / "r10" / "lane2_r4f_equal_epochs" / "selected_ar_bc_r4f_candidate.pt"))
    parser.add_argument("--refgames-dir", default=str(r4f.DEFAULT_REGEN3))
    parser.add_argument("--val-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
