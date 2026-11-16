from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from deepmew.arena.models import MatrixSummary, PairSummary


def format_pair_report(pair: PairSummary) -> str:
    lines = [
        f"# Pair: {pair.agent_a} vs {pair.agent_b}",
        "",
        f"games={pair.games} completed={pair.completed} crashes={pair.crashes} timeouts={pair.timeouts} avg_turns={pair.avg_turns:.2f}",
        f"{pair.agent_a}: {pair.a_wins} wins",
        f"{pair.agent_b}: {pair.b_wins} wins",
        f"draws: {pair.draws}",
        "",
        "## Safety counters",
    ]
    for agent, metrics in pair.metrics.items():
        lines.append(
            f"- {agent}: decisions={metrics['decisions']} "
            f"end_with_useful={metrics['end_with_useful_options']} "
            f"attack_skipped={metrics['attack_available_but_not_picked']} "
            f"unknown_action={metrics['unknown_action']} "
            f"risky_over_safer={metrics['risky_over_safer']}"
        )
    return "\n".join(lines)


def format_matrix_report(summary: MatrixSummary) -> str:
    labels = summary.agents
    lines = [
        "# Arena Matrix",
        "",
        f"games_per_pair={summary.games_per_pair} seed={summary.seed} include_self={summary.include_self}",
        "",
        "## Win-Rate Matrix",
        "",
        "| agent | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for agent in labels:
        row = []
        for opponent in labels:
            cell = summary.matrix[agent][opponent]
            row.append("-" if cell["win_rate"] is None else f"{cell['win_rate']:.2f}")
        lines.append(f"| {agent} | " + " | ".join(row) + " |")

    lines.extend(["", "## Rankings", ""])
    for index, item in enumerate(summary.rankings, start=1):
        lines.append(
            f"{index}. {item['agent']}: point_rate={item['point_rate']:.3f} "
            f"record={item['wins']}-{item['losses']}-{item['draws']} "
            f"crashes={item['crashes']} timeouts={item['timeouts']}"
        )
    lines.extend(["", "## Pair Details", ""])
    for pair in summary.pairs:
        lines.append(
            f"- {pair.agent_a} vs {pair.agent_b}: "
            f"{pair.a_wins}-{pair.b_wins}-{pair.draws}, "
            f"completed={pair.completed}/{pair.games}, "
            f"crashes={pair.crashes}, timeouts={pair.timeouts}, "
            f"avg_turns={pair.avg_turns:.2f}"
        )
    return "\n".join(lines)


def write_summary(summary: PairSummary | MatrixSummary, out_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(summary, PairSummary):
        stem = f"pair_{summary.agent_a}_vs_{summary.agent_b}_{stamp}"
        report = format_pair_report(summary)
    else:
        stem = f"matrix_{stamp}"
        report = format_matrix_report(summary)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    json_path = output_dir / f"{safe_stem}.json"
    report_path = output_dir / f"{safe_stem}.md"
    payload = asdict(summary)
    if isinstance(summary, MatrixSummary):
        payload["output_json"] = str(json_path)
        payload["output_report"] = str(report_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(report + "\n", encoding="utf-8")
    return json_path, report_path
