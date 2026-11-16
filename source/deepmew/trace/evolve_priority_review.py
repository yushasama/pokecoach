"""Review artifact builder for the experimental evolve_priority ablation."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_json, write_markdown, write_text
from deepmew.reports.markdown import bullets, section, table, title


RUN_KIND = "deck_evolve_priority"
HANDOFF_KIND = "babymew_evolve_priority_review"


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_handoff_dir(
    *,
    out_root: Path | str = ".research/handoff_bundles",
    kind: str = HANDOFF_KIND,
    now: datetime | None = None,
) -> Path:
    created = now or _utc_now()
    bundle_dir = Path(out_root) / created.strftime("%Y%m%d") / make_run_id(kind, now=created)
    bundle_dir.mkdir(parents=True)
    return bundle_dir


def _metric(audit: dict[str, Any], key: str) -> Any:
    return (audit.get("metrics") or {}).get(key)


def _tag_count(audit: dict[str, Any], key: str) -> int:
    return int(((audit.get("metrics") or {}).get("tag_counts") or {}).get(key, 0))


def _agent_win_rate(pair: dict[str, Any], agent: str) -> float | None:
    completed = int(pair.get("completed") or 0)
    if completed <= 0:
        return None
    if pair.get("agent_a") == agent:
        return float(pair.get("a_wins") or 0) / completed
    if pair.get("agent_b") == agent:
        return float(pair.get("b_wins") or 0) / completed
    return None


def _matrix_point_rate(matrix: dict[str, Any], agent: str) -> float | None:
    for row in matrix.get("rankings") or []:
        if row.get("agent") == agent:
            return row.get("point_rate")
    return None


def build_comparison(
    *,
    pair_summary: dict[str, Any],
    matrix_summary: dict[str, Any],
    default_audit: dict[str, Any],
    ablation_audit: dict[str, Any],
) -> dict[str, Any]:
    default_play = _tag_count(default_audit, "play_beat_evolve")
    ablation_play = _tag_count(ablation_audit, "play_beat_evolve")
    default_non_target = _tag_count(default_audit, "chosen_play_generic_or_nonessential")
    ablation_non_target = _tag_count(ablation_audit, "chosen_play_generic_or_nonessential")
    ablation_agent = ablation_audit.get("agent_label") or "v2.4[evolve_priority=new]"
    default_agent = default_audit.get("agent_label") or "v2.4"
    pair_win_rate = _agent_win_rate(pair_summary, str(ablation_agent))
    matrix_point_rate = _matrix_point_rate(matrix_summary, str(ablation_agent))
    default_point_rate = _matrix_point_rate(matrix_summary, str(default_agent))

    if pair_win_rate is None and matrix_point_rate is None:
        win_rate_assessment = "mixed"
    elif pair_win_rate is not None and pair_win_rate > 0.5:
        win_rate_assessment = "improved"
    elif pair_win_rate is not None and pair_win_rate < 0.5:
        win_rate_assessment = "worsened"
    elif matrix_point_rate is not None and default_point_rate is not None and matrix_point_rate > default_point_rate:
        win_rate_assessment = "improved"
    elif matrix_point_rate is not None and default_point_rate is not None and matrix_point_rate < default_point_rate:
        win_rate_assessment = "worsened"
    else:
        win_rate_assessment = "mixed"

    return {
        "default_agent": default_agent,
        "ablation_agent": ablation_agent,
        "play_beat_evolve": {
            "default": default_play,
            "ablation": ablation_play,
            "delta": ablation_play - default_play,
            "decreased": ablation_play < default_play,
        },
        "non_target_play_beat_evolve": {
            "default": default_non_target,
            "ablation": ablation_non_target,
            "delta": ablation_non_target - default_non_target,
            "decreased": ablation_non_target < default_non_target,
        },
        "same_frame_evolve_chosen": {
            "default": _metric(default_audit, "same_frame_chosen_count"),
            "ablation": _metric(ablation_audit, "same_frame_chosen_count"),
        },
        "evolve_never_observed": {
            "default": _metric(default_audit, "never_observed_count"),
            "ablation": _metric(ablation_audit, "never_observed_count"),
        },
        "pair": {
            "agent_a": pair_summary.get("agent_a"),
            "agent_b": pair_summary.get("agent_b"),
            "completed": pair_summary.get("completed"),
            "a_wins": pair_summary.get("a_wins"),
            "b_wins": pair_summary.get("b_wins"),
            "draws": pair_summary.get("draws"),
            "crashes": pair_summary.get("crashes"),
            "timeouts": pair_summary.get("timeouts"),
            "ablation_win_rate": pair_win_rate,
        },
        "matrix": {
            "agents": matrix_summary.get("agents"),
            "rankings": matrix_summary.get("rankings"),
            "ablation_point_rate": matrix_point_rate,
            "default_point_rate": default_point_rate,
        },
        "win_rate_assessment": win_rate_assessment,
        "recommendation": "keep experimental and expand test" if ablation_play < default_play else "revise rule before promotion",
    }


def build_report(comparison: dict[str, Any]) -> str:
    play = comparison["play_beat_evolve"]
    non_target = comparison["non_target_play_beat_evolve"]
    pair = comparison["pair"]
    matrix = comparison["matrix"]
    lines = [title("BabyMew Evolve Priority Review")]
    lines.append(section("Scope"))
    lines.append(
        bullets(
            [
                "Experimental ablation only: `v2.4[evolve_priority=new]`.",
                "Default `v2.4` remains unchanged.",
                "The rule delays non-target PLAY when an important Mega-line EVOLVE is already legal and no KO/prize-pressure ATTACK is legal.",
                "ATTACH and ATTACK scores are not modified.",
                "`attach_timing=new` remains separate and experimental.",
            ]
        )
    )
    lines.append(section("Comparison"))
    lines.append(
        table(
            ["Metric", "Default", "Ablation", "Delta"],
            [
                ["PLAY beat EVOLVE", play["default"], play["ablation"], play["delta"]],
                ["Non-target PLAY beat EVOLVE", non_target["default"], non_target["ablation"], non_target["delta"]],
                [
                    "Same-frame EVOLVE chosen",
                    comparison["same_frame_evolve_chosen"]["default"],
                    comparison["same_frame_evolve_chosen"]["ablation"],
                    "",
                ],
                [
                    "EVOLVE never observed",
                    comparison["evolve_never_observed"]["default"],
                    comparison["evolve_never_observed"]["ablation"],
                    "",
                ],
            ],
        )
    )
    lines.append(section("Arena"))
    lines.append(
        bullets(
            [
                f"Pair: {pair['agent_a']} vs {pair['agent_b']} completed={pair['completed']} record={pair['a_wins']}-{pair['b_wins']}-{pair['draws']} crashes={pair['crashes']} timeouts={pair['timeouts']}.",
                f"Ablation pair win rate: {pair['ablation_win_rate']}.",
                f"Matrix ablation point rate: {matrix['ablation_point_rate']}; default point rate: {matrix['default_point_rate']}.",
                f"Win-rate assessment: {comparison['win_rate_assessment']}.",
            ]
        )
    )
    lines.append(section("Recommendation"))
    lines.append(bullets([comparison["recommendation"]]))
    return "\n".join(lines)


def export_bundle(
    *,
    run_dir: Path,
    comparison: dict[str, Any],
    report: str,
    handoff_root: Path | str,
) -> Path:
    bundle_dir = _create_handoff_dir(out_root=handoff_root)
    write_markdown(bundle_dir / "README.md", "# BabyMew Evolve Priority Review Bundle\n\nInspect `executive_summary.md` first.\n")
    write_markdown(bundle_dir / "executive_summary.md", report)
    write_json(bundle_dir / "comparison.json", comparison)
    write_text(
        bundle_dir / "relevant_paths.txt",
        "\n".join(
            [
                str(run_dir),
                "deepmew/policy_rules/evolve_priority.py",
                "deepmew/policy/scoring.py",
                "deepmew/arena/specs.py",
                "scripts/audit_deck_v3_evolve_sequencing.py",
                "tests/test_evolve_priority.py",
            ]
        )
        + "\n",
    )
    write_manifest(
        bundle_dir / "manifest.json",
        build_manifest(
            run_id=bundle_dir.name,
            kind=HANDOFF_KIND,
            command="export evolve priority review bundle",
            inputs={"run_dir": str(run_dir)},
            outputs={
                "README.md": str(bundle_dir / "README.md"),
                "executive_summary.md": str(bundle_dir / "executive_summary.md"),
                "comparison.json": str(bundle_dir / "comparison.json"),
                "relevant_paths.txt": str(bundle_dir / "relevant_paths.txt"),
            },
            summary=comparison,
        ),
    )
    return bundle_dir


def write_review_outputs(
    *,
    run_dir: Path,
    command: str,
    pair_summary_path: Path,
    matrix_summary_path: Path,
    default_audit_path: Path,
    ablation_audit_path: Path,
    export_handoff_bundle: bool = False,
    handoff_root: Path | str = ".research/handoff_bundles",
) -> dict[str, Any]:
    pair_summary = load_json(pair_summary_path)
    matrix_summary = load_json(matrix_summary_path)
    default_audit = load_json(default_audit_path)
    ablation_audit = load_json(ablation_audit_path)
    comparison = build_comparison(
        pair_summary=pair_summary,
        matrix_summary=matrix_summary,
        default_audit=default_audit,
        ablation_audit=ablation_audit,
    )
    report = build_report(comparison)
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "report": run_dir / "report.md",
        "pair_summary": run_dir / "pair_summary.json",
        "matrix_summary": run_dir / "matrix_summary.json",
        "default_evolve_counts": run_dir / "default_evolve_counts.json",
        "ablation_evolve_counts": run_dir / "ablation_evolve_counts.json",
        "comparison": run_dir / "comparison.json",
    }
    write_markdown(outputs["report"], report)
    shutil.copyfile(pair_summary_path, outputs["pair_summary"])
    shutil.copyfile(matrix_summary_path, outputs["matrix_summary"])
    write_json(outputs["default_evolve_counts"], default_audit.get("metrics") or {})
    write_json(outputs["ablation_evolve_counts"], ablation_audit.get("metrics") or {})
    write_json(outputs["comparison"], comparison)

    bundle_dir = None
    if export_handoff_bundle:
        bundle_dir = export_bundle(
            run_dir=run_dir,
            comparison=comparison,
            report=report,
            handoff_root=handoff_root,
        )

    manifest = build_manifest(
        run_id=run_dir.name,
        kind=RUN_KIND,
        command=command,
        inputs={
            "pair_summary": str(pair_summary_path),
            "matrix_summary": str(matrix_summary_path),
            "default_audit": str(default_audit_path),
            "ablation_audit": str(ablation_audit_path),
        },
        outputs={key: str(value) for key, value in outputs.items()},
        summary={**comparison, "handoff_bundle": str(bundle_dir) if bundle_dir else None},
    )
    write_manifest(run_dir / "manifest.json", manifest)
    return {"run_dir": run_dir, "bundle_dir": bundle_dir, "comparison": comparison}
