"""Build StrategicActionScorer v0 artifacts.

Loads CardFunctionClassifier v2 tags + DeckProfileInference v0 decklists, rebuilds the
MatchupThreatInference v0 surfaces in-process, materialises candidate actions from each
deck's own cards, scores them against meta-representative opponents, and writes CSV +
JSONL + report + manifest. Offline/training-only: no gameplay code, no engine import,
no submission change, no trained labels, and no win rates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_markdown
from deepmew.deck_profile_inference.card_functions import DEFAULT_V2_CSV, load_v2_cards
from deepmew.deck_profile_inference.decklists import find_latest_decklists, load_decklists
from deepmew.strategic_action_scoring.dataset import build_score_rows, select_representative_opponents
from deepmew.matchup_threat_inference.inference import build_all_profiles
from deepmew.matchup_threat_inference.schemas import MatchupConfig
from deepmew.strategic_action_scoring.io import write_scores_csv, write_scores_jsonl
from deepmew.strategic_action_scoring.report import build_report
from deepmew.strategic_action_scoring.schemas import ARTIFACT_VERSION, ScorerConfig

DEFAULT_OUT_DIR = ".research/strategic_action_scoring_v0"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build StrategicActionScorer v0 artifacts.")
    parser.add_argument("--v2-csv", default=DEFAULT_V2_CSV, help="CardFunctionClassifier v2 CSV.")
    parser.add_argument("--decklists", default=None, help="deck_zoo_decklists.jsonl (default: latest run).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args(argv)


def run(v2_csv: str, decklists_path: str | None, out_dir: str) -> dict[str, object]:
    config = ScorerConfig()
    v2map = load_v2_cards(v2_csv)
    resolved_decklists = decklists_path or find_latest_decklists()
    decks = load_decklists(resolved_decklists)

    rows = build_score_rows(decks, v2map, config)
    threat_profiles = build_all_profiles(decks, v2map, MatchupConfig())
    opponents = select_representative_opponents(threat_profiles, config.max_opponents)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_out = out / config.output_csv_name
    jsonl_out = out / config.output_jsonl_name
    report_out = out / config.report_name

    write_scores_csv(csv_out, rows)
    write_scores_jsonl(jsonl_out, rows)
    report = build_report(
        rows,
        decklists_path=str(resolved_decklists),
        v2_csv=v2_csv,
        matchup_artifact="deepmew.matchup_threat_inference.v0 (in-process)",
        csv_output=str(csv_out),
        jsonl_output=str(jsonl_out),
        n_subjects=len(threat_profiles),
        n_opponents=len(opponents),
    )
    write_markdown(report_out, report)

    tag_freq: Counter = Counter(t for r in rows for t in r["reason_tags"])
    conf_freq = {level: sum(1 for r in rows if r["confidence"] == level) for level in ("high", "medium", "low")}

    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "rows": len(rows),
        "subjects": len(threat_profiles),
        "opponents": len(opponents),
        "confidence": conf_freq,
        "top_reason_tags": tag_freq.most_common(10),
        "claims_win_rate": False,
        "trains_labels": False,
    }
    manifest = build_manifest(
        run_id=make_run_id("strategic_action_scoring"),
        kind="strategic_action_scoring",
        command=f"build_strategic_action_scoring_v0 --v2-csv {v2_csv}",
        inputs={"v2_csv": v2_csv, "decklists": str(resolved_decklists)},
        outputs={"csv": str(csv_out), "jsonl": str(jsonl_out), "report": str(report_out)},
        summary=summary,
    )
    write_manifest(out / "manifest.json", manifest)
    return {
        "csv": str(csv_out),
        "jsonl": str(jsonl_out),
        "report": str(report_out),
        "manifest": str(out / "manifest.json"),
        **summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.v2_csv, args.decklists, args.out_dir)
    print(f"[strategic_action_scoring] scored {result['rows']} candidate-vs-opponent rows")
    print(f"[strategic_action_scoring] subjects={result['subjects']} opponents={result['opponents']}")
    print(f"[strategic_action_scoring] top reason tags: {result['top_reason_tags']}")
    print(f"[strategic_action_scoring] csv={result['csv']}")
    print(f"[strategic_action_scoring] jsonl={result['jsonl']}")
    print(f"[strategic_action_scoring] report={result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
