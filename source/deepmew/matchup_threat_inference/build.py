"""Build MatchupThreatInference v0 artifacts.

Loads CardFunctionClassifier v2 tags + DeckProfileInference v0 decklists, derives a
per-deck matchup-threat surface (presents/exposes/coverage) and a bounded set of
subject-vs-opponent pairings, and writes CSV + JSONL + pairings JSONL + report +
manifest. Offline/training-only: no gameplay code, no engine import, no submission
change, and no matchup win rates.

Examples
--------
    python scripts/build_matchup_threat_inference_v0.py
    python scripts/build_matchup_threat_inference_v0.py --decklists path/to/deck_zoo_decklists.jsonl
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
from deepmew.matchup_threat_inference.inference import build_all_profiles, build_pairings
from deepmew.matchup_threat_inference.io import (
    write_pairings_jsonl,
    write_profiles_csv,
    write_profiles_jsonl,
)
from deepmew.matchup_threat_inference.report import build_report
from deepmew.matchup_threat_inference.schemas import ARTIFACT_VERSION, MatchupConfig

DEFAULT_OUT_DIR = ".research/matchup_threat_inference_v0"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MatchupThreatInference v0 artifacts.")
    parser.add_argument("--v2-csv", default=DEFAULT_V2_CSV, help="CardFunctionClassifier v2 CSV.")
    parser.add_argument("--decklists", default=None, help="deck_zoo_decklists.jsonl (default: latest run).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args(argv)


def run(v2_csv: str, decklists_path: str | None, out_dir: str) -> dict[str, object]:
    config = MatchupConfig()
    v2map = load_v2_cards(v2_csv)
    resolved_decklists = decklists_path or find_latest_decklists()
    decks = load_decklists(resolved_decklists)
    profiles = build_all_profiles(decks, v2map, config)
    pairings = build_pairings(profiles, config)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_out = out / config.output_csv_name
    jsonl_out = out / config.output_jsonl_name
    pairings_out = out / config.pairings_jsonl_name
    report_out = out / config.report_name

    write_profiles_csv(csv_out, profiles)
    write_profiles_jsonl(jsonl_out, profiles)
    write_pairings_jsonl(pairings_out, pairings)
    report = build_report(
        profiles,
        pairings,
        decklists_path=str(resolved_decklists),
        v2_csv=v2_csv,
        csv_output=str(csv_out),
        jsonl_output=str(jsonl_out),
        pairings_output=str(pairings_out),
    )
    write_markdown(report_out, report)

    confidence = {level: sum(1 for p in profiles if p.threat_confidence == level) for level in ("high", "medium", "low")}
    review = sum(1 for p in profiles if p.manual_review_reasons)
    gap_freq = Counter(g for pr in pairings for g in pr.coverage_gaps)

    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "profiles": len(profiles),
        "pairings": len(pairings),
        "threat_confidence": confidence,
        "needs_review": review,
        "top_coverage_gaps": gap_freq.most_common(8),
        "claims_win_rate": False,
    }
    manifest = build_manifest(
        run_id=make_run_id("matchup_threat_inference"),
        kind="matchup_threat_inference",
        command=f"build_matchup_threat_inference_v0 --v2-csv {v2_csv}",
        inputs={"v2_csv": v2_csv, "decklists": str(resolved_decklists)},
        outputs={
            "csv": str(csv_out),
            "jsonl": str(jsonl_out),
            "pairings": str(pairings_out),
            "report": str(report_out),
        },
        summary=summary,
    )
    write_manifest(out / "manifest.json", manifest)
    return {
        "csv": str(csv_out),
        "jsonl": str(jsonl_out),
        "pairings": str(pairings_out),
        "report": str(report_out),
        "manifest": str(out / "manifest.json"),
        **summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.v2_csv, args.decklists, args.out_dir)
    print(f"[matchup_threat_inference] generated {result['profiles']} deck threat surfaces")
    print(f"[matchup_threat_inference] generated {result['pairings']} matchup pairings")
    conf = result["threat_confidence"]
    print(
        f"[matchup_threat_inference] threat confidence high={conf['high']} "
        f"medium={conf['medium']} low={conf['low']}"
    )
    print(f"[matchup_threat_inference] needs_review={result['needs_review']}")
    print(f"[matchup_threat_inference] top coverage gaps: {result['top_coverage_gaps']}")
    print(f"[matchup_threat_inference] csv={result['csv']}")
    print(f"[matchup_threat_inference] jsonl={result['jsonl']}")
    print(f"[matchup_threat_inference] pairings={result['pairings']}")
    print(f"[matchup_threat_inference] report={result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
