"""Build the CardFunctionClassifier v2 artifacts.

Classifies every unique card in ``EN_Card_Data.csv`` into function tags and writes
``all_cards_function_tags_v2.csv`` + ``card_function_classifier_v2_report.md`` (plus
a run manifest) under the output directory. Offline/training-only.

Examples
--------
    python scripts/build_card_function_classifier_v2.py
    python scripts/build_card_function_classifier_v2.py --csv data/.../EN_Card_Data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_markdown
from deepmew.card_functions.classify import classify_all
from deepmew.card_functions.io import write_cards_csv
from deepmew.card_functions.parsing import DEFAULT_CSV_PATH, load_card_inputs
from deepmew.card_functions.priority import DEFAULT_HANDOFF_DIR, load_priority_hints
from deepmew.card_functions.report import build_report
from deepmew.card_functions.schemas import ARTIFACT_VERSION, ClassifierConfig
from deepmew.card_functions.traps import evaluate_all

DEFAULT_OUT_DIR = ".research/card_function_classifier_v2"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CardFunctionClassifier v2 artifacts.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to EN_Card_Data.csv.")
    parser.add_argument("--handoff-dir", default=DEFAULT_HANDOFF_DIR, help="Handoff bundle for priority hints.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for CSV + report.")
    return parser.parse_args(argv)


def run(csv_path: str, handoff_dir: str | None, out_dir: str) -> dict[str, object]:
    config = ClassifierConfig()
    cards = load_card_inputs(csv_path)
    hints = load_priority_hints(handoff_dir)
    classified = classify_all(cards, hints, config)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_out = out / config.output_csv_name
    report_out = out / config.report_name
    write_cards_csv(csv_out, classified)
    report = build_report(
        classified, handoff_dir=handoff_dir, csv_path=csv_path, csv_output=str(csv_out)
    )
    write_markdown(report_out, report)

    trap_results = evaluate_all(classified)
    trap_pass = sum(1 for r in trap_results if r["passed"])
    confidence = {level: sum(1 for c in classified if c.confidence == level) for level in ("high", "medium", "low")}
    review = sum(1 for c in classified if c.needs_manual_review)

    run_id = make_run_id("card_functions")
    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "cards_classified": len(classified),
        "confidence": confidence,
        "needs_manual_review": review,
        "trap_cases_passed": trap_pass,
        "trap_cases_total": len(trap_results),
    }
    manifest = build_manifest(
        run_id=run_id,
        kind="card_functions",
        command=f"build_card_function_classifier_v2 --csv {csv_path}",
        inputs={"csv": csv_path, "handoff_dir": handoff_dir},
        outputs={"csv": str(csv_out), "report": str(report_out)},
        summary=summary,
    )
    write_manifest(out / "manifest.json", manifest)
    return {
        "csv": str(csv_out),
        "report": str(report_out),
        "manifest": str(out / "manifest.json"),
        **summary,
        "trap_failures": [r for r in trap_results if not r["passed"]],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.csv, args.handoff_dir, args.out_dir)
    print(f"[card_functions] classified {result['cards_classified']} cards")
    print(
        f"[card_functions] confidence high={result['confidence']['high']} "
        f"medium={result['confidence']['medium']} low={result['confidence']['low']}"
    )
    print(f"[card_functions] needs_manual_review={result['needs_manual_review']}")
    print(f"[card_functions] trap cases passed {result['trap_cases_passed']}/{result['trap_cases_total']}")
    if result["trap_failures"]:
        print(f"[card_functions] TRAP FAILURES: {result['trap_failures']}")
    print(f"[card_functions] csv={result['csv']}")
    print(f"[card_functions] report={result['report']}")
    return 1 if result["trap_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
