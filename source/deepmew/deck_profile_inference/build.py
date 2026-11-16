"""Build DeckProfileInference v0 artifacts.

Loads CardFunctionClassifier v2 tags + mined 60-card decklists, infers a
count-weighted deck profile per deck, and writes CSV + JSONL + report (+ manifest).
Offline/training-only: no gameplay code, no engine import, no submission change.

Examples
--------
    python scripts/build_deck_profile_inference_v0.py
    python scripts/build_deck_profile_inference_v0.py --decklists path/to/deck_zoo_decklists.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import make_run_id
from deepmew.artifacts.writers import write_markdown
from deepmew.deck_profile_inference.card_functions import DEFAULT_V2_CSV, load_v2_cards
from deepmew.deck_profile_inference.decklists import find_latest_decklists, load_decklists
from deepmew.deck_profile_inference.inference import infer_all
from deepmew.deck_profile_inference.io import write_profiles_csv, write_profiles_jsonl
from deepmew.deck_profile_inference.report import build_report
from deepmew.deck_profile_inference.schemas import ARTIFACT_VERSION, ProfileConfig

DEFAULT_OUT_DIR = ".research/deck_profile_inference_v0"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeckProfileInference v0 artifacts.")
    parser.add_argument("--v2-csv", default=DEFAULT_V2_CSV, help="CardFunctionClassifier v2 CSV.")
    parser.add_argument("--decklists", default=None, help="deck_zoo_decklists.jsonl (default: latest run).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args(argv)


def run(v2_csv: str, decklists_path: str | None, out_dir: str) -> dict[str, object]:
    config = ProfileConfig()
    v2map = load_v2_cards(v2_csv)
    resolved_decklists = decklists_path or find_latest_decklists()
    decks = load_decklists(resolved_decklists)
    profiles = infer_all(decks, v2map, config)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_out = out / config.output_csv_name
    jsonl_out = out / config.output_jsonl_name
    report_out = out / config.report_name

    write_profiles_csv(csv_out, profiles)
    write_profiles_jsonl(jsonl_out, profiles)
    report = build_report(
        profiles,
        v2map=v2map,
        decklists_path=str(resolved_decklists),
        csv_output=str(csv_out),
        jsonl_output=str(jsonl_out),
    )
    write_markdown(report_out, report)

    confidence = {level: sum(1 for p in profiles if p.profile_confidence == level) for level in ("high", "medium", "low")}
    review = sum(1 for p in profiles if p.manual_review_reasons)
    from collections import Counter

    reason_freq = Counter(t for p in profiles for t in p.reason_tags)

    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "profiles": len(profiles),
        "confidence": confidence,
        "needs_review": review,
        "top_reason_tags": reason_freq.most_common(8),
    }
    manifest = build_manifest(
        run_id=make_run_id("deck_profile_inference"),
        kind="deck_profile_inference",
        command=f"build_deck_profile_inference_v0 --v2-csv {v2_csv}",
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
    print(f"[deck_profile_inference] generated {result['profiles']} profiles")
    print(
        f"[deck_profile_inference] confidence high={result['confidence']['high']} "
        f"medium={result['confidence']['medium']} low={result['confidence']['low']}"
    )
    print(f"[deck_profile_inference] needs_review={result['needs_review']}")
    print(f"[deck_profile_inference] top reason tags: {result['top_reason_tags']}")
    print(f"[deck_profile_inference] csv={result['csv']}")
    print(f"[deck_profile_inference] jsonl={result['jsonl']}")
    print(f"[deck_profile_inference] report={result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
