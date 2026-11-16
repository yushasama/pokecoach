"""R17: build per-ELO-bucket opponent deck pools from the R16 replay sections.

Scans `.research/kaggle_replays/r16-elo-001-<bucket>/` and writes one pool
artifact with per-bucket deduplicated 60-card decklists + game/win counts +
shell labels (for punisher targeting).

Usage: python scripts/build_r17_bucket_pool.py [--sections-root .research/kaggle_replays]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_r14_ladder_replays import DEFAULT_CARD_CSV, load_card_names, shell_label  # noqa: E402
from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_json, write_markdown  # noqa: E402
from deepmew.bucket_arena import STAGE_ORDER, build_bucket_pool  # noqa: E402

SCHEMA_VERSION = "deepmew.r17_bucket_pool.v1"
SECTION_PREFIX = "r16-elo-001-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections-root", default=str(ROOT / ".research" / "kaggle_replays"))
    parser.add_argument("--card-csv", default=str(DEFAULT_CARD_CSV))
    args = parser.parse_args()

    cards = load_card_names(Path(args.card_csv))
    sections_root = Path(args.sections_root)
    pools: dict[str, list[dict]] = {}
    for bucket in STAGE_ORDER:
        section_dir = sections_root / f"{SECTION_PREFIX}{bucket}"
        if not section_dir.is_dir():
            continue
        entries = build_bucket_pool(section_dir)
        pools[bucket] = [
            {**entry.to_json(), "shell": shell_label(entry.card_ids, cards)}
            for entry in entries
        ]
        print(f"[{bucket}] {len(entries)} unique decks, "
              f"{sum(e.games for e in entries)} deck-games", flush=True)

    run_dir = create_run_dir(kind="bucket_pool", out_root=ROOT / ".research" / "runs")
    payload = {"schema_version": SCHEMA_VERSION, "sections_root": str(sections_root), "pools": pools}
    write_json(run_dir.output_path("bucket_pools.json"), payload)
    lines = ["# R17 bucket deck pools", ""]
    for bucket, entries in pools.items():
        lines.append(f"## {bucket} ({len(entries)} decks)")
        lines.append("")
        lines.append("| deck_hash | shell | games | win_rate | teams |")
        lines.append("|---|---|---|---|---|")
        for entry in entries[:12]:
            lines.append(
                f"| {entry['deck_hash']} | {entry['shell']} | {entry['games']} "
                f"| {entry['win_rate']} | {len(entry['teams'])} |"
            )
        lines.append("")
    write_markdown(run_dir.output_path("report.md"), "\n".join(lines))
    manifest = build_manifest(
        run_id=run_dir.run_id, kind="bucket_pool",
        command=[sys.executable, *sys.argv],
        inputs={"sections_root": str(sections_root)},
        outputs={"pools": "bucket_pools.json", "report": "report.md"},
        summary={bucket: len(entries) for bucket, entries in pools.items()},
    )
    write_manifest(run_dir.manifest_path, manifest)
    print("artifacts:", run_dir.path)


if __name__ == "__main__":
    main()
