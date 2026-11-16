"""R19: build band-stratified replay-pilot tables from the R16 bucket sections.

Produces `band_replay_tables.json` -- one `ReplayPolicyTable` per ELO band plus a
merged global fallback -- so the curriculum arena's `replay` pilot tier plays each
band's opponents the way real players in that band did (faithful, same provenance
as the opponent decks). Archetype/shell labels use the SAME labeler the deck pool
used, so table keys align with the curriculum opponents' shells.

Usage:
  python scripts/build_r19_replay_pilots.py [--sections-root .research/kaggle_replays]
Then feed the artifact to the curriculum:
  python scripts/run_curriculum_arena.py --pools <pools.json> --candidate spec:greedy \
      --replay-tables .research/runs/replay_pilots/<date>/<run>/band_replay_tables.json
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
from deepmew.curriculum.replay_pilots import build_band_replay_tables  # noqa: E402

SCHEMA_VERSION = "deepmew.r19_replay_pilots.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections-root", default=str(ROOT / ".research" / "kaggle_replays"))
    parser.add_argument("--card-csv", default=str(DEFAULT_CARD_CSV))
    parser.add_argument("--section-prefix", default="r16-elo-001-")
    args = parser.parse_args()

    cards = load_card_names(Path(args.card_csv))

    def labeler(card_ids: list[int]) -> str:
        return shell_label(card_ids, cards)

    tables = build_band_replay_tables(
        args.sections_root, shell_labeler=labeler, section_prefix=args.section_prefix
    )
    for band, n in sorted(tables.band_rows.items()):
        arch_keys = len(tables.bands[band].ctx_arch)
        print(f"[{band}] {n} decisions, {arch_keys} (context,archetype) cells", flush=True)
    print(f"[global] {tables.global_table.train_examples} decisions, "
          f"{len(tables.global_table.ctx_arch)} cells", flush=True)

    run_dir = create_run_dir(kind="replay_pilots", out_root=ROOT / ".research" / "runs")
    payload = {"schema_version": SCHEMA_VERSION, "sections_root": str(args.sections_root), **tables.to_json()}
    out = write_json(run_dir.output_path("band_replay_tables.json"), payload)
    lines = ["# R19 band-stratified replay pilots", "",
             "| band | decisions | (context,archetype) cells |", "|---|---|---|"]
    for band, n in sorted(tables.band_rows.items()):
        lines.append(f"| {band} | {n} | {len(tables.bands[band].ctx_arch)} |")
    lines.append(f"| global | {tables.global_table.train_examples} | {len(tables.global_table.ctx_arch)} |")
    write_markdown(run_dir.output_path("report.md"), "\n".join(lines) + "\n")
    manifest = build_manifest(
        run_id=run_dir.run_id, kind="replay_pilots",
        command=[sys.executable, *sys.argv],
        inputs={"sections_root": str(args.sections_root)},
        outputs={"band_replay_tables": "band_replay_tables.json", "report": "report.md"},
        summary={"band_rows": tables.band_rows, "global_rows": tables.global_table.train_examples},
    )
    write_manifest(run_dir.manifest_path, manifest)
    print("artifacts:", run_dir.path)
    print("band_replay_tables:", out)


if __name__ == "__main__":
    main()
