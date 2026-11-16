"""Create an append-only classification correction for a frozen rating overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.current_rating import canonical_json_sha256, utc_now_text  # noqa: E402


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")

    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def classification_units(overlay: dict) -> dict:
    detailed_counts = overlay.get("identity_resolution", {}).get("counts", {})

    return {
        "resolved_canonical_players": len(overlay.get("players", [])),
        "identity_evidence_rows": {
            "CURRENT_RATING_RESOLVED": int(
                detailed_counts.get("RESOLVED_DIRECT_EPISODE_TEAM_ID_AND_EXACT_SEAT_NAME", 0)
            ),
            "CURRENT_RATING_UNRESOLVED": sum(
                int(count)
                for status, count in detailed_counts.items()
                if not str(status).startswith("RESOLVED_")
            ),
        },
    }


def supersede(source_path: Path, output_root: Path) -> Path:
    source = _read_json(source_path)
    source_hash = str(source.pop("overlay_sha256"))
    if canonical_json_sha256(source) != source_hash:
        raise ValueError("source overlay hash does not verify")
    for player in source.get("players", []):
        player["current_rating_classification"] = "CURRENT_RATING_RESOLVED"
    unresolved = source.get("identity_resolution", {}).get("unresolved_or_ambiguous", [])
    for row in unresolved:
        row["current_rating_classification"] = "CURRENT_RATING_UNRESOLVED"
    created_at = utc_now_text()
    source["overlay_version"] = f"{source['overlay_version']}+classification-hotfix.2"
    source["supersedes_overlay_sha256"] = source_hash
    source["classification_correction"] = {
        "created_at_utc": created_at,
        "reason": "add exact mission-required CURRENT_RATING_UNRESOLVED classification without changing detailed status",
        "rating_snapshot_changed": False,
        "identity_mapping_changed": False,
        "replay_assets_changed": False,
    }
    corrected_hash = canonical_json_sha256(source)
    source["overlay_sha256"] = corrected_hash
    receipt = {
        "schema_version": "deepmew.current_rating_overlay_superseding_receipt.v1",
        "created_at_utc": created_at,
        "source_overlay_path": str(source_path),
        "source_overlay_sha256": source_hash,
        "corrected_overlay_sha256": corrected_hash,
        "classification_units": classification_units(source),
        "old_snapshot_preserved": True,
        "rating_snapshot_changed": False,
    }
    output_dir = output_root / created_at.replace(":", "-") / "classification-hotfix-2"
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "CURRENT_PLAYER_RATING_OVERLAY_V1.json", source)
    _write_json(output_dir / "CURRENT_PLAYER_RATING_OVERLAY_SUPERSEDING_RECEIPT_V1.json", receipt)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_overlay", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(supersede(args.source_overlay, args.output_root))


if __name__ == "__main__":
    main()
