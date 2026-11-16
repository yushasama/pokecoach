from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.real_meta_arena.decklists import ZooDeck, expand_card_counts, validate_decklist
from deepmew.rl.metadata import normalized_deck_hash


SCHEMA_VERSION = "deepmew.r13.deck_variant_bundles.v0"
DEFAULT_TEMPLATE = ROOT / ".research" / "submissions" / "20260706-r4f-canonical"
DEFAULT_DECK_ZOO = ROOT / ".research" / "top_player_arena" / "20260706-top20-r4f-arena" / "archive_named_deck_zoo_20260706"
DEFAULT_REFERENCE_PANEL = ROOT / ".research" / "top_player_arena" / "20260706-top20-r4f-arena" / "reference_deck_panel_all_r4f_g64" / "combined" / "by_pilot.csv"
DEFAULT_OUT = ROOT / ".research" / "r13" / "lane0_deck_variant_bundles"
DEFAULT_SUBMISSIONS = ROOT / ".research" / "submissions"
STARMIE_CANDIDATES = ("18ffc1e65b611872", "2ccdace10682df96", "92ed7ee5a1969418", "86c3817de0b1db80")
COPIED_POLICY_ID = "top_player_replay"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_deck_rows(deck_zoo: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    path = deck_zoo / "deck_zoo_decklists.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["deck_hash"])] = row

    return rows


def _variant_labels(deck_zoo: Path) -> dict[str, str]:
    path = deck_zoo / "deck_zoo.csv"
    if not path.exists():
        return {}

    return {str(row["deck_hash"]): str(row.get("variant_label") or "") for row in _read_csv(path)}


def _choose_starmie(reference_panel: Path, candidates: tuple[str, ...]) -> dict[str, Any]:
    rows = [row for row in _read_csv(reference_panel) if row.get("pilot_deck_hash") in candidates]
    if not rows:
        raise RuntimeError(f"no Starmie candidates found in {reference_panel}")
    rows.sort(key=lambda row: (-float(row["winrate"]), row["pilot_deck_hash"]))

    return rows[0]


def _deck_from_row(row: dict[str, Any], variant_label: str) -> ZooDeck:
    card_ids = expand_card_counts(row.get("card_counts") or {})

    return ZooDeck(
        deck_hash=str(row["deck_hash"]),
        archetype=str(row.get("archetype") or "Unknown"),
        split=str(row.get("split") or ""),
        is_canonical=bool(row.get("is_canonical")),
        card_ids=card_ids,
        card_count=len(card_ids),
        variant_label=variant_label or str(row.get("archetype") or "Unknown"),
    )


def _write_deck_csv(path: Path, card_ids: tuple[int, ...]) -> None:
    path.write_text("\n".join(str(card_id) for card_id in card_ids) + "\n", encoding="utf-8")


def _copy_template(template_dir: Path, bundle_dir: Path) -> None:
    submissions_root = (ROOT / ".research" / "submissions").resolve()
    resolved = bundle_dir.resolve()
    if not str(resolved).startswith(str(submissions_root)):
        raise RuntimeError(f"refusing to replace bundle outside submissions root: {bundle_dir}")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(
        template_dir,
        bundle_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _make_tar(bundle_dir: Path, tar_path: Path) -> None:
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            archive.add(path, arcname=str(path.relative_to(bundle_dir)).replace("\\", "/"))


def _build_variant(
    *,
    template_dir: Path,
    submissions_dir: Path,
    out_dir: Path,
    deck: ZooDeck,
    bundle_name: str,
    agent_label: str,
    source_note: str,
) -> dict[str, Any]:
    validation = validate_decklist(deck)
    if not validation.valid:
        raise RuntimeError(f"{deck.deck_hash} failed deck validation: {validation.errors}")

    bundle_dir = submissions_dir / bundle_name
    tar_path = submissions_dir / f"{bundle_name}.tar.gz"
    _copy_template(template_dir, bundle_dir)
    _write_deck_csv(bundle_dir / "deck.csv", deck.card_ids)

    normalized_hash = normalized_deck_hash(list(deck.card_ids))
    weights_manifest_path = bundle_dir / "r5_policy_weights.manifest.json"
    weights_manifest = _read_json(weights_manifest_path)
    weights_manifest.update(
        {
            "policy_id": COPIED_POLICY_ID,
            "feature_deck_hash": deck.deck_hash,
            "deck_variant_hash": deck.deck_hash,
            "deck_variant_archetype": deck.archetype,
            "deck_variant_label": deck.variant_label,
            "deck_variant_source": source_note,
            "deck_normalized_sha256": f"sha256:{normalized_hash}",
            "weights": str(bundle_dir / "r5_policy_weights.npz"),
        }
    )
    _write_json(weights_manifest_path, weights_manifest)

    submission_manifest_path = bundle_dir / "submission_manifest.json"
    submission_manifest = _read_json(submission_manifest_path)
    submission_manifest.update(
        {
            "created_at_utc": _now(),
            "submission_id": bundle_name,
            "agent_label": agent_label,
            "bundle_dir": str(bundle_dir),
            "bundle_tar_gz": str(tar_path),
            "deck_variant_hash": deck.deck_hash,
            "deck_variant_archetype": deck.archetype,
            "deck_variant_label": deck.variant_label,
            "deck_variant_source": source_note,
            "feature_policy_id": COPIED_POLICY_ID,
            "feature_deck_hash": deck.deck_hash,
            "deck_normalized_sha256": f"sha256:{normalized_hash}",
        }
    )
    _write_json(submission_manifest_path, submission_manifest)

    _make_tar(bundle_dir, tar_path)
    submission_manifest["tar_gz_sha256"] = f"sha256:{_sha256_file(tar_path)}"
    _write_json(submission_manifest_path, submission_manifest)

    return {
        "bundle_name": bundle_name,
        "bundle_dir": str(bundle_dir),
        "tar_gz": str(tar_path),
        "tar_gz_sha256": submission_manifest["tar_gz_sha256"],
        "agent_label": agent_label,
        "deck_hash": deck.deck_hash,
        "archetype": deck.archetype,
        "variant_label": deck.variant_label,
        "policy_id": COPIED_POLICY_ID,
        "feature_deck_hash": deck.deck_hash,
        "deck_normalized_sha256": f"sha256:{normalized_hash}",
        "card_count": deck.card_count,
        "engine_accepted": validation.engine_accepted,
        "validation_valid": validation.valid,
        "validation_errors": "; ".join(validation.errors),
        "validation_warnings": "; ".join(validation.warnings),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R13 Lane 0 Deck Variant Bundle Build",
        "",
        f"- schema_version: `{payload['schema_version']}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- template: `{payload['inputs']['template_dir']}`",
        f"- deck_zoo: `{payload['inputs']['deck_zoo_dir']}`",
        "",
        "## Starmie Selection",
        "",
        f"Selected `{payload['starmie_selection']['deck_hash']}` because it was the highest winrate candidate in the archived reference panel.",
        "",
        "| candidate | wins-losses-draws | winrate | rank note |",
        "|---|---:|---:|---|",
    ]
    for row in payload["starmie_selection"]["candidate_rows"]:
        lines.append(
            f"| `{row['pilot_deck_hash']}` | {row['wins']}-{row['losses']}-{row['draws']} | "
            f"{row['winrate']} | {row['pilot_label']} |"
        )
    lines.extend(
        [
            "",
            "## Built Bundles",
            "",
            "| bundle | deck | archetype | feature policy | tar sha256 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["bundles"]:
        lines.append(
            f"| `{row['bundle_name']}` | `{row['deck_hash']}` | {row['archetype']} | "
            f"`{row['policy_id']}` / `{row['feature_deck_hash']}` | `{row['tar_gz_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Only bundle-local `deck.csv` and manifest files were changed. Root `deck.csv`, root `main.py`, `cg/`, `deepmew/submissions/`, and default `v2.4` were not modified.",
            "",
            "## Next",
            "",
            "Run `scripts/validate_r13_submission_bundle.py` for each bundle before upload.",
        ]
    )

    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    deck_zoo = Path(args.deck_zoo)
    labels = _variant_labels(deck_zoo)
    deck_rows = _load_deck_rows(deck_zoo)
    starmie_rows = [row for row in _read_csv(Path(args.reference_panel)) if row.get("pilot_deck_hash") in STARMIE_CANDIDATES]
    starmie_rows.sort(key=lambda row: (-float(row["winrate"]), row["pilot_deck_hash"]))
    starmie = _choose_starmie(Path(args.reference_panel), STARMIE_CANDIDATES)
    selected = {
        "bd27": _deck_from_row(deck_rows[args.bd27_hash], labels.get(args.bd27_hash, "Mega Lucario ex")),
        "starmie": _deck_from_row(deck_rows[starmie["pilot_deck_hash"]], labels.get(starmie["pilot_deck_hash"], "Mega Starmie ex / Staryu")),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = [
        _build_variant(
            template_dir=Path(args.template_dir),
            submissions_dir=Path(args.submissions_dir),
            out_dir=out_dir,
            deck=selected["bd27"],
            bundle_name="20260706-r4f-bd27",
            agent_label="selected_ar_bc_r4f_reweighted_bd27",
            source_note="R13 Lane 0 bd27 Mega Lucario/Fighting variant",
        ),
        _build_variant(
            template_dir=Path(args.template_dir),
            submissions_dir=Path(args.submissions_dir),
            out_dir=out_dir,
            deck=selected["starmie"],
            bundle_name=f"20260706-r4f-starmie-{starmie['pilot_deck_hash'][:8]}",
            agent_label=f"selected_ar_bc_r4f_reweighted_starmie_{starmie['pilot_deck_hash'][:8]}",
            source_note="R13 Lane 0 top archived reference-panel Starmie/Staryu variant",
        ),
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "inputs": {
            "template_dir": str(Path(args.template_dir)),
            "deck_zoo_dir": str(deck_zoo),
            "reference_panel": str(Path(args.reference_panel)),
        },
        "starmie_selection": {
            "deck_hash": starmie["pilot_deck_hash"],
            "pilot_label": starmie["pilot_label"],
            "winrate": starmie["winrate"],
            "wins": starmie["wins"],
            "losses": starmie["losses"],
            "draws": starmie["draws"],
            "candidate_rows": starmie_rows,
        },
        "bundles": bundles,
    }
    _write_json(out_dir / "deck_variant_bundle_manifest.json", payload)
    _write_csv(out_dir / "deck_variant_bundles.csv", bundles)
    (out_dir / "deck_variant_bundle_report.md").write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build R13 R4-F deck-variant submission bundles.")
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--deck-zoo", default=str(DEFAULT_DECK_ZOO))
    parser.add_argument("--reference-panel", default=str(DEFAULT_REFERENCE_PANEL))
    parser.add_argument("--submissions-dir", default=str(DEFAULT_SUBMISSIONS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--bd27-hash", default="bd27c5c46ec86161")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
