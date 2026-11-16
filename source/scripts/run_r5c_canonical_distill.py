from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

import main
from deepmew.rl.metadata import normalized_deck_hash


DEFAULT_TEMPLATE = ROOT / ".research" / "submissions" / "20260704-v1b-canonical"
DEFAULT_CHECKPOINT = ROOT / ".research" / "r5b" / "retrain_autopsy" / "checkpoints" / "selected_ar_bc_v1b_rows_4810253.pt"
DEFAULT_BUNDLE = ROOT / ".research" / "submissions" / "rebuilt-r5c-canonical"
SCHEMA_VERSION = "deepmew.r5c_canonical_distill.rebuilt.v0"
BASE_FEATURE_MODE = "base"
SECOND_ORDER_FEATURE_MODE = "second_order"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _linear_indices(state: dict[str, Any]) -> list[int]:
    indices = []
    for key in state:
        if not key.startswith("encoder.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) == 3 and parts[1].isdigit():
            indices.append(int(parts[1]))

    return sorted(indices)


def export_weights(checkpoint: Path, weights_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    arrays: dict[str, np.ndarray] = {}
    linear_indices = _linear_indices(state)
    for out_index, source_index in enumerate(linear_indices):
        arrays[f"encoder.{out_index}.weight"] = state[f"encoder.{source_index}.weight"].detach().cpu().numpy()
        arrays[f"encoder.{out_index}.bias"] = state[f"encoder.{source_index}.bias"].detach().cpu().numpy()
    for prefix in ("option_head", "stop_head"):
        for layer in (0, 2):
            arrays[f"{prefix}.{layer}.weight"] = state[f"{prefix}.{layer}.weight"].detach().cpu().numpy()
            arrays[f"{prefix}.{layer}.bias"] = state[f"{prefix}.{layer}.bias"].detach().cpu().numpy()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_path, **arrays)

    return {
        "source_schema_version": payload.get("schema_version"),
        "input_dim": int(payload["input_dim"]),
        "hidden_dim": int(payload["hidden_dim"]),
        "depth": int(payload["depth"]),
        "embed_dim": int(payload["embed_dim"]),
        "dropout": float(payload.get("dropout", 0.0)),
        "encoder_linear_layers": len(linear_indices),
        "feature_mode": str(payload.get("feature_mode") or BASE_FEATURE_MODE),
    }


def _copy_runtime(template_dir: Path, bundle_dir: Path, deck_csv: Path) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    shutil.copy2(template_dir / "main.py", bundle_dir / "main.py")
    shutil.copy2(template_dir / "r5_distilled_policy.py", bundle_dir / "r5_distilled_policy.py")
    shutil.copy2(deck_csv, bundle_dir / "deck.csv")
    shutil.copytree(ROOT / "cg", bundle_dir / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _patch_runtime_for_second_order(runtime_path: Path) -> None:
    text = runtime_path.read_text(encoding="utf-8")
    import_anchor = "import numpy as np\nfrom cg.api import AreaType, OptionType, SelectContext, SelectType\n"
    import_patch = (
        "import numpy as np\n"
        "from cg.api import AreaType, OptionType, SelectContext, SelectType\n"
        "from second_order_reader import augment_option_features\n"
    )
    feature_anchor = (
        "def build_live_features(obs: dict[str, Any], legal_set: list[dict[str, Any]], *, policy_id: str, deck_hash: str) -> list[list[float]]:\n"
        "    state = [float(value) for value in encode_base(obs)]\n"
        "    state.extend(_bench_danger(obs))\n"
        "    state.extend(_policy_id_features(policy_id))\n"
        "    state.extend(_hash_features(deck_hash))\n"
        "    return [state + build_option_features(_as_dict(option)) for option in legal_set]\n"
    )
    feature_patch = (
        "def build_live_features(\n"
        "    obs: dict[str, Any],\n"
        "    legal_set: list[dict[str, Any]],\n"
        "    *,\n"
        "    policy_id: str,\n"
        "    deck_hash: str,\n"
        "    feature_mode: str = \"base\",\n"
        ") -> list[list[float]]:\n"
        "    state = [float(value) for value in encode_base(obs)]\n"
        "    state.extend(_bench_danger(obs))\n"
        "    state.extend(_policy_id_features(policy_id))\n"
        "    state.extend(_hash_features(deck_hash))\n"
        "    base_features = [state + build_option_features(_as_dict(option)) for option in legal_set]\n"
        "    if feature_mode == \"second_order\":\n"
        "        return augment_option_features(obs, legal_set, base_features)\n"
        "\n"
        "    return base_features\n"
    )
    call_anchor = "        features = build_live_features(obs, table, policy_id=self.policy_id, deck_hash=self.deck_hash)\n"
    call_patch = (
        "        features = build_live_features(\n"
        "            obs,\n"
        "            table,\n"
        "            policy_id=self.policy_id,\n"
        "            deck_hash=self.deck_hash,\n"
        "            feature_mode=str(self.manifest.get(\"feature_mode\") or \"base\"),\n"
        "        )\n"
    )
    if import_anchor not in text or feature_anchor not in text or call_anchor not in text:
        raise RuntimeError("Runtime template did not match expected base distill shape")
    text = text.replace(import_anchor, import_patch, 1)
    text = text.replace(feature_anchor, feature_patch, 1)
    text = text.replace(call_anchor, call_patch, 1)
    runtime_path.write_text(text, encoding="utf-8")


def _enable_second_order_runtime(bundle_dir: Path) -> None:
    shutil.copy2(ROOT / "deepmew" / "second_order" / "reader.py", bundle_dir / "second_order_reader.py")
    _patch_runtime_for_second_order(bundle_dir / "r5_distilled_policy.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _make_tar(bundle_dir: Path, tar_path: Path) -> None:
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            archive.add(path, arcname=str(path.relative_to(bundle_dir)).replace("\\", "/"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    bundle_dir = Path(args.bundle_dir)
    template_dir = Path(args.template_dir)
    deck_csv = Path(args.deck_csv)
    _copy_runtime(template_dir, bundle_dir, deck_csv)
    weights_path = bundle_dir / "r5_policy_weights.npz"
    weight_meta = export_weights(checkpoint, weights_path)
    feature_mode = str(weight_meta.get("feature_mode") or BASE_FEATURE_MODE)
    if feature_mode == SECOND_ORDER_FEATURE_MODE:
        _enable_second_order_runtime(bundle_dir)
    elif feature_mode != BASE_FEATURE_MODE:
        raise ValueError(f"Unsupported feature_mode for canonical distill: {feature_mode}")
    weights_sha = sha256_file(weights_path)
    checkpoint_sha = sha256_file(checkpoint)
    deck = [
        int(row.strip())
        for row in deck_csv.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    deck_hash = normalized_deck_hash(deck)
    weights_manifest = {
        "schema_version": "deepmew.r5_ship.rebuilt.v0",
        "created_at_utc": _now(),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": f"sha256:{checkpoint_sha}",
        "weights": str(weights_path),
        "weights_sha256": f"sha256:{weights_sha}",
        "policy_id": args.policy_id,
        "feature_deck_hash": deck_hash,
        "runtime_contract": "pure Python plus numpy; zero torch imports in submission path",
        "model_kind": "autoregressive",
        **weight_meta,
    }
    _write_json(bundle_dir / "r5_policy_weights.manifest.json", weights_manifest)
    tar_path = Path(args.tar_gz) if args.tar_gz else bundle_dir.with_suffix(".tar.gz")
    submission_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "submission_id": bundle_dir.name,
        "agent_label": args.agent_label,
        "bundle_dir": str(bundle_dir),
        "entrypoint": "main.py",
        "deck": "deck.csv",
        "policy_runtime": "r5_distilled_policy.py",
        "weights": "r5_policy_weights.npz",
        "weights_sha256": f"sha256:{weights_sha}",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": f"sha256:{checkpoint_sha}",
        "torch_runtime_allowed": False,
        "root_surfaces_mutated": False,
        "decision_rule": "canonical epsilon tie-break with option-identity key; STOP last",
    }
    _write_json(bundle_dir / "submission_manifest.json", submission_manifest)
    if not args.no_tar:
        _make_tar(bundle_dir, tar_path)
        submission_manifest["bundle_tar_gz"] = str(tar_path)
        submission_manifest["tar_gz_sha256"] = f"sha256:{sha256_file(tar_path)}"
        _write_json(bundle_dir / "submission_manifest.json", submission_manifest)
    print(json.dumps(submission_manifest, indent=2, sort_keys=True, default=str))

    return submission_manifest


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuilt R5C canonical checkpoint-to-numpy submission distill.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--deck-csv", default=str(ROOT / "deck.csv"))
    parser.add_argument("--tar-gz", default="")
    parser.add_argument("--policy-id", default="reference_abomasnow")
    parser.add_argument("--agent-label", default="rebuilt_R5C_numpy_canonical")
    parser.add_argument("--no-tar", action="store_true")
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
