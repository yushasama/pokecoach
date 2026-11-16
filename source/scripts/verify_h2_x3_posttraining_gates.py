"""Verify final H2/X3 checkpoint, leakage, and protected-runtime gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_PATHS = ("deck.csv", "main.py", "cg", "deepmew/submissions")
EXPECTED_ROOT_HASHES = {
    "deck.csv": "42068a1803902756badcfd418f6f348b7901365a281d78af0692cbf2589f0799",
    "main.py": "2fd5dd6f9ae5914a65b21adc22bdfcd56c408902ac755b560ee769b3feaadec0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _iter_tensors(value: Any, path: str = "root") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{path}[{index}]")


def checkpoint_finiteness(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors = list(_iter_tensors(payload))
    nonfinite = [name for name, tensor in tensors if not bool(torch.isfinite(tensor).all())]

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "byte_size": path.stat().st_size,
        "tensor_count": len(tensors),
        "nonfinite_tensor_paths": nonfinite,
        "finite": bool(tensors) and not nonfinite,
    }


def _git_diff(*extra: str) -> str:
    process = subprocess.run(
        ("git", "diff", *extra, "--", *PROTECTED_PATHS),
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )

    return process.stdout


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")

    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    h2 = checkpoint_finiteness(Path(args.h2_checkpoint).resolve())
    x3 = checkpoint_finiteness(Path(args.x3_checkpoint).resolve())
    leakage_path = Path(args.leakage_receipt).resolve()
    leakage = _read_json(leakage_path)
    h2_training = _read_json(Path(args.h2_training_receipt).resolve())
    x3_training = _read_json(Path(args.x3_training_receipt).resolve())
    root_hashes = {name: _sha256(ROOT / name) for name in EXPECTED_ROOT_HASHES}
    worktree_diff = _git_diff()
    index_diff = _git_diff("--cached")
    training_private_state_clean = all(
        receipt.get("summary", {}).get("protected_surfaces_mutated") is False
        and receipt.get("summary", {}).get("safety") == {
            "crashes": 0,
            "illegal_actions": 0,
            "timeouts": 0,
        }
        for receipt in (h2_training, x3_training)
    )
    leakage_clean = leakage.get("status") == "PASS" and leakage.get("marker_diffs") == 0
    runtime_clean = (
        root_hashes == EXPECTED_ROOT_HASHES
        and not worktree_diff
        and not index_diff
        and training_private_state_clean
    )
    status = "PASS" if h2["finite"] and x3["finite"] and leakage_clean and runtime_clean else "FAIL"

    return {
        "schema_version": "deepmew.h2_x3_posttraining_gate_receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "leakage": not leakage_clean,
        "runtime_contamination": not runtime_clean,
        "h2_checkpoint_finite": h2["finite"],
        "x3_checkpoint_finite": x3["finite"],
        "checkpoints": {"h2": h2, "x3": x3},
        "feature_leakage_audit": {
            "path": str(leakage_path),
            "status": leakage.get("status"),
            "rows_checked": leakage.get("rows_checked"),
            "marker_diffs": leakage.get("marker_diffs"),
        },
        "protected_runtime": {
            "paths": list(PROTECTED_PATHS),
            "expected_root_hashes": EXPECTED_ROOT_HASHES,
            "observed_root_hashes": root_hashes,
            "worktree_diff_empty": not worktree_diff,
            "index_diff_empty": not index_diff,
            "training_safety_and_protected_surface_receipts_clean": training_private_state_clean,
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-checkpoint", required=True)
    parser.add_argument("--x3-checkpoint", required=True)
    parser.add_argument("--h2-training-receipt", required=True)
    parser.add_argument("--x3-training-receipt", required=True)
    parser.add_argument("--leakage-receipt", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = verify(args)
    _atomic_json(Path(args.output).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
