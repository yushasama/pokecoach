"""Run the bounded DEC-107 CAP-030 opaque-root feasibility gate."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pefile


ROOT = Path(__file__).resolve().parents[1]
CAP001_RECEIPT = ROOT / ".research/cap001_cabt132_replay_admission_v0" / (
    "CANONICAL_REPLAY_ADMISSION_RECEIPT.json"
)
FIXTURE_RECEIPT = ROOT / ".research/q_guided_starmie" / (
    "modal_privileged_teacher_smoke_20260712/smoke_receipt.json"
)
OUTPUT = ROOT / ".research/cap030_opaque_root_feasibility_v0"
EXPECTED_NATIVE_EXPORTS = frozenset({
    "AgentStart",
    "AllAttack",
    "AllCard",
    "BattleFinish",
    "BattleStart",
    "GameInitialize",
    "GetBattleData",
    "SearchBegin",
    "SearchEnd",
    "SearchRelease",
    "SearchStep",
    "Select",
    "VisualizeData",
})


class FeasibilityHardStop(RuntimeError):
    """Raised if the bounded gate lacks exact prerequisite evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FeasibilityHardStop(f"receipt root is not an object: {path}")

    return value


def _function_args(path: Path, function_name: str) -> tuple[str, ...]:
    module = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return tuple(argument.arg for argument in node.args.args)

    raise FeasibilityHardStop(f"function is absent: {path}:{function_name}")


def _native_exports(path: Path) -> frozenset[str]:
    binary = pefile.PE(str(path), fast_load=False)
    try:
        directory = getattr(binary, "DIRECTORY_ENTRY_EXPORT", None)
        if directory is None:
            raise FeasibilityHardStop("CABT binary export directory is absent")
        names = {
            symbol.name.decode("ascii")
            for symbol in directory.symbols
            if symbol.name is not None
        }
    finally:
        binary.close()

    return frozenset(names)


def run(output: Path = OUTPUT) -> dict[str, Any]:
    cap001 = _json(CAP001_RECEIPT)
    if cap001.get("verdict") != "RELIABLE":
        raise FeasibilityHardStop("CAP-001 is not RELIABLE")
    fixture = _json(FIXTURE_RECEIPT)
    fixture_checks = fixture.get("checks")
    fixture_safety = fixture.get("safety")
    required_fixture_checks = (
        "branch_executor_safety_clean",
        "persisted_private_content_scan",
        "root_simulator_state_unchanged",
        "search_begin_input_absent",
        "siblings_begin_from_same_root",
        "student_actor_visible_only",
        "teacher_memory_released",
    )
    if (
        fixture.get("status") != "passed_end_to_end_smoke"
        or not isinstance(fixture_checks, dict)
        or not all(fixture_checks.get(check) is True for check in required_fixture_checks)
        or fixture_safety != {"crashes": 0, "illegal_actions": 0, "timeouts": 0}
    ):
        raise FeasibilityHardStop("count-valid fresh fixture receipt is not safety-clean")
    cg_dir = ROOT / "cg"
    vendored_files = tuple(sorted(
        path.name for path in cg_dir.iterdir() if path.is_file()
    ))
    expected_files = {
        "__init__.py", "api.py", "cg.dll", "game.py", "libcg.so", "sim.py", "utils.py"
    }
    if set(vendored_files) != expected_files:
        raise FeasibilityHardStop("vendored CABT surface differs from the bounded inventory")
    binary_path = cg_dir / "cg.dll"
    exports = _native_exports(binary_path)
    if exports != EXPECTED_NATIVE_EXPORTS:
        raise FeasibilityHardStop("native CABT export surface changed; re-audit required")
    battle_start_args = _function_args(cg_dir / "game.py", "battle_start")
    search_begin_args = _function_args(cg_dir / "api.py", "search_begin")
    if battle_start_args != ("deck0", "deck1"):
        raise FeasibilityHardStop("battle_start signature changed; re-audit required")
    expected_search_args = (
        "agent_observation",
        "your_deck",
        "your_prize",
        "opponent_deck",
        "opponent_prize",
        "opponent_hand",
        "opponent_active",
        "manual_coin",
    )
    if search_begin_args != expected_search_args:
        raise FeasibilityHardStop("search_begin signature changed; re-audit required")
    missing_primitives = (
        "historical_state_deserialize",
        "battle_snapshot_restore",
        "battle_start_with_seed_or_rng_state",
        "public_prefix_to_opaque_search_input",
        "explicit_hidden_world_battle_start",
    )
    receipt = {
        "schema_version": "deepmew.cap030.opaque_root_feasibility.v1",
        "capability": "CAP-030",
        "terminal_verdict": "OPAQUE_ROOT_RECONSTRUCTION_NOT_FEASIBLE",
        "bounded_gate_only": True,
        "cap001_prerequisite": {
            "status": "RELIABLE",
            "receipt_sha256": _sha256(CAP001_RECEIPT),
            "canonical_decision_count": cap001.get("counts", {}).get("decision_count"),
        },
        "count_valid_fixture": {
            "status": "PASS_FRESH_LIVE_ROOT_ONLY",
            "receipt_sha256": _sha256(FIXTURE_RECEIPT),
            "world_sample_count": fixture.get("hidden_world_sensitivity", {}).get(
                "world_sample_count"
            ),
            "search_begin_count": fixture.get("teacher_receipt", {}).get(
                "begin_search_count"
            ),
            "safety": fixture_safety,
        },
        "vendor_surface": {
            "vendored_files": list(vendored_files),
            "cg_dll_sha256": _sha256(binary_path),
            "native_exports": sorted(exports),
            "battle_start_arguments": list(battle_start_args),
            "search_begin_arguments": list(search_begin_args),
            "missing_required_primitives": list(missing_primitives),
        },
        "exact_blocker": (
            "CABT can create the opaque input only from a fresh live GetBattleData root, "
            "while the complete current surface has no callable primitive to instantiate "
            "or restore a declared historical hidden world and matched chance/RNG state."
        ),
        "failure_stage": "historical_equivalent_live_root_instantiation",
        "search_or_scoring_attempted": False,
        "long_sampling_attempted": False,
        "historical_private_state_read": False,
        "opaque_input_persisted_hashed_logged_or_returned": False,
        "safety": {"crashes": 0, "illegal_actions": 0, "timeouts": 0},
        "roadmap_effect": (
            "historical branching remains blocked at CAP-030/CAP-029; do not start "
            "CAP-004 for that path unless CABT gains a versioned root/restore API or "
            "equivalent lawful matched-prefix primitive"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "FEASIBILITY_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return receipt


def main() -> int:
    receipt = run()
    print(json.dumps({
        "terminal_verdict": receipt["terminal_verdict"],
        "exact_blocker": receipt["exact_blocker"],
        "safety": receipt["safety"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
