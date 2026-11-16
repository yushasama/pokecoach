from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


SCHEMA_VERSION = "deepmew.r13.submission_bundle_validation.v0"


def _load_real_meta_runner() -> tuple[type[Any], Any]:
    module_name = "_r13_real_meta_runner"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "deepmew" / "real_meta_arena" / "runner.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load deepmew.real_meta_arena.runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module.SpecAgent, module.run_real_meta_matchup


SpecAgent, run_real_meta_matchup = _load_real_meta_runner()


class BundleAgent:
    def __init__(self, module: Any, label: str):
        self.module = module
        self.label = label

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        _ = rng

        return [int(index) for index in self.module.agent(obs)]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_deck(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tar_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        return sorted(member.name for member in archive.getmembers() if member.isfile())


def _import_bundle_main(bundle_dir: Path) -> tuple[Any, bool]:
    for name in ("main", "r5_distilled_policy", "second_order_reader"):
        sys.modules.pop(name, None)
    before_torch = "torch" in sys.modules
    old_path = list(sys.path)
    sys.path.insert(0, str(bundle_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"r13_bundle_main_{bundle_dir.name.replace('-', '_')}", bundle_dir / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {bundle_dir / 'main.py'}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch_loaded_after_import = (not before_torch) and ("torch" in sys.modules)

        return module, torch_loaded_after_import
    finally:
        sys.path[:] = old_path


def _normal_obs(option_types: list[int]) -> dict[str, Any]:
    options = [{"type": option_type} for option_type in option_types]

    return {
        "current": {
            "players": [
                {"active": [], "bench": [], "prize": [], "hand": [], "deckCount": 60},
                {"active": [], "bench": [], "prize": [], "hand": [], "deckCount": 60},
            ],
            "yourIndex": 0,
            "turn": 1,
            "firstPlayer": 0,
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
    }


def _valid_selection(selection: Any, obs: dict[str, Any]) -> bool:
    if not isinstance(selection, list):
        return False
    options = obs["select"]["option"]
    min_count = int(obs["select"].get("minCount") or 0)
    max_count = int(obs["select"].get("maxCount") or len(options))
    if len(selection) < min_count or len(selection) > max_count:
        return False

    return all(isinstance(index, int) and 0 <= index < len(options) for index in selection) and len(selection) == len(set(selection))


def _cwd_asset_check(module: Any) -> bool:
    original = Path.cwd()
    tmp_root = Path("C:/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        os.chdir(tmp)
        try:
            deck = module.read_deck_csv()
        finally:
            os.chdir(original)

    return len(deck) == 60 and all(isinstance(card_id, int) for card_id in deck)


def _runtime_shape(bundle_dir: Path) -> tuple[list[int], bool]:
    weights = np.load(bundle_dir / "r5_policy_weights.npz", allow_pickle=False)
    shape = list(weights["encoder.0.weight"].shape)

    return shape, shape == [512, 304]


def _static_clock_guard(bundle_dir: Path) -> dict[str, Any]:
    text = (bundle_dir / "main.py").read_text(encoding="utf-8")
    value = None
    for line in text.splitlines():
        if line.startswith("MAX_DECISION_SECONDS"):
            try:
                value = float(line.split("=", 1)[1].strip())
            except ValueError:
                value = None
    has_guard = all(snippet in text for snippet in ("MAX_DECISION_SECONDS", "time.perf_counter", "safe_fallback", "valid_selection"))

    return {
        "max_decision_seconds": value,
        "static_guard_present": has_guard,
        "remaining_clock_note": "no-search bundle: fixed hard decision cap; remaining-clock search budgeting is not exercised",
    }


def _cabt_self_play(module: Any, steps: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from kaggle_environments import make
    except Exception as exc:
        return {"status": "not_run", "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def wrapped(obs: dict[str, Any], _config: Any = None) -> list[int]:
        return module.agent(obs)

    try:
        env = make("cabt", debug=True)
        env.reset(2)
        steps_completed = 0
        for _ in range(max(1, steps)):
            actions = []
            for state in env.state:
                status = getattr(state, "status", None)
                observation = getattr(state, "observation", {})
                if status == "ACTIVE":
                    actions.append(wrapped(observation))
                else:
                    actions.append([])
            env.step(actions)
            steps_completed += 1
            statuses_after_step = [getattr(item, "status", None) for item in env.state or []]
            if env.done or any(status in {"ERROR", "INVALID", "TIMEOUT"} for status in statuses_after_step):
                break
        state = env.state or []
        statuses = [getattr(item, "status", None) for item in state]
        rewards = [getattr(item, "reward", None) for item in state]
        errors = [getattr(item, "error", None) for item in state]
        ok = bool(state) and steps_completed > 0 and not any(
            status in {"ERROR", "INVALID", "TIMEOUT"} for status in statuses
        ) and not any(errors)

        return {
            "status": "pass" if ok else "fail",
            "ok": ok,
            "statuses": statuses,
            "rewards": rewards,
            "errors": errors,
            "steps_completed": steps_completed,
            "episode_done": bool(env.done),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "requested_steps": steps,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = Path(args.bundle_dir)
    tar_path = Path(args.tar_gz) if args.tar_gz else Path(f"{args.bundle_dir}.tar.gz")
    out_dir = Path(args.out_dir)
    module, torch_loaded = _import_bundle_main(bundle_dir)
    deck = _read_deck(bundle_dir / "deck.csv")
    tar_members = _tar_members(tar_path)
    required = {"main.py", "deck.csv", "r5_distilled_policy.py", "r5_policy_weights.npz", "r5_policy_weights.manifest.json", "second_order_reader.py", "submission_manifest.json"}
    normal_obs = _normal_obs([0, 1])
    unknown_obs = _normal_obs([999999, 999998])
    normal_selection = module.agent(normal_obs)
    unknown_selection = module.agent(unknown_obs)
    shape, shape_ok = _runtime_shape(bundle_dir)
    clock = _static_clock_guard(bundle_dir)
    cabt = _cabt_self_play(module, int(args.cabt_steps))
    arena = run_real_meta_matchup(
        BundleAgent(module, args.label),
        SpecAgent.from_str("first-legal"),
        deck,
        deck,
        games=int(args.arena_games),
        bucket="r13_bundle_self_check",
        opp_deck_hash=args.deck_hash,
        opp_archetype=args.archetype,
        seed=int(args.seed),
        max_steps=int(args.max_steps),
    )
    safety_clean = arena.crashes == 0 and arena.timeouts == 0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "bundle_dir": str(bundle_dir),
        "tar_gz": str(tar_path),
        "label": args.label,
        "deck_hash": args.deck_hash,
        "archetype": args.archetype,
        "checks": {
            "tar_readable": bool(tar_members),
            "tar_member_count": len(tar_members),
            "required_layout": sorted(required).copy(),
            "required_layout_ok": required.issubset(set(tar_members)),
            "research_member_count": sum(1 for member in tar_members if ".research" in member.replace("\\", "/").split("/")),
            "deck_len": len(deck),
            "deck_all_int": all(isinstance(card_id, int) for card_id in deck),
            "torch_loaded_after_import": torch_loaded,
            "runtime_shape": shape,
            "runtime_shape_ok": shape_ok,
            "normal_selection": normal_selection,
            "normal_selection_valid": _valid_selection(normal_selection, normal_obs),
            "bug_001_unknown_enum_selection": unknown_selection,
            "bug_001_unknown_enum_safe": _valid_selection(unknown_selection, unknown_obs),
            "bug_002_cwd_asset_safe": _cwd_asset_check(module),
            "bug_003_clock_guard": clock,
            "cabt_self_play": cabt,
        },
        "arena": asdict(arena),
        "verdict": {
            "safety_clean": safety_clean,
            "upload_ready": (
                bool(tar_members)
                and required.issubset(set(tar_members))
                and len(deck) == 60
                and all(isinstance(card_id, int) for card_id in deck)
                and not torch_loaded
                and shape_ok
                and _valid_selection(normal_selection, normal_obs)
                and _valid_selection(unknown_selection, unknown_obs)
                and _cwd_asset_check(module)
                and bool(clock["static_guard_present"])
                and bool(cabt.get("ok"))
                and safety_clean
            ),
        },
    }
    _write_json(out_dir / f"{bundle_dir.name}_validation.json", payload)
    report = [
        f"# R13 Bundle Validation - {bundle_dir.name}",
        "",
        f"- upload_ready: `{payload['verdict']['upload_ready']}`",
        f"- deck_hash: `{args.deck_hash}`",
        f"- archetype: `{args.archetype}`",
        f"- tar_member_count: `{payload['checks']['tar_member_count']}`",
        f"- deck_len: `{payload['checks']['deck_len']}`",
        f"- torch_loaded_after_import: `{payload['checks']['torch_loaded_after_import']}`",
        f"- runtime_shape: `{payload['checks']['runtime_shape']}`",
        f"- BUG-001 unknown enum safe: `{payload['checks']['bug_001_unknown_enum_safe']}`",
        f"- BUG-002 cwd asset safe: `{payload['checks']['bug_002_cwd_asset_safe']}`",
        f"- BUG-003 clock guard: `{payload['checks']['bug_003_clock_guard']['static_guard_present']}`",
        f"- cabt_self_play: `{payload['checks']['cabt_self_play'].get('status')}`",
        f"- arena: `{arena.our_wins}-{arena.opp_wins}-{arena.draws}`, crashes `{arena.crashes}`, timeouts `{arena.timeouts}`",
        "",
    ]
    (out_dir / f"{bundle_dir.name}_validation.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    return payload


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an R13 submission bundle and run deck-specific smoke arena.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--tar-gz", default="")
    parser.add_argument("--out-dir", default=str(ROOT / ".research" / "r13" / "lane0_deck_variant_bundles"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--deck-hash", required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--arena-games", type=int, default=32)
    parser.add_argument("--cabt-steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args(argv)
    payload = run(args)

    return 0 if payload["verdict"]["upload_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main_cli())
