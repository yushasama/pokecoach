"""Create and execute ordered prospective decision-trace tranches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from deepmew.artifacts.run_dir import safe_slug  # noqa: E402
from deepmew.artifacts.writers import write_json  # noqa: E402
from deepmew.deck import read_deck  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent  # noqa: E402
from deepmew.replay.information_boundary import InformationBoundary, InformationMode, require_oracle_output_path  # noqa: E402
from deepmew.replay.prospective_trace import (  # noqa: E402
    ActorMetadata,
    PUBLIC_PROJECTION_VERSION,
    PROGRESS_SCHEMA_VERSION,
    ProspectiveGameRecorder,
    assert_public_trace_row_safe,
    atomic_write_gzip_jsonl,
    begin_tranche,
    complete_tranche,
    create_plan,
    fail_tranche_safety,
    load_plan,
    scan_durable_shards,
    tranche_by_id,
    unavailable_rating_snapshot,
    utc_now,
)
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
import run_r17_bucket_arena as arena  # noqa: E402


MANIFEST_SCHEMA_VERSION = "deepmew.prospective_trace.manifest.v1"
POLICY_SOURCE_RECEIPT_SCHEMA_VERSION = "deepmew.named_policy_source_receipt.v1"

NAMED_POLICY_SOURCE_FILES: tuple[str, ...] = (
    "main.py",
    "deepmew/real_meta_arena/runner.py",
)
NAMED_POLICY_SOURCE_DIRECTORIES: tuple[str, ...] = (
    "deepmew/agents",
    "deepmew/arena",
    "deepmew/card_semantics",
    "deepmew/deck",
    "deepmew/deck_profiles",
    "deepmew/options",
    "deepmew/policy",
    "deepmew/policy_rules",
    "deepmew/state_machine",
    "deepmew/turn_plan",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _named_policy_source_receipt() -> dict[str, Any]:
    paths = [ROOT / relative for relative in NAMED_POLICY_SOURCE_FILES]
    for relative in NAMED_POLICY_SOURCE_DIRECTORIES:
        paths.extend(sorted((ROOT / relative).rglob("*.py")))
    unique_paths = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    missing = [path for path in unique_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"named-policy source receipt is missing: {missing[0]}")
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in unique_paths
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return {
        "schema_version": POLICY_SOURCE_RECEIPT_SCHEMA_VERSION,
        "scope": "SpecAgent.from_str/select_action imported DeepMew source closure",
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _optional_identity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()

    return normalized or None


def _policy_snapshot(
    spec: str,
    *,
    league_member_id: str | None = None,
    feature_policy_id: str | None = None,
) -> dict[str, Any]:
    identity = {
        "league_member_id": _optional_identity(league_member_id),
        "feature_policy_id": _optional_identity(feature_policy_id),
    }
    if spec.startswith("checkpoint:"):
        checkpoint = Path(spec.split(":", 1)[1]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return {
            "policy_spec": spec,
            "policy_family": "checkpoint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "source_receipt": None,
            **identity,
        }

    return {
        "policy_spec": spec,
        "policy_family": "named_policy",
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "source_receipt": _named_policy_source_receipt(),
        **identity,
    }


def _verify_policy_snapshot(policy: dict[str, Any]) -> None:
    family = str(policy.get("policy_family") or "")
    if family == "checkpoint":
        checkpoint = Path(str(policy.get("checkpoint_path") or ""))
        expected = str(policy.get("checkpoint_sha256") or "")
        if not checkpoint.is_file() or not expected:
            raise ValueError("checkpoint policy snapshot is missing its file/hash receipt")
        actual = _sha256_file(checkpoint)
        if actual != expected:
            raise ValueError(
                f"checkpoint policy drift detected for {checkpoint}: {actual} != {expected}"
            )

        return
    if family != "named_policy":
        raise ValueError(f"unknown prospective policy family: {family!r}")
    expected_receipt = policy.get("source_receipt")
    if not isinstance(expected_receipt, dict):
        raise ValueError("named-policy snapshot is missing its source receipt")
    if expected_receipt.get("schema_version") != POLICY_SOURCE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("named-policy snapshot has an unsupported source receipt")
    expected = str(expected_receipt.get("source_sha256") or "")
    actual_receipt = _named_policy_source_receipt()
    actual = actual_receipt["source_sha256"]
    if not expected or actual != expected:
        raise ValueError(
            f"named-policy source drift detected for {policy.get('policy_spec')!r}: "
            f"{actual} != {expected or 'missing'}"
        )


def _deck_snapshot(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    deck = read_deck(resolved)
    if len(deck) != 60:
        raise ValueError(f"prospective deck must have 60 cards: {resolved}")

    return {
        "deck_path": str(resolved),
        "deck_file_sha256": _sha256_file(resolved),
        "deck_hash": normalized_deck_hash(deck),
        "card_count": len(deck),
    }


def _frozen_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    boundary = _information_boundary(args)

    return {
        "information_mode": boundary.mode.value,
        "actors": [
            {
                "seat": 0,
                "policy": _policy_snapshot(
                    args.policy0,
                    league_member_id=getattr(args, "policy0_league_member_id", None),
                    feature_policy_id=getattr(args, "policy0_feature_policy_id", None),
                ),
                "deck": _deck_snapshot(args.deck0),
            },
            {
                "seat": 1,
                "policy": _policy_snapshot(
                    args.policy1,
                    league_member_id=getattr(args, "policy1_league_member_id", None),
                    feature_policy_id=getattr(args, "policy1_feature_policy_id", None),
                ),
                "deck": _deck_snapshot(args.deck1),
            },
        ],
        "league_snapshot_id": str(args.league_snapshot_id),
        "rating_snapshot": unavailable_rating_snapshot(),
    }


def _information_boundary(args: argparse.Namespace) -> InformationBoundary:
    requested = getattr(args, "information_mode", None)
    reveal_opponent = bool(getattr(args, "reveal_opponent_private_state", False))
    if reveal_opponent:
        if requested not in (None, InformationMode.PRIVILEGED_ORACLE.value):
            raise ValueError(
                "--reveal-opponent-private-state conflicts with the selected information mode"
            )

        return InformationBoundary(mode=InformationMode.PRIVILEGED_ORACLE)

    return InformationBoundary(mode=InformationMode(requested or InformationMode.PUBLIC_BELIEF.value))


def _build_agent(policy: dict[str, Any], deck: list[int], deck_hash: str) -> Any:
    _verify_policy_snapshot(policy)
    spec = str(policy["policy_spec"])
    if spec.startswith("checkpoint:"):
        return arena.build_exploiter(Path(str(policy["checkpoint_path"])), deck, deck_hash, "reference_abomasnow")

    return SpecAgent.from_str(spec)


def _contract(plan: dict[str, Any], tranche: dict[str, Any], *, games: int, seed: int) -> dict[str, Any]:
    payload = {
        "plan_id": plan["plan_id"],
        "run_id": plan["run_id"],
        "tranche_id": tranche["tranche_id"],
        "tranche_sequence": tranche["tranche_sequence"],
        "frozen_snapshot": tranche["frozen_snapshot"],
        "games": games,
        "seed": seed,
        "public_projection_version": PUBLIC_PROJECTION_VERSION,
        "script_sha256": _sha256_file(Path(__file__)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["contract_sha256"] = hashlib.sha256(encoded).hexdigest()

    return payload


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    boundary = _information_boundary(args)
    require_oracle_output_path(out_dir, boundary)
    out_dir.mkdir(parents=True, exist_ok=False)
    plan_path = out_dir / "plan.json"
    plan = create_plan(
        plan_path,
        plan_id=safe_slug(args.plan_id),
        run_id=safe_slug(args.run_id),
        tranche_ids=[str(value) for value in args.tranche_ids],
        frozen_snapshot=_frozen_snapshot(args),
        league_identity=str(args.league_identity),
        curriculum_round=args.curriculum_round,
    )
    print(json.dumps(plan, indent=2))

    return plan


def _load_progress(path: Path, *, contract: dict[str, Any], resume: bool, shard_dir: Path) -> dict[str, Any]:
    durable = scan_durable_shards(shard_dir)
    if path.exists():
        if not resume:
            raise FileExistsError(f"prospective progress exists; pass --resume: {path}")
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("contract", {}).get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError("prospective resume contract mismatch")
        progress["resume_events"] = int(progress.get("resume_events") or 0) + 1
    else:
        if durable["games"]:
            raise ValueError("durable prospective shards exist without progress contract")
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "contract": contract,
            "resume_events": 0,
            "elapsed_seconds": 0.0,
        }
    progress.update(durable)
    progress["updated_at_utc"] = utc_now()
    progress["status"] = "running"
    write_json(path, progress)

    return progress


def _play_game(
    *,
    game_id: str,
    run_id: str,
    tranche: dict[str, Any],
    seed: int,
    actors: list[ActorMetadata],
    agents: dict[int, Any],
    decks: dict[int, list[int]],
    league_identity: str,
    curriculum_round: int | None,
    boundary: InformationBoundary,
    max_steps: int,
    game_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    started_at = utc_now()
    recorder = ProspectiveGameRecorder(
        game_id=game_id,
        run_id=run_id,
        tranche=tranche,
        seed=seed,
        actors=actors,
        league_identity=league_identity,
        curriculum_round=curriculum_round,
        boundary=boundary,
        game_started_at_utc=started_at,
        game_index=game_index,
    )
    safety = {"completed": 0, "crashes": 0, "timeouts": 0, "illegal": 0}
    obs = None
    try:
        obs, start_data = battle_start(decks[0], decks[1])
        if obs is None:
            raise RuntimeError(f"BattleStart failed: {start_data.errorType}")
        result = -1
        for _step in range(max_steps):
            current = obs.get("current") or {}
            result = int(current.get("result", -1))
            if result != -1:
                break
            actor = int(current["yourIndex"])
            selected = list(agents[actor].select(obs, rng))
            options = (obs.get("select") or {}).get("option") or []
            if any(index < 0 or index >= len(options) for index in selected):
                safety["illegal"] += 1
                raise ValueError(f"illegal selection in {game_id}: {selected}")
            visualizer_before = (
                visualize_data() if boundary.mode is InformationMode.PRIVILEGED_ORACLE else None
            )
            post_obs = battle_select(selected)
            recorder.record_transition(
                obs, selected, post_obs, visualizer_before=visualizer_before,
            )
            obs = post_obs
        else:
            safety["timeouts"] += 1
            raise TimeoutError(f"prospective game exceeded max steps: {game_id}")
        safety["completed"] += 1
        rows = recorder.finalize(
            result=result,
            result_reason="engine_result_only",
            game_completed_at_utc=utc_now(),
        )

        return rows, safety
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            if safety["timeouts"] == 0:
                safety["timeouts"] += 1
        elif safety["illegal"] == 0:
            safety["crashes"] += 1
        setattr(exc, "prospective_safety", dict(safety))
        raise
    finally:
        if obs is not None:
            battle_finish()


def _exception_safety(exc: Exception) -> dict[str, int]:
    attached = getattr(exc, "prospective_safety", None)
    if isinstance(attached, dict):
        parsed = {
            key: int(attached.get(key) or 0)
            for key in ("completed", "crashes", "timeouts", "illegal")
        }
        if any(parsed[key] for key in ("crashes", "timeouts", "illegal")):
            return parsed
    return {
        "completed": 0,
        "crashes": 0 if isinstance(exc, TimeoutError) else 1,
        "timeouts": 1 if isinstance(exc, TimeoutError) else 0,
        "illegal": 0,
    }


def _persist_safety_failure(
    *,
    plan_path: Path,
    tranche_id: str,
    progress_path: Path,
    progress: dict[str, Any],
    shard_dir: Path,
    game_id: str,
    safety: dict[str, int],
    error: Exception,
    elapsed_seconds: float,
) -> None:
    durable = scan_durable_shards(shard_dir)
    failure = {
        "game_id": game_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "safety": dict(safety),
    }
    failures = progress.setdefault("safety_failures", [])
    if not isinstance(failures, list):
        raise ValueError("prospective progress safety_failures must be a list")
    failures.append({**failure, "failed_at_utc": utc_now()})
    progress.update(durable)
    progress.update({
        "updated_at_utc": utc_now(),
        "status": "safety_failed",
        "elapsed_seconds": elapsed_seconds,
        "safety": dict(safety),
        "failed_game_id": game_id,
    })
    write_json(progress_path, progress)
    fail_tranche_safety(plan_path, tranche_id, receipt=failure)


def run_tranche(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    plan_path = Path(args.plan)
    plan = load_plan(plan_path)
    tranche = tranche_by_id(plan, args.tranche_id)
    snapshot = tranche["frozen_snapshot"]
    boundary = InformationBoundary(mode=InformationMode(snapshot["information_mode"]))
    out_dir = plan_path.parent / "tranches" / str(args.tranche_id)
    require_oracle_output_path(out_dir, boundary)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = begin_tranche(plan_path, args.tranche_id, frozen_snapshot=snapshot, resume=bool(args.resume))
    tranche = tranche_by_id(plan, args.tranche_id)
    contract = _contract(plan, tranche, games=int(args.games), seed=int(args.seed))
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    progress = _load_progress(progress_path, contract=contract, resume=bool(args.resume), shard_dir=shard_dir)
    prior_elapsed = float(progress.get("elapsed_seconds") or 0.0)
    completed = set(progress["completed_game_ids"])

    actor_rows = sorted(snapshot["actors"], key=lambda row: int(row["seat"]))
    actors = [
        ActorMetadata(seat=int(row["seat"]), policy=row["policy"], deck=row["deck"])
        for row in actor_rows
    ]
    decks = {actor.seat: read_deck(actor.deck["deck_path"]) for actor in actors}
    agents = {
        actor.seat: _build_agent(actor.policy, decks[actor.seat], str(actor.deck["deck_hash"]))
        for actor in actors
    }
    skipped = 0
    safety = {"completed": len(completed), "crashes": 0, "timeouts": 0, "illegal": 0}
    tranche_run_id = f"{plan['run_id']}:{tranche['tranche_id']}"
    for game_index in range(int(args.games)):
        game_id = f"{tranche_run_id}:g{game_index:05d}"
        if game_id in completed:
            skipped += 1
            continue
        try:
            rows, game_safety = _play_game(
                game_id=game_id,
                run_id=tranche_run_id,
                tranche=tranche,
                seed=int(args.seed) + game_index,
                actors=actors,
                agents=agents,
                decks=decks,
                league_identity=str(plan["league_identity"]),
                curriculum_round=plan.get("curriculum_round"),
                boundary=boundary,
                max_steps=int(args.max_steps),
                game_index=game_index,
            )
        except Exception as exc:
            game_safety = _exception_safety(exc)
            for key in ("crashes", "timeouts", "illegal"):
                safety[key] += game_safety[key]
            _persist_safety_failure(
                plan_path=plan_path,
                tranche_id=str(args.tranche_id),
                progress_path=progress_path,
                progress=progress,
                shard_dir=shard_dir,
                game_id=game_id,
                safety=safety,
                error=exc,
                elapsed_seconds=round(prior_elapsed + time.perf_counter() - started, 3),
            )
            raise
        for key in ("crashes", "timeouts", "illegal"):
            safety[key] += game_safety[key]
        if any(game_safety[key] for key in ("crashes", "timeouts", "illegal")):
            error = RuntimeError(f"prospective safety hard stop: {game_safety}")
            _persist_safety_failure(
                plan_path=plan_path,
                tranche_id=str(args.tranche_id),
                progress_path=progress_path,
                progress=progress,
                shard_dir=shard_dir,
                game_id=game_id,
                safety=safety,
                error=error,
                elapsed_seconds=round(prior_elapsed + time.perf_counter() - started, 3),
            )
            raise error
        for row in rows:
            assert_public_trace_row_safe(row)
        shard_path = shard_dir / f"decisions_shard_{game_index:05d}.jsonl.gz"
        if shard_path.exists():
            raise FileExistsError(f"refusing to overwrite prospective shard: {shard_path}")
        for row in rows:
            row["source_dataset"] = str(out_dir.resolve())
            row["source_shard"] = str(shard_path.resolve())
        atomic_write_gzip_jsonl(shard_path, rows)
        durable = scan_durable_shards(shard_dir)
        completed = set(durable["completed_game_ids"])
        safety["completed"] = len(completed)
        progress.update(durable)
        progress.update({
            "updated_at_utc": utc_now(),
            "status": "running",
            "elapsed_seconds": round(prior_elapsed + time.perf_counter() - started, 3),
            "safety": dict(safety),
        })
        write_json(progress_path, progress)

    durable = scan_durable_shards(shard_dir)
    elapsed = round(prior_elapsed + time.perf_counter() - started, 3)
    progress.update(durable)
    progress.update({
        "updated_at_utc": utc_now(),
        "status": "completed",
        "elapsed_seconds": elapsed,
        "safety": dict(safety),
        "skipped_games_this_invocation": skipped,
    })
    write_json(progress_path, progress)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "plan_id": plan["plan_id"],
        "run_id": tranche_run_id,
        "tranche_id": tranche["tranche_id"],
        "tranche_sequence": tranche["tranche_sequence"],
        "information_mode": boundary.mode.value,
        "public_projection_version": PUBLIC_PROJECTION_VERSION,
        "contract": contract,
        "coverage": durable,
        "safety": safety,
        "resume_events": progress["resume_events"],
        "skipped_games_this_invocation": skipped,
        "elapsed_seconds": elapsed,
    }
    write_json(out_dir / "manifest.json", manifest)
    complete_tranche(plan_path, args.tranche_id, receipt={
        "manifest": str((out_dir / "manifest.json").resolve()),
        "games": durable["games"],
        "decisions": durable["decisions"],
        "information_mode": boundary.mode.value,
    })
    print(json.dumps(manifest, indent=2))

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init-plan")
    init_parser.add_argument("--out-dir", required=True)
    init_parser.add_argument("--plan-id", required=True)
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--tranche-ids", nargs="+", required=True)
    init_parser.add_argument("--policy0", required=True)
    init_parser.add_argument("--policy1", required=True)
    init_parser.add_argument("--policy0-league-member-id")
    init_parser.add_argument("--policy1-league-member-id")
    init_parser.add_argument("--policy0-feature-policy-id")
    init_parser.add_argument("--policy1-feature-policy-id")
    init_parser.add_argument("--deck0", required=True)
    init_parser.add_argument("--deck1", required=True)
    init_parser.add_argument("--league-identity", required=True)
    init_parser.add_argument("--league-snapshot-id", required=True)
    init_parser.add_argument("--curriculum-round", type=int)
    init_parser.add_argument(
        "--information-mode",
        choices=[mode.value for mode in InformationMode],
        default=None,
    )
    init_parser.add_argument("--reveal-opponent-private-state", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--tranche-id", required=True)
    run_parser.add_argument("--games", type=int, default=1)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--max-steps", type=int, default=5000)
    run_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "init-plan":
        initialize(args)
    else:
        run_tranche(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
