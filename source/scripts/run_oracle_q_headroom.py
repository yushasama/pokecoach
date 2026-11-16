"""Privileged-oracle, terminal-outcome counterfactual-Q headroom audit."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cg.api import search_begin, search_end, search_step, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from deepmew.artifacts.writers import write_json, write_jsonl  # noqa: E402
from deepmew.deck import read_deck  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent  # noqa: E402
from deepmew.replay.information_boundary import (  # noqa: E402
    InformationBoundary,
    InformationMode,
    require_oracle_output_path,
)
from deepmew.replay.prospective_trace import public_trace_observation, utc_now  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402
import run_r17_bucket_arena as arena  # noqa: E402


SCHEMA_VERSION = "deepmew.oracle_q_headroom.v1"
ROOT_SCHEMA_VERSION = "deepmew.oracle_q_root.v1"
BRANCH_SCHEMA_VERSION = "deepmew.oracle_q_branch.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _policy_snapshot(spec: str) -> dict[str, Any]:
    if spec.startswith("checkpoint:"):
        path = Path(spec.split(":", 1)[1]).resolve()
        return {
            "policy_spec": spec, "policy_family": "checkpoint",
            "checkpoint_path": str(path), "checkpoint_sha256": _sha256_file(path),
        }

    return {
        "policy_spec": spec, "policy_family": "named_policy",
        "checkpoint_path": None, "checkpoint_sha256": None,
    }


def _build_agent(snapshot: dict[str, Any], deck: list[int], deck_hash: str) -> Any:
    spec = str(snapshot["policy_spec"])
    if spec.startswith("checkpoint:"):
        return arena.build_exploiter(Path(str(snapshot["checkpoint_path"])), deck, deck_hash, "reference_abomasnow")

    return SpecAgent.from_str(spec)


def _card_ids(values: Any) -> list[int]:
    return [
        int(value["id"])
        for value in (values if isinstance(values, list) else [])
        if isinstance(value, dict) and value.get("id") is not None
    ]


def _latest_visualizer(payload: str) -> dict[str, Any]:
    decoded = json.loads(payload)
    if not isinstance(decoded, list) or not decoded or not isinstance(decoded[-1], dict):
        raise ValueError("oracle visualizer payload is not a non-empty observation list")

    return decoded[-1]


def exact_state_hash(observation: dict[str, Any], visualizer_payload: str) -> str:
    visualizer = _latest_visualizer(visualizer_payload)
    private_zones = []
    for player in visualizer["current"]["players"]:
        private_zones.append({
            zone: _card_ids(player.get(zone))
            for zone in ("hand", "deck", "prize", "active")
        })
    payload = {
        "public": public_trace_observation(observation),
        "private_zones": private_zones,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _begin_exact_search(observation: dict[str, Any], visualizer_payload: str):
    visualizer = _latest_visualizer(visualizer_payload)
    actor = int(observation["current"]["yourIndex"])
    players = visualizer["current"]["players"]
    own = players[actor]
    opponent = players[1 - actor]

    return search_begin(
        to_observation_class(observation),
        _card_ids(own.get("deck")),
        _card_ids(own.get("prize")),
        _card_ids(opponent.get("deck")),
        _card_ids(opponent.get("prize")),
        _card_ids(opponent.get("hand")),
        _card_ids(opponent.get("active")),
        manual_coin=True,
    )


def enumerate_legal_selections(select: dict[str, Any], *, limit: int | None = None) -> list[list[int]]:
    options = select.get("option") or []
    if not isinstance(options, list):
        return []
    minimum = max(0, int(select.get("minCount") or 0))
    maximum = min(len(options), int(select.get("maxCount") if select.get("maxCount") is not None else len(options)))
    selections: list[list[int]] = []
    for count in range(minimum, maximum + 1):
        for combination in itertools.combinations(range(len(options)), count):
            selections.append(list(combination))
            if limit is not None and len(selections) > limit:
                return selections

    return selections


def _terminal_payoff(result: int, root_actor: int) -> float:
    if result == root_actor:
        return 1.0
    if result in (0, 1):
        return 0.0

    return 0.5


def _state_reference(observation: dict[str, Any]) -> str:
    encoded = json.dumps(
        public_trace_observation(observation), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _search_root_reference(observation: dict[str, Any]) -> str:
    """SearchBegin reconstructs state/select but not the transient prior-action logs."""
    def remove_null_fields(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: remove_null_fields(item)
                for key, item in value.items()
                if item is not None and key != "playerIndex"
            }
        if isinstance(value, list):
            return [remove_null_fields(item) for item in value]

        return value

    payload = {
        "current": remove_null_fields(public_trace_observation(observation).get("current")),
        "select": remove_null_fields(copy.deepcopy(observation.get("select"))),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def rollout_branch(
    *,
    root_observation: dict[str, Any],
    root_visualizer_payload: str,
    expected_root_hash: str,
    forced_selection: list[int],
    agents: dict[int, Any],
    rollout_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    live_hash = exact_state_hash(root_observation, visualize_data())
    if live_hash != expected_root_hash:
        raise ValueError("live root state changed before sibling branch")
    if exact_state_hash(root_observation, root_visualizer_payload) != expected_root_hash:
        raise ValueError("stored oracle root state hash mismatch")
    root_actor = int(root_observation["current"]["yourIndex"])
    rng = random.Random(rollout_seed)
    trajectory: list[dict[str, Any]] = []
    try:
        root = _begin_exact_search(root_observation, root_visualizer_payload)
        root_search_public = dataclasses.asdict(root.observation)
        if _search_root_reference(root_search_public) != _search_root_reference(root_observation):
            raise ValueError("SearchBegin root public state does not match source root")
        before = root_observation
        state = search_step(root.searchId, forced_selection)
        after = dataclasses.asdict(state.observation)
        trajectory.append({
            "logical_decision_step": 0,
            "actor": root_actor,
            "root_actor": root_actor,
            "public_state_reference": _state_reference(before),
            "public_event_delta": copy.deepcopy(after.get("logs") or []),
            "reward": _terminal_payoff(int(after["current"]["result"]), root_actor)
            if int(after["current"]["result"]) != -1 else 0.0,
            "control_owner_before": root_actor,
            "control_owner_after": int(after["current"]["yourIndex"]),
            "terminal": int(after["current"]["result"]) != -1,
            "seed": rollout_seed,
        })
        steps = 1
        while int(after["current"]["result"]) == -1 and steps < max_steps:
            actor = int(after["current"]["yourIndex"])
            selected = list(agents[actor].select(after, rng))
            before = after
            state = search_step(state.searchId, selected)
            after = dataclasses.asdict(state.observation)
            result = int(after["current"]["result"])
            trajectory.append({
                "logical_decision_step": steps,
                "actor": actor,
                "root_actor": root_actor,
                "public_state_reference": _state_reference(before),
                "public_event_delta": copy.deepcopy(after.get("logs") or []),
                "reward": _terminal_payoff(result, root_actor) if result != -1 else 0.0,
                "control_owner_before": actor,
                "control_owner_after": int(after["current"]["yourIndex"]),
                "terminal": result != -1,
                "seed": rollout_seed,
            })
            steps += 1
        if int(after["current"]["result"]) == -1:
            raise TimeoutError("oracle branch exceeded max logical decisions")
        result = int(after["current"]["result"])

        return {
            "terminal_result": result,
            "terminal_payoff": _terminal_payoff(result, root_actor),
            "logical_decision_steps": steps,
            "trajectory": trajectory,
        }
    finally:
        search_end()


def _turn_band(turn: int) -> str:
    if turn <= 2:
        return "early_0_2"
    if turn <= 5:
        return "mid_3_5"

    return "late_6_plus"


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "max": None}

    return {
        "count": len(values), "min": min(values), "mean": sum(values) / len(values),
        "p50": statistics.median(values), "max": max(values),
    }


def _summarize_root(root: dict[str, Any]) -> dict[str, Any]:
    action_rows = root["actions"]
    means = [float(action["q_mean"]) for action in action_rows if action["q_mean"] is not None]
    ranked = sorted(action_rows, key=lambda action: (-float(action["q_mean"]), action["option_index"]))
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ranked[0]
    chosen = next((action for action in action_rows if action["is_logged_choice"]), None)
    per_seed_best = []
    seeds = sorted({rollout["seed"] for action in action_rows for rollout in action["rollouts"] if not rollout.get("error")})
    for seed in seeds:
        by_seed = []
        for action in action_rows:
            rollout = next((item for item in action["rollouts"] if item.get("seed") == seed and not item.get("error")), None)
            if rollout is not None:
                by_seed.append((float(rollout["terminal_payoff"]), action["option_index"]))
        if by_seed:
            best_value = max(value for value, _index in by_seed)
            per_seed_best.append(sorted(index for value, index in by_seed if value == best_value))
    ranking_stability = (
        sum(choice == per_seed_best[0] for choice in per_seed_best) / len(per_seed_best)
        if per_seed_best else None
    )
    root.update({
        "best_option_index": best["option_index"],
        "chosen_option_index": chosen["option_index"] if chosen else None,
        "chosen_action_regret": (
            float(best["q_mean"]) - float(chosen["q_mean"]) if chosen and chosen["q_mean"] is not None else None
        ),
        "best_minus_second_best_gap": float(best["q_mean"]) - float(second["q_mean"]),
        "chosen_is_q_best": bool(chosen and math.isclose(float(chosen["q_mean"]), float(best["q_mean"]))),
        "ranking_stability_across_paired_seeds": ranking_stability,
        "q_min": min(means), "q_max": max(means),
    })

    return root


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    boundary = InformationBoundary(mode=InformationMode.PRIVILEGED_ORACLE)
    require_oracle_output_path(out_dir, boundary)
    out_dir.mkdir(parents=True, exist_ok=False)
    deck_paths = {0: Path(args.deck0).resolve(), 1: Path(args.deck1).resolve()}
    decks = {seat: read_deck(path) for seat, path in deck_paths.items()}
    deck_snapshots = {
        seat: {
            "deck_path": str(deck_paths[seat]),
            "deck_file_sha256": _sha256_file(deck_paths[seat]),
            "deck_hash": normalized_deck_hash(decks[seat]),
        }
        for seat in (0, 1)
    }
    policies = {0: _policy_snapshot(args.policy0), 1: _policy_snapshot(args.policy1)}
    agents = {
        seat: _build_agent(policies[seat], decks[seat], deck_snapshots[seat]["deck_hash"])
        for seat in (0, 1)
    }
    matchup_id = hashlib.sha256(json.dumps({
        "decks": deck_snapshots, "policies": policies,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    created_at_utc = utc_now()
    run_config = {
        "games": int(args.games), "seed": int(args.seed), "branch_seed": int(args.branch_seed),
        "rollouts_per_action": int(args.rollouts_per_action), "max_roots": int(args.max_roots),
        "max_legal_actions": int(args.max_legal_actions), "max_steps": int(args.max_steps),
        "branch_max_steps": int(args.branch_max_steps),
    }
    script_sha256 = _sha256_file(Path(__file__))
    config_sha256 = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    code_provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha256,
        "config_sha256": config_sha256,
    }
    roots: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    selection_cells: set[tuple[str, str]] = set()
    source_safety = {"completed": 0, "crashes": 0, "timeouts": 0, "illegal": 0}
    branch_safety = {"errors": 0, "timeouts": 0, "illegal": 0}
    excluded_large_legal_sets = 0
    for game_index in range(int(args.games)):
        game_id = f"{safe_run_id(args.run_id)}:g{game_index:04d}"
        rng = random.Random(int(args.seed) + game_index)
        obs = None
        game_root_indices: list[int] = []
        try:
            obs, start_data = battle_start(decks[0], decks[1])
            if obs is None:
                raise RuntimeError(f"BattleStart failed: {start_data.errorType}")
            decision_index = 0
            result = -1
            for _step in range(int(args.max_steps)):
                current = obs["current"]
                result = int(current["result"])
                if result != -1:
                    break
                actor = int(current["yourIndex"])
                selected = list(agents[actor].select(obs, rng))
                select = obs.get("select") or {}
                option_count = len(select.get("option") or [])
                if any(index < 0 or index >= option_count for index in selected):
                    source_safety["illegal"] += 1
                    raise ValueError(f"source policy selected illegal option: {selected}")
                legal = enumerate_legal_selections(select, limit=int(args.max_legal_actions))
                if len(legal) > int(args.max_legal_actions):
                    excluded_large_legal_sets += 1
                else:
                    turn = int(current.get("turn") or 0)
                    band = _turn_band(turn)
                    cell = (str(policies[actor]["policy_spec"]), band)
                    if (cell not in selection_cells and len(legal) >= 2
                            and len(roots) < int(args.max_roots)):
                        selection_cells.add(cell)
                        root_id = f"{game_id}:d{decision_index:05d}"
                        visualizer_payload = visualize_data()
                        root_hash = exact_state_hash(obs, visualizer_payload)
                        semantics = legal_action_table(obs, select)
                        action_results = []
                        for option_index, forced in enumerate(legal):
                            rollouts = []
                            for rollout_index in range(int(args.rollouts_per_action)):
                                paired_seed = int(args.branch_seed) + (10000 * len(roots)) + rollout_index
                                branch_id = f"{root_id}:a{option_index:04d}:r{rollout_index:03d}"
                                try:
                                    outcome = rollout_branch(
                                        root_observation=obs,
                                        root_visualizer_payload=visualizer_payload,
                                        expected_root_hash=root_hash,
                                        forced_selection=forced,
                                        agents=agents,
                                        rollout_seed=paired_seed,
                                        max_steps=int(args.branch_max_steps),
                                    )
                                    rollout = {
                                        "schema_version": BRANCH_SCHEMA_VERSION,
                                        "created_at_utc": created_at_utc,
                                        "information_mode": "privileged_oracle",
                                        "value_scale": "terminal_win_probability",
                                        "code_provenance": copy.deepcopy(code_provenance),
                                        "branch_id": branch_id,
                                        "root_id": root_id,
                                        "option_index": option_index,
                                        "forced_selection_private": forced,
                                        "seed": paired_seed,
                                        "error": None,
                                        **outcome,
                                    }
                                except TimeoutError as exc:
                                    branch_safety["timeouts"] += 1
                                    rollout = {"branch_id": branch_id, "root_id": root_id, "option_index": option_index, "seed": paired_seed, "error": f"TimeoutError:{exc}"}
                                except (ValueError, IndexError) as exc:
                                    branch_safety["illegal"] += 1
                                    rollout = {"branch_id": branch_id, "root_id": root_id, "option_index": option_index, "seed": paired_seed, "error": f"{type(exc).__name__}:{exc}"}
                                except Exception as exc:
                                    branch_safety["errors"] += 1
                                    rollout = {"branch_id": branch_id, "root_id": root_id, "option_index": option_index, "seed": paired_seed, "error": f"{type(exc).__name__}:{exc}"}
                                rollouts.append(rollout)
                                branches.append(rollout)
                            payoffs = [float(row["terminal_payoff"]) for row in rollouts if not row.get("error")]
                            action_results.append({
                                "option_index": option_index,
                                "forced_selection_private": forced,
                                "forced_action_private": [semantics[index] for index in forced],
                                "is_logged_choice": sorted(forced) == sorted(selected),
                                "q_mean": sum(payoffs) / len(payoffs) if payoffs else None,
                                "q_std": statistics.pstdev(payoffs) if len(payoffs) > 1 else (0.0 if payoffs else None),
                                "n_rollouts": len(payoffs),
                                "terminal_rate": len(payoffs) / len(rollouts) if rollouts else None,
                                "rollouts": rollouts,
                            })
                        if all(action["q_mean"] is not None for action in action_results):
                            root = _summarize_root({
                                "schema_version": ROOT_SCHEMA_VERSION,
                                "created_at_utc": created_at_utc,
                                "information_mode": "privileged_oracle",
                                "source": "fresh_live_oracle_trace",
                                "value_scale": "terminal_win_probability",
                                "code_provenance": copy.deepcopy(code_provenance),
                                "run_id": safe_run_id(args.run_id),
                                "game_id": game_id,
                                "decision_id": root_id,
                                "actor_seat": actor,
                                "turn": turn,
                                "phase": str(semantics[0].get("select_context") if semantics else "not_found"),
                                "action_type": None,
                                "turn_band": band,
                                "matchup_id": matchup_id,
                                "actor_policy": copy.deepcopy(policies[actor]),
                                "opponent_policy": copy.deepcopy(policies[1 - actor]),
                                "actor_deck": copy.deepcopy(deck_snapshots[actor]),
                                "opponent_deck": copy.deepcopy(deck_snapshots[1 - actor]),
                                "root_state_hash": root_hash,
                                "logged_selection_private": selected,
                                "legal_action_count": len(legal),
                                "actions": action_results,
                                "source_result": None,
                                "source_outcome_for_actor": None,
                                "warnings": [],
                            })
                            game_root_indices.append(len(roots))
                            roots.append(root)
                post_obs = battle_select(selected)
                obs = post_obs
                decision_index += 1
            else:
                source_safety["timeouts"] += 1
                raise TimeoutError(f"source game exceeded max steps: {game_id}")
            source_safety["completed"] += 1
            for root_index in game_root_indices:
                roots[root_index]["source_result"] = result
                roots[root_index]["source_outcome_for_actor"] = _terminal_payoff(result, int(roots[root_index]["actor_seat"]))
        except (TimeoutError, ValueError, IndexError):
            raise
        except Exception:
            source_safety["crashes"] += 1
            raise
        finally:
            if obs is not None:
                battle_finish()
        if len(roots) >= int(args.max_roots):
            break
    regrets = [float(root["chosen_action_regret"]) for root in roots if root["chosen_action_regret"] is not None]
    gaps = [float(root["best_minus_second_best_gap"]) for root in roots]
    variances = [float(action["q_std"]) for root in roots for action in root["actions"] if action["q_std"] is not None]
    stability = [float(root["ranking_stability_across_paired_seeds"]) for root in roots if root["ranking_stability_across_paired_seeds"] is not None]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "information_mode": "privileged_oracle",
        "private_state_flag_required": True,
        "run_id": safe_run_id(args.run_id),
        "matchup_id": matchup_id,
        "source_policies": policies,
        "source_decks": deck_snapshots,
        "run_config": run_config,
        "code_provenance": code_provenance,
        "source_games": source_safety["completed"],
        "sampled_roots": len(roots),
        "sampled_decision_ids": [root["decision_id"] for root in roots],
        "selection_cells": sorted([list(cell) for cell in selection_cells]),
        "excluded_large_legal_sets": excluded_large_legal_sets,
        "legal_action_count_distribution": dict(sorted(Counter(root["legal_action_count"] for root in roots).items())),
        "total_actions": sum(len(root["actions"]) for root in roots),
        "total_terminal_rollouts": sum(action["n_rollouts"] for root in roots for action in root["actions"]),
        "source_safety": source_safety,
        "branch_safety": branch_safety,
        "chosen_action_regret": _distribution(regrets),
        "best_second_gap": _distribution(gaps),
        "q_std": _distribution(variances),
        "ranking_stability": _distribution(stability),
        "chosen_q_best_fraction": sum(root["chosen_is_q_best"] for root in roots) / len(roots) if roots else None,
        "turn_bands": dict(sorted(Counter(root["turn_band"] for root in roots).items())),
        "phases": dict(sorted(Counter(root["phase"] for root in roots).items())),
        "root_policy_families": dict(sorted(Counter(root["actor_policy"]["policy_spec"] for root in roots).items())),
        "continuation_semantics": "force one exact root selection; then frozen source-game policies for both seats to terminal",
        "paired_seed_method": "same rollout seed index reused across every sibling action at a root; fresh exact SearchBegin per sibling",
        "value_model_used": False,
        "ema_calculated": False,
        "outputs": {
            "roots": str((out_dir / "oracle_q_roots.jsonl").resolve()),
            "branches": str((out_dir / "oracle_q_branches.jsonl").resolve()),
            "summary": str((out_dir / "summary.json").resolve()),
        },
    }
    write_jsonl(out_dir / "oracle_q_roots.jsonl", roots)
    write_jsonl(out_dir / "oracle_q_branches.jsonl", branches)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    return summary


def safe_run_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)

    return cleaned.strip("._-") or "oracle-q"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deck0", required=True)
    parser.add_argument("--deck1", required=True)
    parser.add_argument("--policy0", required=True)
    parser.add_argument("--policy1", required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=770000)
    parser.add_argument("--branch-seed", type=int, default=880000)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--max-roots", type=int, default=6)
    parser.add_argument("--max-legal-actions", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--branch-max-steps", type=int, default=5000)
    parser.add_argument("--reveal-opponent-private-state", action="store_true", required=True)
    args = parser.parse_args(argv)
    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
