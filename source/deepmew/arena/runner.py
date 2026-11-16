from __future__ import annotations

import random
import time
from dataclasses import asdict

import main
from cg.game import battle_finish, battle_select, battle_start
from deepmew.arena.models import AgentSpec, GameRecord, PairSummary
from deepmew.arena.selection import select_action
from deepmew.arena.specs import parse_agent_spec
from deepmew.metrics import PolicyMetrics, update_metrics
from deepmew.trace.record import TraceRecorder


def run_game(
    agent0: AgentSpec,
    agent1: AgentSpec,
    *,
    game_index: int,
    seat_a: int,
    max_steps: int,
    rng: random.Random,
    metrics_by_agent: dict[str, PolicyMetrics] | None = None,
    trace_recorder: TraceRecorder | None = None,
    run_seed: int | None = None,
    game_id_prefix: str = "game",
) -> GameRecord:
    started = time.perf_counter()
    obs = None
    try:
        obs, start_data = battle_start(main.read_deck_csv(), main.read_deck_csv())
        if obs is None:
            raise RuntimeError(
                f"BattleStart failed: errorPlayer={start_data.errorPlayer}, "
                f"errorType={start_data.errorType}"
            )

        decision_index = 0
        game_id = f"{game_id_prefix}-{game_index:04d}"
        for _ in range(max_steps):
            current = obs.get("current") or {}
            if current.get("result", -1) != -1:
                result = int(current["result"])
                turns = int(current.get("turn") or 0)
                winner = agent0.label if result == 0 else agent1.label if result == 1 else None
                if trace_recorder is not None:
                    trace_recorder.finalize_game(
                        game_id=game_id,
                        final_result=result,
                        final_result_reason=str(current.get("resultReason", "")) if current.get("resultReason") is not None else None,
                    )
                return GameRecord(
                    game_index=game_index,
                    agent_a=agent0.label if seat_a == 0 else agent1.label,
                    agent_b=agent1.label if seat_a == 0 else agent0.label,
                    seat_a=seat_a,
                    result=result,
                    winner=winner,
                    draws=1 if result == 2 else 0,
                    turns=turns,
                    crashed=False,
                    timed_out=False,
                    elapsed_seconds=time.perf_counter() - started,
                )

            player_index = int(current.get("yourIndex") or 0)
            spec = agent0 if player_index == 0 else agent1
            selected = select_action(spec, obs, rng)
            if metrics_by_agent is not None:
                update_metrics(metrics_by_agent[spec.label], obs, selected)
            if trace_recorder is not None and obs.get("select") is not None:
                trace_recorder.record_decision(
                    game_id=game_id,
                    seed=run_seed,
                    obs=obs,
                    chosen_indices=selected,
                    decision_index=decision_index,
                    agent_spec=spec.raw,
                    agent_label=spec.label,
                    policy_version=spec.base,
                )
                decision_index += 1
            obs = battle_select(selected)

        if trace_recorder is not None:
            trace_recorder.finalize_game(game_id=game_id, final_result=None, final_result_reason="timeout")
        return GameRecord(
            game_index=game_index,
            agent_a=agent0.label if seat_a == 0 else agent1.label,
            agent_b=agent1.label if seat_a == 0 else agent0.label,
            seat_a=seat_a,
            result=None,
            winner=None,
            draws=0,
            turns=max_steps,
            crashed=False,
            timed_out=True,
            elapsed_seconds=time.perf_counter() - started,
            error=f"game exceeded {max_steps} selections",
        )
    except Exception as exc:
        if trace_recorder is not None:
            trace_recorder.finalize_game(
                game_id=f"{game_id_prefix}-{game_index:04d}",
                final_result=None,
                final_result_reason=f"crash:{type(exc).__name__}",
            )
        return GameRecord(
            game_index=game_index,
            agent_a=agent0.label if seat_a == 0 else agent1.label,
            agent_b=agent1.label if seat_a == 0 else agent0.label,
            seat_a=seat_a,
            result=None,
            winner=None,
            draws=0,
            turns=0,
            crashed=True,
            timed_out=False,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if obs is not None:
            battle_finish()


def run_pair(
    agent_a: str | AgentSpec,
    agent_b: str | AgentSpec,
    *,
    games: int,
    max_steps: int = 5000,
    seed: int | None = None,
    trace_recorder: TraceRecorder | None = None,
    game_id_prefix: str | None = None,
) -> PairSummary:
    spec_a = parse_agent_spec(agent_a) if isinstance(agent_a, str) else agent_a
    spec_b = parse_agent_spec(agent_b) if isinstance(agent_b, str) else agent_b
    rng = random.Random(seed)
    metrics = {spec_a.label: PolicyMetrics(), spec_b.label: PolicyMetrics()}
    records: list[GameRecord] = []
    started = time.perf_counter()
    final_game_id_prefix = game_id_prefix or "pair"

    for index in range(games):
        seat_a = index % 2
        if seat_a == 0:
            agent0, agent1 = spec_a, spec_b
        else:
            agent0, agent1 = spec_b, spec_a
        records.append(
            run_game(
                agent0,
                agent1,
                game_index=index + 1,
                seat_a=seat_a,
                max_steps=max_steps,
                rng=rng,
                metrics_by_agent=metrics,
                trace_recorder=trace_recorder,
                run_seed=seed,
                game_id_prefix=final_game_id_prefix,
            )
        )

    completed = sum(1 for record in records if record.result is not None and not record.crashed and not record.timed_out)
    a_wins = sum(
        1
        for record in records
        if record.result is not None and record.result != 2 and record.result == record.seat_a
    )
    b_wins = sum(
        1
        for record in records
        if record.result is not None and record.result != 2 and record.result != record.seat_a
    )
    draws = sum(record.draws for record in records)
    crashes = sum(1 for record in records if record.crashed)
    timeouts = sum(1 for record in records if record.timed_out)
    turns = sum(record.turns for record in records if record.result is not None)
    metrics_json = {label: asdict(value) for label, value in metrics.items()}
    return PairSummary(
        agent_a=spec_a.label,
        agent_b=spec_b.label,
        games=games,
        completed=completed,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        crashes=crashes,
        timeouts=timeouts,
        avg_turns=turns / completed if completed else 0.0,
        elapsed_seconds=time.perf_counter() - started,
        records=records,
        metrics=metrics_json,
    )
