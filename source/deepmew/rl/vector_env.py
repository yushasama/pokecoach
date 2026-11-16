"""Subprocess vector environment boundary for DeepMew RL.

The Pokemon battle engine owns process-global state. Multiple in-process
`SampledBattleEnv` instances are therefore not a safe scaling boundary on
Windows. This module gives each live battle env its own worker process and a
small request/response protocol.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass, field
from typing import Any

from deepmew.rl.env_adapter import StepResult


@dataclass(frozen=True)
class EnvFactorySpec:
    factory: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


def _load_factory(path: str) -> Any:
    module_name, _, attr_path = path.partition(":")
    if not module_name or not attr_path:
        raise ValueError(f"factory must be '<module>:<attr>', got {path!r}")
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def build_env_from_spec(spec: EnvFactorySpec) -> Any:
    factory = _load_factory(spec.factory)
    return factory(*spec.args, **spec.kwargs)


def _make_sampled_battle_env(*, experiment: str, seed: int, max_steps_per_game: int, env_id: int) -> Any:
    from deepmew.rl.algorithms.masked_ppo import SampledBattleEnv
    from deepmew.rl.registry import build_encoder, build_masker, build_opponent_sampler, build_reward, resolve_experiment
    from deepmew.rl.trainer import RLComponents

    config = resolve_experiment(experiment).validate()
    components = RLComponents(
        config=config,
        encoder=build_encoder(config),
        masker=build_masker(config),
        reward_fn=build_reward(config),
        opponent_sampler=build_opponent_sampler(config, seed=seed + env_id * 1009),
    )
    return SampledBattleEnv(
        components,
        seed=seed + 11 + env_id,
        max_steps_per_game=max_steps_per_game,
    )


def sampled_battle_env_spec(*, experiment: str, seed: int, max_steps_per_game: int, env_id: int) -> EnvFactorySpec:
    return EnvFactorySpec(
        factory="deepmew.rl.vector_env:_make_sampled_battle_env",
        kwargs={
            "experiment": experiment,
            "seed": seed,
            "max_steps_per_game": max_steps_per_game,
            "env_id": env_id,
        },
    )


def _step_payload(env: Any, step: StepResult) -> dict[str, Any]:
    return {
        "obs": step.obs,
        "done": step.done,
        "result": step.result,
        "crashed": step.crashed,
        "timed_out": step.timed_out,
        "info": step.info,
        "select": getattr(env, "select", None),
        "our_seat": getattr(env, "our_seat", 0),
        "opponent_label": getattr(env, "opponent_label", "unknown"),
    }


def _payload_to_step(payload: dict[str, Any]) -> StepResult:
    return StepResult(
        obs=payload.get("obs"),
        done=bool(payload.get("done")),
        result=payload.get("result"),
        crashed=bool(payload.get("crashed")),
        timed_out=bool(payload.get("timed_out")),
        info=dict(payload.get("info") or {}),
    )


def _worker_main(
    env_id: int,
    spec: EnvFactorySpec,
    command_q: mp.Queue,
    response_q: mp.Queue,
) -> None:
    env: Any | None = None
    try:
        env = build_env_from_spec(spec)
        response_q.put({"kind": "ready", "env_id": env_id, "pid": os.getpid()})
        while True:
            msg = command_q.get()
            cmd = msg.get("cmd")
            seq = msg.get("seq")
            try:
                if cmd == "close":
                    if env is not None:
                        env.close()
                    response_q.put({"kind": "closed", "seq": seq, "env_id": env_id})
                    return
                if cmd == "heartbeat":
                    response_q.put({"kind": "heartbeat", "seq": seq, "env_id": env_id, "pid": os.getpid()})
                    continue
                if cmd == "crash_for_test":
                    os._exit(17)
                if cmd == "reset":
                    assert env is not None
                    response_q.put({"kind": "step", "seq": seq, "env_id": env_id, **_step_payload(env, env.reset())})
                    continue
                if cmd == "step":
                    assert env is not None
                    response_q.put(
                        {
                            "kind": "step",
                            "seq": seq,
                            "env_id": env_id,
                            **_step_payload(env, env.step(list(msg.get("selection") or []))),
                        }
                    )
                    continue
                raise ValueError(f"unknown worker command: {cmd!r}")
            except BaseException as exc:
                response_q.put(
                    {
                        "kind": "error",
                        "seq": seq,
                        "env_id": env_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=20),
                    }
                )
    except BaseException as exc:
        response_q.put(
            {
                "kind": "init_error",
                "env_id": env_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )


class SubprocessEnvProxy:
    """One env hosted in one worker process."""

    def __init__(self, env_id: int, spec: EnvFactorySpec, *, timeout_seconds: float = 30.0):
        self.env_id = env_id
        self.spec = spec
        self.timeout_seconds = timeout_seconds
        self._ctx = mp.get_context("spawn")
        self._seq = 0
        self._command_q: mp.Queue | None = None
        self._response_q: mp.Queue | None = None
        self._process: mp.Process | None = None
        self._select: dict[str, Any] | None = None
        self.our_seat = 0
        self.opponent_label = f"worker-{env_id}"
        self.restart_count = 0
        self.last_crash_report: dict[str, Any] | None = None
        self._start()

    @property
    def select(self) -> dict[str, Any] | None:
        return self._select

    def _start(self) -> None:
        self._command_q = self._ctx.Queue()
        self._response_q = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self.env_id, self.spec, self._command_q, self._response_q),
            daemon=True,
        )
        self._process.start()
        ready = self._recv_any()
        if ready.get("kind") != "ready":
            raise RuntimeError(f"worker {self.env_id} failed to start: {ready}")

    def _recv_any(self) -> dict[str, Any]:
        assert self._response_q is not None
        try:
            return self._response_q.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(f"worker {self.env_id} did not respond within {self.timeout_seconds}s") from exc

    def _request(self, cmd: str, **payload: Any) -> dict[str, Any]:
        if self._process is None or not self._process.is_alive():
            return self._restart_after_crash({"cmd": cmd, "reason": "process_not_alive"})
        assert self._command_q is not None
        self._seq += 1
        seq = self._seq
        self._command_q.put({"cmd": cmd, "seq": seq, **payload})
        try:
            response = self._recv_any()
        except TimeoutError:
            return self._restart_after_crash({"cmd": cmd, "reason": "response_timeout"})
        if response.get("seq") != seq and response.get("kind") != "closed":
            raise RuntimeError(f"worker {self.env_id} returned mismatched response: {response}")
        if response.get("kind") == "error":
            raise RuntimeError(f"worker {self.env_id} command {cmd!r} failed: {response}")
        return response

    def _restart_after_crash(self, report: dict[str, Any]) -> dict[str, Any]:
        self.last_crash_report = dict(report)
        self.restart_count += 1
        self._terminate_process()
        self._start()
        return {
            "kind": "step",
            "env_id": self.env_id,
            "obs": None,
            "done": True,
            "result": None,
            "crashed": True,
            "timed_out": False,
            "info": {"worker_restart": True, "report": report},
            "select": None,
            "our_seat": self.our_seat,
            "opponent_label": self.opponent_label,
        }

    def _update_from_payload(self, payload: dict[str, Any]) -> StepResult:
        self._select = payload.get("select")
        self.our_seat = int(payload.get("our_seat", self.our_seat) or 0)
        self.opponent_label = str(payload.get("opponent_label", self.opponent_label))
        return _payload_to_step(payload)

    def reset(self) -> StepResult:
        return self._update_from_payload(self._request("reset"))

    def step(self, selection: list[int]) -> StepResult:
        return self._update_from_payload(self._request("step", selection=list(selection)))

    def heartbeat(self) -> dict[str, Any]:
        return self._request("heartbeat")

    def crash_for_test(self) -> None:
        if self._process is not None and self._process.is_alive() and self._command_q is not None:
            self._seq += 1
            self._command_q.put({"cmd": "crash_for_test", "seq": self._seq})

    def close(self) -> None:
        try:
            if self._process is not None and self._process.is_alive() and self._command_q is not None:
                self._seq += 1
                self._command_q.put({"cmd": "close", "seq": self._seq})
                try:
                    self._recv_any()
                except Exception:
                    pass
        finally:
            self._terminate_process()

    def _terminate_process(self) -> None:
        proc = self._process
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self._process = None


class SubprocessVectorEnv:
    """Batch facade over one `SubprocessEnvProxy` per env."""

    def __init__(self, specs: list[EnvFactorySpec], *, timeout_seconds: float = 30.0):
        if not specs:
            raise ValueError("SubprocessVectorEnv requires at least one env spec")
        self.envs = [
            SubprocessEnvProxy(env_id=index, spec=spec, timeout_seconds=timeout_seconds)
            for index, spec in enumerate(specs)
        ]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def restart_count(self) -> int:
        return sum(env.restart_count for env in self.envs)

    def reset(self, env_id: int) -> StepResult:
        return self.envs[env_id].reset()

    def reset_all(self) -> list[StepResult]:
        return [env.reset() for env in self.envs]

    def step_batch(self, selections: list[list[int]]) -> list[StepResult]:
        if len(selections) != len(self.envs):
            raise ValueError(f"expected {len(self.envs)} selections, got {len(selections)}")
        return [env.step(selection) for env, selection in zip(self.envs, selections)]

    def heartbeat(self) -> list[dict[str, Any]]:
        return [env.heartbeat() for env in self.envs]

    def crash_worker_for_test(self, env_id: int) -> None:
        self.envs[env_id].crash_for_test()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __enter__(self) -> "SubprocessVectorEnv":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()
