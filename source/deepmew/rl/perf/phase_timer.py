"""Low-overhead phase timer for DeepMew RL hot-path profiling."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


DEFAULT_PHASES: tuple[str, ...] = (
    "env_reset",
    "engine_step",
    "engine_reset_start",
    "engine_action_decoding",
    "engine_action_submission",
    "engine_opponent_policy",
    "engine_opponent_action_submission",
    "engine_terminal_check",
    "engine_turn_owner_check",
    "engine_rules_effect_resolution",
    "engine_state_copying",
    "engine_logging",
    "engine_misc",
    "python_native_boundary",
    "allocation_snapshot",
    "serialization_probe",
    "opponent_step",
    "legal_action_enumeration",
    "complete_action_table_build",
    "action_mask_build",
    "observation_encode",
    "policy_forward",
    "mask_logits",
    "action_sample",
    "logprob_entropy",
    "submit_action",
    "reward_done_extract",
    "rollout_record_pack",
    "queue_send",
    "queue_recv",
    "advantage_compute",
    "ppo_minibatch_prepare",
    "ppo_forward",
    "ppo_loss",
    "ppo_backward",
    "optimizer_step",
    "checkpoint_write",
    "eval_game",
)


PHASE_CLASSIFICATION: dict[str, str] = {
    "env_reset": "process_isolation_required",
    "engine_step": "process_isolation_required",
    "engine_reset_start": "process_isolation_required",
    "engine_action_decoding": "process_isolation_required",
    "engine_action_submission": "process_isolation_required",
    "engine_opponent_policy": "process_isolation_required",
    "engine_opponent_action_submission": "process_isolation_required",
    "engine_terminal_check": "process_isolation_required",
    "engine_turn_owner_check": "process_isolation_required",
    "engine_rules_effect_resolution": "unmeasured_inside_native_transition",
    "engine_state_copying": "process_isolation_required",
    "engine_logging": "process_isolation_required",
    "engine_misc": "process_isolation_required",
    "python_native_boundary": "process_isolation_required",
    "allocation_snapshot": "diagnostic_probe",
    "serialization_probe": "diagnostic_probe",
    "opponent_step": "process_isolation_required",
    "legal_action_enumeration": "bookkeeping",
    "complete_action_table_build": "bookkeeping",
    "action_mask_build": "bookkeeping",
    "observation_encode": "batchable",
    "policy_forward": "batchable",
    "mask_logits": "batchable",
    "action_sample": "batchable",
    "logprob_entropy": "batchable",
    "submit_action": "process_isolation_required",
    "reward_done_extract": "bookkeeping",
    "rollout_record_pack": "bookkeeping",
    "queue_send": "queue",
    "queue_recv": "queue",
    "advantage_compute": "learner_only",
    "ppo_minibatch_prepare": "learner_only",
    "ppo_forward": "learner_only",
    "ppo_loss": "learner_only",
    "ppo_backward": "learner_only",
    "optimizer_step": "learner_only",
    "checkpoint_write": "io",
    "eval_game": "eval",
}


def _json_default(value: Any) -> str:
    return str(value)


def _percentile_us(values_ns: list[int], percentile: float) -> float:
    if not values_ns:
        return 0.0
    values = sorted(values_ns)
    if len(values) == 1:
        return values[0] / 1000.0
    rank = (len(values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower] / 1000.0
    fraction = rank - lower
    blended = values[lower] * (1.0 - fraction) + values[upper] * fraction
    return blended / 1000.0


@dataclass
class PhaseTimer:
    """Collects phase timings, counters, and profiler output files."""

    run_id: str
    out_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    phases: tuple[str, ...] = DEFAULT_PHASES
    durations_ns: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    events: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)

    @contextmanager
    def phase(self, name: str, **metadata: Any) -> Iterator[None]:
        started = time.perf_counter_ns()
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ns = time.perf_counter_ns() - started
            self.record_phase(name, elapsed_ns, error=error, **metadata)

    def record_phase(self, name: str, elapsed_ns: int, **metadata: Any) -> None:
        self.durations_ns[name].append(int(elapsed_ns))
        event = {
            "run_id": self.run_id,
            "phase": name,
            "elapsed_ns": int(elapsed_ns),
            "elapsed_us": round(elapsed_ns / 1000.0, 3),
            "timestamp_ns": time.time_ns(),
        }
        if metadata:
            event["metadata"] = metadata
        self.events.append(event)

    def increment(self, name: str, amount: int | float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def set_counter(self, name: str, value: Any) -> None:
        self.counters[name] = value

    def observe(self, name: str, value: int | float) -> None:
        self.observations[name].append(float(value))

    def total_timed_ns(self, *, include_diagnostics: bool = False) -> int:
        total = 0
        for phase, values in self.durations_ns.items():
            classification = PHASE_CLASSIFICATION.get(phase, "unknown")
            if not include_diagnostics and classification == "diagnostic_probe":
                continue
            total += sum(values)
        return total

    def summary_rows(self) -> list[dict[str, Any]]:
        total_ns = max(1, self.total_timed_ns())
        rows: list[dict[str, Any]] = []
        seen = set(self.phases)
        for phase in list(self.phases) + sorted(set(self.durations_ns) - seen):
            values = self.durations_ns.get(phase, [])
            calls = len(values)
            phase_total = sum(values)
            classification = PHASE_CLASSIFICATION.get(phase, "unknown")
            rows.append(
                {
                    "phase": phase,
                    "classification": classification,
                    "calls": calls,
                    "total_ms": round(phase_total / 1_000_000.0, 3),
                    "mean_us": round((phase_total / calls) / 1000.0, 3) if calls else 0.0,
                    "p50_us": round(_percentile_us(values, 0.50), 3),
                    "p95_us": round(_percentile_us(values, 0.95), 3),
                    "p99_us": round(_percentile_us(values, 0.99), 3),
                    "percent_of_total": (
                        0.0
                        if classification == "diagnostic_probe"
                        else round(100.0 * phase_total / total_ns, 4) if phase_total else 0.0
                    ),
                }
            )
        return rows

    def observation_summary(self) -> dict[str, dict[str, float]]:
        payload: dict[str, dict[str, float]] = {}
        for name, values in self.observations.items():
            ordered = sorted(values)
            if not ordered:
                continue
            payload[name] = {
                "count": float(len(ordered)),
                "mean": round(sum(ordered) / len(ordered), 6),
                "p95": round(_percentile_us([int(v * 1000) for v in ordered], 0.95), 6),
                "max": round(max(ordered), 6),
            }
        return payload

    def _write_jsonl(self, path: Path, events: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")

    def _top_rows(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = [row for row in self.summary_rows() if row["classification"] != "diagnostic_probe"]
        return sorted(rows, key=lambda row: float(row["total_ms"]), reverse=True)[:limit]

    def write_outputs(self) -> dict[str, str]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        phase_csv = self.out_dir / "perf_phase_summary.csv"
        rows = self.summary_rows()
        with phase_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        events_path = self.out_dir / "perf_phase_events.jsonl"
        self._write_jsonl(events_path, self.events)

        config_path = self.out_dir / "config.json"
        config_payload = {
            "schema_version": "deepmew.rl_hotpath_profile.v0",
            "run_id": self.run_id,
            "config": self.config,
            "counters": self.counters,
            "observations": self.observation_summary(),
            "total_timed_ms": round(self.total_timed_ns() / 1_000_000.0, 3),
            "diagnostic_timed_ms": round(
                (self.total_timed_ns(include_diagnostics=True) - self.total_timed_ns()) / 1_000_000.0,
                3,
            ),
        }
        config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

        summary_path = self.out_dir / "perf_summary.md"
        summary_path.write_text(self._summary_markdown(), encoding="utf-8")

        safety_path = self.out_dir / "perf_parallel_safety.md"
        safety_path.write_text(self._parallel_safety_markdown(), encoding="utf-8")

        recommendations_path = self.out_dir / "perf_bottleneck_recommendations.md"
        recommendations_path.write_text(self._recommendations_markdown(), encoding="utf-8")

        return {
            "perf_summary": str(summary_path),
            "perf_phase_summary": str(phase_csv),
            "perf_phase_events": str(events_path),
            "perf_parallel_safety": str(safety_path),
            "perf_bottleneck_recommendations": str(recommendations_path),
            "config": str(config_path),
        }

    def _summary_markdown(self) -> str:
        decisions = float(self.counters.get("decisions", 0) or 0)
        total_seconds = self.total_timed_ns() / 1_000_000_000.0
        dps = decisions / total_seconds if total_seconds > 0 else 0.0
        lines = [
            "# RL Hot-Path Profile",
            "",
            f"- run_id: `{self.run_id}`",
            f"- decisions: {int(decisions)}",
            f"- timed_seconds: {total_seconds:.3f}",
            f"- timed_decisions_per_sec: {dps:.3f}",
            f"- illegal_attempts: {self.counters.get('illegal_attempts', 0)}",
            f"- crashes: {self.counters.get('crashes', 0)}",
            f"- timeouts: {self.counters.get('timeouts', 0)}",
            f"- nan_inf_count: {self.counters.get('nan_inf_count', 0)}",
            f"- advantage_backend: {self.counters.get('advantage_backend', self.config.get('advantage_backend', 'unknown'))}",
            "",
            "## Top Phases",
            "",
            "| phase | calls | total_ms | mean_us | p95_us | percent | classification |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in self._top_rows():
            lines.append(
                f"| {row['phase']} | {row['calls']} | {row['total_ms']} | {row['mean_us']} | "
                f"{row['p95_us']} | {row['percent_of_total']} | {row['classification']} |"
            )
        lines.extend(["", "## Counters", ""])
        for key in sorted(self.counters):
            lines.append(f"- {key}: {self.counters[key]}")
        observations = self.observation_summary()
        if observations:
            lines.extend(["", "## Observations", ""])
            for key, value in sorted(observations.items()):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines) + "\n"

    def _parallel_safety_markdown(self) -> str:
        rows = self.summary_rows()
        by_class: dict[str, float] = defaultdict(float)
        for row in rows:
            by_class[str(row["classification"])] += float(row["total_ms"])
        total = max(1e-9, sum(by_class.values()))
        lines = [
            "# Parallel Safety Notes",
            "",
            "Same-process real-engine `num_envs > 1` remains blocked. Treat `process_isolation_required` phases as candidates for subprocess workers or a future native binding only after measurement justifies it.",
            "",
            "| class | total_ms | percent |",
            "| --- | ---: | ---: |",
        ]
        for key, value in sorted(by_class.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| {key} | {value:.3f} | {100.0 * value / total:.3f} |")
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- process_isolation_required: battle-engine lifecycle or stepping; do not parallelize in-process on Windows.",
                "- unmeasured_inside_native_transition: known work that is currently fused inside `battle_select`; needs engine instrumentation before attribution is honest.",
                "- diagnostic_probe: supplemental measurement such as allocation or serialization probes; useful for direction, not a direct rollout phase.",
                "- batchable: candidates for batched policy inference or tighter tensor layout.",
                "- learner_only: PPO/update work; candidates for GPU or minibatch/data movement optimization.",
                "- queue: candidates for shared-memory transport if subprocess rollout becomes worthwhile.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _recommendations_markdown(self) -> str:
        top = [row for row in self._top_rows(5) if row["calls"]]
        lines = ["# Bottleneck Recommendations", ""]
        if not top:
            lines.append("No measured phase calls were recorded.")
            return "\n".join(lines) + "\n"
        lines.append("Ranked by measured wall time in this profile:")
        lines.append("")
        for row in top:
            phase = row["phase"]
            recommendation = "Keep measuring; no canned recommendation for this phase yet."
            if phase in {
                "engine_step",
                "engine_reset_start",
                "engine_action_submission",
                "engine_opponent_action_submission",
                "engine_terminal_check",
                "engine_turn_owner_check",
                "engine_state_copying",
                "engine_misc",
                "opponent_step",
                "env_reset",
                "submit_action",
            }:
                recommendation = "Prioritize isolated rollout workers and fewer engine boundary crossings before native Puffer work."
            elif phase == "engine_opponent_policy":
                recommendation = "Cache immutable opponent inputs where possible and measure whether policy selection or engine submission is the true opponent-side cost."
            elif phase in {"allocation_snapshot", "serialization_probe"}:
                recommendation = "Use this as a direction check; run allocation or queue-specific profiles before rewriting hot paths."
            elif phase in {"policy_forward", "mask_logits", "action_sample", "logprob_entropy"}:
                recommendation = "Batch policy inference and inspect CPU/GPU transfer overhead before changing model semantics."
            elif phase in {"observation_encode", "action_mask_build", "complete_action_table_build", "legal_action_enumeration"}:
                recommendation = "Optimize data layout and avoid rebuilding unchanged action/observation structures."
            elif phase in {"queue_send", "queue_recv"}:
                recommendation = "Use shared memory or compact binary records if subprocess rollout is retained."
            elif phase in {"advantage_compute", "ppo_forward", "ppo_loss", "ppo_backward", "optimizer_step"}:
                recommendation = "Move learner work toward GPU/batched minibatches; keep `puffer_cpu` advantage optional and measured."
            elif phase == "checkpoint_write":
                recommendation = "Reduce checkpoint frequency or write asynchronously if checkpoint IO enters the top phases."
            lines.append(
                f"- {phase}: {row['total_ms']} ms ({row['percent_of_total']}%); {recommendation}"
            )
        return "\n".join(lines) + "\n"
