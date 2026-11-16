"""Reproducible run manifests + summaries for RL runs.

Builds a `deepmew.run_manifest.v1` manifest (via `deepmew.artifacts`) plus a
sibling metrics summary. The manifest records the full reproducibility header
required by DEC-040: experiment id, seed, git status summary, config dict,
config_sha256, observation_schema_sha256, reward name, opponent sampler name,
feature-module names, algorithm/trainer backend, timestamps, output paths, and
output content hashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, utc_now_iso, write_manifest
from deepmew.artifacts.run_dir import RunDirectory, create_run_dir
from deepmew.artifacts.writers import write_json
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.logging import git_status_summary, sha256_json, sha256_text
from deepmew.rl.observation import ObservationSchema

RL_RUN_KIND = "rl_baseline"


def reproducibility_header(
    config: RLExperimentConfig,
    schema: ObservationSchema,
    *,
    seed: int,
    command: str | list[str],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_dict = config.to_dict()
    payload = {
        "experiment_id": config.experiment_id,
        "seed": seed,
        "git": git_status_summary(),
        "config": config_dict,
        "config_sha256": sha256_json(config_dict),
        "observation_schema_sha256": sha256_text("\n".join(schema.names)),
        "observation_schema_dim": schema.dim,
        "reward_name": config.reward,
        "opponent_sampler_name": config.opponent_sampler,
        "feature_module_names": list(config.feature_modules),
        "trainer_backend": config.trainer_backend,
        "algorithm_backend": config.trainer_backend,
        "created_at_utc": utc_now_iso(),
        "command": command,
    }
    if run_metadata is not None:
        payload["run_metadata"] = run_metadata
    return payload


def write_rl_run(
    *,
    config: RLExperimentConfig,
    schema: ObservationSchema,
    seed: int,
    command: str | list[str],
    result: dict[str, Any],
    metrics: dict[str, Any],
    out_root: Path | str = ".research/runs",
    run_id: str | None = None,
    suffix: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> RunDirectory:
    """Write summary.json + manifest.json under a fresh run directory."""
    run_dir = create_run_dir(kind=RL_RUN_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    header = reproducibility_header(config, schema, seed=seed, command=command, run_metadata=run_metadata)

    summary_payload = {
        "schema_version": "deepmew.rl_run_summary.v1",
        "reproducibility": header,
        "result": result,
        "metrics": metrics,
    }
    if run_metadata is not None:
        summary_payload["run_metadata"] = run_metadata
    summary_path = run_dir.output_path("summary.json")
    write_json(summary_path, summary_payload)

    outputs = {
        "summary": str(summary_path),
        "summary_sha256": sha256_json(summary_payload),
    }
    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=RL_RUN_KIND,
        command=command,
        status="completed",
        inputs={
            "experiment_id": config.experiment_id,
            "seed": seed,
            "config_sha256": header["config_sha256"],
            "observation_schema_sha256": header["observation_schema_sha256"],
            "reward_name": config.reward,
            "opponent_sampler_name": config.opponent_sampler,
            "feature_module_names": list(config.feature_modules),
            "trainer_backend": config.trainer_backend,
            "git": header["git"],
            "run_metadata": run_metadata,
        },
        outputs=outputs,
        summary={
            "dry_run": result.get("dry_run"),
            "episodes": result.get("episodes"),
            "decision_steps": result.get("decision_steps"),
            "win_rate": metrics.get("runtime_metrics", {}).get("win_rate"),
            "crashes": result.get("crashes"),
            "timeouts": result.get("timeouts"),
            "illegal_attempts": result.get("illegal_attempts"),
            "agent_deck_hash": (run_metadata or {}).get("agent_deck_hash"),
            "opponent_deck_hash": (run_metadata or {}).get("opponent_deck_hash"),
            "opponent_decks": (run_metadata or {}).get("opponents"),
            "advantage_backend": (run_metadata or {}).get("advantage_backend"),
            "vector_backend": (run_metadata or {}).get("vector_backend"),
            "puffer_vectorized_status": (run_metadata or {}).get("puffer_vectorized_status"),
        },
    )
    write_manifest(run_dir.manifest_path, manifest)
    return run_dir
