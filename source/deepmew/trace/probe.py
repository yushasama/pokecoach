"""Same-frame policy probing for recorded decision frames."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from deepmew.arena import parse_agent_spec, select_action
from deepmew.policy import valid_selection
from deepmew.trace.record import summarize_chosen_options
from deepmew.trace.schema import POLICY_PROBE_SCHEMA_VERSION, PolicyProbe, validate_frame_payload


def resolve_frames_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / "frames.jsonl"
    return path


def read_frames(path: Path | str, *, limit: int | None = None) -> list[dict[str, Any]]:
    frames_path = resolve_frames_path(path)
    frames: list[dict[str, Any]] = []
    with frames_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            validate_frame_payload(payload)
            frames.append(payload)
            if limit is not None and len(frames) >= limit:
                break
    return frames


def _probe_error(frame: dict[str, Any], policy_spec: str, message: str) -> PolicyProbe:
    return PolicyProbe(
        schema_version=POLICY_PROBE_SCHEMA_VERSION,
        frame_id=str(frame.get("frame_id", "")),
        policy_spec=policy_spec,
        chosen_indices=[],
        chosen_option_summaries=[],
        valid_return=False,
        error=message,
        notes=[],
    )


def probe_frame(frame: dict[str, Any], policy_spec: str) -> PolicyProbe:
    raw_observation = frame.get("raw_observation")
    if not isinstance(raw_observation, dict):
        return _probe_error(frame, policy_spec, "raw observation unavailable")
    select = raw_observation.get("select")
    if not isinstance(select, dict):
        return _probe_error(frame, policy_spec, "select payload unavailable")

    try:
        spec = parse_agent_spec(policy_spec)
        seed = frame.get("seed")
        decision_index = int(frame.get("decision_index") or 0)
        rng_seed = int(seed or 0) + decision_index
        chosen = select_action(spec, raw_observation, random.Random(rng_seed))
        valid = valid_selection(chosen, select)
        error = None if valid else "invalid return"
        return PolicyProbe(
            schema_version=POLICY_PROBE_SCHEMA_VERSION,
            frame_id=str(frame.get("frame_id", "")),
            policy_spec=spec.label,
            chosen_indices=list(chosen),
            chosen_option_summaries=summarize_chosen_options(select, list(chosen), raw_observation),
            valid_return=valid,
            error=error,
            notes=[],
        )
    except Exception as exc:
        return _probe_error(frame, policy_spec, f"{type(exc).__name__}: {exc}")


def probe_frames(frames: Iterable[dict[str, Any]], policy_specs: list[str]) -> list[PolicyProbe]:
    probes: list[PolicyProbe] = []
    for frame in frames:
        for policy_spec in policy_specs:
            probes.append(probe_frame(frame, policy_spec))
    return probes
