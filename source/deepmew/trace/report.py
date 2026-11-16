"""Markdown reports for trace and probe runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from deepmew.reports.markdown import bullets, section, table, title
from deepmew.trace.classify import classify_frame_diff


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    return dict(value)


def _top(counter: Counter[str], limit: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(limit) or [("none", 0)]


def build_probe_report(
    *,
    run_id: str,
    frames: Iterable[dict[str, Any]],
    probes: Iterable[Any],
    diffs: Iterable[Any],
    agents: list[str],
    source_trace: str,
    crashes: int | None = None,
    timeouts: int | None = None,
    skipped_frames: int = 0,
) -> str:
    frame_list = list(frames)
    frame_by_id = {str(frame.get("frame_id")): frame for frame in frame_list}
    probe_payloads = [_payload(probe) for probe in probes]
    diff_payloads = [_payload(diff) for diff in diffs]
    diff_counter: Counter[str] = Counter()
    symptom_counter: Counter[str] = Counter()
    examples: list[list[Any]] = []

    for diff in diff_payloads:
        for tag in diff.get("diff_tags") or []:
            diff_counter[str(tag)] += 1
        frame = frame_by_id.get(str(diff.get("frame_id")), {})
        symptoms = classify_frame_diff(frame, diff)
        for symptom in symptoms:
            symptom_counter[symptom] += 1
        if len(examples) < 8 and not diff.get("same_choice"):
            examples.append(
                [
                    diff.get("frame_id"),
                    ", ".join(diff.get("diff_tags") or []),
                    ", ".join(symptoms),
                    diff.get("base_choice"),
                    diff.get("compare_choice"),
                ]
            )

    errored = sum(1 for probe in probe_payloads if probe.get("error"))
    valid = sum(1 for probe in probe_payloads if probe.get("valid_return"))
    lines = [
        title("Trace Probe Report"),
        f"- run_id: `{run_id}`\n",
        f"- source_trace: `{source_trace}`\n",
        f"- agents: {', '.join(agents)}\n",
        f"- games: {len({frame.get('game_id') for frame in frame_list})}\n",
        f"- frames: {len(frame_list)}\n",
        f"- probed_frames: {len({probe.get('frame_id') for probe in probe_payloads})}\n",
        f"- probes: {len(probe_payloads)}\n",
        f"- valid_probe_returns: {valid}\n",
        f"- probe_errors: {errored}\n",
        f"- skipped_frames: {skipped_frames}\n",
        f"- crashes: {crashes if crashes is not None else 'unknown'}\n",
        f"- timeouts: {timeouts if timeouts is not None else 'unknown'}\n",
        "\n",
        section("Top Diff Tags"),
        table(["tag", "count"], _top(diff_counter)),
        "\n",
        section("Top Symptom Buckets"),
        "These are observed symptom tags, likely causes are not proven causal full-game counterfactuals after a divergent action.\n\n",
        table(["symptom", "count"], _top(symptom_counter)),
        "\n",
        section("Example Divergences"),
    ]
    if examples:
        lines.append(table(["frame_id", "diff_tags", "symptoms", "base_choice", "compare_choice"], examples))
    else:
        lines.append("No divergent examples.\n")
    lines.extend(
        [
            "\n",
            section("Limitations"),
            bullets(
                [
                    "Same-frame probing re-evaluates one recorded decision frame only.",
                    "After policies diverge, full-game counterfactual claims are polluted and not proven causal.",
                    "Symptom buckets are conservative diagnostic labels, not broad card semantics or replay learning.",
                ]
            ),
        ]
    )
    return "".join(lines)


def build_trace_report(*, run_id: str, frames: Iterable[dict[str, Any]], agents: list[str]) -> str:
    frame_list = list(frames)
    return "".join(
        [
            title("Trace Run Report"),
            f"- run_id: `{run_id}`\n",
            f"- agents: {', '.join(agents)}\n",
            f"- games: {len({frame.get('game_id') for frame in frame_list})}\n",
            f"- frames: {len(frame_list)}\n",
            "\n",
            "This is ordered decision-frame evidence, not replay learning.\n",
        ]
    )
