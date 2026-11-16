"""Ingest a Kaggle episode JSON into DeepMew opponent-deck and replay artifacts.

Deck extraction is the primary, robust output. Replay action-row conversion is
best-effort: it maps a step to a `deepmew.replay_action_row.v1` row only when the
per-agent observation carries `select.option` and the agent's action is a list of
indices; steps that cannot be mapped are skipped and counted, never faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_jsonl, write_markdown
from deepmew.kaggle_replays.decks import extract_decklists, write_deck_artifacts
from deepmew.kaggle_replays.episode import EpisodeMetadata, episode_metadata, load_episode, result_from_rewards
from deepmew.replay.action_schema import SCHEMA_VERSION as ACTION_SCHEMA_VERSION
from deepmew.replay.action_schema import ReplayActionSchemaError, validate_action_row

KAGGLE_INGEST_KIND = "kaggle_ingest"
KAGGLE_SOURCE_KIND = "kaggle_replay"
ACTION_TABLE_FILENAME = "replay_action_table.jsonl"


@dataclass
class ActionRowStats:
    steps_total: int = 0
    select_decisions: int = 0
    rows: int = 0
    skipped_no_chosen_indices: int = 0
    malformed: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_total": self.steps_total,
            "select_decisions": self.select_decisions,
            "rows": self.rows,
            "skipped_no_chosen_indices": self.skipped_no_chosen_indices,
            "malformed": self.malformed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class KaggleIngestResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    deck_results: list[dict[str, Any]]
    action_table_path: Path | None
    action_stats: dict[str, Any] | None
    summary: dict[str, Any]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _chosen_indices(action: Any) -> list[int] | None:
    if not isinstance(action, list) or not action:
        return None
    out: list[int] = []
    for item in action:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        out.append(item)
    return out


def build_kaggle_action_rows(
    data: dict[str, Any],
    meta: EpisodeMetadata,
    decks_by_index: dict[int, dict[str, Any]],
    *,
    source_run_id: str,
    source_path: str,
) -> tuple[list[dict[str, Any]], ActionRowStats]:
    stats = ActionRowStats()
    steps = data.get("steps")
    if not isinstance(steps, list):
        stats.notes.append("episode has no 'steps' list; no action rows produced")
        return [], stats
    stats.steps_total = len(steps)

    result = result_from_rewards(meta.rewards)
    reason = data.get("info", {}).get("reason") if isinstance(data.get("info"), dict) else None
    rows: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        for seat, agent in enumerate(step):
            if not isinstance(agent, dict):
                continue
            obs = agent.get("observation")
            select = obs.get("select") if isinstance(obs, dict) else None
            if not (isinstance(select, dict) and isinstance(select.get("option"), list)):
                continue
            stats.select_decisions += 1
            chosen = _chosen_indices(agent.get("action"))
            if chosen is None:
                stats.skipped_no_chosen_indices += 1
                continue
            current = obs.get("current") if isinstance(obs.get("current"), dict) else {}
            player_id = _int_or_none(current.get("yourIndex"))
            if player_id is None:
                player_id = seat
            deck = decks_by_index.get(player_id, {})
            row = {
                "schema_version": ACTION_SCHEMA_VERSION,
                "source_kind": KAGGLE_SOURCE_KIND,
                "source_run_id": source_run_id,
                "source_path": source_path,
                "game_id": meta.episode_id,
                "player_id": player_id if player_id in (0, 1) else None,
                "turn": _int_or_none(current.get("turn")),
                "step": step_index,
                "frame_id": f"{meta.episode_id}:s{step_index:04d}:p{player_id}",
                "seed": meta.seed,
                "state_json": obs,
                "legal_options_json": select.get("option"),
                "chosen_indices": chosen,
                "result": result,
                "result_reason": reason if isinstance(reason, str) else None,
                "deck_id": deck.get("deck_id"),
                "deck_hash": deck.get("deck_hash"),
                "source_rating": None,
            }
            try:
                validate_action_row(row, index=stats.rows)
            except ReplayActionSchemaError as exc:
                stats.malformed += 1
                if len(stats.notes) < 10:
                    stats.notes.append(f"step {step_index} seat {seat}: {exc}")
                continue
            rows.append(row)
            stats.rows += 1
    return rows, stats


def _report(
    meta: EpisodeMetadata,
    deck_results: list[dict[str, Any]],
    action_stats: dict[str, Any] | None,
    *,
    source_path: str,
) -> str:
    lines = [
        "# Kaggle Replay Ingest",
        "",
        f"episode_id={meta.episode_id}",
        f"team_names={meta.team_names}",
        f"rewards={meta.rewards} statuses={meta.statuses}",
        f"seed={meta.seed} step_count={meta.step_count} agent_count={meta.agent_count}",
        f"source_episode={source_path}",
        "",
        "## Opponent decks",
        "",
        "| player | team | cards | standard_60 | deck_hash | warnings |",
        "|---:|---|---:|---|---|---|",
    ]
    for deck in deck_results:
        lines.append(
            f"| {deck['player_index']} | {deck['team_name']} | {deck['card_count']} | "
            f"{deck['is_standard_size']} | {deck['deck_hash']} | {deck['warnings'] or '-'} |"
        )
    for deck in deck_results:
        summary = deck.get("name_counts")
        if summary:
            lines += ["", f"### Player {deck['player_index']} / {deck['team_name']}", ""]
            for name, count in summary.items():
                lines.append(f"- {count} {name}")

    lines += ["", "## Replay action rows", ""]
    if action_stats is None:
        lines.append("- Action-row conversion was not run (`--extract-decks-only`).")
    else:
        lines.append(
            f"- steps={action_stats['steps_total']} select_decisions={action_stats['select_decisions']} "
            f"rows={action_stats['rows']} skipped_no_chosen_indices={action_stats['skipped_no_chosen_indices']} "
            f"malformed={action_stats['malformed']}"
        )
        if action_stats["rows"] == 0:
            lines.append(
                "- No `deepmew.replay_action_row.v1` rows were produced: the episode's per-agent "
                "observations did not expose `select.option` with a list action at decision time. "
                "Deck extraction still completed. Missing fields were not fabricated."
            )
        for note in action_stats.get("notes", [])[:5]:
            lines.append(f"  - {note}")

    lines += [
        "",
        "## Boundary",
        "",
        "- Infrastructure only: reads a downloaded Kaggle episode and writes opponent-deck + replay artifacts.",
        "- No gameplay/default `v2.4` change; root `main.py`/`deck.csv`/`cg/` untouched; never loaded by the eval path.",
    ]
    return "\n".join(lines)


def ingest_episode(
    episode_json: str | Path,
    *,
    extract_decks_only: bool = False,
    out_root: str | Path = ".research",
    opponent_decks_root: str | Path = ".research/opponent_decks",
    command: str | None = None,
) -> KaggleIngestResult:
    episode_path = str(Path(episode_json))
    data = load_episode(episode_json)

    decks = extract_decklists(data)
    meta = episode_metadata(
        data,
        path=episode_json,
        name_fallback=[deck.team_name for deck in decks] if decks else None,
    )

    run_dir = create_run_dir(kind=KAGGLE_INGEST_KIND, out_root=out_root, suffix=meta.episode_id)

    deck_results: list[dict[str, Any]] = []
    decks_by_index: dict[int, dict[str, Any]] = {}
    for deck in decks:
        written = write_deck_artifacts(
            deck,
            episode_id=meta.episode_id,
            source_episode_path=episode_path,
            out_root=opponent_decks_root,
            command=command,
        )
        written["name_counts"] = deck.name_counts
        deck_results.append(written)
        decks_by_index[deck.player_index] = written

    action_table_path: Path | None = None
    action_stats: dict[str, Any] | None = None
    if not extract_decks_only:
        rows, stats = build_kaggle_action_rows(
            data, meta, decks_by_index, source_run_id=run_dir.run_id, source_path=episode_path
        )
        action_stats = stats.to_dict()
        if rows:
            action_table_path = write_jsonl(run_dir.output_path(ACTION_TABLE_FILENAME), rows)

    report = _report(meta, deck_results, action_stats, source_path=episode_path)
    report_path = write_markdown(run_dir.output_path("report.md"), report)

    summary = {
        "episode_id": meta.episode_id,
        "team_names": meta.team_names,
        "rewards": meta.rewards,
        "statuses": meta.statuses,
        "seed": meta.seed,
        "step_count": meta.step_count,
        "agent_count": meta.agent_count,
        "result": result_from_rewards(meta.rewards),
        "decks": [
            {key: deck[key] for key in ("player_index", "team_name", "deck_id", "deck_hash", "card_count", "is_standard_size", "warnings")}
            for deck in deck_results
        ],
        "action_rows": action_stats,
    }
    outputs = {"report": str(report_path)}
    for deck in deck_results:
        outputs[f"deck_p{deck['player_index']}"] = deck["out_dir"]
    if action_table_path is not None:
        outputs["replay_action_table"] = str(action_table_path)

    manifest = build_manifest(
        run_id=run_dir.run_id,
        kind=KAGGLE_INGEST_KIND,
        command=command or "",
        inputs={
            "episode_json": episode_path,
            "episode_id": meta.episode_id,
            "extract_decks_only": extract_decks_only,
            "seed": meta.seed,
        },
        outputs=outputs,
        summary=summary,
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)

    return KaggleIngestResult(
        run_dir=run_dir.path,
        manifest_path=manifest_path,
        report_path=report_path,
        deck_results=deck_results,
        action_table_path=action_table_path,
        action_stats=action_stats,
        summary=summary,
    )
