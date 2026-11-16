from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deepmew.agents import canonical_agent_label, is_experimental, parse_agent_spec
from deepmew.arena.models import AgentSpec, GameRecord
from deepmew.arena.runner import run_game
from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import RunDirectory, create_run_dir
from deepmew.artifacts.writers import write_json, write_markdown, write_text
from deepmew.deck.io import read_deck
from deepmew.opponents.mixture import OpponentMixture
from deepmew.opponents.registry import get_mixture
from deepmew.opponents.sampler import MixtureSampler, SampledOpponent


RUN_KIND = "opponent_mixtures"
BUNDLE_KIND = "babymew_opponent_mixture_review"


@dataclass(frozen=True)
class OpponentMixtureRunResult:
    run_dir: Path
    manifest_path: Path
    report_path: Path
    mixture_summary_path: Path
    matchups_path: Path
    results_path: Path
    bundle_dir: Path | None
    summary: dict[str, Any]


def deck_metadata(path: str | Path = "deck.csv") -> dict[str, Any]:
    deck_path = Path(path)
    try:
        card_ids = read_deck(deck_path)
    except FileNotFoundError:
        return {"deck_path": str(deck_path), "deck_id": None, "deck_hash": None, "card_count": None}
    normalized = "\n".join(str(card_id) for card_id in card_ids) + "\n"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {
        "deck_path": str(deck_path),
        "deck_id": deck_path.stem,
        "deck_hash": digest[:16],
        "deck_hash_sha256": digest,
        "card_count": len(card_ids),
    }


def _game_row(
    *,
    agent: AgentSpec,
    mixture: OpponentMixture,
    sample: SampledOpponent,
    record: GameRecord,
    deck: dict[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    agent_win = record.result is not None and record.result != 2 and record.result == record.seat_a
    opponent_win = record.result is not None and record.result != 2 and record.result != record.seat_a
    agent_policy = agent.policy_spec
    opponent_policy = sample.agent_spec.policy_spec
    opponent_deck_id = sample.deck_id or deck.get("deck_id")
    opponent_deck_hash = sample.deck_hash or deck.get("deck_hash")
    return {
        "mixture_id": mixture.mixture_id,
        "mixture_name": mixture.name,
        "mixture_version": mixture.version,
        "run_seed": seed,
        "game_index": record.game_index,
        "agent_spec": agent.label,
        "agent_canonical_label": canonical_agent_label(agent),
        "agent_policy_id": agent_policy.policy_id,
        "agent_policy_options": _policy_options_json(agent),
        "opponent_spec": sample.agent_spec.label,
        "opponent_canonical_label": sample.canonical_label,
        "opponent_policy_id": opponent_policy.policy_id,
        "opponent_policy_options": _policy_options_json(sample.agent_spec),
        "opponent_entry_index": sample.index,
        "opponent_weight": sample.weight,
        "opponent_normalized_weight": sample.normalized_weight,
        "agent_deck_id": deck.get("deck_id"),
        "agent_deck_hash": deck.get("deck_hash"),
        "opponent_deck_id": opponent_deck_id,
        "opponent_deck_hash": opponent_deck_hash,
        "agent_seat": record.seat_a,
        "result": record.result,
        "agent_win": agent_win,
        "opponent_win": opponent_win,
        "draw": record.result == 2,
        "crashed": record.crashed,
        "timed_out": record.timed_out,
        "turns": record.turns,
        "elapsed_seconds": record.elapsed_seconds,
        "error": record.error,
    }


def _policy_options_json(spec: AgentSpec) -> list[dict[str, str | int]]:
    return [
        {"key": key, "value": value}
        for key, value in spec.policy_spec.options
    ]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _summarize_rows(rows: list[dict[str, Any]], *, games: int, seed: int | None) -> dict[str, Any]:
    completed = sum(1 for row in rows if row["result"] is not None and not row["crashed"] and not row["timed_out"])
    agent_wins = sum(1 for row in rows if row["agent_win"])
    opponent_wins = sum(1 for row in rows if row["opponent_win"])
    draws = sum(1 for row in rows if row["draw"])
    crashes = sum(1 for row in rows if row["crashed"])
    timeouts = sum(1 for row in rows if row["timed_out"])
    sampled_counts = Counter(str(row["opponent_canonical_label"]) for row in rows)
    by_opponent: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opponent_canonical_label"])].append(row)
    for opponent, opponent_rows in sorted(grouped.items()):
        opp_completed = sum(1 for row in opponent_rows if row["result"] is not None and not row["crashed"] and not row["timed_out"])
        by_opponent[opponent] = {
            "games": len(opponent_rows),
            "completed": opp_completed,
            "agent_wins": sum(1 for row in opponent_rows if row["agent_win"]),
            "opponent_wins": sum(1 for row in opponent_rows if row["opponent_win"]),
            "draws": sum(1 for row in opponent_rows if row["draw"]),
            "crashes": sum(1 for row in opponent_rows if row["crashed"]),
            "timeouts": sum(1 for row in opponent_rows if row["timed_out"]),
        }
    return {
        "games": games,
        "seed": seed,
        "completed": completed,
        "agent_wins": agent_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "crashes": crashes,
        "timeouts": timeouts,
        "point_rate": (agent_wins + draws * 0.5) / completed if completed else 0.0,
        "sampled_opponent_counts": dict(sampled_counts),
        "by_opponent": by_opponent,
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "mixture_id",
        "mixture_name",
        "mixture_version",
        "run_seed",
        "game_index",
        "agent_spec",
        "agent_canonical_label",
        "agent_policy_id",
        "agent_policy_options",
        "opponent_spec",
        "opponent_canonical_label",
        "opponent_policy_id",
        "opponent_policy_options",
        "opponent_entry_index",
        "opponent_weight",
        "opponent_normalized_weight",
        "agent_deck_id",
        "agent_deck_hash",
        "opponent_deck_id",
        "opponent_deck_hash",
        "agent_seat",
        "result",
        "agent_win",
        "opponent_win",
        "draw",
        "crashed",
        "timed_out",
        "turns",
        "elapsed_seconds",
        "error",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return stream.getvalue()


def format_mixture_report(
    *,
    agent: AgentSpec,
    mixture: OpponentMixture,
    summary: dict[str, Any],
    deck: dict[str, Any],
) -> str:
    lines = [
        f"# Opponent Mixture Run: {agent.label} vs {mixture.name}",
        "",
        f"mixture_id={mixture.mixture_id} version={mixture.version} seed={summary['seed']} games={summary['games']}",
        f"agent_label={agent.label} agent_canonical_label={canonical_agent_label(agent)}",
        f"deck_id={deck.get('deck_id')} deck_hash={deck.get('deck_hash')}",
        "",
        "## Result Summary",
        "",
        (
            f"completed={summary['completed']} agent_wins={summary['agent_wins']} "
            f"opponent_wins={summary['opponent_wins']} draws={summary['draws']} "
            f"crashes={summary['crashes']} timeouts={summary['timeouts']} "
            f"point_rate={summary['point_rate']:.3f}"
        ),
        "",
        "## Mixture Entries",
        "",
        "| opponent | weight | normalized | diagnostic |",
        "|---|---:|---:|---|",
    ]
    for entry in mixture.to_manifest()["entries"]:
        lines.append(
            f"| {entry['canonical_label']} | {entry['weight']} | "
            f"{entry['normalized_weight']:.3f} | {mixture.diagnostic_only} |"
        )
    lines.extend(["", "## Sampled Matchups", "", "| opponent | games | agent_wins | opponent_wins | draws | crashes | timeouts |", "|---|---:|---:|---:|---:|---:|---:|"])
    for opponent, row in summary["by_opponent"].items():
        lines.append(
            f"| {opponent} | {row['games']} | {row['agent_wins']} | {row['opponent_wins']} | "
            f"{row['draws']} | {row['crashes']} | {row['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is fixed local mixture infrastructure, not replay learning, value modeling, RL, or Deep RL.",
            "- Local mixtures are stress tests and future training infrastructure, not solved metas or ladder truth.",
            "- No gameplay default, deck, `cg/`, or root `main.py` behavior is changed by this run.",
        ]
    )
    return "\n".join(lines)


def build_mixture_run_manifest(
    *,
    run_id: str,
    command: str,
    agent: AgentSpec,
    mixture: OpponentMixture,
    games: int,
    seed: int | None,
    deck: dict[str, Any],
    outputs: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return build_manifest(
        run_id=run_id,
        kind=RUN_KIND,
        command=command,
        inputs={
            "agent": {
                "agent_spec": agent.label,
                "agent_raw": agent.raw,
                "canonical_label": canonical_agent_label(agent),
            },
            "mixture": mixture.to_manifest(),
            "games": games,
            "seed": seed,
            "deck": deck,
        },
        outputs=outputs,
        summary=summary,
    )


def _bundle_readme(run_dir: RunDirectory) -> str:
    return "\n".join(
        [
            "# BabyMew Opponent Mixture Review",
            "",
            "Compact review bundle for a fixed opponent-mixture arena run.",
            "",
            f"- source run: `{run_dir.path}`",
            "- inspect first: `report.md`, then `mixture_summary.json`, `matchups.csv`, and `results.json`.",
            "- bundle kind: `babymew_opponent_mixture_review`.",
            "",
        ]
    )


def _relevant_paths(run_dir: RunDirectory, bundle_dir: Path) -> str:
    paths = [
        run_dir.path,
        run_dir.output_path("manifest.json"),
        run_dir.output_path("report.md"),
        run_dir.output_path("mixture_summary.json"),
        run_dir.output_path("matchups.csv"),
        run_dir.output_path("results.json"),
        bundle_dir,
        Path("deepmew/agents/"),
        Path("deepmew/opponents/"),
        Path("deepmew/arena/specs.py"),
        Path("scripts/run_opponent_mixture.py"),
        Path("tests/test_agent_registry.py"),
        Path("tests/test_opponent_mixtures.py"),
        Path("PTCG_Bible.md"),
        Path(".research/BIBLE_COMPANION.md"),
        Path(".research/STATE.md"),
        Path(".research/CURRENT_STATE_COMPACT.md"),
        Path("HANDOFF_SUMMARY.md"),
    ]
    return "\n".join(str(path) for path in paths) + "\n"


def _write_bundle(
    *,
    run_dir: RunDirectory,
    report: str,
    mixture_summary: dict[str, Any],
    results: dict[str, Any],
    matchups_csv: str,
    bundle_root: str | Path,
    command: str,
) -> Path:
    bundle_run_id = run_dir.run_id.replace(RUN_KIND, BUNDLE_KIND, 1)
    bundle_dir = Path(bundle_root) / run_dir.day / bundle_run_id
    if bundle_dir.exists():
        raise FileExistsError(f"bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)

    outputs = {
        "readme": str(bundle_dir / "README.md"),
        "report": str(bundle_dir / "report.md"),
        "mixture_summary": str(bundle_dir / "mixture_summary.json"),
        "matchups_csv": str(bundle_dir / "matchups.csv"),
        "results": str(bundle_dir / "results.json"),
        "relevant_paths": str(bundle_dir / "relevant_paths.txt"),
    }
    write_markdown(bundle_dir / "README.md", _bundle_readme(run_dir))
    write_markdown(bundle_dir / "report.md", report)
    write_json(bundle_dir / "mixture_summary.json", mixture_summary)
    write_text(bundle_dir / "matchups.csv", matchups_csv)
    write_json(bundle_dir / "results.json", results)
    write_text(bundle_dir / "relevant_paths.txt", _relevant_paths(run_dir, bundle_dir))
    manifest = build_manifest(
        run_id=bundle_dir.name,
        kind=BUNDLE_KIND,
        command=command,
        inputs={"source_run": str(run_dir.path), "source_run_id": run_dir.run_id},
        outputs=outputs,
        summary=results["summary"],
    )
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def run_opponent_mixture(
    *,
    agent: str,
    mixture_name: str,
    games: int,
    max_steps: int = 5000,
    seed: int | None = None,
    out_root: str | Path = ".research/runs",
    bundle_root: str | Path = ".research/handoff_bundles",
    write_bundle: bool = True,
    command: str | None = None,
) -> OpponentMixtureRunResult:
    if games <= 0:
        raise ValueError("games must be positive")
    agent_spec = parse_agent_spec(agent)
    mixture = get_mixture(mixture_name)
    deck = deck_metadata()
    run_dir = create_run_dir(
        kind=RUN_KIND,
        out_root=out_root,
        suffix=f"{canonical_agent_label(agent_spec)}__vs__{mixture.name}",
    )
    command_text = command or "python " + " ".join(sys.argv)

    sampler = MixtureSampler(mixture, seed=seed)
    samples = sampler.sample_sequence(games)
    rng = random.Random(None if seed is None else seed + 7919)
    records: list[GameRecord] = []
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        seat_a = (index - 1) % 2
        if seat_a == 0:
            agent0, agent1 = agent_spec, sample.agent_spec
        else:
            agent0, agent1 = sample.agent_spec, agent_spec
        record = run_game(
            agent0,
            agent1,
            game_index=index,
            seat_a=seat_a,
            max_steps=max_steps,
            rng=rng,
            run_seed=seed,
            game_id_prefix=f"mixture-{run_dir.run_id}",
        )
        records.append(record)
        rows.append(_game_row(agent=agent_spec, mixture=mixture, sample=sample, record=record, deck=deck, seed=seed))

    summary = _summarize_rows(rows, games=games, seed=seed)
    mixture_summary = {
        "mixture": mixture.to_manifest(),
        "sampled_sequence": [
            {
                "game_index": index,
                "opponent_entry_index": sample.index,
                "opponent_spec": sample.agent_spec.label,
                "canonical_label": sample.canonical_label,
                "experimental": is_experimental(sample.agent_spec),
                "deck_id": sample.deck_id or deck.get("deck_id"),
                "deck_hash": sample.deck_hash or deck.get("deck_hash"),
                "policy_id": sample.agent_spec.policy_spec.policy_id,
                "policy_options": _policy_options_json(sample.agent_spec),
                "weight": sample.weight,
                "normalized_weight": sample.normalized_weight,
            }
            for index, sample in enumerate(samples, start=1)
        ],
        "sampled_counts": summary["sampled_opponent_counts"],
    }
    results = {
        "agent": {
            "agent_spec": agent_spec.label,
            "canonical_label": canonical_agent_label(agent_spec),
        },
        "mixture_id": mixture.mixture_id,
        "mixture_version": mixture.version,
        "deck": deck,
        "summary": summary,
        "matchups": rows,
        "records": [asdict(record) for record in records],
    }
    report = format_mixture_report(agent=agent_spec, mixture=mixture, summary=summary, deck=deck)
    matchups_csv = _csv_text(rows)

    outputs = {
        "report": str(run_dir.output_path("report.md")),
        "mixture_summary": str(run_dir.output_path("mixture_summary.json")),
        "matchups_csv": str(run_dir.output_path("matchups.csv")),
        "results": str(run_dir.output_path("results.json")),
    }
    bundle_dir = None
    if write_bundle:
        bundle_dir = _write_bundle(
            run_dir=run_dir,
            report=report,
            mixture_summary=mixture_summary,
            results=results,
            matchups_csv=matchups_csv,
            bundle_root=bundle_root,
            command=command_text,
        )
        outputs["handoff_bundle"] = str(bundle_dir)

    write_markdown(run_dir.output_path("report.md"), report)
    write_json(run_dir.output_path("mixture_summary.json"), mixture_summary)
    write_text(run_dir.output_path("matchups.csv"), matchups_csv)
    write_json(run_dir.output_path("results.json"), results)
    manifest = build_mixture_run_manifest(
        run_id=run_dir.run_id,
        command=command_text,
        agent=agent_spec,
        mixture=mixture,
        games=games,
        seed=seed,
        deck=deck,
        outputs=outputs,
        summary=summary,
    )
    write_manifest(run_dir.manifest_path, manifest)
    return OpponentMixtureRunResult(
        run_dir=run_dir.path,
        manifest_path=run_dir.manifest_path,
        report_path=run_dir.output_path("report.md"),
        mixture_summary_path=run_dir.output_path("mixture_summary.json"),
        matchups_path=run_dir.output_path("matchups.csv"),
        results_path=run_dir.output_path("results.json"),
        bundle_dir=bundle_dir,
        summary=summary,
    )
