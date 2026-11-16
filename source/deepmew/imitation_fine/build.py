"""K4.1 fine-grained imitation builder + K5-lite filtered/reward-weighted BC.

Streams real ladder action rows, joins each decision to its deck's K2A archetype,
K3 split, and K3 deck-zoo bucket, recovers option/card/target identity via
``deepmew.options.identity``, derives deterministic intent / punishment / risk /
last-stand pseudo-labels, and exports labeled examples + summaries. It then builds
two K5-lite behavior-cloning tables (filtered, reward-weighted), compares them
against an unweighted context baseline on a held-out split, and reports which
better reproduces the real fine-grained choice. Offline/training-only: nothing
here is imported by the submission/eval path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_json, write_jsonl, write_markdown
from deepmew.imitation.schemas import option_type_name, select_context_name
from deepmew.imitation_fine import io as fio
from deepmew.imitation_fine.action_value import (
    build_action_value_table,
    evaluate_action_value,
)
from deepmew.imitation_fine.archetype_goals import goal_of_deck as goal_of_deck_for
from deepmew.imitation_fine.bc import (
    build_context_baseline_table,
    build_filtered_bc_table,
    build_reward_weighted_bc_table,
    evaluate_fine_bc,
)
from deepmew.imitation_fine.board import read_board
from deepmew.imitation_fine.labelers import label_decision
from deepmew.imitation_fine.report import render_k4_1_report, render_k5_lite_report
from deepmew.imitation_fine.schemas import (
    ARTIFACT_VERSION,
    IMITATION_FINE_KIND,
    MAIN_CONTEXT_ID,
    OUTPUT_FILES,
    PUNISHMENT_VOCAB,
    FineConfig,
    FineExample,
)
from deepmew.imitation_fine.teacher import is_trusted_teacher, reward_weight
from deepmew.meta_kg.io import json_cell, write_csv
from deepmew.options.identity import resolve_option_identity
from deepmew.rank_meta.loader import DEFAULT_SCOUT_ROOT, load_episode_seat_rank, load_team_rank_map
from deepmew.rank_meta.schemas import MISSING_RANK_INFO

LABEL_SUMMARY_FIELDS = ["dimension", "value", "n_examples"]
SINGLE_SUMMARY_FIELDS = ["label", "n_examples", "share"]
TEACHER_FIELDS = ["dimension", "value", "n_examples", "trusted", "trusted_rate", "avg_reward_weight"]
COMPARISON_FIELDS = ["context_scope", "method", "accuracy", "coverage", "n_test", "n_train"]
VALUE_EVAL_FIELDS = ["scope", "n_test", "coverage", "agree_rate", "agreed_win_rate", "disagreed_win_rate", "lift"]

K5_OUTPUT_FILES = {
    "filtered_bc_table": "filtered_bc_table.json",
    "reward_weighted_bc_table": "reward_weighted_bc_table.json",
    "context_baseline_table": "context_baseline_table.json",
    "action_value_table": "action_value_table.json",
    "method_comparison": "k5_lite_method_comparison.csv",
    "value_eval": "action_value_eval.csv",
    "k5_report": "k5_lite_report.md",
}

UNKNOWN_LABEL = {"archetype": "Unknown", "variant_label": "Unknown", "confidence": 0.0, "classification_status": "unknown"}


@dataclass
class FineResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]


def _confidence_band(confidence: float, cfg: FineConfig) -> str:
    if confidence >= cfg.confidence_high:
        return "high"
    if confidence >= cfg.confidence_low:
        return "medium"
    return "low"


def _build_examples(
    ingest_root: str | Path,
    label_map: dict[str, dict[str, Any]],
    split_map: dict[str, str],
    bucket_map: dict[str, str],
    episode_seat_decks: dict[str, dict[int, str]],
    episode_seat_rank: dict[tuple[str, int], Any],
    cfg: FineConfig,
) -> tuple[list[FineExample], dict[str, int]]:
    examples: list[FineExample] = []
    stats = {
        "rows_seen": 0, "skipped_no_seat": 0, "skipped_no_chosen": 0,
        "missing_label": 0, "missing_opponent": 0, "unresolved_identity": 0, "invalid_board": 0,
    }

    for row in fio.iter_action_rows(ingest_root):
        stats["rows_seen"] += 1
        seat = row.get("player_id")
        if seat not in (0, 1):
            stats["skipped_no_seat"] += 1
            continue
        state = row.get("state_json") or {}
        legal = (state.get("select") or {}).get("option") if isinstance(state, dict) else None
        legal = legal if isinstance(legal, list) else (row.get("legal_options_json") or [])
        chosen = [i for i in (row.get("chosen_indices") or []) if isinstance(i, int) and 0 <= i < len(legal)]
        if not chosen:
            stats["skipped_no_chosen"] += 1
            continue
        primary = legal[chosen[0]]
        if not isinstance(primary, dict):
            stats["skipped_no_chosen"] += 1
            continue

        option_type = option_type_name(primary.get("type"))
        identity = resolve_option_identity(state, primary)
        missing: list[str] = []
        if option_type in ("PLAY", "EVOLVE") and identity.card_id is None:
            missing.append("card_id")
        if option_type in ("ATTACH", "EVOLVE") and identity.target_area is None:
            missing.append("target")
        if not identity.resolved:
            stats["unresolved_identity"] += 1

        board = read_board(state)
        if not board.valid:
            stats["invalid_board"] += 1

        episode_id = str(row.get("game_id") or "")
        deck_hash = str(row.get("deck_hash") or "")
        seat_decks = episode_seat_decks.get(episode_id, {})
        opp_hash = seat_decks.get(1 - seat, "")
        if not opp_hash:
            stats["missing_opponent"] += 1
            missing.append("opponent")

        label = label_map.get(deck_hash)
        if label is None:
            stats["missing_label"] += 1
            label = UNKNOWN_LABEL
            missing.append("archetype")
        opp_label = label_map.get(opp_hash, UNKNOWN_LABEL)

        goal_deck, _gdc = goal_of_deck_for(label["archetype"])
        labels = label_decision(option_type, identity.card_id, identity.target_area or "", board, cfg)

        pilot_info = episode_seat_rank.get((episode_id, seat), MISSING_RANK_INFO)
        opp_info = episode_seat_rank.get((episode_id, 1 - seat), MISSING_RANK_INFO)
        if pilot_info.missing:
            missing.append("pilot_rank")

        context_id = (state.get("select") or {}).get("context") if isinstance(state, dict) else None
        result = row.get("result")
        result_int = result if isinstance(result, int) else -1
        ls = labels.last_stand

        ex = FineExample(
            episode_id=episode_id, seat=seat,
            turn=int(row.get("turn") or 0), step=int(row.get("step") or 0),
            deck_hash=deck_hash, archetype=label["archetype"], variant_label=label["variant_label"],
            archetype_confidence=float(label["confidence"]), classification_status=label["classification_status"],
            opponent_archetype=opp_label["archetype"],
            split=split_map.get(deck_hash, "train"), deck_zoo_bucket=bucket_map.get(deck_hash, ""),
            select_context_id=int(context_id) if isinstance(context_id, int) else -1,
            select_context=select_context_name(context_id),
            option_type=option_type, chosen_indices=list(chosen),
            chosen_option_identity=identity.reason,
            chosen_card_id=identity.card_id, chosen_card_name=identity.card_name or "",
            chosen_target_area=identity.target_area or "", chosen_target_slot=identity.target_index,
            chosen_target_card_id=identity.target_card_id, chosen_target_card_name=identity.target_card_name or "",
            goal_of_deck=goal_deck, goal_of_play=labels.goal_of_play,
            punishment_labels=labels.punishment_labels, risk_justification=labels.risk_justification,
            passive_line_loses=ls.passive_line_loses, last_stand_triggered=ls.last_stand_triggered,
            comeback_equity_line=ls.comeback_equity_line, slow_bleed_risk=ls.slow_bleed_risk,
            comeback_equity_score=ls.comeback_equity_score,
            result=result_int, won=(result_int == seat), is_draw=(result_int == 2),
            result_reason=str(row.get("result_reason") or ""),
            teacher_weight=0.0, label_confidence=labels.label_confidence,
            missing_fields=sorted(set(missing)),
            source_run_id=str(row.get("source_run_id") or ""),
            pilot_rank=pilot_info.rank, pilot_rank_bucket=pilot_info.bucket,
            pilot_rank_weight=pilot_info.teacher_factor, pilot_rank_missing=pilot_info.missing,
            pilot_team_id=pilot_info.team_id or "",
            opponent_rank=opp_info.rank, opponent_rank_bucket=opp_info.bucket,
            opponent_rank_missing=opp_info.missing,
        )
        ex.teacher_weight = reward_weight(ex, cfg)
        examples.append(ex)
    return examples, stats


def _label_summary(examples: list[FineExample], cfg: FineConfig) -> list[dict[str, Any]]:
    dims: dict[str, Counter] = {
        "split": Counter(), "deck_zoo_bucket": Counter(), "select_context": Counter(),
        "option_type": Counter(), "goal_of_deck": Counter(), "goal_of_play": Counter(),
        "risk_justification": Counter(), "classification_status": Counter(),
        "label_confidence_band": Counter(), "has_card_id": Counter(), "has_target": Counter(),
        "pilot_rank_bucket": Counter(), "opponent_rank_bucket": Counter(),
    }
    for ex in examples:
        dims["split"][ex.split] += 1
        dims["deck_zoo_bucket"][ex.deck_zoo_bucket or "(none)"] += 1
        dims["select_context"][ex.select_context] += 1
        dims["option_type"][ex.option_type] += 1
        dims["goal_of_deck"][ex.goal_of_deck] += 1
        dims["goal_of_play"][ex.goal_of_play] += 1
        dims["risk_justification"][ex.risk_justification or "(none)"] += 1
        dims["classification_status"][ex.classification_status] += 1
        dims["label_confidence_band"][_confidence_band(ex.label_confidence, cfg)] += 1
        dims["has_card_id"][str(ex.chosen_card_id is not None)] += 1
        dims["has_target"][str(bool(ex.chosen_target_area))] += 1
        dims["pilot_rank_bucket"][ex.pilot_rank_bucket] += 1
        dims["opponent_rank_bucket"][ex.opponent_rank_bucket] += 1
    rows: list[dict[str, Any]] = []
    for dimension, counter in dims.items():
        for value, n in counter.most_common():
            rows.append({"dimension": dimension, "value": value, "n_examples": n})
    return rows


def _single_summary(counter: Counter, total: int) -> list[dict[str, Any]]:
    return [
        {"label": label, "n_examples": n, "share": round(n / total, 4) if total else 0.0}
        for label, n in counter.most_common()
    ]


def _punishment_summary(examples: list[FineExample]) -> list[dict[str, Any]]:
    counter: Counter = Counter()
    rows_with_any = 0
    for ex in examples:
        if ex.punishment_labels:
            rows_with_any += 1
        for label in ex.punishment_labels:
            counter[label] += 1
    # Ensure every vocabulary label appears (0 is meaningful evidence).
    for label in PUNISHMENT_VOCAB:
        counter.setdefault(label, 0)
    total = len(examples)
    rows = [
        {"label": label, "n_examples": counter[label], "share": round(counter[label] / total, 4) if total else 0.0}
        for label in sorted(counter, key=lambda k: (-counter[k], k))
    ]
    rows.append({"label": "_rows_with_any_punishment", "n_examples": rows_with_any,
                 "share": round(rows_with_any / total, 4) if total else 0.0})
    return rows


def _last_stand_summary(examples: list[FineExample]) -> list[dict[str, Any]]:
    total = len(examples)
    flags = {
        "passive_line_loses": sum(1 for e in examples if e.passive_line_loses),
        "last_stand_triggered": sum(1 for e in examples if e.last_stand_triggered),
        "comeback_equity_line": sum(1 for e in examples if e.comeback_equity_line),
        "slow_bleed_risk": sum(1 for e in examples if e.slow_bleed_risk),
    }
    rows = [
        {"label": k, "n_examples": v, "share": round(v / total, 4) if total else 0.0}
        for k, v in sorted(flags.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    triggered = [e for e in examples if e.last_stand_triggered]
    avg_equity = round(sum(e.comeback_equity_score for e in triggered) / len(triggered), 4) if triggered else 0.0
    rows.append({"label": "_avg_comeback_equity_when_triggered", "n_examples": len(triggered), "share": avg_equity})
    return rows


def _teacher_summary(examples: list[FineExample], cfg: FineConfig) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[FineExample]] = defaultdict(list)
    for ex in examples:
        groups[("split", ex.split)].append(ex)
        groups[("goal_of_deck", ex.goal_of_deck)].append(ex)
        groups[("classification_status", ex.classification_status)].append(ex)
        groups[("pilot_rank_bucket", ex.pilot_rank_bucket)].append(ex)
    rows: list[dict[str, Any]] = []
    for (dimension, value), group in groups.items():
        trusted = sum(1 for e in group if is_trusted_teacher(e, cfg))
        avg_w = round(sum(e.teacher_weight for e in group) / len(group), 4) if group else 0.0
        rows.append({
            "dimension": dimension, "value": value, "n_examples": len(group),
            "trusted": trusted, "trusted_rate": round(trusted / len(group), 4) if group else 0.0,
            "avg_reward_weight": avg_w,
        })
    rows.sort(key=lambda r: (r["dimension"], -r["n_examples"], str(r["value"])))
    return rows


def _comparison(examples: list[FineExample], cfg: FineConfig) -> list[dict[str, Any]]:
    """Held-out top-1 fine-target accuracy for each method, at ALL and MAIN scopes."""
    rows: list[dict[str, Any]] = []
    scopes = {
        "ALL": examples,
        "MAIN": [e for e in examples if e.select_context_id == MAIN_CONTEXT_ID],
    }
    for scope_name, scope in scopes.items():
        train = [e for e in scope if e.split == "train"]
        test = [e for e in scope if e.split in ("eval", "holdout")]
        if not (train and test):
            continue
        tables = {
            "context_baseline": build_context_baseline_table(train, cfg, split=None),
            "filtered_bc": build_filtered_bc_table(train, cfg, split=None),
            "reward_weighted_bc": build_reward_weighted_bc_table(train, cfg, split=None),
        }
        for table in tables.values():
            ev = evaluate_fine_bc(table, test)
            rows.append({
                "context_scope": scope_name, "method": ev.method, "accuracy": ev.accuracy,
                "coverage": ev.coverage, "n_test": ev.n_test, "n_train": ev.n_train,
            })
    return rows


def build_imitation_fine(
    meta_kg_dir: str | Path,
    deck_zoo_dir: str | Path,
    ingest_root: str | Path = fio.DEFAULT_INGEST_ROOT,
    opponent_decks_root: str | Path = fio.DEFAULT_OPPONENT_DECKS,
    scout_root: str | Path = DEFAULT_SCOUT_ROOT,
    out_root: str | Path = ".research/runs",
    config: FineConfig | None = None,
    *,
    command: str = "",
    run_id: str | None = None,
    suffix: str | None = None,
) -> FineResult:
    cfg = config or FineConfig()
    label_map = fio.load_label_map(meta_kg_dir)
    split_map = fio.load_split_map(deck_zoo_dir)
    bucket_map = fio.load_bucket_map(deck_zoo_dir)
    episode_seat_decks = fio.load_episode_seat_decks(opponent_decks_root)
    # K1.7 rank join: team_name -> leaderboard rank -> (episode, seat) -> RankInfo.
    # Best-effort: missing scout data leaves every row rank-missing (back-compat).
    team_rank_map, _rank_warnings = load_team_rank_map(scout_root)
    episode_seat_rank = load_episode_seat_rank(team_rank_map, opponent_decks_root)

    examples, build_stats = _build_examples(
        ingest_root, label_map, split_map, bucket_map, episode_seat_decks, episode_seat_rank, cfg
    )
    if not examples:
        raise ValueError(f"no fine imitation examples produced from {ingest_root}")

    total = len(examples)
    label_summary = _label_summary(examples, cfg)
    goal_deck_summary = _single_summary(Counter(e.goal_of_deck for e in examples), total)
    goal_play_summary = _single_summary(Counter(e.goal_of_play for e in examples), total)
    punishment_summary = _punishment_summary(examples)
    risk_summary = _single_summary(Counter(e.risk_justification or "(none)" for e in examples), total)
    last_stand_summary = _last_stand_summary(examples)
    teacher_summary = _teacher_summary(examples, cfg)
    comparison = _comparison(examples, cfg)

    # K5-lite tables (trained on train split; serialized for inspection/eval reuse).
    filtered_table = build_filtered_bc_table(examples, cfg, split="train")
    reward_table = build_reward_weighted_bc_table(examples, cfg, split="train")
    baseline_table = build_context_baseline_table(examples, cfg, split="train")

    # K5 conservative offline action-value scorer (not BC argmax): trained on train,
    # directional offline eval on the held-out split at ALL and MAIN scopes.
    value_table = build_action_value_table(examples, split="train")
    value_eval_rows: list[dict[str, Any]] = []
    for scope_name, scope in (("ALL", examples), ("MAIN", [e for e in examples if e.select_context_id == MAIN_CONTEXT_ID])):
        test = [e for e in scope if e.split in ("eval", "holdout")]
        if test:
            ev = evaluate_action_value(value_table, test, scope=scope_name)
            value_eval_rows.append({
                "scope": ev.scope, "n_test": ev.n_test, "coverage": ev.coverage,
                "agree_rate": ev.agree_rate, "agreed_win_rate": ev.agreed_win_rate,
                "disagreed_win_rate": ev.disagreed_win_rate, "lift": ev.lift,
            })

    run_dir = create_run_dir(kind=IMITATION_FINE_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    out: dict[str, str] = {}
    out["examples"] = str(write_jsonl(run_dir.output_path(OUTPUT_FILES["examples"]), (e.to_dict() for e in examples)))
    out["label_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["label_summary"]), LABEL_SUMMARY_FIELDS, label_summary))
    out["goal_of_deck_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["goal_of_deck_summary"]), SINGLE_SUMMARY_FIELDS, goal_deck_summary))
    out["goal_of_play_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["goal_of_play_summary"]), SINGLE_SUMMARY_FIELDS, goal_play_summary))
    out["punishment_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["punishment_summary"]), SINGLE_SUMMARY_FIELDS, punishment_summary))
    out["risk_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["risk_summary"]), SINGLE_SUMMARY_FIELDS, risk_summary))
    out["last_stand_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["last_stand_summary"]), SINGLE_SUMMARY_FIELDS, last_stand_summary))
    out["teacher_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["teacher_summary"]), TEACHER_FIELDS, teacher_summary))
    out["filtered_bc_table"] = str(write_json(run_dir.output_path(K5_OUTPUT_FILES["filtered_bc_table"]), filtered_table.to_dict()))
    out["reward_weighted_bc_table"] = str(write_json(run_dir.output_path(K5_OUTPUT_FILES["reward_weighted_bc_table"]), reward_table.to_dict()))
    out["context_baseline_table"] = str(write_json(run_dir.output_path(K5_OUTPUT_FILES["context_baseline_table"]), baseline_table.to_dict()))
    out["action_value_table"] = str(write_json(run_dir.output_path(K5_OUTPUT_FILES["action_value_table"]), value_table.to_dict()))
    out["method_comparison"] = str(write_csv(run_dir.output_path(K5_OUTPUT_FILES["method_comparison"]), COMPARISON_FIELDS, comparison))
    out["value_eval"] = str(write_csv(run_dir.output_path(K5_OUTPUT_FILES["value_eval"]), VALUE_EVAL_FIELDS, value_eval_rows))

    counts = _build_counts(examples, build_stats, comparison, filtered_table, reward_table, value_table, value_eval_rows)
    ctx = {
        "built_at_utc": _utc(), "artifact_version": ARTIFACT_VERSION,
        "meta_kg_dir": str(meta_kg_dir), "deck_zoo_dir": str(deck_zoo_dir),
        "output_dir": str(run_dir.path), "counts": counts,
        "label_summary": label_summary, "goal_deck_summary": goal_deck_summary,
        "goal_play_summary": goal_play_summary, "punishment_summary": punishment_summary,
        "risk_summary": risk_summary, "last_stand_summary": last_stand_summary,
        "teacher_summary": teacher_summary, "comparison": comparison,
        "value_eval": value_eval_rows, "output_paths": out,
    }
    out["report"] = str(write_markdown(run_dir.output_path(OUTPUT_FILES["report"]), render_k4_1_report(ctx)))
    out["k5_report"] = str(write_markdown(run_dir.output_path(K5_OUTPUT_FILES["k5_report"]), render_k5_lite_report(ctx)))

    manifest = build_manifest(
        run_id=run_dir.run_id, kind=IMITATION_FINE_KIND, command=command,
        inputs={"meta_kg_dir": str(meta_kg_dir), "deck_zoo_dir": str(deck_zoo_dir), "ingest_root": str(ingest_root)},
        outputs=out, summary={"artifact_version": ARTIFACT_VERSION, **counts},
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return FineResult(run_dir.path, manifest_path, out, counts)


def _build_counts(examples, build_stats, comparison, filtered_table, reward_table, value_table, value_eval_rows) -> dict[str, Any]:
    main = [e for e in examples if e.select_context_id == MAIN_CONTEXT_ID]
    comp = {(r["context_scope"], r["method"]): r["accuracy"] for r in comparison}
    value_eval = {r["scope"]: r for r in value_eval_rows}
    rank_known = sum(1 for e in examples if not e.pilot_rank_missing)
    return {
        "example_count": len(examples),
        "main_context_count": len(main),
        "episode_count": len({e.episode_id for e in examples}),
        "deck_count": len({e.deck_hash for e in examples}),
        "archetype_count": len({e.archetype for e in examples}),
        "labeled_examples": sum(1 for e in examples if e.classification_status != "unknown"),
        "pilot_rank_known_examples": rank_known,
        "pilot_rank_known_share": round(rank_known / len(examples), 4) if examples else 0.0,
        "pilot_rank_top10_examples": sum(1 for e in examples if e.pilot_rank_bucket == "top10"),
        "opponent_rank_known_examples": sum(1 for e in examples if not e.opponent_rank_missing),
        "rows_with_card_id": sum(1 for e in examples if e.chosen_card_id is not None),
        "rows_with_target": sum(1 for e in examples if e.chosen_target_area),
        "rows_with_punishment": sum(1 for e in examples if e.punishment_labels),
        "rows_with_risk": sum(1 for e in examples if e.risk_justification),
        "last_stand_triggered": sum(1 for e in examples if e.last_stand_triggered),
        "missing_label_rows": build_stats["missing_label"],
        "missing_opponent_rows": build_stats["missing_opponent"],
        "unresolved_identity_rows": build_stats["unresolved_identity"],
        "filtered_bc_train_examples": filtered_table.train_examples,
        "reward_weighted_bc_train_examples": reward_table.train_examples,
        "acc_main_context_baseline": comp.get(("MAIN", "context_baseline")),
        "acc_main_filtered_bc": comp.get(("MAIN", "filtered_bc")),
        "acc_main_reward_weighted_bc": comp.get(("MAIN", "reward_weighted_bc")),
        "acc_all_context_baseline": comp.get(("ALL", "context_baseline")),
        "acc_all_filtered_bc": comp.get(("ALL", "filtered_bc")),
        "acc_all_reward_weighted_bc": comp.get(("ALL", "reward_weighted_bc")),
        "value_table_train_examples": value_table.train_examples,
        "value_eval_main_lift": value_eval.get("MAIN", {}).get("lift"),
        "value_eval_main_agreed_win_rate": value_eval.get("MAIN", {}).get("agreed_win_rate"),
        "value_eval_main_disagreed_win_rate": value_eval.get("MAIN", {}).get("disagreed_win_rate"),
        "value_eval_all_lift": value_eval.get("ALL", {}).get("lift"),
    }


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build K4.1 fine labels + K5-lite filtered/reward-weighted BC.")
    parser.add_argument("--meta-kg", default=None, help="K2A run dir (default: latest)")
    parser.add_argument("--deck-zoo", default=None, help="K3 deck zoo run dir (default: latest)")
    parser.add_argument("--ingest-root", default=fio.DEFAULT_INGEST_ROOT)
    parser.add_argument("--opponent-decks", default=fio.DEFAULT_OPPONENT_DECKS)
    parser.add_argument("--scout-root", default=DEFAULT_SCOUT_ROOT, help="scout root for the K1.7 rank join")
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    meta_kg_dir = args.meta_kg or fio.find_latest_meta_kg()
    deck_zoo_dir = args.deck_zoo or fio.find_latest_deck_zoo()
    result = build_imitation_fine(
        meta_kg_dir=meta_kg_dir, deck_zoo_dir=deck_zoo_dir,
        ingest_root=args.ingest_root, opponent_decks_root=args.opponent_decks,
        scout_root=args.scout_root,
        out_root=args.out_root, command="python -m deepmew.imitation_fine.build " + " ".join(argv or []),
        suffix=args.suffix,
    )
    print(f"imitation_fine build complete: {result.run_dir}")
    print(f"  meta_kg: {meta_kg_dir}")
    print(f"  deck_zoo: {deck_zoo_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
