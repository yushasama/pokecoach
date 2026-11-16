"""K4 imitation dataset builder: action rows + K2A/K3 labels -> BC examples + eval.

Streams real ladder action rows, joins each decision to its deck's K2A archetype
and K3 split, exports behavior-cloning examples, summarizes buckets, fits a
count-table BC baseline, and reports held-out top-1 accuracy. Offline/training-
only: nothing here is imported by the submission/eval path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import create_run_dir
from deepmew.artifacts.writers import write_jsonl, write_markdown
from deepmew.deck_zoo.io import find_latest_meta_kg
from deepmew.imitation import io as imi_io
from deepmew.imitation.baseline import evaluate_baselines
from deepmew.imitation.report import render_report
from deepmew.imitation.schemas import (
    ARTIFACT_VERSION,
    IMITATION_KIND,
    ImitationConfig,
    ImitationExample,
    MAIN_CONTEXT_ID,
    OUTPUT_FILES,
    option_type_name,
    select_context_name,
)
from deepmew.meta_kg.io import json_cell, write_csv

DATASET_SUMMARY_FIELDS = ["dimension", "value", "n_examples"]
BUCKET_SUMMARY_FIELDS = [
    "bucket_type", "bucket_value", "n_examples", "n_main_context", "n_episodes",
    "n_decks", "decision_win_rate", "top_action_types",
]
BASELINE_FIELDS = ["model", "accuracy", "coverage", "n_test", "n_train", "context_scope"]


@dataclass
class ImitationResult:
    run_dir: Path
    manifest_path: Path
    output_paths: dict[str, str]
    counts: dict[str, Any]


def _confidence_bucket(confidence: float, cfg: ImitationConfig) -> str:
    if confidence >= cfg.confidence_high:
        return "high"
    if confidence >= cfg.confidence_low:
        return "medium"
    return "low"


def _build_examples(
    ingest_root: str | Path,
    label_map: dict[str, dict[str, Any]],
    split_map: dict[str, str],
    episode_seat_decks: dict[str, dict[int, str]],
    cfg: ImitationConfig,
) -> tuple[list[ImitationExample], dict[str, int]]:
    examples: list[ImitationExample] = []
    stats = {"rows_seen": 0, "skipped_no_seat": 0, "skipped_no_chosen": 0, "missing_label": 0, "missing_opponent": 0}
    unknown_label = {"archetype": "Unknown", "variant_label": "Unknown", "confidence": 0.0, "classification_status": "unknown"}

    for row in imi_io.iter_action_rows(ingest_root):
        stats["rows_seen"] += 1
        seat = row.get("player_id")
        if seat not in (0, 1):
            stats["skipped_no_seat"] += 1
            continue
        legal = row.get("legal_options_json") or []
        chosen = row.get("chosen_indices") or []
        chosen_types = [option_type_name(legal[i].get("type")) for i in chosen if isinstance(i, int) and 0 <= i < len(legal) and isinstance(legal[i], dict)]
        if not chosen_types:
            stats["skipped_no_chosen"] += 1
            continue

        episode_id = str(row.get("game_id") or "")
        deck_hash = str(row.get("deck_hash") or "")
        seat_decks = episode_seat_decks.get(episode_id, {})
        opponent_deck_hash = seat_decks.get(1 - seat, "")
        if not opponent_deck_hash:
            stats["missing_opponent"] += 1

        label = label_map.get(deck_hash)
        if label is None:
            stats["missing_label"] += 1
            label = unknown_label
        opp_label = label_map.get(opponent_deck_hash, unknown_label)

        state = row.get("state_json") or {}
        select = state.get("select") if isinstance(state, dict) else {}
        context_id = (select or {}).get("context")
        result = row.get("result")
        result_int = result if isinstance(result, int) else -1

        examples.append(
            ImitationExample(
                episode_id=episode_id,
                seat=seat,
                turn=int(row.get("turn") or 0),
                step=int(row.get("step") or 0),
                deck_hash=deck_hash,
                archetype=label["archetype"],
                variant_label=label["variant_label"],
                confidence=label["confidence"],
                confidence_bucket=_confidence_bucket(label["confidence"], cfg),
                classification_status=label["classification_status"],
                opponent_deck_hash=opponent_deck_hash,
                opponent_archetype=opp_label["archetype"],
                split=split_map.get(deck_hash, "train"),
                select_context_id=int(context_id) if isinstance(context_id, int) else -1,
                select_context=select_context_name(context_id),
                n_legal_options=len(legal),
                legal_option_types=sorted({option_type_name(o.get("type")) for o in legal if isinstance(o, dict)}),
                chosen_option_type=chosen_types[0],
                chosen_option_types=chosen_types,
                result=result_int,
                won=(result_int == seat),
                is_draw=(result_int == 2),
                source_run_id=str(row.get("source_run_id") or ""),
            )
        )
    return examples, stats


def _dataset_summary(examples: list[ImitationExample]) -> list[dict[str, Any]]:
    dims: dict[str, Counter] = {
        "split": Counter(), "select_context": Counter(), "chosen_option_type": Counter(),
        "archetype": Counter(), "opponent_archetype": Counter(), "confidence_bucket": Counter(),
        "classification_status": Counter(),
    }
    for ex in examples:
        dims["split"][ex.split] += 1
        dims["select_context"][ex.select_context] += 1
        dims["chosen_option_type"][ex.chosen_option_type] += 1
        dims["archetype"][ex.archetype] += 1
        dims["opponent_archetype"][ex.opponent_archetype] += 1
        dims["confidence_bucket"][ex.confidence_bucket] += 1
        dims["classification_status"][ex.classification_status] += 1
    rows: list[dict[str, Any]] = []
    for dimension, counter in dims.items():
        for value, n in counter.most_common():
            rows.append({"dimension": dimension, "value": value, "n_examples": n})
    return rows


def _bucket_summary(examples: list[ImitationExample]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[ImitationExample]] = defaultdict(list)
    for ex in examples:
        buckets[("our_archetype", ex.archetype)].append(ex)
        buckets[("opponent_archetype", ex.opponent_archetype)].append(ex)
        buckets[("matchup", f"{ex.archetype} vs {ex.opponent_archetype}")].append(ex)
        buckets[("split", ex.split)].append(ex)
        buckets[("confidence_bucket", ex.confidence_bucket)].append(ex)
    rows: list[dict[str, Any]] = []
    for (bucket_type, bucket_value), group in buckets.items():
        main = [e for e in group if e.select_context_id == MAIN_CONTEXT_ID]
        decided = [e for e in group if not e.is_draw and e.result >= 0]
        win_rate = round(sum(1 for e in decided if e.won) / len(decided), 4) if decided else 0.0
        top_actions = Counter(e.chosen_option_type for e in main).most_common(5)
        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket_value": bucket_value,
                "n_examples": len(group),
                "n_main_context": len(main),
                "n_episodes": len({e.episode_id for e in group}),
                "n_decks": len({e.deck_hash for e in group}),
                "decision_win_rate": win_rate,
                "top_action_types": json_cell(top_actions),
            }
        )
    rows.sort(key=lambda r: (r["bucket_type"], -r["n_examples"], str(r["bucket_value"])))
    return rows


def build_imitation_dataset(
    meta_kg_dir: str | Path,
    deck_zoo_dir: str | Path,
    ingest_root: str | Path = imi_io.DEFAULT_INGEST_ROOT,
    opponent_decks_root: str | Path = imi_io.DEFAULT_OPPONENT_DECKS,
    out_root: str | Path = ".research/runs",
    config: ImitationConfig | None = None,
    *,
    command: str = "",
    run_id: str | None = None,
    suffix: str | None = None,
) -> ImitationResult:
    cfg = config or ImitationConfig()
    label_map = imi_io.load_label_map(meta_kg_dir)
    split_map = imi_io.load_split_map(deck_zoo_dir)
    episode_seat_decks = imi_io.load_episode_seat_decks(opponent_decks_root)

    examples, build_stats = _build_examples(ingest_root, label_map, split_map, episode_seat_decks, cfg)
    if not examples:
        raise ValueError(f"no imitation examples produced from {ingest_root}")

    dataset_summary = _dataset_summary(examples)
    bucket_summary = _bucket_summary(examples)

    # Behavior-cloning baseline at two scopes: ALL contexts (where the select
    # context is highly predictive) and MAIN only (the strategic phase, where
    # archetype conditioning should matter most if it matters anywhere).
    main_examples = [e for e in examples if e.select_context_id == MAIN_CONTEXT_ID]
    scopes = {"ALL": examples, "MAIN": main_examples}
    baseline_rows: list[dict[str, Any]] = []
    train_n = test_n = 0
    for scope_name, scope_examples in scopes.items():
        train = [e for e in scope_examples if e.split == "train"]
        test = [e for e in scope_examples if e.split in ("eval", "holdout")]
        if scope_name == "MAIN":
            train_n, test_n = len(train), len(test)
        if not (train and test):
            continue
        for ev in evaluate_baselines(train, test, cfg):
            baseline_rows.append({
                "model": ev.name, "accuracy": ev.accuracy, "coverage": ev.coverage,
                "n_test": ev.n, "n_train": len(train), "context_scope": scope_name,
            })

    run_dir = create_run_dir(kind=IMITATION_KIND, out_root=out_root, run_id=run_id, suffix=suffix)
    output_paths: dict[str, str] = {}
    output_paths["examples"] = str(write_jsonl(run_dir.output_path(OUTPUT_FILES["examples"]), (e.to_dict() for e in examples)))
    output_paths["dataset_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["dataset_summary"]), DATASET_SUMMARY_FIELDS, dataset_summary))
    output_paths["bucket_summary"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["bucket_summary"]), BUCKET_SUMMARY_FIELDS, bucket_summary))
    output_paths["baseline_eval"] = str(write_csv(run_dir.output_path(OUTPUT_FILES["baseline_eval"]), BASELINE_FIELDS, baseline_rows))

    counts = _build_counts(examples, train_n, test_n, baseline_rows, build_stats)
    report_ctx = {
        "built_at_utc": _utc(),
        "artifact_version": ARTIFACT_VERSION,
        "meta_kg_dir": str(meta_kg_dir),
        "deck_zoo_dir": str(deck_zoo_dir),
        "output_dir": str(run_dir.path),
        "counts": counts,
        "dataset_summary": dataset_summary,
        "bucket_summary": bucket_summary,
        "baseline_rows": baseline_rows,
        "output_paths": output_paths,
    }
    output_paths["report"] = str(write_markdown(run_dir.output_path(OUTPUT_FILES["report"]), render_report(report_ctx)))

    manifest = build_manifest(
        run_id=run_dir.run_id, kind=IMITATION_KIND, command=command,
        inputs={"meta_kg_dir": str(meta_kg_dir), "deck_zoo_dir": str(deck_zoo_dir), "ingest_root": str(ingest_root)},
        outputs=output_paths,
        summary={"artifact_version": ARTIFACT_VERSION, **counts},
    )
    manifest_path = write_manifest(run_dir.manifest_path, manifest)
    return ImitationResult(run_dir.path, manifest_path, output_paths, counts)


def _build_counts(examples, train_n, test_n, baseline_rows, build_stats) -> dict[str, Any]:
    main = {r["model"]: r["accuracy"] for r in baseline_rows if r["context_scope"] == "MAIN"}
    all_ctx = {r["model"]: r["accuracy"] for r in baseline_rows if r["context_scope"] == "ALL"}
    return {
        "example_count": len(examples),
        "main_context_count": sum(1 for e in examples if e.select_context_id == MAIN_CONTEXT_ID),
        "episode_count": len({e.episode_id for e in examples}),
        "deck_count": len({e.deck_hash for e in examples}),
        "archetype_count": len({e.archetype for e in examples}),
        "labeled_examples": sum(1 for e in examples if e.classification_status != "unknown"),
        "missing_label_rows": build_stats["missing_label"],
        "missing_opponent_rows": build_stats["missing_opponent"],
        "baseline_main_train_n": train_n,
        "baseline_main_test_n": test_n,
        "all_acc_global": all_ctx.get("global_most_common"),
        "all_acc_by_context": all_ctx.get("by_context"),
        "all_acc_by_context_archetype": all_ctx.get("by_context_archetype"),
        "main_acc_global": main.get("global_most_common"),
        "main_acc_by_context_archetype": main.get("by_context_archetype"),
        "main_acc_by_context_archetype_opponent": main.get("by_context_archetype_opponent"),
    }


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the K4 imitation dataset + BC baseline eval.")
    parser.add_argument("--meta-kg", default=None, help="K2A run dir (default: latest)")
    parser.add_argument("--deck-zoo", default=None, help="K3 deck zoo run dir (default: latest)")
    parser.add_argument("--ingest-root", default=imi_io.DEFAULT_INGEST_ROOT)
    parser.add_argument("--opponent-decks", default=imi_io.DEFAULT_OPPONENT_DECKS)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args(argv)

    meta_kg_dir = args.meta_kg or find_latest_meta_kg()
    deck_zoo_dir = args.deck_zoo or _find_latest_deck_zoo()
    result = build_imitation_dataset(
        meta_kg_dir=meta_kg_dir, deck_zoo_dir=deck_zoo_dir,
        ingest_root=args.ingest_root, opponent_decks_root=args.opponent_decks,
        out_root=args.out_root, command="python -m deepmew.imitation.build " + " ".join(argv or []),
        suffix=args.suffix,
    )
    print(f"imitation dataset build complete: {result.run_dir}")
    print(f"  meta_kg: {meta_kg_dir}")
    print(f"  deck_zoo: {deck_zoo_dir}")
    for key, value in result.counts.items():
        print(f"  {key}: {value}")
    return 0


def _find_latest_deck_zoo(runs_root: str | Path = ".research/runs/deck_zoo") -> Path:
    root = Path(runs_root)
    runs = [c for c in sorted(root.glob("*/*")) if (c / "deck_zoo.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"no deck_zoo run with deck_zoo.csv under {root}")
    return runs[-1]


if __name__ == "__main__":
    raise SystemExit(main())
