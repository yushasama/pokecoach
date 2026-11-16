"""Benchmark admitted E1 replay materialization through production encoding."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import psutil
import numpy as np
import torch

import run_enhanced_generation_smoke as smoke
from deepmew.rl.enhanced_replay import ReplayBatchMaterializer
from deepmew.rl.enhanced_replay_derived import DerivedReplayIndex


DEFAULT_CACHE = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_cache_v2\7459a801-77afa18a")
DEFAULT_SPLIT = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\SPLIT_MANIFEST.json")
DEFAULT_R7 = Path(r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt")
DEFAULT_V0 = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_runs\phasef4_grouped_crossfit_raw_s1_s915002_v3\full_train\value.pt")
DEFAULT_CALIBRATOR = ROOT / ".research/value_q_experiment/artifacts/phasef5_grouped_calibration_raw_s1_s915002_v3"
DEFAULT_DECK = "sha256:109a82f280159691594de831d3f3f507ea11d10d2bc8eaf54b23ba4ce1aee107"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))

    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-manifest", required=True)
    parser.add_argument("--derived-manifest")
    parser.add_argument("--binding-row-start", action="append", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-state", choices=("cold_process", "warm_process"), required=True)
    args = parser.parse_args()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    io_before = process.io_counters()
    tracemalloc.start()
    total_started = time.perf_counter()
    init_started = time.perf_counter()
    materializer = ReplayBatchMaterializer(
        DEFAULT_CACHE,
        split_manifest_path=DEFAULT_SPLIT,
        exact_deck_identity=DEFAULT_DECK,
        admission_manifest_path=args.admission_manifest,
    )
    derived = DerivedReplayIndex(args.derived_manifest) if args.derived_manifest else None
    v0 = smoke._load_v0(SimpleNamespace(
        v0_checkpoint=str(DEFAULT_V0),
        v0_checkpoint_sha256="cd1339e14fa943382069128bc9bf8657de9c441286a49bdef0f42450d56eb0a9",
        v0_calibrator_root=str(DEFAULT_CALIBRATOR),
        v0_calibrator_sha256="6a0868b1da31c76605620a74572f0621f74f3ce362ed04a833aae9c7f8304398",
    ))
    r7_payload = torch.load(DEFAULT_R7, map_location="cpu", weights_only=False)
    input_dim = int(r7_payload["input_dim"])
    init_seconds = time.perf_counter() - init_started
    starts = set(args.binding_row_start)
    bindings = [binding for binding in materializer.bindings if binding.row_start in starts]
    if {binding.row_start for binding in bindings} != starts:
        raise SystemExit("benchmark binding is missing")
    rows = []
    for binding in bindings:
        for global_row in range(binding.row_start, binding.row_end):
            if (
                materializer.data.meta["split_idx"][global_row] == materializer.train_index
                and materializer.data.meta["is_target_eligible"][global_row]
            ):
                rows.append(global_row)
    batch_latencies = []
    validation_seconds = 0.0
    tensorization_seconds = 0.0
    examples = []
    forced = 0
    cardinalities = {}
    first_identity = None
    final_identity = None
    load_started = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset:offset + args.batch_size]
        batch_started = time.perf_counter()
        validation_started = time.perf_counter()
        if derived is None:
            records = materializer.load(batch_rows)
            validation_seconds += time.perf_counter() - validation_started
            tensor_started = time.perf_counter()
            probabilities = np.asarray(
                v0.score(np.stack([np.array(record.s1, dtype=np.float32, copy=True) for record in records])),
                dtype=np.float64,
            )
            encoded = [
                smoke._encode_replay_record(
                    record,
                    v0=v0,
                    input_dim=input_dim,
                    v0_probability=float(probability),
                )
                for record, probability in zip(records, probabilities, strict=True)
            ]
            tensorization_seconds += time.perf_counter() - tensor_started
            record_identities = [record.decision_id for record in records]
        else:
            encoded = list(derived.load(batch_rows))
            validation_seconds += time.perf_counter() - validation_started
            records = None
            record_identities = [example.row_id for example in encoded]
        # Production trainer consumes the aligned example sequence directly.
        examples.extend(encoded)
        batch_latencies.append(time.perf_counter() - batch_started)
        for example in encoded:
            cardinality = int(example.legal_action_count or example.option_features.shape[0])
            cardinalities[str(cardinality)] = cardinalities.get(str(cardinality), 0) + 1
            forced += int(cardinality == 1)
        first_identity = first_identity or record_identities[0]
        final_identity = record_identities[-1]
    measured_seconds = time.perf_counter() - load_started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_after = process.cpu_times()
    io_after = process.io_counters()
    total_seconds = time.perf_counter() - total_started
    output = {
        "schema_version": "deepmew.e1_replay_benchmark.v1",
        "cache_state": args.cache_state,
        "admission_manifest": str(Path(args.admission_manifest).resolve()),
        "derived_manifest": str(Path(args.derived_manifest).resolve()) if args.derived_manifest else None,
        "binding_row_starts": sorted(starts),
        "source_files_crossed": len(bindings),
        "rows": len(rows),
        "unique_rows": len(set(rows)),
        "duplicates": len(rows) - len(set(rows)),
        "batch_size": args.batch_size,
        "batches": len(batch_latencies),
        "init_seconds": init_seconds,
        "measured_seconds": measured_seconds,
        "total_process_seconds": total_seconds,
        "rows_per_second": len(rows) / measured_seconds,
        "batches_per_second": len(batch_latencies) / measured_seconds,
        "p50_batch_seconds": statistics.median(batch_latencies),
        "p95_batch_seconds": _percentile(batch_latencies, 0.95),
        "validation_seconds": validation_seconds,
        "tensorization_seconds": tensorization_seconds,
        "forced_rows": forced,
        "nonforced_rows": len(rows) - forced,
        "legal_action_cardinality": dict(sorted(cardinalities.items(), key=lambda item: int(item[0]))),
        "first_row_identity": first_identity,
        "final_row_identity": final_identity,
        "peak_tracemalloc_bytes": peak,
        "cpu_user_seconds": cpu_after.user - cpu_before.user,
        "cpu_system_seconds": cpu_after.system - cpu_before.system,
        "bytes_read": io_after.read_bytes - io_before.read_bytes,
        "read_operations": io_after.read_count - io_before.read_count,
        "actor_weight_sum": float(len(rows) - forced),
        "encoded_examples": len(examples),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
