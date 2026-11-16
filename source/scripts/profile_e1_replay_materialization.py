"""Profile the real E1 replay materializer on declared accepted rows."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
import tracemalloc
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.rl.enhanced_replay import ReplayBatchMaterializer


DEFAULT_CACHE = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_cache_v2\7459a801-77afa18a")
DEFAULT_SPLIT = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\SPLIT_MANIFEST.json")
DEFAULT_DECK = "sha256:109a82f280159691594de831d3f3f507ea11d10d2bc8eaf54b23ba4ce1aee107"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", nargs="+", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_path = output.with_suffix(".prof")
    profiler = cProfile.Profile()
    tracemalloc.start()
    started = time.perf_counter()
    profiler.enable()
    init_started = time.perf_counter()
    materializer = ReplayBatchMaterializer(
        DEFAULT_CACHE,
        split_manifest_path=DEFAULT_SPLIT,
        exact_deck_identity=DEFAULT_DECK,
    )
    init_seconds = time.perf_counter() - init_started
    load_started = time.perf_counter()
    records = materializer.load(args.rows)
    load_seconds = time.perf_counter() - load_started
    profiler.disable()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler)
    top = []
    for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:30]:
        filename, line, function = key
        call_count, primitive_calls, total_time, cumulative_time, _ = value
        top.append({
            "file": filename,
            "line": line,
            "function": function,
            "calls": call_count,
            "primitive_calls": primitive_calls,
            "total_seconds": total_time,
            "cumulative_seconds": cumulative_time,
        })
    payload = {
        "schema_version": "deepmew.e1_replay_profile.v1",
        "rows": args.rows,
        "record_identities": [record.decision_id for record in records],
        "process_startup_and_imports_excluded": True,
        "init_seconds": init_seconds,
        "load_seconds": load_seconds,
        "total_seconds": elapsed,
        "peak_tracemalloc_bytes": peak,
        "profile_path": str(profile_path),
        "top_cumulative": top,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
