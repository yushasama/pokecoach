"""Build a resumable content-addressed E1 replay observation admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.rl.enhanced_replay import ReplayBatchMaterializer
from deepmew.rl.enhanced_replay_admission import ReplayAdmissionBuilder, bindings_from_materializer


DEFAULT_CACHE = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_cache_v2\7459a801-77afa18a")
DEFAULT_SPLIT = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\SPLIT_MANIFEST.json")
DEFAULT_DECK = "sha256:109a82f280159691594de831d3f3f507ea11d10d2bc8eaf54b23ba4ce1aee107"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--binding-row-start", action="append", type=int)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    materializer = ReplayBatchMaterializer(
        DEFAULT_CACHE,
        split_manifest_path=DEFAULT_SPLIT,
        exact_deck_identity=DEFAULT_DECK,
    )
    bindings = bindings_from_materializer(materializer)
    if args.binding_row_start:
        selected = set(args.binding_row_start)
        bindings = tuple(binding for binding in bindings if binding.row_start in selected)
        if {binding.row_start for binding in bindings} != selected:
            raise SystemExit("one requested binding row start was not found")
    corpus_identity = {
        "cache_manifest_sha256": materializer.cache_manifest_sha256,
        "split_manifest_sha256": materializer.split_manifest_sha256,
        "action_schema_identity": "deepmew.actor_transition.v1",
        "observation_schema_identity": "deepmew.replay_action_row.v1:state_json actor_visible",
        "eligible_train_rows": int((
            (materializer.data.meta["split_idx"] == materializer.train_index)
            & materializer.data.meta["is_target_eligible"]
        ).sum()),
    }
    builder = ReplayAdmissionBuilder(
        output / "admission.sqlite3",
        corpus_identity=corpus_identity,
        bindings=bindings,
        split_idx=materializer.data.meta["split_idx"],
        is_target_eligible=materializer.data.meta["is_target_eligible"],
        train_index=materializer.train_index,
    )
    collection = builder.collect_requests()
    observations = builder.materialize_observations()
    receipt = builder.finalize(output / "admission.manifest.json")
    print(json.dumps({"collection": collection, "observations": observations, "receipt": receipt}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
