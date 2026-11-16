"""Build resumable training-ready E1 replay rows from an admitted corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch

import run_enhanced_generation_smoke as smoke
from deepmew.replay.contract.preflight import require_accepted_canonical_decision_receipt
from deepmew.rl.enhanced_replay import ReplayBatchMaterializer
from deepmew.rl.enhanced_replay_derived import DerivedReplayBuilder, DerivedReplayIdentity


DEFAULT_CACHE = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_cache_v2\7459a801-77afa18a")
DEFAULT_SPLIT = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\SPLIT_MANIFEST.json")
DEFAULT_R7 = Path(r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt")
DEFAULT_V0 = Path(r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_runs\phasef4_grouped_crossfit_raw_s1_s915002_v3\full_train\value.pt")
DEFAULT_CALIBRATOR = ROOT / ".research/value_q_experiment/artifacts/phasef5_grouped_calibration_raw_s1_s915002_v3"
DEFAULT_DECK = "sha256:109a82f280159691594de831d3f3f507ea11d10d2bc8eaf54b23ba4ce1aee107"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--admission-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--binding-row-start", action="append", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    accepted_receipt = require_accepted_canonical_decision_receipt(
        args.canonical_decision_receipt,
        consumer=__name__,
    )
    args.canonical_decision_receipt = str(accepted_receipt.path)
    admission = json.loads(Path(args.admission_manifest).read_text(encoding="ascii"))
    materializer = ReplayBatchMaterializer(
        DEFAULT_CACHE,
        split_manifest_path=DEFAULT_SPLIT,
        exact_deck_identity=DEFAULT_DECK,
        admission_manifest_path=args.admission_manifest,
    )
    v0 = smoke._load_v0(SimpleNamespace(
        v0_checkpoint=str(DEFAULT_V0),
        v0_checkpoint_sha256="cd1339e14fa943382069128bc9bf8657de9c441286a49bdef0f42450d56eb0a9",
        v0_calibrator_root=str(DEFAULT_CALIBRATOR),
        v0_calibrator_sha256="6a0868b1da31c76605620a74572f0621f74f3ce362ed04a833aae9c7f8304398",
    ))
    r7 = torch.load(DEFAULT_R7, map_location="cpu", weights_only=False)
    input_dim = int(r7["input_dim"])
    encoder_identity = hashlib.sha256(
        (_sha256(SCRIPTS / "run_enhanced_generation_smoke.py") + _sha256(SCRIPTS / "run_terminal_league_ppo.py")).encode("ascii")
    ).hexdigest()
    identity = DerivedReplayIdentity(
        admission_receipt_sha256=admission["receipt_sha256"],
        cache_manifest_sha256=materializer.cache_manifest_sha256,
        split_manifest_sha256=materializer.split_manifest_sha256,
        r7_checkpoint_sha256=_sha256(DEFAULT_R7),
        v0_checkpoint_sha256=_sha256(DEFAULT_V0),
        v0_calibrator_sha256="6a0868b1da31c76605620a74572f0621f74f3ce362ed04a833aae9c7f8304398",
        action_schema_identity="deepmew.actor_transition.v1",
        observation_schema_identity="deepmew.replay_action_row.v1:state_json actor_visible",
        encoder_code_sha256=encoder_identity,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    builder = DerivedReplayBuilder(output / "derived.sqlite3", identity)
    starts = set(args.binding_row_start or ())
    bindings = [
        binding
        for binding in materializer.bindings
        if not starts or binding.row_start in starts
    ]
    rows = [
        global_row
        for binding in bindings
        for global_row in range(binding.row_start, binding.row_end)
        if materializer.data.meta["split_idx"][global_row] == materializer.train_index
        and materializer.data.meta["is_target_eligible"][global_row]
    ]
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = builder.missing_global_rows(rows[offset:offset + args.batch_size])
        if not batch_rows:
            continue
        records = materializer.load(batch_rows)
        probabilities = np.asarray(v0.score(np.stack([record.s1 for record in records])), dtype=np.float64)
        examples = [
            smoke._encode_replay_record(
                record,
                v0=v0,
                input_dim=input_dim,
                v0_probability=float(probability),
            )
            for record, probability in zip(records, probabilities, strict=True)
        ]
        builder.add(batch_rows, records, examples)
    receipt = builder.finalize(output / "derived.manifest.json", expected_rows=len(rows))
    print(json.dumps(receipt, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
