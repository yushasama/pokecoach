"""Seal one strict AWR trainer manifest for a registered league candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.curriculum.awr_train_receipt import (  # noqa: E402
    seal_awr_train_receipt,
    write_awr_train_receipt_atomic,
)
from deepmew.curriculum.league_generation import GenerationPhase, load_generation  # noqa: E402
from deepmew.curriculum.league_registry import (  # noqa: E402
    LeagueRole,
    load_registry,
    sha256_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="data_ready league_generation.v1 receipt")
    parser.add_argument("--candidate-member-id", required=True)
    parser.add_argument("--checkpoint", required=True, help="registered candidate checkpoint")
    parser.add_argument("--trainer-manifest", required=True, help="deepmew.r18.awr.v1 manifest")
    parser.add_argument("--data-manifest", required=True, help="bound league datagen v2 manifest")
    parser.add_argument("--out", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = load_generation(args.state)
    if state.phase is not GenerationPhase.DATA_READY:
        raise ValueError("AWR receipt sealing requires a data_ready generation")
    data_artifact = state.artifacts.get("data_manifest")
    if data_artifact is None:
        raise ValueError("data_ready generation has no data_manifest artifact")
    data_path = Path(args.data_manifest).resolve(strict=True)
    if data_path != Path(data_artifact.path).resolve():
        raise ValueError("requested data manifest path does not match the generation")
    if sha256_path(data_path) != data_artifact.sha256:
        raise ValueError("requested data manifest bytes do not match the generation")
    registry = load_registry(state.registry_path)
    candidate = registry.member(args.candidate_member_id)
    if candidate is None or candidate.role is not LeagueRole.CANDIDATE:
        raise ValueError("receipt sealing requires a registered candidate member")
    if candidate.parent_member_id != state.champion_member_id:
        raise ValueError("candidate parent does not match the generation champion")
    if candidate.feature_policy_id != state.feature_policy_id:
        raise ValueError("candidate feature policy does not match the generation")
    if candidate.deck_sha256 != state.champion_deck_sha256:
        raise ValueError("candidate deck does not match the generation champion deck")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if sha256_path(checkpoint) != candidate.artifact_sha256:
        raise ValueError("candidate checkpoint bytes do not match the registry")
    if sha256_path(candidate.deck_path) != candidate.deck_sha256:
        raise ValueError("candidate deck bytes do not match the registry")
    receipt = seal_awr_train_receipt(
        generation_id=state.generation_id,
        snapshot_id=state.snapshot_id,
        champion_member_id=state.champion_member_id,
        candidate_member_id=candidate.league_member_id,
        candidate_deck_sha256=candidate.deck_sha256,
        feature_policy_id=state.feature_policy_id,
        input_data_manifest_sha256=data_artifact.sha256,
        init_checkpoint_sha256=state.champion_artifact_sha256,
        output_checkpoint_sha256=candidate.artifact_sha256,
        trainer_manifest_path=args.trainer_manifest,
    )
    sealed_output = Path(receipt["receipts"]["output_checkpoint"]["path"]).resolve()
    if sealed_output != checkpoint:
        raise ValueError("trainer output path does not match the registered candidate checkpoint")
    write_awr_train_receipt_atomic(args.out, receipt)
    print(json.dumps({
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "generation_id": receipt["generation_id"],
        "candidate_member_id": receipt["candidate_member_id"],
        "output_checkpoint_sha256": receipt["output_checkpoint_sha256"],
        "out": str(Path(args.out).resolve()),
    }, sort_keys=True, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
