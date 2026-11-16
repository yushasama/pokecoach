"""ELO-bucket curriculum arena (R17): per-band opponent pools, progressive
stages, and PFSP reweighting. Offline/eval-only (INVARIANT #8): never imported
by `main.py`/`cg`."""

from deepmew.bucket_arena.pool import DeckPoolEntry, build_bucket_pool, entry_from_json
from deepmew.bucket_arena.stage import STAGE_ORDER, StageOpponent, build_stage, pfsp_weights

__all__ = [
    "DeckPoolEntry",
    "STAGE_ORDER",
    "StageOpponent",
    "build_bucket_pool",
    "build_stage",
    "entry_from_json",
    "pfsp_weights",
]
