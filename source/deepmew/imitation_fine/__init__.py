"""K4.1 fine-grained imitation + K5-lite offline behavior cloning (eval-only).

K4 broad action-category behavior cloning was an honest negative: select context
already determines which *category* a player picks, so archetype labels added
nothing at that granularity. K4.1 moves the imitation target to where deck identity
actually lives -- which card, which target, which option advances the deck's plan
while avoiding punishment -- and attaches deterministic intent / punishment / risk /
last-stand pseudo-labels. K5-lite then compares two offline, corpus-only policy
improvements over those labels: filtered behavior cloning and reward-weighted
behavior cloning.

Everything here is offline/training-only. Nothing in this package is imported by
``main.py`` / the submission runtime; root `main.py`, `deck.csv`, `cg/`, and the
default `v2.4` policy stay frozen.
"""

from __future__ import annotations

from deepmew.imitation_fine.action_value import (
    ActionValueConfig,
    ConservativeActionValueTable,
    build_action_value_table,
    evaluate_action_value,
)
from deepmew.imitation_fine.bc import (
    FineBCEval,
    FineBCTable,
    build_context_baseline_table,
    build_filtered_bc_table,
    build_reward_weighted_bc_table,
    evaluate_fine_bc,
)
from deepmew.imitation_fine.board import read_board
from deepmew.imitation_fine.build import FineResult, build_imitation_fine
from deepmew.imitation_fine.labelers import label_decision
from deepmew.imitation_fine.schemas import BoardSnapshot, FineConfig, FineExample
from deepmew.imitation_fine.teacher import is_trusted_teacher, reward_weight

__all__ = [
    "ActionValueConfig",
    "BoardSnapshot",
    "ConservativeActionValueTable",
    "FineBCEval",
    "FineBCTable",
    "FineConfig",
    "FineExample",
    "FineResult",
    "build_action_value_table",
    "build_context_baseline_table",
    "build_filtered_bc_table",
    "build_imitation_fine",
    "build_reward_weighted_bc_table",
    "evaluate_action_value",
    "evaluate_fine_bc",
    "is_trusted_teacher",
    "label_decision",
    "read_board",
    "reward_weight",
]
