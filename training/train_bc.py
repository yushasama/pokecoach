"""Compatibility entry point forwarding to the single admitted S0 trainer.

The former real-corpus/auto-device flags are intentionally unavailable in S0.
"""
from __future__ import annotations

import sys

from deepmew_ex.cli import main
from deepmew_ex.models.mlp import DeepMewEX
from deepmew_ex.training.bc import set_seed, train_epoch

__all__ = ['DeepMewEX', 'set_seed', 'train_epoch']

if __name__ == '__main__':
    main(['experiment', 'run', *sys.argv[1:]])
