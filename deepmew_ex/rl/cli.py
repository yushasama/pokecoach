"""Light entrypoint for RL commands (keeps torch out of spawned lean farm workers).

    python -m deepmew_ex.rl.cli train --config experiments/deepmew_ex/rl-league-pilot.json --run-id <id> [--resume <ckpt>]
    python -m deepmew_ex.rl.cli farm-eval --run-id <id> --deck <deck> --participant bc=<pt>@<sha> --participant rl=<pt>
    python -m deepmew_ex.rl.cli evaluate --run-id <id> --deck <deck> --participant ...

Multiprocessing ``spawn`` re-imports the launching ``__main__`` module in
every child; launching through this module (no torch at import time) keeps
lean workers at ~70 MB. ``python -m deepmew_ex.rl.train`` still works but its
children import torch.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in ('train', 'farm-eval', 'evaluate'):
        raise SystemExit(__doc__)
    command, rest = arguments[0], arguments[1:]
    if command == 'train':
        from deepmew_ex.rl.train import main as run
    elif command == 'farm-eval':
        from deepmew_ex.rl.farm_eval import main as run
    else:
        from deepmew_ex.rl.evaluate import main as run
    run(rest)


if __name__ == '__main__':
    main()
