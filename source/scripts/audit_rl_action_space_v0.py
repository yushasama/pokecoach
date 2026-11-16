"""Run a bounded real-engine audit of DeepMew's RL action-space contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.rl.action_space_audit import run_action_space_audit, write_action_space_audit  # noqa: E402


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-per-game", type=int, default=2000)
    parser.add_argument("--out-dir", default=".research/rl_ablation_roadmap_v0")
    parser.add_argument("--opponents", nargs="+", default=["first-legal", "v2.4"])
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _build_args(argv)
    audit = run_action_space_audit(
        decisions=max(1, args.decisions),
        seed=args.seed,
        max_steps_per_game=args.max_steps_per_game,
        opponents=tuple(args.opponents),
    )
    json_path, md_path = write_action_space_audit(audit, args.out_dir)
    payload = audit.to_json()
    print(
        "[audit_rl_action_space_v0] "
        f"decisions={payload['decisions']} max_options={payload['max_option_count']} "
        f"multiselect={payload['multiselect_decisions']} "
        f"zero_select={payload['zero_select_decisions']} crashes={payload['crashes']} "
        f"timeouts={payload['timeouts']}"
    )
    print(f"  json={json_path}")
    print(f"  markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
