"""Eval-only ladder surrogate from bucket-arena matchup outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.curriculum.ladder_surrogate import load_matchup_rates, simulate_ladder_surrogate  # noqa: E402


def _format_metric(value: object) -> str:
    if value is None:
        return "unavailable"

    return str(value)


def _report(result: dict) -> str:
    lines = [
        "# Ladder Surrogate",
        "",
        "**Eval-only surrogate. Not Kaggle ladder truth. Score margin ignored.**",
        "",
        f"- expected_final_mu: `{result['expected_final_mu']}`",
        f"- median_final_mu: `{result['median_final_mu']}`",
        f"- mu_after_8: `{_format_metric(result['mu_after_8'])}`",
        f"- mu_after_32: `{_format_metric(result['mu_after_32'])}`",
        f"- P(reach 800): `{result['P_reach_800']}`",
        f"- P(reach 900): `{result['P_reach_900']}`",
        f"- P(reach 1000): `{result['P_reach_1000']}`",
        f"- P(reach 1100): `{result['P_reach_1100']}`",
        f"- P(0-2 start): `{result['P_0_2_start']}`",
        f"- P(2-6 or worse first 8): `{result['P_2_6_or_worse_first_8']}`",
        f"- draw_rate: `{result['draw_rate']}`",
        f"- collapse_rate_after_900: `{result['collapse_rate_after_900']}`",
    ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchups", required=True, help="bucket-arena results.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--start-mu", type=float, default=600.0)
    parser.add_argument("--start-sigma", type=float, default=120.0)
    parser.add_argument("--min-sigma", type=float, default=35.0)
    parser.add_argument("--sigma-decay", type=float, default=0.985)
    parser.add_argument("--base-k", type=float, default=28.0)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args(argv)

    rates = load_matchup_rates(args.matchups)
    result = simulate_ladder_surrogate(
        rates,
        episodes=args.episodes,
        games=args.games,
        start_mu=args.start_mu,
        start_sigma=args.start_sigma,
        min_sigma=args.min_sigma,
        sigma_decay=args.sigma_decay,
        base_k=args.base_k,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ladder_surrogate.json"
    md_path = out_dir / "ladder_surrogate.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"outputs": {"json": str(json_path), "markdown": str(md_path)},
                      "summary": {k: result[k] for k in ("expected_final_mu", "median_final_mu", "mu_after_8", "P_reach_900")}}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
