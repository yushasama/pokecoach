"""R17: evaluate a candidate against the ELO-bucket curriculum arena.

Stage k = top decks of the target bucket + carried top decks of every earlier
bucket (progressive inclusion). Opponent pilots come in tiers:
  - `first-legal` (the low-band field),
  - `v2.4` (heuristic, harder),
  - `exploiter` = champion checkpoint piloting the opponent deck
    (AlphaStar-league main-exploiter analog; fixed snapshot, NOT self-play —
    INVARIANT #2 respected: no equilibrium signal, just best-response targets).

Candidates:
  --candidate bundle:<dir>            shipped distilled runtime
  --candidate bundle+guard:<dir>      same + futility guard
  --candidate checkpoint:<path.pt>    torch R4F checkpoint on our deck
  --candidate checkpoint+guard:<path>

Outputs per-opponent rows, per-bucket/tier aggregates, the stage gate verdict
(default threshold 0.70 on the target bucket), and PFSP weights (AlphaStar
prioritized fictitious self-play: opponents that beat us get more weight) for
the next training round. Arena is a filter, not the ladder oracle (INVARIANT #5).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from main import read_deck_csv as read_root_deck  # noqa: E402
import run_r9_r4f_retrain as r4f  # noqa: E402
from deepmew.arena import parse_agent_spec, select_action  # noqa: E402
from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_json, write_markdown  # noqa: E402
from deepmew.bucket_arena import build_stage, entry_from_json, pfsp_weights  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.futility_guard import FutilityGuard  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup  # noqa: E402
from run_r12_deck_pilot_matrix import PilotPolicyAdapter  # noqa: E402

SCHEMA_VERSION = "deepmew.r17_bucket_arena.v2"
DEFAULT_CHAMPION = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
SAFETY_KEYS = ("illegal_actions", "crashes", "timeouts")


def matchup_safety_counts(summary: Any) -> dict[str, int]:
    return {
        "illegal_actions": int(summary.illegal),
        "crashes": int(summary.crashes),
        "timeouts": int(summary.timeouts),
    }


def merge_safety_counts(total: dict[str, int], update: dict[str, int]) -> None:
    if set(total) != set(SAFETY_KEYS) or set(update) != set(SAFETY_KEYS):
        raise ValueError("bucket arena safety field inventory mismatch")
    for name in SAFETY_KEYS:
        value = update[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"bucket arena safety count {name} is invalid")
        total[name] += value


def assert_safety_clean(safety: dict[str, int]) -> None:
    if set(safety) != set(SAFETY_KEYS):
        raise ValueError("bucket arena safety field inventory mismatch")
    if any(safety[name] != 0 for name in SAFETY_KEYS):
        raise RuntimeError("bucket arena stopped after an illegal action, crash, or timeout")


class GuardedAgent:
    """Wrap any SeatAgent-style select with the futility guard."""

    def __init__(self, inner: Any, label: str | None = None) -> None:
        self.inner = inner
        self.label = label or f"{inner.label}+guard"
        self.guard = FutilityGuard()
        self.guard_swaps = 0

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        choice = self.inner.select(obs, rng)
        select = obs.get("select") if isinstance(obs, dict) else None
        if not isinstance(select, dict):
            return choice
        adjusted = self.guard.process(obs, select, list(choice))
        if adjusted != choice:
            self.guard_swaps += 1
        return adjusted


class BundleAgent:
    """SeatAgent adapter around a shipped bundle's DistilledPolicy."""

    def __init__(
        self,
        bundle_dir: Path,
        label: str,
        feature_policy_id: str = "",
        engine_deck: list[int] | None = None,
    ) -> None:
        self.label = label
        if str(bundle_dir) not in sys.path:
            sys.path.insert(0, str(bundle_dir))
        import r5_distilled_policy as runtime  # noqa: PLC0415

        self._runtime = runtime
        self._policy = runtime.DistilledPolicy(
            str(bundle_dir / "r5_policy_weights.npz"),
            str(bundle_dir / "r5_policy_weights.manifest.json"),
        )
        if feature_policy_id:
            self._policy.policy_id = feature_policy_id
        bundled_deck = [
            int(v) for v in (bundle_dir / "deck.csv").read_text(encoding="utf-8").split()
        ][:60]
        self.deck = list(engine_deck) if engine_deck is not None else bundled_deck

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        select = obs.get("select") if isinstance(obs, dict) else None
        if not isinstance(select, dict):
            return []
        try:
            choice = self._policy.select(obs)
        except Exception:
            choice = self._runtime.safe_fallback(select)
        return self._runtime.guard_selection([int(i) for i in choice], select)


class CheckpointAgent:
    """SeatAgent adapter around an R4F torch checkpoint piloting our deck.

    `deck_binding` optionally rebinds the pilot to a different deck slot
    (cards + feature deck_hash + policy_id), matching how the shipped deck
    -variant bundles (e.g. bd27) run on the ladder."""

    def __init__(
        self,
        checkpoint: Path,
        label: str,
        deck_policy: str,
        deck_binding: dict[str, Any] | None = None,
        feature_policy_id: str = "",
    ) -> None:
        self.label = label
        self._policy = r4f.R4FLivePolicy(label, checkpoint, deck_policy)
        if deck_binding:
            self._policy.deck = list(deck_binding["deck"])
            self._policy.deck_hash = str(deck_binding["deck_hash"])
            self._policy.policy_id = str(deck_binding["policy_id"])
        if feature_policy_id:
            self._policy.policy_id = feature_policy_id
        self.deck = list(self._policy.deck)

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        return self._policy.select(obs)


class SpecCandidateAgent:
    """SeatAgent piloting root deck.csv with an arena base spec.

    Lets a scriptless heuristic base (`greedy`, `first-legal`, `v2.4`) run as the
    candidate against the bucket pool, e.g. `--candidate spec:greedy`. The deck is
    the frozen root `deck.csv`; action selection reuses the arena `select_action`
    dispatch, so no policy is duplicated."""

    def __init__(self, spec_text: str) -> None:
        self.label = f"spec:{spec_text}"
        self._spec = parse_agent_spec(spec_text)
        self.deck = [int(v) for v in read_root_deck()][:60]

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        return select_action(self._spec, obs, rng)


class ReferenceTeacherCandidateAgent:
    """Adapter for frozen reference teachers on their own immutable decks."""

    def __init__(self, teacher_id: str) -> None:
        from deepmew.rl1_teacher_bc.teachers import make_teacher

        self.label = f"spec:{teacher_id}"
        self._teacher = make_teacher(teacher_id)
        self.deck = [int(card_id) for card_id in self._teacher.deck]

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        return self._teacher.select(obs, rng)


def _read_deck_file(path: Path) -> list[int]:
    return [int(v) for v in path.read_text(encoding="utf-8").split()][:60]


def _checkpoint_deck_binding(binding_ref: str) -> dict[str, Any] | None:
    """Resolve a checkpoint candidate's deck slot from a bundle dir or deck:<csv>."""
    if not binding_ref:
        return None
    kind, _, target = binding_ref.partition(":")
    if kind == "deck" and target:
        path = Path(target)
        deck = _read_deck_file(path)
        return {
            "deck": deck,
            "deck_hash": normalized_deck_hash(deck),
            "policy_id": f"deck:{path.stem}",
        }

    bind_dir = Path(binding_ref)
    manifest = json.loads((bind_dir / "r5_policy_weights.manifest.json").read_text(encoding="utf-8"))
    return {
        "deck": _read_deck_file(bind_dir / "deck.csv"),
        "deck_hash": str(manifest["feature_deck_hash"]),
        "policy_id": str(manifest["policy_id"]),
    }


def build_candidate(
    spec: str,
    deck_policy: str,
    deck_binding_dir: str = "",
    feature_policy_id: str = "",
) -> Any:
    kind, _, target = spec.partition(":")
    if not target:
        raise ValueError(f"candidate spec needs kind:path, got {spec!r}")
    guarded = kind.endswith("+guard")
    kind = kind.removesuffix("+guard")
    if kind == "spec":
        if target.startswith("teacher_"):
            agent: Any = ReferenceTeacherCandidateAgent(target)
        else:
            agent = SpecCandidateAgent(target)
    elif kind == "bundle":
        binding = _checkpoint_deck_binding(deck_binding_dir)
        agent = BundleAgent(
            Path(target).resolve(),
            Path(target).name,
            feature_policy_id=feature_policy_id,
            engine_deck=list(binding["deck"]) if binding is not None else None,
        )
    elif kind == "checkpoint":
        binding = _checkpoint_deck_binding(deck_binding_dir)
        agent = CheckpointAgent(
            Path(target),
            Path(target).stem,
            deck_policy,
            deck_binding=binding,
            feature_policy_id=feature_policy_id,
        )
    else:
        raise ValueError(f"unknown candidate kind {kind!r}")
    return GuardedAgent(agent) if guarded else agent


def build_exploiter(
    checkpoint: Path,
    deck: list[int],
    deck_hash: str,
    deck_policy: str,
    feature_policy_id: str = "bucket_exploiter",
) -> Any:
    policy = r4f.R4FLivePolicy("exploiter", checkpoint, deck_policy)
    adapter = PilotPolicyAdapter(policy)
    adapter.bind_deck(deck, deck_hash, feature_policy_id)
    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", required=True, help="bucket_pools.json from build_r17_bucket_pool.py")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stage", default="600-800")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--carry-n", type=int, default=4)
    parser.add_argument("--games", type=int, default=32, help="games per opponent deck per pilot tier")
    # DEFAULT = smart opponents. first-legal is a dumb baseline that INFLATES win
    # rates and mispredicts the ladder (observed R18: it rated bd27 0.66 vs the field
    # where v2.4+exploiter rate it 0.60). Keep first-legal opt-in for fast smokes only.
    parser.add_argument("--pilots", nargs="+", default=["v2.4", "exploiter"],
                        choices=["first-legal", "random", "greedy", "legacy-v2", "v2.4", "v4", "exploiter"])
    parser.add_argument("--exploiter-top", type=int, default=8, help="exploiter (champion policy) pilots the top-N stage decks")
    parser.add_argument("--champion", default=str(DEFAULT_CHAMPION))
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--candidate-deck-binding", default="",
                        help="bundle dir, or deck:<deck.csv>, whose deck slot a checkpoint candidate pilots")
    parser.add_argument(
        "--feature-policy-id",
        default="",
        help="explicit feature encoder policy ID for bundle/checkpoint candidates",
    )
    parser.add_argument("--gate", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=170707)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    pools_payload = json.loads(Path(args.pools).read_text(encoding="utf-8"))
    pools = {
        bucket: [entry_from_json(row) for row in rows]
        for bucket, rows in pools_payload["pools"].items()
    }
    shells = {
        row["deck_hash"]: str(row.get("shell") or "Unknown")
        for rows in pools_payload["pools"].values()
        for row in rows
    }
    stage = build_stage(pools, args.stage, top_n=args.top_n, carry_n=args.carry_n)
    if not stage:
        raise SystemExit(f"stage {args.stage} is empty")

    candidate = build_candidate(
        args.candidate,
        args.deck_policy,
        args.candidate_deck_binding,
        feature_policy_id=args.feature_policy_id,
    )
    our_deck = list(getattr(candidate, "deck", None) or getattr(getattr(candidate, "inner", None), "deck", None) or [])
    if len(our_deck) != 60:
        raise SystemExit("candidate did not expose a 60-card deck")

    rows: list[dict[str, Any]] = []
    per_deck_wr: dict[str, float] = {}
    safety = {name: 0 for name in SAFETY_KEYS}
    for opp_index, opponent in enumerate(stage):
        for pilot in args.pilots:
            if pilot == "exploiter" and opp_index >= args.exploiter_top:
                continue
            if pilot == "exploiter":
                opp_agent = build_exploiter(Path(args.champion), list(opponent.card_ids), opponent.deck_hash, args.deck_policy)
            else:
                opp_agent = SpecAgent.from_str(pilot)
            summary = run_real_meta_matchup(
                candidate, opp_agent, our_deck, list(opponent.card_ids),
                games=args.games, bucket=opponent.bucket,
                opp_deck_hash=opponent.deck_hash,
                opp_archetype=shells.get(opponent.deck_hash, "Unknown"),
                seed=args.seed + opp_index,
                stop_on_safety_failure=True,
            )
            merge_safety_counts(safety, matchup_safety_counts(summary))
            decided = summary.our_wins + summary.opp_wins
            wr = summary.our_wins / decided if decided else 0.0
            key = f"{opponent.deck_hash}:{pilot}"
            per_deck_wr[key] = wr
            rows.append({
                "bucket": opponent.bucket, "carried": opponent.carried,
                "deck_hash": opponent.deck_hash, "shell": shells.get(opponent.deck_hash, "Unknown"),
                "pilot": pilot, "games": summary.games,
                "wins": summary.our_wins, "losses": summary.opp_wins, "draws": summary.draws,
                "illegal_actions": summary.illegal,
                "crashes": summary.crashes,
                "timeouts": summary.timeouts,
                "attempted": summary.attempted,
                "stopped_early": summary.stopped_early,
                "winrate": round(wr, 4),
                "guard_swaps": getattr(candidate, "guard_swaps", 0),
                "errors": summary.errors,
            })
            print(f"[{opponent.bucket}{' carry' if opponent.carried else ''}] "
                  f"{shells.get(opponent.deck_hash,'?')[:28]:28s} x {pilot:11s} "
                  f"{summary.our_wins}-{summary.opp_wins}-{summary.draws} wr={wr:.3f}", flush=True)
            if any(safety.values()):
                print(json.dumps({"status": "hard_stop_safety_failure", "safety": safety}, indent=2))
                assert_safety_clean(safety)

    target_rows = [r for r in rows if r["bucket"] == args.stage]
    target_wins = sum(r["wins"] for r in target_rows)
    target_losses = sum(r["losses"] for r in target_rows)
    target_wr = target_wins / max(1, target_wins + target_losses)
    all_wins = sum(r["wins"] for r in rows)
    all_losses = sum(r["losses"] for r in rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate": args.candidate,
        "feature_policy_id": args.feature_policy_id or "artifact/default",
        "stage": args.stage,
        "pilots": args.pilots,
        "games_per_cell": args.games,
        "seed": args.seed,
        "rows": rows,
        "target_bucket_winrate": round(target_wr, 4),
        "overall_winrate": round(all_wins / max(1, all_wins + all_losses), 4),
        "gate": args.gate,
        "gate_passed": target_wr >= args.gate,
        "safety": safety,
        "pfsp_weights": pfsp_weights({k: v for k, v in per_deck_wr.items()}),
    }
    print(json.dumps({k: result[k] for k in ("stage", "target_bucket_winrate", "overall_winrate", "gate_passed", "safety")}, indent=2))
    if args.no_write:
        return
    run_dir = create_run_dir(kind="bucket_arena", out_root=ROOT / ".research" / "runs")
    write_json(run_dir.output_path("results.json"), result)
    lines = [
        "# R17 bucket arena", "",
        f"- candidate: `{args.candidate}`  stage: {args.stage}  gate: {args.gate} -> "
        f"**{'PASS' if result['gate_passed'] else 'FAIL'}** (target wr {result['target_bucket_winrate']})",
        f"- overall wr {result['overall_winrate']}  safety {safety}", "",
        "| bucket | shell | pilot | W-L-D | wr |", "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['bucket']}{' (carry)' if r['carried'] else ''} | {r['shell']} | {r['pilot']} "
                     f"| {r['wins']}-{r['losses']}-{r['draws']} | {r['winrate']} |")
    write_markdown(run_dir.output_path("report.md"), "\n".join(lines) + "\n")
    manifest = build_manifest(
        run_id=run_dir.run_id, kind="bucket_arena",
        command=[sys.executable, *sys.argv],
        inputs={"pools": args.pools, "candidate": args.candidate},
        outputs={"results": "results.json", "report": "report.md"},
        summary={"stage": args.stage, "target_bucket_winrate": result["target_bucket_winrate"],
                 "gate_passed": result["gate_passed"], "safety": safety},
    )
    write_manifest(run_dir.manifest_path, manifest)
    print("artifacts:", run_dir.path)


if __name__ == "__main__":
    main()
