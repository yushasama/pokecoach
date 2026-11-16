"""Curriculum arena driver: run a candidate up the difficulty ladder.

Composes the four "harder opponent" levers into one auto-advancing curriculum
(see `deepmew/curriculum/`):

  1. DECK BAND   -- `build_stage` pulls the rung's target ELO bucket + carried
                    earlier buckets (progressive inclusion).
  2. PILOT TIERS -- ascending brains pilot those decks:
                      first-legal / v2.4  -> SpecAgent (heuristic)
                      replay              -> ReplayAgent (K4 action-category dist)
                      exploiter           -> champion checkpoint piloting opp deck
                      adversarial         -> a best-response checkpoint trained vs
                                             the current candidate (--adversarial)
                    Tiers whose data is absent are SKIPPED with a logged reason
                    (so the run still completes with whatever brains exist).
  3. COUNTERS    -- structural hard-counter shells (Archaludon/Crustle/Grimmsnarl)
                    pulled from the pool by shell match and injected as extra
                    opponents piloted by the rung's hardest available tier.
  4. GATE        -- target-band win rate needed to advance. On pass we auto-advance
                    to the next (harder) rung and freeze the candidate into the
                    league; on fail we stop (train more vs this rung).

INVARIANT #2: opponents are a FROZEN population (frozen decks + fixed-snapshot
brains). The league only GROWS by adding frozen members; nothing co-adapts to
equilibrium. INVARIANT #5: this is a filter + training-target generator, not the
ladder oracle. Offline/eval-only (INVARIANT #8): submission runtime untouched.

Examples:
  # Cheap composition check, no games:
  python scripts/run_curriculum_arena.py --pools <bucket_pools.json> \
      --candidate spec:greedy --plan
  # Run one rung with heuristic + replay pilots:
  python scripts/run_curriculum_arena.py --pools <pools.json> \
      --candidate spec:greedy --rung 1 --games 16 \
      --replay-table .research/runs/imitation/<run>/ --no-write
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

from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_json, write_markdown  # noqa: E402
from deepmew.bucket_arena import build_stage, entry_from_json  # noqa: E402
from deepmew.curriculum import (  # noqa: E402
    CurriculumConfig,
    League,
    RungResult,
    advance,
    default_curriculum,
    is_complete,
    pilot_rank,
)
from deepmew.real_meta_arena.replay_policy import ReplayAgent, build_replay_policy_table  # noqa: E402
from deepmew.real_meta_arena.runner import SpecAgent, run_real_meta_matchup  # noqa: E402
import run_r17_bucket_arena as r17  # noqa: E402

SCHEMA_VERSION = "deepmew.curriculum_arena.v1"
HEURISTIC_TIERS = {"first-legal", "random", "greedy", "legacy-v2", "v2.4", "v4"}


class OpponentFactory:
    """Builds an opponent SeatAgent for a (pilot tier, deck) pair, or explains why not.

    Caches the champion policy / replay tables so repeated builds are cheap. A tier
    that cannot be built (missing torch, missing table/checkpoint) is reported once
    via `skips` and simply not played.

    The `replay` tier prefers band-stratified R16 tables (`--replay-tables`, faithful
    per-band pilots) and falls back to a single K4 table (`--replay-table` dir)."""

    def __init__(self, *, champion: Path, deck_policy: str,
                 replay_table_dir: str, replay_tables_path: str, adversarial_ckpt: str) -> None:
        self.champion = champion
        self.deck_policy = deck_policy
        self.replay_table_dir = replay_table_dir
        self.replay_tables_path = replay_tables_path
        self.adversarial_ckpt = adversarial_ckpt
        self.skips: dict[str, str] = {}
        self._replay_table = None
        self._band_tables = None

    def _table(self):
        if self._replay_table is None:
            self._replay_table = build_replay_policy_table(self.replay_table_dir)
        return self._replay_table

    def _bands(self):
        if self._band_tables is None:
            from deepmew.curriculum.replay_pilots import BandReplayTables  # noqa: PLC0415
            self._band_tables = BandReplayTables.from_json(
                json.loads(Path(self.replay_tables_path).read_text(encoding="utf-8"))
            )
        return self._band_tables

    def available(self, tier: str) -> tuple[bool, str]:
        """Cheap check used by --plan and to pre-filter tiers (no game run)."""
        if tier in HEURISTIC_TIERS:
            return True, "heuristic"
        if tier == "replay":
            if self.replay_tables_path:
                if not Path(self.replay_tables_path).exists():
                    return False, f"band replay tables missing: {self.replay_tables_path}"
                return True, "band-stratified R16 replay pilots (faithful per-band)"
            if not self.replay_table_dir:
                return False, "no --replay-tables / --replay-table given"
            if not (Path(self.replay_table_dir) / "imitation_examples.jsonl").exists():
                return False, f"no imitation_examples.jsonl under {self.replay_table_dir}"
            return True, "K4 replay distribution (global, not band-stratified)"
        if tier == "exploiter":
            if not self.champion.exists():
                return False, f"champion checkpoint missing: {self.champion}"
            return True, "champion best-response (needs torch at run time)"
        if tier == "adversarial":
            if not self.adversarial_ckpt:
                return False, "no --adversarial checkpoint given"
            if not Path(self.adversarial_ckpt).exists():
                return False, f"adversarial checkpoint missing: {self.adversarial_ckpt}"
            return True, "trained best-response vs current candidate"
        return False, f"unknown tier {tier!r}"

    def build(self, tier: str, deck: list[int], deck_hash: str, shell: str, bucket: str = "") -> Any:
        if tier in HEURISTIC_TIERS:
            return SpecAgent.from_str(tier)
        if tier == "replay":
            table = self._bands().table_for(bucket) if self.replay_tables_path else self._table()
            return ReplayAgent(archetype=shell, table=table)
        if tier == "exploiter":
            return r17.build_exploiter(self.champion, deck, deck_hash, self.deck_policy)
        if tier == "adversarial":
            return r17.build_exploiter(Path(self.adversarial_ckpt), deck, deck_hash, self.deck_policy)
        raise ValueError(f"unknown tier {tier!r}")


def collect_counters(pools: dict[str, list[Any]], shells: dict[str, str],
                     counter_shells: tuple[str, ...]) -> list[Any]:
    """Pull decks from any band whose shell matches a counter shell (substring, ci).

    Structural counters are counters regardless of band, so we scan all pools and
    keep the most-played matching deck per shell. Shell labels live in the pool
    JSON (keyed by deck_hash), not on the reconstructed DeckPoolEntry."""
    if not counter_shells:
        return []
    wanted = [s.lower() for s in counter_shells]
    best: dict[str, Any] = {}
    for entries in pools.values():
        for entry in entries:
            shell = str(shells.get(entry.deck_hash, "") or "").lower()
            for target in wanted:
                if target in shell:
                    prev = best.get(target)
                    if prev is None or entry.games > prev.games:
                        best[target] = entry
    return list(best.values())


def run_rung(config: CurriculumConfig, rung_id: int, *, pools: dict[str, list[Any]],
             shells: dict[str, str], candidate: Any, our_deck: list[int],
             factory: OpponentFactory, games: int, seed: int) -> tuple[RungResult, list[dict[str, Any]]]:
    rung = config.rung(rung_id)
    stage = build_stage(pools, rung.target_bucket, top_n=rung.top_n, carry_n=rung.carry_n)
    if not stage:
        raise SystemExit(f"rung {rung_id} target bucket {rung.target_bucket} is empty in this pool")

    # Only run tiers whose data actually exists.
    active_tiers: list[str] = []
    for tier in sorted(rung.pilot_tiers, key=pilot_rank):
        ok, reason = factory.available(tier)
        if ok:
            active_tiers.append(tier)
        else:
            factory.skips[tier] = reason
    if not active_tiers:
        raise SystemExit(f"rung {rung_id}: no pilot tiers available ({factory.skips})")
    hardest_active = active_tiers[-1]

    # Counter decks: extra opponents piloted by the hardest available tier.
    counter_entries = collect_counters(pools, shells, rung.counter_shells)

    rows: list[dict[str, Any]] = []
    per_opp: dict[str, float] = {}
    safety = {"crashes": 0, "timeouts": 0}

    def _play(deck: list[int], deck_hash: str, bucket: str, shell: str,
              tier: str, tag: str, opp_index: int) -> None:
        opp_agent = factory.build(tier, deck, deck_hash, shell, bucket=bucket)
        summary = run_real_meta_matchup(
            candidate, opp_agent, our_deck, list(deck),
            games=games, bucket=bucket, opp_deck_hash=deck_hash,
            opp_archetype=shell, seed=seed + opp_index,
        )
        safety["crashes"] += summary.crashes
        safety["timeouts"] += summary.timeouts
        decided = summary.our_wins + summary.opp_wins
        wr = summary.our_wins / decided if decided else 0.0
        per_opp[f"{deck_hash}:{tier}"] = wr
        rows.append({
            "kind": tag, "bucket": bucket, "shell": shell, "deck_hash": deck_hash,
            "pilot": tier, "wins": summary.our_wins, "losses": summary.opp_wins,
            "draws": summary.draws, "winrate": round(wr, 4),
            "crashes": summary.crashes, "timeouts": summary.timeouts,
        })
        print(f"  [{bucket:9s}] {shell[:26]:26s} x {tier:11s} "
              f"{summary.our_wins}-{summary.opp_wins}-{summary.draws} wr={wr:.3f}", flush=True)

    for opp_index, opponent in enumerate(stage):
        shell = shells.get(opponent.deck_hash, "Unknown")
        for tier in active_tiers:
            if tier in {"exploiter", "adversarial"} and opp_index >= rung.exploiter_top:
                continue
            _play(list(opponent.card_ids), opponent.deck_hash, opponent.bucket, shell,
                   tier, "stage", opp_index)

    for c_index, entry in enumerate(counter_entries):
        _play(list(entry.card_ids), entry.deck_hash, "counter",
               shells.get(entry.deck_hash, "Unknown"), hardest_active,
               "counter", 10_000 + c_index)

    target_rows = [r for r in rows if r["bucket"] == rung.target_bucket]
    tw = sum(r["wins"] for r in target_rows)
    tl = sum(r["losses"] for r in target_rows)
    target_wr = tw / max(1, tw + tl)
    aw = sum(r["wins"] for r in rows)
    al = sum(r["losses"] for r in rows)
    result = RungResult(
        rung_id=rung_id, target_bucket=rung.target_bucket,
        target_winrate=target_wr, overall_winrate=aw / max(1, aw + al),
        gate=rung.gate, passed=target_wr >= rung.gate,
        safety=safety, per_opponent_winrate=per_opp,
    )
    return result, rows


def plan(config: CurriculumConfig, pools: dict[str, list[Any]], shells: dict[str, str],
         factory: OpponentFactory, league: League) -> dict[str, Any]:
    """Dry-run: show the composed curriculum without running any game."""
    out: dict[str, Any] = {"league": league.to_json(), "rungs": []}
    for rung in config.rungs:
        stage = build_stage(pools, rung.target_bucket, top_n=rung.top_n, carry_n=rung.carry_n)
        tiers = []
        for tier in sorted(rung.pilot_tiers, key=pilot_rank):
            ok, reason = factory.available(tier)
            tiers.append({"tier": tier, "available": ok, "reason": reason})
        counters = collect_counters(pools, shells, rung.counter_shells)
        out["rungs"].append({
            "rung_id": rung.rung_id, "name": rung.name, "target_bucket": rung.target_bucket,
            "gate": rung.gate, "difficulty": rung.difficulty,
            "stage_decks": len(stage),
            "pilot_tiers": tiers,
            "counter_decks": [f"{shells.get(e.deck_hash, '?')}:{e.deck_hash}" for e in counters],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True, help="bucket_pools.json from build_r17_bucket_pool.py")
    parser.add_argument("--candidate", required=True, help="spec:<base> | bundle:<dir> | checkpoint:<path>")
    parser.add_argument("--config", default="", help="curriculum config JSON (default: built-in 5-rung ladder)")
    parser.add_argument("--league", default="", help="league JSON to resume (default: fresh)")
    parser.add_argument("--rung", type=int, default=-1, help="run a single rung id (default: league round)")
    parser.add_argument("--auto", action="store_true", help="chain rungs from --rung, advancing on each gate pass")
    parser.add_argument("--plan", action="store_true", help="dry-run: print composed curriculum, run no games")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--deck-policy", default="reference_abomasnow")
    parser.add_argument("--champion", default=str(r17.DEFAULT_CHAMPION))
    parser.add_argument("--replay-tables", default="",
                        help="band_replay_tables.json from build_r19_replay_pilots.py (faithful per-band pilots)")
    parser.add_argument("--replay-table", default="",
                        help="fallback: K4 imitation run dir (global, non-band-stratified replay pilot)")
    parser.add_argument("--adversarial", default="", help="checkpoint trained as best-response vs the candidate")
    parser.add_argument("--freeze-candidate", action="store_true",
                        help="on gate pass, freeze the candidate into the league as a new opponent")
    parser.add_argument("--seed", type=int, default=170708)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = (CurriculumConfig.from_json(json.loads(Path(args.config).read_text(encoding="utf-8")))
              if args.config else default_curriculum())
    league = (League.from_json(json.loads(Path(args.league).read_text(encoding="utf-8")))
              if args.league else League())

    pools_payload = json.loads(Path(args.pools).read_text(encoding="utf-8"))
    pools = {b: [entry_from_json(r) for r in rows] for b, rows in pools_payload["pools"].items()}
    shells = {r["deck_hash"]: str(r.get("shell") or "Unknown")
              for rows in pools_payload["pools"].values() for r in rows}

    factory = OpponentFactory(
        champion=Path(args.champion), deck_policy=args.deck_policy,
        replay_table_dir=args.replay_table, replay_tables_path=args.replay_tables,
        adversarial_ckpt=args.adversarial,
    )

    if args.plan:
        report = plan(config, pools, shells, factory, league)
        print(json.dumps(report, indent=2))
        return

    candidate = r17.build_candidate(args.candidate, args.deck_policy)
    our_deck = list(getattr(candidate, "deck", None)
                    or getattr(getattr(candidate, "inner", None), "deck", None) or [])
    if len(our_deck) != 60:
        raise SystemExit("candidate did not expose a 60-card deck")

    start = args.rung if args.rung >= 0 else min(league.round, len(config.rungs) - 1)
    rung_id: int | None = start
    results: list[RungResult] = []
    all_rows: list[dict[str, Any]] = []
    while rung_id is not None:
        print(f"== rung {rung_id}: {config.rung(rung_id).name} "
              f"(gate {config.rung(rung_id).gate}) ==", flush=True)
        result, rows = run_rung(config, rung_id, pools=pools, shells=shells,
                                candidate=candidate, our_deck=our_deck,
                                factory=factory, games=args.games, seed=args.seed)
        results.append(result)
        all_rows.extend(rows)
        print(f"   -> target wr {result.target_winrate:.4f} vs gate {result.gate} "
              f"=> {'PASS' if result.passed else 'FAIL'}  safety {result.safety}", flush=True)
        if args.freeze_candidate and result.passed:
            league.freeze_candidate(args.candidate, note=f"passed rung {rung_id}")
        if not args.auto:
            break
        nxt = advance(config, result)
        rung_id = nxt.rung_id if nxt is not None else None

    last = results[-1]
    pfsp = league.pfsp(last.per_opponent_winrate)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate": args.candidate,
        "results": [r.to_json() for r in results],
        "curriculum_complete": is_complete(config, last),
        "final_passed": last.passed,
        "tier_skips": factory.skips,
        "league": league.to_json(),
        "pfsp_weights": pfsp,
        "rows": all_rows,
    }
    print(json.dumps({
        "final_rung": last.rung_id, "target_winrate": last.target_winrate,
        "passed": last.passed, "curriculum_complete": summary["curriculum_complete"],
        "tier_skips": factory.skips,
        "safety": {"crashes": sum(r.safety["crashes"] for r in results),
                   "timeouts": sum(r.safety["timeouts"] for r in results)},
    }, indent=2))

    if args.no_write:
        return
    run_dir = create_run_dir(kind="curriculum_arena", out_root=ROOT / ".research" / "runs")
    write_json(run_dir.output_path("results.json"), summary)
    lines = ["# Curriculum arena", "",
             f"- candidate: `{args.candidate}`", f"- tier skips: {factory.skips}",
             f"- curriculum complete: {summary['curriculum_complete']}", "",
             "| rung | bucket | gate | target wr | pass |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.rung_id} | {r.target_bucket} | {r.gate} | "
                     f"{r.target_winrate:.4f} | {'PASS' if r.passed else 'FAIL'} |")
    write_markdown(run_dir.output_path("report.md"), "\n".join(lines) + "\n")
    if args.freeze_candidate:
        write_json(run_dir.output_path("league.json"), league.to_json())
    manifest = build_manifest(
        run_id=run_dir.run_id, kind="curriculum_arena",
        command=[sys.executable, *sys.argv],
        inputs={"pools": args.pools, "candidate": args.candidate},
        outputs={"results": "results.json", "report": "report.md"},
        summary={"final_rung": last.rung_id, "final_passed": last.passed,
                 "curriculum_complete": summary["curriculum_complete"]},
    )
    write_manifest(run_dir.manifest_path, manifest)
    print("artifacts:", run_dir.path)


if __name__ == "__main__":
    main()
