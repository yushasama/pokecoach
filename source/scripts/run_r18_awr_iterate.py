"""R18/R19: AWR policy-ITERATION loop for Starmie (the piece that can exceed the teacher).

Each round: (1) self-play the CURRENT policy vs the hard curriculum/league,
(2) AWR-train a new checkpoint warm-started from + KL-anchored to the current policy,
(3) eval the new checkpoint vs the greedy tempo pilot, (4) ADOPT it only if it beats the
running best (hill-climb; a regressing round is discarded, so the loop can't drift down).
Round 0 establishes the baseline (the init checkpoint = the R4-F Starmie prior).

INVARIANT #2: reward = observed game outcome vs a FIXED opponent snapshot/league,
not self-play-to-Nash. INVARIANT #1: every sub-run reports crashes/timeouts.

Example
-------
    python scripts/run_r18_awr_iterate.py --rounds 4 --workers 8 \
        --init-checkpoint <r4f.pt> --out-root .research/r18/awr_iterate
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
PY = sys.executable
_pd = sorted((ROOT / ".research/runs/bucket_pool").rglob("bucket_pools.json"))
POOLS_DEFAULT = _pd[-1] if _pd else (ROOT / "bucket_pools.json")  # container passes --pools explicitly
STARMIE_DEFAULT = ROOT / ".research/submissions/20260706-r4f-starmie-18ffc1e6"
_WR = re.compile(r'"overall_winrate":\s*([0-9.]+)')


def _write_history(root: Path, payload: dict) -> None:
    tmp = root / "history.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(root / "history.json")


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_text(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _elapsed_since_utc(value: str) -> float:
    started = _parse_utc_text(value)
    if started is None:
        return 0.0

    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _clear_partial_round_dir(root: Path, path: Path) -> None:
    """Remove a retried round directory while refusing paths outside this run."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(f"refusing to clear path outside run root: {path}")
    if path.exists():
        shutil.rmtree(path)


def _load_resume_payload(root: Path) -> dict | None:
    hist = root / "history.json"
    if not hist.exists():
        return None

    payload = json.loads(hist.read_text(encoding="utf-8"))
    history = payload.get("history") or []
    if not history:
        raise RuntimeError(f"resume history is empty: {hist}")
    best_ckpt = Path(payload["best_ckpt"])
    if not best_ckpt.exists():
        raise RuntimeError(f"resume best checkpoint is missing: {best_ckpt}")

    return payload


def _run(cmd: list[str], **kw) -> str:
    print(f"  $ {' '.join(str(c) for c in cmd[:6])} ...", flush=True)
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=str(ROOT), **kw)
    if p.returncode != 0:
        print(p.stdout[-1500:]); print(p.stderr[-1500:])
        raise RuntimeError(f"subrun failed ({p.returncode}): {cmd[1] if len(cmd) > 1 else cmd}")
    return p.stdout + p.stderr


def _league_eval_deck_binding(a: argparse.Namespace) -> str:
    if a.eval_deck_binding:
        return a.eval_deck_binding
    if a.league_config:
        from deepmew.curriculum.league_config import load_league_config  # noqa: PLC0415
        return load_league_config(a.league_config).our.deck
    return a.starmie_bundle


def _league_feature_policy_id(a: argparse.Namespace) -> str:
    if a.feature_policy_id:
        return a.feature_policy_id
    if a.league_config:
        from deepmew.curriculum.league_config import load_league_config  # noqa: PLC0415
        return load_league_config(a.league_config).our.effective_feature_policy_id

    return ""


def _assert_safety_clean(manifest: dict) -> None:
    totals = manifest.get("totals") or {}
    bad = {k: int(totals.get(k, 0) or 0) for k in ("illegal", "crashes", "timeouts")
           if int(totals.get(k, 0) or 0)}
    if bad:
        raise RuntimeError(f"safety failure in datagen manifest: {bad}")


def eval_vs_greedy(a: argparse.Namespace, ckpt: Path) -> float:
    deck_binding = _league_eval_deck_binding(a)
    feature_policy_id = _league_feature_policy_id(a)
    out = _run([PY, "scripts/run_r17_bucket_arena.py", "--pools", a.pools,
                "--candidate", f"checkpoint:{ckpt}", "--candidate-deck-binding", deck_binding,
                "--feature-policy-id", feature_policy_id,
                "--stage", a.eval_stage, "--pilots", *a.eval_pilots.split(","), "--games", str(a.eval_games),
                "--top-n", str(a.top_n), "--exploiter-top", str(a.exploiter_top),
                "--champion", a.champion, "--no-write"])
    m = _WR.findall(out)
    return float(m[-1]) if m else float("nan")


def datagen(a: argparse.Namespace, ckpt: Path, out_dir: Path) -> dict:
    if a.league_config:
        cmd = [PY, "scripts/run_league_datagen.py", "--config", a.league_config,
               "--pools", a.pools, "--out-dir", str(out_dir),
               "--our-pilot", f"checkpoint:{ckpt}", "--workers", str(a.workers),
               "--our-feature-policy-id", _league_feature_policy_id(a),
               "--seed", str(a.seed), "--max-steps", str(a.max_steps)]
        if a.league_champion:
            cmd += ["--champion", a.league_champion]
        elif a.champion:
            cmd += ["--champion", a.champion]
        if a.league_replay_tables:
            cmd += ["--replay-tables", a.league_replay_tables]
        out = _run(cmd)
    else:
        cmd = [PY, "scripts/run_r18_curriculum_datagen.py", "--pools", a.pools,
               "--starmie-bundle", a.starmie_bundle, "--stages", *a.stages.split(","),
               "--our-pilot", f"checkpoint:{ckpt}", "--opp-pilots", *a.opp_pilots.split(","),
               "--games", str(a.games), "--top-n", str(a.top_n), "--exploiter-top", str(a.exploiter_top),
               "--workers", str(a.workers), "--seed", str(a.seed), "--out-dir", str(out_dir),
               "--champion", a.champion, "--max-steps", str(a.max_steps)]
        if a.replay_tables:
            cmd += ["--replay-tables", a.replay_tables]
        out = _run(cmd)
    try:
        return json.loads((out_dir / "manifest.json").read_text())
    except Exception:
        return {}


def awr_train(a: argparse.Namespace, init_ckpt: Path, data_dir: Path, out_dir: Path) -> Path:
    cmd = [PY, "scripts/run_r18_awr_trainer.py", "--refgames-dir", str(data_dir),
           "--init-checkpoint", str(init_ckpt), "--out-dir", str(out_dir),
           "--temperature", str(a.temperature), "--kl-beta", str(a.kl_beta),
           "--max-seconds", str(a.max_seconds)]
    if a.ladder_weighting_config:
        cmd += ["--ladder-weighting-config", a.ladder_weighting_config]
    _run(cmd)
    return out_dir / "selected_ar_bc_r4f_candidate.pt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--pools", default=str(POOLS_DEFAULT))
    ap.add_argument("--starmie-bundle", default=str(STARMIE_DEFAULT))
    ap.add_argument("--league-config", default="", help="YAML league roster; overrides --opp-pilots datagen when set")
    ap.add_argument("--league-champion", default="", help="champion override passed to --league-config datagen")
    ap.add_argument("--league-replay-tables", default="", help="replay-table override passed to --league-config datagen")
    ap.add_argument("--eval-deck-binding", default="", help="deck slot for checkpoint eval; default = league our.deck or --starmie-bundle")
    ap.add_argument(
        "--feature-policy-id",
        default="",
        help="feature encoder policy ID; defaults to league our.feature_policy_id/label",
    )
    ap.add_argument("--stages", default="600-800,800-900")
    ap.add_argument("--opp-pilots", default="greedy,v2.4")
    ap.add_argument("--champion", default=str(ROOT / ".research/r12/lane1_r4f_focal_reweight/selected_ar_bc_r4f_reweighted_candidate.pt"))
    ap.add_argument("--replay-tables", default="", help="band_replay_tables.json for old --opp-pilots replay tier")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--exploiter-top", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--eval-pilots", default="greedy,v2.4", help="opponent pilots for the adopt-eval (richer than greedy-only)")
    ap.add_argument("--eval-stage", default="600-800")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--kl-beta", type=float, default=0.03)
    ap.add_argument("--max-seconds", type=float, default=150.0)
    ap.add_argument("--ladder-weighting-config", default="",
                    help="YAML/JSON row-weight config passed to run_r18_awr_trainer.py")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock cap on the WHOLE loop (0=off); stops after a round if exceeded, keeping best")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                    help="resume from out-root/history.json if present")
    ap.add_argument("--seed", type=int, default=181818)
    ap.add_argument("--out-root", default=str(ROOT / ".research/r18/awr_iterate"))
    a = ap.parse_args(argv)

    import time as _time
    _loop_start = _time.perf_counter()
    root = Path(a.out_root); root.mkdir(parents=True, exist_ok=True)
    base_seed = int(a.seed)
    resume_payload = _load_resume_payload(root) if a.resume else None
    durable_elapsed_seconds = 0.0
    resumed_active_round = None
    if resume_payload:
        budget = resume_payload.get("budget") or {}
        durable_elapsed_seconds = float(budget.get("elapsed_seconds", 0.0) or 0.0)
        active_started = str(budget.get("active_attempt_started_at_utc") or "")
        resumed_active_round = budget.get("active_round")
        if active_started:
            recovered = _elapsed_since_utc(active_started)
            durable_elapsed_seconds += recovered
            print(f"[iterate] resume budget accounting: added {recovered:.1f}s "
                  f"from active round {resumed_active_round}", flush=True)
        history = resume_payload["history"]
        best_wr = float(resume_payload["best_wr"])
        current = Path(resume_payload["best_ckpt"])
        last_round = max(int(row["round"]) for row in history)
        next_round = last_round + 1
        a.seed = base_seed + (1000 * last_round)
        print(f"[iterate] resume: loaded {len(history)} history rows; "
              f"next_round={next_round} best={best_wr:.4f} ckpt={current}", flush=True)
    else:
        current = Path(a.init_checkpoint)
        print("[iterate] round 0: baseline eval of the init prior...", flush=True)
        best_wr = eval_vs_greedy(a, current)
        history = [{"round": 0, "wr": best_wr, "adopted": True, "source": "init_prior"}]
        next_round = 1
        print(f"[iterate] round 0 baseline WR vs greedy = {best_wr:.4f}", flush=True)

    def elapsed_seconds() -> float:
        return durable_elapsed_seconds + (_time.perf_counter() - _loop_start)

    def budget_state(active_round: int | None = None, active_started_at_utc: str = "") -> dict:
        state = {"elapsed_seconds": round(elapsed_seconds(), 3)}
        if active_round is not None:
            state["active_round"] = int(active_round)
            state["active_attempt_started_at_utc"] = active_started_at_utc or _utc_now_text()

        return state

    def persist_history(active_round: int | None = None, active_started_at_utc: str = "") -> None:
        _write_history(root, {"init": str(a.init_checkpoint), "best_wr": best_wr,
                              "best_ckpt": str(current), "history": history,
                              "budget": budget_state(active_round, active_started_at_utc)})

    persist_history()

    for r in range(next_round, int(a.rounds) + 1):
        if a.max_minutes and elapsed_seconds() / 60.0 >= float(a.max_minutes):
            print(f"[iterate] wall-clock cap {a.max_minutes}min reached before round {r}; stopping.", flush=True)
            break
        print(f"\n[iterate] === round {r} ===", flush=True)
        data_dir = root / f"r{r}_data"
        ckpt_dir = root / f"r{r}_ckpt"
        _clear_partial_round_dir(root, data_dir)
        _clear_partial_round_dir(root, ckpt_dir)
        active_started_at_utc = _utc_now_text()
        persist_history(active_round=r, active_started_at_utc=active_started_at_utc)
        a.seed += 1000  # fresh self-play each round
        dm = datagen(a, current, data_dir)
        _assert_safety_clean(dm)
        new_ckpt = awr_train(a, current, data_dir, ckpt_dir)
        wr = eval_vs_greedy(a, new_ckpt)
        adopt = wr >= best_wr  # hill-climb; ties adopt (fresher policy)
        selfplay_wr = dm.get("win_rate", dm.get("our_win_rate"))
        print(f"[iterate] round {r}: selfplay_wr={selfplay_wr} new_ckpt_wr={wr:.4f} "
              f"best={best_wr:.4f} -> {'ADOPT' if adopt else 'reject'}", flush=True)
        if adopt:
            current, best_wr = new_ckpt, wr
        history.append({"round": r, "wr": wr, "selfplay_wr": selfplay_wr,
                        "adopted": adopt, "best_wr": best_wr, "ckpt": str(new_ckpt)})
        persist_history()
        if a.max_minutes and elapsed_seconds() / 60.0 >= float(a.max_minutes):
            print(f"[iterate] wall-clock cap {a.max_minutes}min reached after round {r}; stopping.", flush=True)
            break

    print(f"\n[iterate] DONE. baseline={history[0]['wr']:.4f} -> best={best_wr:.4f} "
          f"({'IMPROVED' if best_wr > history[0]['wr'] else 'no improvement'}). best_ckpt={current}", flush=True)
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
