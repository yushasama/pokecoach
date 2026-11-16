"""Modal fan-out for Starmie curriculum self-play data-gen (BUDGET-GUARDED, CPU-ONLY).

Runs `run_r18_curriculum_datagen` shards across N Modal CPU containers, then pulls
the shards back locally. Designed for a TIGHT budget (~$30 credits):
  * CPU ONLY - never requests a GPU (the engine is CPU-native anyway; GPU would just
    burn credits for nothing).
  * Prints an up-front cost ESTIMATE and refuses to run above --max-usd without --yes.
  * Per-container timeout so a hung shard can't drain credits.
  * Light path by default (v2.4 / greedy pilots) so no torch checkpoint is shipped.

Launch (from repo root):
    modal run scripts/modal_datagen.py --workers 8 --games 12 --max-usd 2.0
    modal run scripts/modal_datagen.py --workers 40 --games 12 --yes --max-usd 10.0

Nothing runs until you invoke `modal run`; importing this file is free.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

# Approx Modal on-demand CPU price (physical core-hour). Verify against modal.com/pricing;
# used only for the pre-flight estimate + guardrail, not billing.
USD_PER_CORE_HOUR = 0.135

# League opponent artifacts shipped into the container so the `exploiter` and `replay`
# tiers can run on Modal (not just the light heuristic path). Both are small (~few MB).
CHAMPION = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"
_BAND_TABLE_HITS = sorted((ROOT / ".research/runs/replay_pilots").rglob("band_replay_tables.json"))
BAND_TABLES = _BAND_TABLE_HITS[-1] if _BAND_TABLE_HITS else None

# CPU-only image with just what the data-gen needs (torch is pulled transitively by the
# run_r9/run_r17 import chain; forced to the CPU wheel so we never ship CUDA).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(str(ROOT / "cg"), "/repo/cg", copy=True)
    .add_local_dir(str(ROOT / "deepmew"), "/repo/deepmew", copy=True)
    .add_local_dir(str(ROOT / "scripts"), "/repo/scripts", copy=True)
    .add_local_dir(str(ROOT / "data"), "/repo/data", copy=True)
    .add_local_file(str(ROOT / "main.py"), "/repo/main.py", copy=True)
    .add_local_file(str(ROOT / "deck.csv"), "/repo/deck.csv", copy=True)
)
if CHAMPION.exists():
    image = image.add_local_file(str(CHAMPION), "/repo/champion.pt", copy=True)
if BAND_TABLES is not None:
    image = image.add_local_file(str(BAND_TABLES), "/repo/band_replay_tables.json", copy=True)

# For the YAML-league entrypoint: ship the shipped bundles (reference agents, bd27,
# current Starmie) + the example config so fixed-deck sparring partners run on Modal.
_SUBMISSIONS = ROOT / ".research" / "submissions"
_LEAGUE_YAML = ROOT / ".research" / "curriculum" / "league.example.yaml"
league_image = image
if _SUBMISSIONS.is_dir():
    for _b in ("20260708-starmie-awr", "20260706-r4f-bd27", "20260706-r4f-canonical", "20260704-v1b-canonical"):
        _d = _SUBMISSIONS / _b
        if _d.is_dir():
            league_image = league_image.add_local_dir(str(_d), f"/repo/.research/submissions/{_b}", copy=True)
if _LEAGUE_YAML.exists():
    league_image = league_image.add_local_file(str(_LEAGUE_YAML), "/repo/league.yaml", copy=True)

app = modal.App("deepmew-datagen")


@app.function(image=image, cpu=1.0, timeout=1800, retries=1)
def datagen_shard(shard_index: int, shard_count: int, pools_bytes: bytes,
                  starmie_tar: bytes, opts: dict) -> bytes:
    """Run one worker slice in-container; return its shard.jsonl.gz bytes (or b'')."""
    import subprocess
    Path("/repo/bucket_pools.json").write_bytes(pools_bytes)
    sb = Path("/repo/starmie_bundle"); sb.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(starmie_tar)) as t:
        t.extractall(sb)
    out = Path(f"/repo/out_{shard_index}")
    cmd = [sys.executable, "-u", "/repo/scripts/run_r18_curriculum_datagen.py",
           "--pools", "/repo/bucket_pools.json", "--starmie-bundle", str(sb),
           "--stages", *opts["stages"], "--our-pilot", opts["our_pilot"],
           "--opp-pilots", *opts["opp_pilots"], "--games", str(opts["games"]),
           "--top-n", str(opts["top_n"]), "--exploiter-top", str(opts["exploiter_top"]),
           "--seed", str(opts["seed"]), "--out-dir", str(out),
           "--champion", "/repo/champion.pt",
           "--shard-count", str(shard_count), "--shard-index", str(shard_index)]
    if "replay" in opts["opp_pilots"] and Path("/repo/band_replay_tables.json").exists():
        cmd += ["--replay-tables", "/repo/band_replay_tables.json"]
    subprocess.run(cmd, cwd="/repo", check=True)
    shards = sorted((out / "shards").glob("*.jsonl.gz"))
    return b"".join(s.read_bytes() for s in shards)


@app.function(image=league_image, cpu=1.0, timeout=3600, retries=1)
def league_shard(seed: int, pools_bytes: bytes) -> bytes:
    """Run the whole YAML league roster once (one seed = one independent sample set).
    Bundles/champion/tables are baked into `league_image`; return shard bytes."""
    import subprocess
    Path("/repo/bucket_pools.json").write_bytes(pools_bytes)
    out = Path(f"/repo/league_out_{seed}")
    cmd = [sys.executable, "-u", "/repo/scripts/run_league_datagen.py",
           "--config", "/repo/league.yaml", "--pools", "/repo/bucket_pools.json",
           "--champion", "/repo/champion.pt", "--replay-tables", "/repo/band_replay_tables.json",
           "--seed", str(seed), "--out-dir", str(out)]
    subprocess.run(cmd, cwd="/repo", check=True)
    shards = sorted((out / "shards").glob("*.jsonl.gz"))
    return b"".join(s.read_bytes() for s in shards)


@app.local_entrypoint()
def league(workers: int = 8, base_seed: int = 181818, est_secs_per_container: float = 1500.0,
           max_usd: float = 6.0, yes: bool = False,
           out_dir: str = "D:/DeepMew_artifacts/datasets/r19/modal_league"):
    """YAML-league data-gen on Modal: N containers each run the full roster with a
    distinct seed (independent samples), shards merged locally. The roster (reference
    agents, bd27, current Starmie, band field) lives in league.example.yaml."""
    est = workers * (est_secs_per_container / 3600.0) * USD_PER_CORE_HOUR
    print(f"[budget] {workers} CPU containers x ~{est_secs_per_container:.0f}s -> EST ${est:.2f} (cap ${max_usd:.2f})")
    if est > max_usd and not yes:
        print(f"[budget] ABORT: est ${est:.2f} > --max-usd ${max_usd:.2f}. Re-run with --yes or lower --workers.")
        return
    pools_bytes = sorted((ROOT / ".research/runs/bucket_pool").rglob("bucket_pools.json"))[-1].read_bytes()
    out = Path(out_dir); (out / "shards").mkdir(parents=True, exist_ok=True)
    args = [(base_seed + i * 1_000_000, pools_bytes) for i in range(workers)]
    n = 0
    for i, chunk in enumerate(league_shard.starmap(args)):
        if chunk:
            (out / "shards" / f"decisions_shard_{i:03d}.jsonl.gz").write_bytes(chunk)
            n += 1
    (out / "manifest.json").write_text(json.dumps(
        {"schema_version": "deepmew.r19.modal_league.v0", "workers": workers,
         "shards": n, "base_seed": base_seed, "est_usd": round(est, 2)}, indent=2), encoding="utf-8")
    print(f"[modal] league: pulled {n} shards -> {out}")


@app.local_entrypoint()
def main(workers: int = 8, games: int = 12, stages: str = "600-800,800-900,900-1000,1000-1100",
         our_pilot: str = "v2.4", opp_pilots: str = "greedy,v2.4,replay,exploiter", top_n: int = 8,
         exploiter_top: int = 4, seed: int = 181818, est_secs_per_container: float = 900.0,
         max_usd: float = 2.0, yes: bool = False,
         out_dir: str = "D:/DeepMew_artifacts/datasets/r18/modal_curriculum"):
    est = workers * (est_secs_per_container / 3600.0) * USD_PER_CORE_HOUR
    print(f"[budget] {workers} CPU containers x ~{est_secs_per_container:.0f}s "
          f"@ ${USD_PER_CORE_HOUR}/core-hr -> EST ${est:.2f} (cap ${max_usd:.2f})")
    if est > max_usd and not yes:
        print(f"[budget] ABORT: estimate ${est:.2f} > --max-usd ${max_usd:.2f}. "
              f"Re-run with --yes to override, or lower --workers/--games.")
        return
    pools_path = sorted((ROOT / ".research/runs/bucket_pool").rglob("bucket_pools.json"))[-1]
    pools_bytes = pools_path.read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        t.add(ROOT / ".research/submissions/20260706-r4f-starmie-18ffc1e6", arcname=".")
    starmie_tar = buf.getvalue()
    opts = {"stages": stages.split(","), "our_pilot": our_pilot,
            "opp_pilots": opp_pilots.split(","), "games": games, "top_n": top_n,
            "exploiter_top": exploiter_top, "seed": seed}
    args = [(i, workers, pools_bytes, starmie_tar, opts) for i in range(workers)]
    out = Path(out_dir); (out / "shards").mkdir(parents=True, exist_ok=True)
    n = 0
    for i, chunk in enumerate(datagen_shard.starmap(args)):
        if chunk:
            (out / "shards" / f"decisions_shard_{i:03d}.jsonl.gz").write_bytes(chunk)
            n += 1
    (out / "manifest.json").write_text(json.dumps(
        {"schema_version": "deepmew.r18.modal_datagen.v0", "workers": workers,
         "shards": n, "opts": opts, "est_usd": round(est, 2)}, indent=2), encoding="utf-8")
    print(f"[modal] pulled {n} shards -> {out}")
