"""Modal: run the AWR policy-iteration loop on a CPU container for N hours (BUDGET-GUARDED).

Runs `run_r18_awr_iterate` inside one Modal CPU container (cpu cores parallelize the
self-play data-gen via --workers) for a wall-clock cap, then pulls the best checkpoint
+ history back locally. CPU-ONLY (never GPU: the engine is CPU-native, GPU would just
burn credits). Prints a cost estimate and refuses to exceed --max-usd without --yes.

Coherence gate: refuses to train a deck that fails check_deck_coherence (dead cards).

    modal run scripts/modal_train.py --deck bd27 --minutes 240 --cpu 4 --max-usd 5
    modal run scripts/modal_train.py --deck bd27 --minutes 5 --cpu 2 --max-usd 0.5   # cheap smoke
    modal run scripts/modal_train.py --deck starmie --league-config .research/curriculum/league.example.yaml --minutes 240 --cpu 4 --max-usd 5
"""
from __future__ import annotations

import io
import json
import re
import sys
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
USD_PER_CORE_HOUR = 0.135
NONPREEMPTIBLE_PRICE_MULTIPLIER = 3.0
MODAL_VOLUME_NAME = "deepmew-awr-train"
MODAL_OUT_ROOT = Path("/modal_out")
R4F = ROOT / ".research/r12/lane1_r4f_focal_reweight/selected_ar_bc_r4f_reweighted_candidate.pt"
LEAGUE_CONFIG_DIR = ROOT / ".research/curriculum"
TOP_STARYU_DECK = ROOT / ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"
BAND_TABLES = ROOT / ".research/r19/band_replay_tables.json"
# NOTE: this module is imported BOTH locally (to build the image) and INSIDE the
# container (which lacks .research/runs). Guard the glob so container import doesn't
# crash; the container uses the mounted /repo/bucket_pools.json via the function.
_pools_matches = sorted((ROOT / ".research/runs/bucket_pool").rglob("bucket_pools.json"))
POOLS = _pools_matches[-1] if _pools_matches else Path("/repo/bucket_pools.json")
DECKS = {
    "bd27": ROOT / ".research/submissions/20260706-r4f-bd27",
    "abomasnow": ROOT / ".research/submissions/20260706-r4f-canonical",
    "starmie": TOP_STARYU_DECK,
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyyaml", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(str(ROOT / "cg"), "/repo/cg", copy=True)
    .add_local_dir(str(ROOT / "configs"), "/repo/configs", copy=True)
    .add_local_dir(str(ROOT / "deepmew"), "/repo/deepmew", copy=True)
    .add_local_dir(str(ROOT / "scripts"), "/repo/scripts", copy=True)
    .add_local_dir(str(ROOT / "data"), "/repo/data", copy=True)
    .add_local_dir(str(LEAGUE_CONFIG_DIR), "/repo/.research/curriculum", copy=True)
    .add_local_file(str(ROOT / "main.py"), "/repo/main.py", copy=True)
    .add_local_file(str(ROOT / "deck.csv"), "/repo/deck.csv", copy=True)
    .add_local_file(str(R4F), "/repo/init.pt", copy=True)
    .add_local_file(str(R4F), "/repo/champion.pt", copy=True)
    .add_local_file(str(POOLS), "/repo/bucket_pools.json", copy=True)
)
if BAND_TABLES.exists():
    image = image.add_local_file(str(BAND_TABLES), "/repo/band_replay_tables.json", copy=True)
for _bundle in ("20260708-starmie-awr", "20260706-r4f-bd27", "20260706-r4f-canonical", "20260704-v1b-canonical"):
    _path = ROOT / ".research/submissions" / _bundle
    if _path.is_dir():
        image = image.add_local_dir(str(_path), f"/repo/.research/submissions/{_bundle}", copy=True)

app = modal.App("deepmew-train")
TRAIN_VOLUME = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


def _safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")

    return cleaned or "run"


def _container_repo_path(path_text: str, fallback: str) -> str:
    if not path_text:
        return fallback
    path = Path(path_text)
    if path.is_absolute():
        return fallback

    return str(Path("/repo") / path)


def _run_train_loop(deck_tar: bytes, init_ckpt: bytes, opts: dict) -> bytes:
    import subprocess
    bundle = Path("/repo/deck_bundle"); bundle.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(deck_tar)) as t:
        t.extractall(bundle)
    if init_ckpt:
        Path("/repo/init.pt").write_bytes(init_ckpt)
    run_id = _safe_run_id(str(opts["run_id"]))
    out = MODAL_OUT_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    hist = out / "history.json"
    if hist.exists() and not opts.get("resume", True):
        raise RuntimeError(f"durable Modal output already exists for run_id={run_id}; "
                           "use --resume or choose --run-id")
    cmd = [sys.executable, "-u", "/repo/scripts/run_r18_awr_iterate.py",
           "--pools", "/repo/bucket_pools.json", "--init-checkpoint", "/repo/init.pt",
           "--starmie-bundle", str(bundle), "--out-root", str(out),
           "--rounds", "999", "--max-minutes", str(opts["minutes"]),
           "--workers", str(opts["workers"]), "--games", str(opts["games"]),
           "--eval-games", str(opts["eval_games"]), "--eval-pilots", opts["eval_pilots"],
           "--exploiter-top", str(opts["exploiter_top"]), "--max-seconds", str(opts["max_seconds"]),
           "--kl-beta", str(opts["kl_beta"]), "--temperature", str(opts["temperature"]),
           "--champion", "/repo/champion.pt"]
    if opts.get("resume", True):
        cmd += ["--resume"]
    if opts.get("league_config"):
        league_cfg = _container_repo_path(str(opts["league_config"]), "/repo/.research/curriculum/league.example.yaml")
        cmd += ["--league-config", league_cfg,
                "--league-champion", "/repo/champion.pt",
                "--league-replay-tables", "/repo/band_replay_tables.json"]
    else:
        cmd += ["--stages", ",".join(opts["stages"]), "--opp-pilots", ",".join(opts["opp_pilots"])]
        if "replay" in opts["opp_pilots"] and Path("/repo/band_replay_tables.json").exists():
            cmd += ["--replay-tables", "/repo/band_replay_tables.json"]
    if opts.get("eval_deck_binding"):
        cmd += ["--eval-deck-binding", opts["eval_deck_binding"]]
    if opts.get("ladder_weighting_config"):
        cfg = _container_repo_path(str(opts["ladder_weighting_config"]), "")
        cmd += ["--ladder-weighting-config", cfg]
    subprocess.run(cmd, cwd="/repo", check=True)
    TRAIN_VOLUME.commit()
    if not hist.exists():
        return b""
    h = json.loads(hist.read_text())
    best = Path(h["best_ckpt"])
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if best.exists():
            tar.add(best, arcname="best_ckpt.pt")
        tar.add(hist, arcname="history.json")
    return buf.getvalue()


@app.function(image=image, timeout=15600, volumes={str(MODAL_OUT_ROOT): TRAIN_VOLUME})  # cpu set per-call via .with_options
def train_loop(deck_tar: bytes, init_ckpt: bytes, opts: dict) -> bytes:
    return _run_train_loop(deck_tar, init_ckpt, opts)


@app.function(image=image, timeout=15600, volumes={str(MODAL_OUT_ROOT): TRAIN_VOLUME}, nonpreemptible=True)
def train_loop_nonpreemptible(deck_tar: bytes, init_ckpt: bytes, opts: dict) -> bytes:
    return _run_train_loop(deck_tar, init_ckpt, opts)


@app.local_entrypoint()
def main(deck: str = "bd27", minutes: int = 240, cpu: int = 4, workers: int = 4,
         games: int = 16, stages: str = "600-800,800-900",
         opp_pilots: str = "greedy,v2.4,v4,exploiter", eval_pilots: str = "greedy,v2.4,exploiter",
         exploiter_top: int = 4, eval_games: int = 32, max_seconds: float = 600.0,
         kl_beta: float = 0.03, temperature: float = 0.5, max_usd: float = 5.0,
         yes: bool = False, out_dir: str = "", league_config: str = "", eval_deck_binding: str = "",
         run_id: str = "", resume: bool = True, nonpreemptible: bool = False,
         init_checkpoint: str = "", ladder_weighting_config: str = ""):
    if deck not in DECKS:
        print(f"unknown deck {deck}; choices: {list(DECKS)}"); return
    if league_config and deck != "starmie":
        print("[gate] --league-config trains the config.our side; use --deck starmie for the shipped guide config.")
        return
    # coherence gate (local, free)
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_deck_coherence import _load_cards, _resolve_deck, check
    res = check(_resolve_deck(DECKS[deck]), _load_cards())
    if not res["coherent"]:
        print(f"[gate] REFUSING: {deck} is INCOHERENT ({res['dead_slots']} dead slots). Fix the deck first.")
        for cid, nm, n, why in res["dead"]:
            print(f"   {n}x [{cid}] {nm}: {why}")
        return
    price_multiplier = NONPREEMPTIBLE_PRICE_MULTIPLIER if nonpreemptible else 1.0
    est = cpu * (minutes / 60.0) * USD_PER_CORE_HOUR * price_multiplier
    mode = "nonpreemptible" if nonpreemptible else "preemptible-resumable"
    print(f"[budget] {deck}: {cpu} cores x {minutes}min @ ${USD_PER_CORE_HOUR}/core-hr "
          f"x {price_multiplier:g} ({mode}) -> EST ${est:.2f} (cap ${max_usd})")
    if est > max_usd and not yes:
        print(f"[budget] ABORT: ${est:.2f} > --max-usd ${max_usd}. Re-run with --yes or lower --minutes/--cpu.")
        return
    init_path = Path(init_checkpoint) if init_checkpoint else R4F
    if not init_path.exists():
        print(f"[gate] REFUSING: init checkpoint does not exist: {init_path}")
        return
    if not eval_deck_binding and DECKS[deck].suffix == ".csv" and not league_config:
        eval_deck_binding = "deck:/repo/deck_bundle/deck.csv"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        src = DECKS[deck]
        if src.suffix == ".csv":
            t.add(src, arcname="deck.csv")
        else:
            t.add(src, arcname=".")
    out = Path(out_dir) if out_dir else (ROOT / f".research/r18/modal_train_{deck}")
    run_id = _safe_run_id(run_id or out.name)
    opts = {"minutes": minutes, "workers": workers, "games": games,
            "stages": stages.split(","), "opp_pilots": opp_pilots.split(","),
            "eval_pilots": eval_pilots, "exploiter_top": exploiter_top,
            "eval_games": eval_games, "max_seconds": max_seconds,
            "kl_beta": kl_beta, "temperature": temperature,
            "league_config": league_config, "eval_deck_binding": eval_deck_binding,
            "run_id": run_id, "resume": resume,
            "ladder_weighting_config": ladder_weighting_config}
    fn = train_loop_nonpreemptible if nonpreemptible else train_loop
    print(f"[modal] durable volume={MODAL_VOLUME_NAME} run_id={run_id} resume={resume} init={init_path}")
    result = fn.with_options(cpu=cpu).remote(buf.getvalue(), init_path.read_bytes(), opts)
    out.mkdir(parents=True, exist_ok=True)
    if not result:
        print("[modal] no history returned (loop may have failed); check Modal logs.")
        return
    with tarfile.open(fileobj=io.BytesIO(result)) as t:
        t.extractall(out)
    h = json.loads((out / "history.json").read_text())
    print(f"[modal] {deck} DONE: baseline={h['history'][0]['wr']} -> best_wr={h['best_wr']} "
          f"({'IMPROVED' if h['best_wr'] > h['history'][0]['wr'] else 'no improvement'}); ckpt -> {out}/best_ckpt.pt")
