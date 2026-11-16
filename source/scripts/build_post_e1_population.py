"""Build the frozen post-E1 population registry and connected payoff evidence.

Reads only already-completed evaluation artifacts (no new games, no paid work):

  - E1 Stage-6 final evaluation      (23,040 games; r7 / D1 / E1)
  - Starmie plain-PPO control panel  (30,720 games; r7 / D1)
  - Grimmsnarl plain-PPO control     (30,720 games; r7 / Grimmsnarl control)
  - r23 R0 shared panel              ( 2,560 games; r7 / r23 R0 candidate)
  - M3 rejected-Q anchored panel     (30,720 games; r7 / M3)
  - deck-control fixed-v2.4 panel    ( 5,760 games; v2.4 on D0/D1/D3 decks)
  - r23 R0 direct payoff             (   128 games; r7 vs r23 R0 candidate)

Outputs (all under .research/post_e1_population_v0/):

  POST_E1_POPULATION_REGISTRY.json
  POST_E1_EXISTING_EVALUATION_INVENTORY.json
  POST_E1_REFERENCE_PAYOFF_MATRIX.json
  POST_E1_CONNECTEDNESS.json
  POST_E1_RATINGS.json
  POST_E1_COMPETENCE_BUCKETS.json
  POST_E1_DECK_FEATURES.json
  POST_E1_GAME_SHAPE_FEATURES.json

Member identity rule (deepmew.post_e1_member_id.v1):
  member_id = "pm1-" + SHA256(policy_key + "|" + deck_key)
  policy_key = lowercase checkpoint/bundle SHA-256 when the policy is a
               trained artifact, else "named:<policy_name>:<implementation_sha256>"
  deck_key   = lowercase deck SHA-256 (deck file SHA-256 when a deck file
               exists, else the normalized 64-hex deck identity from the pool)

Usage (from repo root):
  python scripts/build_post_e1_population.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / ".research" / "post_e1_population_v0"

E1_DIR = REPO / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "e1_starmie_complete_enhanced_generation_20260714"
STAR_CTRL = REPO / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "control_closeout_starmie_20260713" / "anchored_eval"
GRIM_CTRL = REPO / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "control_closeout_grimmsnarl_20260713" / "anchored_eval"
R23_PANEL = REPO / ".research" / "league_temporal_rl" / "r23-next-after-l0-l2-20260710" / "r0_shared_panel"
R23_PAYOFF = REPO / ".research" / "league_temporal_rl" / "r23-next-after-l0-l2-20260710" / "r0_payoff"
M3_PANEL = REPO / ".research" / "q_guided_starmie" / "anchored_eval_m3_20260713"
DECK_CTRL = REPO / ".research" / "deck_control_v0" / "d0_d1_d3_v24_20260712"
BUCKET_POOL = Path("D:/DeepMew/.research/runs/bucket_pool/20260707/20260707T141406Z-bucket_pool-1c29c3dd/bucket_pools.json")
CARD_CSV = REPO / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"

STARMIE_DECK_CSV = REPO / ".research" / "curriculum" / "decks" / "top_staryu_windecks_b15853cdfb83.deck.csv"
ROOT_DECK_CSV = REPO / "deck.csv"
D2_DECK_CSV = REPO / ".research" / "opponent_decks" / "01641f8a-79c0-11f1-bdeb-0242ac130203" / "0-Eduardo_Rocha_de_Andrade" / "deck.csv"

SHARED_IMPL_SHA = "84b31402d3d05f7ec4a143b66df067310cffeb01151f883dedcc5acbe6e1b1ba"
STARMIE_DECK_FILE_SHA = "2636fcaf8aedd7a156d0b2e33185b0e3878a9c2e2b77425f9203cc53e96e7b28"
STARMIE_DECK_NORMALIZED = "b15853cdfb8304f39f879638102a7fd161f1102d81e6d3f36d5beca4db9e01b0"
EXPLOITER_CKPT_SHA = "ffd5dcf3104b015a5e11ed68bc194dccc0bbac072055a12f672f1b9236fddb8a"

WLD = ("wins", "losses", "draws")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def member_id(policy_key: str, deck_key: str) -> str:
    return "pm1-" + hashlib.sha256(f"{policy_key}|{deck_key}".encode("utf-8")).hexdigest()


def wilson95(wins: int, losses: int) -> list[float] | None:
    n = wins + losses
    if n == 0:
        return None
    p = wins / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 6), round(center + half, 6)]


def rates(w: int, l: int, d: int) -> dict:
    completed = w + l + d
    decided = w + l
    return {
        "wins": w,
        "losses": l,
        "draws": d,
        "completed": completed,
        "point_rate": round((w + 0.5 * d) / completed, 6) if completed else None,
        "decisive_win_rate": round(w / decided, 6) if decided else None,
        "wilson95_decisive": wilson95(w, l),
    }


# ---------------------------------------------------------------- game parsing

def parse_games(path: Path, label_to_alias: dict[str, str], source_id: str):
    """Return per (alias, cell_id) accumulation dicts from a games.jsonl file."""
    acc: dict[tuple[str, str], dict] = {}
    safety = {"crashes": 0, "timeouts": 0, "illegal": 0, "errors": 0, "games": 0}
    turn_lists: dict[tuple[str, str], dict] = defaultdict(lambda: {"win": [], "loss": [], "draw": []})
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            alias = label_to_alias.get(r["participant"])
            if alias is None:
                continue
            cell = r["cell_id"]
            key = (alias, cell)
            a = acc.get(key)
            if a is None:
                a = acc[key] = {
                    "source": source_id,
                    "wins": 0, "losses": 0, "draws": 0,
                    "crashes": 0, "timeouts": 0, "illegal": 0,
                    "by_seed": defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0}),
                    "by_seat": {0: {"wins": 0, "losses": 0, "draws": 0},
                                1: {"wins": 0, "losses": 0, "draws": 0}},
                }
            out = r["outcome"]
            slot = {"win": "wins", "loss": "losses", "draw": "draws"}[out]
            a[slot] += 1
            a["by_seed"][str(r["panel_seed"])][slot] += 1
            a["by_seat"][int(r["candidate_seat"])][slot] += 1
            if r.get("crashed"):
                a["crashes"] += 1
                safety["crashes"] += 1
            if r.get("timed_out"):
                a["timeouts"] += 1
                safety["timeouts"] += 1
            if r.get("illegal"):
                a["illegal"] += 1
                safety["illegal"] += 1
            if r.get("error"):
                safety["errors"] += 1
            safety["games"] += 1
            t = r.get("turns")
            if isinstance(t, int):
                turn_lists[key][out].append(t)
    return acc, safety, turn_lists


def evidence_record(a: dict) -> dict:
    rec = rates(a["wins"], a["losses"], a["draws"])
    rec["source"] = a["source"]
    rec["safety"] = {"crashes": a["crashes"], "timeouts": a["timeouts"], "illegal": a["illegal"]}
    rec["by_seed"] = {
        seed: rates(v["wins"], v["losses"], v["draws"]) for seed, v in sorted(a["by_seed"].items())
    }
    rec["by_seat"] = {
        str(seat): rates(v["wins"], v["losses"], v["draws"]) for seat, v in sorted(a["by_seat"].items())
    }
    return rec


# ---------------------------------------------------------------- main build

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    e1_cfg = json.load(open(E1_DIR / "E1_FINAL_EVALUATION_23040_CONFIG.json", encoding="utf-8"))
    cells_cfg = e1_cfg["configuration"]["cells"]
    e1_results_path = E1_DIR / "production_e1_20260716" / "stage_06_final_evaluation.results.json"
    e1_games_path = E1_DIR / "production_e1_20260716" / "stage_06_final_evaluation.games.jsonl"

    root_deck_sha = sha256_file(ROOT_DECK_CSV)
    starmie_deck_sha = sha256_file(STARMIE_DECK_CSV)
    d2_deck_sha = sha256_file(D2_DECK_CSV) if D2_DECK_CSV.exists() else None

    # ---------------- reference cell members (the frozen anchored panel) -----
    pilot_policy_key = {
        "greedy": f"named:greedy:{SHARED_IMPL_SHA}",
        "v2.4": f"named:v2.4:{SHARED_IMPL_SHA}",
        "exploiter": EXPLOITER_CKPT_SHA,
    }
    cell_members = []
    cell_member_by_id: dict[str, dict] = {}
    for c in cells_cfg:
        pk = pilot_policy_key[c["opponent_policy_identity"]]
        dk = c["opponent_deck_identity"]
        mid = member_id(pk, dk)
        m = {
            "member_id": mid,
            "alias": f"REF|{c['shell']}|{c['opponent_policy_identity']}",
            "cell_id": c["cell_id"],
            "cell_index": c["cell_index"],
            "role": "REFERENCE",
            "status": "FROZEN_ACTIVE_REFERENCE",
            "policy_identity": c["opponent_policy_identity"],
            "policy_key": pk,
            "policy_checkpoint_sha256": c.get("opponent_policy_checkpoint_sha256"),
            "policy_implementation_sha256": c.get("opponent_policy_implementation_sha256"),
            "deck_identity_normalized": dk,
            "deck_sha256": dk,
            "shell": c["shell"],
            "anchored_bucket": c["anchored_bucket"],
            "parent_lineage": None,
            "action_schema": "cg.select_context.v_current",
            "observation_schema": "public_engine_state",
            "safety_status": "clean_in_all_recovered_runs",
        }
        cell_members.append(m)
        cell_member_by_id[c["cell_id"]] = m

    # ---------------- participant members ------------------------------------
    participants = {
        "r7": {
            "alias": "r7",
            "role": "HISTORICAL",
            "status": "FROZEN_HISTORICAL_PARENT_REFERENCE",
            "policy_key": "e293433756fc7edad865f887f9e130d5cec1e4d4067da30446ddebf9b714aee2",
            "policy_checkpoint_sha256": "e293433756fc7edad865f887f9e130d5cec1e4d4067da30446ddebf9b714aee2",
            "policy_checkpoint_path": "D:/DeepMew_artifacts/checkpoints/r22-staryu-full-league-a4-8c-r7-BEST.pt",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_identity_normalized": STARMIE_DECK_NORMALIZED,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "parent_lineage": ["r18-starmie-awr-iter2", "r22-staryu-full-league round 7"],
            "existing_league_member_id": "lm1-7bedd0a0851b5df1fa6aa83d1f437c0b",
        },
        "D1": {
            "alias": "D1",
            "role": "GENERALIST",
            "status": "CURRENT_STRONGEST_BROAD_REFERENCE",
            "policy_key": "b402035e53c3dfa87d6b0e5c962bb61269ffbb10a3c814b5269f25524abf3564",
            "policy_checkpoint_sha256": "b402035e53c3dfa87d6b0e5c962bb61269ffbb10a3c814b5269f25524abf3564",
            "policy_checkpoint_path": ".research/terminal_league_starmie_grimmsnarl_v0/control_closeout_starmie_20260713/remote/final_actor.pt",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_identity_normalized": STARMIE_DECK_NORMALIZED,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "parent_lineage": ["r7 lineage", "plain league-PPO fixed 21,600s floor (DEC-093/094)"],
            "existing_league_member_id": None,
        },
        "E1": {
            "alias": "E1",
            "role": "GENERALIST",
            "role_note": "behaviorally distinct enhanced generation; FAIL_COMPLETE_ENHANCED vs D1; permanent frozen opponent and possible expert/hybrid parent",
            "status": "FROZEN_EVALUATED_DISTINCT_OPPONENT",
            "policy_key": "0473d802de80d590eddb45da5289028b428627f23713b482b9c0229fb4b1cd85",
            "policy_checkpoint_sha256": "0473d802de80d590eddb45da5289028b428627f23713b482b9c0229fb4b1cd85",
            "policy_checkpoint_path": ".research/terminal_league_starmie_grimmsnarl_v0/e1_starmie_complete_enhanced_generation_20260714/production_e1_20260716/stage_05_terminal_ppo.pt.actor.pt",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_identity_normalized": STARMIE_DECK_NORMALIZED,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "parent_lineage": ["r7 lineage (must_not_initialize_from_d1)",
                               "V0 + corrected replay + branch-Q + curriculum + terminal PPO (DEC-101)"],
            "existing_league_member_id": None,
        },
        "GRIMM_CONTROL": {
            "alias": "GRIMM_CONTROL",
            "role": "HISTORICAL",
            "status": "FROZEN_EVALUATED_CONTROL_UNSELECTED",
            "policy_key": "f4019faba273b41689f776d83e9ab76674f03d1d0aa2612b18ef0e144902d5c0",
            "policy_checkpoint_sha256": "f4019faba273b41689f776d83e9ab76674f03d1d0aa2612b18ef0e144902d5c0",
            "policy_checkpoint_path": ".research/terminal_league_starmie_grimmsnarl_v0/control_closeout_grimmsnarl_20260713/remote/final_actor.pt",
            "deck_sha256": "c960f71296a4ca797d8b8421f8c3d059752296ba34721108e6b625e02ca0410a",
            "deck_identity_normalized": None,
            "deck_path": str(D2_DECK_CSV.relative_to(REPO)),
            "shell": "Marnie's Grimmsnarl ex",
            "parent_lineage": ["plain league-PPO fixed 14,400s floor (DEC-093/094)"],
            "existing_league_member_id": None,
        },
        "R23_R0_CANDIDATE": {
            "alias": "R23_R0_CANDIDATE",
            "role": "HISTORICAL",
            "status": "FROZEN_REJECTED_CANDIDATE",
            "policy_key": "4e02475f3b99e1ceac9385bd8c65a2eaea095bbea0e28ba1a9d762a56ea77ba0",
            "policy_checkpoint_sha256": "4e02475f3b99e1ceac9385bd8c65a2eaea095bbea0e28ba1a9d762a56ea77ba0",
            "policy_checkpoint_path": ".research/league_temporal_rl/r23-next-after-l0-l2-20260710/r0_train/selected_ar_bc_r4f_candidate.pt",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_identity_normalized": STARMIE_DECK_NORMALIZED,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "parent_lineage": ["r7 lineage; r23 strict generation R0 (DEC-089)"],
            "existing_league_member_id": "lm1-c8e920c2207b4f87cf1c7101cb5a9d65",
        },
        "M3_REJECTED_Q": {
            "alias": "M3_REJECTED_Q",
            "role": "HISTORICAL",
            "status": "FROZEN_REJECTED_DIAGNOSTIC_CONTROL",
            "policy_key": "13408160a7b43022243c21af205cbfa6fdd537354bdfdba88a81b8f6b1eb2b34",
            "policy_checkpoint_sha256": "13408160a7b43022243c21af205cbfa6fdd537354bdfdba88a81b8f6b1eb2b34",
            "policy_checkpoint_path": "D:/DeepMew_artifacts/checkpoints/q-guided-starmie-m3-candidate-134081*.pt",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_identity_normalized": STARMIE_DECK_NORMALIZED,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "parent_lineage": ["tiny-data Q first wave M3 (DEC-092)"],
            "existing_league_member_id": None,
        },
        "V24_ON_D0_BUNDLE_DECK": {
            "alias": "V24_ON_D0_BUNDLE_DECK",
            "role": "REFERENCE",
            "status": "FROZEN_DECK_SUBSTRATE_CONTROL",
            "policy_key": f"named:v2.4:{SHARED_IMPL_SHA}",
            "policy_checkpoint_sha256": None,
            "policy_checkpoint_path": None,
            "deck_sha256": "f265b59b3ca7b524e593e9edeaae9a76f31d5179705947cbb35da2bab1ec9068",
            "deck_identity_normalized": None,
            "deck_path": ".research/submissions/20260708-starmie-awr.tar.gz (bundle deck)",
            "shell": "exact submitted Starmie control bundle deck (incoherent Cinderace/Staryu)",
            "parent_lineage": ["deck_control_v0 D0 (DEC-091)"],
            "existing_league_member_id": None,
        },
        "V24_ON_STARMIE_DECK": {
            "alias": "V24_ON_STARMIE_DECK",
            "role": "REFERENCE",
            "status": "FROZEN_DECK_SUBSTRATE_CONTROL",
            "policy_key": f"named:v2.4:{SHARED_IMPL_SHA}",
            "policy_checkpoint_sha256": None,
            "policy_checkpoint_path": None,
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_identity_normalized": STARMIE_DECK_NORMALIZED,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "parent_lineage": ["deck_control_v0 D1 arm (DEC-091)"],
            "existing_league_member_id": None,
        },
        "V24_ON_LUCARIO_DECK": {
            "alias": "V24_ON_LUCARIO_DECK",
            "role": "REFERENCE",
            "status": "FROZEN_DECK_SUBSTRATE_CONTROL",
            "policy_key": f"named:v2.4:{SHARED_IMPL_SHA}",
            "policy_checkpoint_sha256": None,
            "policy_checkpoint_path": None,
            "deck_sha256": "7d3bdf2379fe66cf7bbbc75feb550ddacc6918db11a28ae3de570312483daecc",
            "deck_identity_normalized": None,
            "deck_path": ".research/submissions/20260706-r4f-bd27.tar.gz (bundle deck)",
            "shell": "Mega Lucario (bd27)",
            "parent_lineage": ["deck_control_v0 D3 arm (DEC-091)"],
            "existing_league_member_id": None,
        },
    }
    for alias, p in participants.items():
        p["member_id"] = member_id(p["policy_key"], p["deck_sha256"])
        p["action_schema"] = "cg.select_context.v_current"
        p["observation_schema"] = ("starmie feature_policy_id public observation"
                                   if p["policy_checkpoint_sha256"] else "public_engine_state")
        p["safety_status"] = "clean_in_all_recovered_runs"

    # members registered without panel evidence (identity-only registration)
    extra_members = [
        {
            "alias": "greedy-root-anchor",
            "role": "REFERENCE",
            "status": "REGISTERED_NO_ANCHORED_PANEL_EDGES",
            "policy_key": f"named:greedy:{SHARED_IMPL_SHA}",
            "policy_checkpoint_sha256": None,
            "deck_sha256": root_deck_sha,
            "deck_path": "deck.csv",
            "shell": "root Abomasnow deck",
            "existing_league_member_id": "lm1-aa8fd1a5ee079c1b28381fae1d9dcfa0",
            "parent_lineage": None,
        },
        {
            "alias": "v2.4-root-calibration",
            "role": "REFERENCE",
            "status": "REGISTERED_NO_ANCHORED_PANEL_EDGES",
            "policy_key": f"named:v2.4:{SHARED_IMPL_SHA}",
            "policy_checkpoint_sha256": None,
            "deck_sha256": root_deck_sha,
            "deck_path": "deck.csv",
            "shell": "root Abomasnow deck",
            "existing_league_member_id": "lm1-3cfb093616778552fd3d8e633349ecbe",
            "parent_lineage": None,
        },
        {
            "alias": "r22-from-r7-r5-candidate",
            "role": "HISTORICAL",
            "status": "REGISTERED_NO_ANCHORED_PANEL_EDGES",
            "policy_key": "81ae2aa09ecf09c1e5bcab8f4de87002737f7d187a47effd744209a1de538c8d",
            "policy_checkpoint_sha256": "81ae2aa09ecf09c1e5bcab8f4de87002737f7d187a47effd744209a1de538c8d",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "existing_league_member_id": "lm1-1165402f88f757b222e757c278878d5e",
            "parent_lineage": ["r7 continuation round 5 (DEC-074)"],
        },
        {
            "alias": "r18-r2-historical-bundle",
            "role": "HISTORICAL",
            "status": "REGISTERED_NO_ANCHORED_PANEL_EDGES",
            "policy_key": "4a34de9ee7011bcbcb5d0e91c8fbcfd534b12b04eb001458c95a80969b6d0eb1",
            "policy_checkpoint_sha256": "4a34de9ee7011bcbcb5d0e91c8fbcfd534b12b04eb001458c95a80969b6d0eb1",
            "deck_sha256": STARMIE_DECK_FILE_SHA,
            "deck_path": str(STARMIE_DECK_CSV.relative_to(REPO)),
            "shell": "Staryu/Mega Starmie",
            "existing_league_member_id": "lm1-29d39e615a544d5476917f7c6fb34671",
            "parent_lineage": ["r18 AWR iterate round 2"],
        },
        {
            "alias": "r12-lane1-exploiter-checkpoint",
            "role": "REFERENCE",
            "role_note": "champion-exploiter pilot used inside the anchored panel exploiter cells",
            "status": "FROZEN_ACTIVE_REFERENCE_PILOT",
            "policy_key": EXPLOITER_CKPT_SHA,
            "policy_checkpoint_sha256": EXPLOITER_CKPT_SHA,
            "deck_sha256": None,
            "deck_path": "piloted panel bucket decks (see exploiter cells)",
            "shell": None,
            "existing_league_member_id": None,
            "parent_lineage": ["r12 lane1 focal reweight selected candidate"],
        },
    ]
    for m in extra_members:
        m["member_id"] = member_id(m["policy_key"], m["deck_sha256"] or "no_fixed_deck")
        m["action_schema"] = "cg.select_context.v_current"
        m["observation_schema"] = "public_engine_state"
        m["safety_status"] = "clean_in_all_recovered_runs"

    registry = {
        "schema_version": "deepmew.post_e1_population_registry.v1",
        "created_on": "2026-07-17",
        "member_id_rule": {
            "id": "deepmew.post_e1_member_id.v1",
            "rule": "member_id = 'pm1-' + SHA256(policy_key + '|' + deck_key); policy_key = checkpoint/bundle sha256 or 'named:<name>:<impl_sha256>'; deck_key = deck file sha256 else normalized deck identity else 'no_fixed_deck'",
            "note": "human labels (D1, E1, r7, greedy, v2.4, ...) are aliases only; existing lm1- control-plane ids recorded where present",
        },
        "frozen_roles": {
            "D1": "CURRENT_STRONGEST_BROAD_REFERENCE",
            "E1": "FROZEN_EVALUATED_DISTINCT_OPPONENT",
            "r7": "FROZEN_HISTORICAL_PARENT_REFERENCE",
        },
        "participants": [participants[k] for k in sorted(participants)],
        "reference_cells": cell_members,
        "additional_members": extra_members,
    }
    json.dump(registry, open(OUT / "POST_E1_POPULATION_REGISTRY.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- parse all games sources --------------------------------
    sources = {}

    acc_e1, safety_e1, turns_e1 = parse_games(
        e1_games_path,
        {"frozen_r7": "r7", "frozen_starmie_d1": "D1", "frozen_e1": "E1"},
        "e1_stage6_final_20260717",
    )
    sources["e1_stage6_final_20260717"] = {
        "path": str(e1_games_path.relative_to(REPO)),
        "results_path": str(e1_results_path.relative_to(REPO)),
        "run_id": "e1-starmie-complete-enhanced-final-20260717",
        "panel_seeds": [613249997, 613250019, 613250041],
        "games_per_participant_cell_seed": 128,
        "participants": ["r7", "D1", "E1"],
        "safety": safety_e1,
        "receipts": [
            str((E1_DIR / "production_e1_20260716" / "STAGE_06_EVALUATION_COMPLETION_RECEIPT.json").relative_to(REPO)),
            str((E1_DIR / "production_e1_20260716" / "STAGE_06_BOUNDARY_RECEIPT.json").relative_to(REPO)),
        ],
    }

    acc_sc, safety_sc, turns_sc = parse_games(
        STAR_CTRL / "games.jsonl",
        {"r22-r7": "r7", "starmie-plain-ppo-control": "D1"},
        "starmie_control_anchored_20260713",
    )
    sources["starmie_control_anchored_20260713"] = {
        "path": str((STAR_CTRL / "games.jsonl").relative_to(REPO)),
        "results_path": str((STAR_CTRL / "results.json").relative_to(REPO)),
        "run_id": "starmie-plain-ppo-control-anchored-20260713",
        "panel_seeds": [40009, 50021, 60013],
        "games_per_participant_cell_seed": 256,
        "participants": ["r7", "D1"],
        "safety": safety_sc,
        "receipts": [str((STAR_CTRL / "results.json").relative_to(REPO))],
    }

    acc_gc, safety_gc, turns_gc = parse_games(
        GRIM_CTRL / "games.jsonl",
        {"r22-r7": "r7", "grimmsnarl-plain-ppo-control": "GRIMM_CONTROL"},
        "grimmsnarl_control_anchored_20260713",
    )
    sources["grimmsnarl_control_anchored_20260713"] = {
        "path": str((GRIM_CTRL / "games.jsonl").relative_to(REPO)),
        "results_path": str((GRIM_CTRL / "results.json").relative_to(REPO)),
        "run_id": "grimmsnarl-plain-ppo-control-anchored-20260713",
        "panel_seeds": [40009, 50021, 60013],
        "games_per_participant_cell_seed": 256,
        "participants": ["r7", "GRIMM_CONTROL"],
        "safety": safety_gc,
        "receipts": [str((GRIM_CTRL / "results.json").relative_to(REPO))],
    }

    acc_r23, safety_r23, turns_r23 = parse_games(
        R23_PANEL / "games.jsonl",
        {"r22-r7": "r7", "r23-starmie-r0": "R23_R0_CANDIDATE"},
        "r23_r0_shared_panel_20260710",
    )
    sources["r23_r0_shared_panel_20260710"] = {
        "path": str((R23_PANEL / "games.jsonl").relative_to(REPO)),
        "results_path": str((R23_PANEL / "results.json").relative_to(REPO)),
        "run_id": "r23-r0-shared-panel",
        "panel_seeds": [181818],
        "games_per_participant_cell_seed": 64,
        "participants": ["r7", "R23_R0_CANDIDATE"],
        "safety": safety_r23,
        "receipts": [str((R23_PANEL / "results.json").relative_to(REPO))],
    }

    acc_m3, safety_m3, turns_m3 = parse_games(
        M3_PANEL / "games.jsonl",
        {"r22-r7": "r7", "q-guided-starmie-m3": "M3_REJECTED_Q"},
        "m3_anchored_eval_20260713",
    )
    sources["m3_anchored_eval_20260713"] = {
        "path": str((M3_PANEL / "games.jsonl").relative_to(REPO)),
        "results_path": str((M3_PANEL / "results.json").relative_to(REPO)),
        "run_id": "m3-anchored-eval-20260713",
        "panel_seeds": [40009, 50021, 60013],
        "games_per_participant_cell_seed": 256,
        "participants": ["r7", "M3_REJECTED_Q"],
        "safety": safety_m3,
        "receipts": [str((M3_PANEL / "results.json").relative_to(REPO))],
    }

    # deck control: per-cell aggregates only (no games.jsonl)
    dc = json.load(open(DECK_CTRL / "results.json", encoding="utf-8"))
    dc_alias = {"D0": "V24_ON_D0_BUNDLE_DECK", "D1": "V24_ON_STARMIE_DECK", "D3": "V24_ON_LUCARIO_DECK"}
    acc_dc: dict[tuple[str, str], dict] = {}
    for label, alias in dc_alias.items():
        for cell in dc["participants"][label]["cells"]:
            acc_dc[(alias, cell["cell_id"])] = {
                "source": "deck_control_v24_20260712",
                "wins": cell["wins"], "losses": cell["losses"], "draws": cell["draws"],
                "crashes": cell["crashes"], "timeouts": cell["timeouts"], "illegal": cell["illegal"],
                "by_seed": {}, "by_seat": {},
            }
    sources["deck_control_v24_20260712"] = {
        "path": str((DECK_CTRL / "results.json").relative_to(REPO)),
        "results_path": str((DECK_CTRL / "results.json").relative_to(REPO)),
        "run_id": dc.get("run_id"),
        "panel_seeds": [9271201, 9271202, 9271203],
        "games_per_participant_cell_seed": 32,
        "participants": sorted(dc_alias.values()),
        "granularity": "cell_aggregate_only_no_per_game_rows",
        "safety": {"crashes": 0, "timeouts": 0, "illegal": 0, "errors": 0, "games": 5760},
        "receipts": [str((DECK_CTRL / "results.json").relative_to(REPO))],
    }

    # direct payoff edge r7 <-> r23 candidate
    po = json.load(open(R23_PAYOFF / "results.json", encoding="utf-8"))
    pair = po["asymmetric_pairs"][0]
    direct_edges = [{
        "left": "r7",
        "right": "R23_R0_CANDIDATE",
        "left_member_id": participants["r7"]["member_id"],
        "right_member_id": participants["R23_R0_CANDIDATE"]["member_id"],
        "games_per_direction": 64,
        "left_win_rate_as_left": pair["W_left_right"],
        "right_win_rate_as_left": pair["W_right_left"],
        "zero_sum_consistency_gap": pair["zero_sum_consistency_gap"],
        "source": "r23_r0_direct_payoff_20260710",
        "receipt": str((R23_PAYOFF / "results.json").relative_to(REPO)),
    }]
    sources["r23_r0_direct_payoff_20260710"] = {
        "path": str((R23_PAYOFF / "results.json").relative_to(REPO)),
        "results_path": str((R23_PAYOFF / "results.json").relative_to(REPO)),
        "run_id": po.get("run_id"),
        "participants": ["r7", "R23_R0_CANDIDATE"],
        "granularity": "directional_pair_aggregate",
        "safety": {"crashes": 0, "timeouts": 0, "illegal": 0, "errors": 0, "games": 128},
        "receipts": [str((R23_PAYOFF / "results.json").relative_to(REPO))],
    }

    inventory = {
        "schema_version": "deepmew.post_e1_evaluation_inventory.v1",
        "created_on": "2026-07-17",
        "note": "all runs used the identical frozen 20-cell anchored 600-800 panel (same cell_ids) except the direct payoff pair; no game listed here may be re-run under the same identity",
        "sources": sources,
    }
    json.dump(inventory, open(OUT / "POST_E1_EXISTING_EVALUATION_INVENTORY.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- payoff matrix ------------------------------------------
    all_acc = {}
    for acc in (acc_e1, acc_sc, acc_gc, acc_r23, acc_m3, acc_dc):
        for key, a in acc.items():
            all_acc.setdefault(key, []).append(a)

    cell_ids = [c["cell_id"] for c in cells_cfg]
    edges = []
    merged_by_edge: dict[tuple[str, str], dict] = {}
    for (alias, cell_id), recs in sorted(all_acc.items()):
        evidence = [evidence_record(a) for a in recs]
        w = sum(a["wins"] for a in recs)
        l = sum(a["losses"] for a in recs)
        d = sum(a["draws"] for a in recs)
        merged = rates(w, l, d)
        merged_by_edge[(alias, cell_id)] = merged
        edges.append({
            "participant": alias,
            "participant_member_id": participants[alias]["member_id"],
            "reference_cell_id": cell_id,
            "reference_member_id": cell_member_by_id[cell_id]["member_id"],
            "shell": cell_member_by_id[cell_id]["shell"],
            "pilot": cell_member_by_id[cell_id]["policy_identity"],
            "merged": merged,
            "evidence": evidence,
        })

    matrix = {
        "schema_version": "deepmew.post_e1_reference_payoff_matrix.v1",
        "created_on": "2026-07-17",
        "orientation": "rates are from the participant (row) perspective vs the reference cell (column)",
        "cells": cell_ids,
        "participant_reference_edges": edges,
        "direct_participant_edges": direct_edges,
    }
    json.dump(matrix, open(OUT / "POST_E1_REFERENCE_PAYOFF_MATRIX.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- connectedness ------------------------------------------
    node_edges = defaultdict(set)
    for (alias, cell_id) in merged_by_edge:
        node_edges[alias].add(cell_id)
        node_edges[cell_id].add(alias)
    node_edges["r7"].add("R23_R0_CANDIDATE")
    node_edges["R23_R0_CANDIDATE"].add("r7")
    seen, comps = set(), []
    for n in sorted(node_edges):
        if n in seen:
            continue
        comp, stack = [], [n]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(node_edges[x] - seen)
        comps.append(sorted(comp))
    participant_aliases = sorted(participants)
    covered = {a: sorted({c for (p, c) in merged_by_edge if p == a}) for a in participant_aliases}
    missing_panel_edges = {a: sorted(set(cell_ids) - set(covered[a])) for a in participant_aliases}
    no_edge_members = [m["alias"] for m in extra_members if m["status"].startswith("REGISTERED_NO")]
    connectedness = {
        "schema_version": "deepmew.post_e1_connectedness.v1",
        "created_on": "2026-07-17",
        "components": comps,
        "single_connected_component": len(comps) == 1,
        "panel_cell_coverage_per_participant": {a: len(covered[a]) for a in participant_aliases},
        "missing_panel_edges": missing_panel_edges,
        "members_with_no_payoff_edges": no_edge_members,
        "absent_direct_participant_pairs": [
            ["D1", "E1"], ["D1", "r7"], ["E1", "r7"],
            ["D1", "GRIMM_CONTROL"], ["E1", "GRIMM_CONTROL"],
        ],
        "note": "absent direct pairs are inferable through the shared frozen panel; they are optional evidence, not required by the anchored-panel mission definition",
    }
    json.dump(connectedness, open(OUT / "POST_E1_CONNECTEDNESS.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- FROZEN_PANEL_BT_INDEX ----------------------------------
    nodes = participant_aliases + cell_ids
    idx = {n: i for i, n in enumerate(nodes)}
    pair_wl: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (alias, cell_id), m in merged_by_edge.items():
        i, j = idx[alias], idx[cell_id]
        w = m["wins"] + 0.5 * m["draws"]
        l = m["losses"] + 0.5 * m["draws"]
        pair_wl[(i, j)][0] += w
        pair_wl[(i, j)][1] += l
    # direct pair: r7 34-30 as left, candidate 34-30 as left -> r7 64W 64L total
    i, j = idx["r7"], idx["R23_R0_CANDIDATE"]
    pair_wl[(i, j)][0] += 64.0
    pair_wl[(i, j)][1] += 64.0

    n = len(nodes)
    theta = np.zeros(n)
    lam = 1e-6
    for _ in range(500):
        grad = -2 * lam * theta
        H = np.full((n, n), 0.0)
        np.fill_diagonal(H, -2 * lam)
        for (i, j), (w, l) in pair_wl.items():
            p = 1.0 / (1.0 + math.exp(-(theta[i] - theta[j])))
            g = w - (w + l) * p
            grad[i] += g
            grad[j] -= g
            h = (w + l) * p * (1 - p)
            H[i, i] -= h
            H[j, j] -= h
            H[i, j] += h
            H[j, i] += h
        step = np.linalg.solve(H - 1e-9 * np.eye(n), grad)
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    ref_mean = float(np.mean([theta[idx[c]] for c in cell_ids]))
    theta = theta - ref_mean
    fisher = np.zeros((n, n))
    for (i, j), (w, l) in pair_wl.items():
        p = 1.0 / (1.0 + math.exp(-(theta[i] - theta[j])))
        h = (w + l) * p * (1 - p)
        fisher[i, i] += h
        fisher[j, j] += h
        fisher[i, j] -= h
        fisher[j, i] -= h
    cov = np.linalg.pinv(fisher)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    index_per_logit = 400.0 / math.log(10.0)
    total_games_by_node = defaultdict(float)
    for (i, j), (w, l) in pair_wl.items():
        total_games_by_node[i] += w + l
        total_games_by_node[j] += w + l
    ratings = {
        "schema_version": "deepmew.post_e1_frozen_panel_bt_index.v1",
        "created_on": "2026-07-17",
        "name": "FROZEN_PANEL_BT_INDEX",
        "model": "Bradley-Terry on merged payoff edges; draws counted as half-win half-loss; L2 reg 1e-6; anchor: mean of the 20 reference-cell strengths = 0; panel-relative display baseline = 1000",
        "semantic_boundary": "panel-relative competence index only; not a Kaggle ladder rating and not externally comparable",
        "index_per_logit": index_per_logit,
        "external_ladder_rating_equivalence": False,
        "kaggle_ladder_rating": False,
        "not_terminal_reward": True,
        "nodes": [
            {
                "node": nname,
                "kind": "participant" if nname in participants else "reference_cell",
                "member_id": (participants[nname]["member_id"] if nname in participants
                              else cell_member_by_id[nname]["member_id"]),
                "theta": round(float(theta[idx[nname]]), 6),
                "theta_se": round(float(se[idx[nname]]), 6),
                "frozen_panel_bt_index": round(1000.0 + index_per_logit * float(theta[idx[nname]]), 2),
                "frozen_panel_bt_index_se": round(index_per_logit * float(se[idx[nname]]), 2),
                "weighted_games": int(total_games_by_node[idx[nname]]),
            }
            for nname in nodes
        ],
    }
    ratings["nodes"].sort(key=lambda r: -r["theta"])
    json.dump(ratings, open(OUT / "POST_E1_RATINGS.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- competence buckets -------------------------------------
    def field_value(alias: str, source_filter=None):
        cell_prs, var_terms = [], []
        for cid in cell_ids:
            recs = all_acc.get((alias, cid))
            if not recs:
                return None, None, 0
            if source_filter:
                recs = [a for a in recs if a["source"] in source_filter]
                if not recs:
                    return None, None, 0
            w = sum(a["wins"] for a in recs)
            l = sum(a["losses"] for a in recs)
            d = sum(a["draws"] for a in recs)
            ng = w + l + d
            pr = (w + 0.5 * d) / ng
            cell_prs.append(pr)
            var_terms.append(pr * (1 - pr) / max(ng, 1))
        v = sum(cell_prs) / len(cell_prs)
        se_v = math.sqrt(sum(var_terms)) / len(cell_prs)
        games = sum(sum(a["wins"] + a["losses"] + a["draws"] for a in all_acc[(alias, cid)])
                    for cid in cell_ids)
        return v, se_v, games

    THRESHOLDS = [("B0_FIELD_FRONTIER", 0.02), ("B1_STRONG_ADAPTED", 0.08),
                  ("B2_INTERMEDIATE", 0.20), ("B3_WEAK_OR_REFERENCE", float("inf"))]

    def bucket_for(regret: float) -> str:
        for name, ub in THRESHOLDS:
            if regret <= ub:
                return name
        return "B3_WEAK_OR_REFERENCE"

    field_rows = []
    for alias in participant_aliases:
        v, se_v, games = field_value(alias)
        if v is None:
            continue
        field_rows.append({"member": alias, "V": v, "V_se": se_v, "games": games})
    v_best_row = max(field_rows, key=lambda r: r["V"])
    for r in field_rows:
        r["field_regret"] = v_best_row["V"] - r["V"]
        r["field_regret_se"] = math.sqrt(r["V_se"] ** 2 + v_best_row["V_se"] ** 2) if r is not v_best_row else 0.0
        r["min_support_met"] = r["games"] >= 1920
        r["bucket"] = bucket_for(r["field_regret"]) if r["min_support_met"] else "UNBUCKETED_INSUFFICIENT_SUPPORT"
        for k in ("V", "V_se", "field_regret", "field_regret_se"):
            r[k] = round(r[k], 6)

    # references: field = stage-6 frozen trio (r7, D1, E1), reference perspective
    ref_rows = []
    for cid in cell_ids:
        w = l = d = 0
        for alias in ("r7", "D1", "E1"):
            a = acc_e1[(alias, cid)]
            # reference perspective: reference wins when participant loses
            w += a["losses"]
            l += a["wins"]
            d += a["draws"]
        ng = w + l + d
        pr = (w + 0.5 * d) / ng
        ref_rows.append({"member": cid, "V": pr, "V_se": math.sqrt(pr * (1 - pr) / ng), "games": ng})
    ref_best = max(ref_rows, key=lambda r: r["V"])
    for r in ref_rows:
        r["field_regret"] = ref_best["V"] - r["V"]
        r["field_regret_se"] = math.sqrt(r["V_se"] ** 2 + ref_best["V_se"] ** 2) if r is not ref_best else 0.0
        r["min_support_met"] = r["games"] >= 1152
        r["bucket"] = bucket_for(r["field_regret"]) if r["min_support_met"] else "UNBUCKETED_INSUFFICIENT_SUPPORT"
        for k in ("V", "V_se", "field_regret", "field_regret_se"):
            r[k] = round(r[k], 6)

    participant_buckets = {row["member"]: row["bucket"] for row in field_rows}
    preserved_bucket_assignments = {
        "D1": "B0_FIELD_FRONTIER",
        "E1": "B1_STRONG_ADAPTED",
        "r7": "B2_INTERMEDIATE",
    }
    if any(participant_buckets.get(alias) != bucket for alias, bucket in preserved_bucket_assignments.items()):
        raise ValueError("frozen D1/E1/r7 panel-relative competence buckets drifted")

    buckets = {
        "schema_version": "deepmew.post_e1_competence_buckets.v1",
        "created_on": "2026-07-17",
        "semantic_boundary": "panel-relative competence buckets only; not Kaggle ladder ratings and not externally comparable",
        "preserved_panel_relative_assignments": preserved_bucket_assignments,
        "frozen_rule": {
            "id": "deepmew.field_regret_bucket_rule.v1",
            "field_value": "equal-weight mean of per-cell point rates over the frozen 20-cell anchored panel",
            "regret": "FieldRegret(i,f) = V_best(f) - V_i(f), absolute point-rate units, no rescaling",
            "thresholds": {"B0_FIELD_FRONTIER": "<=0.02", "B1_STRONG_ADAPTED": "<=0.08",
                           "B2_INTERMEDIATE": "<=0.20", "B3_WEAK_OR_REFERENCE": ">0.20"},
            "min_support": {"participants_field": 1920, "references_field": 1152},
            "honesty_caveat": "aggregate point rates for r7/D1/E1 were already public in the E1 closeout before this rule was written down; thresholds are round numbers and are frozen prospectively for all future assignments",
        },
        "fields": {
            "F_ANCHORED_PANEL_20": {
                "description": "participant vs the frozen 20-cell 600-800 anchored panel (merged across recovered runs)",
                "V_best_member": v_best_row["member"],
                "members": sorted(field_rows, key=lambda r: r["field_regret"]),
            },
            "F_STAGE6_PARTICIPANT_TRIO": {
                "description": "reference cell vs the frozen r7/D1/E1 trio, reference perspective, stage-6 games only",
                "V_best_member": ref_best["member"],
                "members": sorted(ref_rows, key=lambda r: r["field_regret"]),
                "caveat": "reference competence conflates deck strength and pilot strength; role stays REFERENCE regardless of bucket",
            },
        },
        "not_terminal_reward": True,
        "role_is_not_competence": True,
    }
    json.dump(buckets, open(OUT / "POST_E1_COMPETENCE_BUCKETS.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- deck features ------------------------------------------
    cards = {}
    with open(CARD_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (ValueError, KeyError):
                continue
            if cid not in cards:
                cards[cid] = row

    def deck_features(card_ids: list[int]) -> dict:
        feats = {
            "cards_total": len(card_ids),
            "unknown_card_ids": 0,
            "pokemon": 0, "trainer": 0, "energy": 0,
            "basic_pokemon": 0, "stage1_pokemon": 0, "stage2_pokemon": 0, "mega_pokemon": 0,
            "ex_rule_cards": 0, "max_hp": 0,
            "energy_types": {},
            "unique_cards": len(set(card_ids)),
        }
        for cid in card_ids:
            row = cards.get(cid)
            if row is None:
                feats["unknown_card_ids"] += 1
                continue
            cat = (row.get("Category") or "").strip().lower()
            stage = (row.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip()
            name = (row.get("Card Name") or "").strip()
            rule = (row.get("Rule") or "").strip().lower()
            if "energy" in name.lower() and stage.lower().startswith("basic energy"):
                feats["energy"] += 1
                t = (row.get("Type") or "").strip()
                feats["energy_types"][t] = feats["energy_types"].get(t, 0) + 1
                continue
            hp = row.get("HP") or ""
            is_pokemon = hp.strip().isdigit()
            if is_pokemon:
                feats["pokemon"] += 1
                feats["max_hp"] = max(feats["max_hp"], int(hp))
                sl = stage.lower()
                if sl.startswith("basic"):
                    feats["basic_pokemon"] += 1
                elif "1" in sl:
                    feats["stage1_pokemon"] += 1
                elif "2" in sl:
                    feats["stage2_pokemon"] += 1
                if "mega" in name.lower():
                    feats["mega_pokemon"] += 1
                if " ex" in name.lower() or "ex" in rule:
                    feats["ex_rule_cards"] += 1
            else:
                feats["trainer"] += 1
        return feats

    pool = json.load(open(BUCKET_POOL, encoding="utf-8")) if BUCKET_POOL.exists() else {"pools": {}}
    pool_rows = {row["deck_hash"]: row for rows in pool["pools"].values() for row in rows}

    deck_entries = []
    seen_hashes = set()
    for c in cells_cfg:
        dh = c["opponent_deck_identity"]
        if dh in seen_hashes:
            continue
        seen_hashes.add(dh)
        row = pool_rows.get(dh)
        entry = {
            "deck_identity": dh,
            "shell": c["shell"],
            "source": "r17_bucket_pool_600_800",
            "replay_provenance": ({"games": row["games"], "wins": row["wins"],
                                   "win_rate": row["win_rate"], "teams": len(row["teams"])}
                                  if row else None),
            "features": deck_features(row["card_ids"]) if row else None,
        }
        deck_entries.append(entry)

    def read_deck_csv(path: Path) -> list[int]:
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().split(",")[0]
            if line.isdigit():
                ids.append(int(line))
        return ids

    for name, path, sha, shell in [
        ("starmie_participant_deck", STARMIE_DECK_CSV, starmie_deck_sha, "Staryu/Mega Starmie"),
        ("root_abomasnow_deck", ROOT_DECK_CSV, root_deck_sha, "Mega Abomasnow (root)"),
        ("grimmsnarl_d2_deck", D2_DECK_CSV, d2_deck_sha, "Marnie's Grimmsnarl ex"),
    ]:
        if not path.exists():
            continue
        deck_entries.append({
            "deck_identity": sha,
            "shell": shell,
            "source": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "replay_provenance": None,
            "features": deck_features(read_deck_csv(path)),
        })

    deck_manifest = {
        "schema_version": "deepmew.post_e1_deck_features.v1",
        "created_on": "2026-07-17",
        "feature_note": "interpretable v0 features from immutable card lists + EN_Card_Data.csv; richer semantic axes (win conditions, acceleration, disruption, prize geometry) remain a K3/meta-findings join, listed as follow-up",
        "decks": deck_entries,
    }
    json.dump(deck_manifest, open(OUT / "POST_E1_DECK_FEATURES.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- game-shape (playstyle v0) features ----------------------
    turn_sources = {
        "e1_stage6_final_20260717": turns_e1,
        "starmie_control_anchored_20260713": turns_sc,
        "grimmsnarl_control_anchored_20260713": turns_gc,
        "r23_r0_shared_panel_20260710": turns_r23,
        "m3_anchored_eval_20260713": turns_m3,
    }
    shape_rows = []
    per_participant: dict[str, dict] = defaultdict(lambda: {"win": [], "loss": [], "draw": []})
    for src, tl in turn_sources.items():
        for (alias, cid), outs in tl.items():
            for out, lst in outs.items():
                per_participant[alias][out].extend(lst)
            allt = outs["win"] + outs["loss"] + outs["draw"]
            if not allt:
                continue
            shape_rows.append({
                "participant": alias, "cell_id": cid, "source": src,
                "games": len(allt),
                "mean_turns": round(statistics.fmean(allt), 3),
                "median_turns": statistics.median(allt),
                "mean_turns_wins": round(statistics.fmean(outs["win"]), 3) if outs["win"] else None,
                "mean_turns_losses": round(statistics.fmean(outs["loss"]), 3) if outs["loss"] else None,
            })
    participant_summaries = {}
    for alias, outs in per_participant.items():
        allt = outs["win"] + outs["loss"] + outs["draw"]
        seat_stats = {}
        for (a2, cid), recs in all_acc.items():
            if a2 != alias:
                continue
            for a in recs:
                for seat, v in a.get("by_seat", {}).items():
                    s = seat_stats.setdefault(seat, {"wins": 0, "losses": 0, "draws": 0})
                    for k in WLD:
                        s[k] += v[k]
        participant_summaries[alias] = {
            "games_with_turns": len(allt),
            "mean_turns": round(statistics.fmean(allt), 3) if allt else None,
            "p25_turns": float(np.percentile(allt, 25)) if allt else None,
            "median_turns": statistics.median(allt) if allt else None,
            "p75_turns": float(np.percentile(allt, 75)) if allt else None,
            "mean_turns_wins": round(statistics.fmean(outs["win"]), 3) if outs["win"] else None,
            "mean_turns_losses": round(statistics.fmean(outs["loss"]), 3) if outs["loss"] else None,
            "draw_rate": round(len(outs["draw"]) / len(allt), 6) if allt else None,
            "by_seat": {str(seat): rates(v["wins"], v["losses"], v["draws"])
                        for seat, v in sorted(seat_stats.items())},
        }
    shape = {
        "schema_version": "deepmew.post_e1_game_shape_features.v1",
        "created_on": "2026-07-17",
        "scope_note": "public-trajectory v0 features only (turn counts, outcomes, seats); action-level playstyle axes (aggression, resource commitment, gust usage, ...) require the prospective actor-visible trace corpus and are preregistered as follow-up, not fabricated here",
        "participant_summaries": {k: participant_summaries[k] for k in sorted(participant_summaries)},
        "participant_cell_rows": sorted(shape_rows, key=lambda r: (r["participant"], r["cell_id"], r["source"])),
    }
    json.dump(shape, open(OUT / "POST_E1_GAME_SHAPE_FEATURES.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)

    # ---------------- console summary ----------------------------------------
    print("post_e1_population_v0 build complete")
    print(f"  members: {len(registry['participants'])} participants, {len(cell_members)} reference cells, {len(extra_members)} additional")
    print(f"  payoff edges: {len(edges)} participant-reference (+{len(direct_edges)} direct)")
    print(f"  connected: {connectedness['single_connected_component']} ({len(comps)} component(s))")
    print("  field table (F_ANCHORED_PANEL_20):")
    for r in sorted(field_rows, key=lambda x: x["field_regret"]):
        print(f"    {r['member']:22s} V={r['V']:.4f} +/-{r['V_se']:.4f} regret={r['field_regret']:.4f} games={r['games']} bucket={r['bucket']}")
    print("  top FROZEN_PANEL_BT_INDEX values:")
    for row in ratings["nodes"][:12]:
        print(f"    {row['node'][:52]:52s} index={row['frozen_panel_bt_index']:8.2f} +/-{row['frozen_panel_bt_index_se']:.2f} kind={row['kind']}")


if __name__ == "__main__":
    sys.exit(main())
