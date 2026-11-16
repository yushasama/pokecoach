"""R17: build band-winner BC example rows from the R16 ELO-bucket replays.

Teachers = seats that WON games inside the chosen ELO buckets (default
800-1100: what actually beats the 600-900 field we must climb through).
Advantage weighting is MARWIL/AWR-lite and applied via **weighted reservoir
sampling** (Efraimidis-Spirakis A-Res: key = u^(1/w)): rows from higher
buckets and punisher shells (Garchomp / Lucario / Dragapult / Starmie /
Archaludon; R16 punish plan) get proportionally higher sampling weight, and
the output is capped at --max-rows so the R14 fine-tune recipe (which loads
every example row into RAM) stays memory-safe. Emphasis lives in the sampling
probability, not in duplicate rows.

Row schema matches build_top_player_examples.py exactly (consumed by
`encode_top_player_row_r9`); `rank_bucket` carries the ELO bucket and `rank`
a pseudo-rank (bucket floor) so downstream reporting keeps working.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_r14_ladder_replays import DEFAULT_CARD_CSV, load_card_names, shell_label  # noqa: E402
from deepmew.kaggle_replays.decks import DeckExtractionError, extract_decklists  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402
from deepmew.rl1_teacher_bc.semantics import legal_action_table  # noqa: E402

SCHEMA_VERSION = "deepmew.r17_bucket_examples.v1"
SECTION_PREFIX = "r16-elo-001-"
PUNISHER_TOKENS = ("Garchomp", "Lucario", "Dragapult", "Starmie", "Archaludon")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def emit_winner_rows(
    data: dict[str, Any],
    *,
    episode_id: str,
    bucket: str,
    exclude_teams: set[str],
    cards: dict[int, dict[str, Any]],
    vs_shell_tokens: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    info = _as_dict(data.get("info"))
    team_names = [str(name) for name in _as_list(info.get("TeamNames"))]
    rewards = _as_list(data.get("rewards"))
    if len(team_names) != 2 or len(rewards) != 2:
        return []
    try:
        decks = extract_decklists(data)
    except DeckExtractionError:
        return []
    if len(decks) != 2:
        return []
    pseudo_rank = int(bucket.split("-")[0])
    rows: list[dict[str, Any]] = []
    for seat, team in enumerate(team_names):
        if team in exclude_teams:
            continue
        reward = rewards[seat]
        if not (isinstance(reward, (int, float)) and reward > 0):
            continue  # winners only: these seats beat this band
        if vs_shell_tokens:
            opp_cards = sorted(int(c) for c in decks[1 - seat].card_ids)
            opp_shell = shell_label(opp_cards, cards)
            if not any(tok in opp_shell for tok in vs_shell_tokens):
                continue  # anti-leak mode: teacher must have BEATEN a leak shell
        card_ids = sorted(int(c) for c in decks[seat].card_ids)
        deck_hash = normalized_deck_hash(card_ids)
        shell = shell_label(card_ids, cards)
        for step_index, step in enumerate(_as_list(data.get("steps"))):
            agents = _as_list(step)
            if len(agents) != 2:
                continue
            agent = _as_dict(agents[seat])
            obs = _as_dict(agent.get("observation"))
            select = _as_dict(obs.get("select"))
            options = _as_list(select.get("option"))
            action = agent.get("action")
            if not (obs and options and isinstance(action, list) and action):
                continue
            selected = [i for i in action if isinstance(i, int) and 0 <= i < len(options)]
            if not selected:
                continue
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode_id,
                "step": step_index,
                "seat": seat,
                "team_name": team,
                "rank": pseudo_rank,
                "rank_bucket": bucket,
                "deck_hash": deck_hash,
                "shell": shell,
                "reward": 1,
                "observation": obs,
                "legal_action_table": legal_action_table(obs, select),
                "selected_action_indices": selected,
                "action_mask": {
                    "min_count": select.get("minCount"),
                    "max_count": select.get("maxCount"),
                },
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections-root", default=str(ROOT / ".research" / "kaggle_replays"))
    parser.add_argument("--buckets", nargs="+", default=["800-900", "900-1000", "1000-1100"])
    parser.add_argument("--bucket-weights", nargs="+", type=float, default=[1.0, 2.0, 3.0],
                        help="per-bucket sampling weight (advantage), aligned with --buckets")
    parser.add_argument("--punisher-weight", type=float, default=1.5,
                        help="sampling-weight multiplier for punisher shells (1.0 disables)")
    parser.add_argument("--max-rows", type=int, default=40000,
                        help="output cap; the R14 trainer loads all rows into RAM")
    parser.add_argument("--seed", type=int, default=170707)
    parser.add_argument("--vs-shells", nargs="*", default=[],
                        help="anti-leak mode: keep only winner seats whose OPPONENT shell contains one of these tokens")
    parser.add_argument("--exclude-team", nargs="*", default=["DeepMew"])
    parser.add_argument("--card-csv", default=str(DEFAULT_CARD_CSV))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.bucket_weights) != len(args.buckets):
        raise SystemExit("--bucket-weights must align with --buckets")

    cards = load_card_names(Path(args.card_csv))
    exclude = set(args.exclude_team)
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Efraimidis-Spirakis A-Res weighted reservoir: keep max-rows rows with
    # inclusion probability proportional to weight; min-heap on key = u^(1/w).
    reservoir: list[tuple[float, int, str]] = []  # (key, tiebreak, serialized row)
    tiebreak = 0
    seen: Counter[str] = Counter()
    for bucket, bucket_weight in zip(args.buckets, args.bucket_weights):
        section = Path(args.sections_root) / f"{SECTION_PREFIX}{bucket}"
        if not section.is_dir():
            print(f"[skip] missing section {section}")
            continue
        for path in sorted(section.glob("episode-*-replay.json")):
            episode_id = path.stem.replace("episode-", "").replace("-replay", "")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in emit_winner_rows(
                data, episode_id=episode_id, bucket=bucket,
                exclude_teams=exclude, cards=cards,
                vs_shell_tokens=tuple(args.vs_shells),
            ):
                weight = bucket_weight
                if args.punisher_weight != 1.0 and any(tok in row["shell"] for tok in PUNISHER_TOKENS):
                    weight *= args.punisher_weight
                key = rng.random() ** (1.0 / max(1e-9, weight))
                seen[bucket] += 1
                tiebreak += 1
                if len(reservoir) < args.max_rows:
                    heapq.heappush(reservoir, (key, tiebreak, json.dumps(row)))
                elif key > reservoir[0][0]:
                    heapq.heapreplace(reservoir, (key, tiebreak, json.dumps(row)))
        print(f"[{bucket}] winner rows seen: {seen[bucket]}", flush=True)

    kept_by_bucket: Counter[str] = Counter()
    kept_by_shell: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for _key, _tb, serialized in sorted(reservoir, key=lambda item: -item[0]):
            handle.write(serialized + "\n")
            row = json.loads(serialized)
            kept_by_bucket[row["rank_bucket"]] += 1
            kept_by_shell[row["shell"]] += 1
    print("rows seen:", sum(seen.values()), "kept:", len(reservoir))
    print("kept by bucket:", dict(kept_by_bucket))
    print("kept top shells:", kept_by_shell.most_common(8))
    print("out:", out_path)


if __name__ == "__main__":
    main()
