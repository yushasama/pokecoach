"""Band-stratified replay pilots: reproduce each ELO band's real players.

The curriculum's `replay` pilot tier needs *faithful* opponents -- brains that play
the way real ~600-1000 ELO ladder players did, per archetype. The existing
`ReplayAgent` samples an action *category* from a `ReplayPolicyTable` keyed by
(select_context, archetype); this module builds those tables directly from the R16
ELO-bucket replay sections -- the SAME source the curriculum's opponent decks come
from -- so pilots and decks share provenance and are stratified by band. A 900-1000
Archaludon pilot then plays like a real 900-1000 Archaludon player, not like the
off-distribution Abomasnow champion the R17 exploiter tier fumbled foreign decks with.

Honest limitation (inherited from ReplayPolicy): this shapes which action CATEGORY
the opponent takes per (context, archetype), not the exact card/target. It is a
faithful opponent *distribution*, not a full replay clone. Uncertain decisions fall
back to first-legal, so the opponent is always legal (INVARIANT #1 for the arena).

Offline/eval-only (INVARIANT #8): never imported by `main.py`/`cg`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cg.api import OptionType, SelectContext
from deepmew.kaggle_replays.decks import DeckExtractionError, extract_decklists
from deepmew.kaggle_replays.episode import episode_metadata, load_episode
from deepmew.kaggle_replays.ingest import build_kaggle_action_rows
from deepmew.real_meta_arena.replay_policy import TABLE_SCHEMA_VERSION, ReplayPolicyTable

BAND_TABLES_SCHEMA = "deepmew.curriculum_band_replay_tables.v1"


def _context_name(value: Any) -> str:
    try:
        return SelectContext(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _option_type_name(value: Any) -> str:
    try:
        return OptionType(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


class _TableAccumulator:
    """Tallies (context, archetype) -> {option_type: count} from action rows."""

    def __init__(self) -> None:
        self.ctx_arch: dict[str, Counter] = defaultdict(Counter)
        self.ctx: dict[str, Counter] = defaultdict(Counter)
        self.global_counts: Counter = Counter()
        self.rows = 0

    def add_row(self, row: dict[str, Any], archetype: str) -> None:
        """One replay_action_row.v1: derive context + each chosen option's type."""
        state = row.get("state_json") or {}
        select = state.get("select") if isinstance(state, dict) else None
        options = row.get("legal_options_json") or []
        if not isinstance(select, dict) or not isinstance(options, list):
            return
        context = _context_name(select.get("context"))
        for idx in row.get("chosen_indices") or []:
            if not isinstance(idx, int) or idx < 0 or idx >= len(options):
                continue
            option = options[idx]
            if not isinstance(option, dict):
                continue
            action = _option_type_name(option.get("type"))
            self.ctx_arch[ReplayPolicyTable._key(context, archetype)][action] += 1
            self.ctx[context][action] += 1
            self.global_counts[action] += 1
            self.rows += 1

    def to_table(self, *, split: str = "r16") -> ReplayPolicyTable:
        return ReplayPolicyTable(
            schema_version=TABLE_SCHEMA_VERSION,
            split=split,
            ctx_arch={k: dict(v) for k, v in self.ctx_arch.items()},
            ctx={k: dict(v) for k, v in self.ctx.items()},
            global_counts=dict(self.global_counts),
            train_examples=self.rows,
        )


@dataclass
class BandReplayTables:
    """Per-band replay tables plus a global fallback (all bands merged)."""

    bands: dict[str, ReplayPolicyTable]
    global_table: ReplayPolicyTable
    band_rows: dict[str, int]

    def table_for(self, bucket: str) -> ReplayPolicyTable:
        """Band-appropriate table, falling back to the merged global table."""
        return self.bands.get(bucket, self.global_table)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": BAND_TABLES_SCHEMA,
            "band_rows": dict(self.band_rows),
            "bands": {b: t.to_dict() for b, t in self.bands.items()},
            "global": self.global_table.to_dict(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BandReplayTables":
        return cls(
            bands={b: ReplayPolicyTable.from_dict(t) for b, t in (data.get("bands") or {}).items()},
            global_table=ReplayPolicyTable.from_dict(data.get("global") or {}),
            band_rows={b: int(n) for b, n in (data.get("band_rows") or {}).items()},
        )


def build_band_replay_tables(
    sections_root: str | Path,
    *,
    shell_labeler,
    section_prefix: str = "r16-elo-001-",
    bands: tuple[str, ...] | None = None,
) -> BandReplayTables:
    """Scan R16 bucket sections and build one replay table per band + a global one.

    `shell_labeler(card_ids) -> str` maps a seat's 60 card ids to the archetype/shell
    label (pass the same labeler the deck pool used, so archetype keys align with the
    curriculum's opponent shells).
    """
    sections_root = Path(sections_root)
    from deepmew.bucket_arena import STAGE_ORDER  # local import: avoids a heavy dep at import time

    band_names = bands if bands is not None else STAGE_ORDER
    per_band: dict[str, _TableAccumulator] = {}
    merged = _TableAccumulator()

    for band in band_names:
        section_dir = sections_root / f"{section_prefix}{band}"
        if not section_dir.is_dir():
            continue
        acc = _TableAccumulator()
        for path in sorted(section_dir.glob("episode-*-replay.json")):
            try:
                data = load_episode(path)
                decks = extract_decklists(data)
            except (OSError, ValueError, DeckExtractionError):
                continue
            if len(decks) != 2:
                continue
            shell_by_index: dict[int, str] = {}
            decks_by_index: dict[int, dict[str, Any]] = {}
            for deck in decks:
                card_ids = sorted(int(c) for c in deck.card_ids)
                if len(card_ids) != 60:
                    continue
                shell_by_index[deck.player_index] = shell_labeler(card_ids)
                decks_by_index[deck.player_index] = {"deck_id": None, "deck_hash": None}
            try:
                meta = episode_metadata(data, path=path,
                                        name_fallback=[d.team_name for d in decks])
                rows, _ = build_kaggle_action_rows(
                    data, meta, decks_by_index,
                    source_run_id="r19-replay-pilots", source_path=str(path),
                )
            except (ValueError, KeyError):
                continue
            for row in rows:
                archetype = shell_by_index.get(row.get("player_id"), "Unknown")
                acc.add_row(row, archetype)
                merged.add_row(row, archetype)
        if acc.rows:
            per_band[band] = acc

    return BandReplayTables(
        bands={b: acc.to_table() for b, acc in per_band.items()},
        global_table=merged.to_table(),
        band_rows={b: acc.rows for b, acc in per_band.items()},
    )
