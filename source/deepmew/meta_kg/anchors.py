"""Anchor resolution: config-seeded Pokemon/engine anchors plus auto anchors.

Anchors are card-family identity hooks. Manual (config) anchors take priority
over auto-discovered ones. DSU-style family grouping is for cards, never for
merging decks: terminal cards dominate, shared basics are supporting evidence.
"""

from __future__ import annotations

from typing import Any

from deepmew.meta_kg.schemas import (
    AnchorDefinition,
    BuildWarning,
    CardCatalogRow,
    CardId,
    K2ABuildConfig,
    UniqueDeck,
)

# Bare "Mega " is intentionally excluded: every Mega Pokemon is "Mega X ex" (so
# " ex" already covers it), while Trainers like "Mega Signal" would false-match.
TERMINAL_NAME_PATTERNS = (" ex", "VMAX", "VSTAR", " V")


def _name_to_ids(catalog: list[CardCatalogRow]) -> dict[str, list[CardId]]:
    mapping: dict[str, list[CardId]] = {}
    for row in catalog:
        if row.name_missing:
            continue
        mapping.setdefault(row.card_name.strip().lower(), []).append(row.card_id)
    return mapping


def _resolve_names(
    names: list[str],
    name_index: dict[str, list[CardId]],
    *,
    anchor_id: str,
    warnings: dict[str, BuildWarning] | None,
) -> tuple[CardId, ...]:
    resolved: list[CardId] = []
    for name in names:
        ids = name_index.get(str(name).strip().lower())
        if not ids:
            if warnings is not None:
                warnings.setdefault(
                    "anchor_name_unresolved",
                    BuildWarning("anchor_name_unresolved", "anchor card name not found in catalog"),
                ).bump(f"{anchor_id}:{name}")
            continue
        if len(ids) > 1 and warnings is not None:
            warnings.setdefault(
                "anchor_name_ambiguous",
                BuildWarning("anchor_name_ambiguous", "anchor card name maps to multiple ids"),
            ).bump(f"{anchor_id}:{name}->{ids}")
        resolved.extend(ids)
    # Deterministic, de-duplicated.
    seen: dict[CardId, None] = {}
    for cid in resolved:
        seen.setdefault(cid, None)
    return tuple(seen)


def resolve_anchor_config(
    raw_anchors: list[dict[str, Any]],
    raw_engine_packages: list[dict[str, Any]],
    catalog: list[CardCatalogRow],
    warnings: dict[str, BuildWarning] | None = None,
) -> list[AnchorDefinition]:
    """Resolve name-based config anchors and engine packages to card-ID anchors."""
    name_index = _name_to_ids(catalog)
    anchors: list[AnchorDefinition] = []

    for raw in raw_anchors:
        anchor_id = str(raw.get("anchor_id") or raw.get("label") or "anchor")
        terminal = _resolve_names(raw.get("terminal_names", []) or [], name_index, anchor_id=anchor_id, warnings=warnings)
        line = _resolve_names(raw.get("line_names", []) or [], name_index, anchor_id=anchor_id, warnings=warnings)
        support = _resolve_names(raw.get("support_names", []) or [], name_index, anchor_id=anchor_id, warnings=warnings)
        # Allow direct ID specification too.
        terminal = terminal + tuple(str(c) for c in raw.get("terminal_card_ids", []) or [])
        line = line + tuple(str(c) for c in raw.get("line_card_ids", []) or [])
        if not terminal and not line:
            continue
        anchors.append(
            AnchorDefinition(
                anchor_id=anchor_id,
                label=str(raw.get("label") or anchor_id),
                anchor_type=str(raw.get("anchor_type") or "pokemon_core"),
                terminal_card_ids=terminal,
                line_card_ids=line,
                support_card_ids=support,
                priority=int(raw.get("priority", 100)),
                source="config",
            )
        )

    for raw in raw_engine_packages:
        anchor_id = str(raw.get("package_id") or raw.get("label") or "engine")
        engine_ids = _resolve_names(raw.get("card_names", []) or [], name_index, anchor_id=anchor_id, warnings=warnings)
        engine_ids = engine_ids + tuple(str(c) for c in raw.get("card_ids", []) or [])
        if not engine_ids:
            continue
        anchors.append(
            AnchorDefinition(
                anchor_id=anchor_id,
                label=str(raw.get("label") or anchor_id),
                anchor_type="engine",
                engine_card_ids=engine_ids,
                priority=int(raw.get("priority", 100)),
                source="config",
            )
        )

    return anchors


def _looks_terminal(name: str) -> bool:
    return any(pattern in name for pattern in TERMINAL_NAME_PATTERNS)


def _is_attacker_grade(meta: Any) -> bool:
    """A terminal Pokemon worth anchoring: ex / Mega / an evolved stage."""
    return bool(meta.is_ex or meta.is_mega or meta.stage_guess in ("stage1", "stage2"))


def splash_card_ids(
    assignments: list[Any],
    unique_decks: dict[str, Any],
    config: K2ABuildConfig,
    *,
    card_meta: dict[CardId, Any] | None = None,
) -> set[CardId]:
    """Pokemon that appear across many reliable-core archetypes are splash/engine.

    Spread is measured from a reliable-core-only classification, so a card that
    shows up in >= ``splash_min_archetypes`` distinct real archetypes (e.g.
    Dudunsparce, Fezandipiti ex) is a cross-deck engine, never an identity.
    Configured/auto terminal cores are exempt.
    """
    from collections import defaultdict

    card_meta = card_meta or {}
    archetypes_of_card: dict[CardId, set[str]] = defaultdict(set)
    for assignment in assignments:
        archetype = assignment.primary_archetype
        if archetype in ("Unknown", "Mixed"):
            continue
        deck = unique_decks.get(assignment.deck_hash)
        if deck is None:
            continue
        for cid in deck.cards:
            archetypes_of_card[cid].add(archetype)
    splash: set[CardId] = set()
    for cid, archetypes in archetypes_of_card.items():
        if len(archetypes) >= config.splash_min_archetypes:
            splash.add(cid)
    return splash


def generate_auto_anchors(
    catalog: list[CardCatalogRow],
    config: K2ABuildConfig,
    *,
    existing: list[AnchorDefinition],
    card_meta: dict[CardId, Any] | None = None,
    reliable_only: bool = False,
    exclude_ids: set[CardId] | None = None,
) -> list[AnchorDefinition]:
    """Conservative auto anchors for repeated terminal Pokemon.

    With card metadata, a candidate must be a *Pokemon* that nothing evolves from
    (terminal) and is attacker-grade (ex/Mega/evolved); Trainers, Energy, and
    bare basics are excluded, and pre-evolutions become line evidence. Without
    metadata, it falls back to terminal-looking card names. Auto anchors are
    lower priority than manual anchors.

    ``reliable_only`` keeps only win-condition-grade terminals (ex/Mega/Stage2),
    used for the first spread-measuring pass. ``exclude_ids`` drops cross-deck
    splash/engine Pokemon discovered by that pass.
    """
    if not config.auto_anchor_enabled:
        return []
    card_meta = card_meta or {}
    exclude_ids = exclude_ids or set()
    configured_terminals = {cid for anchor in existing for cid in anchor.terminal_card_ids}
    configured_labels = {anchor.label.strip().lower() for anchor in existing}
    present_ids = {row.card_id for row in catalog}
    name_index = _name_to_ids(catalog)
    auto: list[AnchorDefinition] = []
    for row in catalog:
        if row.name_missing or row.card_id in configured_terminals or row.card_id in exclude_ids:
            continue
        if row.card_name.strip().lower() in configured_labels:
            continue
        if row.deck_count < config.auto_anchor_min_deck_count:
            continue
        meta = card_meta.get(row.card_id)
        if meta is not None:
            if meta.type_guess != "pokemon" or not meta.is_terminal or not _is_attacker_grade(meta):
                continue
            wincon = bool(meta.is_ex or meta.is_mega or meta.stage_guess == "stage2")
            if reliable_only and not wincon:
                continue
            from deepmew.meta_kg.card_metadata import evolution_line_ids

            line = tuple(evolution_line_ids(row.card_id, card_meta, name_index, present_ids))
        elif _looks_terminal(row.card_name):
            wincon = True
            line = ()
        else:
            continue
        auto.append(
            AnchorDefinition(
                anchor_id=f"auto_{row.card_id}",
                label=row.card_name,
                anchor_type="pokemon_core",
                terminal_card_ids=(row.card_id,),
                line_card_ids=line,
                priority=10,
                source="auto",
                terminal_is_wincon=wincon,
            )
        )
    return auto


def merge_anchors(
    *anchor_lists: list[AnchorDefinition],
) -> list[AnchorDefinition]:
    """Earlier lists win; later anchors fill gaps for unclaimed terminal cards.

    Pass higher-priority anchors first (manual, then auto, then fallback).
    """
    claimed: set[CardId] = set()
    merged: list[AnchorDefinition] = []
    for anchors in anchor_lists:
        for anchor in anchors:
            terminals = set(anchor.terminal_card_ids) or set(anchor.engine_card_ids)
            if terminals & claimed:
                continue
            merged.append(anchor)
            claimed |= terminals
    merged.sort(key=lambda a: (-a.priority, a.anchor_id))
    return merged


def generate_fallback_anchors(
    unknown_decks: dict[str, "UniqueDeck"],
    catalog: list[CardCatalogRow],
    config: K2ABuildConfig,
    *,
    existing: list[AnchorDefinition],
    staple_card_ids: set[CardId],
    card_meta: dict[CardId, Any] | None = None,
) -> list[AnchorDefinition]:
    """Discover recurring non-staple Pokemon families among still-Unknown decks.

    A fallback anchor is a group of Pokemon-type cards (Trainers/Energy excluded
    via the catalog ``type_guess``, which is metadata-accurate when available)
    that co-occur across enough Unknown decks. The group's terminal/most-covered
    card becomes the label; the rest become line cards. This catches non-``ex``
    evolution archetypes (e.g. ``Hop's Trevenant``) without hardcoding, while
    decks that stay genuinely unique remain Unknown.
    """
    if not config.auto_anchor_enabled or not unknown_decks:
        return []
    card_meta = card_meta or {}
    min_deck_count = max(2, config.fallback_anchor_min_deck_count)
    pokemon_ids = {row.card_id for row in catalog if row.type_guess == "pokemon" and not row.name_missing}
    name_by_id = {row.card_id: row.card_name for row in catalog}
    freq_by_id = {row.card_id: row.global_deck_freq for row in catalog}
    avg_copies = {row.card_id: row.avg_copies_when_present for row in catalog}

    def is_terminal(cid: CardId) -> int:
        meta = card_meta.get(cid)
        return 1 if (meta is not None and meta.is_terminal) else 0

    claimed = {cid for anchor in existing for cid in anchor.terminal_card_ids + anchor.line_card_ids}

    candidate_ids = pokemon_ids - claimed - staple_card_ids
    deck_count: dict[CardId, int] = {}
    decks_with: dict[CardId, set[str]] = {}
    for deck_hash, deck in unknown_decks.items():
        for cid in deck.cards:
            if cid in candidate_ids:
                decks_with.setdefault(cid, set()).add(deck_hash)
    deck_count = {cid: len(hashes) for cid, hashes in decks_with.items()}
    candidates = sorted([cid for cid, n in deck_count.items() if n >= min_deck_count])
    if not candidates:
        return []

    parent: dict[CardId, CardId] = {cid: cid for cid in candidates}

    def find(x: CardId) -> CardId:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: CardId, b: CardId) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=lambda c: (deck_count[c], c))] = min(ra, rb, key=lambda c: (deck_count[c], c))

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            shared = len(decks_with[a] & decks_with[b])
            if shared >= 0.6 * min(deck_count[a], deck_count[b]):
                union(a, b)

    groups: dict[CardId, list[CardId]] = {}
    for cid in candidates:
        groups.setdefault(find(cid), []).append(cid)

    anchors: list[AnchorDefinition] = []
    for members in groups.values():
        group_decks: set[str] = set()
        for cid in members:
            group_decks |= decks_with[cid]
        if len(group_decks) < min_deck_count:
            continue
        # Representative: prefer a metadata terminal, then most-covered, then the
        # more-evolved card (fewer copies per deck) so the label leans terminal.
        rep = max(
            members,
            key=lambda c: (is_terminal(c), deck_count[c], -avg_copies.get(c, 0.0), freq_by_id.get(c, 0.0), _id_num(c)),
        )
        line = tuple(sorted((c for c in members if c != rep), key=_id_num))
        rep_meta = card_meta.get(rep)
        wincon = bool(rep_meta is None or rep_meta.is_ex or rep_meta.is_mega or rep_meta.stage_guess == "stage2")
        anchors.append(
            AnchorDefinition(
                anchor_id=f"fallback_{rep}",
                label=name_by_id.get(rep, f"Unknown Card {rep}"),
                anchor_type="pokemon_core",
                terminal_card_ids=(rep,),
                line_card_ids=line,
                priority=5,
                source="fallback",
                terminal_is_wincon=wincon,
            )
        )
    anchors.sort(key=lambda a: a.anchor_id)
    return anchors


def _id_num(card_id: CardId) -> int:
    try:
        return int(card_id)
    except (TypeError, ValueError):
        return 0
