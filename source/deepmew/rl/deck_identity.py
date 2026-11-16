"""Fail-closed policy/deck identity for RL and arena opponents."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
ROOT_DECK_PATH = Path("deck.csv")
DEFAULT_CARD_POOL_VERSION = "cg_api_current"
R4F_FEATURE_DECK_HASH_SCHEMA_VERSION = "deepmew.r4f_feature_deck_hash.ordered_ids.v0"
EXPLICIT_ROOT_DECK_IDS = {"agent_root_deck", "quarantined_failure_baseline"}
GUARDED_POLICY_PREFIXES = ("reference_", "teacher_", "top_player_")


def resolve_repo_path(path: str | Path) -> Path:
    deck_path = Path(path)
    if deck_path.is_absolute():
        return deck_path

    # Repository paths are serialized into manifests on both Windows and POSIX.
    # Treat relative manifest paths as platform-neutral rather than allowing the
    # current host to interpret the other platform's separator as a filename.
    normalized_relative_path = Path(str(path).replace("\\", "/"))

    return ROOT / normalized_relative_path


def is_root_deck_path(path: str | Path | None) -> bool:
    if path is None or str(path) == "":
        return False
    try:
        return resolve_repo_path(path).resolve() == (ROOT / ROOT_DECK_PATH).resolve()
    except OSError:
        return str(path).replace("\\", "/") == "deck.csv"


def is_guarded_policy_id(policy_id: str) -> bool:
    return str(policy_id).startswith(GUARDED_POLICY_PREFIXES)


def normalized_deck_payload(
    deck: Sequence[int],
    *,
    card_pool_version: str = DEFAULT_CARD_POOL_VERSION,
) -> dict[str, Any]:
    counts: dict[int, int] = {}
    for card_id in deck:
        key = int(card_id)
        counts[key] = counts.get(key, 0) + 1
    return {
        "card_pool_version": card_pool_version,
        "cards": [
            {"card_id": card_id, "count": counts[card_id]}
            for card_id in sorted(counts)
        ],
    }


def normalized_deck_hash(
    deck: Sequence[int],
    *,
    card_pool_version: str = DEFAULT_CARD_POOL_VERSION,
) -> str:
    payload = normalized_deck_payload(deck, card_pool_version=card_pool_version)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def r4f_feature_deck_hash(deck: Sequence[int]) -> str:
    """Historical ordered deck hash consumed by the frozen R4F feature encoder."""

    payload = ",".join(str(int(card_id)) for card_id in deck)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_deck_any(path: str | Path) -> list[int]:
    """Read one-id-per-line or counted CSV decklists without falling back."""

    deck_path = resolve_repo_path(path)
    raw = deck_path.read_text(encoding="utf-8-sig").splitlines()
    if not raw:
        return []
    first = raw[0].strip().lower()
    if "," in first or "card" in first or "count" in first:
        deck: list[int] = []
        with deck_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            fields = {field.lower(): field for field in reader.fieldnames}
            id_field = fields.get("card_id") or fields.get("id") or reader.fieldnames[0]
            count_field = fields.get("count") or fields.get("copies")
            for row in reader:
                value = str(row.get(id_field) or "").strip()
                if not value:
                    continue
                try:
                    card_id = int(value)
                except ValueError:
                    continue
                count = 1
                if count_field:
                    try:
                        count = int(str(row.get(count_field) or "1").strip())
                    except ValueError:
                        count = 1
                deck.extend([card_id] * max(0, count))
        return deck

    deck = []
    for line_number, line in enumerate(raw, start=1):
        value = line.strip()
        if not value:
            continue
        try:
            deck.append(int(value.split(",")[0]))
        except ValueError as exc:
            raise ValueError(f"{deck_path}:{line_number}: invalid card id {value!r}") from exc
    return deck


@dataclass(frozen=True)
class DeckIdentity:
    policy_id: str
    policy_role: str
    deck_id: str
    deck_role: str
    deck_hash: str
    deck_status: str
    pairing_role: str
    deck_path: str = ""
    deck_source_id: str = ""
    card_count: int = 0
    notes: str = ""

    @property
    def deck_source_path(self) -> str:
        return self.deck_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_role": self.policy_role,
            "deck_id": self.deck_id,
            "deck_role": self.deck_role,
            "deck_path": self.deck_path,
            "deck_source_id": self.deck_source_id,
            "deck_source_path": self.deck_source_path,
            "deck_hash": self.deck_hash,
            "deck_status": self.deck_status,
            "pairing_role": self.pairing_role,
            "card_count": self.card_count,
            "notes": self.notes,
        }


def validate_deck_identity(identity: DeckIdentity) -> DeckIdentity:
    if not identity.policy_id:
        raise ValueError("deck identity requires policy_id")
    if not identity.deck_id:
        raise ValueError(f"{identity.policy_id}: deck identity requires deck_id")
    if not identity.deck_hash:
        raise ValueError(f"{identity.policy_id}: deck identity requires deck_hash")
    if identity.deck_status == "runnable_full_deck" and identity.card_count != 60:
        raise ValueError(
            f"{identity.policy_id}: runnable_full_deck must contain 60 cards, got {identity.card_count}"
        )
    rootish = is_root_deck_path(identity.deck_path)
    if rootish and identity.deck_id not in EXPLICIT_ROOT_DECK_IDS:
        raise ValueError(
            f"{identity.policy_id}: root deck.csv requires explicit root deck_id, got {identity.deck_id!r}"
        )
    if rootish and is_guarded_policy_id(identity.policy_id) and identity.pairing_role != "debug":
        raise ValueError(
            f"{identity.policy_id}: guarded opponents may not silently use root deck.csv"
        )
    return identity


def deck_identity_from_deck(
    *,
    policy_id: str,
    policy_role: str,
    deck_id: str,
    deck_role: str,
    deck: Sequence[int],
    deck_status: str,
    pairing_role: str,
    deck_path: str | Path = "",
    deck_source_id: str = "",
    notes: str = "",
) -> DeckIdentity:
    identity = DeckIdentity(
        policy_id=policy_id,
        policy_role=policy_role,
        deck_id=deck_id,
        deck_role=deck_role,
        deck_path=str(deck_path) if deck_path else "",
        deck_source_id=deck_source_id,
        deck_hash=normalized_deck_hash(deck),
        deck_status=deck_status,
        pairing_role=pairing_role,
        card_count=len(deck),
        notes=notes,
    )
    return validate_deck_identity(identity)


def root_deck_identity(
    *,
    policy_id: str,
    policy_role: str = "baseline",
    pairing_role: str = "baseline",
) -> DeckIdentity:
    deck = read_deck_any(ROOT_DECK_PATH)
    return deck_identity_from_deck(
        policy_id=policy_id,
        policy_role=policy_role,
        deck_id="agent_root_deck",
        deck_role="root_shipping_surface",
        deck=deck,
        deck_status="quarantined_failure_baseline",
        pairing_role=pairing_role,
        deck_path=ROOT_DECK_PATH,
        deck_source_id="root deck.csv",
        notes="Explicit root shipping surface; quarantined as a failure baseline, not an implicit opponent default.",
    )


def deck_identity_from_path(
    *,
    policy_id: str,
    policy_role: str,
    deck_id: str,
    deck_role: str,
    deck_path: str | Path,
    deck_status: str,
    pairing_role: str,
    deck_source_id: str = "",
    notes: str = "",
) -> DeckIdentity:
    deck = read_deck_any(deck_path)
    return deck_identity_from_deck(
        policy_id=policy_id,
        policy_role=policy_role,
        deck_id=deck_id,
        deck_role=deck_role,
        deck=deck,
        deck_status=deck_status,
        pairing_role=pairing_role,
        deck_path=deck_path,
        deck_source_id=deck_source_id,
        notes=notes,
    )


def agent_deck_identity_from_config(config: Any) -> DeckIdentity:
    deck_id = str(getattr(config, "agent_deck_id", "agent_root_deck"))
    deck_path = str(getattr(config, "agent_deck_path", "deck.csv"))
    if is_root_deck_path(deck_path):
        return root_deck_identity(
            policy_id=str(getattr(config, "experiment_id", "agent_policy")),
            policy_role="shipping_candidate",
            pairing_role="shipping_candidate",
        )
    return deck_identity_from_path(
        policy_id=str(getattr(config, "experiment_id", "agent_policy")),
        policy_role="research_candidate",
        deck_id=deck_id,
        deck_role="research_agent_deck",
        deck_path=deck_path,
        deck_status="runnable_full_deck",
        pairing_role="research_candidate",
        deck_source_id=deck_id,
    )
