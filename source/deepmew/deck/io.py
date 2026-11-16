from __future__ import annotations

from pathlib import Path


def read_deck(path: str | Path) -> list[int]:
    deck_path = Path(path)
    card_ids = []
    for line_number, line in enumerate(deck_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            card_ids.append(int(stripped))
        except ValueError as exc:
            raise ValueError(f"{deck_path}:{line_number}: invalid card id {stripped!r}") from exc
    return card_ids
