from __future__ import annotations

from pathlib import Path

from cg.game import battle_finish, battle_start
from deepmew.deck.io import read_deck
from deepmew.deck.models import ValidationResult
from deepmew.deck.tables import card_table


def validate_deck(path: str | Path, *, check_engine: bool = True) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    engine_accepted: bool | None = None
    engine_error: str | None = None

    try:
        deck = read_deck(path)
    except Exception as exc:
        return ValidationResult(False, (str(exc),), (), None, None)

    cards_by_id = card_table()
    if len(deck) != 60:
        errors.append(f"deck must contain exactly 60 cards; found {len(deck)}")

    missing = sorted({card_id for card_id in deck if card_id not in cards_by_id})
    if missing:
        errors.append(f"unknown card id(s): {', '.join(str(card_id) for card_id in missing)}")

    ace_count = sum(1 for card_id in deck if card_id in cards_by_id and cards_by_id[card_id].aceSpec)
    if ace_count > 1:
        errors.append(f"deck may contain at most 1 ACE SPEC; found {ace_count}")

    if not any(cards_by_id[card_id].basic for card_id in deck if card_id in cards_by_id):
        warnings.append("deck has no Basic Pokemon")

    if check_engine and not errors:
        obs = None
        try:
            obs, start_data = battle_start(deck, deck)
            engine_accepted = obs is not None
            if obs is None:
                engine_error = (
                    f"BattleStart failed: errorPlayer={start_data.errorPlayer}, "
                    f"errorType={start_data.errorType}"
                )
                errors.append(engine_error)
        except Exception as exc:
            engine_accepted = False
            engine_error = f"{type(exc).__name__}: {exc}"
            errors.append(engine_error)
        finally:
            if obs is not None:
                battle_finish()

    return ValidationResult(not errors, tuple(errors), tuple(warnings), engine_accepted, engine_error)
