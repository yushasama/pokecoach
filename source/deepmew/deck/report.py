from __future__ import annotations

from deepmew.deck.models import DeckSummary, ValidationResult


def format_report(summary: DeckSummary, validation: ValidationResult) -> str:
    lines = [
        f"# Deck Report: {summary.path}",
        "",
        "## Validation",
        "",
        f"- valid: {validation.valid}",
        f"- engine_accepted: {validation.engine_accepted}",
        f"- errors: {len(validation.errors)}",
        f"- warnings: {len(validation.warnings)}",
    ]
    for error in validation.errors:
        lines.append(f"  - ERROR: {error}")
    for warning in validation.warnings:
        lines.append(f"  - WARNING: {warning}")

    lines.extend(
        [
            "",
            "## Composition",
            "",
            f"- total_cards: {summary.total_cards}",
            f"- unique_cards: {summary.unique_cards}",
            f"- pokemon_count: {summary.pokemon_count}",
            f"- trainer_count: {summary.trainer_count}",
            f"- energy_count: {summary.energy_count}",
            f"- basics: {summary.basic_count}",
            f"- evolutions: {summary.evolution_count}",
            f"- ex: {summary.ex_count}",
            f"- mega_ex: {summary.mega_ex_count}",
            f"- ace_spec: {summary.ace_spec_count}",
            "",
            "## Cards",
            "",
            "| count | id | name | type | flags | attacks |",
            "|---:|---:|---|---|---|---|",
        ]
    )
    for card in summary.cards:
        flags = []
        if card.basic:
            flags.append("basic")
        if card.stage1:
            flags.append("stage1")
        if card.stage2:
            flags.append("stage2")
        if card.ex:
            flags.append("ex")
        if card.mega_ex:
            flags.append("mega_ex")
        if card.ace_spec:
            flags.append("ace_spec")
        if card.evolves_from:
            flags.append(f"from {card.evolves_from}")
        attack_text = "; ".join(
            f"{attack['name']} {attack['damage']} ({'/'.join(attack['energies'])})"
            for attack in card.attacks
        )
        lines.append(
            f"| {card.count} | {card.card_id} | {card.name} | {card.card_type} | "
            f"{', '.join(flags) or '-'} | {attack_text or '-'} |"
        )

    lines.extend(["", "## Visible Main Attackers", ""])
    for card in summary.main_attackers:
        best_damage = max((int(attack["damage"]) for attack in card.attacks), default=0)
        lines.append(f"- {card.name} x{card.count}: HP {card.hp}, best visible damage {best_damage}")

    return "\n".join(lines)
