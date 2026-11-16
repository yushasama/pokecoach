"""Minimal Markdown rendering utilities for evidence reports."""

from __future__ import annotations

from typing import Iterable, Sequence


def title(text: str, level: int = 1) -> str:
    level = max(1, min(level, 6))
    return f"{'#' * level} {text}\n"


def section(text: str) -> str:
    return title(text, level=2)


def bullets(items: Iterable[str]) -> str:
    return "".join(f"- {item}\n" for item in items)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    header_line = "| " + " | ".join(_cell(header) for header in headers) + " |\n"
    divider_line = "| " + " | ".join("---" for _ in headers) + " |\n"
    body = "".join("| " + " | ".join(_cell(cell) for cell in row) + " |\n" for row in rows)
    return header_line + divider_line + body
