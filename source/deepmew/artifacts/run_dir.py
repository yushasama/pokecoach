"""Run-directory layout helpers for generated evidence."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    kind: str
    day: str
    run_id: str

    @property
    def path(self) -> Path:
        return self.root / self.kind / self.day / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def output_path(self, name: str) -> Path:
        return self.path / name


def safe_slug(value: str, *, fallback: str = "run") -> str:
    slug = _SAFE_CHARS.sub("_", value.strip()).strip("._-")
    return slug or fallback


def _coerce_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def make_run_id(kind: str, *, now: datetime | None = None, suffix: str | None = None) -> str:
    stamp = _coerce_utc(now).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")
    safe_kind = safe_slug(kind)
    safe_suffix = safe_slug(suffix or uuid.uuid4().hex[:8], fallback="run")
    return f"{stamp}-{safe_kind}-{safe_suffix}"


def create_run_dir(
    *,
    kind: str,
    out_root: Path | str = ".research/runs",
    run_id: str | None = None,
    now: datetime | None = None,
    suffix: str | None = None,
    overwrite: bool = False,
) -> RunDirectory:
    created = _coerce_utc(now)
    safe_kind = safe_slug(kind)
    day = created.strftime("%Y%m%d")
    final_run_id = safe_slug(run_id) if run_id else make_run_id(safe_kind, now=created, suffix=suffix)
    run_dir = RunDirectory(root=Path(out_root), kind=safe_kind, day=day, run_id=final_run_id)
    if run_dir.path.exists() and not overwrite:
        raise FileExistsError(f"run directory already exists: {run_dir.path}")
    run_dir.path.mkdir(parents=True, exist_ok=overwrite)
    return run_dir
