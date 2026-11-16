"""Build and validate frozen DeepMew submission bundles."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from deepmew.submissions.manifest import (
    SCHEMA_VERSION,
    sha256_file,
    utc_now_iso,
    validate_submission_manifest,
    write_submission_manifest,
)


ENTRYPOINT = "main.py"
DECK_PATH = "deck.csv"
SUPPORTED_AGENT = "v2.4"
SUBMISSION_README = "SUBMISSION_README.md"
SUBMISSION_MANIFEST = "submission_manifest.json"
DEFAULT_OUT_ROOT = ".research/submissions"

EXCLUDED_PATHS = (
    ".research/",
    "tests/",
    "__pycache__/",
    ".pytest_cache/",
    ".git/",
    ".agents/",
    ".codex/",
    "data/",
    "scripts/",
    "deepmew/replay/",
    "deepmew/trace/",
    ".research/runs/",
    ".research/handoff_bundles/",
    "handoff_bundle/",
    "handoff_bundles/",
    "replay/action/prior artifacts",
)

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_LOCAL_ROOTS = {"deepmew", "cg"}
_DISCOVER_ROOTS = {"deepmew"}
_IGNORED_NAMES = {"__pycache__", ".pytest_cache"}


class SubmissionBundleError(RuntimeError):
    """Raised when a bundle cannot be planned, staged, or validated."""


@dataclass(frozen=True)
class SubmissionPlan:
    repo_root: Path
    out_root: Path
    day: str
    submission_id: str
    bundle_dir: Path
    agent_spec: str
    canonical_label: str
    included_paths: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...] = EXCLUDED_PATHS

    def to_summary(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "bundle_dir": str(self.bundle_dir),
            "agent_spec": self.agent_spec,
            "canonical_label": self.canonical_label,
            "included_paths": list(self.included_paths),
            "dependency_paths": list(self.dependency_paths),
            "excluded_paths": list(self.excluded_paths),
        }


@dataclass(frozen=True)
class SubmissionBuildResult:
    plan: SubmissionPlan
    manifest: dict[str, Any] | None
    tar_gz_path: Path | None
    zip_path: Path | None
    dry_run: bool

    @property
    def bundle_dir(self) -> Path:
        return self.plan.bundle_dir

    def to_summary(self) -> dict[str, Any]:
        summary = self.plan.to_summary()
        summary.update(
            {
                "dry_run": self.dry_run,
                "manifest_path": str(self.bundle_dir / SUBMISSION_MANIFEST) if self.manifest else None,
                "bundle_tar_gz": str(self.tar_gz_path) if self.tar_gz_path else None,
                "tar_gz_hash": self.manifest.get("tar_gz_hash") if self.manifest else None,
                "bundle_zip": str(self.zip_path) if self.zip_path else None,
                "zip_hash": self.manifest.get("zip_hash") if self.manifest else None,
                "validation": self.manifest.get("validation") if self.manifest else None,
            }
        )
        return summary


def _coerce_utc(now: datetime | None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _safe_slug(value: str, *, fallback: str = "submission") -> str:
    slug = _SAFE_CHARS.sub("_", value.strip()).strip("._-")
    return slug or fallback


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _assert_supported_agent(agent: str) -> str:
    if agent != SUPPORTED_AGENT:
        raise SubmissionBundleError(
            f"submission bundling currently freezes only default {SUPPORTED_AGENT}; got {agent!r}"
        )
    return SUPPORTED_AGENT


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _git_commit(repo_root: Path) -> str | None:
    return _git_value(repo_root, "rev-parse", "HEAD")


def _git_dirty(repo_root: Path) -> bool | None:
    value = _git_value(repo_root, "status", "--porcelain")
    if value is None:
        return None
    return bool(value)


def _module_parts(module: str) -> tuple[str, ...]:
    return tuple(part for part in module.split(".") if part)


def _module_path(repo_root: Path, module: str) -> Path | None:
    parts = _module_parts(module)
    if not parts or parts[0] not in _LOCAL_ROOTS:
        return None
    file_path = repo_root.joinpath(*parts).with_suffix(".py")
    if file_path.is_file():
        return file_path
    init_path = repo_root.joinpath(*parts, "__init__.py")
    if init_path.is_file():
        return init_path
    return None


def _module_name_for_path(repo_root: Path, path: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    parts = rel.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _add_parent_inits(repo_root: Path, module: str, files: set[Path]) -> None:
    parts = _module_parts(module)
    if not parts or parts[0] not in _LOCAL_ROOTS:
        return
    for end in range(1, len(parts) + 1):
        init_path = repo_root.joinpath(*parts[:end], "__init__.py")
        if init_path.is_file():
            files.add(init_path)


def _resolve_relative_import(current_module: str, level: int, module: str | None, *, is_package: bool) -> str | None:
    current_parts = current_module.split(".")
    package_parts = current_parts if is_package else current_parts[:-1]
    if level > len(package_parts) + 1:
        return None
    base = package_parts[: max(0, len(package_parts) - level + 1)]
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def _iter_import_modules(path: Path, module_name: str, *, is_package: bool = False) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.split(".", 1)[0] in _DISCOVER_ROOTS:
                    yield name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative_import(module_name, node.level, node.module, is_package=is_package)
            else:
                base = node.module
            if not base:
                continue
            if base.split(".", 1)[0] in _DISCOVER_ROOTS:
                yield base
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    yield f"{base}.{alias.name}"


def discover_import_dependencies(repo_root: Path, entrypoint: str = ENTRYPOINT) -> tuple[str, ...]:
    root = Path(repo_root)
    entry = root / entrypoint
    if not entry.is_file():
        raise SubmissionBundleError(f"missing submission entrypoint: {entrypoint}")

    discovered: set[Path] = set()
    seen_modules: set[str] = set()
    queue: list[str] = []

    for imported in _iter_import_modules(entry, "main"):
        if imported.split(".", 1)[0] in _LOCAL_ROOTS:
            queue.append(imported)

    while queue:
        module = queue.pop(0)
        if module in seen_modules:
            continue
        seen_modules.add(module)
        path = _module_path(root, module)
        if path is None:
            continue
        discovered.add(path)
        _add_parent_inits(root, module, discovered)
        parsed_module = _module_name_for_path(root, path)
        for imported in _iter_import_modules(path, parsed_module, is_package=path.name == "__init__.py"):
            if imported not in seen_modules:
                queue.append(imported)

    # cg is copied as a required submission-surface directory, so dependency
    # paths report only the project modules beyond main.py/deck.csv/cg.
    dependency_paths = sorted(_rel(path, root) for path in discovered if path.parts and path.parts[-1] != "cg")
    return tuple(dependency_paths)


def _iter_cg_files(repo_root: Path) -> Iterable[Path]:
    cg_root = repo_root / "cg"
    if not cg_root.is_dir():
        raise SubmissionBundleError("missing required cg/ directory")
    for path in cg_root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in _IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        yield path


def plan_submission(
    *,
    repo_root: Path | str = ".",
    agent: str,
    out_root: Path | str = DEFAULT_OUT_ROOT,
    now: datetime | None = None,
) -> SubmissionPlan:
    root = Path(repo_root).resolve()
    canonical_label = _assert_supported_agent(agent)
    created = _coerce_utc(now)
    day = created.strftime("%Y%m%d")

    deck = root / DECK_PATH
    if not deck.is_file():
        raise SubmissionBundleError(f"missing required deck file: {DECK_PATH}")
    entrypoint = root / ENTRYPOINT
    if not entrypoint.is_file():
        raise SubmissionBundleError(f"missing required entrypoint: {ENTRYPOINT}")

    deck_hash = sha256_file(deck)[:8]
    stamp = created.replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")
    submission_id = f"{stamp}-{_safe_slug(canonical_label)}-{deck_hash}"
    output_root = Path(out_root)
    bundle_dir = output_root / day / submission_id

    dependency_paths = discover_import_dependencies(root)
    included = {
        ENTRYPOINT,
        DECK_PATH,
        *(_rel(path, root) for path in _iter_cg_files(root)),
        *dependency_paths,
        SUBMISSION_MANIFEST,
        SUBMISSION_README,
    }
    return SubmissionPlan(
        repo_root=root,
        out_root=output_root,
        day=day,
        submission_id=submission_id,
        bundle_dir=bundle_dir,
        agent_spec=agent,
        canonical_label=canonical_label,
        included_paths=tuple(sorted(included)),
        dependency_paths=dependency_paths,
    )


def _copy_included_files(plan: SubmissionPlan) -> None:
    if plan.bundle_dir.exists():
        raise SubmissionBundleError(f"bundle directory already exists: {plan.bundle_dir}")
    plan.bundle_dir.mkdir(parents=True, exist_ok=False)
    for rel_path in plan.included_paths:
        if rel_path in {SUBMISSION_MANIFEST, SUBMISSION_README}:
            continue
        source = plan.repo_root / rel_path
        if not source.is_file():
            raise SubmissionBundleError(f"planned source file is missing: {rel_path}")
        target = plan.bundle_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _read_deck_cards(path: Path) -> list[int]:
    cards: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            cards.append(int(stripped))
        except ValueError as exc:
            raise SubmissionBundleError(f"{path}:{line_number}: invalid card id {stripped!r}") from exc
    return cards


def _validation_check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "ok": ok, **details}


def validate_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    checks: list[dict[str, Any]] = []

    entrypoint = bundle / ENTRYPOINT
    deck_path = bundle / DECK_PATH
    cg_path = bundle / "cg"

    checks.append(_validation_check("entrypoint_exists", entrypoint.is_file(), path=ENTRYPOINT))
    checks.append(_validation_check("deck_exists", deck_path.is_file(), path=DECK_PATH))
    checks.append(_validation_check("cg_exists", cg_path.is_dir(), path="cg/"))

    deck_len: int | None = None
    if deck_path.is_file():
        try:
            deck_cards = _read_deck_cards(deck_path)
            deck_len = len(deck_cards)
            checks.append(_validation_check("deck_has_60_cards", deck_len == 60, count=deck_len))
        except SubmissionBundleError as exc:
            checks.append(_validation_check("deck_has_60_cards", False, error=str(exc)))

    smoke = {
        "ok": False,
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }
    if entrypoint.is_file() and deck_path.is_file() and cg_path.is_dir():
        code = r"""
import importlib
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path = [str(root)] + [item for item in sys.path if item not in ("", str(root))]
main = importlib.import_module("main")
agent = getattr(main, "agent", None)
if not callable(agent):
    raise RuntimeError("main.agent is missing or not callable")
deck = main.read_deck_csv()
if len(deck) != 60:
    raise RuntimeError(f"main.read_deck_csv returned {len(deck)} cards, expected 60")
initial = agent({"select": None, "logs": [], "current": None, "search_begin_input": None})
if initial != deck:
    raise RuntimeError("main.agent initial deck response does not match read_deck_csv")
print(json.dumps({"agent_callable": True, "deck_len": len(deck), "initial_agent_len": len(initial)}))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(bundle)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                cwd=bundle,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            smoke.update(
                {
                    "ok": result.returncode == 0,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "returncode": result.returncode,
                }
            )
        except subprocess.TimeoutExpired as exc:
            smoke.update({"stderr": f"import smoke timed out after {exc.timeout}s", "returncode": "timeout"})
        except OSError as exc:
            smoke.update({"stderr": str(exc), "returncode": "oserror"})

    smoke_details = {key: value for key, value in smoke.items() if key != "ok"}
    checks.append(_validation_check("staged_import_and_initial_agent_smoke", bool(smoke["ok"]), **smoke_details))
    checks.append(
        _validation_check(
            "local_self_play_smoke",
            True,
            skipped=True,
            reason=(
                "Existing scripts/self_play.py imports repo diagnostics and metrics outside the minimal "
                "submission closure; staged import/deck/agent smoke is used instead."
            ),
        )
    )

    required_ok = all(check["ok"] for check in checks if not check.get("skipped"))
    return {
        "passed": required_ok,
        "checks": checks,
        "deck_card_count": deck_len,
    }


def _archive_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    if name.endswith(".zip"):
        return "zip"
    return "unknown"


def _normalize_archive_member(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _archive_members(path: Path) -> tuple[str, ...]:
    archive_type = _archive_type(path)
    if archive_type == "tar.gz":
        with tarfile.open(path, "r:gz") as archive:
            return tuple(sorted(_normalize_archive_member(member.name) for member in archive.getmembers()))
    if archive_type == "zip":
        with zipfile.ZipFile(path) as archive:
            return tuple(sorted(_normalize_archive_member(name) for name in archive.namelist()))
    raise SubmissionBundleError(f"unsupported archive type for validation: {path}")


def _archive_exclusion_hit(name: str) -> bool:
    normalized = name.strip("/")
    parts = normalized.split("/")
    if not normalized:
        return False
    if normalized.endswith(".pyc") or "__pycache__" in parts:
        return True
    excluded_prefixes = (
        ".research",
        ".pytest_cache",
        ".git",
        ".agents",
        ".codex",
        "tests",
        "data",
        "scripts",
        "handoff_bundle",
        "handoff_bundles",
        "deepmew/replay",
        "deepmew/trace",
    )
    return any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in excluded_prefixes)


def validate_archive(
    archive_path: Path | str,
    *,
    required_dependency_paths: Iterable[str] = (),
) -> dict[str, Any]:
    path = Path(archive_path)
    checks: list[dict[str, Any]] = []
    archive_type = _archive_type(path)
    try:
        names = _archive_members(path)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, SubmissionBundleError) as exc:
        return {
            "passed": False,
            "archive_path": str(path),
            "archive_type": archive_type,
            "member_count": 0,
            "checks": [_validation_check("archive_readable", False, error=str(exc))],
        }

    name_set = set(names)
    deepmew_required = tuple(sorted(rel_path for rel_path in required_dependency_paths if rel_path.startswith("deepmew/")))
    missing_dependencies = [dep for dep in deepmew_required if dep not in name_set]
    exclusion_hits = sorted(name for name in name_set if _archive_exclusion_hit(name))

    checks.append(_validation_check("archive_readable", True, archive_type=archive_type))
    checks.append(_validation_check("top_level_entrypoint", ENTRYPOINT in name_set, path=ENTRYPOINT))
    checks.append(_validation_check("top_level_deck", DECK_PATH in name_set, path=DECK_PATH))
    checks.append(_validation_check("cg_included", any(name == "cg" or name.startswith("cg/") for name in name_set), path="cg/"))
    checks.append(
        _validation_check(
            "deepmew_dependency_closure_included",
            not missing_dependencies,
            required_count=len(deepmew_required),
            missing=missing_dependencies,
        )
    )
    checks.append(
        _validation_check(
            "excluded_paths_absent",
            not exclusion_hits,
            hits=exclusion_hits[:20],
            hit_count=len(exclusion_hits),
        )
    )

    return {
        "passed": all(check["ok"] for check in checks),
        "archive_path": str(path),
        "archive_type": archive_type,
        "member_count": len(names),
        "checks": checks,
    }


def _artifact_hashes(bundle_dir: Path, included_paths: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel_path in sorted(included_paths):
        if rel_path == SUBMISSION_MANIFEST:
            continue
        path = bundle_dir / rel_path
        if path.is_file():
            hashes[rel_path] = sha256_file(path)
    return hashes


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    validation = manifest["validation"]
    lines = [
        "# DeepMew Submission Bundle",
        "",
        f"- submission_id: `{manifest['submission_id']}`",
        f"- agent_spec: `{manifest['agent_spec']}`",
        f"- canonical_label: `{manifest['canonical_label']}`",
        f"- entrypoint: `{manifest['entrypoint']}`",
        f"- deck_path: `{manifest['deck_path']}`",
        f"- deck_hash: `{manifest['deck_hash']}`",
        f"- validation_passed: `{validation['passed']}`",
        "",
        "Manual upload step: upload the generated .tar.gz through Kaggle or the competition UI.",
        "",
        "This bundle is a frozen baseline v2.4 artifact. It does not promote v4, replay priors, value modeling, RL, search, or any experimental policy flag.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _create_tar_gz(bundle_dir: Path, tar_gz_path: Path) -> Path:
    if tar_gz_path.exists():
        raise SubmissionBundleError(f"tar.gz already exists: {tar_gz_path}")
    tar_gz_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz_path, "w:gz") as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.add(path, arcname=path.relative_to(bundle_dir).as_posix(), recursive=False)
    return tar_gz_path


def _create_zip(bundle_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        raise SubmissionBundleError(f"zip already exists: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, path.relative_to(bundle_dir).as_posix())
    return zip_path


def _validation_with_archives(
    validation: dict[str, Any],
    archive_validations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(validation)
    updated["archives"] = archive_validations
    updated["passed"] = bool(validation["passed"]) and all(item["passed"] for item in archive_validations.values())
    return updated


def _raise_if_archive_validation_failed(kind: str, validation: dict[str, Any]) -> None:
    if validation["passed"]:
        return
    failed = [check for check in validation["checks"] if not check["ok"]]
    raise SubmissionBundleError(f"{kind} archive failed validation: {failed}")


def _build_manifest(
    plan: SubmissionPlan,
    validation: dict[str, Any],
    *,
    bundle_tar_gz: str | None,
    tar_gz_hash: str | None,
    bundle_zip: str | None,
    zip_hash: str | None,
) -> dict[str, Any]:
    deck_hash = sha256_file(plan.bundle_dir / DECK_PATH)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": plan.submission_id,
        "created_at": utc_now_iso(),
        "agent_spec": plan.agent_spec,
        "canonical_label": plan.canonical_label,
        "entrypoint": ENTRYPOINT,
        "deck_path": DECK_PATH,
        "deck_hash": deck_hash,
        "git_commit": _git_commit(plan.repo_root),
        "git_dirty": _git_dirty(plan.repo_root),
        "included_paths": list(plan.included_paths),
        "excluded_paths": list(plan.excluded_paths),
        "artifact_hashes": _artifact_hashes(plan.bundle_dir, plan.included_paths),
        "bundle_tar_gz": bundle_tar_gz,
        "tar_gz_hash": tar_gz_hash,
        "bundle_zip": bundle_zip,
        "zip_hash": zip_hash,
        "validation": validation,
        "notes": [
            "Default eval-safe behavior remains v2.4.",
            "Root main.py imports deepmew.policy, so the staged bundle includes the discovered local import closure.",
            "The .tar.gz artifact is the Kaggle-ready upload artifact; zip is retained only when explicitly emitted.",
            "No replay-prior online integration, value model, RL, PPO, Deep RL, belief model, or search is included.",
            "submission_manifest.json is not self-hashed inside artifact_hashes; archive hashes are recorded in the local manifest after archive creation.",
        ],
    }
    return validate_submission_manifest(manifest)


def build_submission(
    *,
    repo_root: Path | str = ".",
    agent: str,
    out_root: Path | str = DEFAULT_OUT_ROOT,
    dry_run: bool = False,
    make_zip: bool = True,
    make_tar_gz: bool = True,
    now: datetime | None = None,
) -> SubmissionBuildResult:
    plan = plan_submission(repo_root=repo_root, agent=agent, out_root=out_root, now=now)
    if dry_run:
        return SubmissionBuildResult(plan=plan, manifest=None, tar_gz_path=None, zip_path=None, dry_run=True)

    _copy_included_files(plan)
    validation = validate_bundle(plan.bundle_dir)
    if not validation["passed"]:
        failed = [check for check in validation["checks"] if not check["ok"] and not check.get("skipped")]
        raise SubmissionBundleError(f"staged submission bundle failed validation: {failed}")

    manifest = _build_manifest(
        plan,
        validation,
        bundle_tar_gz=None,
        tar_gz_hash=None,
        bundle_zip=None,
        zip_hash=None,
    )
    _write_readme(plan.bundle_dir / SUBMISSION_README, manifest)
    manifest["artifact_hashes"] = _artifact_hashes(plan.bundle_dir, plan.included_paths)
    write_submission_manifest(plan.bundle_dir / SUBMISSION_MANIFEST, manifest)

    tar_gz_path: Path | None = None
    archive_validations: dict[str, dict[str, Any]] = {}
    if make_tar_gz:
        tar_gz_path = plan.bundle_dir.parent / f"{plan.bundle_dir.name}.tar.gz"
        _create_tar_gz(plan.bundle_dir, tar_gz_path)
        tar_gz_validation = validate_archive(tar_gz_path, required_dependency_paths=plan.dependency_paths)
        _raise_if_archive_validation_failed("tar.gz", tar_gz_validation)
        archive_validations["tar_gz"] = tar_gz_validation
        manifest["bundle_tar_gz"] = tar_gz_path.name
        manifest["tar_gz_hash"] = sha256_file(tar_gz_path)

    zip_path: Path | None = None
    if make_zip:
        zip_path = plan.bundle_dir.parent / f"{plan.bundle_dir.name}.zip"
        _create_zip(plan.bundle_dir, zip_path)
        zip_validation = validate_archive(zip_path, required_dependency_paths=plan.dependency_paths)
        _raise_if_archive_validation_failed("zip", zip_validation)
        archive_validations["zip"] = zip_validation
        manifest["bundle_zip"] = zip_path.name
        manifest["zip_hash"] = sha256_file(zip_path)

    if archive_validations:
        manifest["validation"] = _validation_with_archives(validation, archive_validations)
    write_submission_manifest(plan.bundle_dir / SUBMISSION_MANIFEST, manifest)

    return SubmissionBuildResult(
        plan=plan,
        manifest=manifest,
        tar_gz_path=tar_gz_path,
        zip_path=zip_path,
        dry_run=False,
    )
