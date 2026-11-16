"""Conservative repository organization scanner for DeepMew."""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepmew.reports.markdown import bullets, section, table, title


SCAN_SCHEMA_VERSION = "deepmew.repo_scan.v1"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "build",
    "dist",
    ".eggs",
    ".tox",
}

ROOT_DOC_NAMES = {
    ".gitignore",
    "FUTURE_IDEAS.md",
    "HANDOFF_SUMMARY.md",
    "LICENSE",
    "OPEN_ARCHITECTURE_DECISIONS.md",
    "PTCG_Bible.md",
    "README.md",
    "READ_ME_FIRST.md",
}

RESEARCH_SOURCE_NAMES = {
    "BENCHMARKS.md",
    "BIBLE_COMPANION.md",
    "BUGS.md",
    "CODE_ORGANIZATION.md",
    "CONSTRAINTS.md",
    "CURRENT_STATE_COMPACT.md",
    "DECISION_LOG.md",
    "ENVIRONMENT.md",
    "EVALUATION.md",
    "EXTERNAL_LLM_PROTOCOL.md",
    "HANDOFF.md",
    "HARDWARE.md",
    "INVARIANTS.md",
    "MAINTENANCE.md",
    "MATH.md",
    "PRIOR_WORK.md",
    "REPORT_INDEX.md",
    "REPO_ARCHITECTURE_PLAN.md",
    "README.md",
    "READ_POLICY.md",
    "REPRODUCIBILITY.md",
    "STATE.md",
    "SYSTEM_PROMPT.md",
    "TASKS.md",
    "ptcg_builder_spec.md",
}

TOP_LEVEL_ALLOWED = {
    ".gitignore",
    "FUTURE_IDEAS.md",
    "HANDOFF_SUMMARY.md",
    "LICENSE",
    "OPEN_ARCHITECTURE_DECISIONS.md",
    "PTCG_Bible.md",
    "README.md",
    "deck.csv",
    "main.py",
}

DEV_IMPORT_MARKERS = (
    "deepmew.artifacts",
    "deepmew.repo",
    "deepmew.reports",
    "scripts.",
    ".research",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _classify(rel_path: str, is_dir: bool = False) -> str:
    parts = rel_path.split("/")
    name = parts[-1]
    if is_dir and name in SKIP_DIR_NAMES:
        return "cache_or_temp"
    if rel_path == "main.py":
        return "submission_entrypoint"
    if rel_path == "deck.csv":
        return "shipped_deck"
    if parts[0] == "cg":
        return "vendored_engine"
    if parts[0] == "deepmew":
        if name == "__pycache__":
            return "cache_or_temp"
        if name.endswith(".py"):
            return "importable_deepmew_code"
        return "unknown"
    if parts[0] == "scripts":
        if name == "__pycache__":
            return "cache_or_temp"
        if name.endswith(".py"):
            return "cli_script"
        return "unknown"
    if parts[0] == "tests":
        if name == "__pycache__":
            return "cache_or_temp"
        if name.endswith(".py"):
            return "test"
        return "unknown"
    if parts[0] == "data":
        return "competition_data_reference"
    if parts[0] == ".research":
        if len(parts) > 1 and parts[1] in {"architecture", "environment", "governance"}:
            return "research_source_of_truth"
        if len(parts) > 1 and parts[1] == "reports":
            return "research_audit_or_report"
        if len(parts) > 1 and parts[1] == "skills":
            return "research_skill"
        if len(parts) > 1 and parts[1] == "prompts":
            return "research_source_of_truth"
        if rel_path == ".research/runs/README.md":
            return "research_source_of_truth"
        if len(parts) > 1 and parts[1] == "runs":
            return "generated_run_artifact"
        if len(parts) > 1 and parts[1] == "arena_runs":
            return "generated_legacy_arena_artifact"
        if name in RESEARCH_SOURCE_NAMES:
            return "research_source_of_truth"
        if name.endswith(".md") or name.endswith(".txt"):
            return "research_audit_or_report"
        return "unknown"
    if len(parts) == 1 and name in ROOT_DOC_NAMES:
        return "project_doc"
    return "unknown"


def _warning(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _read_text_if_small(path: Path, max_bytes: int = 300_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _load_report_index(root: Path) -> str:
    index = root / ".research" / "REPORT_INDEX.md"
    if index.exists():
        return _read_text_if_small(index, max_bytes=500_000)
    return ""


def _scan_file_warnings(root: Path, path: Path, rel_path: str, role: str, report_index: str) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    name = path.name
    parts = rel_path.split("/")
    size = path.stat().st_size
    if len(parts) == 1 and name.endswith(".py") and name != "main.py":
        warnings.append(
            _warning(
                "root_python_file_not_whitelisted",
                rel_path,
                "Root Python files other than main.py can blur submission vs tooling boundaries.",
            )
        )
    if parts[0] in {"deepmew", "scripts", "tests"} and name.endswith(".json"):
        warnings.append(
            _warning(
                "generated_json_in_source_dir",
                rel_path,
                "JSON under source or test directories may be generated evidence in the wrong place.",
            )
        )
    if role == "research_audit_or_report" and name.endswith(".md") and report_index and rel_path not in report_index:
        warnings.append(
            _warning(
                "generated_md_without_index",
                rel_path,
                "Research report is not mentioned in .research/REPORT_INDEX.md.",
            )
        )
    if role == "generated_legacy_arena_artifact":
        warnings.append(
            _warning(
                "arena_run_flat_output_legacy",
                rel_path,
                "Legacy arena evidence lives outside .research/runs and may not have a manifest.",
            )
        )
    if role == "unknown" and len(parts) == 1:
        warnings.append(
            _warning(
                "unknown_top_level_file",
                rel_path,
                "Unknown top-level files should be classified or moved behind an explicit convention.",
            )
        )
    if size > 5_000_000 and parts[0] != "cg":
        warnings.append(
            _warning(
                "large_file_under_repo",
                rel_path,
                "Large non-vendored file may be generated evidence or cache data.",
            )
        )
    if name.endswith(".py") and parts[0] not in {"cg"}:
        text = _read_text_if_small(path)
        import_lines = [
            line.strip()
            for line in text.splitlines()
            if line.lstrip().startswith(("import ", "from "))
        ]
        if any(".research" in line for line in import_lines):
            warnings.append(
                _warning(
                    "import_from_research_possible",
                    rel_path,
                    "Runtime code appears to import from .research, which is docs/evidence only.",
                )
            )
        if rel_path == "main.py" and any(marker in text for marker in DEV_IMPORT_MARKERS):
            warnings.append(
                _warning(
                    "submission_imports_dev_tooling_possible",
                    rel_path,
                    "main.py appears to import dev/reporting/repo-scan modules.",
                )
            )
        if parts[0] == "scripts":
            function_count = text.count("\ndef ") + text.count("\nclass ")
            line_count = text.count("\n") + 1
            if line_count > 250 or function_count > 8:
                warnings.append(
                    _warning(
                        "script_contains_reusable_logic_possible",
                        rel_path,
                        "Large script may contain reusable logic that belongs under deepmew/.",
                    )
                )
    return warnings


def scan_repo(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    report_index = _load_report_index(root)
    files: list[dict[str, Any]] = []
    skipped_dirs: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_dir = "." if current == root else _rel(current, root)
        kept_dirnames: list[str] = []
        for dirname in sorted(dirnames):
            rel_path = dirname if rel_dir == "." else f"{rel_dir}/{dirname}"
            if dirname in SKIP_DIR_NAMES:
                role = _classify(rel_path, is_dir=True)
                skipped_dirs.append({"path": rel_path, "role": role})
                if dirname != ".git":
                    warnings.append(
                        _warning(
                            "cache_or_pycache_present",
                            rel_path,
                            "Cache/build directory is present under the repo and was skipped.",
                        )
                    )
            else:
                kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in sorted(filenames):
            path = current / filename
            rel_path = _rel(path, root)
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            role = _classify(rel_path)
            files.append({"path": rel_path, "role": role, "size": size})
            warnings.extend(_scan_file_warnings(root, path, rel_path, role, report_index))

    deepmew_flat_modules = [
        item for item in files if item["path"].startswith("deepmew/") and item["path"].count("/") == 1 and item["path"].endswith(".py")
    ]
    if len(deepmew_flat_modules) > 8:
        warnings.append(
            _warning(
                "flat_deepmew_module_count_high",
                "deepmew",
                "Many flat deepmew modules are present; consider subpackages for new reusable systems.",
            )
        )

    seen_run_roots: set[Path] = set()
    for item in files:
        rel_path = item["path"]
        parts = rel_path.split("/")
        if len(parts) >= 5 and parts[0] == ".research" and parts[1] == "runs":
            run_root = root / parts[0] / parts[1] / parts[2] / parts[3] / parts[4]
            if run_root in seen_run_roots:
                continue
            seen_run_roots.add(run_root)
            if run_root.is_dir() and not (run_root / "manifest.json").exists():
                warnings.append(
                    _warning(
                        "research_artifact_without_manifest",
                        "/".join(parts[:5]),
                        "Generated run directory is missing manifest.json.",
                    )
                )

    role_counts = Counter(item["role"] for item in files)
    top_level = sorted({item["path"].split("/")[0] for item in files})
    unknown_files = sum(1 for item in files if item["role"] == "unknown")
    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "root": str(root),
        "generated_at_utc": _utc_now_iso(),
        "files": files,
        "summary": {
            "total_files": len(files),
            "role_counts": dict(sorted(role_counts.items())),
            "skipped_dirs": skipped_dirs,
            "top_level_entries": top_level,
            "unknown_files": unknown_files,
            "warning_count": len(warnings),
        },
        "warnings": warnings,
    }


def render_scan_markdown(scan: dict[str, Any]) -> str:
    summary = scan["summary"]
    role_rows = sorted(summary["role_counts"].items())
    warning_counts = Counter(item["code"] for item in scan["warnings"])
    warning_rows = sorted(warning_counts.items())
    examples = scan["warnings"][:20]

    parts = [
        title("DeepMew Repo Scan"),
        f"- Schema: `{scan['schema_version']}`\n",
        f"- Root: `{scan['root']}`\n",
        f"- Generated: `{scan['generated_at_utc']}`\n",
        f"- Files scanned: {summary['total_files']}\n",
        f"- Warnings: {summary['warning_count']}\n\n",
        section("Role Counts"),
        table(["Role", "Count"], role_rows),
        "\n",
        section("Top-Level Entries"),
        bullets(f"`{entry}`" for entry in summary["top_level_entries"]),
        "\n",
        section("Warning Counts"),
        table(["Warning", "Count"], warning_rows or [("none", 0)]),
        "\n",
        section("Warning Examples"),
    ]
    if examples:
        parts.append(table(["Code", "Path", "Message"], [[w["code"], w["path"], w["message"]] for w in examples]))
    else:
        parts.append("No warnings.\n")
    parts.extend(
        [
            "\n",
            section("Notes"),
            bullets(
                [
                    "This scan is conservative and may emit possible-boundary warnings for human review.",
                    "`cg/` is treated as vendored engine glue.",
                    "`data/` is treated as local competition/reference data, not submission runtime.",
                    "Generated evidence should move toward `.research/runs/<kind>/<YYYYMMDD>/<run_id>/` with a manifest.",
                ]
            ),
        ]
    )
    return "".join(parts)
