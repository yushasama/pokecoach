"""Conservative code-quality and artifact-output audit for DeepMew."""

from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from deepmew.reports.markdown import bullets, section, table, title


CODE_QUALITY_SCHEMA_VERSION = "deepmew.code_quality.v1"
RUN_MANIFEST_SCHEMA_VERSION = "deepmew.run_manifest.v1"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "data",
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
    "README.md",
    "READ_POLICY.md",
    "REPORT_INDEX.md",
    "REPO_ARCHITECTURE_PLAN.md",
    "REPRODUCIBILITY.md",
    "STATE.md",
    "SYSTEM_PROMPT.md",
    "TASKS.md",
    "ptcg_builder_spec.md",
}

GENERIC_MODULE_NAMES = {"base", "common", "core", "helpers", "misc", "shared", "utils"}
DEV_IMPORT_MARKERS = (
    "deepmew.artifacts",
    "deepmew.repo",
    "deepmew.reports",
    "deepmew.trace",
    "scripts.",
    ".research",
    "requests",
    "urllib",
    "socket",
    "torch",
    "tensorflow",
    "sklearn",
)
MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "kind",
    "command",
    "inputs",
    "outputs",
    "status",
    "summary",
}

CARD_ID_RE = re.compile(r"\b[A-Z][A-Z0-9_]*(?:CARD|ENERGY|POKEMON|TRAINER|LINE|ATTACKER|SETUP)[A-Z0-9_]*_IDS?\s*=")
ATTACK_ID_RE = re.compile(r"\b[A-Z][A-Z0-9_]*_ATTACK_ID\s*=")
MAGIC_SCORE_RE = re.compile(r"\bscore\s*[+-]=\s*-?\d+\b")
SCORE_DELTA_RE = re.compile(r"\b(?:attach|attack|play|setup|score)_score_delta\b|\b_score_delta\b", re.IGNORECASE)
JSON_WRITE_RE = re.compile(r"json\.(?:dump|dumps)\s*\(|write_json\s*\(")
DIRECT_JSON_WRITE_RE = re.compile(r"json\.(?:dump|dumps)\s*\(|\.write_text\s*\([^)]*json", re.DOTALL)
DIRECT_MARKDOWN_WRITE_RE = re.compile(r"\.write_text\s*\([^)]*(?:report|markdown|md_path|\\.md)", re.DOTALL)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _warning(code: str, path: str, message: str, *, line: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "path": path, "message": message}
    if line is not None:
        item["line"] = line
    return item


def _line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _read_text_if_small(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in SKIP_DIR_NAMES)
        current = Path(dirpath)
        for filename in sorted(filenames):
            yield current / filename


def _is_python_source(rel_path: str) -> bool:
    return rel_path.endswith(".py") and not rel_path.startswith("cg/")


def _is_script(rel_path: str) -> bool:
    return rel_path.startswith("scripts/") and rel_path.endswith(".py")


def _is_profile_path(rel_path: str) -> bool:
    return (
        rel_path.startswith("deepmew/cards/")
        or rel_path.startswith("deepmew/card_profiles/")
        or rel_path.startswith("deepmew/deck_profiles/")
    )


def _is_generated_extension(rel_path: str) -> bool:
    return Path(rel_path).suffix.lower() in {".json", ".jsonl", ".md", ".txt"}


def _loads_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_python(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _python_stats(rel_path: str, text: str) -> dict[str, Any]:
    tree = _parse_python(text)
    functions: list[str] = []
    classes: list[str] = []
    dataclasses: list[str] = []
    enums: list[str] = []
    import_lines = [line.strip() for line in text.splitlines() if line.lstrip().startswith(("import ", "from "))]
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                decorator_names = {
                    getattr(decorator, "id", "")
                    or getattr(getattr(decorator, "func", None), "id", "")
                    or getattr(decorator, "attr", "")
                    for decorator in node.decorator_list
                }
                base_names = {getattr(base, "id", "") or getattr(base, "attr", "") for base in node.bases}
                if "dataclass" in decorator_names:
                    dataclasses.append(node.name)
                if "Enum" in base_names or "StrEnum" in base_names:
                    enums.append(node.name)

    lines = text.count("\n") + (1 if text else 0)
    report_markers = sum(marker in text for marker in ("report", "markdown", "format_", "render_", "table("))
    policy_markers = sum(marker in text for marker in ("score", "OptionType", "SelectContext", "choose_action", "policy"))
    arena_markers = sum(marker in text for marker in ("battle_start", "battle_select", "run_pair", "run_matrix"))
    artifact_markers = sum(marker in text for marker in ("manifest", "write_json", "write_markdown", "write_text", ".research"))
    responsibility_score = report_markers + policy_markers + arena_markers + artifact_markers
    return {
        "path": rel_path,
        "lines": lines,
        "functions": len(functions),
        "classes": len(classes),
        "dataclasses": len(dataclasses),
        "enums": len(enums),
        "imports": len(import_lines),
        "function_names": functions,
        "class_names": classes,
        "responsibility_score": responsibility_score,
    }


def _test_candidates_for(rel_path: str) -> set[str]:
    parts = rel_path.split("/")
    stem = Path(rel_path).stem
    candidates = {f"tests/test_{stem}.py"}
    if len(parts) >= 3:
        package = parts[1]
        candidates.add(f"tests/test_{package}.py")
        candidates.add(f"tests/test_{package}_{stem}.py")
    if len(parts) == 2:
        candidates.add(f"tests/test_{stem}_*.py")
    return candidates


def _has_test_for(rel_path: str, test_files: set[str]) -> bool:
    candidates = _test_candidates_for(rel_path)
    if candidates & test_files:
        return True
    stem = Path(rel_path).stem
    return any(path.startswith(f"tests/test_{stem}") and path.endswith(".py") for path in test_files)


def _scan_python_file(rel_path: str, text: str, test_files: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    stats = _python_stats(rel_path, text)
    warnings: list[dict[str, Any]] = []
    helper_names: list[str] = []
    lines = stats["lines"]

    if lines > 400:
        warnings.append(
            _warning(
                "large_module_possible_split",
                rel_path,
                f"Module has {lines} lines; review for a future targeted split.",
            )
        )
    if lines > 500 and stats["responsibility_score"] >= 3:
        warnings.append(
            _warning(
                "god_module_candidate",
                rel_path,
                "Module mixes several responsibility markers; review recommended before adding more behavior.",
            )
        )
    if _is_script(rel_path) and (stats["dataclasses"] or stats["classes"] or stats["enums"] or lines > 250 or stats["functions"] > 8):
        warnings.append(
            _warning(
                "script_contains_reusable_logic_possible",
                rel_path,
                "Script contains data models, many helpers, or substantial logic; reusable pieces may belong under deepmew/.",
            )
        )
    if rel_path == "main.py" and any(marker in text for marker in DEV_IMPORT_MARKERS):
        warnings.append(
            _warning(
                "main_imports_dev_tooling_possible",
                rel_path,
                "Submission entrypoint appears to import dev, report, trace, repo, network, or heavy tooling.",
            )
        )
    if rel_path == "deepmew/policy/scoring.py" and ("score +=" in text or "score -=" in text) and "OptionType" in text:
        warnings.append(
            _warning(
                "policy_scoring_boundary_review",
                rel_path,
                "`deepmew/policy/scoring.py` still owns several scoring branches; future new multi-line rules should live under deepmew/policy_rules/.",
            )
        )
    if Path(rel_path).stem in GENERIC_MODULE_NAMES and not rel_path.startswith("cg/"):
        warnings.append(
            _warning(
                "module_name_too_generic",
                rel_path,
                "Generic module names make ownership hard to infer; prefer domain-specific names.",
            )
        )
    if rel_path.startswith("deepmew/") and rel_path.endswith(".py") and Path(rel_path).name != "__init__.py":
        if not _has_test_for(rel_path, test_files):
            warnings.append(
                _warning(
                    "missing_tests_for_new_module_possible",
                    rel_path,
                    "No obvious matching test module was found; review whether coverage exists under a broader test file.",
                )
            )

    if _is_script(rel_path) and re.search(r"\.write_text\s*\(|open\s*\([^)]*['\"]w|json\.(?:dump|dumps)\s*\(", text, re.DOTALL):
        warnings.append(
            _warning(
                "script_writes_artifacts_directly",
                rel_path,
                "Script appears to write artifact files directly; prefer deepmew.artifacts helpers unless this is a documented compatibility path.",
            )
        )
    if _is_script(rel_path) and DIRECT_JSON_WRITE_RE.search(text):
        warnings.append(
            _warning(
                "manual_json_write_in_script",
                rel_path,
                "Script appears to write JSON manually; prefer deepmew.artifacts.writers.write_json.",
            )
        )
    if _is_script(rel_path) and DIRECT_MARKDOWN_WRITE_RE.search(text):
        warnings.append(
            _warning(
                "manual_markdown_write_in_script",
                rel_path,
                "Script appears to write Markdown manually; prefer deepmew.artifacts.writers.write_markdown.",
            )
        )
    if (
        not rel_path.startswith("deepmew/artifacts/")
        and not rel_path.startswith("tests/")
        and re.search(r"\.write_text\s*\(|json\.(?:dump|dumps)\s*\(", text, re.DOTALL)
        and _is_python_source(rel_path)
    ):
        warnings.append(
            _warning(
                "artifact_helper_bypass_possible",
                rel_path,
                "File writes JSON/Markdown/text directly; review whether it should use deepmew.artifacts writers.",
            )
        )
    if re.search(r"json\.(?:load|loads)\s*\(", text) and "schema_version" not in text and "read_manifest" not in text:
        warnings.append(
            _warning(
                "artifact_reader_schema_unchecked_possible",
                rel_path,
                "JSON reader does not visibly check schema_version; review before relying on old artifacts.",
            )
        )

    if not _is_profile_path(rel_path) and not rel_path.startswith("tests/"):
        for match in CARD_ID_RE.finditer(text):
            warnings.append(
                _warning(
                    "hardcoded_card_id_outside_profile",
                    rel_path,
                    "Card/deck ID constant outside a card profile; centralize before it spreads further.",
                    line=_line_number(text, match.start()),
                )
            )
        for match in ATTACK_ID_RE.finditer(text):
            warnings.append(
                _warning(
                    "hardcoded_attack_id_outside_profile",
                    rel_path,
                    "Attack ID constant outside a card profile; centralize before it spreads further.",
                    line=_line_number(text, match.start()),
                )
            )

    for match in MAGIC_SCORE_RE.finditer(text):
        warnings.append(
            _warning(
                "magic_score_delta",
                rel_path,
                "Anonymous numeric score adjustment; prefer named constants or weight dataclasses.",
                line=_line_number(text, match.start()),
            )
        )
    for match in SCORE_DELTA_RE.finditer(text):
        warnings.append(
            _warning(
                "magic_score_delta",
                rel_path,
                "Score-delta field or variable found; review whether its owner, scope, and ablation are clear.",
                line=_line_number(text, match.start()),
            )
        )

    helper_names.extend(name for name in stats["function_names"] if name in {"_enum_name", "_get", "_card_id"})
    return stats, warnings, helper_names


def _load_report_index(root: Path) -> str:
    path = root / ".research" / "REPORT_INDEX.md"
    if path.exists():
        return _read_text_if_small(path, max_bytes=500_000)
    return ""


def _is_research_report_candidate(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if len(parts) >= 3 and parts[0] == ".research" and parts[1] == "reports" and Path(rel_path).suffix.lower() == ".md":
        return True
    if len(parts) != 2 or parts[0] != ".research":
        return False
    name = parts[1]
    if name in RESEARCH_SOURCE_NAMES:
        return False
    return name.endswith(".md")


def _scan_non_python_file(rel_path: str, report_index: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    parts = rel_path.split("/")
    suffix = Path(rel_path).suffix.lower()
    if parts[0] in {"deepmew", "scripts", "tests"} and suffix in {".json", ".jsonl", ".md", ".txt"}:
        warnings.append(
            _warning(
                "generated_artifact_in_source_tree",
                rel_path,
                "Generated-looking artifact lives under source/script/test directories.",
            )
        )
    if rel_path.startswith(".research/arena_runs/") and suffix in {".json", ".md"}:
        warnings.append(
            _warning(
                "legacy_flat_arena_output",
                rel_path,
                "Legacy flat arena output is preserved evidence; do not delete or silently move.",
            )
        )
    elif rel_path.startswith(".research/") and not rel_path.startswith(".research/runs/") and suffix in {".json", ".jsonl"}:
        warnings.append(
            _warning(
                "generated_output_outside_research_runs",
                rel_path,
                "Generated JSON/JSONL outside `.research/runs`; migrate only through a future compatibility-aware task.",
            )
        )
        warnings.append(
            _warning(
                "flat_generated_output",
                rel_path,
                "Artifact output is in a flat folder rather than a manifest-backed run directory.",
            )
        )
    elif rel_path.startswith(".research/reports/"):
        return warnings
    elif rel_path.startswith(".research/") and not rel_path.startswith(".research/runs/") and suffix == ".txt":
        warnings.append(
            _warning(
                "generated_output_outside_research_runs",
                rel_path,
                "Generated-looking text output lives outside `.research/runs`; preserve unless a migration task approves moving it.",
            )
        )
    if _is_research_report_candidate(rel_path) and report_index and rel_path not in report_index:
        warnings.append(
            _warning(
                "missing_report_index_entry",
                rel_path,
                "Research report is not mentioned in `.research/REPORT_INDEX.md`.",
            )
        )
    return warnings


def _run_directories(root: Path) -> list[Path]:
    runs_root = root / ".research" / "runs"
    if not runs_root.exists():
        return []
    run_dirs: list[Path] = []
    for kind_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        for day_dir in sorted(path for path in kind_dir.iterdir() if path.is_dir()):
            for run_dir in sorted(path for path in day_dir.iterdir() if path.is_dir()):
                run_dirs.append(run_dir)
    return run_dirs


def _scan_manifest(root: Path, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rel_run_dir = _rel(run_dir, root)
    manifest_path = run_dir / "manifest.json"
    item: dict[str, Any] = {"path": rel_run_dir, "has_manifest": manifest_path.exists(), "missing_fields": []}
    warnings: list[dict[str, Any]] = []
    if not manifest_path.exists():
        warnings.append(
            _warning(
                "run_output_without_manifest",
                rel_run_dir,
                "Generated run directory is missing manifest.json.",
            )
        )
        return item, warnings

    manifest = _loads_json(manifest_path)
    if manifest is None:
        item["manifest_valid_json"] = False
        warnings.append(
            _warning(
                "missing_run_schema_version",
                _rel(manifest_path, root),
                "Manifest is not a JSON object or could not be parsed.",
            )
        )
        return item, warnings

    item["manifest_valid_json"] = True
    missing = sorted(MANIFEST_REQUIRED_FIELDS - set(manifest))
    item["missing_fields"] = missing
    if "schema_version" in missing or manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        warnings.append(
            _warning(
                "missing_run_schema_version",
                _rel(manifest_path, root),
                "Manifest schema_version is missing or unsupported.",
            )
        )
    for field, code in (
        ("command", "manifest_missing_command"),
        ("inputs", "manifest_missing_inputs"),
        ("outputs", "manifest_missing_outputs"),
        ("summary", "manifest_missing_summary"),
    ):
        if field in missing:
            warnings.append(_warning(code, _rel(manifest_path, root), f"Manifest is missing `{field}`."))
    return item, warnings


def _artifact_discipline(
    root: Path,
    files: list[str],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dirs = _run_directories(root)
    run_items: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        item, manifest_warnings = _scan_manifest(root, run_dir)
        run_items.append(item)
        warnings.extend(manifest_warnings)

    legacy_flat_outputs = [
        rel_path
        for rel_path in files
        if rel_path.startswith(".research/arena_runs/") and Path(rel_path).suffix.lower() in {".json", ".md"}
    ]
    outside_runs = [
        warning["path"]
        for warning in warnings
        if warning["code"] == "generated_output_outside_research_runs"
    ]
    direct_writer_scripts = sorted(
        {warning["path"] for warning in warnings if warning["code"] in {"script_writes_artifacts_directly", "manual_json_write_in_script", "manual_markdown_write_in_script"}}
    )
    migration_tasks = []
    if "scripts/arena.py" in direct_writer_scripts or "deepmew/arena.py" in {w["path"] for w in warnings}:
        migration_tasks.append("migrate `scripts/arena.py` default output to `.research/runs/arena/<YYYYMMDD>/<run_id>/` with manifests")
    if legacy_flat_outputs:
        migration_tasks.append("keep `.research/arena_runs/` readable as legacy evidence; do not delete or silently move old outputs")
    if direct_writer_scripts:
        migration_tasks.append("centralize direct JSON/Markdown writers behind `deepmew.artifacts.writers` or document compatibility exceptions")
    if any(path.startswith("scripts/diagnose_") for path in direct_writer_scripts):
        migration_tasks.append("add manifest support to diagnosis scripts still writing manually")
    if outside_runs:
        migration_tasks.append("index or migrate artifact outputs outside `.research/runs/` through explicit future tasks")

    return {
        "generated_run_directories_scanned": len(run_items),
        "run_directories_with_manifests": sum(1 for item in run_items if item["has_manifest"]),
        "run_directories_missing_manifests": sum(1 for item in run_items if not item["has_manifest"]),
        "legacy_flat_outputs": len(legacy_flat_outputs),
        "legacy_flat_output_paths": legacy_flat_outputs,
        "generated_outputs_outside_research_runs": sorted(set(outside_runs)),
        "scripts_that_write_json_or_markdown_directly": direct_writer_scripts,
        "top_recommended_migration_tasks": migration_tasks,
    }


def scan_code_quality(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    report_index = _load_report_index(root)
    all_files = sorted(_rel(path, root) for path in _iter_files(root))
    test_files = {path for path in all_files if path.startswith("tests/") and path.endswith(".py")}
    modules: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    helper_paths: dict[str, list[str]] = defaultdict(list)

    for rel_path in all_files:
        path = root / rel_path
        if _is_python_source(rel_path):
            text = _read_text_if_small(path)
            stats, file_warnings, helper_names = _scan_python_file(rel_path, text, test_files)
            modules.append(stats)
            warnings.extend(file_warnings)
            for helper_name in helper_names:
                helper_paths[helper_name].append(rel_path)
        elif _is_generated_extension(rel_path):
            warnings.extend(_scan_non_python_file(rel_path, report_index))

    top_level_deepmew_modules = [
        module["path"]
        for module in modules
        if module["path"].startswith("deepmew/") and module["path"].count("/") == 1 and module["path"].endswith(".py")
    ]
    if len(top_level_deepmew_modules) > 6:
        warnings.append(
            _warning(
                "top_level_deepmew_sprawl",
                "deepmew",
                f"{len(top_level_deepmew_modules)} flat top-level deepmew modules exist; new reusable code should use subpackages.",
            )
        )

    for helper_name, paths in sorted(helper_paths.items()):
        if len(paths) <= 1:
            continue
        code = "duplicate_enum_helper_possible" if helper_name == "_enum_name" else "duplicate_get_helper_possible"
        warnings.append(
            _warning(
                code,
                ", ".join(sorted(paths)),
                f"Helper `{helper_name}` appears in multiple files; consider a shared helper only if it stays domain-neutral.",
            )
        )

    artifact = _artifact_discipline(root, all_files, warnings)
    warning_counts = Counter(item["code"] for item in warnings)
    top_modules = sorted(modules, key=lambda item: (-item["lines"], item["path"]))
    responsibility_smells = sorted(
        (module for module in modules if module["responsibility_score"] >= 3 or module["lines"] > 400),
        key=lambda item: (-item["responsibility_score"], -item["lines"], item["path"]),
    )
    return {
        "schema_version": CODE_QUALITY_SCHEMA_VERSION,
        "root": str(root),
        "generated_at_utc": _utc_now_iso(),
        "summary": {
            "files_scanned": len(all_files),
            "python_modules_scanned": len(modules),
            "warning_count": len(warnings),
            "warning_counts": dict(sorted(warning_counts.items())),
            "top_level_deepmew_module_count": len(top_level_deepmew_modules),
        },
        "modules": modules,
        "top_modules": top_modules[:20],
        "responsibility_smell_modules": responsibility_smells[:20],
        "artifact_discipline": artifact,
        "warnings": warnings,
    }


def _warning_rows(audit: dict[str, Any], code: str, limit: int = 12) -> list[list[object]]:
    rows = []
    for warning in audit["warnings"]:
        if warning["code"] != code:
            continue
        location = warning["path"]
        if "line" in warning:
            location = f"{location}:{warning['line']}"
        rows.append([location, warning["message"]])
        if len(rows) >= limit:
            break
    return rows


def _task_rows(tasks: Iterable[str]) -> list[str]:
    default_tasks = [
        "centralize current-deck card and attack IDs under `deepmew/deck_profiles/` before adding more card-specific rules",
        "keep `deepmew/arena/` split by specs, selection, runner, matrix, and reports",
        "keep `deepmew/policy/` split by models, selection, attack pressure, scoring, and action selection",
        "keep the migrated `deepmew/state_machine/` package split by state, tables, readiness, transition, effects, and rendering",
        "do not add new flat `deepmew/*.py` modules unless they are tiny compatibility shims",
        "review artifact-output paths before adding new diagnosis/probe scripts; default to manifest-backed `.research/runs/` output",
    ]
    return list(dict.fromkeys([*tasks, *default_tasks]))


def render_code_quality_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    artifact = audit["artifact_discipline"]
    warning_rows = sorted(summary["warning_counts"].items())
    top_module_rows = [
        [item["path"], item["lines"], item["functions"], item["classes"], item["responsibility_score"]]
        for item in audit["top_modules"][:12]
    ]
    smell_rows = [
        [item["path"], item["lines"], item["responsibility_score"], item["dataclasses"], item["classes"]]
        for item in audit["responsibility_smell_modules"][:12]
    ]
    card_rows = _warning_rows(audit, "hardcoded_card_id_outside_profile") + _warning_rows(audit, "hardcoded_attack_id_outside_profile")
    score_rows = _warning_rows(audit, "magic_score_delta")

    parts = [
        title("DeepMew Code Quality Audit"),
        f"- Schema: `{audit['schema_version']}`\n",
        f"- Root: `{audit['root']}`\n",
        f"- Generated: `{audit['generated_at_utc']}`\n",
        f"- Files scanned: {summary['files_scanned']}\n",
        f"- Python modules scanned: {summary['python_modules_scanned']}\n",
        f"- Warnings: {summary['warning_count']} review-recommended signals\n\n",
        section("Warning Counts"),
        table(["Warning", "Count"], warning_rows or [["none", 0]]),
        "\n",
        section("Top Module-Size Table"),
        table(["Module", "Lines", "Functions", "Classes", "Responsibility score"], top_module_rows or [["none", 0, 0, 0, 0]]),
        "\n",
        section("Top Responsibility-Smell Modules"),
        table(["Module", "Lines", "Responsibility score", "Dataclasses", "Classes"], smell_rows or [["none", 0, 0, 0, 0]]),
        "\n",
        section("Card-ID Hardcoding Locations"),
    ]
    parts.append(table(["Location", "Message"], card_rows) if card_rows else "No card/attack ID hardcoding warnings.\n")
    parts.extend(["\n", section("Magic Score-Delta Locations")])
    parts.append(table(["Location", "Message"], score_rows) if score_rows else "No magic score-delta warnings.\n")
    parts.extend(
        [
            "\n",
            section("Artifact-Output Discipline Findings"),
            f"- Generated run directories scanned: {artifact['generated_run_directories_scanned']}\n",
            f"- With manifests: {artifact['run_directories_with_manifests']}\n",
            f"- Missing manifests: {artifact['run_directories_missing_manifests']}\n",
            f"- Legacy flat outputs: {artifact['legacy_flat_outputs']}\n",
            f"- Scripts that appear to write JSON/Markdown directly: {len(artifact['scripts_that_write_json_or_markdown_directly'])}\n\n",
            "Scripts with direct JSON/Markdown writes:\n",
            bullets(f"`{path}`" for path in artifact["scripts_that_write_json_or_markdown_directly"])
            if artifact["scripts_that_write_json_or_markdown_directly"]
            else "None detected.\n",
            "\nTop recommended migration tasks:\n",
            bullets(artifact["top_recommended_migration_tasks"]) if artifact["top_recommended_migration_tasks"] else "None required by this scan.\n",
            "\n",
            section("Recommended Future Refactor Tasks"),
            bullets(_task_rows(artifact["top_recommended_migration_tasks"])),
            "\n",
            section("Do Not Do Now"),
            bullets(
                [
                    "Do not delete or move old `.research/arena_runs/` outputs.",
                    "Do not refactor all artifact writers in this pass.",
                    "Do not promote or alter gameplay policy based on this audit.",
                    "Do not add new flat `deepmew/*.py` modules; use the existing package homes.",
                ]
            ),
            "\n",
            section("Notes"),
            bullets(
                [
                    "Warnings are conservative review signals, not bug claims.",
                    "Legacy paths are labeled as legacy evidence when applicable.",
                    "Manifests are the preferred reproducibility boundary for future trace, probe, diagnosis, and audit runs.",
                ]
            ),
        ]
    )
    return "".join(parts)
