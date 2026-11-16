"""Build and verify the sealed DeepMew Modal payload V2.1.

The manifest is based on logical files selected by the trainer's import closure
and the deserialization closure of every frozen checkpoint, plus explicit
runtime assets. Host paths, mount counts, timestamps, and files outside the
sealed payload root are deliberately outside the identity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MISSION = ROOT / ".research" / "x1_x2_direct_exploiter_first_wave"
MANIFEST_PATH = MISSION / "DEEPMEW_MODAL_PAYLOAD_MANIFEST_V2.json"
INVENTORY_PATH = MISSION / "RUNTIME_PAYLOAD_INVENTORY.json"
PACKAGING_AUDIT_PATH = MISSION / "MODAL_PAYLOAD_V2_PACKAGING_METHOD_AUDIT.json"
REMOTE_PAYLOAD_ROOT = PurePosixPath("/opt/deepmew_payload")
CHECKPOINT_ASSET_ROOT = PurePosixPath(
    ".research/x1_x2_direct_exploiter_first_wave/assets"
)
MANIFEST_VERSION = "2.2.0"
MANIFEST_SCHEMA = "deepmew.modal_payload_manifest.v2"
MANIFEST_CONTROL_NAME = "DEEPMEW_MODAL_PAYLOAD_MANIFEST_V2.json"
REQUIRED_ROLES = frozenset(
    {
        "REQUIRED_CODE",
        "REQUIRED_CONFIGURATION",
        "REQUIRED_CHECKPOINT",
        "REQUIRED_DECK",
        "REQUIRED_SCHEMA",
        "REQUIRED_STATIC_ASSET",
    }
)
EXCLUDED_ROLES = frozenset(
    {"OPTIONAL_DIAGNOSTIC", "LOCAL_ONLY", "DISALLOWED_REMOTE"}
)


class PayloadV2Error(RuntimeError):
    """The logical payload or its sealed staging tree violated the V2 contract."""


@dataclass(frozen=True)
class SourceSpec:
    source_path: Path
    logical_relative_path: str
    intended_remote_path: str
    payload_role: str
    owning_package_or_component: str
    required_import_or_consumer: str


@dataclass(frozen=True)
class FrozenCheckpointSpec:
    source_path: Path
    asset_name: str
    required_import_or_consumer: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def manifest_identity(entries: Sequence[Mapping[str, Any]]) -> str:
    logical_rows = [
        {
            "logical_relative_path": str(entry["logical_relative_path"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in sorted(entries, key=lambda row: str(row["logical_relative_path"]))
    ]

    return hashlib.sha256(_canonical_json_bytes(logical_rows)).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="ascii",
            newline="\n",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(payload), stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_repository_identity(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise PayloadV2Error("source repository commit identity is invalid")

    return f"git:{commit}:dirty-worktree-content-addressed"


def frozen_checkpoint_specs(root: Path = ROOT) -> tuple[FrozenCheckpointSpec, ...]:
    return (
        FrozenCheckpointSpec(
            root
            / ".research"
            / "terminal_league_starmie_grimmsnarl_v0"
            / "control_closeout_starmie_20260713"
            / "remote"
            / "final_actor.pt",
            "d1_actor.pt",
            "frozen D1 target",
        ),
        FrozenCheckpointSpec(
            root
            / ".research"
            / "terminal_league_starmie_grimmsnarl_v0"
            / "e1_starmie_complete_enhanced_generation_20260714"
            / "production_e1_20260716"
            / "stage_05_terminal_ppo.pt.actor.pt",
            "e1_actor.pt",
            "frozen E1 target",
        ),
        FrozenCheckpointSpec(
            Path("D:/DeepMew_artifacts/checkpoints/r22-staryu-full-league-a4-8c-r7-BEST.pt"),
            "r7.pt",
            "frozen r7 base/history",
        ),
        FrozenCheckpointSpec(
            root
            / ".research"
            / "r12"
            / "lane1_r4f_focal_reweight"
            / "selected_ar_bc_r4f_reweighted_candidate.pt",
            "r12_exploiter.pt",
            "frozen hard-tail checkpoint",
        ),
    )


def staged_checkpoint_paths(payload_root: Path) -> dict[str, Path]:
    asset_root = payload_root.joinpath(*CHECKPOINT_ASSET_ROOT.parts)

    return {
        checkpoint.asset_name: asset_root / checkpoint.asset_name
        for checkpoint in frozen_checkpoint_specs()
    }


def load_frozen_checkpoints(checkpoint_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Deserialize frozen checkpoints without inspecting or persisting tensor values."""
    import torch

    expected_names = {checkpoint.asset_name for checkpoint in frozen_checkpoint_specs()}
    observed_names = set(checkpoint_paths)
    if observed_names != expected_names:
        raise PayloadV2Error(
            "checkpoint deserialization set mismatch: "
            f"expected={sorted(expected_names)} observed={sorted(observed_names)}"
        )
    receipts: list[dict[str, Any]] = []
    for asset_name in sorted(checkpoint_paths):
        checkpoint_path = Path(checkpoint_paths[asset_name])
        if not checkpoint_path.is_file():
            raise PayloadV2Error(f"frozen checkpoint is missing: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_type = type(checkpoint)
        top_level_keys = (
            sorted(str(key) for key in checkpoint)
            if isinstance(checkpoint, Mapping)
            else []
        )
        receipts.append(
            {
                "asset_name": asset_name,
                "sha256": sha256_file(checkpoint_path),
                "python_type": (
                    f"{checkpoint_type.__module__}.{checkpoint_type.__qualname__}"
                ),
                "top_level_keys": top_level_keys,
                "deserialized": True,
            }
        )
        del checkpoint

    return {
        "schema_version": "deepmew.modal_checkpoint_deserialization_receipt.v1",
        "status": "PASS",
        "map_location": "cpu",
        "weights_only": False,
        "checkpoint_count": len(receipts),
        "checkpoints": receipts,
        "tensor_values_persisted": False,
    }


def _canonical_module_name(logical_path: str) -> str:
    path = PurePosixPath(logical_path)
    parts = list(path.parts)
    if not parts or path.suffix != ".py":
        raise PayloadV2Error(f"cannot derive Python module name from {logical_path}")
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = path.stem

    return ".".join(parts)


def _is_exact_case_file(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        exact_child = next(
            (child for child in current.iterdir() if child.name == part),
            None,
        )
        if exact_child is None:
            return False
        current = exact_child

    return current.is_file()


def _project_module_files(root: Path, module_name: str) -> list[str]:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return []
    logicals: list[str] = []
    for index in range(1, len(parts)):
        package_init = root.joinpath(*parts[:index], "__init__.py")
        if _is_exact_case_file(root, package_init):
            logicals.append(package_init.relative_to(root).as_posix())
    module_file = root.joinpath(*parts).with_suffix(".py")
    package_file = root.joinpath(*parts, "__init__.py")
    if _is_exact_case_file(root, module_file):
        logicals.append(module_file.relative_to(root).as_posix())
    elif _is_exact_case_file(root, package_file):
        logicals.append(package_file.relative_to(root).as_posix())

    return logicals


def _static_project_imports(root: Path, logical_path: str) -> set[str]:
    source_path = root / logical_path
    tree = ast.parse(
        source_path.read_text(encoding="utf-8-sig"),
        filename=str(source_path),
    )
    module_name = _canonical_module_name(logical_path)
    is_package = PurePosixPath(logical_path).name == "__init__.py"
    package_name = module_name if is_package else module_name.rpartition(".")[0]
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if not package_name:
                    continue
                relative_name = "." * node.level + (node.module or "")
                try:
                    base_name = importlib.util.resolve_name(relative_name, package_name)
                except (ImportError, ValueError):
                    continue
            else:
                base_name = node.module or ""
            if base_name:
                imported_modules.add(base_name)
                imported_modules.update(
                    f"{base_name}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            continue
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_import_module = (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
        ) or (
            isinstance(function, ast.Name)
            and function.id in {"import_module", "__import__"}
        )
        first_argument = node.args[0]
        if (
            is_import_module
            and isinstance(first_argument, ast.Constant)
            and isinstance(first_argument.value, str)
        ):
            imported_modules.add(first_argument.value)
    logicals: set[str] = set()
    for imported_module in imported_modules:
        logicals.update(_project_module_files(root, imported_module))

    return logicals


def _module_source_specs(root: Path) -> list[SourceSpec]:
    scripts = root / "scripts"
    for search_path in (root, scripts):
        if str(search_path) not in sys.path:
            sys.path.insert(0, str(search_path))
    importlib.import_module("run_terminal_league_ppo")
    load_frozen_checkpoints(
        {
            checkpoint.asset_name: checkpoint.source_path
            for checkpoint in frozen_checkpoint_specs(root)
        }
    )
    consumers: dict[str, set[str]] = {}
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(root)
        except (OSError, ValueError):
            continue
        if not path.is_file() or ".venv" in relative.parts or "__pycache__" in relative.parts:
            continue
        logical = relative.as_posix()
        if logical.startswith("deepmew/submissions/"):
            raise PayloadV2Error("protected submission source entered the import closure")
        if logical == "deck.csv":
            raise PayloadV2Error("protected root deck entered the import closure")
        if not (
            logical == "main.py"
            or logical.startswith("cg/")
            or logical.startswith("deepmew/")
            or logical.startswith("scripts/")
        ):
            continue
        consumers.setdefault(logical, set()).add(module_name)
    queue = list(consumers)
    scanned: set[str] = set()
    while queue:
        logical = queue.pop()
        if logical in scanned or not logical.endswith(".py"):
            continue
        scanned.add(logical)
        for imported_logical in sorted(_static_project_imports(root, logical)):
            if imported_logical.startswith("deepmew/submissions/"):
                raise PayloadV2Error(
                    "protected submission source entered the static import closure"
                )
            if imported_logical == "deck.csv":
                raise PayloadV2Error("protected root deck entered the static import closure")
            if imported_logical not in consumers:
                queue.append(imported_logical)
            consumers.setdefault(imported_logical, set()).add(
                f"static import closure:{logical}"
            )
    consumers.setdefault("scripts/modal_payload_v2.py", set()).add("modal_payload_v2")
    consumers.setdefault("scripts/modal_direct_exploiter.py", set()).add(
        "Modal function source; sealed duplicate parity"
    )
    specs: list[SourceSpec] = []
    for logical, module_names in sorted(consumers.items()):
        source = root / logical
        if not source.is_file():
            raise PayloadV2Error(f"required source file is missing: {logical}")
        owner = logical.split("/", 1)[0]
        specs.append(
            SourceSpec(
                source_path=source,
                logical_relative_path=logical,
                intended_remote_path=str(REMOTE_PAYLOAD_ROOT / logical),
                payload_role="REQUIRED_CODE",
                owning_package_or_component=owner,
                required_import_or_consumer=",".join(sorted(module_names)),
            )
        )

    return specs


def _explicit_source_specs(root: Path) -> list[SourceSpec]:
    mission = root / ".research" / "x1_x2_direct_exploiter_first_wave"
    panel = root / ".research" / "post_e1_request_a_m0_m1_v1" / "corpus" / "decks"
    specs = [
        SourceSpec(
            root / "cg" / "libcg.so",
            "cg/libcg.so",
            str(REMOTE_PAYLOAD_ROOT / "cg/libcg.so"),
            "REQUIRED_STATIC_ASSET",
            "CABT engine",
            "cg.game,cg.sim ctypes runtime",
        ),
        SourceSpec(
            mission / "DECK_MANIFEST.json",
            ".research/x1_x2_direct_exploiter_first_wave/DECK_MANIFEST.json",
            str(REMOTE_PAYLOAD_ROOT / ".research/x1_x2_direct_exploiter_first_wave/DECK_MANIFEST.json"),
            "REQUIRED_SCHEMA",
            "X1/X2 campaign",
            "deepmew.rl.opponents policy/deck binding resolver",
        ),
        SourceSpec(
            mission / "X1_CAMPAIGN_CONFIG.json",
            ".research/x1_x2_direct_exploiter_first_wave/X1_CAMPAIGN_CONFIG.json",
            str(REMOTE_PAYLOAD_ROOT / ".research/x1_x2_direct_exploiter_first_wave/X1_CAMPAIGN_CONFIG.json"),
            "REQUIRED_CONFIGURATION",
            "X1 campaign",
            "scripts.run_terminal_league_ppo",
        ),
        SourceSpec(
            mission / "X2_CAMPAIGN_CONFIG.json",
            ".research/x1_x2_direct_exploiter_first_wave/X2_CAMPAIGN_CONFIG.json",
            str(REMOTE_PAYLOAD_ROOT / ".research/x1_x2_direct_exploiter_first_wave/X2_CAMPAIGN_CONFIG.json"),
            "REQUIRED_CONFIGURATION",
            "X2 campaign",
            "scripts.run_terminal_league_ppo",
        ),
        SourceSpec(
            mission / "X1_REMOTE_ARTIFACT_MAP_V2.json",
            ".research/x1_x2_direct_exploiter_first_wave/X1_REMOTE_ARTIFACT_MAP_V2.json",
            str(REMOTE_PAYLOAD_ROOT / ".research/x1_x2_direct_exploiter_first_wave/X1_REMOTE_ARTIFACT_MAP_V2.json"),
            "REQUIRED_CONFIGURATION",
            "X1 Modal packaging",
            "scripts.run_terminal_league_ppo artifact remapping",
        ),
        SourceSpec(
            mission / "X2_REMOTE_ARTIFACT_MAP_V2.json",
            ".research/x1_x2_direct_exploiter_first_wave/X2_REMOTE_ARTIFACT_MAP_V2.json",
            str(REMOTE_PAYLOAD_ROOT / ".research/x1_x2_direct_exploiter_first_wave/X2_REMOTE_ARTIFACT_MAP_V2.json"),
            "REQUIRED_CONFIGURATION",
            "X2 Modal packaging",
            "scripts.run_terminal_league_ppo artifact remapping",
        ),
        SourceSpec(
            mission / "PROTECTED_SURFACE_BASELINE_V2.json",
            ".research/x1_x2_direct_exploiter_first_wave/PROTECTED_SURFACE_BASELINE_V2.json",
            str(REMOTE_PAYLOAD_ROOT / ".research/x1_x2_direct_exploiter_first_wave/PROTECTED_SURFACE_BASELINE_V2.json"),
            "REQUIRED_SCHEMA",
            "DeepMew protected surfaces",
            "Modal V2 runtime contamination verifier",
        ),
        SourceSpec(
            root / ".research" / "curriculum" / "decks" / "top_staryu_windecks_b15853cdfb83.deck.csv",
            ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv",
            str(REMOTE_PAYLOAD_ROOT / ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"),
            "REQUIRED_DECK",
            "D1/E1 Starmie deck",
            "X1 learner and D1/E1/r7 bindings",
        ),
    ]
    deck_names = (
        "09b937915fb787c62365c02c76bbd2d25b847fbd06f62292dd20f251fbdce19a.deck.csv",
        "6644aa1457640c4c3d0c132d2b5e493ec808f97162a8ed59ec15448732b9f007.deck.csv",
        "7a41b8db186a8f5056b573182dcd412c4efc97f8aef1c3678e7fdf9624b8b248.deck.csv",
    )
    for name in deck_names:
        logical = f".research/post_e1_request_a_m0_m1_v1/corpus/decks/{name}"
        specs.append(
            SourceSpec(
                panel / name,
                logical,
                str(REMOTE_PAYLOAD_ROOT / logical),
                "REQUIRED_DECK",
                "frozen X1/X2 reference panel",
                "X1/X2 policy/deck bindings",
            )
        )
    for checkpoint in frozen_checkpoint_specs(root):
        logical = str(CHECKPOINT_ASSET_ROOT / checkpoint.asset_name)
        specs.append(
            SourceSpec(
                checkpoint.source_path,
                logical,
                str(REMOTE_PAYLOAD_ROOT / logical),
                "REQUIRED_CHECKPOINT",
                "X1/X2 frozen model assets",
                checkpoint.required_import_or_consumer,
            )
        )

    return specs


def source_specs(root: Path = ROOT) -> list[SourceSpec]:
    by_logical: dict[str, SourceSpec] = {}
    for spec in _module_source_specs(root) + _explicit_source_specs(root):
        prior = by_logical.get(spec.logical_relative_path)
        if prior is not None and prior.source_path.resolve() != spec.source_path.resolve():
            raise PayloadV2Error(f"duplicate logical source: {spec.logical_relative_path}")
        by_logical[spec.logical_relative_path] = spec

    return [by_logical[name] for name in sorted(by_logical)]


def build_manifest(root: Path = ROOT) -> dict[str, Any]:
    repository_identity = _source_repository_identity(root)
    entries: list[dict[str, Any]] = []
    for spec in source_specs(root):
        if spec.payload_role not in REQUIRED_ROLES:
            raise PayloadV2Error(f"manifest entry has non-required role: {spec.payload_role}")
        if not spec.source_path.is_file():
            raise PayloadV2Error(f"required payload source is missing: {spec.source_path}")
        entries.append(
            {
                "logical_relative_path": spec.logical_relative_path,
                "intended_remote_path": spec.intended_remote_path,
                "sha256": sha256_file(spec.source_path),
                "byte_size": spec.source_path.stat().st_size,
                "payload_role": spec.payload_role,
                "owning_package_or_component": spec.owning_package_or_component,
                "required_import_or_consumer": spec.required_import_or_consumer,
                "source_repository_identity": repository_identity,
                "manifest_version": MANIFEST_VERSION,
            }
        )
    identity = manifest_identity(entries)

    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "status": "LOCAL_PARITY_CANDIDATE",
        "sealed_remote_root": str(REMOTE_PAYLOAD_ROOT),
        "manifest_control_remote_path": str(REMOTE_PAYLOAD_ROOT / MANIFEST_CONTROL_NAME),
        "identity_algorithm": "sha256(canonical sorted logical_relative_path and sha256 rows)",
        "payload_identity_sha256": identity,
        "required_entry_count": len(entries),
        "source_repository_identity": repository_identity,
        "entries": entries,
    }


def _staging_path(staging_root: Path, entry: Mapping[str, Any]) -> Path:
    intended = PurePosixPath(str(entry["intended_remote_path"]))
    try:
        relative = intended.relative_to(REMOTE_PAYLOAD_ROOT)
    except ValueError as exc:
        raise PayloadV2Error(f"entry escapes sealed root: {intended}") from exc

    return staging_root.joinpath(*relative.parts)


def construct_staging_tree(
    staging_root: Path,
    manifest: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> None:
    if staging_root.exists():
        raise PayloadV2Error(f"staging root already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    specs = {spec.logical_relative_path: spec for spec in source_specs(root)}
    for entry in manifest["entries"]:
        spec = specs[str(entry["logical_relative_path"])]
        target = _staging_path(staging_root, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec.source_path, target)
    _atomic_json(staging_root / MANIFEST_CONTROL_NAME, manifest)


def verify_staging_tree(
    staging_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise PayloadV2Error("unexpected V2 manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PayloadV2Error("V2 manifest entries are missing")
    observed_identity = manifest_identity(entries)
    if observed_identity != manifest.get("payload_identity_sha256"):
        raise PayloadV2Error("V2 manifest identity does not match its logical entries")
    expected_files = {MANIFEST_CONTROL_NAME}
    verified_roles: dict[str, int] = {}
    for entry in entries:
        path = _staging_path(staging_root, entry)
        relative = path.relative_to(staging_root).as_posix()
        if relative in expected_files:
            raise PayloadV2Error(f"duplicate staged payload path: {relative}")
        expected_files.add(relative)
        if not path.is_file():
            raise PayloadV2Error(f"required staged file is missing: {relative}")
        size = path.stat().st_size
        if size != int(entry["byte_size"]):
            raise PayloadV2Error(
                f"required staged file size mismatch: {relative}: expected {entry['byte_size']}, got {size}"
            )
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise PayloadV2Error(
                f"required staged file hash mismatch: {relative}: expected {entry['sha256']}, got {digest}"
            )
        role = str(entry["payload_role"])
        verified_roles[role] = verified_roles.get(role, 0) + 1
    observed_files = {
        path.relative_to(staging_root).as_posix()
        for path in staging_root.rglob("*")
        if path.is_file()
    }
    undeclared = sorted(observed_files - expected_files)
    missing = sorted(expected_files - observed_files)
    if undeclared:
        raise PayloadV2Error(f"undeclared file inside sealed payload root: {undeclared[0]}")
    if missing:
        raise PayloadV2Error(f"declared file missing from sealed payload root: {missing[0]}")

    return {
        "schema_version": "deepmew.modal_payload_v2_staging_verification.v1",
        "status": "PASS",
        "payload_identity_sha256": observed_identity,
        "required_entry_count": len(entries),
        "observed_declared_file_count": len(observed_files),
        "verified_roles": dict(sorted(verified_roles.items())),
        "undeclared_files": [],
        "missing_files": [],
    }


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise PayloadV2Error("V2 manifest must be a JSON object")

    return payload


def verify_runtime_imports(
    payload_root: Path,
    *,
    probe_modules: Sequence[str] = (),
) -> dict[str, Any]:
    root_text = str(payload_root)
    scripts_text = str(payload_root / "scripts")
    for search_path in (root_text, scripts_text):
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
    previous = Path.cwd()
    os.chdir(payload_root)
    try:
        trainer = importlib.import_module("run_terminal_league_ppo")
        benchmark = Path(payload_root / "scripts" / "run_terminal_league_ppo.py")
        if not benchmark.is_file() or not callable(getattr(trainer, "main", None)):
            raise PayloadV2Error("benchmark entry point did not resolve")
        trainer_path = Path(trainer.__file__).resolve()
        try:
            trainer_path.relative_to(payload_root.resolve())
        except ValueError as exc:
            raise PayloadV2Error(
                f"trainer imported outside sealed payload root: {trainer_path}"
            ) from exc
        probed_modules: dict[str, str] = {}
        for module_name in probe_modules:
            module = importlib.import_module(module_name)
            module_path = Path(str(getattr(module, "__file__", ""))).resolve()
            try:
                module_path.relative_to(payload_root.resolve())
            except ValueError as exc:
                raise PayloadV2Error(
                    f"probe module imported outside sealed payload root: "
                    f"{module_name}={module_path}"
                ) from exc
            probed_modules[module_name] = str(module_path)
    finally:
        os.chdir(previous)

    return {
        "trainer_module": str(trainer_path),
        "trainer_imported": True,
        "benchmark_entry_point": str(benchmark),
        "benchmark_entry_point_resolved": True,
        "probed_modules": dict(sorted(probed_modules.items())),
    }


def excluded_inventory() -> list[dict[str, str]]:
    return [
        {"path_or_pattern": ".git/", "classification": "DISALLOWED_REMOTE", "reason": "repository metadata"},
        {"path_or_pattern": "**/__pycache__/", "classification": "LOCAL_ONLY", "reason": "generated bytecode cache"},
        {"path_or_pattern": ".venv/", "classification": "DISALLOWED_REMOTE", "reason": "local virtual environment"},
        {"path_or_pattern": "deepmew/submissions/", "classification": "DISALLOWED_REMOTE", "reason": "protected submission surface"},
        {"path_or_pattern": "deck.csv", "classification": "DISALLOWED_REMOTE", "reason": "protected root deck; all runtime decks explicitly bound"},
        {"path_or_pattern": "cg/cg.dll", "classification": "LOCAL_ONLY", "reason": "Windows CABT library; remote image is Linux"},
        {"path_or_pattern": "data/", "classification": "LOCAL_ONLY", "reason": "not opened by trainer import/runtime contracts"},
        {"path_or_pattern": "configs/", "classification": "LOCAL_ONLY", "reason": "not opened by the frozen trainer import/runtime contracts"},
        {"path_or_pattern": ".research/**/*.sqlite*", "classification": "DISALLOWED_REMOTE", "reason": "no database consumer in frozen workload"},
        {"path_or_pattern": ".research/**/*replay*", "classification": "DISALLOWED_REMOTE", "reason": "no replay-store consumer in terminal PPO workload"},
        {"path_or_pattern": ".research/**/*REPORT*", "classification": "LOCAL_ONLY", "reason": "historical/reporting artifacts"},
        {"path_or_pattern": "tests/", "classification": "LOCAL_ONLY", "reason": "local verification only"},
        {"path_or_pattern": "scripts/* outside recorded import closure", "classification": "LOCAL_ONLY", "reason": "unrelated scripts"},
        {"path_or_pattern": "PAYLOAD_IDENTITY.json", "classification": "LOCAL_ONLY", "reason": "retired V1 historical identity"},
        {"path_or_pattern": "COST_LEDGER.json", "classification": "LOCAL_ONLY", "reason": "local budget ledger"},
    ]


def write_inventory_and_audit(manifest: Mapping[str, Any]) -> None:
    inventory = {
        "schema_version": "deepmew.modal_payload_inventory.v2",
        "status": "DECLARED_FROM_RUNTIME_IMPORT_AND_CHECKPOINT_DESERIALIZATION_CLOSURE",
        "required_entries": manifest["entries"],
        "excluded": excluded_inventory(),
    }
    audit = {
        "schema_version": "deepmew.modal_payload_packaging_audit.v2",
        "sealed_remote_root": str(REMOTE_PAYLOAD_ROOT),
        "sources": [
            {
                "local_source_path": "mission-owned deterministic staging tree",
                "packaging_method": "modal.Image.add_local_dir(copy=True, ignore=())",
                "intended_remote_path": str(REMOTE_PAYLOAD_ROOT),
                "filtering_or_ignore_rules": "none at packaging time; staging tree is allowlist-built from manifest",
                "non_python_assets_included": True,
                "dot_prefixed_files_excluded": False,
                "packaging_phase": "image build",
            },
            {
                "local_source_path": ".research/modal/q_gpu_runtime",
                "packaging_method": "modal.Image.uv_sync(frozen=True)",
                "intended_remote_path": "system Python environment outside payload parity root",
                "filtering_or_ignore_rules": "uv.lock exact dependency resolution",
                "non_python_assets_included": "dependency-owned",
                "dot_prefixed_files_excluded": "not applicable",
                "packaging_phase": "image build",
            },
            {
                "local_source_path": "scripts/modal_direct_exploiter.py",
                "packaging_method": "Modal function-source mount plus identical sealed manifest copy",
                "intended_remote_path": "/root/modal_direct_exploiter.py and /opt/deepmew_payload/scripts/modal_direct_exploiter.py",
                "filtering_or_ignore_rules": "exact file",
                "non_python_assets_included": False,
                "dot_prefixed_files_excluded": "not applicable",
                "packaging_phase": "container startup/function hydration and image build",
            },
        ],
        "identity_semantics": "sorted logical path plus content SHA-256; no mount counts, host paths, timestamps, or inode metadata",
    }
    _atomic_json(INVENTORY_PATH, inventory)
    _atomic_json(PACKAGING_AUDIT_PATH, audit)


def build_command(staging_root: Path) -> dict[str, Any]:
    manifest = build_manifest(ROOT)
    _atomic_json(MANIFEST_PATH, manifest)
    construct_staging_tree(staging_root, manifest, root=ROOT)
    verification = verify_staging_tree(staging_root, manifest)
    write_inventory_and_audit(manifest)

    return {
        "manifest_path": str(MANIFEST_PATH),
        "staging_root": str(staging_root),
        "manifest_identity_sha256": manifest["payload_identity_sha256"],
        "verification": verification,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_command(Path(args.staging_root).resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
