"""PufferLib adapter boundary for DeepMew RL.

The local Windows environment may not have PufferLib installed as a normal
package. This module is therefore intentionally import-light: it describes the
compatibility contract, can discover a locally patched PufferLib source checkout,
and checks availability without importing Puffer training modules on the default
RL path.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PufferCompatibility:
    available: bool
    native_available: bool
    blocker: str | None
    python: str
    platform: str
    adapter_status: str
    version: str | None = None
    module_path: str | None = None
    source_path: str | None = None
    native_blocker: str | None = None
    runtime_dll_dirs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "native_available": self.native_available,
            "blocker": self.blocker,
            "python": self.python,
            "platform": self.platform,
            "adapter_status": self.adapter_status,
            "version": self.version,
            "module_path": self.module_path,
            "source_path": self.source_path,
            "native_blocker": self.native_blocker,
            "runtime_dll_dirs": list(self.runtime_dll_dirs),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_source_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("DEEPMEW_PUFFERLIB_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_repo_root() / ".research" / "local" / "pufferlib_windows_compat" / "PufferLib-4.0")
    return candidates


def _source_has_package(path: Path) -> bool:
    return (path / "pufferlib" / "__init__.py").exists()


def _ensure_local_source_on_path() -> Path | None:
    """Add a local PufferLib source checkout to `sys.path` when available."""
    for candidate in _candidate_source_paths():
        resolved = candidate.resolve()
        if not _source_has_package(resolved):
            continue
        resolved_str = str(resolved)
        if resolved_str not in sys.path:
            sys.path.insert(0, resolved_str)
            importlib.invalidate_caches()
        return resolved
    return None


_DLL_DIRECTORY_HANDLES: dict[str, Any] = {}


def _candidate_dll_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dirs = os.environ.get("DEEPMEW_PUFFERLIB_DLL_DIR")
    if env_dirs:
        candidates.extend(Path(part) for part in env_dirs.split(os.pathsep) if part)
    candidates.append(Path("C:/msys64/ucrt64/bin"))
    return candidates


def _register_windows_dll_dirs() -> tuple[str, ...]:
    if platform.system().lower() != "windows" or not hasattr(os, "add_dll_directory"):
        return ()
    registered: list[str] = []
    for candidate in _candidate_dll_dirs():
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in _DLL_DIRECTORY_HANDLES:
            registered.append(resolved)
            continue
        _DLL_DIRECTORY_HANDLES[resolved] = os.add_dll_directory(resolved)
        registered.append(resolved)
    return tuple(dict.fromkeys(registered))


def _version_from_importable_package() -> str | None:
    try:
        module = importlib.import_module("pufferlib")
    except Exception:
        try:
            return importlib.metadata.version("pufferlib")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
    module_version = getattr(module, "__version__", None)
    if module_version is not None:
        return str(module_version)
    try:
        return importlib.metadata.version("pufferlib")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _missing_puffer_blocker() -> tuple[str, str]:
    if platform.system().lower() == "windows":
        return (
            "pufferlib is not importable in the active Python environment. "
            "Local install probes on this Windows/Python target failed before "
            "a Puffer API could be inspected: pufferlib 3.0.0 and 2.0.6 reject "
            "Windows during wheel metadata with `ValueError: Unsupported system: "
            "Windows`; pufferlib 0.7.3 requires legacy numpy==1.23.3, which does "
            "not build on Python 3.12 here. For the local PufferLib-4.0 source "
            "zip, run `python scripts/setup_pufferlib_windows_v0.py` to create "
            "a Python-importable compatibility checkout; use Linux/WSL/Modal for "
            "the native scale backend until `_C` is ported."
        ), "blocked_install_failed_windows_python312"
    return (
        "pufferlib is not importable in the active Python environment. Per "
        "INVARIANT #10, install/configure/verify pufferlib on this target before "
        "falling back to the Torch reference backend."
    ), "blocked_not_installed"


def _native_backend_status() -> tuple[bool, str | None, str]:
    native_spec = importlib.util.find_spec("pufferlib._C")
    if native_spec is None:
        return (
            False,
            "pufferlib Python package is importable, but native module `pufferlib._C` "
            "is absent. PufferLib 4.0 builds `_C` through the upstream Bash/C++/CUDA "
            "path; that native path is not Windows-compatible in this checkout.",
            "python_importable_native_missing",
        )
    try:
        native = importlib.import_module("pufferlib._C")
    except Exception as exc:
        return (
            False,
            f"pufferlib._C exists but failed to import: {type(exc).__name__}: {exc}",
            "python_importable_native_import_failed",
        )
    if getattr(native, "native_available", True) is False:
        reason = str(
            getattr(
                native,
                "native_blocker",
                "compatibility stub loaded; native PufferLib backend is unavailable",
            )
        )
        return False, reason, "python_importable_native_stub"
    required = ("create_vec", "puff_advantage_cpu")
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        return (
            False,
            f"pufferlib._C imported, but is missing required CPU symbols: {', '.join(missing)}",
            "python_importable_native_incomplete",
        )
    if platform.system().lower() == "windows" and not bool(getattr(native, "gpu", False)):
        return True, None, "pufferlib_windows_cpu_create_vec_importable"
    return True, None, "pufferlib_native_importable_api_probe_required"


def check_puffer_compatibility() -> PufferCompatibility:
    source_path = _ensure_local_source_on_path()
    runtime_dll_dirs = _register_windows_dll_dirs()
    spec = importlib.util.find_spec("pufferlib")
    if spec is None:
        blocker, status = _missing_puffer_blocker()
        return PufferCompatibility(
            available=False,
            native_available=False,
            blocker=blocker,
            python=platform.python_version(),
            platform=platform.platform(),
            adapter_status=status,
            source_path=str(source_path) if source_path else None,
            runtime_dll_dirs=runtime_dll_dirs,
        )

    version = _version_from_importable_package()
    native_available, native_blocker, status = _native_backend_status()
    blocker = None
    if not native_available:
        blocker = (
            "PufferLib is Python-importable for local API inspection, but the "
            f"native training backend is unavailable: {native_blocker}"
        )
    return PufferCompatibility(
        available=True,
        native_available=native_available,
        blocker=blocker,
        python=platform.python_version(),
        platform=platform.platform(),
        adapter_status=status,
        version=version,
        module_path=spec.origin,
        source_path=str(source_path) if source_path else None,
        native_blocker=native_blocker,
        runtime_dll_dirs=runtime_dll_dirs,
    )


def inspect_puffer_api() -> dict[str, Any]:
    """Inspect the installed/local PufferLib API surface without assuming docs."""
    compatibility = check_puffer_compatibility()
    modules: dict[str, dict[str, Any]] = {}
    for module_name in (
        "pufferlib",
        "pufferlib._C",
        "pufferlib.pufferl",
        "pufferlib.torch_pufferl",
        "pufferlib.models",
        "pufferlib.vector",
        "pufferlib.emulation",
    ):
        spec = importlib.util.find_spec(module_name)
        entry: dict[str, Any] = {"importable": spec is not None, "origin": spec.origin if spec else None}
        if spec is not None:
            try:
                module = importlib.import_module(module_name)
                entry["public_names"] = [name for name in dir(module) if not name.startswith("_")][:160]
            except Exception as exc:
                entry["import_error"] = f"{type(exc).__name__}: {exc}"
        modules[module_name] = entry
    native_symbols = {}
    native = modules.get("pufferlib._C", {})
    if native.get("importable") and "import_error" not in native:
        module = importlib.import_module("pufferlib._C")
        for name in ("create_vec", "create_pufferl", "rollouts", "train", "puff_advantage_cpu"):
            native_symbols[name] = hasattr(module, name)
    return {
        "compatibility": compatibility.to_dict(),
        "modules": modules,
        "native_symbols": native_symbols,
        "selected_wrapper": None,
        "selected_wrapper_status": (
            "blocked: installed Windows PufferLib has native _C.create_vec for compiled Ocean envs, "
            "but no pufferlib.vector/emulation Python wrapper module and no DeepMew native env/action binding"
        ),
    }


class DeepMewPufferEnvSpec:
    """Small descriptor for the planned Puffer vector env wrapper.

    The live implementation should preserve the same observation schema,
    autoregressive action semantics, reward function, opponent sampler, and
    checkpoint/eval contract as `masked_ppo`.
    """

    observation_kind = "fixed_float_vector"
    action_kind = "autoregressive_masked_selection_with_stop"
    reward_kind = "configured_reward_fn"
    vectorization_goal = "fastest compatible Python/vectorized Puffer path first"
    windows_compat_goal = "Windows CPU Puffer smoke first; DeepMew vector/action adapter before real training"
