"""R16: build a futility-guarded variant of a shipped distilled bundle.

Takes an existing validated bundle tarball (e.g. 20260706-r4f-canonical.tar.gz),
adds `futility_guard.py` (verbatim copy of deepmew/futility_guard/core.py), and
patches `main.py` to post-process the policy choice through the guard. Anchored
string replacement: the build FAILS if the template text drifted.

The source bundle is not modified. Output: a new bundle dir + tar.gz under
`.research/submissions/`. Validate with scripts/validate_r13_submission_bundle.py.

Usage:
  python scripts/build_r16_guarded_bundle.py \
      --source .research/submissions/20260706-r4f-canonical.tar.gz \
      --label 20260707-r4f-futility-guard
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMPORT_ANCHOR = "from r5_distilled_policy import DistilledPolicy, guard_selection, safe_fallback, valid_selection"
IMPORT_PATCH = (
    IMPORT_ANCHOR
    + "\nfrom futility_guard import FutilityGuard\n\n_FUTILITY_GUARD = FutilityGuard()"
)
CHOICE_ANCHOR = """    choice = guard_selection([int(index) for index in choice], select)
    if not valid_selection(choice, select):"""
CHOICE_PATCH = """    choice = guard_selection([int(index) for index in choice], select)
    try:
        adjusted = _FUTILITY_GUARD.process(obs, select, choice)
        choice = guard_selection([int(index) for index in adjusted], select)
    except Exception:
        pass
    if not valid_selection(choice, select):"""


def build(source: Path, label: str, out_root: Path) -> Path:
    bundle_dir = out_root / label
    if bundle_dir.exists():
        raise FileExistsError(f"refusing to overwrite {bundle_dir}")
    bundle_dir.mkdir(parents=True)

    if source.is_dir():
        for item in source.iterdir():
            if item.name == "__pycache__":
                continue
            if item.is_dir():
                shutil.copytree(item, bundle_dir / item.name, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(item, bundle_dir / item.name)
    else:
        with tarfile.open(source, "r:gz") as tar:
            tar.extractall(bundle_dir, filter="data")

    guard_source = ROOT / "deepmew" / "futility_guard" / "core.py"
    shutil.copy2(guard_source, bundle_dir / "futility_guard.py")

    main_path = bundle_dir / "main.py"
    text = main_path.read_text(encoding="utf-8")
    if IMPORT_ANCHOR not in text or CHOICE_ANCHOR not in text:
        raise RuntimeError("main.py template drifted: patch anchors not found")
    if "FutilityGuard" in text:
        raise RuntimeError("main.py already patched")
    text = text.replace(IMPORT_ANCHOR, IMPORT_PATCH, 1)
    text = text.replace(CHOICE_ANCHOR, CHOICE_PATCH, 1)
    main_path.write_text(text, encoding="utf-8")

    tar_path = out_root / f"{label}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in sorted(bundle_dir.rglob("*")):
            if item.is_file() and "__pycache__" not in item.parts:
                tar.add(item, arcname=str(item.relative_to(bundle_dir)))
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-root", default=str(ROOT / ".research" / "submissions"))
    args = parser.parse_args()
    tar_path = build(Path(args.source), args.label, Path(args.out_root))
    print("built:", tar_path)


if __name__ == "__main__":
    main()
