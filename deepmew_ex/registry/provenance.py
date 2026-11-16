"""Exact executable-source identity shared by training and read-only audits."""

from __future__ import annotations

import subprocess
from pathlib import Path

from deepmew_ex.registry.store import canonical, digest, file_hash


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(project), *arguments], text=True).strip()


def code_identity(project: Path) -> dict[str, object]:
    files = sorted([*project.joinpath('deepmew_ex').rglob('*.py'), *project.joinpath('training').rglob('*.py')])
    hashes = {path.relative_to(project).as_posix(): file_hash(path) for path in files}
    return {'git_commit': git_output(project, 'rev-parse', 'HEAD'), 'git_status': git_output(project, 'status', '--porcelain'), 'source_files_sha256': hashes, 'source_manifest_sha256': digest(canonical(hashes)), 'dirty_patch_hash': digest(subprocess.check_output(['git', '-C', str(project), 'diff', '--binary', 'HEAD']))}
