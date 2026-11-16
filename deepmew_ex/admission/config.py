"""Typed paths and trusted manifest inputs for a read-only V1 audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepmew_ex.experiments.config import OutputPaths, PROJECT_ROOT, component


@dataclass(frozen=True)
class AuditConfig:
    root: Path
    audit_id: str
    output_paths: OutputPaths
    expected_manifest: Path = PROJECT_ROOT / 'DATASET_MANIFEST.json'
    validation_receipt: Path = PROJECT_ROOT / 'VALIDATION_RECEIPT.json'
    project: Path = PROJECT_ROOT
    selection_seed: int = 42
    train_selection_limit: int = 4096
    validation_selection_limit: int = 1024

    @property
    def corpus(self) -> Path:
        return self.root.resolve(strict=True) / 'data' / 'clean_replay'

    @property
    def output(self) -> Path:
        return self.output_paths.run('deepmew_ex', 'v1_admission', self.audit_id)

    def validate(self) -> None:
        component(self.audit_id)
        self.output_paths.validate(self.project)
        if not self.corpus.is_dir() or not self.expected_manifest.is_file() or not self.validation_receipt.is_file():
            raise ValueError('corpus, trusted dataset manifest and validation receipt are required')
        if self.output.is_relative_to(self.corpus.resolve()) or self.corpus.resolve().is_relative_to(self.output):
            raise ValueError('audit outputs and source corpus must be separate')
        if self.train_selection_limit < 1 or self.validation_selection_limit < 1:
            raise ValueError('selection limits must be positive')
