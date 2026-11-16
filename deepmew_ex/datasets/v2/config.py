"""One output owner and fixed bounded golden conversion limits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from deepmew_ex.datasets.v2.schema import Payload, payload
from deepmew_ex.experiments.config import OutputPaths, PROJECT_ROOT, component
from deepmew_ex.registry.records import validate_hash


@dataclass(frozen=True)
class GoldenConfig:
    inventory: Path
    run_id: str
    paths: OutputPaths
    max_episodes: int = 48
    max_episode_bytes: int = 16 * 1024**2
    max_source_bytes: int = 192 * 1024**2
    max_output_bytes: int = 512 * 1024**2
    max_steps: int = 2000

    @property
    def output(self) -> Path:
        return self.paths.run('deepmew_ex', 'v2_golden', self.run_id)

    def load(self) -> Payload:
        component(self.run_id)
        self.paths.validate()
        if not 1 <= self.max_episodes <= 48 or not 1 <= self.max_episode_bytes <= 16 * 1024**2:
            raise ValueError('golden scope exceeds approved bounds')
        if not 1 <= self.max_source_bytes <= 192 * 1024**2 or not 1 <= self.max_output_bytes <= 512 * 1024**2 or not 3 <= self.max_steps <= 2000:
            raise ValueError('golden worker bounds exceeded')
        value = payload(json.loads(self.inventory.read_text(encoding='utf-8')))
        if value['schema'] != 'deepmew.ex.golden_inventory.v1' or value['purpose'] != 'golden_validation_only':
            raise ValueError('unsupported inventory or use; model training is not authorized')
        sources = value['sources']
        if not isinstance(sources, list) or not 1 <= len(sources) <= self.max_episodes:
            raise ValueError('empty or oversized golden inventory')
        hashes, games = set(), set()
        for source in sources:
            row = payload(source)
            validate_hash(row['sha256'])
            if row['sha256'] in hashes or row['native_game_id'] in games:
                raise ValueError('duplicate bytes or game identity across splits')
            hashes.add(row['sha256'])
            games.add(row['native_game_id'])
            if row['split'] not in ('golden_development', 'golden_validation') or row['module_version'] not in ('1.30.2', '1.32.0'):
                raise ValueError('unsupported split/source family')
            path = Path(row['path']).resolve()
            if path.is_relative_to(self.output) or path == self.output:
                raise ValueError('source and output locations must be separate')
        return value
