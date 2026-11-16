"""Block a legacy trainer's snapshot boundary without interrupting raw training."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil


@dataclass(frozen=True)
class BoundaryConfig:
    run: Path
    output: Path


def acquire_boundary(index: Path) -> sqlite3.Connection:
    if not index.is_file():
        raise FileNotFoundError(index)
    connection = sqlite3.connect(f'file:{index.resolve().as_posix()}?mode=rw', uri=True, timeout=30)
    try:
        connection.execute('BEGIN EXCLUSIVE')
    except BaseException:
        connection.close()
        raise
    return connection


def guard(config: BoundaryConfig) -> None:
    environment = json.loads((config.run / 'environment.json').read_text())
    process = psutil.Process(environment['pid'])
    if config.run.name not in process.cmdline():
        raise ValueError('process identity does not match the run')
    progress = json.loads((config.run / 'progress.jsonl').read_text().splitlines()[-1])
    if progress['phase'] != 'raw' or progress['updates'] < 1:
        raise ValueError('boundary guard requires active raw temporal training')
    config.output.mkdir(parents=True, exist_ok=False)
    connection = acquire_boundary(config.run / 'snapshot-index.sqlite3')
    record = {'run': str(config.run), 'trainer_pid': process.pid, 'trainer_created': process.create_time(), 'guard_pid': psutil.Process().pid, 'status': 'armed', 'armed_utc': datetime.now(timezone.utc).isoformat(), 'observed_progress': progress, 'mechanism': 'exclusive read barrier on snapshot index; no data changes; raw training and validation use other files', 'authority': 'User directed temporal phase to continue uninterrupted and prohibited subsequent snapshot training'}
    path = config.output / 'boundary.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    try:
        while process.is_running():
            time.sleep(1)
    finally:
        connection.rollback()
        connection.close()
    record.update(status='released_after_trainer_exit', released_utc=datetime.now(timezone.utc).isoformat())
    failure = config.run / 'failure.json'
    record['trainer_failure'] = json.loads(failure.read_text()) if failure.exists() else None
    record['temporal_checkpoints'] = [str(item) for item in sorted((config.run / 'checkpoints').glob('phase-end-*-raw-*.pt'))]
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard(BoundaryConfig(args.run.resolve(), args.output.resolve()))


if __name__ == '__main__':
    main()
