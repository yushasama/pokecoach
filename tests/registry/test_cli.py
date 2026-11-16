from __future__ import annotations

import json
import subprocess
import sys

from deepmew_ex.experiments.config import PROJECT_ROOT


def test_read_only_cli_does_not_import_models_or_engine() -> None:
    script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'cg'):
        raise AssertionError('model or engine imported by metadata command')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from deepmew_ex.cli import main
main(['registry', 'import-legacy', '--dry-run'])
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report['checkpoint_payloads_opened'] == 0 and report['records'] == 332
