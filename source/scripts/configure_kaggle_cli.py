"""Persist the Kaggle CLI executable path for DeepMew replay infrastructure.

Writes ``.research/local/kaggle_cli.json`` so the K1.5/K1.6 downloader and scout can
resolve the Kaggle CLI without manual PATH debugging on every run. Resolution order
used by the tools is: explicit ``--kaggle-exe`` > ``DEEPMEW_KAGGLE_EXE`` env var >
this config file > PATH lookup.

Examples
--------
    python scripts/configure_kaggle_cli.py --kaggle-exe "C:/Users/you/AppData/Local/Programs/Python/Python313/Scripts/kaggle.exe"
    python scripts/configure_kaggle_cli.py --kaggle-exe kaggle --no-verify
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.download import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    KaggleExecutable,
    ensure_kaggle_cli,
    kaggle_config_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist the Kaggle CLI executable path for DeepMew.")
    parser.add_argument("--kaggle-exe", required=True, help="Path to the Kaggle CLI executable (or a name on PATH).")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Research root that holds local/kaggle_cli.json.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the `kaggle --help` health check.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kaggle_exe = str(args.kaggle_exe).strip()
    if not kaggle_exe:
        print("configure failed: --kaggle-exe must not be empty", file=sys.stderr)
        return 1

    config_path = kaggle_config_path(args.out_root)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    verified = False
    if not args.no_verify:
        import subprocess

        try:
            ensure_kaggle_cli(
                KaggleExecutable(command=kaggle_exe, explicit=True, resolution="explicit"),
                runner=lambda command, **kwargs: subprocess.run(list(command), **kwargs),
            )
            verified = True
        except Exception as exc:  # noqa: BLE001 - surface any health-check failure to the operator
            print(f"configure failed: Kaggle CLI health check failed for {kaggle_exe!r}: {exc}", file=sys.stderr)
            print("Re-run with --no-verify to store the path anyway.", file=sys.stderr)
            return 1

    payload = {
        "kaggle_exe": kaggle_exe,
        "verified": verified,
        "configured_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"kaggle_exe={kaggle_exe}")
    print(f"verified={verified}")
    print(f"config_path={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
