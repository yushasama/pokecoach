"""Report-only diagnostics over generated AWR refgame rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.curriculum.refgame_diagnostics import (  # noqa: E402
    analyze_refgame_rows, iter_refgame_rows, write_diagnostics_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refgames-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    rows = iter_refgame_rows(Path(args.refgames_dir), limit=int(args.limit) or None)
    result = analyze_refgame_rows(rows)
    outputs = write_diagnostics_outputs(result, Path(args.out_dir))
    print(json.dumps({"summary": {k: result[k] for k in ("rows", "games", "unavailable_fields")},
                      "outputs": outputs}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
