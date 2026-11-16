# Vendored: 7th-place Pokémon TCG agent (Z3rox-dev)

- Upstream: https://github.com/Z3rox-dev/pokemon-tcg-7th-place-transformer-agent
- Commit: `1dde9eca485bb2088913c855bf0164afd47998c5`
- License: MIT, copyright (c) 2026 Giovanni (github.com/Z3rox-dev). See `UPSTREAM_LICENSE`.
- Contents: the upstream `kaggle_submission/` directory, unchanged (Python caches removed).
  The upstream README is kept as `UPSTREAM_README.md`.

PokéCoach runs it through `deepmew_ex/coach/external_worker.py` in an isolated
`python -I` subprocess, as the coach and as one of the random opponents.
