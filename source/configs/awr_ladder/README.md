# AWR ladder weighting configs

These configs are training-only inputs for `scripts/run_r18_awr_trainer.py` and
`scripts/run_r18_awr_iterate.py`. They do not affect `main.py`, submission
bundles, `cg/`, root `deck.csv`, or default `v2.4`.

Run a direct trainer job with:

```powershell
uv run python scripts/run_r18_awr_trainer.py `
  --refgames-dir <refgames-dir> `
  --init-checkpoint <checkpoint.pt> `
  --out-dir <out-dir> `
  --ladder-weighting-config configs/awr_ladder/A1_better_baseline.yaml
```

Run an iteration job with:

```powershell
uv run python scripts/run_r18_awr_iterate.py `
  --init-checkpoint <checkpoint.pt> `
  --league-config .research/curriculum/league.fast.yaml `
  --out-root .research/r18/awr_ladder_probe `
  --ladder-weighting-config configs/awr_ladder/A2_phase_bucket.yaml
```

Run the old-vs-new proof harness with:

```powershell
uv run python scripts/run_awr_ladder_ablation.py `
  --refgames-dir <refgames-dir> `
  --init-checkpoint <checkpoint.pt> `
  --out-dir .research/r20_ladder_aware_awr/a0_vs_a4_probe `
  --configs configs/awr_ladder/A0_current.yaml configs/awr_ladder/A4_exploiter.yaml
```

- `A0_current.yaml`: historical AWR behavior.
- `A1_better_baseline.yaml`: metadata-aware baseline grouping only.
- `A2_phase_bucket.yaml`: A1 plus phase and bucket multipliers.
- `A3_archetype.yaml`: A2 plus opponent-shell multipliers.
- `A4_exploiter.yaml`: A3 plus opponent-pilot multipliers.

`A4_exploiter.yaml` is an explicit ablation arm, not an automatic default. Use
the proof harness before treating it as better than `A0_current.yaml`.
