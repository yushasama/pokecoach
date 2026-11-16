# Runnable DeepMew EX training

The [V2 retraining execution guide](../docs/v2-bc-league-results.md) documents the separately authorized real semantic-data path, causal history controls, checkpoint identities and local league. It uses the same experiment CLI, shared option scorer, AdamW update function, registry and immutable store described here. Supported S2 configurations are `experiments/deepmew_ex/v2-{snapshot,null,ordered,shuffled}.json`; they require a separately admitted and cached V2 dataset. Unsupported combinations still fail before loading data.

## Preserved S0 baseline

The S0 path adapts PR #1's shared state-option scorer and weighted per-decision BC loop. S0 executes synthetic CPU plumbing only. The original `deepmew_ex.Decision` and `iter_decisions` imports and V1 loader bytes are preserved.

## Run and test

From the repository root with Python 3.11+ (verified on Python 3.12.7, NumPy 2.4.3, PyTorch 2.10.0+cpu):

```powershell
python -m pip install -e ".[test]"
python -m deepmew_ex.cli experiment validate experiments/deepmew_ex/s0.json
python -m deepmew_ex.cli experiment run experiments/deepmew_ex/s0.json --run-id s0-local-001
python -m pytest tests -q --tb=short -p no:cacheprovider --s0-run-path=outputs/deepmew_ex/snapshot_bc_s0_v1/s0-local-001
python -m deepmew_ex.cli experiment report --run-id s0-local-001
python -m deepmew_ex.cli registry rebuild
```

Use a new run ID each time. With no `--s0-run-path`, pytest creates one bounded synthetic training fixture in its unique configured temporary output directory. It never reads real replay data. No release download is required.

The compatibility command `python -m training.train_bc experiments/deepmew_ex/s0.json --run-id s0-local-002` forwards to the same runner. Former real-corpus, `--device auto`, epoch and `runs/ex_bc` output flags are unavailable in this S0 implementation. Existing historical output bytes are not moved.

## Frozen smoke

The config fixes seed 42, CPU/one thread, width 64, 64 raw training decisions, 16 raw validation decisions, 32 optimizer updates and validation every eight updates. Each split includes one forced, zero-weight, multi-select and optional-empty exclusion. This leaves 60/12 eligible decisions. The 32 updates present 32 distinct training rows once; available support and actual consumption are separate ledgers.

The pilot's shared 304-option/144-state network is unchanged. These engineered state features do not create a critic. Each update minimizes actor weight times categorical cross-entropy, followed by clipping at 1.0 and AdamW (lr 0.0003, weight decay 0.01, default betas/epsilon). Evaluation reports both unweighted and weight-normalized loss/top1, with exact sums and denominators. Best selection maximizes unweighted validation top1; ties retain the earliest evaluated update. `final.pt` always means update 32.

## Outputs and registry

New runs live in `outputs/<family>/<variant>/<run_id>/`. Files include resolved config, code/source hashes, model specification, synthetic dataset/split identities, exclusions, actual consumption, metrics, best/final checkpoints, independent best-checkpoint re-evaluation, environment, lineage and a sealed manifest. Files are published atomically under an exclusive publisher lock and cannot be replaced through the store. Completion/failure seals make the entire run write-protected through this API; checksums detect subsequent byte changes. Abrupt process termination leaves an explicitly incomplete attempt; a new attempt requires a fresh ID. A parent-link allocation helper preserves a verified parent and records that optimizer/RNG restoration has not occurred; resumed training is unsupported.

`registry/{families,models,runs}.json` are rebuildable references to run manifests, not new evidence authorities. Local checkpoints remain in gitignored outputs; no remote retention is claimed. Registry metadata commands never deserialize a model, import the native engine or fetch release assets:

```powershell
python -m deepmew_ex.cli registry list
python -m deepmew_ex.cli registry show D1
python -m deepmew_ex.cli registry lineage D1 --format json
python -m deepmew_ex.cli registry import-legacy --dry-run
```

Historical alias lookup returns all source assertions when ambiguous. It never guesses which receipt wins. New run lookup/lineage accepts run, model, checkpoint digest and evaluation IDs. Checkpoint-to-run and run-to-dataset links are in the run manifest and lineage receipt.

## Limits and stop point

S0 admission rejects real V1 training, V2/entity inputs, temporal BC, TD, multi-select objectives, GPU, over-budget runs and non-finite configuration before data loading or run allocation. Empty eligible splits, bad targets and non-finite actor inputs fail explicitly and leave failure receipts after allocation. Holdout has no S0 reader. The S2 path has its own typed admission for the four explicit V2 BC compositions.

The S0 one-factor comparison and deliberately invalid two-factor fixture validate declarations only. Its input-allowlist tests cover synthetic fixed-feature boundaries, not the upstream raw replay privacy contract. See [S0 migration findings](../docs/registry-migration-report.md) and [executed S0 evidence](../docs/s0-baseline-results.md) for that stage's limits. Later V2 training, history and local-game evidence is recorded separately in [the V2 execution guide](../docs/v2-bc-league-results.md). Learned multi-select, critic/PPO, paid compute and promotion remain outside this implementation.
