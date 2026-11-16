# RL league training (PPO from the BC actor)

Owner-approved on 2026-11-15: league RL from the run-e behaviour-cloned actor
(`bc-8m-2ep-20261114-e`, `phase-end-e1-raw-52213-u104426.pt`, sha256
`911fb34b…714a74`). Free local compute only; no promotion, merge or
competition submission.

## Entry points

| What | Command |
| --- | --- |
| Train | `python -m deepmew_ex.rl.cli train --config experiments/deepmew_ex/rl-league-pilot.json --run-id <id>` (the light `cli` entrypoint keeps torch out of spawned lean workers) |
| Resume | add `--resume outputs/rl-runs/<id>/checkpoints/iter-NNNNNN.pt` (same run dir) |
| Stop gracefully | create `outputs/rl-runs/<id>/STOP` (checkpoints, then exits) |
| League-harness eval | `python -m deepmew_ex.rl.cli evaluate --run-id <id> --deck <deck> --participant bc=<bc.pt>@<sha> --participant rl=<iter.pt>` |
| High-volume eval | `python -m deepmew_ex.rl.cli farm-eval --run-id <id> --deck <deck> --games-per-opponent 200 --participant ...` |
| Tests | `python -m pytest tests/rl -q` |

Run outputs: `outputs/rl-runs/<id>/` — `config.json`, `code.json`,
`progress.jsonl` (one row per iteration: farm speed, per-opponent scores,
league shares/win rates, PPO stats, free disk), `games.jsonl` (one row per
game), `checkpoints/iter-*.pt` + `.pt.json` SHA-256 sidecars,
`snapshot-*.pt`, `summary.json` or `failure.json`.

## Design

**Game farm** (`deepmew_ex/rl/farm.py`). The native engine holds one battle
per process, so N spawn worker processes each play one game at a time. A
worker builds the actor-visible full-prefix history (`FullPrefixHistory`,
same as the league adapter), runs frozen outside opponents locally, and steps
the engine with the unchanged `run_real_meta_game`. Each decision of a
learned seat is sent to the main process as only its *new* records (as
`leaf_tables`, computed in the worker) and new events. The main-process
`InferenceServer` answers every pending request of every worker in one GPU
batch.

**Cached inference** (`deepmew_ex/rl/incremental.py`). Records are
append-only and encoded independently, and the context transformer is causal,
norm-first and dropout-free, so nothing already computed ever changes.
`CachePool` keeps record encodings, per-layer keys/values and context rows in
slot-indexed GPU buffers; a batch runs all games' new tokens through each
layer with one padded, masked attention call. `RecordEncoder` caches each
string's GRU vector (weight-version scoped). Tests compare against
`TemporalPolicy.encode` and `encode_padded` on CPU and CUDA, including buffer
growth and slot reuse.

**Action sampling** (`deepmew_ex/rl/sampling.py` and the server). Structured
actions are drawn from exactly the distributions BC trained: pick-one
softmax; unordered subsets with the count head (count drawn from its
marginal, subset by conditional-Poisson sampling over elementary symmetric
sums); ordered picks autoregressively with a stop action. Tests check
log-probabilities equal the BC training NLL (`losses`, `policy_terms`) and the
subset sampler's empirical distribution; `greedy=True` reproduces
`TemporalPolicy.decode`.

**PPO** (`deepmew_ex/rl/ppo.py`, `train.py`). Reward = +1 win / -1 loss / 0
draw at the last decision; GAE(lambda = 0.95, gamma = 1); clipped ratio 0.2;
entropy bonus on exact pick-one entropies; Adam (policy 2e-5, value 3e-4),
grad clip 1, two epochs of 8-game minibatches with a KL early stop (0.03).
Synchronous and on-policy: each iteration plays 192 games with the current
weights, then trains while the workers idle. The value head is a small MLP on
the same causal state; by default its gradient does **not** enter the BC
trunk, and the first three iterations train only the value head. UPGO
(`upgo`) is implemented and tested but off in the pilot; V-trace is not used
because the loop is on-policy (it reduces to clipped TD(lambda) here).

**League** (`deepmew_ex/rl/league.py`). PFSP over frozen outside opponents —
D1 and the four reference notebooks from `league/registry.json`, plus the
downloaded 7th-place agent in its isolated `python -I` subprocess (reset per
game) — with weights `0.15 + (1 - win_rate)^2` from an EMA of our games.
Snapshots of our own policy are added every 10 iterations (max 4) but capped
at 20% of games. **Invariant #2 note:** DeepMew rejects a self-play-to-Nash
training signal, so the league is deliberately anchored on fixed outside
opponents that always carry most of the mass; `RLConfig.validate` rejects a
snapshot share of 0.5 or more.

**Deck.** D1's Starmie list
(`source/.research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv`).
A sampled-BC scan of 336 games per deck against the six outside opponents:
D1's deck 39 wins (12%), the 7th-place Dragapult/Blaziken list 28 wins (8%).
D1's deck gives the stronger start, more win signal for a short pilot and a
direct comparison with the earlier BC league evaluation.

## Measured farm throughput (BC policy, D1 deck, D1 + 4 references)

| Setup | Games/s | Decisions/s |
| --- | --- | --- |
| Single process, full-prefix model (baseline) | ~1.1 | ~45 |
| 4 workers, cached server v1 | 2.5 | 85 |
| 8 workers, slot pool + batched sampling | 9.6 | 333 |
| 10 workers | 10.7 | 391 |

Per our-model decision in the baseline: history 2 ms, model 13 ms (growing
with game length), opponent D1 1.5 ms, engine 1.5 ms. Workers use ~0.52 GB RAM
each; each 7th-place agent subprocess ~1.3 GB, so it runs on 2 workers and is
capped at 10% of training games.

## Second optimization pass (2026-11-15, during the pilot)

Profiling the live pilot: the inference server was busy 91% of each
collection phase at a near-constant ~5-9 ms per batch regardless of batch
size; the torch profiler showed only ~0.5 ms of GPU work per batch (~150
small kernels), i.e. host-bound. Changes, all parity-tested:

- **Exact record cache** (`RecordEncoder`): 20,760 records sent per 80 games
  but only 3,497 distinct; a record encoding is keyed by the bytes of its leaf
  rows in global string ids (exact, no hashing) and reused until the weights
  change. A per-leaf cache was tried and measured slower (missed records are
  large state records; per-leaf dictionary lookups cost more than the matmul)
  and removed.
- **One sync per batch**: host-side sizes instead of `repeat_interleave` /
  `.max()` reads, and index uploads through reused pinned buffers
  (non-blocking).
- **Lean workers**: torch was 450 of a worker's 516 MB. The seat adapters
  moved to the torch-free `deepmew_ex/evaluation/league/seats.py`
  (re-exported from `policies.py` / `temporal.py`) and the worker side to the
  torch-free `deepmew_ex/rl/worker.py`. Only `torch_workers` workers can run
  checkpoint opponents (D1); the rest refuse torch imports (`TorchBlocker`).
  Launch through `python -m deepmew_ex.rl.cli` so spawned workers do not
  re-import a torch-loading `__main__`.

Same contended machine, same 200 games (BC vs D1 + 4 references): 5 full
workers **4.5 games/s** -> 14 workers (3 with torch) **17.7 games/s**; mean
batch 2.2 -> 4.8 requests. The pilot itself ran on the older code.

## Pilot `rl-pilot-20261115-a`

Config `experiments/deepmew_ex/rl-league-pilot.json`; run dir
`outputs/rl-runs/rl-pilot-20261115-a`. Stopped on its 4 h wall budget:
**405 iterations, 77,760 games**, final checkpoint `iter-000405.pt`
(sha256 `6b14f4b4da9aa28bc458ac2e467a67a793d0447bd08924e1c0d238ad5fcf259f`).

Incidents (environmental, both while another user job, `D:\alphajong`
`future_spread.py`, held the GPU at 99% and most free RAM): iteration 161
died on `CUDA illegal memory access` at the same second as three NVIDIA
driver reset events (`nvlddmkm` 153, 19:54-19:55); the first resume died on
`fatal: Memory allocation failure` (host RAM). Resumed from `iter-000160.pt`
with `rl-league-pilot-lowmem.json` (5 workers, checkpoint every 5; learning
settings identical). Records kept as `failure-*.json`.

Training games (sampled policy), score by 100-iteration block:

| Iterations | D1 | Abomasnow | Dragapult | Iono | Lucario | 7th place | Entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-100 | 0.28 | 0.19 | 0.32 | 0.14 | 0.22 | 0.02 | 0.50 |
| 101-200 | 0.37 | 0.24 | 0.41 | 0.18 | 0.29 | 0.02 | 0.24 |
| 201-300 | 0.39 | 0.26 | 0.43 | 0.19 | 0.29 | 0.03 | 0.22 |
| 301-405 | 0.42 | 0.27 | 0.41 | 0.18 | 0.32 | 0.03 | 0.20 |

### Evaluation (argmax, D1's deck, frozen opponents, unpaired engine chance)

League-harness pattern (`deepmew_ex.rl.evaluate`, unchanged full-prefix
`FullPrefixLivePolicy`, 3 seeds x 2 seats x 2 reps = 12 games per opponent;
`outputs/rl-evals/rl-pilot-a-harness-20261115`):

| Opponent | BC W-L-D | RL iter 405 W-L-D |
| --- | --- | --- |
| D1 | 4-8-0 | 8-4-0 |
| ref_abomasnow | 0-12-0 | 4-8-0 |
| ref_dragapult | 5-7-0 | 7-5-0 |
| ref_iono | 1-11-0 | 3-9-0 |
| ref_lucario | 4-8-0 | 2-10-0 |
| 7th place | 0-11-1 | 0-12-0 |
| **Equal-opponent score** | **0.201** [0.076, 0.333] | **0.333** [0.153, 0.528] |
| Crashes / illegal / timeouts | 0 / 0 / 0 | 0 / 0 / 0 |
| Decision latency mean / p99 / max | 25.6 / 39.8 / 716 ms | 21.6 / 40.8 / 116 ms |

High volume (`deepmew_ex.rl.farm_eval`, cached path, 200 games per opponent;
`outputs/rl-evals/rl-pilot-a-farm200-20261115`, `...-farm-ext60-20261115`):

| Opponent | BC wins (Wilson 95%) | RL iter 200 | RL iter 405 |
| --- | --- | --- | --- |
| D1 | 57/200 [0.23, 0.35] | 66/200 [0.27, 0.40] | **92/200 [0.39, 0.53]** |
| ref_abomasnow | 34/200 [0.12, 0.23] | 52/200 | 64/200 [0.26, 0.39] |
| ref_dragapult | 72/200 [0.30, 0.43] | 83/200 | 91/200 [0.39, 0.52] |
| ref_iono | 29/200 [0.10, 0.20] | 27/200 | 44/200 [0.17, 0.28] |
| ref_lucario | 48/200 [0.19, 0.30] | 65/200 | 61/200 [0.25, 0.37] |
| **Score (5 opponents)** | **0.243** | **0.296** | **0.355** |
| 7th place (60 games) | 1/60 | - | 2/60 |

All evaluation games: 0 crashes, 0 illegal, 0 timeouts.

**Reading.** Exploratory, one training seed, unpaired engine chance, no
promotion. League RL from the BC actor improved the greedy policy steadily
(+11 points equal-opponent score over 1,000 games per model; vs D1 on D1's
own deck 29% -> 46%), still rising at the end (no plateau seen). The
7th-place agent remains unbeaten (deck plus its league training and lethal
search). The earlier 0-12 head-to-head script fed the BC history only its own
prompts; these evaluations notify the history of both seats, as the league
runner does.

**Next.** Longer run on the lean farm (about 4x faster); league EMA 0.1 is
noisy, try 0.02; consider a short lethal search at play time (helped the
7th-place team); a second deck / generalist; resolve the GPU/RAM sharing with
other jobs before multi-hour runs.
