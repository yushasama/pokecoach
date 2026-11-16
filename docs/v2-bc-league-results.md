# Executed V2 behavior cloning and local league

**Historical October 25 execution report.** Later work trained/evaluated a full-prefix baseline and published the enlarged temporal corpus. For current capabilities and live PR status, read [the November 12 system handoff](system-handoff.md); PRs #6–#8 are now merged, while #9 remains open. The original results and consumed identities below remain unchanged.

This stage connects admitted semantic replay data to the existing DeepMew EX experiment, scorer, optimizer, registry and immutable output store. It trains four real-data conditions and evaluates their selected checkpoints in the preserved local arena. PRs #6, #7 and #8 remain open and unmerged; this implementation stacks on #8.

## Reproduce the executed path

Use the existing local archive and the full-port artifact described in [the migration guide](v2-full-corpus-port.md). Python 3.12.7, NumPy 2.4.3 and PyTorch 2.10.0+cpu were used. No asset download or paid compute was needed. Commands are from the repository root. Every repeated run needs a fresh ID and matching cache paths in the four configs.

```powershell
python -m deepmew_ex.cli dataset admit-v2 --corpus outputs/deepmew_ex/v2_corpus/v2-full-20261024 --run-id v2-admission-20261025
python -m deepmew_ex.cli dataset cache-v2 --admission outputs/deepmew_ex/v2_admission/v2-admission-20261025 --run-id v2-cache-20261025
foreach ($treatment in @('snapshot', 'null', 'ordered', 'shuffled')) {
    python -m deepmew_ex.cli experiment run "experiments/deepmew_ex/v2-$treatment.json" --run-id "v2-$treatment-20261025"
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $treatment" }
}
python -m deepmew_ex.cli league canary --run-id v2-d1-canary-20261025
python -m deepmew_ex.cli league canary --run-id v2-reference-canary-20261025 --opponent d1 --opponent ref_abomasnow --opponent ref_lucario --opponent ref_iono --opponent ref_dragapult
python -m deepmew_ex.cli league evaluate --run-id v2-league-corrected-20261025 --run outputs/deepmew_ex/v2_bc_snapshot_v1/v2-snapshot-20261025 --run outputs/deepmew_ex/v2_bc_null_v1/v2-null-20261025 --run outputs/deepmew_ex/v2_bc_ordered_v1/v2-ordered-20261025 --run outputs/deepmew_ex/v2_bc_shuffled_v1/v2-shuffled-20261025
python -m pytest tests -q --tb=short --durations=5 -p no:cacheprovider --s0-run-path=outputs/deepmew_ex/snapshot_bc_s0_v1/s0-integration-20261024 --junitxml=outputs/test-results/v2-bc-reference-fix-tests.xml
```

Admission and caching happened once, before training. The canary happened before the full league. Training was CPU/one thread, seed42, width64, embedding24, batch32, AdamW lr0.0003/weight decay0.01, 1,024 updates and 32,768 presentations (two epochs). Validation ran initially and every 256 updates. Selection minimizes unweighted validation CE among trained checkpoints, retaining the earliest tie; the initial model is reported separately. Best and final checkpoints are distinct artifacts.

## What was consumed

The full port retains 9,300,876 canonical semantic references; these are not all admitted temporal examples. Fresh admission audited the complete supported causal raw pool: 3,251 episodes and 439,349 requests. Game identity, game content, effective splits and current semantic signatures are checked across reserved splits. Historical target masks are preserved evidence; they are not reused as a new training mask. Membership ranking uses seed, native game, actor and request coordinate, independently of target/outcome.

| Partition | Raw requests | Forced | Multi-select | Optional-empty-capable | Cross-split signature | Eligible | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 280,231 | 19,269 | 7,653 | 38,009 | 2,081 | 213,219 | 16,384 |
| Validation | 59,757 | 4,124 | 1,752 | 8,102 | 435 | 45,344 | 2,048 |

The optional exclusion means the prompt permits an empty selection, not that every demonstration chose empty. Training uses 2,041 games and validation 422. Test, holdout, calibration and golden-validation contribute only conservative admission/deduplication checks; they are not fitted, used for vocabulary construction, scored or used for checkpoint selection.

The compressed cache contains 2,463 per-game NPZs (46,131,115 bytes). Its train-owned vocabulary has 2,628 path tokens and 675 value tokens including PAD/UNK. Training had zero unknown leaves; validation had 52 unknown paths and 794 unknown values among 7,742,662 leaves. Missing/changed source bytes, empty eligible sets and unsupported data/objectives fail explicitly.

## Temporal model semantics

Current actor inputs contain visible state, request context and that request's conservative public log snapshot. They exclude the current chosen action and outcomes. Semantic path/value/numeric leaves replace the fixed state projection; card and zone relationships remain in their paths. Raw semantics remain preserved independently in the parent corpus. Card text/rules catalog enrichment is not implemented.

`snapshot` pools the current semantic observation and uses the reused shared per-option scorer. `null`, `ordered` and `shuffled` use the same recurrent architecture and identical initialization. Ordered history is up to eight completed requests by the same actor, including visible state, past action and public logs observed on that actor's own request. Shuffled history permutes those causal chunks within the window. Null history is zeroed. Forced/multi/optional prompts can supply history but bear no categorical training loss.

Every prediction starts GRU h0 at zero and backpropagates through the finite history window. The live policy rebuilds the same actor/game-local window and clears it each game/seat. This is request-by-request context, not every internal simulator tick or full-game BPTT. Live/offline logits match in tests for all four conditions.

Snapshot has 111,913 parameters; recurrent controls have 140,969. The exact D1 runtime model has 1,220,367. These models are roughly 9â€“11 times smaller than D1 and use a bounded training subset. D1 also has different training, engineered priors and deck exposure. D1 comparisons measure practical policy performance; they cannot isolate representation quality or establish a capacity ceiling.

## Local arena contract

The fresh D1 anchor and each of four selected candidates play the frozen ten-member `latest_training_population`: d1, h2, e1, r7, x2, ref_abomasnow, ref_lucario, ref_iono, ref_dragapult and hardtail_r12. This is not all 34 historical P0 registry members. All candidates use the frozen D1 deck. Model/deck hashes are verified before loading. Reference teachers execute their actual notebooks with exact frozen deck bytes in scoped run directories. Under the common conservative projection, they use the shared preserved greedy fallback on DECK/LOOKING option requests. These four opponents are restricted-input notebook/greedy hybrids, not reproductions of unrestricted notebook strength. Unexpected notebook errors stop the run; silent proxy substitution is rejected. All reference routes are recorded.

The frozen schedule uses policy seeds 40009/50021/60013, both seats and two repetitions: 12 games per opponent/participant, 600 total. Policy RNG streams are isolated, but native engine chance is unpaired: the consumed BattleStart API exposes no engine-seed argument. The engine and imported source hashes are recorded. Every game receives new policy buffers/teacher objects. No bit-reproducible-game claim is made.

Forced, optional, compound and over-budget live requests use the declared preserved greedy fallback. Reports distinguish model choices from fallback routes, and a fallback-only candidate is rejected. The runner records latency and stops on the first illegal action, crash, timeout or resource limit. Limits are 2,400 seconds overall, 4,000 selections/game and 5 seconds/decision. Scores use draw=.5, equal opponent weights, per-opponent Wilson win intervals and opponent-cluster bootstrap aggregate intervals. One training seed and unpaired engine chance make this exploratory, not E11/E23 confirmation or a promotion decision.

## Migration and modularity

The existing package owns data admission/caching, semantic models, shared BC optimization, experiment execution, registry/lineage and league adapters in separate domain modules. V1 imports/loader semantics and the synthetic S0 path remain intact. There is no second framework, critic, PPO, TD, search, learned multi-select, unused family or champion update.

Model registry rebuilding now examines manifest kind before verifying payloads, avoiding repeated multi-GB reads of unrelated dataset runs; relevant model runs still undergo hash verification. Frozen reference notebooks expect a working-directory `deck.csv`; scoped per-opponent runtime copies resolve this without editing source notebooks or a global deck. They also require explicit `select.deck`/`current.looking` dataclass keys. Supplying `None` preserves the projection; requests that require the excluded buffers use the declared shared fallback. Exact original historical checkpoints were available locally, even though optional portable copies were absent; no download was needed.

The first league attempt, `v2-league-20261025`, was stopped and sealed as failed after 427 games when its counters exposed 8,292 reference exception fallbacks. Every reference call in that attempt had fallen back because of missing decoder keys. Its game safety counters were zero, but its scores are invalid and excluded. A bounded follow-up probe established that merely inserting null keys still fails on deck-search requests. [The append-only reference amendment](../experiments/deepmew_ex/v2-league-reference-amendment.json) declares the corrected restricted-input routing before a fresh canary/full schedule; it changes neither training nor score selection.

New runs stay under `outputs/<family>/<variant>/<run_id>/`, with exclusive allocation, sealed manifests, source/config/data identities, counts, metrics, checkpoint hashes and failures. The store rejects replacement and verifies subsequent bytes; it is not an OS permission boundary. Checkpoint/NPZ bytes remain local and gitignored. Reproduction requires the recorded archive and native Windows engine. Original source, historical model/league/D1 assets, pilot-v1 and postmortems were not modified. Eighteen old partial R2 shards and unsupported raw contracts remain quarantined.

## Executed training results

All four runs used clean training revision 61db330d79eb30a5f8cdc8b52fdbc2303110660e. Each completed 1,024 finite optimizer updates, changed parameter tensors, and independently reloaded/rescored its selected checkpoint with exact parity. Ordered and shuffled treatments each consumed 250,226 prior-request presentations (102,102 unique context records).

| Condition | Initial CE | Selected CE | Selected top1 | Update | Seconds |
|---|---:|---:|---:|---:|---:|
| snapshot | 1.769182 | 1.460177 | 44.09% | 512 | 72.79 |
| null | 1.750744 | 1.458140 | 44.82% | 1024 | 73.74 |
| ordered | 1.750778 | 1.458364 | 44.38% | 1024 | 243.90 |
| shuffled | 1.750784 | 1.458111 | 44.63% | 1024 | 243.55 |

All conditions learned relative to initialization. Their selected losses are close; the ordered-history condition does not beat its matched zero-history control on validation CE. This run does not demonstrate an imitation benefit from ordered history. Checkpoint selection used validation, so these are selected validation diagnostics rather than an untouched final test estimate.

| Selected checkpoint | SHA-256 |
|---|---|
| snapshot | 143b9d93e3365ef1b9128733721f43509ba99be341e5bb127deaa623223fb180 |
| null | 12fdb7fe1ddc8ed6d4c9704e263b4a1af04bd441d414f3ff0961b4ebdacc8044 |
| ordered | 6f84e7a7d070aa661e87983b508da98ecf0ff4704c58c7c54fe6144b319c2e18 |
| shuffled | 93e996d61070620ce9e2ec051ca451454c8718df79c2a7ac1dc301e8e1bf01aa |

## Executed league and verification

The corrected schedule completed **600/600 games in 175.82 seconds** at clean evaluator revision `9a8655837570ea05b0c7c9519809fff46fbd2a87`: 120 games per participant, 12 against each opponent. Illegal actions, crashes, timeouts and unexpected notebook errors were all **zero**. The preceding reference/D1 canary completed ten games, both seats for each reference and D1, with 561 actual notebook choices and 54 declared restricted-zone fallback choices; no notebook errors.

| Participant | Equal-opponent score | Opponent-cluster bootstrap 95% interval |
|---|---:|---:|
| D1 anchor | 40.42% | 30.00–50.00% |
| Snapshot | 22.50% | 16.67–30.00% |
| Zero history | 22.50% | 15.83–30.00% |
| Ordered history | 20.00% | 12.50–27.50% |
| Shuffled history | 27.50% | 17.92–37.08% |

These are whole-policy scores including the declared fallback. Every new policy reached thousands of learned choices:

| Condition | Model choices | Forced fallback | Multi-select fallback | Optional fallback |
|---|---:|---:|---:|---:|
| Snapshot | 3,140 | 387 | 172 | 735 |
| Zero history | 3,309 | 405 | 174 | 796 |
| Ordered history | 3,418 | 408 | 186 | 856 |
| Shuffled history | 3,285 | 415 | 198 | 839 |

No candidate used over-budget semantic fallback in this schedule. Across the four restricted-input references, there were 14,581 actual notebook choices and 1,529 explicitly routed restricted-zone fallbacks, with zero exception fallbacks. Maximum observed candidate decision latency was 150.93 ms; D1 anchor maximum was 8.36 ms. These maxima are from this schedule, not a latency guarantee. Per-opponent wins/losses/draws, Wilson intervals, route counts and exact assets are in [the league receipt](receipts/v2-league-corrected-20261025.json).

The ordered model does not improve either selected validation CE or the exploratory league point estimate over its matched null control. Shuffled history has the highest new-model league point estimate, with overlapping exploratory intervals. This is not evidence that shuffling improves strategy. All four trail D1's point estimate. The result establishes a working learned path and a negative initial history diagnostic under this exact budget; it does not reject temporal modeling, V2 semantics or larger/data-richer models. No checkpoint was promoted and no additional tuning was performed.

**134 tests passed in 40.37 seconds**, zero failures/errors/skips. The final source tree includes the native decoder/fallback corrections; JUnit SHA-256 is `14cc9b7fdd74d635c41433327afa838bfeecbb1ab1feac6f1e28615696c4805c`. Coverage includes finite gradients and real updates, option-permutation equivariance, batched pilot-scorer parity, exact checkpoint save/load, private/future/cross-seat isolation, train-only vocabulary, empty support, unsupported objectives before I/O, sealed output immutability, valid comparison and changed-optimizer rejection, isolated RNGs, live/offline logits for all treatments, scoped reference decks, explicit missing native fields, and rejection of silent notebook proxies. Tests use synthetic fixtures and the saved S0 run; the four real training runs and native games are separate executed evidence.

All completed training/cache/admission/canary/league manifests and the failed initial league manifest were independently hash-verified. The ten historical checkpoint/deck bindings were rechecked after the league. Protected paths have no diff from the consumed PR #8 head. The complete [delivery receipt](receipts/v2-bc-delivery-20261025.json) binds data, consumption, metrics, exact source maps, best/final hashes, the excluded invalid attempt and changed paths. Full per-model lineage receipts are [snapshot](receipts/v2-snapshot-20261025.json), [null](receipts/v2-null-20261025.json), [ordered](receipts/v2-ordered-20261025.json) and [shuffled](receipts/v2-shuffled-20261025.json).

## Exact inputs and implementation ownership

PR heads were refreshed and matched the revisions consumed. None was merged by this work:

| PR | Consumed head |
|---|---|
| #6 | `e9f5ae2e706b3cd638783750d3cc11bb0f7fd849` |
| #7 | `661c1d0ffcae87123999238b91297de224cc9759` |
| #8, implementation base | `2888f897d6a1c2e7166b8556b03e494975384488` |

Admission executed at clean `f104825e96b2dfca7d0648a74ea67aea0252e196`; caching and the initial D1 canary at clean `866e0eb787dcb84609bcd8e6ea373c31f2db4f35`; training at clean `61db330d79eb30a5f8cdc8b52fdbc2303110660e`; corrected canary/league at clean `9a8655837570ea05b0c7c9519809fff46fbd2a87`. Original recorded identities are retained rather than relabeled as the final documentation commit.

- `deepmew_ex/datasets/v2/learning/`: objective-specific admission, train-owned semantic vocabulary and ragged game caches.
- `deepmew_ex/models/semantic.py`, `training/semantic.py`: semantic/history encoders and BC batching; existing `models/mlp.py` and `training/bc.py` retain the reused scorer/update owner.
- `deepmew_ex/experiments/v2/`: typed S2 composition, checkpoint execution and comparison validation; existing CLI/config/output/registry remain shared.
- `deepmew_ex/evaluation/league/`: exact historical bindings, live actor history, counted fallbacks, independent seats and native evaluation.
- `experiments/deepmew_ex/`, mirrored domain tests, and `docs/receipts/`: frozen configuration, validation and reviewable evidence.

Implementation stops at these working BC/history baselines and their local exploratory evaluation. Training the remaining corpus, salvaging quarantined data, adding card/rules metadata, increasing capacity, extra seeds, new objectives and qualification are separate follow-ups.
