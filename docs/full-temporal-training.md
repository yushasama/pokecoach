# Full historical training execution

**November 12 handoff:** the new full-scale prepared temporal corpus is complete, with 14,643,384 admitted training decisions. The owner transferred execution to Claude before any 8.09M training update. Read [the current handoff](full-scale-temporal-handoff.md); the commands and configuration below belong to the preserved old baseline and must not launch the new run.

**Current status, 2026-11-08:** The execution description below is historical. Later owner instructions prohibited legacy snapshot training; the raw temporal phase completed and the boundary stop preserved it before any snapshot consumption. Training remains stopped and the original run unsealed. The independent [frozen-D1 gameplay evaluation](temporal-d1-evaluation.md) is now complete: 0 wins, 12 losses, 0 draws for the temporal checkpoint, with all 24 candidate/control games safety-clean. This supersedes the historical statement below that gameplay is unmeasured and does not authorize resuming either training phase.

The user authorized training on 2026-10-26, superseding the earlier review-before-training hold. Implementation starts from PR #9 (`1992c450d5cc95c196cd7fd2b7249dd68c3ac081`) and preserves its completed four-model and 600-game evidence. This run does not repeat the 16,384-row diagnostic.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -B -u -m deepmew_ex.experiments.v2.full --config experiments/deepmew_ex/full-temporal.json --run-id full-temporal-20261026-v3
```

The runner indexes the complete frozen 9,300,876-reference corpus, reconciles raw/canonical native-game aliases, resolves canonical reserved populations and builds their semantic-signature guard **before any optimizer update**. It then trains one epoch over every eligible raw actor sequence and every resolvable eligible snapshot, retaining the historical split exclusions. There is no decision/sample cap and the last snapshot batch is retained.

The actor uses complete same-actor request/action prefixes, two causal attention layers, width128, four heads and UTF-8 path/value encodings with no vocabulary/OOV truncation. Exact unordered subset likelihood includes empty choices; ordered selections use a masked autoregressive likelihood. Every recorded request remains context, including forced prompts. Targets and actions become history only after their own prediction.

Recorded post-action states enter model context only when also observable at an actor request. Other post-state identities and original immutable episode bytes remain available for auditing; inactive-seat-only observations are not silently exposed. Snapshot-only sources have explicit unknown-history/reset inputs. These sources do not become temporal data through file order. Learned encodings are compressed representations; preservation of source inputs is not proof that the model learns to use every distinction.

The run uses the local RTX4060, at most60% of its memory, two CPU threads, one epoch,64GiB output allowance and10GiB free-disk reserve. After two zero-update preparation corrections, the remaining wall budget is23h, within the original24h envelope. No paid compute, cloud submission or promotion is authorized by this run. A resource stop preserves the last committed checkpoint and reports incomplete coverage.

Outputs live under the absolute NVMe `output_root` in the typed experiment configuration, followed by `deepmew_ex/v2_full_temporal_v1/<run-id>/`: configuration, exact source/parser identities, parameter counts, progress, checkpoint/hash pairs, SQLite eligible/consumed/context ledgers and validation. The original D: source corpus remains read-only. Final validation rereads both temporal and snapshot populations against the exact final checkpoint; phase metrics are separately hash-bound. Native gameplay strength and the benefit of memory remain unmeasured.

Recovery uses a fresh attempt ID and an immutable checkpoint from a terminated parent:

```powershell
.venv/Scripts/python.exe -B -u -m deepmew_ex.experiments.v2.full --config experiments/deepmew_ex/full-temporal.json --run-id <NEW_ID> --resume <ABSOLUTE_CHECKPOINT_PATH>
```

Resume checks code/config/corpus/parser identity, restores model/optimizer/RNG and committed cursor/coverage, and retains cumulative elapsed time. A mid-update interruption is never published as a resumable checkpoint. Existing runs are not overwritten.

The frozen port is the execution universe. Quarantined raw sources, old partial archive shards and unresolved prior-request alignments are recorded gaps, not invented valid examples. A completed run establishes consumption of the resolved eligible universe, not that every historical archive has become valid, that temporal memory is useful, or that gameplay has improved.

Validation includes exact subset normalization, ordered-action normalization, future-action masking, unseen-value distinction, optional/multiselect gradients, GPU updates, partial batches, reserved-signature inclusion even for historically masked rows, parser drift rejection, complete fixture consumption, final-checkpoint validation and bit-identical CPU recovery after an injected post-optimizer failure. Bit-identical CUDA recovery has not been established.
