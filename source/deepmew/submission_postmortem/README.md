# Kaggle Submission Postmortem Suite

This research-only suite starts from an external Kaggle submission ID, fetches
the complete episode catalog through the official Kaggle SDK, caches every
selected public replay, reconstructs the submitted actor when an exact local
bundle resolves, and writes loss/matchup/action/timeline artifacts.

## Run all episodes

From the repository root:

```powershell
python scripts/analyze_submission_postmortem.py --submission-id 54788723
```

Use already-downloaded replays without making Kaggle replay calls:

```powershell
python scripts/analyze_submission_postmortem.py `
  --submission-id 54788723 `
  --replay-dir .research/kaggle_replays/submission-54788723-all `
  --no-download
```

Validation episodes are included by default. The SDK catalog is authoritative
for submission/team IDs and validation status. CLI and offline catalogs remain
explicitly `metadata_limited`.

## Filters

Filter families combine with AND; repeated values inside a family combine with
OR.

```powershell
# Exact IDs
python scripts/analyze_submission_postmortem.py --submission-id 54788723 `
  --episode-id 86530455 --episode-id 86531083

# Inclusive numeric episode-ID range
python scripts/analyze_submission_postmortem.py --submission-id 54788723 `
  --episode-range 86530455:86534047

# Inclusive 1-based chronological range after (create_time, episode_id) sorting
python scripts/analyze_submission_postmortem.py --submission-id 54788723 `
  --ordinal-range 11:25

# Team substring, exact team/submission IDs, or actor-relative result
python scripts/analyze_submission_postmortem.py --submission-id 54788723 `
  --opponent sukesuke012 --result loss
```

Opponent name filtering over an offline catalog is post-parse and uses replay
team names. Exact opponent team/submission-ID filters require a full SDK
catalog.

## Exact ratings and fail-closed binding

The external submission ID resolves through
`configs/submission_postmortem/submission_bindings.json` or an explicit
`--bundle-dir`. Resolution verifies the submitted deck, NumPy weights, feature
mode/hash, runtime code, and second-order reader. It never guesses a checkpoint
from deck identity or behavioral similarity.

The actor encoder is vectorized across every legal option in a cache miss. The
autoregressive option/STOP heads reuse those embeddings. Complete score shards
are cached by replay SHA-256 plus runtime-binding SHA-256, so filters reuse the
expensive work. Use `--refresh-scores` only when deliberately invalidating
those shards.

For D1, exact deployed policy logits/ranks/probabilities/entropy are available.
The submitted bundle contains no critic, so value and counterfactual blunder
ratings are `not_instrumented`. A policy preference is not a Q-value.

## Outputs

Each run writes:

- `report.md`, `summary.json`, and a hash-bound `manifest.json`
- `episodes.csv/json` and one safe episode -> turn -> decision trajectory JSON
  per replay
- `decisions.csv/jsonl` and full actor-only `decision_options.jsonl`
- matchup shell and exact-deck tables
- loss-signal, first/second, opponent-action, opponent-playstyle, rating, and
  chronological timeline tables
- catalog/cache/parse/privacy receipts plus a phase `checkpoint.json`

The hierarchy is an observed trajectory, not a counterfactual game tree.

## Hash-verified statistical addendum

The addendum CLI extends a completed postmortem run without reparsing replays,
rescoring the actor, or replacing any base-run artifact:

```powershell
python scripts/analyze_submission_postmortem_addendum.py `
  --base-run .research/submission_postmortem_suite/submission-54788723-full-v0 `
  --human-addendum .research/submission_postmortem_suite/ADDENDUM.md
```

Before analysis, the runner requires a `complete` base manifest, a passing base
privacy receipt, and an exact SHA-256 and byte-size match for every output bound
by the base manifest. It then reads only the already-sanitized `episodes.json`
and `decisions.jsonl`. A mismatch, missing artifact, incomplete manifest, or
failed privacy receipt stops the run.

By default, machine-readable and rendered outputs go under
`BASE_RUN/addendum-v0/`. Use `--out-dir` to choose another additive output
directory. `--human-addendum` optionally writes a second copy of the rendered
Markdown; it should point to a new addendum path, never the authoritative base
`report.md` or the human `REPORT.md`. This convenience copy is outside the
addendum manifest; the hash-bound Markdown is `addendum-v0/ADDENDUM.md`. The
deterministic defaults are random seed `54788723` and `40,000` trend
permutations; override them with `--random-seed` and `--permutations` when
deliberately running a sensitivity check.

The existing base report, summary, tables, trajectories, privacy receipt,
checkpoint, and manifest remain byte-for-byte untouched. Rerunning the
addendum may replace files inside its own output directory, but never mutates
the completed base artifacts.

### Addendum phases and outputs

| Phase | Analysis | Primary output |
|---|---|---|
| A | Win-rate intervals, sampling variance, posterior threshold probabilities, and sampled-mixture Elo uncertainty | `rating_stability.json` |
| B | Cumulative and rolling 5/10/20-game performance plus selection-adjusted trend tests | `rolling_performance.json`, `rolling_performance.csv` |
| C | Exposure-normalized matchup losses, intervals, loss ratios, and Bayesian shrinkage | `matchup_normalization.json`, `matchup_normalized.csv` |
| D | First/second loss odds ratio, relative risk, intervals, and descriptive standardization | `initiative_effect.json` |
| E | Per-loss observable and sustained-disadvantage timing proxies | `loss_timeline_summary.json`, `loss_timeline.csv` |
| F | Action-confidence and entropy histograms plus outcome, matchup, and game-phase associations | `confidence_diagnostics.json`, `confidence_*.csv`, `entropy_histogram.csv` |
| G | Regularized retrospective loss-association model with uncertainty and held-out predictions | `observational_loss_model.json`, `loss_model_*.csv` |
| H | Exploratory repeated-failure clusters, stability, membership, and representative episodes | `failure_clustering.json`, `failure_cluster*.csv` |
| I | Replay-cited engineering tasks ranked by an evidence-priority proxy | `engineering_priority.json`, `engineering_priorities.csv` |
| J | Expected uncertainty at 500/1,000/5,000/10,000 evaluation games | `data_value.json`, `data_value.csv` |
| K | Research decision memo separating known, suspected, and unidentifiable claims | `decision_memo.json`, `decision_memo.md` |

`statistical_addendum.json` contains all phases in one schema-bound payload.
`ADDENDUM.md` renders the complete addendum. The addendum's own `manifest.json`
binds the base manifest/report/episode/decision hashes, analysis parameters,
all generated outputs inside the addendum directory, and its passing
`privacy_sweep.json`.

### Statistical and identifiability boundaries

- The inferential unit is one opponent-attributable decisive game. Exact
  self-mirrors and draws are excluded; actor decisions are not treated as
  independent samples.
- Offline runs without authoritative creation timestamps use the stored
  ordinal/numeric episode-ID order as an `episode_id_order_proxy`. Rolling and
  change-point findings are not claimed as timestamp-verified ladder history.
- Opponent decks remain postgame matchup metadata. Matchup, initiative,
  trajectory, regression, and clustering results are observational and do not
  establish causal effects.
- Elo output is a performance differential against the sampled opponent mix,
  not an absolute ladder rating. Threshold probabilities use the stated
  Bayesian prior and are not frequentist guarantees.
- Policy chosen-action probability measures action-selection concentration,
  not probability of winning or action correctness. Without an outcome target,
  global probability calibration remains `not_identifiable`.
- A realized replay can identify an observable or sustained disadvantage
  proxy, but not the first irreversible or effectively lost state. Those fields
  remain `not_identifiable` without valid counterfactual value or branching.
- The observational loss model must not be read as a prospective policy model;
  full-trajectory variables can occur after the state they help describe.
- Expected realized Elo gain per engineering week is not identifiable from
  observational replays. Engineering ordering is explicitly an evidence and
  implementation-cost priority proxy.
- Extra-game projections quantify evaluation uncertainty reduction. They do
  not numerically decide data collection versus policy iteration without
  policy-gain and cost priors.

## Information boundary

- Policy rescoring uses only the submitted actor seat's projected
  `actor_visible` pre-action observation.
- Raw `search_begin_input`, logs, looking data, deck order, and raw observations
  are neither cached nor persisted.
- Opponent rows retain realized public actions and public turn context only;
  opponent private state and unselected legal alternatives are discarded.
- Postgame decklists are matchup metadata only and never enter actor rescoring.
- CABT module `1.32.0` uses the bounded previous-step request -> current-step
  action rule and is labelled `structurally_verified_unadmitted`, not canonical.

## Research capability and information-value audit

Phase L is maintained separately from the A-K statistical computation:

- `.research/submission_postmortem_suite/RESEARCH_CAPABILITY_AUDIT.md` contains
  the unknown inventory, instrumentation catalog, information-value ordering,
  dependency DAG, maturity model, readiness gates, and research milestones.
- `.research/capability_debt/capability_debt_ledger.json` is the living source
  of truth.
- `.research/capability_debt/CAPABILITY_DEBT_LEDGER.md` and
  `capability_debt_ledger.csv` are deterministic generated views.
- `.research/capability_debt/README.md` documents evidence-bearing status
  transitions and the `RELIABLE` gate.

Validate or regenerate those views without rerunning replay analysis:

```powershell
python scripts\manage_capability_debt_ledger.py validate
python scripts\manage_capability_debt_ledger.py render
```

The ledger is capability metadata only. It contains no replay observations,
hidden zones, private engine state, model tensors, or Elo-gain estimates.
