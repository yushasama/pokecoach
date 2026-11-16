# PokéCoach

Build a Pokémon TCG deck, play it against AI opponents on the real competition
engine, and get a coach that highlights the strongest move and explains why.
CECS 451 Team 6 term project, built on the DeepMew EX agent work.
Everything we did not make ourselves is credited in [CREDITS.md](CREDITS.md).

## Run it

1. Python 3.12 with PyTorch (CPU is enough for the app), NumPy and orjson.
2. Copy the Kaggle competition engine (`cg` package) into `source/cg/`. It is
   not redistributable, so it is not in this repo; see
   [source/cg/README.md](source/cg/README.md).
3. Start the app:

```
python -m deepmew_ex.coach.server --port 8765
```

Open http://127.0.0.1:8765. Full UI guide: [deepmew_ex/coach/README.md](deepmew_ex/coach/README.md).
Tests: `python -m pytest tests/coach tests/rl tests/models -q`.

## What is in this repo

| Path | What |
| --- | --- |
| `deepmew_ex/coach/` | The app: stdlib HTTP server, game session on the engine, brains, explanations, deck builder UI |
| `deepmew_ex/` (rest) | DeepMew EX: actor-visible history, byte-level temporal transformer, BC training stack |
| `deepmew_ex/rl/` | League RL: cached-inference game farm, PPO, PFSP league, evaluation |
| `source/cg/` | Where you put the competition game engine (CABT); not included |
| `source/deepmew/`, `source/scripts/` | BabyMew rules agent, arena runner and the legacy loaders the league uses |
| `league/` | Frozen league registry and deck lists (the Kaggle sample opponent notebooks are not included) |
| `models/coach/` | Our BC model, weights only |
| `third_party/z3rox_7th/` | The 7th-place Kaggle agent (MIT, vendored unchanged) |
| `docs/` | Training write-ups: BC, BC league results, RL league |

## The models

| Role in PokéCoach | Model |
| --- | --- |
| Coach (move ranking + lethal check) | 7th-place league agent by Z3rox-dev |
| Random opponents | our DeepMew BC model (any sample deck), BabyMew rules, the 7th-place agent |

### Our BC model (`models/coach/bc-8m-run-e`)

An 8,090,949-parameter policy that reads the whole game so far, the way a
player does:

1. **Inputs.** Every decision is turned into what that seat could see: the
   board, its hand, the request and its legal options, the public game log,
   and its own past actions. Each item becomes a list of `(path, value,
   number)` leaves.
2. **Encoder.** A GRU reads the UTF-8 bytes of every path and value string, so
   there is no vocabulary and unseen cards never collapse to "unknown". Leaf
   vectors are mean- and max-pooled into one vector per record.
3. **Memory.** A 2-layer causal transformer (width 464, 8 heads) runs over the
   game's records in order, so each decision sees the full history and nothing
   from the future.
4. **Actions.** A state-conditioned scorer rates each legal option. Exact heads
   cover single picks, unordered multi-picks (with a learned count) and ordered
   picks with a stop action.
5. **Training.** Behaviour cloning on 14,643,384 recorded decisions from
   104,482 historical games, 2 epochs (AdamW, lr 3e-4), on one RTX 4060 in
   17.9 hours. A bitmap ledger proves every decision was used exactly once per
   epoch. Validation NLL went from 0.493 after epoch 1 to 0.360 after epoch 2.

Write-up: [docs/full-temporal-training.md](docs/full-temporal-training.md),
league results: [docs/v2-bc-league-results.md](docs/v2-bc-league-results.md).

### League RL from the BC model (code included, checkpoint not uploaded)

PPO fine-tuning against a league of frozen outside opponents (DeepMew's D1,
four rule-based reference decks, the 7th-place agent) plus capped snapshots
of itself, sampled by how often we lose to each. Reward is win/loss; GAE
λ = 0.95, clip 0.2. Games run in parallel worker processes against one
batched GPU server that caches each game's encoder and attention state, so a
decision only pays for what is new.

A 4-hour pilot (77,760 games) raised the greedy policy's score against D1 and
the reference decks from 0.243 to 0.355 over 1,000 games per model, and
against D1 on D1's own deck from 29% to 46%. The 7th-place agent still wins
almost every game (BC 1/60, RL 2/60). Details:
[docs/rl-league-training.md](docs/rl-league-training.md).

### BabyMew rules

The hand-written DeepMew v2.4 strategy, in `source/deepmew/policy/`.

### 7th-place agent (`third_party/z3rox_7th/`)

[Z3rox-dev/pokemon-tcg-7th-place-transformer-agent](https://github.com/Z3rox-dev/pokemon-tcg-7th-place-transformer-agent)
at commit `1dde9eca`, MIT license (`UPSTREAM_LICENSE`). A league-trained
transformer with lethal search at play time, piloting its own
Dragapult/Blaziken deck. It runs in an isolated `python -I` subprocess and is
reset at the start of every game. We did not train or modify it.

## What we uploaded (and what we did not)

We only uploaded the models PokéCoach uses: our BC model (weights only; the
optimizer and RNG state were stripped from the 97 MB training checkpoint) and
the vendored 7th-place agent. Not uploaded: the training corpus, D1 and the
other league checkpoints, the RL checkpoints and run outputs. The training and
RL code is here, but rerunning it needs those assets, which live in a separate
private data repo.

Also not included, because we may not redistribute them: the competition
game engine and the four Kaggle sample rule-based opponent notebooks. Both are
credited, with download instructions, in [CREDITS.md](CREDITS.md).

Pokémon and all card names and artwork are trademarks and © of Nintendo,
Creatures Inc., GAME FREAK inc. and The Pokémon Company. This is an
unaffiliated, non-commercial student project.
