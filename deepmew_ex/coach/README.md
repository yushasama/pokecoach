# DeepMew Coach (CECS 451 Team 6 MVP)

Build a Pokémon TCG deck, then practice against a random AI opponent while a
coach highlights the strongest move and explains why. You are free to ignore it.

## Run it

Follow the [PokéCoach installation and launch instructions](../../README.md#run-it).

Open http://127.0.0.1:8765. Our models run on CPU (a training run can keep the GPU).

## Flow

1. **Deck Builder** — search/filter the game's 1,267-card pool (real card art), click to add, +/− to adjust.
   Rules are enforced like the real game: exactly 60 cards, max 4 copies of a card (Basic Energy unlimited),
   at least 1 Basic Pokémon, at most 1 ACE SPEC. Start from a sample deck if you like, name it, **Save**.
2. **Play** — pick your saved deck → **Find opponent**. You face a random AI from our models, each piloting a deck it knows.
3. **Coach** — glows gold on the recommended card/spot, lists *why* (damage & KOs incl. weakness/resistance,
   prize math, energy still needed for each attack, evolution HP/attacks, what the trainer does), and how it
   rates the alternatives. "Play coach's move" does it for you. Turn hints off for real practice.

## Controls (same flow as digital TCG clients)

- **Play a card:** click a hand card, then where it goes — or drag it. Pokémon → Bench, Energy/Tools → a Pokémon,
  evolutions → the Pokémon they evolve from, Trainers → the middle play zone.
- **Attack / Retreat / Ability:** click your Active Pokémon → attack panel (energy cost, damage, text).
- **End Turn:** button on the right of the mat. **Searches, discards, yes/no:** a card picker pops up.
- **Inspect:** hover any card to enlarge it; click a discard pile to browse it.

## Who's who

| Role | Model |
|---|---|
| Coach | **7th-place league agent** (Z3rox-dev, MIT, deck-agnostic transformer + lethal search) — the strongest general agent available; falls back to our BC model if not downloaded |
| Opponents (random) | our **DeepMew BC model** (any sample deck), **BabyMew rules** (its deck), the **7th-place agent** (Dragapult/Blaziken) |

The 7th-place agent runs isolated in its own `python -I` subprocess (`external_worker.py`).
It is vendored in `third_party/z3rox_7th/kaggle_submission` (MIT, unchanged); a local download at `outputs/external_agents/z3rox_7th_1dde9eca/kaggle_submission` takes precedence.

## Files

| File | What it does |
|---|---|
| `server.py` | stdlib HTTP server + JSON API (`/api/meta`, `/api/cards`, `/api/deck`, `/api/decks/*`, `/api/new`, `/api/move`, `/api/coach-move`) |
| `session.py` | One game on the real competition engine (`cg/`): the AI plays until it's your turn |
| `brains.py` | Opponents and coaches: BC model, BabyMew rules, greedy, external Kaggle agents |
| `explain.py` | Plain-language reasons for a move, computed from engine data |
| `decks.py` | Card pool catalog, deck rules, saved decks (`outputs/coach_decks/*.json`) |
| `labels.py` | Readable text for cards, moves, board and log |
| `static/index.html` | The whole UI (vanilla HTML/JS, no build step) |
| `static/card_images.json` | Card ID → set/number from the Kaggle `EN_Card_Data.csv`; art from the public Limitless TCG CDN (needs internet) |

Tests: `python -m pytest tests/coach -q`.

## Easy things to add (pick one!)

- **LLM explanations** — swap/extend `explain.explain()`; it already gets the full observation and the chosen move.
- **Difficulty levels** — sample opponents' moves with a temperature instead of always the top pick.
- **Deck import/export** — paste a deck list (e.g. PTCG Live format) and map names → card IDs in `decks.py`.
- **Match history & stats** — append `session.view()` results to a file; show win rate per deck.
- **More opponents** — any Kaggle submission folder works as an `ExternalBrain`.
- **Animations / sounds** — `static/index.html` (`render`, `monCard`).

## Known limits

- One game at a time (the engine is process-global).
- Damage in explanations is an estimate (weakness ×2, resistance −30); card effects can change it.
- Coach percentages are the model's preference among legal moves, not win probabilities.
