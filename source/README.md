# DeepMew

Runnable submission for the Pokemon TCG AI Battle competition.

This repository root intentionally mirrors `data/pokemon-tcg-ai-battle/sample_submission`:

- `main.py`
- `deck.csv`
- `cg/`

V2 adds a small anti-blunder heuristic policy. It is still generic and option-category based: no search, belief sampling, opponent modeling, trained models, or `cg/` changes.

## Local Self-Play

Run one game:

```powershell
python scripts\self_play.py --games 1
```

Run the crash check:

```powershell
python scripts\self_play.py --games 50
```

The same root layout also supports Kaggle-style asset lookup because `main.py` reads `deck.csv` locally first and falls back to `/kaggle_simulations/agent/deck.csv`.

## Observation Inspection

Run one verbose inspection game:

```powershell
python scripts\self_play.py --games 1 --verbose
```

The observed API shape is summarized in `.research/reports/observations/OBSERVATION_SCHEMA.md`.

## V2 Policy

`main.py` keeps the initial deck selection unchanged, then delegates legal action choices to `deepmew.policy`.

The V2 policy scores legal options by `SelectContext` and `OptionType`, guards every return for valid length/index/duplicate constraints, and falls back to a deterministic legal selection if scoring fails. It prefers attacks, visible skills, evolution/setup, energy attachment, play/draw/search style actions, and low-risk development over `END`. It penalizes unknown, discard, retreat, and other risky actions unless forced by the legal context.

Run behavior probes:

```powershell
python scripts\self_play.py --games 1 --verbose --option-limit 3
```

Compare cheap baselines:

```powershell
python scripts\self_play.py --games 50 --compare
```

