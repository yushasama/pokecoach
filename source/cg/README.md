# Game engine goes here (not included)

PokéCoach plays every game on the CABT engine from the Kaggle Pokémon TCG AI
Battle competition. The engine has no redistribution license, so it is not in
this repository (see `CREDITS.md`).

Copy the competition's `cg` package into this folder so it contains:

```
source/cg/__init__.py
source/cg/api.py
source/cg/game.py
source/cg/sim.py
source/cg/utils.py
source/cg/cg.dll      (Windows)
source/cg/libcg.so    (Linux)
```

Engine documentation: https://matsuoinstitute.github.io/cabt/
