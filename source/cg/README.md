# Game engine goes here (not included)

PokéCoach plays every game on the CABT engine from the Kaggle Pokémon TCG AI
Battle competition. The engine has no redistribution license, so it is not in
this repository (see `CREDITS.md`).

Get the SDK from the [competition data](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data)
or use an existing installation. Find the directory containing `cg/api.py`
and the native library for your OS. Set `PYTHONPATH` to the **parent of cg**
in the same shell that starts PokéCoach. For example, for an SDK extracted
to `C:\cabt\cg`, run:

```powershell
$env:PYTHONPATH = 'C:\cabt'
python -m deepmew_ex.coach.server
```

On Linux, use `PYTHONPATH=/path/to/sdk python -m deepmew_ex.coach.server`.
An already importable SDK needs no path setting. The pinned
`kaggle-environments` wheel alone omits `cg.api`.

Alternatively, copy the competition's `cg` package into this folder:

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
