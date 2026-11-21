# CABT engine setup

PokéCoach plays every game on the CABT engine from the Kaggle Pokémon TCG AI
Battle competition. The engine has no redistribution license, so it is not in
this repository (see `CREDITS.md`).

The [one-time setup command](../../README.md#run-it), `python -m deepmew_ex.coach.engine_setup`,
installs and verifies the SDK in `outputs/cabt-sdk/`. It downloads [cg-lib version 8](https://www.kaggle.com/datasets/kiyotah/cg-lib/versions/8)
and, on Windows x64, Linux ARM64 and Apple Silicon, the matching
[Kaggle native library](https://github.com/Kaggle/kaggle-environments/tree/03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303/kaggle_environments/envs/cabt/cg).
The archive includes the Linux x86-64 library. The downloads require internet access during setup.

For an existing manual installation, find the directory containing `cg/api.py`
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
