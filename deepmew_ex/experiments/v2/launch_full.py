"""Light entrypoint for ``deepmew_ex.experiments.v2.full``.

Spawned loader workers re-import the parent's ``__main__`` module. Launching
the trainer through this module keeps that re-import free of torch, which cuts
each worker's committed memory from about 1.4 GB to about 0.5 GB on Windows.
Same CLI as ``python -m deepmew_ex.experiments.v2.full``.
"""

if __name__ == '__main__':
    from deepmew_ex.experiments.v2.full import main

    main()
