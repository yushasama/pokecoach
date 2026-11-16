from __future__ import annotations

from deepmew.opponents.mixture import MixtureEntry, OpponentMixture


def _uniform(specs: list[str]) -> tuple[MixtureEntry, ...]:
    return tuple(MixtureEntry.from_text(spec, 1.0) for spec in specs)


BUILTIN_MIXTURES: dict[str, OpponentMixture] = {
    "smoke": OpponentMixture(
        name="smoke",
        version="v1",
        entries=_uniform(["random", "first-legal", "v2.4", "v4"]),
        description="Tiny sanity population covering floors, stable default, and arena-only V4.",
        intended_use="Fast integration smoke and manifest/report verification.",
        diagnostic_only=True,
        warning="Diagnostic only; not ladder truth.",
    ),
    "v3_experiments": OpponentMixture(
        name="v3_experiments",
        version="v1",
        entries=_uniform(
            [
                "v2.4",
                "v2.4[attach_timing=new]",
                "v2.4[evolve_priority=new]",
                "v2.4[semantics=off]",
                "first-legal",
            ]
        ),
        description="V3 experimental ablations plus default and first-legal control.",
        intended_use="Stress default or candidate policies against currently quarantined V3 signals.",
        diagnostic_only=True,
        warning="Do not promote any ablation from this mixture alone.",
    ),
    "v4_compare": OpponentMixture(
        name="v4_compare",
        version="v1",
        entries=_uniform(
            [
                "v2.4",
                "v4",
                "first-legal",
                "v2.4[attach_timing=new]",
                "v2.4[evolve_priority=new]",
            ]
        ),
        description="Comparison set for arena-only V4 against stable and gated V3 baselines.",
        intended_use="Local V4 regression/stress testing before any broader V4 review.",
        diagnostic_only=True,
        warning="V4 is arena-only and experimental; mixture results are diagnostic.",
    ),
    "baseline_zoo": OpponentMixture(
        name="baseline_zoo",
        version="v1",
        entries=_uniform(
            [
                "random",
                "first-legal",
                "v2.4",
                "v4",
                "v2.4[semantics=off]",
                "v2.4[attach_timing=new]",
                "v2.4[evolve_priority=new]",
            ]
        ),
        description="Simple baseline and current experimental-policy zoo with full support.",
        intended_use="Default local opponent-population smoke before replay/RL infrastructure.",
        diagnostic_only=True,
        warning="Local mixture is a stress test, not a solved meta or ladder proxy.",
    ),
}


def get_mixture(name: str) -> OpponentMixture:
    try:
        return BUILTIN_MIXTURES[name].validate()
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_MIXTURES))
        raise ValueError(f"unknown opponent mixture: {name}; known mixtures: {known}") from exc


def list_mixtures() -> tuple[str, ...]:
    return tuple(sorted(BUILTIN_MIXTURES))
