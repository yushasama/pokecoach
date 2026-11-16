"""Experiment + component registry for the RL ablation ladder.

`EXPERIMENTS` resolves the RL0..RL6 ladder (each rung extends the previous one).
The component factories return the encoder / masker / reward / opponent sampler /
feature modules for a given config, so the trainer can assemble a run purely from
config without hardwiring any ablation.
"""

from __future__ import annotations

from deepmew.rl.action_mask import ActionMasker
from deepmew.rl.config import RLExperimentConfig, extend
from deepmew.rl.features import build_feature_modules
from deepmew.rl.observation import ObservationEncoder
from deepmew.rl.opponents import make_opponent_sampler
from deepmew.rl.rewards import make_reward


STARMIE_D1_DECK_PATH = ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"
GRIMMSNARL_D2_DECK_PATH = (
    ".research/opponent_decks/01641f8a-79c0-11f1-bdeb-0242ac130203/"
    "0-Eduardo_Rocha_de_Andrade/deck.csv"
)


def _build_experiments() -> dict[str, RLExperimentConfig]:
    rl0 = RLExperimentConfig(
        experiment_id="RL0_pure_masked_ppo",
        observation="base",
        action_mask="legal_only",
        reward="terminal_win_loss",
        opponent_sampler="simple_baseline_mix",
        feature_modules=(),
        notes="masked-PPO control baseline",
    ).validate()
    starmie_health = extend(
        rl0,
        "RL0_starmie_health_v0",
        opponent_sampler="starmie_d1_health_v0",
        agent_deck_id="D1",
        agent_deck_path=STARMIE_D1_DECK_PATH,
        opponent_deck_id="D1",
        opponent_deck_path=STARMIE_D1_DECK_PATH,
        max_steps_per_game=5000,
        notes="bounded terminal-reward health smoke on exact Starmie D1 policy/deck pairs",
    ).validate()
    starmie_league = extend(
        rl0,
        "RL0_starmie_league_terminal_v0",
        opponent_sampler="starmie_league_initial_v0",
        agent_deck_id="D1",
        agent_deck_path=STARMIE_D1_DECK_PATH,
        opponent_deck_id="per_binding_manifest",
        opponent_deck_path="",
        max_steps_per_game=5000,
        notes="terminal win/loss Starmie league learner with immutable generation-zero opponent pairs",
    ).validate()
    grimmsnarl_health = extend(
        rl0,
        "RL0_grimmsnarl_d2_health_v0",
        opponent_sampler="grimmsnarl_d2_health_v0",
        agent_deck_id="D2",
        agent_deck_path=GRIMMSNARL_D2_DECK_PATH,
        opponent_deck_id="D2",
        opponent_deck_path=GRIMMSNARL_D2_DECK_PATH,
        max_steps_per_game=5000,
        notes="bounded terminal-reward health smoke with exact Grimmsnarl D2 on both seats",
    ).validate()
    grimmsnarl_league = extend(
        rl0,
        "RL0_grimmsnarl_d2_terminal_v0",
        opponent_sampler="grimmsnarl_league_initial_v0",
        agent_deck_id="D2",
        agent_deck_path=GRIMMSNARL_D2_DECK_PATH,
        opponent_deck_id="per_binding_manifest",
        opponent_deck_path="",
        max_steps_per_game=5000,
        notes="terminal win/loss Grimmsnarl league learner with exact D2 generation-zero bindings",
    ).validate()

    rl1 = extend(rl0, "RL1_bc_warm_start", bc_init=True).validate()
    rl1_top_player = extend(rl0, "RL1_top_player_bc_warm_start", bc_init=True).validate()
    rl1_top_player_canary = extend(rl0, "RL1_top_player_bc_canary_v0", bc_init=True).validate()
    rl1_top_player_ppo = extend(rl0, "RL1_top_player_bc_init_ppo_probe_v0", bc_init=True).validate()
    baseline_teacher_probe = extend(
        rl0,
        "baseline_teacher_bc_probe_v0",
        bc_init=True,
    ).validate()
    baseline_teacher_ppo = extend(
        rl0,
        "baseline_teacher_bc_init_ppo_probe_v0",
        bc_init=True,
    ).validate()
    rl1_local_teacher = extend(
        rl0,
        "RL1_local_teacher_bc_probe_v0",
        bc_init=True,
    ).validate()
    rl1a_teacher_abomasnow_bc = extend(
        rl0,
        "RL1A_teacher_abomasnow_bc_v0",
        bc_init=True,
    ).validate()
    rl1a_teacher_abomasnow_bc_ppo = extend(
        rl0,
        "RL1A_teacher_abomasnow_bc_init_ppo_probe_v0",
        bc_init=True,
    ).validate()
    rl2 = extend(rl1, "RL2_prize_shaping", reward="terminal_win_loss+prize_delta").validate()
    rl3 = extend(
        rl2, "RL3_support_features",
        feature_modules=("support_chain",), support_features=True,
    ).validate()
    rl4 = extend(rl3, "RL4_opponent_mixture", opponent_sampler="frozen_mixture").validate()
    rl4_arena_canary = extend(
        rl0,
        "RL4_arena_opponent_mix_canary",
        reward="terminal_win_loss+prize_delta",
        opponent_sampler="reference_weighted_arena_mix",
        agent_deck_id="pilot_sample_abomasnow_v0",
        agent_deck_path=".research/decks/pilot_sample_abomasnow_v0/decklist.csv",
        opponent_deck_id="opponent_policy_default_or_reference_deck",
        notes="200k canary config: pilot deck, reference-weighted arena mix, no BC warm start, credit monitors only",
    ).validate()
    rl4b_public_belief = extend(
        rl4_arena_canary,
        "RL4B_public_belief",
        feature_modules=("public_belief",),
        notes="RL4 + public-belief features (DEC-052) in actor+critic: DEC-038 public-only posterior "
              "+ public board/prize/threat; same deck/reward/opponent mix as RL4 control; no BC, no search",
    ).validate()
    rl5_floor_raise_potential = extend(
        rl4_arena_canary,
        "RL5_floor_raise_potential",
        reward="terminal_win_loss+potential_shaping",
        notes="RL5 floor-raise Phase A (DEC-054): RL4 control config with potential-based "
              "reward shaping (Ng 1999, public board potential); same pilot deck / opponent "
              "mix / features as RL4 control so reward is the only changed variable",
    ).validate()
    rl5 = extend(
        rl4, "RL5_dsu_belief",
        feature_modules=("support_chain", "dsu_belief"), dsu=True, belief=True,
    ).validate()
    rl6 = extend(
        rl5, "RL6_robust_posterior",
        feature_modules=("support_chain", "dsu_belief", "robust_posterior"),
        static_priors=True,
    ).validate()

    experiments = [
        rl0,
        starmie_health,
        starmie_league,
        grimmsnarl_health,
        grimmsnarl_league,
        rl1,
        rl1_top_player,
        rl1_top_player_canary,
        rl1_top_player_ppo,
        baseline_teacher_probe,
        baseline_teacher_ppo,
        rl1_local_teacher,
        rl1a_teacher_abomasnow_bc,
        rl1a_teacher_abomasnow_bc_ppo,
        rl2,
        rl3,
        rl4,
        rl4_arena_canary,
        rl4b_public_belief,
        rl5,
        rl6,
        rl5_floor_raise_potential,
    ]
    return {config.experiment_id: config for config in experiments}


EXPERIMENTS: dict[str, RLExperimentConfig] = _build_experiments()

# Short aliases (RL0..RL6) for CLI convenience.
EXPERIMENT_ALIASES: dict[str, str] = {}
for config in EXPERIMENTS.values():
    EXPERIMENT_ALIASES.setdefault(config.experiment_id.split("_", 1)[0], config.experiment_id)


def resolve_experiment(name: str) -> RLExperimentConfig:
    if name in EXPERIMENTS:
        return EXPERIMENTS[name]
    if name in EXPERIMENT_ALIASES:
        return EXPERIMENTS[EXPERIMENT_ALIASES[name]]
    raise ValueError(
        f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)} or {sorted(EXPERIMENT_ALIASES)}"
    )


def build_encoder(config: RLExperimentConfig) -> ObservationEncoder:
    return ObservationEncoder(feature_modules=build_feature_modules(config.feature_modules))


def build_masker(config: RLExperimentConfig, *, max_options: int = 64) -> ActionMasker:
    return ActionMasker(max_options=max_options)


def build_reward(config: RLExperimentConfig):
    return make_reward(config.reward)


def build_opponent_sampler(config: RLExperimentConfig, *, seed: int | None = None):
    return make_opponent_sampler(
        config.opponent_sampler, seed=config.seed if seed is None else seed
    )
