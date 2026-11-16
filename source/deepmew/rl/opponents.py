"""Opponent sampling for RL training and evaluation.

Opponents are drawn from existing arena agent specs (`first-legal`, `v2.4`,
`legacy-v2`, `random`, and gated ablations). Sampling is deterministic under a
fixed seed. RL0 starts from a small fixed baseline mix; later experiments swap
to a frozen mixture / league without touching the trainer (the trainer only
calls ``sample``).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from deepmew.arena.selection import select_action
from deepmew.arena.specs import parse_agent_spec
from deepmew.rl.deck_identity import (
    ROOT,
    DeckIdentity,
    deck_identity_from_deck,
    r4f_feature_deck_hash,
    read_deck_any,
    resolve_repo_path,
    root_deck_identity,
)


BINDING_PREFIX = "binding:"
POLICY_SPEC_KINDS = frozenset({"spec", "checkpoint", "bundle"})


@dataclass(frozen=True)
class PolicyDeckBinding:
    """Run-frozen research policy + public deck-manifest pair."""

    binding_id: str
    label: str
    policy_spec: str
    policy_artifact_sha256: str
    deck_manifest_path: str
    deck_id: str
    expected_deck_file_sha256: str
    deck_manifest_entry_id: str = ""
    policy_artifact_file: str = ""
    registered_bundle_sha256: str = ""
    feature_policy_id: str = ""
    feature_deck_hash: str = ""
    deck_policy_id: str = "reference_abomasnow"
    policy_role: str = "frozen_league_member"
    pairing_role: str = "league_training"
    strength_bucket: str = "historical_intermediate"

    @property
    def policy_kind(self) -> str:
        return self.policy_spec.partition(":")[0]

    @property
    def policy_target(self) -> str:
        return self.policy_spec.partition(":")[2]

    @property
    def deck_manifest_spec(self) -> str:
        suffix = f"#{self.deck_manifest_entry_id}" if self.deck_manifest_entry_id else ""

        return f"manifest:{self.deck_manifest_path}{suffix}"

    def validate(self) -> "PolicyDeckBinding":
        if not self.binding_id or not self.label:
            raise ValueError("policy/deck binding requires binding_id and label")
        if self.policy_kind not in POLICY_SPEC_KINDS or not self.policy_target:
            raise ValueError(
                f"{self.binding_id}: policy_spec must be one of "
                f"{sorted(POLICY_SPEC_KINDS)} with a non-empty target"
            )
        if not self.deck_manifest_path or not self.deck_id:
            raise ValueError(f"{self.binding_id}: deck manifest and deck_id are required")
        _require_sha256(self.expected_deck_file_sha256, field="expected_deck_file_sha256")
        if self.policy_kind == "spec":
            if self.policy_artifact_sha256:
                raise ValueError(f"{self.binding_id}: spec policy must not claim an artifact hash")
        else:
            _require_sha256(self.policy_artifact_sha256, field="policy_artifact_sha256")
            if not self.feature_deck_hash:
                raise ValueError(f"{self.binding_id}: artifact policy requires feature_deck_hash")
        if self.policy_kind == "bundle":
            _require_sha256(
                self.registered_bundle_sha256,
                field="registered_bundle_sha256",
            )
        elif self.registered_bundle_sha256:
            raise ValueError(
                f"{self.binding_id}: only bundle policies may claim a registered bundle hash"
            )

        return self


@dataclass(frozen=True)
class ResolvedPolicyDeckBinding:
    binding: PolicyDeckBinding
    deck: tuple[int, ...]
    deck_path: Path
    deck_file_sha256: str
    deck_identity: DeckIdentity
    policy_artifact_path: Path | None


PolicySpecLoader = Callable[[PolicyDeckBinding, tuple[int, ...], DeckIdentity], Any]
POLICY_SPEC_LOADERS: dict[str, PolicySpecLoader] = {}


def _require_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256")

    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def register_policy_spec_loader(
    policy_kind: str,
    loader: PolicySpecLoader,
    *,
    replace: bool = False,
) -> None:
    """Register an actor-visible runtime adapter for one artifact kind."""

    if policy_kind not in POLICY_SPEC_KINDS:
        raise ValueError(f"unknown policy loader kind {policy_kind!r}")
    if policy_kind in POLICY_SPEC_LOADERS and not replace:
        raise ValueError(f"policy loader {policy_kind!r} is already registered")
    POLICY_SPEC_LOADERS[policy_kind] = loader


def register_policy_deck_binding(
    binding: PolicyDeckBinding,
    *,
    replace_existing: bool = False,
) -> None:
    """Register one immutable research policy/deck pair for this process."""

    binding.validate()
    existing = POLICY_DECK_BINDINGS.get(binding.binding_id)
    if existing is not None and existing != binding and not replace_existing:
        raise ValueError(f"policy/deck binding {binding.binding_id!r} is already registered")
    POLICY_DECK_BINDINGS[binding.binding_id] = binding
    resolve_policy_deck_binding.cache_clear()


def remap_policy_artifact_paths(overrides: Mapping[str, str | Path]) -> None:
    """Rebind immutable artifact bytes to portable paths without changing identity."""

    unknown = set(overrides) - set(POLICY_DECK_BINDINGS)
    if unknown:
        raise ValueError(f"unknown policy/deck binding overrides: {sorted(unknown)}")
    for binding_id, raw_path in overrides.items():
        binding = POLICY_DECK_BINDINGS[binding_id]
        if binding.policy_kind == "spec":
            raise ValueError(f"{binding_id}: spec policies do not have artifact paths")
        path = Path(raw_path)
        if not str(path):
            raise ValueError(f"{binding_id}: artifact override path must be non-empty")
        POLICY_DECK_BINDINGS[binding_id] = replace(
            binding,
            policy_spec=f"{binding.policy_kind}:{path}",
        )
    resolve_policy_deck_binding.cache_clear()


def _manifest_deck_path(binding: PolicyDeckBinding) -> tuple[Path, Path]:
    manifest_path = resolve_repo_path(binding.deck_manifest_path).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if binding.deck_manifest_entry_id:
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"{binding.binding_id}: deck registry manifest has no entries")
        matches = [row for row in entries if str(row.get("id")) == binding.deck_manifest_entry_id]
        if len(matches) != 1:
            raise ValueError(
                f"{binding.binding_id}: expected one deck entry {binding.deck_manifest_entry_id!r}, "
                f"got {len(matches)}"
            )
        deck_ref = str(matches[0].get("deck_path") or "")
        deck_path = resolve_repo_path(deck_ref)
    elif isinstance(payload.get("outputs"), dict) and payload["outputs"].get("deck_csv"):
        deck_path = resolve_repo_path(str(payload["outputs"]["deck_csv"]))
    elif payload.get("deck"):
        deck_path = manifest_path.parent / str(payload["deck"])
    else:
        raise ValueError(f"{binding.binding_id}: deck manifest does not identify a deck file")
    deck_path = deck_path.resolve(strict=True)

    return manifest_path, deck_path


def _policy_artifact_path(binding: PolicyDeckBinding) -> Path | None:
    if binding.policy_kind == "spec":
        return None
    target = resolve_repo_path(binding.policy_target).resolve(strict=True)
    artifact_path = target / binding.policy_artifact_file if target.is_dir() else target
    artifact_path = artifact_path.resolve(strict=True)
    actual = _file_sha256(artifact_path)
    expected = _require_sha256(binding.policy_artifact_sha256, field="policy_artifact_sha256")
    if actual != expected:
        raise ValueError(
            f"{binding.binding_id}: policy artifact hash mismatch: expected {expected}, got {actual}"
        )

    return artifact_path


@lru_cache(maxsize=None)
def resolve_policy_deck_binding(binding: PolicyDeckBinding) -> ResolvedPolicyDeckBinding:
    binding.validate()
    manifest_path, deck_path = _manifest_deck_path(binding)
    deck_file_sha256 = _file_sha256(deck_path)
    expected_deck_sha256 = _require_sha256(
        binding.expected_deck_file_sha256,
        field="expected_deck_file_sha256",
    )
    if deck_file_sha256 != expected_deck_sha256:
        raise ValueError(
            f"{binding.binding_id}: deck file hash mismatch: "
            f"expected {expected_deck_sha256}, got {deck_file_sha256}"
        )
    deck = tuple(read_deck_any(deck_path))
    if len(deck) != 60:
        raise ValueError(f"{binding.binding_id}: bound deck must contain 60 cards, got {len(deck)}")
    if binding.policy_kind == "checkpoint":
        observed_feature_deck_hash = r4f_feature_deck_hash(deck)
        if observed_feature_deck_hash != binding.feature_deck_hash:
            raise ValueError(
                f"{binding.binding_id}: R4F feature deck hash mismatch: "
                f"expected {binding.feature_deck_hash}, got {observed_feature_deck_hash}"
            )
    artifact_path = _policy_artifact_path(binding)
    artifact_note = (
        f" policy_artifact_sha256={_require_sha256(binding.policy_artifact_sha256, field='policy_artifact_sha256')};"
        if binding.policy_artifact_sha256
        else ""
    )
    identity = deck_identity_from_deck(
        policy_id=binding.binding_id,
        policy_role=binding.policy_role,
        deck_id=binding.deck_id,
        deck_role="manifest_bound_full_deck",
        deck=deck,
        deck_status="runnable_full_deck",
        pairing_role=binding.pairing_role,
        deck_path=deck_path,
        deck_source_id=str(manifest_path),
        notes=(
            f"Run-frozen binding {binding.binding_id}; deck_file_sha256={deck_file_sha256};"
            f"{artifact_note} strength_bucket={binding.strength_bucket}."
        ),
    )

    return ResolvedPolicyDeckBinding(
        binding=binding,
        deck=deck,
        deck_path=deck_path,
        deck_file_sha256=deck_file_sha256,
        deck_identity=identity,
        policy_artifact_path=artifact_path,
    )


class _BoundPolicy:
    """Lazy delegate; artifact loaders receive actor-visible observations only."""

    def __init__(self, resolved: ResolvedPolicyDeckBinding):
        self.resolved = resolved
        self.delegate: Any | None = None

    def materialize(self) -> None:
        """Load the bound delegate exactly once before checkpoint restoration."""

        if self.delegate is not None:
            return
        kind = self.resolved.binding.policy_kind
        loader = POLICY_SPEC_LOADERS.get(kind)
        if loader is None:
            raise ImportError(
                f"{self.resolved.binding.binding_id}: no actor-visible {kind!r} policy loader registered"
            )
        delegate = loader(
            self.resolved.binding,
            self.resolved.deck,
            self.resolved.deck_identity,
        )
        if delegate is None:
            raise ValueError(
                f"{self.resolved.binding.binding_id}: {kind!r} policy loader returned no delegate"
            )
        self.delegate = delegate

    def reset(self) -> None:
        reset = getattr(self.delegate, "reset", None)
        if callable(reset):
            try:
                reset()
            except TypeError:
                reset(0)

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        self.materialize()
        if self.delegate is None:
            raise ValueError("bound policy materialization completed without a delegate")
        selected = self.delegate.select(obs, rng)

        return [int(index) for index in selected]


def _load_bound_spec(
    binding: PolicyDeckBinding,
    deck: tuple[int, ...],
    _identity: DeckIdentity,
) -> Any:
    target = binding.policy_target
    from deepmew.rl1_teacher_bc.teachers import TEACHER_IDS, make_teacher

    teacher_id = REFERENCE_TEACHER_ALIASES.get(target, target)
    if teacher_id in TEACHER_IDS:
        teacher = make_teacher(teacher_id)
        if tuple(teacher.deck) != deck:
            raise ValueError(
                f"{binding.binding_id}: reference-teacher deck disagrees with frozen binding"
            )

        return teacher

    spec = parse_agent_spec(target)

    class BoundSpecPolicy:
        def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
            return select_action(spec, obs, rng)

    return BoundSpecPolicy()


register_policy_spec_loader("spec", _load_bound_spec)


REFERENCE_TEACHER_ALIASES = {
    "reference_abomasnow": "teacher_abomasnow",
    "reference_lucario": "teacher_lucario",
    "reference_iono": "teacher_iono",
    "reference_dragapult": "teacher_dragapult",
}

CROSS_DECK_STRESS_SPECS: dict[str, tuple[str, str]] = {
    "reference_iono_on_topplayer_dragapult_ex": ("reference_iono", "top_player_snapshot_02"),
    "reference_lucario_on_topplayer_trevenant": ("reference_lucario", "top_player_snapshot_01"),
}

TOP_PLAYER_SNAPSHOT_JSON = ROOT / ".research" / "arena" / "top_player_snapshots_v0.json"

D0_SUBMISSION_MANIFEST = ".research/submissions/20260708-starmie-awr/submission_manifest.json"
D0_BUNDLE = ".research/submissions/20260708-starmie-awr"
D0_DECK_FILE_SHA256 = "f265b59b3ca7b524e593e9edeaae9a76f31d5179705947cbb35da2bab1ec9068"
D0_BUNDLE_WEIGHTS_SHA256 = "c6ac0bf451689536a29e15d8116f1fcac7fdcb771d242d5eac037bf2c808d2dc"
D0_REGISTERED_BUNDLE_SHA256 = "4a34de9ee7014fd3480e168042eb21e1f1473ca43bdbd15ddda9283a7deefa9c"
D1_DECK_REGISTRY = ".research/DECK_REGISTRY.json"
D1_DECK_FILE_SHA256 = "2636fcaf8aedd7a156d0b2e33185b0e3878a9c2e2b77425f9203cc53e96e7b28"
D2_DECK_MANIFEST = (
    ".research/opponent_decks/01641f8a-79c0-11f1-bdeb-0242ac130203/"
    "0-Eduardo_Rocha_de_Andrade/deck_manifest.json"
)
D2_DECK_FILE_SHA256 = "c960f71296a4ca797d8b8421f8c3d059752296ba34721108e6b625e02ca0410a"
R7_CHECKPOINT = "D:/DeepMew_artifacts/checkpoints/r22-staryu-full-league-a4-8c-r7-BEST.pt"
R7_CHECKPOINT_SHA256 = "e293433756fc7edad865f887f9e130d5cec1e4d4067da30446ddebf9b714aee2"
D1_R4F_FEATURE_DECK_HASH = "b15853cdfb8304f39f879638102a7fd161f1102d81e6d3f36d5beca4db9e01b0"
D2_R4F_FEATURE_DECK_HASH = "e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e"
R23_CHECKPOINT = (
    ".research/league_temporal_rl/r23-next-after-l0-l2-20260710/"
    "r0_train/selected_ar_bc_r4f_candidate.pt"
)
R23_CHECKPOINT_SHA256 = "4e02475f3b997622ee5f9d8201998b6a3e7759ccd40a7183359b280485b7bf9d"
EXPLOITER_CHECKPOINT = ".research/r12/lane1_r4f_focal_reweight/selected_ar_bc_r4f_reweighted_candidate.pt"
EXPLOITER_CHECKPOINT_SHA256 = "ffd5dcf3104b015a5e11ed68bc194dccc0bbac072055a12f672f1b9236fddb8a"


POLICY_DECK_BINDINGS: dict[str, PolicyDeckBinding] = {
    "starmie-r7-d1": PolicyDeckBinding(
        binding_id="starmie-r7-d1",
        label="starmie-r7-d1",
        policy_spec=f"checkpoint:{R7_CHECKPOINT}",
        policy_artifact_sha256=R7_CHECKPOINT_SHA256,
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        feature_policy_id="starmie",
        feature_deck_hash=D1_R4F_FEATURE_DECK_HASH,
        policy_role="local_frontier_anchor",
        pairing_role="starmie_league_training",
        strength_bucket="local_frontier",
    ),
    "starmie-r23-d1": PolicyDeckBinding(
        binding_id="starmie-r23-d1",
        label="starmie-r23-d1",
        policy_spec=f"checkpoint:{R23_CHECKPOINT}",
        policy_artifact_sha256=R23_CHECKPOINT_SHA256,
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        feature_policy_id="starmie",
        feature_deck_hash=D1_R4F_FEATURE_DECK_HASH,
        policy_role="rejected_historical_league_member",
        pairing_role="starmie_league_training",
        strength_bucket="historical_intermediate",
    ),
    "starmie-submitted-6779-d0": PolicyDeckBinding(
        binding_id="starmie-submitted-6779-d0",
        label="starmie-submitted-6779-d0",
        policy_spec=f"bundle:{D0_BUNDLE}",
        policy_artifact_sha256=D0_BUNDLE_WEIGHTS_SHA256,
        policy_artifact_file="r5_policy_weights.npz",
        registered_bundle_sha256=D0_REGISTERED_BUNDLE_SHA256,
        deck_manifest_path=D0_SUBMISSION_MANIFEST,
        deck_id="D0",
        expected_deck_file_sha256=D0_DECK_FILE_SHA256,
        feature_policy_id="top_player_replay",
        feature_deck_hash="18ffc1e65b611872",
        policy_role="submitted_anchor",
        pairing_role="starmie_league_training",
        strength_bucket="submitted",
    ),
    "starmie-historical-legacy-d1": PolicyDeckBinding(
        binding_id="starmie-historical-legacy-d1",
        label="starmie-historical-legacy-d1",
        policy_spec="spec:legacy-v2",
        policy_artifact_sha256="",
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        policy_role="historical_agent",
        pairing_role="starmie_league_training",
        strength_bucket="historical_intermediate",
    ),
    "starmie-mirror-g00-d1": PolicyDeckBinding(
        binding_id="starmie-mirror-g00-d1",
        label="starmie-mirror-g00-d1",
        policy_spec=f"checkpoint:{R7_CHECKPOINT}",
        policy_artifact_sha256=R7_CHECKPOINT_SHA256,
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        feature_policy_id="starmie",
        feature_deck_hash=D1_R4F_FEATURE_DECK_HASH,
        policy_role="generation_zero_mirror_bootstrap",
        pairing_role="starmie_league_training",
        strength_bucket="local_frontier",
    ),
    "starmie-exploiter-d1": PolicyDeckBinding(
        binding_id="starmie-exploiter-d1",
        label="starmie-exploiter-d1",
        policy_spec=f"checkpoint:{EXPLOITER_CHECKPOINT}",
        policy_artifact_sha256=EXPLOITER_CHECKPOINT_SHA256,
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        feature_policy_id="starmie",
        feature_deck_hash=D1_R4F_FEATURE_DECK_HASH,
        policy_role="provisional_exploiter_slot",
        pairing_role="starmie_league_training",
        strength_bucket="unverified_exploiter",
    ),
    "starmie-hard-extreme-d1": PolicyDeckBinding(
        binding_id="starmie-hard-extreme-d1",
        label="starmie-hard-extreme-d1",
        policy_spec="spec:v4",
        policy_artifact_sha256="",
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        policy_role="provisional_hard_extreme_slot",
        pairing_role="starmie_league_training",
        strength_bucket="unverified_hard_extreme",
    ),
    "starmie-v24-d1": PolicyDeckBinding(
        binding_id="starmie-v24-d1",
        label="starmie-v24-d1",
        policy_spec="spec:v2.4",
        policy_artifact_sha256="",
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        policy_role="health_baseline",
        pairing_role="starmie_health_smoke",
        strength_bucket="reference",
    ),
    "starmie-first-legal-d1": PolicyDeckBinding(
        binding_id="starmie-first-legal-d1",
        label="starmie-first-legal-d1",
        policy_spec="spec:first-legal",
        policy_artifact_sha256="",
        deck_manifest_path=D1_DECK_REGISTRY,
        deck_manifest_entry_id="D1",
        deck_id="D1",
        expected_deck_file_sha256=D1_DECK_FILE_SHA256,
        policy_role="health_baseline",
        pairing_role="starmie_health_smoke",
        strength_bucket="reference",
    ),
    "grimmsnarl-v0-d2": PolicyDeckBinding(
        binding_id="grimmsnarl-v0-d2",
        label="grimmsnarl-v0-d2",
        policy_spec="spec:v2.4",
        policy_artifact_sha256="",
        deck_manifest_path=D2_DECK_MANIFEST,
        deck_id="D2",
        expected_deck_file_sha256=D2_DECK_FILE_SHA256,
        feature_policy_id="grimmsnarl",
        policy_role="generation_zero_baseline",
        pairing_role="grimmsnarl_league_training",
        strength_bucket="reference",
    ),
    "grimmsnarl-mirror-g00-d2": PolicyDeckBinding(
        binding_id="grimmsnarl-mirror-g00-d2",
        label="grimmsnarl-mirror-g00-d2",
        policy_spec=f"checkpoint:{R7_CHECKPOINT}",
        policy_artifact_sha256=R7_CHECKPOINT_SHA256,
        deck_manifest_path=D2_DECK_MANIFEST,
        deck_id="D2",
        expected_deck_file_sha256=D2_DECK_FILE_SHA256,
        feature_policy_id="grimmsnarl",
        feature_deck_hash=D2_R4F_FEATURE_DECK_HASH,
        policy_role="generation_zero_current_learner_snapshot",
        pairing_role="grimmsnarl_league_training",
        strength_bucket="unverified_transfer_bootstrap",
    ),
    "grimmsnarl-first-legal-d2": PolicyDeckBinding(
        binding_id="grimmsnarl-first-legal-d2",
        label="grimmsnarl-first-legal-d2",
        policy_spec="spec:first-legal",
        policy_artifact_sha256="",
        deck_manifest_path=D2_DECK_MANIFEST,
        deck_id="D2",
        expected_deck_file_sha256=D2_DECK_FILE_SHA256,
        feature_policy_id="grimmsnarl",
        policy_role="health_baseline",
        pairing_role="grimmsnarl_health_smoke",
        strength_bucket="reference",
    ),
}


def _arena_policy_role(label: str) -> str:
    base = label.split("[", 1)[0]
    if base in {"random", "first-legal", "legacy-v2", "v2.4"} and "[" not in label:
        return "baseline"
    return "research_candidate"


@dataclass(frozen=True)
class OpponentAgent:
    """Adapter so an arena agent spec fits the env's opponent interface."""

    spec: Any
    label: str
    arena_spec: bool = True
    deck_identity_override: DeckIdentity | None = None
    deck_override: tuple[int, ...] | None = None
    binding: PolicyDeckBinding | None = None
    policy_artifact_path: str = ""

    @classmethod
    def from_str(cls, raw: str) -> "OpponentAgent":
        if raw.startswith(BINDING_PREFIX):
            binding_id = raw.removeprefix(BINDING_PREFIX)
            binding = POLICY_DECK_BINDINGS.get(binding_id)
            if binding is None:
                raise ValueError(
                    f"unknown policy/deck binding {binding_id!r}; known: {sorted(POLICY_DECK_BINDINGS)}"
                )

            return _make_policy_deck_binding_opponent(binding)
        if raw in CROSS_DECK_STRESS_SPECS:
            return _make_cross_deck_stress_opponent(raw)
        if raw.startswith("top_player_snapshot_"):
            return _make_top_player_snapshot_opponent(raw)
        if raw in REFERENCE_TEACHER_ALIASES:
            from deepmew.rl1_teacher_bc.teachers import make_teacher

            teacher = make_teacher(REFERENCE_TEACHER_ALIASES[raw])
            identity = _reference_identity(raw, teacher)
            return cls(
                spec=teacher,
                label=raw,
                arena_spec=False,
                deck_identity_override=identity,
                deck_override=tuple(teacher.deck),
            )
        spec = parse_agent_spec(raw)
        role = _arena_policy_role(spec.label)
        identity = root_deck_identity(
            policy_id=spec.label,
            policy_role=role,
            pairing_role="baseline" if role == "baseline" else "research_candidate",
        )
        return cls(
            spec=spec,
            label=spec.label,
            deck_identity_override=identity,
            deck_override=tuple(read_deck_any("deck.csv")),
        )

    @property
    def deck(self) -> list[int] | None:
        if self.deck_override is not None:
            return list(self.deck_override)
        deck = getattr(self.spec, "deck", None)
        return list(deck) if deck is not None else None

    @property
    def deck_identity(self) -> DeckIdentity:
        if self.deck_identity_override is not None:
            return self.deck_identity_override
        role = _arena_policy_role(self.label)
        return root_deck_identity(
            policy_id=self.label,
            policy_role=role,
            pairing_role="baseline" if role == "baseline" else "research_candidate",
        )

    @property
    def pair_identity(self) -> dict[str, Any]:
        identity = self.deck_identity
        if self.binding is None:
            return {
                "binding_id": "",
                "label": self.label,
                "policy_spec": self.label,
                "checkpoint_sha256": None,
                "bundle_sha256": None,
                "bundle_npz_sha256": None,
                "policy_artifact_sha256": None,
                "policy_artifact_path": "",
                "feature_deck_hash": None,
                "deck_spec": f"deck:{identity.deck_source_path}",
                "deck_manifest": "",
                "deck_hash": identity.deck_hash,
                "deck_id": identity.deck_id,
                "policy_role": identity.policy_role,
                "pairing_role": identity.pairing_role,
                "strength_bucket": "unknown",
            }
        binding = self.binding
        artifact_sha256 = _require_sha256(
            binding.policy_artifact_sha256,
            field="policy_artifact_sha256",
        ) if binding.policy_artifact_sha256 else None

        return {
            "binding_id": binding.binding_id,
            "label": binding.label,
            "policy_spec": binding.policy_spec,
            "checkpoint_sha256": artifact_sha256 if binding.policy_kind == "checkpoint" else None,
            "bundle_sha256": (
                binding.registered_bundle_sha256
                if binding.policy_kind == "bundle"
                else None
            ),
            "bundle_npz_sha256": (
                artifact_sha256 if binding.policy_kind == "bundle" else None
            ),
            "policy_artifact_sha256": artifact_sha256,
            "policy_artifact_path": self.policy_artifact_path,
            "feature_deck_hash": binding.feature_deck_hash or None,
            "deck_spec": f"deck:{identity.deck_source_path}",
            "deck_manifest": binding.deck_manifest_spec,
            "deck_hash": identity.deck_hash,
            "deck_id": identity.deck_id,
            "policy_role": identity.policy_role,
            "pairing_role": identity.pairing_role,
            "strength_bucket": binding.strength_bucket,
        }

    def reset(self) -> None:
        reset = getattr(self.spec, "reset", None)
        if callable(reset):
            reset()

    def materialize(self) -> None:
        """Materialize a lazy bound policy without changing its gameplay state."""

        materialize = getattr(self.spec, "materialize", None)
        if callable(materialize):
            materialize()

    def select(self, obs: dict[str, Any], rng: random.Random) -> list[int]:
        if not self.arena_spec:
            return self.spec.select(obs, rng)
        return select_action(self.spec, obs, rng)


def _make_policy_deck_binding_opponent(binding: PolicyDeckBinding) -> OpponentAgent:
    resolved = resolve_policy_deck_binding(binding)

    return OpponentAgent(
        spec=_BoundPolicy(resolved),
        label=binding.label,
        arena_spec=False,
        deck_identity_override=resolved.deck_identity,
        deck_override=resolved.deck,
        binding=binding,
        policy_artifact_path=str(resolved.policy_artifact_path or ""),
    )


def _reference_identity(label: str, teacher: Any, *, pairing_role: str = "canonical_policy_own_deck") -> DeckIdentity:
    deck = list(getattr(teacher, "deck"))
    teacher_id = str(getattr(teacher, "teacher_id", label))
    deck_id = str(getattr(teacher, "deck_id", f"sample_notebook_{teacher_id}_deck"))
    notes = "Sample/reference policy uses its notebook/list-defined deck."
    if teacher_id == "teacher_abomasnow":
        notes += " Hash matches pilot_sample_abomasnow_v0; treat as in_distribution_anchor, not a transfer target."
    return deck_identity_from_deck(
        policy_id=label,
        policy_role="reference_agent",
        deck_id=deck_id,
        deck_role="sample_reference_deck",
        deck=deck,
        deck_status="runnable_full_deck",
        pairing_role=pairing_role,
        deck_path=f"NOTEBOOK_DECKLISTS/{teacher_id}",
        deck_source_id=str(getattr(teacher, "teacher_source", "sample_rule_based")),
        notes=notes,
    )


def _snapshot_payload(snapshot_id: str) -> dict[str, Any]:
    import json

    if not TOP_PLAYER_SNAPSHOT_JSON.exists():
        raise ValueError(f"top-player snapshot metadata missing: {TOP_PLAYER_SNAPSHOT_JSON}")
    payload = json.loads(TOP_PLAYER_SNAPSHOT_JSON.read_text(encoding="utf-8"))
    for snapshot in payload.get("snapshots", []):
        if str(snapshot.get("snapshot_id")) == snapshot_id:
            return snapshot
    raise ValueError(f"unknown top-player snapshot opponent {snapshot_id!r}")


def _top_player_snapshot_deck(snapshot_id: str) -> tuple[dict[str, Any], list[int]]:
    snapshot = _snapshot_payload(snapshot_id)
    deck_path = snapshot.get("decklist_path")
    if not deck_path:
        raise ValueError(f"{snapshot_id}: missing decklist_path; refusing root deck fallback")
    deck = read_deck_any(Path(str(deck_path)))
    if len(deck) != 60:
        raise ValueError(f"{snapshot_id}: full runnable deck required, got {len(deck)} cards")
    return snapshot, deck


def _top_player_identity(
    snapshot_id: str,
    snapshot: dict[str, Any],
    deck: list[int],
    *,
    pairing_role: str = "top_player_clone",
    policy_id: str | None = None,
    policy_role: str = "top_player_clone",
) -> DeckIdentity:
    deck_path = str(snapshot.get("decklist_path") or "")
    core = str(snapshot.get("core_key") or "unknown")
    return deck_identity_from_deck(
        policy_id=policy_id or snapshot_id,
        policy_role=policy_role,
        deck_id=snapshot_id,
        deck_role=f"top_player_snapshot_{core}_full_deck",
        deck=deck,
        deck_status="runnable_full_deck",
        pairing_role=pairing_role,
        deck_path=deck_path,
        deck_source_id=str(snapshot.get("deck_hash") or ""),
        notes="Full legal snapshot deck; no root deck fallback allowed.",
    )


def _make_top_player_snapshot_opponent(raw: str) -> OpponentAgent:
    from deepmew.top_player_snapshot_adapters.adapters import (
        ReplayPreferenceTable,
        SnapshotAdapterPolicy,
        SnapshotAdapterSpec,
    )

    snapshot, deck = _top_player_snapshot_deck(raw)
    table = ReplayPreferenceTable(0, {}, {}, {})
    adapter_spec = SnapshotAdapterSpec(
        snapshot_id=raw,
        core=str(snapshot.get("core_key") or "unknown"),
        deck_hash=str(snapshot.get("deck_hash") or ""),
        decklist_path=Path(str(snapshot.get("decklist_path"))),
        deck=tuple(deck),
        source_rank=str(snapshot.get("source_rank") or ""),
        rank_priority=float(snapshot.get("rank_weighted_priority") or 0.0),
        adapter_type=str(snapshot.get("policy_adapter_type") or "heuristic_archetype_policy"),
        callable_status="callable",
        readiness=str(snapshot.get("readiness_status") or "deck_ready_policy_adapter_pending"),
        table=table,
        notes="Identity path uses the full deck and conservative legal fallback; exact replay table is loaded only by adapter audits.",
    )
    policy = SnapshotAdapterPolicy(adapter_spec)
    return OpponentAgent(
        spec=policy,
        label=raw,
        arena_spec=False,
        deck_identity_override=_top_player_identity(raw, snapshot, deck),
        deck_override=tuple(deck),
    )


def _make_cross_deck_stress_opponent(raw: str) -> OpponentAgent:
    base_policy, snapshot_id = CROSS_DECK_STRESS_SPECS[raw]
    base = OpponentAgent.from_str(base_policy)
    snapshot, deck = _top_player_snapshot_deck(snapshot_id)
    identity = _top_player_identity(
        snapshot_id,
        snapshot,
        deck,
        pairing_role="cross_deck_stress",
        policy_id=raw,
        policy_role=f"cross_deck_stress_policy_source:{base_policy}",
    )
    return OpponentAgent(base.spec, raw, base.arena_spec, identity, tuple(deck))


@dataclass(frozen=True)
class SampledOpponent:
    index: int
    agent: OpponentAgent
    weight: float

    @property
    def pair_identity(self) -> dict[str, Any]:
        return self.agent.pair_identity


class OpponentSampler:
    """Deterministic weighted sampler over a fixed opponent set."""

    def __init__(self, name: str, entries: list[tuple[str, float]], *, seed: int = 0):
        if not entries:
            raise ValueError("opponent sampler needs at least one entry")
        self.name = name
        self.entries = [(str(raw), float(weight)) for raw, weight in entries]
        self.agents = [OpponentAgent.from_str(raw) for raw, _ in self.entries]
        self.weights = [weight for _, weight in self.entries]
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("opponent weights must sum to a positive number")
        self.normalized = [weight / total for weight in self.weights]
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def labels(self) -> list[str]:
        return [agent.label for agent in self.agents]

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(self.seed if seed is None else seed)

    def materialize_all(self) -> None:
        """Load every lazy opponent without advancing the sampler cursor."""

        for agent in self.agents:
            agent.materialize()

    def sample(self) -> SampledOpponent:
        threshold = self._rng.random()
        cumulative = 0.0
        selected = len(self.agents) - 1
        for index, weight in enumerate(self.normalized):
            cumulative += weight
            if threshold <= cumulative:
                selected = index
                break
        return SampledOpponent(
            index=selected, agent=self.agents[selected], weight=self.normalized[selected]
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the exact deterministic cursor plus immutable sampler identity."""

        return {
            "name": self.name,
            "seed": self.seed,
            "labels": list(self.labels),
            "weights": list(self.weights),
            "normalized": list(self.normalized),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor only when its frozen sampler identity still matches."""

        expected = {
            "name": self.name,
            "seed": self.seed,
            "labels": list(self.labels),
            "weights": list(self.weights),
            "normalized": list(self.normalized),
        }
        if not isinstance(state, Mapping):
            raise ValueError("opponent sampler state must be a mapping")
        observed = {key: state.get(key) for key in expected}
        if observed != expected:
            raise ValueError("opponent sampler state identity is incompatible")
        rng_state = state.get("rng_state")
        if rng_state is None:
            raise ValueError("opponent sampler state is missing its RNG cursor")
        self._rng.setstate(rng_state)


# Named opponent sets. RL0 = simple baseline mix; RL4+ = frozen mixture/league.
OPPONENT_SETS: dict[str, list[tuple[str, float]]] = {
    "simple_baseline_mix": [("first-legal", 1.0), ("v2.4", 1.0)],
    "starmie_d1_health_v0": [
        ("binding:starmie-first-legal-d1", 0.50),
        ("binding:starmie-v24-d1", 0.50),
    ],
    "grimmsnarl_d2_health_v0": [
        ("binding:grimmsnarl-first-legal-d2", 0.50),
        ("binding:grimmsnarl-v0-d2", 0.50),
    ],
    "starmie_league_initial_v0": [
        ("binding:starmie-r7-d1", 0.25),
        ("binding:starmie-r23-d1", 0.15),
        ("binding:starmie-submitted-6779-d0", 0.15),
        ("reference_abomasnow", 0.0375),
        ("reference_lucario", 0.0375),
        ("reference_iono", 0.0375),
        ("reference_dragapult", 0.0375),
        ("binding:starmie-historical-legacy-d1", 0.10),
        ("binding:starmie-mirror-g00-d1", 0.10),
        ("binding:starmie-exploiter-d1", 0.05),
        ("binding:starmie-hard-extreme-d1", 0.05),
    ],
    "grimmsnarl_league_initial_v0": [
        ("binding:grimmsnarl-v0-d2", 0.20),
        ("binding:grimmsnarl-mirror-g00-d2", 0.15),
        ("binding:starmie-r7-d1", 0.20),
        ("binding:starmie-submitted-6779-d0", 0.10),
        ("reference_abomasnow", 0.0375),
        ("reference_lucario", 0.0375),
        ("reference_iono", 0.0375),
        ("reference_dragapult", 0.0375),
        ("binding:starmie-historical-legacy-d1", 0.10),
        ("binding:starmie-exploiter-d1", 0.05),
        ("binding:starmie-hard-extreme-d1", 0.05),
    ],
    "frozen_mixture": [
        ("first-legal", 1.0),
        ("v2.4", 1.0),
        ("legacy-v2", 0.5),
        ("random", 0.25),
    ],
    "reference_weighted_arena_mix": [
        ("reference_iono", 0.18),
        ("reference_abomasnow", 0.12),
        ("reference_dragapult", 0.10),
        ("reference_lucario", 0.10),
        ("first-legal", 0.15),
        ("v4", 0.15),
        ("v2.4", 0.12),
        ("legacy-v2", 0.06),
        ("random", 0.02),
    ],
    # RL4 arm C (opponent diversity): a FLATTER distribution over the same nine
    # opponents so the policy cannot memorize the exact scrim proportions of the
    # weighted mix. Combine with per-chunk seed rotation in the orchestrator.
    "reference_weighted_arena_mix_wide": [
        ("reference_iono", 0.13),
        ("reference_abomasnow", 0.12),
        ("reference_dragapult", 0.12),
        ("reference_lucario", 0.12),
        ("first-legal", 0.12),
        ("v4", 0.12),
        ("v2.4", 0.12),
        ("legacy-v2", 0.10),
        ("random", 0.05),
    ],
    # RL4 arm D (hard arena shocks): high-pressure blocks inserted periodically as
    # stress inoculation (NOT constant from step 0). Built only from opponents that
    # resolve through arena specs; top-player snapshots / exploiters are not wired
    # into parse_agent_spec, so they are a documented gap, not silently included.
    "reference_block_shock": [
        ("reference_iono", 0.34),
        ("reference_abomasnow", 0.22),
        ("reference_dragapult", 0.22),
        ("reference_lucario", 0.22),
    ],
    "reference_iono_heavy_shock": [
        ("reference_iono", 0.60),
        ("reference_abomasnow", 0.14),
        ("reference_dragapult", 0.13),
        ("reference_lucario", 0.13),
    ],
    "deck_identity_eval_smoke": [
        ("reference_iono", 1.0),
        ("reference_abomasnow", 1.0),
        ("reference_dragapult", 1.0),
        ("reference_lucario", 1.0),
        ("top_player_snapshot_01", 1.0),
        ("top_player_snapshot_02", 1.0),
        ("reference_iono_on_topplayer_dragapult_ex", 1.0),
    ],
}


def make_opponent_sampler(name: str, *, seed: int = 0) -> OpponentSampler:
    if name not in OPPONENT_SETS:
        raise ValueError(
            f"unknown opponent sampler {name!r}; known: {sorted(OPPONENT_SETS)}"
        )
    return OpponentSampler(name, OPPONENT_SETS[name], seed=seed)


def make_eval_suite(name: str | None = None) -> list[OpponentAgent]:
    """Fixed evaluation opponents reported per-bucket (no training on these)."""
    if name in OPPONENT_SETS:
        seen: set[str] = set()
        suite: list[OpponentAgent] = []
        for raw, _weight in OPPONENT_SETS[name]:
            if raw in seen:
                continue
            seen.add(raw)
            suite.append(OpponentAgent.from_str(raw))
        return suite
    return [OpponentAgent.from_str(raw) for raw in ("first-legal", "v2.4")]
