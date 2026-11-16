"""DeepMew replay table layer.

Ingest completed opponent-mixture run artifacts into normalized, versioned
per-game replay rows suitable for later replay priors, value-model training, and
shallow-search data plumbing. This is infrastructure only; it does not change
gameplay policy, defaults, or any root artifact.
"""

from deepmew.replay.action_ingest import (
    ActionIngestResult,
    ActionRowResult,
    ReplayActionIngestError,
    build_action_row,
    iter_action_rows,
)
from deepmew.replay.action_ingest import ingest_frames as ingest_action_frames
from deepmew.replay.action_schema import (
    ACTION_ROW_FIELDS,
    ReplayActionSchemaError,
    validate_action_row,
)
from deepmew.replay.action_schema import SCHEMA_VERSION as ACTION_SCHEMA_VERSION
from deepmew.replay.action_table import (
    ActionTableExport,
    build_action_table,
    read_action_table,
)
from deepmew.replay.decision_ingest import (
    DecisionIngestResult,
    ReplayDecisionIngestError,
    deck_metadata_from_path,
    ingest_frames,
    resolve_frames_path,
)
from deepmew.replay.decision_schema import (
    DECISION_ROW_FIELDS,
    ReplayDecisionSchemaError,
    validate_decision_row,
)
from deepmew.replay.decision_schema import SCHEMA_VERSION as DECISION_SCHEMA_VERSION
from deepmew.replay.decision_table import (
    DECISION_TABLE_KIND,
    DecisionTableExport,
    decision_table_csv_text,
    decision_table_summary,
    read_decision_replay_table,
    validate_decision_rows,
    write_decision_replay_table,
)
from deepmew.replay.ingest import (
    IngestResult,
    ReplayIngestError,
    ingest_matchups_csv,
    ingest_results_json,
    ingest_run_dir,
)
from deepmew.replay.information_boundary import (
    InformationBoundary,
    InformationMode,
    merge_visualizer_private_state,
    redact_public_observation,
    reject_privileged_rows,
)
from deepmew.replay.prior_build import (
    PriorBuildResult,
    ReplayPriorBuildError,
    build_replay_prior,
    read_decision_rows,
)
from deepmew.replay.prior_schema import (
    ReplayPriorSchemaError,
    validate_prior_artifact,
)
from deepmew.replay.prior_schema import SCHEMA_VERSION as PRIOR_SCHEMA_VERSION
from deepmew.replay.prior_table import (
    PriorExport,
    read_replay_prior,
    write_replay_prior,
)
from deepmew.replay.schema import (
    MATCHUP_IDENTITY_FIELDS,
    PROVENANCE_FIELDS,
    REPLAY_ROW_FIELDS,
    SCHEMA_VERSION,
    ReplaySchemaError,
    make_replay_row,
    validate_matchup_row,
    validate_replay_row,
)
from deepmew.replay.table import (
    REPLAY_TABLE_KIND,
    ReplayTableExport,
    read_replay_table,
    table_csv_text,
    table_summary,
    validate_rows,
    write_replay_table,
)

__all__ = [
    "ACTION_ROW_FIELDS",
    "ACTION_SCHEMA_VERSION",
    "ActionIngestResult",
    "ActionRowResult",
    "ActionTableExport",
    "DECISION_ROW_FIELDS",
    "DECISION_SCHEMA_VERSION",
    "DECISION_TABLE_KIND",
    "DecisionIngestResult",
    "DecisionTableExport",
    "PRIOR_SCHEMA_VERSION",
    "PriorBuildResult",
    "PriorExport",
    "ReplayActionIngestError",
    "ReplayActionSchemaError",
    "ReplayPriorBuildError",
    "ReplayPriorSchemaError",
    "build_action_row",
    "build_action_table",
    "build_replay_prior",
    "ingest_action_frames",
    "iter_action_rows",
    "read_action_table",
    "read_decision_rows",
    "read_replay_prior",
    "validate_action_row",
    "validate_prior_artifact",
    "write_replay_prior",
    "IngestResult",
    "InformationBoundary",
    "InformationMode",
    "MATCHUP_IDENTITY_FIELDS",
    "PROVENANCE_FIELDS",
    "REPLAY_ROW_FIELDS",
    "REPLAY_TABLE_KIND",
    "ReplayDecisionIngestError",
    "ReplayDecisionSchemaError",
    "ReplayIngestError",
    "ReplaySchemaError",
    "ReplayTableExport",
    "SCHEMA_VERSION",
    "decision_table_csv_text",
    "decision_table_summary",
    "deck_metadata_from_path",
    "ingest_frames",
    "ingest_matchups_csv",
    "ingest_results_json",
    "ingest_run_dir",
    "make_replay_row",
    "merge_visualizer_private_state",
    "redact_public_observation",
    "read_decision_replay_table",
    "read_replay_table",
    "resolve_frames_path",
    "reject_privileged_rows",
    "table_csv_text",
    "table_summary",
    "validate_decision_row",
    "validate_decision_rows",
    "validate_matchup_row",
    "validate_replay_row",
    "validate_rows",
    "write_decision_replay_table",
    "write_replay_table",
]
