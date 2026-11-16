"""Kaggle replay ingestion: read downloaded episode JSONs into DeepMew artifacts.

Offline infrastructure only. Extracts public opponent decklists from the initial
visualize payload and (best-effort) maps steps into `deepmew.replay_action_row.v1`
rows. Never loaded by the eval/runtime submission path; never writes root
`main.py`/`deck.csv`/`cg/`.
"""

from deepmew.kaggle_replays.decks import (
    DeckExtractionError,
    PlayerDeck,
    extract_decklists,
    find_visualize_players,
    write_deck_artifacts,
)
from deepmew.kaggle_replays.download import (
    DEFAULT_COMPETITION,
    DEFAULT_LOG_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPLAY_ROOT,
    KAGGLE_EXE_ENV_VAR,
    DownloadOptions,
    KaggleDownloadError,
    KaggleDownloadResult,
    KaggleExecutable,
    download_kaggle_replays,
    kaggle_config_path,
    load_configured_kaggle_exe,
    log_output_path,
    parse_episode_file,
    replay_output_path,
    resolve_kaggle_executable,
)
from deepmew.kaggle_replays.corpus import (
    CorpusOptions,
    CorpusResult,
    archetype_hint,
    report_kaggle_replay_corpus,
)
from deepmew.kaggle_replays.episode import (
    EpisodeMetadata,
    KaggleEpisodeError,
    episode_metadata,
    load_episode,
    result_from_rewards,
)
from deepmew.kaggle_replays.episode_sources import (
    EpisodeSourceError,
    EpisodeSourceResult,
    build_episode_source,
    extract_episode_ids_from_text,
    scan_replay_dir,
)
from deepmew.kaggle_replays.ingest import (
    KaggleIngestResult,
    build_kaggle_action_rows,
    ingest_episode,
)
from deepmew.kaggle_replays.replay_queue import (
    QueueOptions,
    QueueResult,
    ReplayQueueError,
    mine_replay_queue,
)
from deepmew.kaggle_replays.scout import (
    KaggleScoutError,
    LeaderboardRow,
    ScoutOptions,
    ScoutResult,
    parse_episode_ids_from_episodes_output,
    parse_leaderboard,
    parse_rank_ranges,
    parse_submission_ids,
    scout_kaggle_episodes,
    select_rows_evenly,
)

__all__ = [
    "CorpusOptions",
    "CorpusResult",
    "DeckExtractionError",
    "DEFAULT_COMPETITION",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_REPLAY_ROOT",
    "KAGGLE_EXE_ENV_VAR",
    "KaggleExecutable",
    "kaggle_config_path",
    "load_configured_kaggle_exe",
    "resolve_kaggle_executable",
    "DownloadOptions",
    "EpisodeMetadata",
    "EpisodeSourceError",
    "EpisodeSourceResult",
    "KaggleDownloadError",
    "KaggleDownloadResult",
    "KaggleEpisodeError",
    "KaggleIngestResult",
    "KaggleScoutError",
    "LeaderboardRow",
    "PlayerDeck",
    "QueueOptions",
    "QueueResult",
    "ReplayQueueError",
    "mine_replay_queue",
    "ScoutOptions",
    "ScoutResult",
    "archetype_hint",
    "build_episode_source",
    "build_kaggle_action_rows",
    "download_kaggle_replays",
    "episode_metadata",
    "extract_decklists",
    "extract_episode_ids_from_text",
    "find_visualize_players",
    "ingest_episode",
    "log_output_path",
    "load_episode",
    "parse_episode_ids_from_episodes_output",
    "parse_episode_file",
    "parse_leaderboard",
    "parse_rank_ranges",
    "parse_submission_ids",
    "report_kaggle_replay_corpus",
    "replay_output_path",
    "result_from_rewards",
    "scan_replay_dir",
    "scout_kaggle_episodes",
    "select_rows_evenly",
    "write_deck_artifacts",
]
