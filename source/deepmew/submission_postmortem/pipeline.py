"""End-to-end orchestration for a hash-bound submission postmortem."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from deepmew.options.identity import card_name_for_id
from deepmew.deck.tables import attack_table, card_table, type_name
from deepmew.submission_postmortem.catalog import (
    CatalogFetchResult,
    ReplayCacheRecord,
    SubmissionCatalogError,
    cache_submission_replays,
    catalog_from_replay_dir,
    catalog_sha256,
    fetch_submission_catalog,
    filter_catalog_episodes,
    stable_sort_episodes,
    validate_replay_file,
)
from deepmew.submission_postmortem.io import write_csv, write_json, write_jsonl, write_text
from deepmew.submission_postmortem.models import CatalogEpisode, CatalogQuery, ReplayDecision, ReplayEpisode, SCHEMA_VERSION
from deepmew.submission_postmortem.replay import ReplayExtractionError, load_replay
from deepmew.submission_postmortem.scoring import (
    DEFAULT_BINDINGS_PATH,
    ExactRuntimeActorScorer,
    RuntimeBinding,
    SubmissionBindingError,
    resolve_runtime_binding,
    score_episodes_cached,
    sha256_file,
)
from deepmew.submission_postmortem.signals import (
    aggregate_loss_signals,
    aggregate_matchups,
    aggregate_timeline,
    build_decision_signal,
    extract_state_snapshot,
    summarize_episode,
)


DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_OUTPUT_ROOT = Path(".research/submission_postmortems")
DEFAULT_CACHE_ROOT = Path(".research/submission_postmortem/cache")
PIPELINE_SCHEMA_VERSION = "deepmew.submission_postmortem.run.v0"
FORBIDDEN_OUTPUT_TOKENS = (
    '"search_begin_input"',
    '"deckOrder"',
    '"trueDeckOrder"',
    '"looking"',
)


class PostmortemPipelineError(RuntimeError):
    """Raised when a fail-closed postmortem gate does not pass."""


@dataclass(frozen=True)
class PostmortemOptions:
    submission_id: str
    competition: str = DEFAULT_COMPETITION
    query: CatalogQuery = CatalogQuery()
    replay_dir: Path | None = None
    online_catalog: bool = False
    download_missing: bool = True
    allow_cli_catalog_fallback: bool = False
    kaggle_executable: str | None = None
    bundle_dir: Path | None = None
    bindings_path: Path = DEFAULT_BINDINGS_PATH
    output_dir: Path | None = None
    output_root: Path = DEFAULT_OUTPUT_ROOT
    cache_root: Path = DEFAULT_CACHE_ROOT
    download_workers: int = 4
    parse_workers: int = 4
    encoder_batch_size: int = 8192
    refresh_downloads: bool = False
    refresh_scores: bool = False
    skip_scoring: bool = False
    result_filter: str = "all"
    strict: bool = True
    command: str = ""


@dataclass(frozen=True)
class PostmortemResult:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    summary_path: Path
    summary: Mapping[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_output_dir(options: PostmortemOptions) -> Path:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")

    return Path(options.output_root) / f"submission-{options.submission_id}" / timestamp


def _validate_options(options: PostmortemOptions) -> None:
    if not options.submission_id.isdigit():
        raise PostmortemPipelineError("submission_id must contain digits")
    if options.download_workers <= 0 or options.parse_workers <= 0 or options.encoder_batch_size <= 0:
        raise PostmortemPipelineError("worker and batch sizes must be positive")
    if options.result_filter not in {"all", "win", "loss", "draw"}:
        raise PostmortemPipelineError("result_filter must be all, win, loss, or draw")
    if options.replay_dir is not None and not Path(options.replay_dir).is_dir():
        raise PostmortemPipelineError(f"replay_dir not found: {options.replay_dir}")


def _phase_checkpoint(
    output_dir: Path,
    *,
    options: PostmortemOptions,
    phase: str,
    status: str = "running",
    details: Mapping[str, Any] | None = None,
) -> None:
    write_json(
        output_dir / "checkpoint.json",
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "submission_id": options.submission_id,
            "phase": phase,
            "status": status,
            "updated_at_utc": _utc_now().isoformat(),
            "details": dict(details or {}),
        },
    )


def _catalog_without_opponent_filters(query: CatalogQuery) -> CatalogQuery:
    return replace(
        query,
        opponent_terms=(),
        opponent_team_ids=frozenset(),
        opponent_submission_ids=frozenset(),
    )


def _fetch_catalog(options: PostmortemOptions) -> tuple[CatalogFetchResult, tuple[CatalogEpisode, ...]]:
    offline_only = options.replay_dir is not None and not options.online_catalog
    if offline_only:
        catalog = catalog_from_replay_dir(options.replay_dir, submission_id=options.submission_id)
        query = _catalog_without_opponent_filters(options.query)
    else:
        catalog = fetch_submission_catalog(
            options.submission_id,
            kaggle_executable=options.kaggle_executable,
            offline_replay_dir=options.replay_dir,
            allow_cli_fallback=options.allow_cli_catalog_fallback,
        )
        query = options.query
    selected = filter_catalog_episodes(
        catalog.episodes,
        submission_id=options.submission_id,
        query=query,
    )

    return catalog, selected


def _find_offline_replay(root: Path, episode_id: str) -> Path | None:
    direct = root / f"episode-{episode_id}-replay.json"
    if direct.is_file():
        return direct
    matches = list(root.rglob(f"episode-{episode_id}-replay.json"))

    return matches[0] if len(matches) == 1 else None


def _replay_records(
    options: PostmortemOptions,
    catalog: CatalogFetchResult,
    episodes: Sequence[CatalogEpisode],
) -> tuple[ReplayCacheRecord, ...]:
    if options.replay_dir is not None and not options.download_missing:
        records = []
        for episode in episodes:
            path = _find_offline_replay(Path(options.replay_dir), episode.episode_id)
            if path is None:
                records.append(
                    ReplayCacheRecord(
                        episode_id=episode.episode_id,
                        path=str(Path(options.replay_dir) / f"episode-{episode.episode_id}-replay.json"),
                        status="failed",
                        error="selected offline replay is missing or ambiguous",
                    )
                )
            else:
                records.append(validate_replay_file(path, expected_episode_id=episode.episode_id))

        return tuple(records)

    kaggle_executable = catalog.kaggle_executable if options.download_missing else None
    if options.download_missing and kaggle_executable is None:
        kaggle_executable = options.kaggle_executable
    cached = cache_submission_replays(
        episodes,
        cache_dir=Path(options.cache_root) / "replays" / f"submission-{options.submission_id}",
        offline_replay_dir=options.replay_dir,
        kaggle_executable=kaggle_executable,
        workers=options.download_workers,
        force=options.refresh_downloads,
    )

    return cached.records


def _load_one_replay(
    record: ReplayCacheRecord,
    *,
    options: PostmortemOptions,
    catalog_episode: CatalogEpisode,
    binding: RuntimeBinding | None,
) -> ReplayEpisode:
    return load_replay(
        record.path,
        submission_id=options.submission_id,
        catalog_episode=catalog_episode,
        submitted_deck_hash=None if binding is None else binding.submitted_deck_hash,
    )


def _parse_replays(
    records: Sequence[ReplayCacheRecord],
    *,
    options: PostmortemOptions,
    selected: Sequence[CatalogEpisode],
    binding: RuntimeBinding | None,
) -> tuple[list[ReplayEpisode], list[dict[str, Any]]]:
    catalog_by_id = {episode.episode_id: episode for episode in selected}
    episodes: dict[str, ReplayEpisode] = {}
    errors: list[dict[str, Any]] = []
    valid_records = [record for record in records if record.is_valid]
    for record in records:
        if not record.is_valid:
            errors.append({"episode_id": record.episode_id, "phase": "replay_cache", "error": record.error})
    with ThreadPoolExecutor(
        max_workers=min(options.parse_workers, max(1, len(valid_records))),
        thread_name_prefix="postmortem-parse",
    ) as executor:
        futures = {
            executor.submit(
                _load_one_replay,
                record,
                options=options,
                catalog_episode=catalog_by_id[record.episode_id],
                binding=binding,
            ): record
            for record in valid_records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                episode = future.result()
            except (ReplayExtractionError, OSError, ValueError) as exc:
                errors.append({"episode_id": record.episode_id, "phase": "replay_parse", "error": str(exc)})
                continue
            episodes[episode.episode_id] = episode

    ordered = [episodes[item.episode_id] for item in stable_sort_episodes(selected) if item.episode_id in episodes]

    return ordered, errors


def _perspective_result(episode: ReplayEpisode, seat: int) -> str:
    reward = episode.rewards[seat] if seat < len(episode.rewards) else None
    if reward is None:
        return "unknown"
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"

    return "draw"


def _catalog_agent(episode: CatalogEpisode, seat: int) -> Any:
    for agent in episode.agents:
        if agent.index == seat:
            return agent

    return None


def _filtered_actor_seats(
    replay: ReplayEpisode,
    catalog_episode: CatalogEpisode,
    *,
    options: PostmortemOptions,
) -> tuple[int, ...]:
    retained = []
    terms = tuple(term.casefold() for term in options.query.opponent_terms if term)
    for actor_seat in replay.actor_seats:
        if options.result_filter != "all" and _perspective_result(replay, actor_seat) != options.result_filter:
            continue
        opponent_seats = [seat for seat in range(len(replay.team_names)) if seat != actor_seat]
        if terms and not any(any(term in replay.team_names[seat].casefold() for term in terms) for seat in opponent_seats):
            continue
        if options.query.opponent_team_ids:
            candidates = [_catalog_agent(catalog_episode, seat) for seat in opponent_seats]
            if not any(agent is not None and agent.team_id in options.query.opponent_team_ids for agent in candidates):
                continue
        if options.query.opponent_submission_ids:
            candidates = [_catalog_agent(catalog_episode, seat) for seat in opponent_seats]
            if not any(agent is not None and agent.submission_id in options.query.opponent_submission_ids for agent in candidates):
                continue
        retained.append(actor_seat)

    return tuple(retained)


def _filter_perspectives(
    episodes: Sequence[ReplayEpisode],
    *,
    selected: Sequence[CatalogEpisode],
    options: PostmortemOptions,
) -> list[ReplayEpisode]:
    catalog_by_id = {episode.episode_id: episode for episode in selected}
    filtered = []
    for episode in episodes:
        seats = _filtered_actor_seats(episode, catalog_by_id[episode.episode_id], options=options)
        if seats:
            filtered.append(replace(episode, actor_seats=seats))

    return filtered


def _validate_safety(episodes: Sequence[ReplayEpisode], *, binding: RuntimeBinding | None) -> None:
    failures = []
    for episode in episodes:
        if episode.alignment_status != "structurally_verified_unadmitted" or episode.errors:
            failures.append(f"{episode.episode_id}: alignment={episode.alignment_status} errors={list(episode.errors)}")
        if episode.statuses != ("DONE", "DONE"):
            failures.append(f"{episode.episode_id}: terminal statuses={episode.statuses}")
        if not episode.actor_seats:
            failures.append(f"{episode.episode_id}: actor seat unresolved")
        if binding is not None and episode.decks:
            mismatched = [
                seat
                for seat in episode.actor_seats
                if not any(deck.seat == seat and deck.stable_hash == binding.submitted_deck_hash for deck in episode.decks)
            ]
            if mismatched:
                failures.append(f"{episode.episode_id}: catalog actor seat deck mismatch={mismatched}")
    if failures:
        raise PostmortemPipelineError("safety gate failed:\n" + "\n".join(failures))


def _annotate_option(option: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(option)
    for id_field, name_field in (
        ("source_card_id", "source_card_name"),
        ("target_card_id", "target_card_name"),
    ):
        value = result.get(id_field)
        if isinstance(value, int):
            result[name_field] = card_name_for_id(value)

    return result


def _selected_semantics(
    decision: ReplayDecision,
    scorer: ExactRuntimeActorScorer | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    select = decision.observation.get("select")
    if not isinstance(select, dict):
        return [], []
    if scorer is not None:
        legal_table = [
            _annotate_option(option)
            for option in scorer.runtime.legal_action_table(dict(decision.observation), select)
        ]
    else:
        options = select.get("option")
        legal_table = [dict(option) for option in options] if isinstance(options, list) else []
    selected = [legal_table[index] for index in decision.action if 0 <= index < len(legal_table)]

    return legal_table, selected


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rating_summary(score: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if score is None:
        return None
    prefixes = score.get("prefix_ratings")
    prefix_rows = [row for row in prefixes if isinstance(row, Mapping)] if isinstance(prefixes, list) else []
    entropies = [float(row["entropy"]) for row in prefix_rows if isinstance(row.get("entropy"), (int, float))]
    probabilities = [
        float(row["chosen_probability"])
        for row in prefix_rows
        if isinstance(row.get("chosen_probability"), (int, float))
    ]
    ranks = [int(row["chosen_rank"]) for row in prefix_rows if isinstance(row.get("chosen_rank"), int)]
    margins = [
        float(row["chosen_to_top_margin"])
        for row in prefix_rows
        if isinstance(row.get("chosen_to_top_margin"), (int, float))
    ]

    return {
        "status": score.get("actor_rating_status"),
        "binding_status": "exact_hash_bound_runtime",
        "value_status": score.get("value_rating_status"),
        "value": score.get("value"),
        "entropy": _mean(entropies),
        "chosen_rank": max(ranks) if ranks else None,
        "chosen_probability": math.exp(float(score["sequence_log_probability"]))
        if isinstance(score.get("sequence_log_probability"), (int, float))
        else _mean(probabilities),
        "margin_to_best": min(margins) if margins else None,
        "sequence_log_probability": score.get("sequence_log_probability"),
        "predicted_matches_action": score.get("model_action_agreement"),
        "action_agreement": score.get("model_action_agreement"),
    }


def _decision_signals(
    episodes: Sequence[ReplayEpisode],
    *,
    scorer: ExactRuntimeActorScorer | None,
    scores_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    results = []
    for episode in episodes:
        score_by_id = {
            str(row.get("decision_id")): row
            for row in scores_by_episode.get(episode.episode_id, ())
        }
        actor_seats = set(episode.actor_seats)
        for decision in episode.decisions:
            role = "actor" if decision.seat in actor_seats else "opponent"
            legal_table, selected = _selected_semantics(decision, scorer)
            score = score_by_id.get(decision.decision_id) if role == "actor" else None
            results.append(
                build_decision_signal(
                    decision,
                    legal_table=legal_table,
                    selected_options=selected,
                    rating_summary=_rating_summary(score),
                    role=role,
                )
            )

    return results


def _terminal_state_snapshot(episode: ReplayEpisode, seat: int) -> dict[str, Any] | None:
    """Project the terminal actor view and immediately reduce it to safe counts."""

    from deepmew.submission_postmortem.replay import project_actor_observation

    try:
        payload = json.loads(Path(episode.source_path).read_text(encoding="utf-8"))
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return None
        observation = None
        terminal_step = None
        for step_index in range(len(steps) - 1, -1, -1):
            step = steps[step_index]
            if not isinstance(step, list) or seat >= len(step) or not isinstance(step[seat], dict):
                continue
            candidate = step[seat].get("observation")
            if isinstance(candidate, dict):
                observation = project_actor_observation(candidate)
                terminal_step = step_index
                break
        if observation is None or terminal_step is None:
            return None
        synthetic = ReplayDecision(
            decision_id=f"{episode.episode_id}:terminal:s{seat}",
            episode_id=episode.episode_id,
            request_step=terminal_step,
            action_step=terminal_step,
            seat=seat,
            action=(),
            observation=observation,
        )

        return extract_state_snapshot(synthetic)
    except (OSError, json.JSONDecodeError, ReplayExtractionError, ValueError):
        return None


def _card_strength(card: Any, attacks: Mapping[int, Any]) -> tuple[int, int, int, str]:
    damage = 0
    for attack_id in getattr(card, "attacks", ()) or ():
        attack = attacks.get(int(attack_id))
        if attack is not None:
            damage = max(damage, int(getattr(attack, "damage", 0) or 0))
    special = 2 if bool(getattr(card, "megaEx", False)) else 1 if bool(getattr(card, "ex", False)) else 0

    return special, damage, int(getattr(card, "hp", 0) or 0), str(getattr(card, "name", ""))


def _deck_shell(episode: ReplayEpisode, opponent_seat: int) -> str | None:
    deck = next((item for item in episode.decks if item.seat == opponent_seat), None)
    if deck is None:
        return None
    cards = card_table()
    attacks = attack_table()
    candidates = []
    for card_id, count in deck.card_counts.items():
        card = cards.get(card_id)
        if card is None or type_name(card) != "POKEMON":
            continue
        candidates.append((_card_strength(card, attacks), int(count), str(card.name)))
    candidates.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[1], item[2]), reverse=True)
    names = []
    for _strength, _count, name in candidates:
        if name not in names:
            names.append(name)
        if len(names) == 2:
            break

    return " / ".join(names) if names else None


def _focal_evolution_ids(binding: RuntimeBinding | None) -> tuple[int, ...]:
    if binding is None:
        return ()
    cards = card_table()
    deck_ids = [int(line.strip()) for line in binding.deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = []
    for card_id in sorted(set(deck_ids)):
        card = cards.get(card_id)
        if card is None or type_name(card) != "POKEMON":
            continue
        strength = _card_strength(card, attack_table())
        if strength[0] > 0 or bool(getattr(card, "stage1", False)) or bool(getattr(card, "stage2", False)):
            candidates.append((strength, card_id))
    if not candidates:
        return ()
    best_special = max(item[0][0] for item in candidates)
    if best_special:
        return tuple(sorted(card_id for strength, card_id in candidates if strength[0] == best_special))
    candidates.sort(reverse=True)

    return (candidates[0][1],)


def _episode_summaries(
    episodes: Sequence[ReplayEpisode],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    binding: RuntimeBinding | None,
    ordinal_by_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    focal_ids = _focal_evolution_ids(binding)
    summaries = []
    for episode in episodes:
        for actor_seat in episode.actor_seats:
            opponent_seat = 1 - actor_seat
            summary = summarize_episode(
                episode,
                actor_seat=actor_seat,
                decision_signals=decision_rows,
                ordinal=ordinal_by_id.get(episode.episode_id),
                opponent_label=_deck_shell(episode, opponent_seat),
                terminal_snapshot=_terminal_state_snapshot(episode, actor_seat),
                focal_evolution_card_ids=focal_ids,
            )
            summary["excluded_from_opponent_attribution"] = bool(episode.self_mirror)
            summary["source_sha256"] = episode.source_sha256
            summaries.append(summary)

    return summaries


def _flatten_aggregate(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = payload.get(key)

    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _opponent_action_rows(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result_by_perspective = {
        (str(summary.get("episode_id")), int(summary.get("opponent_seat", -1))): str(summary.get("result"))
        for summary in episode_summaries
        if not summary.get("excluded_from_opponent_attribution")
    }
    games_by_signal: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in decision_rows:
        if row.get("role") != "opponent":
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or turn <= 0:
            continue
        key = (str(row.get("episode_id")), int(row.get("seat", -1)))
        result = result_by_perspective.get(key)
        if result not in {"win", "loss", "draw"}:
            continue
        episode_id = key[0]
        for option in row.get("selected_options") or ():
            if not isinstance(option, Mapping):
                continue
            category = str(option.get("category") or "UNKNOWN")
            signals = [("category", category)]
            if option.get("source_card_name"):
                signals.append(("card", str(option["source_card_name"])))
            elif option.get("source_card_id") is not None:
                signals.append(("card_id", str(option["source_card_id"])))
            if category == "ATTACK" and option.get("attack_id") is not None:
                signals.append(("attack_id", str(option["attack_id"])))
            for signal in signals:
                games_by_signal[signal][result].add(episode_id)
    total_wins = sum(summary.get("result") == "win" and not summary.get("excluded_from_opponent_attribution") for summary in episode_summaries)
    total_losses = sum(summary.get("result") == "loss" and not summary.get("excluded_from_opponent_attribution") for summary in episode_summaries)
    rows = []
    for (kind, signal), outcomes in games_by_signal.items():
        wins = len(outcomes.get("win", set()))
        losses = len(outcomes.get("loss", set()))
        draws = len(outcomes.get("draw", set()))
        games = wins + losses + draws
        rows.append(
            {
                "signal_kind": kind,
                "signal": signal,
                "games_present": games,
                "wins_present": wins,
                "losses_present": losses,
                "draws_present": draws,
                "loss_rate_when_present": losses / (wins + losses) if wins + losses else None,
                "loss_presence_rate": losses / total_losses if total_losses else None,
                "win_presence_rate": wins / total_wins if total_wins else None,
                "loss_minus_win_presence": (losses / total_losses if total_losses else 0.0)
                - (wins / total_wins if total_wins else 0.0),
                "provenance": "public_event",
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["losses_present"]),
            -float(row["loss_minus_win_presence"]),
            -int(row["games_present"]),
            str(row["signal"]),
        )
    )

    return rows


def _rating_outcome_rows(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result_by_perspective = {
        (str(summary.get("episode_id")), int(summary.get("actor_seat", -1))): str(summary.get("result"))
        for summary in episode_summaries
    }
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: Counter[tuple[str, str]] = Counter()
    for row in decision_rows:
        if row.get("role") != "actor":
            continue
        outcome = result_by_perspective.get((str(row.get("episode_id")), int(row.get("seat", -1))))
        if outcome is None:
            continue
        rating = row.get("rating")
        if not isinstance(rating, Mapping):
            continue
        counts[(outcome, "decisions")] += 1
        for metric in ("entropy", "chosen_probability", "margin_to_best", "sequence_log_probability"):
            value = rating.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                groups[(outcome, metric)].append(float(value))
    rows = []
    for outcome in ("win", "loss", "draw"):
        if counts[(outcome, "decisions")] == 0:
            continue
        row: dict[str, Any] = {"result": outcome, "decisions": counts[(outcome, "decisions")], "provenance": "policy_rescore"}
        for metric in ("entropy", "chosen_probability", "margin_to_best", "sequence_log_probability"):
            values = groups.get((outcome, metric), [])
            row[f"mean_{metric}"] = _mean(values)
            row[f"median_{metric}"] = statistics.median(values) if values else None
        rows.append(row)

    return rows


def _position_rows(episode_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for position in ("first", "second", "unavailable"):
        values = [
            summary
            for summary in episode_summaries
            if str(summary.get("first_second_proxy") or "unavailable") == position
        ]
        if not values:
            continue
        wins = sum(value.get("result") == "win" for value in values)
        losses = sum(value.get("result") == "loss" for value in values)
        draws = sum(value.get("result") == "draw" for value in values)
        rows.append(
            {
                "position": position,
                "games": len(values),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "decisive_win_rate": wins / (wins + losses) if wins + losses else None,
                "provenance": "actor_visible",
            }
        )

    return rows


def _opponent_playstyle_rows(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_by_perspective = {
        (str(summary.get("episode_id")), int(summary.get("opponent_seat", -1))): str(summary.get("result"))
        for summary in episode_summaries
        if not summary.get("excluded_from_opponent_attribution")
    }
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        if row.get("role") != "opponent":
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or turn <= 0:
            continue
        grouped[(str(row.get("episode_id")), int(row.get("seat", -1)))].append(row)

    game_rows = []
    for key, rows in grouped.items():
        result = result_by_perspective.get(key)
        if result is None:
            continue
        counts: Counter[str] = Counter()
        turns = set()
        first_attack_turn = None
        for row in rows:
            turn = int(row["turn"])
            turns.add(turn)
            for category in row.get("selected_categories") or ():
                counts[str(category)] += 1
                if str(category) == "ATTACK":
                    first_attack_turn = turn if first_attack_turn is None else min(first_attack_turn, turn)
        turn_count = max(1, len(turns))
        game_rows.append(
            {
                "episode_id": key[0],
                "opponent_seat": key[1],
                "result": result,
                "battle_turn_count": len(turns),
                "decision_count": len(rows),
                "actions_per_turn": sum(counts.values()) / turn_count,
                "attacks_per_turn": counts["ATTACK"] / turn_count,
                "abilities_per_turn": counts["ABILITY"] / turn_count,
                "evolutions_per_turn": counts["EVOLVE"] / turn_count,
                "attachments_per_turn": counts["ATTACH"] / turn_count,
                "plays_per_turn": counts["PLAY"] / turn_count,
                "first_attack_turn": first_attack_turn,
                "action_counts": dict(sorted(counts.items())),
                "provenance": "public_event",
            }
        )
    metrics = (
        "battle_turn_count",
        "decision_count",
        "actions_per_turn",
        "attacks_per_turn",
        "abilities_per_turn",
        "evolutions_per_turn",
        "attachments_per_turn",
        "plays_per_turn",
        "first_attack_turn",
    )
    aggregate_rows = []
    for metric in metrics:
        win_values = [float(row[metric]) for row in game_rows if row["result"] == "win" and row.get(metric) is not None]
        loss_values = [float(row[metric]) for row in game_rows if row["result"] == "loss" and row.get(metric) is not None]
        win_mean = _mean(win_values)
        loss_mean = _mean(loss_values)
        aggregate_rows.append(
            {
                "metric": metric,
                "win_games_with_evidence": len(win_values),
                "loss_games_with_evidence": len(loss_values),
                "mean_when_actor_won": win_mean,
                "mean_when_actor_lost": loss_mean,
                "loss_minus_win_mean": loss_mean - win_mean
                if loss_mean is not None and win_mean is not None
                else None,
                "provenance": "public_event",
            }
        )

    return game_rows, aggregate_rows


def _rating_phase_rows(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result_by_perspective = {
        (str(summary.get("episode_id")), int(summary.get("actor_seat", -1))): str(summary.get("result"))
        for summary in episode_summaries
    }
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        if row.get("role") != "actor":
            continue
        turn = row.get("turn")
        if not isinstance(turn, int):
            phase = "unavailable"
        elif turn <= 0:
            phase = "setup"
        elif turn <= 4:
            phase = "early"
        elif turn <= 8:
            phase = "middle"
        else:
            phase = "late"
        result = result_by_perspective.get((str(row.get("episode_id")), int(row.get("seat", -1))))
        if result is not None:
            groups[(result, phase)].append(row)
    output = []
    for (result, phase), rows in sorted(groups.items()):
        entropies = []
        probabilities = []
        log_probabilities = []
        for row in rows:
            rating = row.get("rating") if isinstance(row.get("rating"), Mapping) else {}
            for target, key in (
                (entropies, "entropy"),
                (probabilities, "chosen_probability"),
                (log_probabilities, "sequence_log_probability"),
            ):
                value = rating.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    target.append(float(value))
        output.append(
            {
                "result": result,
                "phase": phase,
                "decisions": len(rows),
                "mean_entropy": _mean(entropies),
                "mean_chosen_probability": _mean(probabilities),
                "mean_sequence_log_probability": _mean(log_probabilities),
                "provenance": "policy_rescore",
            }
        )

    return output


def _episode_csv_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        prizes = summary.get("prizes") if isinstance(summary.get("prizes"), Mapping) else {}
        tempo = summary.get("tempo") if isinstance(summary.get("tempo"), Mapping) else {}
        actor_deck = summary.get("actor_deck") if isinstance(summary.get("actor_deck"), Mapping) else {}
        opponent_deck = summary.get("opponent_deck") if isinstance(summary.get("opponent_deck"), Mapping) else {}
        tag_evidence = summary.get("tag_evidence") if isinstance(summary.get("tag_evidence"), Mapping) else {}
        action_counts = summary.get("action_counts") if isinstance(summary.get("action_counts"), Mapping) else {}
        actor_counts = action_counts.get("actor") if isinstance(action_counts.get("actor"), Mapping) else {}
        opponent_counts = action_counts.get("opponent") if isinstance(action_counts.get("opponent"), Mapping) else {}
        row = {
            "episode_id": summary.get("episode_id"),
            "ordinal": summary.get("ordinal"),
            "actor_seat": summary.get("actor_seat"),
            "opponent_seat": summary.get("opponent_seat"),
            "result": summary.get("result"),
            "reward": summary.get("reward"),
            "actor_team": summary.get("actor_team"),
            "opponent_team": summary.get("opponent_team"),
            "opponent_shell": summary.get("opponent_label"),
            "actor_deck_hash": actor_deck.get("stable_hash"),
            "opponent_deck_hash": opponent_deck.get("stable_hash"),
            "first_second_proxy": summary.get("first_second_proxy"),
            "prizes_taken": prizes.get("taken"),
            "prizes_conceded": prizes.get("conceded"),
            "maximum_prize_lead": tempo.get("maximum_prize_lead"),
            "first_attack_turn": tempo.get("first_attack_turn"),
            "first_evolution_turn": tempo.get("first_evolution_turn"),
            "first_attachment_turn": tempo.get("first_attachment_turn"),
            "terminal_failure_class": summary.get("terminal_failure_class"),
            "active_tags": summary.get("active_tags"),
            "actor_action_counts": actor_counts,
            "opponent_action_counts": opponent_counts,
            "self_mirror": summary.get("self_mirror"),
            "excluded_from_opponent_attribution": summary.get("excluded_from_opponent_attribution"),
            "seat_binding_status": summary.get("seat_binding_status"),
            "alignment_status": summary.get("alignment_status"),
            "source_sha256": summary.get("source_sha256"),
        }
        for name, evidence in tag_evidence.items():
            if isinstance(evidence, Mapping):
                row[f"signal_{name}"] = evidence.get("value")
                row[f"signal_{name}_provenance"] = evidence.get("provenance")
        rows.append(row)

    return rows


def _decision_csv_rows(decision_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for decision in decision_rows:
        rating = decision.get("rating") if isinstance(decision.get("rating"), Mapping) else {}
        snapshot = decision.get("state_snapshot") if isinstance(decision.get("state_snapshot"), Mapping) else {}
        own = snapshot.get("own") if isinstance(snapshot.get("own"), Mapping) else {}
        opponent = snapshot.get("opponent") if isinstance(snapshot.get("opponent"), Mapping) else {}
        rows.append(
            {
                "decision_id": decision.get("decision_id"),
                "episode_id": decision.get("episode_id"),
                "request_step": decision.get("request_step"),
                "action_step": decision.get("action_step"),
                "seat": decision.get("seat"),
                "role": decision.get("role"),
                "turn": decision.get("turn"),
                "select_type": decision.get("select_type"),
                "select_context": decision.get("select_context"),
                "legal_count": decision.get("legal_count"),
                "selected_count": decision.get("selected_count"),
                "selected_categories": decision.get("selected_categories"),
                "selected_options": decision.get("selected_options"),
                "first_second_proxy": snapshot.get("first_second_proxy", decision.get("first_second_proxy")),
                "prize_lead": snapshot.get("prize_lead"),
                "own_prizes_remaining": own.get("prizes_remaining"),
                "opponent_prizes_remaining": opponent.get("prizes_remaining"),
                "own_board_count": own.get("board_count"),
                "opponent_board_count": opponent.get("board_count"),
                "own_deck_count": own.get("deck_count"),
                "opponent_deck_count": opponent.get("deck_count"),
                "rating_status": rating.get("status"),
                "value_status": rating.get("value_status"),
                "entropy": rating.get("entropy"),
                "chosen_probability": rating.get("chosen_probability"),
                "chosen_rank": rating.get("chosen_rank"),
                "margin_to_best": rating.get("margin_to_best"),
                "sequence_log_probability": rating.get("sequence_log_probability"),
                "action_agreement": rating.get("action_agreement"),
                "provenance": decision.get("provenance"),
            }
        )

    return rows


def _trajectory_payload(
    episode: ReplayEpisode,
    decision_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in decision_rows if row.get("episode_id") == episode.episode_id]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        turn = row.get("turn")
        groups[str(turn) if turn is not None else "unavailable"].append(row)
    turns = []
    for turn_key, values in groups.items():
        values = sorted(values, key=lambda row: (int(row.get("action_step") or 0), int(row.get("seat") or 0)))
        turns.append(
            {
                "turn": None if turn_key == "unavailable" else int(turn_key),
                "decisions": values,
            }
        )
    turns.sort(key=lambda row: (row["turn"] is None, int(row["turn"] or 0)))

    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "structure": "observed_trajectory_episode_to_turn_to_decision",
        "counterfactual_branches_available": False,
        "episode": episode.to_dict(),
        "perspectives": [summary for summary in summaries if summary.get("episode_id") == episode.episode_id],
        "turns": turns,
        "boundary": "no raw observations, hidden zones, opponent legal alternatives, or search inputs",
    }


def _output_hashes(output_dir: Path, relative_paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    outputs = {}
    for relative in relative_paths:
        path = output_dir / relative
        if path.is_file():
            outputs[relative] = {"sha256": sha256_file(path), "byte_size": path.stat().st_size}

    return outputs


def _privacy_sweep(output_dir: Path) -> dict[str, Any]:
    violations = []
    checked = 0
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        checked += 1
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for token in FORBIDDEN_OUTPUT_TOKENS:
            if token in text:
                violations.append({"path": str(path), "token": token})

    return {
        "checked_files": checked,
        "forbidden_tokens": list(FORBIDDEN_OUTPUT_TOKENS),
        "violations": violations,
        "passed": not violations,
    }


def _headline_summary(
    *,
    options: PostmortemOptions,
    catalog: CatalogFetchResult,
    selected_count: int,
    episodes: Sequence[ReplayEpisode],
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    scores_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
    cache_stats: Mapping[str, Any],
    binding: RuntimeBinding | None,
    matchup_shell: Mapping[str, Any],
    loss_signals: Mapping[str, Any],
    timeline: Mapping[str, Any],
    opponent_actions: Sequence[Mapping[str, Any]],
    rating_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attributable = [summary for summary in episode_summaries if not summary.get("excluded_from_opponent_attribution")]
    wins = sum(summary.get("result") == "win" for summary in attributable)
    losses = sum(summary.get("result") == "loss" for summary in attributable)
    draws = sum(summary.get("result") == "draw" for summary in attributable)
    all_scores = [row for rows in scores_by_episode.values() for row in rows]
    agreements = sum(row.get("model_action_agreement") is True for row in all_scores)
    actor_decisions = sum(row.get("role") == "actor" for row in decision_rows)
    opponent_decisions = sum(row.get("role") == "opponent" for row in decision_rows)

    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "submission_id": options.submission_id,
        "competition": options.competition,
        "catalog_metadata_status": catalog.metadata_status,
        "catalog_episode_count": len(catalog.episodes),
        "selected_catalog_episode_count": selected_count,
        "analyzed_episode_count": len(episodes),
        "actor_perspective_count": len(episode_summaries),
        "attributable_game_count": len(attributable),
        "self_mirror_episode_count": sum(episode.self_mirror for episode in episodes),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive_win_rate": wins / (wins + losses) if wins + losses else None,
        "causal_decision_count": sum(len(episode.decisions) for episode in episodes),
        "actor_decision_count": actor_decisions,
        "opponent_public_decision_count": opponent_decisions,
        "actor_scored_decision_count": len(all_scores),
        "actor_action_agreement_count": agreements,
        "actor_action_agreement_rate": agreements / len(all_scores) if all_scores else None,
        "actor_rating_status": "exact_runtime_actor" if binding is not None and all_scores else "not_instrumented",
        "value_rating_status": "not_instrumented",
        "counterfactual_blunder_status": "not_instrumented",
        "counterfactual_blunder_reason": "the submitted runtime has policy preferences but no deployed critic or counterfactual Q source",
        "score_cache": dict(cache_stats),
        "top_matchups_by_losses": _flatten_aggregate(matchup_shell, "matchups")[:10],
        "top_loss_signals": _flatten_aggregate(loss_signals, "signals")[:10],
        "top_opponent_actions_by_losses": list(opponent_actions[:15]),
        "rating_by_outcome": list(rating_outcomes),
        "timeline_phases": timeline.get("phases"),
        "careful_language": "observed associations and reconstructed policy preferences; no causal or counterfactual blunder claims",
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], *, limit: int = 10) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = ["| " + " | ".join(label for _key, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:limit]:
        values = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value if value is not None else "n/a").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")

    return lines


def _report_markdown(
    summary: Mapping[str, Any],
    *,
    matchup_rows: Sequence[Mapping[str, Any]],
    loss_signal_rows: Sequence[Mapping[str, Any]],
    opponent_action_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Submission Postmortem",
        "",
        f"Submission `{summary['submission_id']}`: **{summary['wins']}W / {summary['losses']}L / {summary['draws']}D** over "
        f"{summary['attributable_game_count']} attributable games "
        f"(decisive win rate {float(summary['decisive_win_rate'] or 0.0):.1%}).",
        "",
        "## Evidence receipt",
        "",
        f"- Episodes analyzed: {summary['analyzed_episode_count']} ({summary['self_mirror_episode_count']} self-mirror excluded from opponent attribution)",
        f"- Causally aligned CABT 1.32 decisions: {summary['causal_decision_count']}",
        f"- Exact actor rescoring: {summary['actor_action_agreement_count']}/{summary['actor_scored_decision_count']} replay actions matched",
        f"- Actor rating status: `{summary['actor_rating_status']}`",
        f"- Value rating status: `{summary['value_rating_status']}`",
        f"- Counterfactual blunder status: `{summary['counterfactual_blunder_status']}`",
        "",
        "The episode hierarchy is an observed trajectory, not a reconstructed counterfactual game tree. Policy logits, ranks, margins, and entropy are exact reconstructed preferences; they are not action values or proof of a blunder.",
        "",
        "## Matchup shells with the most observed losses",
        "",
    ]
    lines.extend(_markdown_table(matchup_rows, (("matchup", "opponent shell"), ("games", "games"), ("wins", "wins"), ("losses", "losses"), ("decisive_win_rate", "win rate"), ("loss_share", "loss share"))))
    lines.extend(["", "## Loss-signal co-occurrence", ""])
    lines.extend(_markdown_table(loss_signal_rows, (("signal", "signal"), ("losses_with_signal", "losses"), ("loss_cooccurrence_rate", "loss rate"), ("wins_with_signal", "wins"), ("win_cooccurrence_rate", "win rate"), ("loss_minus_win_cooccurrence", "loss-win"))))
    lines.extend(["", "## Opponent public actions most associated with losses", ""])
    lines.extend(_markdown_table(opponent_action_rows, (("signal_kind", "kind"), ("signal", "signal"), ("games_present", "games"), ("losses_present", "losses"), ("loss_rate_when_present", "loss rate"), ("loss_minus_win_presence", "loss-win presence")), limit=15))
    lines.extend(
        [
            "",
            "## Timing",
            "",
            f"- Early: {json.dumps(summary.get('timeline_phases', {}).get('early', {}), sort_keys=True)}",
            f"- Middle: {json.dumps(summary.get('timeline_phases', {}).get('middle', {}), sort_keys=True)}",
            f"- Late: {json.dumps(summary.get('timeline_phases', {}).get('late', {}), sort_keys=True)}",
            "",
            "### First versus second",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            list(summary.get("first_second") or ()),
            (("position", "position"), ("games", "games"), ("wins", "wins"), ("losses", "losses"), ("decisive_win_rate", "win rate")),
        )
    )
    lines.extend(
        [
            "",
            "### Opponent public playstyle averages",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            list(summary.get("opponent_playstyle_by_outcome") or ()),
            (("metric", "metric"), ("mean_when_actor_won", "actor wins"), ("mean_when_actor_lost", "actor losses"), ("loss_minus_win_mean", "loss-win")),
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Opponent decklists are postgame matchup metadata only and never enter policy rescoring.",
            "- Opponent behavior rows retain realized public actions and public turn context only.",
            "- Raw observations, search inputs, hidden zones, and opponent legal alternatives are not persisted.",
            "- Matchup, action, timeline, and loss-signal tables are descriptive associations, not causal estimates.",
        ]
    )

    return "\n".join(lines)


def _resolve_binding_for_run(options: PostmortemOptions) -> tuple[RuntimeBinding | None, str | None]:
    try:
        binding = resolve_runtime_binding(
            options.submission_id,
            bundle_dir=options.bundle_dir,
            registry_path=options.bindings_path,
        )

        return binding, None
    except SubmissionBindingError as exc:
        if options.bundle_dir is not None:
            raise PostmortemPipelineError(f"explicit bundle binding failed: {exc}") from exc

        return None, str(exc)


def _run_postmortem(options: PostmortemOptions, output_dir: Path) -> PostmortemResult:
    binding, binding_warning = _resolve_binding_for_run(options)
    write_json(
        output_dir / "binding.json",
        binding.to_dict()
        if binding is not None
        else {
            "schema_version": "deepmew.submission_postmortem.binding.v0",
            "external_submission_id": options.submission_id,
            "actor_rating_status": "unresolved_submission_binding",
            "value_rating_status": "not_instrumented",
            "error": binding_warning,
        },
    )
    _phase_checkpoint(output_dir, options=options, phase="catalog")
    catalog, selected = _fetch_catalog(options)
    if not selected:
        raise PostmortemPipelineError("catalog filters selected zero episodes")
    write_json(output_dir / "catalog.json", catalog.to_dict())
    write_json(
        output_dir / "selection.json",
        {
            "submission_id": options.submission_id,
            "query": options.query.to_dict(),
            "result_filter": options.result_filter,
            "selected_count_before_postparse_filters": len(selected),
            "episodes": [episode.to_dict() for episode in selected],
            "catalog_sha256": catalog_sha256(catalog.episodes),
        },
    )

    _phase_checkpoint(output_dir, options=options, phase="replay_cache", details={"selected": len(selected)})
    records = _replay_records(options, catalog, selected)
    write_json(
        output_dir / "replay_cache.json",
        {
            "records": [record.to_dict() for record in records],
            "complete": all(record.is_valid for record in records),
        },
    )
    if not all(record.is_valid for record in records):
        failures = [record.to_dict() for record in records if not record.is_valid]
        raise PostmortemPipelineError(f"replay cache incomplete: {json.dumps(failures, sort_keys=True)}")

    _phase_checkpoint(output_dir, options=options, phase="parse", details={"replays": len(records)})
    parsed, parse_errors = _parse_replays(
        records,
        options=options,
        selected=selected,
        binding=binding,
    )
    write_json(output_dir / "parse_errors.json", parse_errors)
    if parse_errors:
        raise PostmortemPipelineError(f"replay parse failed for {len(parse_errors)} selected episodes")
    _validate_safety(parsed, binding=binding)
    episodes = _filter_perspectives(parsed, selected=selected, options=options)
    if not episodes:
        raise PostmortemPipelineError("post-parse opponent/result filters selected zero actor perspectives")
    _validate_safety(episodes, binding=binding)

    runtime_scorer = ExactRuntimeActorScorer(binding, encoder_batch_size=options.encoder_batch_size) if binding is not None else None
    scores_by_episode: dict[str, list[dict[str, Any]]] = {}
    cache_stats: Mapping[str, Any] = {"hit_episodes": 0, "miss_episodes": 0, "scored_decisions": 0}
    if runtime_scorer is not None and not options.skip_scoring:
        _phase_checkpoint(
            output_dir,
            options=options,
            phase="actor_rescore",
            details={"actor_decisions": sum(len(episode.decisions_for_actor()) for episode in episodes)},
        )
        scores_by_episode, cache_stats = score_episodes_cached(
            episodes,
            runtime_scorer,
            cache_root=options.cache_root,
            refresh=options.refresh_scores,
        )
        mismatches = [
            row
            for rows in scores_by_episode.values()
            for row in rows
            if row.get("model_action_agreement") is not True
        ]
        if mismatches:
            write_json(output_dir / "actor_mismatches.json", mismatches)
            raise PostmortemPipelineError(
                f"exact runtime reconstruction disagreed on {len(mismatches)} decisions; ratings quarantined"
            )

    _phase_checkpoint(output_dir, options=options, phase="signals")
    decision_rows = _decision_signals(
        episodes,
        scorer=runtime_scorer,
        scores_by_episode=scores_by_episode,
    )
    ordinal_by_id = {
        episode.episode_id: ordinal
        for ordinal, episode in enumerate(stable_sort_episodes(catalog.episodes), start=1)
    }
    episode_summaries = _episode_summaries(
        episodes,
        decision_rows,
        binding=binding,
        ordinal_by_id=ordinal_by_id,
    )
    attributable_summaries = [
        summary for summary in episode_summaries if not summary.get("excluded_from_opponent_attribution")
    ]
    matchup_shell = aggregate_matchups(attributable_summaries)
    exact_matchup_summaries = [dict(summary, opponent_label=None) for summary in attributable_summaries]
    matchup_exact = aggregate_matchups(exact_matchup_summaries)
    loss_signals = aggregate_loss_signals(attributable_summaries)
    timeline = aggregate_timeline(attributable_summaries)
    opponent_actions = _opponent_action_rows(attributable_summaries, decision_rows)
    opponent_action_lifts = sorted(
        (row for row in opponent_actions if int(row.get("games_present") or 0) >= 3),
        key=lambda row: (
            -float(row.get("loss_minus_win_presence") or 0.0),
            -int(row.get("losses_present") or 0),
            -int(row.get("games_present") or 0),
            str(row.get("signal") or ""),
        ),
    )
    position_rows = _position_rows(attributable_summaries)
    playstyle_game_rows, playstyle_rows = _opponent_playstyle_rows(attributable_summaries, decision_rows)
    rating_outcomes = _rating_outcome_rows(episode_summaries, decision_rows)
    rating_phase_rows = _rating_phase_rows(episode_summaries, decision_rows)
    option_rows = [
        option
        for scores in scores_by_episode.values()
        for score in scores
        for option in score.get("option_ratings", ())
        if isinstance(option, dict)
    ]

    _phase_checkpoint(output_dir, options=options, phase="write_outputs")
    write_jsonl(output_dir / "decisions.jsonl", decision_rows)
    write_csv(output_dir / "decisions.csv", _decision_csv_rows(decision_rows))
    write_jsonl(output_dir / "decision_options.jsonl", option_rows)
    write_csv(output_dir / "episodes.csv", _episode_csv_rows(episode_summaries))
    write_json(output_dir / "episodes.json", episode_summaries)
    write_csv(output_dir / "matchups.csv", _flatten_aggregate(matchup_shell, "matchups"))
    write_csv(output_dir / "matchups_exact_deck.csv", _flatten_aggregate(matchup_exact, "matchups"))
    write_csv(output_dir / "loss_signals.csv", _flatten_aggregate(loss_signals, "signals"))
    write_json(output_dir / "loss_signals.json", loss_signals)
    write_csv(output_dir / "opponent_actions.csv", opponent_actions)
    write_csv(output_dir / "opponent_action_lifts.csv", opponent_action_lifts)
    write_csv(output_dir / "opponent_playstyle_games.csv", playstyle_game_rows)
    write_csv(output_dir / "opponent_playstyles.csv", playstyle_rows)
    write_csv(output_dir / "first_second.csv", position_rows)
    write_csv(output_dir / "rating_by_outcome.csv", rating_outcomes)
    write_csv(output_dir / "rating_by_phase.csv", rating_phase_rows)
    write_json(output_dir / "timeline.json", timeline)
    write_csv(output_dir / "timeline.csv", _flatten_aggregate(timeline, "rolling"))
    trajectory_dir = output_dir / "episodes"
    for episode in episodes:
        write_json(
            trajectory_dir / f"episode-{episode.episode_id}.json",
            _trajectory_payload(episode, decision_rows, episode_summaries),
        )

    summary = _headline_summary(
        options=options,
        catalog=catalog,
        selected_count=len(selected),
        episodes=episodes,
        episode_summaries=episode_summaries,
        decision_rows=decision_rows,
        scores_by_episode=scores_by_episode,
        cache_stats=cache_stats,
        binding=binding,
        matchup_shell=matchup_shell,
        loss_signals=loss_signals,
        timeline=timeline,
        opponent_actions=opponent_actions,
        rating_outcomes=rating_outcomes,
    )
    summary["binding_warning"] = binding_warning
    summary["catalog_warnings"] = list(catalog.warnings)
    summary["first_second"] = position_rows
    summary["top_discriminative_opponent_actions"] = opponent_action_lifts[:15]
    summary["opponent_playstyle_by_outcome"] = playstyle_rows
    summary["rating_by_phase"] = rating_phase_rows
    write_json(output_dir / "summary.json", summary)
    report = _report_markdown(
        summary,
        matchup_rows=_flatten_aggregate(matchup_shell, "matchups"),
        loss_signal_rows=_flatten_aggregate(loss_signals, "signals"),
        opponent_action_rows=opponent_action_lifts,
    )
    report_path = write_text(output_dir / "report.md", report)

    privacy = _privacy_sweep(output_dir)
    write_json(output_dir / "privacy_sweep.json", privacy)
    if not privacy["passed"]:
        raise PostmortemPipelineError(f"privacy sweep found forbidden output fields: {privacy['violations']}")

    output_names = [
        "binding.json",
        "catalog.json",
        "selection.json",
        "replay_cache.json",
        "parse_errors.json",
        "decisions.jsonl",
        "decisions.csv",
        "decision_options.jsonl",
        "episodes.csv",
        "episodes.json",
        "matchups.csv",
        "matchups_exact_deck.csv",
        "loss_signals.csv",
        "loss_signals.json",
        "opponent_actions.csv",
        "opponent_action_lifts.csv",
        "opponent_playstyle_games.csv",
        "opponent_playstyles.csv",
        "first_second.csv",
        "rating_by_outcome.csv",
        "rating_by_phase.csv",
        "timeline.json",
        "timeline.csv",
        "summary.json",
        "report.md",
        "privacy_sweep.json",
    ]
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at_utc": _utc_now().isoformat(),
        "submission_id": options.submission_id,
        "competition": options.competition,
        "command": options.command,
        "query": options.query.to_dict(),
        "result_filter": options.result_filter,
        "binding": None if binding is None else binding.to_dict(),
        "catalog_metadata_status": catalog.metadata_status,
        "source_catalog_sha256": catalog_sha256(catalog.episodes),
        "replay_sources": [
            {"episode_id": episode.episode_id, "path": episode.source_path, "sha256": episode.source_sha256}
            for episode in episodes
        ],
        "alignment_rule": "previous-step request to current-step action for CABT module 1.32.0",
        "alignment_status": "structurally_verified_unadmitted",
        "information_boundary": {
            "policy_rescore": "submitted actor seat actor_visible only",
            "opponent_behavior": "realized public actions only",
            "opponent_decks": "posthoc game metadata only",
            "raw_observations_persisted": False,
        },
        "summary": summary,
        "outputs": _output_hashes(output_dir, output_names),
        "episode_trajectory_count": len(list(trajectory_dir.glob("episode-*.json"))),
        "privacy_sweep": privacy,
        "status": "complete",
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    _phase_checkpoint(
        output_dir,
        options=options,
        phase="complete",
        status="complete",
        details={"manifest": str(manifest_path), "report": str(report_path)},
    )

    return PostmortemResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        summary_path=output_dir / "summary.json",
        summary=summary,
    )


def run_postmortem(options: PostmortemOptions) -> PostmortemResult:
    """Run the suite with phase checkpoints and fail-closed receipts."""

    _validate_options(options)
    output_dir = Path(options.output_dir) if options.output_dir is not None else _default_output_dir(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    _phase_checkpoint(output_dir, options=options, phase="start")
    try:
        return _run_postmortem(options, output_dir)
    except Exception as exc:
        _phase_checkpoint(
            output_dir,
            options=options,
            phase="failed",
            status="failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
