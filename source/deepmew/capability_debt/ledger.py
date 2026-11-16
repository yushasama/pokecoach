"""Validated, history-preserving research capability-debt ledger.

The ledger is deliberately independent of replay parsing and model code. Future
research missions can update one capability row and regenerate the human and
tabular views without rerunning a submission postmortem.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any


LEDGER_SCHEMA_VERSION = "deepmew.capability_debt_ledger.v1"
MANIFEST_SCHEMA_VERSION = "deepmew.capability_debt_ledger.manifest.v1"

ALLOWED_STATUSES = (
    "MISSING",
    "DESIGNING",
    "IMPLEMENTING",
    "PARTIALLY_AVAILABLE",
    "VALIDATING",
    "RELIABLE",
    "DEFERRED",
    "REJECTED",
)
ALLOWED_PRIORITIES = ("P0", "P1", "P2", "P3")
MATURITY_LEVELS = {
    0: "Absent",
    1: "Prototype",
    2: "Reliable",
    3: "Production",
    4: "Research-complete",
}

REQUIRED_ROW_FIELDS = (
    "capability_id",
    "capability",
    "current_maturity_level",
    "status",
    "blocks",
    "unlocks",
    "dependencies",
    "evidence_of_current_limitation",
    "required_deliverable",
    "verification_criteria",
    "estimated_implementation_effort",
    "maintenance_burden",
    "technical_risk",
    "expected_information_value",
    "expected_reuse",
    "priority",
    "owner",
    "decision_log_reference",
    "last_updated",
)

CSV_FIELDS = (
    "Capability ID",
    "Capability",
    "Scope",
    "Current maturity level",
    "Status",
    "Blocks",
    "Unlocks",
    "Dependencies",
    "Evidence of current limitation",
    "Required deliverable",
    "Verification criteria",
    "Estimated implementation effort",
    "Maintenance burden",
    "Technical risk",
    "Expected information value",
    "Expected information value rationale",
    "Information rate rank",
    "Expected reuse",
    "Priority",
    "Owner",
    "Decision-log reference",
    "Last updated",
)

RELIABILITY_EVIDENCE_FIELDS = (
    "implemented_deliverable",
    "verification_artifacts",
    "compatible_contracts",
    "safety_and_leakage_checks",
    "documented_limitations",
)

_CAPABILITY_ID = re.compile(r"^CAP-[0-9]{3}$")
_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class CapabilityLedgerError(ValueError):
    """Raised when a ledger or transition violates the durable contract."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityLedgerError(f"{name} must be a non-empty string")

    return value.strip()


def _require_date(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if _ISO_DATE.fullmatch(text) is None:
        raise CapabilityLedgerError(f"{name} must be an ISO date")
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise CapabilityLedgerError(f"{name} must be a valid ISO date") from exc

    return text


def _require_string_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CapabilityLedgerError(f"{name} must be a list of strings")
    output = [_require_text(item, f"{name} item") for item in value]
    if not allow_empty and not output:
        raise CapabilityLedgerError(f"{name} must not be empty")

    return output


def _validate_reliability_evidence(row: Mapping[str, Any]) -> None:
    evidence = row.get("reliability_evidence")
    if not isinstance(evidence, Mapping):
        raise CapabilityLedgerError("RELIABLE rows require reliability_evidence")
    missing = [field for field in RELIABILITY_EVIDENCE_FIELDS if field not in evidence]
    if missing:
        raise CapabilityLedgerError(
            "RELIABLE rows are missing reliability evidence: " + ", ".join(missing)
        )
    _require_text(evidence["implemented_deliverable"], "implemented_deliverable")
    _require_string_list(evidence["verification_artifacts"], "verification_artifacts")
    _require_string_list(evidence["compatible_contracts"], "compatible_contracts")
    _require_string_list(
        evidence["safety_and_leakage_checks"],
        "safety_and_leakage_checks",
    )
    _require_string_list(evidence["documented_limitations"], "documented_limitations")


def _validate_history(
    history: Any,
    capability_id: str,
    *,
    current_status: str,
    current_maturity_level: int,
) -> None:
    if not isinstance(history, list):
        raise CapabilityLedgerError(f"{capability_id} history must be a list")
    previous_status: str | None = None
    previous_maturity: int | None = None
    previous_date: str | None = None
    for index, event in enumerate(history):
        if not isinstance(event, Mapping):
            raise CapabilityLedgerError(f"{capability_id} history[{index}] must be an object")
        for field in (
            "changed_on",
            "from_status",
            "to_status",
            "from_maturity_level",
            "to_maturity_level",
            "evidence",
            "validating_artifact",
            "decision_log_reference",
            "owner",
            "blocks_before",
            "blocks_after",
            "unlocks_before",
            "unlocks_after",
        ):
            if field not in event:
                raise CapabilityLedgerError(
                    f"{capability_id} history[{index}] is missing {field}"
                )
        changed_on = _require_date(
            event["changed_on"],
            f"{capability_id} history changed_on",
        )
        if event["from_status"] not in ALLOWED_STATUSES:
            raise CapabilityLedgerError(f"{capability_id} history from_status is invalid")
        if event["to_status"] not in ALLOWED_STATUSES:
            raise CapabilityLedgerError(f"{capability_id} history to_status is invalid")
        for field in ("from_maturity_level", "to_maturity_level"):
            maturity = event[field]
            if isinstance(maturity, bool) or maturity not in MATURITY_LEVELS:
                raise CapabilityLedgerError(
                    f"{capability_id} history {field} must be 0..4"
                )
        for field in ("evidence", "validating_artifact", "decision_log_reference", "owner"):
            _require_text(event[field], f"{capability_id} history {field}")
        for field in ("blocks_before", "blocks_after", "unlocks_before", "unlocks_after"):
            _require_string_list(event[field], f"{capability_id} history {field}")
        if previous_status is not None and event["from_status"] != previous_status:
            raise CapabilityLedgerError(f"{capability_id} history status chain is discontinuous")
        if previous_maturity is not None and event["from_maturity_level"] != previous_maturity:
            raise CapabilityLedgerError(f"{capability_id} history maturity chain is discontinuous")
        if previous_date is not None and changed_on < previous_date:
            raise CapabilityLedgerError(f"{capability_id} history dates are not monotonic")
        previous_status = event["to_status"]
        previous_maturity = event["to_maturity_level"]
        previous_date = changed_on
    if history and previous_status != current_status:
        raise CapabilityLedgerError(f"{capability_id} history does not end at current status")
    if history and previous_maturity != current_maturity_level:
        raise CapabilityLedgerError(f"{capability_id} history does not end at current maturity")


def validate_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping):
        raise CapabilityLedgerError("capability row must be an object")
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise CapabilityLedgerError("capability row is missing: " + ", ".join(missing))
    capability_id = _require_text(row["capability_id"], "capability_id")
    if _CAPABILITY_ID.fullmatch(capability_id) is None:
        raise CapabilityLedgerError(f"invalid capability_id: {capability_id}")
    _require_text(row["capability"], f"{capability_id} capability")
    maturity = row["current_maturity_level"]
    if isinstance(maturity, bool) or maturity not in MATURITY_LEVELS:
        raise CapabilityLedgerError(f"{capability_id} maturity must be 0..4")
    status = row["status"]
    if status not in ALLOWED_STATUSES:
        raise CapabilityLedgerError(f"{capability_id} status is invalid: {status}")
    if status == "RELIABLE" and maturity < 2:
        raise CapabilityLedgerError(f"{capability_id} RELIABLE maturity must be at least 2")
    priority = row["priority"]
    if priority not in ALLOWED_PRIORITIES:
        raise CapabilityLedgerError(f"{capability_id} priority is invalid: {priority}")
    for field in ("blocks", "unlocks", "dependencies", "verification_criteria"):
        _require_string_list(row[field], f"{capability_id} {field}")
    _require_string_list(
        row["evidence_of_current_limitation"],
        f"{capability_id} evidence_of_current_limitation",
    )
    _require_string_list(
        row.get("verification_artifacts", []),
        f"{capability_id} verification_artifacts",
        allow_empty=True,
    )
    _require_string_list(
        row.get("critical_path_projects_blocked", []),
        f"{capability_id} critical_path_projects_blocked",
        allow_empty=True,
    )
    for field in (
        "required_deliverable",
        "estimated_implementation_effort",
        "maintenance_burden",
        "technical_risk",
        "expected_information_value",
        "expected_reuse",
        "owner",
        "decision_log_reference",
    ):
        _require_text(row[field], f"{capability_id} {field}")
    _require_text(row.get("scope", "project-wide"), f"{capability_id} scope")
    _require_text(
        row.get("expected_information_value_rationale", "not supplied"),
        f"{capability_id} expected_information_value_rationale",
    )
    _require_date(row["last_updated"], f"{capability_id} last_updated")
    _validate_history(
        row.get("history", []),
        capability_id,
        current_status=status,
        current_maturity_level=maturity,
    )
    if status == "RELIABLE":
        _validate_reliability_evidence(row)
    if status == "REJECTED":
        _require_text(row.get("rejection_rationale"), f"{capability_id} rejection_rationale")


def validate_ledger(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise CapabilityLedgerError("ledger must be a JSON object")
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise CapabilityLedgerError("unsupported capability ledger schema")
    _require_text(payload.get("project"), "project")
    _require_text(payload.get("scope"), "scope")
    _require_date(payload.get("last_updated"), "last_updated")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CapabilityLedgerError("ledger rows must be a non-empty list")
    identifiers = []
    for row in rows:
        validate_row(row)
        identifiers.append(row["capability_id"])
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise CapabilityLedgerError("duplicate capability ids: " + ", ".join(duplicates))
    information_rate_ranks = []
    for row in rows:
        rank = row.get("information_rate_rank")
        if rank is None:
            continue
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise CapabilityLedgerError(
                f"{row['capability_id']} information_rate_rank must be a positive integer"
            )
        information_rate_ranks.append(rank)
    duplicate_ranks = sorted(
        rank for rank, count in Counter(information_rate_ranks).items() if count > 1
    )
    if duplicate_ranks:
        raise CapabilityLedgerError(
            "duplicate information-rate ranks: "
            + ", ".join(str(rank) for rank in duplicate_ranks)
        )
    expected_ranks = list(range(1, len(information_rate_ranks) + 1))
    if sorted(information_rate_ranks) != expected_ranks:
        raise CapabilityLedgerError(
            "information-rate ranks must be contiguous from 1 through the ranked-row count"
        )


def load_ledger(path: str | Path) -> dict[str, Any]:
    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityLedgerError(f"cannot read ledger {ledger_path}: {exc}") from exc
    validate_ledger(payload)

    return payload


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_ledger(path: str | Path, payload: Mapping[str, Any]) -> None:
    validate_ledger(payload)
    _atomic_write_text(Path(path), _canonical_json(payload))


def _transition_change(ledger: dict[str, Any], change: Mapping[str, Any]) -> None:
    capability_id = _require_text(change.get("capability_id"), "change capability_id")
    matches = [row for row in ledger["rows"] if row["capability_id"] == capability_id]
    if len(matches) != 1:
        raise CapabilityLedgerError(f"transition target not found: {capability_id}")
    row = matches[0]
    expected = _require_text(change.get("expected_current_status"), "expected_current_status")
    if row["status"] != expected:
        raise CapabilityLedgerError(
            f"{capability_id} stale transition: expected {expected}, found {row['status']}"
        )
    new_status = _require_text(change.get("new_status"), "new_status")
    if new_status not in ALLOWED_STATUSES:
        raise CapabilityLedgerError(f"invalid new_status: {new_status}")
    new_maturity = change.get("new_maturity_level")
    if isinstance(new_maturity, bool) or new_maturity not in MATURITY_LEVELS:
        raise CapabilityLedgerError("new_maturity_level must be 0..4")
    changed_on = _require_date(change.get("changed_on"), "changed_on")
    evidence = _require_text(change.get("evidence"), "evidence")
    artifact = _require_text(change.get("validating_artifact"), "validating_artifact")
    decision = _require_text(change.get("decision_log_reference"), "decision_log_reference")
    owner = _require_text(change.get("owner"), "owner")
    blocks = _require_string_list(change.get("blocks"), "change blocks")
    unlocks = _require_string_list(change.get("unlocks"), "change unlocks")
    history = list(row.get("history", []))
    history.append({
        "changed_on": changed_on,
        "from_status": row["status"],
        "to_status": new_status,
        "from_maturity_level": row["current_maturity_level"],
        "to_maturity_level": new_maturity,
        "evidence": evidence,
        "validating_artifact": artifact,
        "decision_log_reference": decision,
        "owner": owner,
        "blocks_before": list(row["blocks"]),
        "blocks_after": blocks,
        "unlocks_before": list(row["unlocks"]),
        "unlocks_after": unlocks,
    })
    row.update({
        "status": new_status,
        "current_maturity_level": new_maturity,
        "blocks": blocks,
        "unlocks": unlocks,
        "owner": owner,
        "decision_log_reference": decision,
        "last_updated": changed_on,
        "history": history,
    })
    for field in (
        "dependencies",
        "evidence_of_current_limitation",
        "required_deliverable",
        "verification_criteria",
        "estimated_implementation_effort",
        "maintenance_burden",
        "technical_risk",
        "expected_information_value",
        "expected_information_value_rationale",
        "expected_reuse",
        "priority",
        "verification_artifacts",
        "critical_path_projects_blocked",
        "reliability_evidence",
        "rejection_rationale",
    ):
        if field in change:
            row[field] = copy.deepcopy(change[field])
    ledger["last_updated"] = changed_on


def _add_change(ledger: dict[str, Any], change: Mapping[str, Any]) -> None:
    row = change.get("row")
    if not isinstance(row, Mapping):
        raise CapabilityLedgerError("add change requires a row object")
    added = copy.deepcopy(dict(row))
    added.setdefault("history", [])
    validate_row(added)
    identifiers = {current["capability_id"] for current in ledger["rows"]}
    if added["capability_id"] in identifiers:
        raise CapabilityLedgerError(f"duplicate capability id: {added['capability_id']}")
    ledger["rows"].append(added)
    ledger["rows"].sort(key=lambda item: item["capability_id"])
    ledger["last_updated"] = added["last_updated"]


def apply_change(payload: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    validate_ledger(payload)
    if not isinstance(change, Mapping):
        raise CapabilityLedgerError("change must be a JSON object")
    operation = _require_text(change.get("operation"), "operation")
    updated = copy.deepcopy(dict(payload))
    if operation == "transition":
        _transition_change(updated, change)
    elif operation == "add":
        _add_change(updated, change)
    else:
        raise CapabilityLedgerError("operation must be transition or add; deletion is unsupported")
    validate_ledger(updated)

    return updated


def apply_change_file(ledger_path: str | Path, change_path: str | Path) -> dict[str, Any]:
    ledger = load_ledger(ledger_path)
    try:
        change = json.loads(Path(change_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityLedgerError(f"cannot read change {change_path}: {exc}") from exc
    updated = apply_change(ledger, change)
    write_ledger(ledger_path, updated)

    return updated


def _display_list(values: Sequence[str]) -> str:
    return "; ".join(values)


def _csv_text(payload: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in payload["rows"]:
        writer.writerow({
            "Capability ID": row["capability_id"],
            "Capability": row["capability"],
            "Scope": row.get("scope", "project-wide"),
            "Current maturity level": row["current_maturity_level"],
            "Status": row["status"],
            "Blocks": _display_list(row["blocks"]),
            "Unlocks": _display_list(row["unlocks"]),
            "Dependencies": _display_list(row["dependencies"]),
            "Evidence of current limitation": _display_list(row["evidence_of_current_limitation"]),
            "Required deliverable": row["required_deliverable"],
            "Verification criteria": _display_list(row["verification_criteria"]),
            "Estimated implementation effort": row["estimated_implementation_effort"],
            "Maintenance burden": row["maintenance_burden"],
            "Technical risk": row["technical_risk"],
            "Expected information value": row["expected_information_value"],
            "Expected information value rationale": row.get(
                "expected_information_value_rationale",
                "not supplied",
            ),
            "Information rate rank": row.get("information_rate_rank", ""),
            "Expected reuse": row["expected_reuse"],
            "Priority": row["priority"],
            "Owner": row["owner"],
            "Decision-log reference": row["decision_log_reference"],
            "Last updated": row["last_updated"],
        })

    return stream.getvalue()


def _markdown_list(values: Sequence[str]) -> str:
    return "\n".join(f"  - {value}" for value in values)


def _markdown_text(payload: Mapping[str, Any]) -> str:
    status_counts = Counter(row["status"] for row in payload["rows"])
    priority_counts = Counter(row["priority"] for row in payload["rows"])
    lines = [
        "# DeepMew Capability Debt Ledger",
        "",
        f"Schema: `{payload['schema_version']}`",
        "",
        f"Last updated: `{payload['last_updated']}`",
        f"Scope: {payload['scope']}",
        "",
        "This is a research capability ledger, not a speculative feature wishlist or an Elo roadmap.",
        "Rows are retained when rejected, and status transitions preserve their prior state in JSON history.",
        "",
        "## Summary",
        "",
        f"- Capabilities: **{len(payload['rows'])}**",
        "- Statuses: " + ", ".join(f"`{key}`={status_counts[key]}" for key in ALLOWED_STATUSES if status_counts[key]),
        "- Priorities: " + ", ".join(f"`{key}`={priority_counts[key]}" for key in ALLOWED_PRIORITIES if priority_counts[key]),
        "",
        "## Priority view",
        "",
        "| Priority | Rate rank | ID | Capability | Maturity | Status | Information value | Reuse | Effort |",
        "|---|---:|---|---|---:|---|---|---|---|",
    ]
    priority_order = {priority: index for index, priority in enumerate(ALLOWED_PRIORITIES)}
    for row in sorted(payload["rows"], key=lambda item: (priority_order[item["priority"]], item["capability_id"])):
        lines.append(
            f"| {row['priority']} | {row.get('information_rate_rank', '—')} | "
            f"{row['capability_id']} | {row['capability']} | "
            f"{row['current_maturity_level']} | {row['status']} | {row['expected_information_value']} | "
            f"{row['expected_reuse']} | {row['estimated_implementation_effort']} |"
        )
    lines.extend(["", "## Capability records", ""])
    for row in sorted(payload["rows"], key=lambda item: item["capability_id"]):
        lines.extend([
            f"### {row['capability_id']} — {row['capability']}",
            "",
            f"- Scope: {row.get('scope', 'project-wide')}",
            f"- Maturity: **{row['current_maturity_level']} — {MATURITY_LEVELS[row['current_maturity_level']]}**",
            f"- Status: **{row['status']}**",
            f"- Priority: **{row['priority']}**",
            f"- Expected information value: **{row['expected_information_value']}** — {row.get('expected_information_value_rationale', 'not supplied')}",
            f"- Information-rate rank: **{row.get('information_rate_rank', 'not ranked')}**",
            f"- Expected reuse: **{row['expected_reuse']}**",
            f"- Estimated effort: {row['estimated_implementation_effort']}",
            f"- Maintenance burden: {row['maintenance_burden']}",
            f"- Technical risk: {row['technical_risk']}",
            f"- Owner: {row['owner']}",
            f"- Decision-log reference: {row['decision_log_reference']}",
            f"- Last updated: {row['last_updated']}",
            "",
            "Blocks:",
            _markdown_list(row["blocks"]),
            "",
            "Unlocks:",
            _markdown_list(row["unlocks"]),
            "",
            "Dependencies:",
            _markdown_list(row["dependencies"]),
            "",
            "Evidence of current limitation:",
            _markdown_list(row["evidence_of_current_limitation"]),
            "",
            f"Required deliverable: {row['required_deliverable']}",
            "",
            "Verification criteria:",
            _markdown_list(row["verification_criteria"]),
            "",
        ])
        if row.get("verification_artifacts"):
            lines.extend(["Current verification artifacts:", _markdown_list(row["verification_artifacts"]), ""])
        reliability = row.get("reliability_evidence")
        if reliability:
            lines.extend([
                "Reliability evidence:",
                f"  - Implemented deliverable: {reliability['implemented_deliverable']}",
                "  - Verification artifacts: " + _display_list(reliability["verification_artifacts"]),
                "  - Compatible contracts: " + _display_list(reliability["compatible_contracts"]),
                "  - Safety and leakage checks: "
                + _display_list(reliability["safety_and_leakage_checks"]),
                "  - Documented limitations: "
                + _display_list(reliability["documented_limitations"]),
                "",
            ])
        if row.get("rejection_rationale"):
            lines.extend([f"Rejection rationale: {row['rejection_rationale']}", ""])
        history = row.get("history", [])
        lines.append(f"History entries: {len(history)}")
        for event in history:
            lines.append(
                f"  - {event['changed_on']}: {event['from_status']} "
                f"(maturity {event['from_maturity_level']}) -> {event['to_status']} "
                f"(maturity {event['to_maturity_level']}); evidence: {event['evidence']}; "
                f"artifact: {event['validating_artifact']}; decision: "
                f"{event['decision_log_reference']}; owner: {event['owner']}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def render_outputs(
    ledger_path: str | Path,
    markdown_path: str | Path,
    csv_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    ledger_source = Path(ledger_path)
    markdown_output = Path(markdown_path)
    csv_output = Path(csv_path)
    manifest_output = Path(manifest_path)
    payload = load_ledger(ledger_source)
    _atomic_write_text(markdown_output, _markdown_text(payload))
    _atomic_write_text(csv_output, _csv_text(payload))
    implementation_path = Path(__file__)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "last_updated": payload["last_updated"],
        "row_count": len(payload["rows"]),
        "status_counts": dict(sorted(Counter(row["status"] for row in payload["rows"]).items())),
        "priority_counts": dict(sorted(Counter(row["priority"] for row in payload["rows"]).items())),
        "information_boundary": (
            "capability metadata and repository evidence paths only; no replay observations, "
            "hidden zones, private engine state, model tensors, or Elo-gain estimates"
        ),
        "inputs": {
            "ledger_json": {
                "path": _manifest_path(ledger_source),
                "sha256": _sha256_path(ledger_source),
            },
            "implementation": {
                "path": _manifest_path(implementation_path),
                "sha256": _sha256_path(implementation_path),
            },
        },
        "outputs": {
            "markdown": {
                "path": _manifest_path(markdown_output),
                "sha256": _sha256_path(markdown_output),
            },
            "csv": {
                "path": _manifest_path(csv_output),
                "sha256": _sha256_path(csv_output),
            },
        },
    }
    _atomic_write_text(manifest_output, _canonical_json(manifest))

    return manifest


__all__ = [
    "ALLOWED_PRIORITIES",
    "ALLOWED_STATUSES",
    "CapabilityLedgerError",
    "LEDGER_SCHEMA_VERSION",
    "MATURITY_LEVELS",
    "apply_change",
    "apply_change_file",
    "load_ledger",
    "render_outputs",
    "validate_ledger",
    "validate_row",
    "write_ledger",
]
