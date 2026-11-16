"""Invoke a pinned, already-deployed terminal-league Modal Function.

Run this module with ordinary Python. It sends only the expected ledger SHA-256
and the smoke flag to one of the two approved deployed Functions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import modal


APP_NAME_PREFIX = "deepmew-terminal-league-"
GENERIC_APP_NAME = "deepmew-terminal-league"
RECEIPT_SCHEMA_VERSION = "deepmew.modal_terminal_league_invocation.v3"
FUNCTION_BY_LANE = {
    "starmie": "starmie_terminal_league",
    "grimmsnarl": "grimmsnarl_terminal_league",
}


class TerminalLeagueInvocationError(RuntimeError):
    """An invocation request or returned smoke result failed closed."""


def _app_name(value: str) -> str:
    if value == GENERIC_APP_NAME:
        raise TerminalLeagueInvocationError("generic terminal-league app name is forbidden")
    if (
        not isinstance(value, str)
        or len(value) > 64
        or len(value) <= len(APP_NAME_PREFIX)
        or not value.startswith(APP_NAME_PREFIX)
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise TerminalLeagueInvocationError(
            "app name must use the unique deepmew-terminal-league- namespace and Modal name characters"
        )

    return value


def _lane_name(value: str) -> str:
    if not isinstance(value, str):
        raise TerminalLeagueInvocationError("lane must be a string")
    normalized = value.strip().lower()
    if normalized not in FUNCTION_BY_LANE:
        allowed = ", ".join(sorted(FUNCTION_BY_LANE))
        raise TerminalLeagueInvocationError(f"lane must be one of: {allowed}")

    return normalized


def _ledger_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TerminalLeagueInvocationError("ledger SHA-256 must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise TerminalLeagueInvocationError(
            "ledger SHA-256 must be exactly 64 hexadecimal characters"
        )

    return normalized


def _function_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or re.fullmatch(
            r"fu-[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?",
            value,
        )
        is None
    ):
        raise TerminalLeagueInvocationError(
            "expected Function ID must use the Modal fu- identifier format"
        )

    return value


def _smoke_status(result: Any, *, lane: str) -> str:
    if not isinstance(result, Mapping):
        raise TerminalLeagueInvocationError("deployed smoke result must be an object")
    if result.get("status") != "smoke_complete":
        raise TerminalLeagueInvocationError("deployed smoke did not report smoke_complete")
    if result.get("lane") != lane:
        raise TerminalLeagueInvocationError("deployed smoke result lane does not match request")
    returncode = result.get("returncode")
    if isinstance(returncode, bool) or returncode != 0:
        raise TerminalLeagueInvocationError("deployed smoke did not return exit code 0")

    return "smoke_complete"


def _resolve_function(
    *,
    app_name: str,
    lane: str,
    expected_function_id: str,
) -> tuple[Any, str]:
    function_name = FUNCTION_BY_LANE[lane]
    try:
        function = modal.Function.from_name(app_name, function_name)
        function.hydrate()
        resolved_function_id = function.object_id
    except Exception as exc:
        raise TerminalLeagueInvocationError(
            f"could not hydrate named deployed Function: {type(exc).__name__}"
        ) from exc
    if resolved_function_id != expected_function_id:
        raise TerminalLeagueInvocationError(
            "resolved deployed Function ID does not match the expected Function ID"
        )

    return function, resolved_function_id


def invoke_terminal_league(
    *,
    app_name: str,
    lane: str,
    expected_ledger_sha256: str,
    expected_function_id: str,
    smoke: bool,
    detached_ack: bool,
) -> dict[str, Any]:
    """Invoke an approved deployed Function and return a sanitized receipt."""

    normalized_app_name = _app_name(app_name)
    normalized_lane = _lane_name(lane)
    normalized_ledger_sha256 = _ledger_sha256(expected_ledger_sha256)
    normalized_expected_function_id = _function_id(expected_function_id)
    if not isinstance(smoke, bool) or not isinstance(detached_ack, bool):
        raise TerminalLeagueInvocationError("smoke and detached acknowledgment must be boolean")
    if not smoke and not detached_ack:
        raise TerminalLeagueInvocationError(
            "long invocation requires an explicit detached acknowledgment"
        )
    function_name = FUNCTION_BY_LANE[normalized_lane]
    function, resolved_function_id = _resolve_function(
        app_name=normalized_app_name,
        lane=normalized_lane,
        expected_function_id=normalized_expected_function_id,
    )
    if smoke:
        try:
            smoke_function = function.with_options(timeout=900, retries=0)
            result = smoke_function.remote(normalized_ledger_sha256, True)
        except Exception as exc:
            raise TerminalLeagueInvocationError(
                f"deployed smoke invocation failed: {type(exc).__name__}"
            ) from exc
        status = _smoke_status(result, lane=normalized_lane)

        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": status,
            "app": normalized_app_name,
            "function": function_name,
            "deployment_version": None,
            "identity_mode": "unique_app_name_single_history_v1_function_id_v1",
            "expected_function_id": normalized_expected_function_id,
            "resolved_function_id": resolved_function_id,
            "ledger_sha256": normalized_ledger_sha256,
            "lane": normalized_lane,
            "smoke": True,
            "invocation_method": "remote_sync",
            "call_id": None,
            "dashboard_url": None,
        }
    try:
        call = function.spawn(normalized_ledger_sha256, False)
        call_id = call.object_id
        dashboard_url = call.get_dashboard_url()
    except Exception as exc:
        raise TerminalLeagueInvocationError(
            f"deployed long invocation failed: {type(exc).__name__}"
        ) from exc
    if not isinstance(call_id, str) or not call_id:
        raise TerminalLeagueInvocationError("spawned invocation did not return a call id")
    if not isinstance(dashboard_url, str) or not dashboard_url:
        raise TerminalLeagueInvocationError("spawned invocation did not return a dashboard URL")

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "spawned",
        "app": normalized_app_name,
        "function": function_name,
        "deployment_version": None,
        "identity_mode": "unique_app_name_single_history_v1_function_id_v1",
        "expected_function_id": normalized_expected_function_id,
        "resolved_function_id": resolved_function_id,
        "ledger_sha256": normalized_ledger_sha256,
        "lane": normalized_lane,
        "smoke": False,
        "invocation_method": "spawn_async",
        "call_id": call_id,
        "dashboard_url": dashboard_url,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invoke a pinned, already-deployed terminal-league Function.",
    )
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--lane", required=True, choices=sorted(FUNCTION_BY_LANE))
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--expected-function-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--detached-ack", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        receipt = invoke_terminal_league(
            app_name=arguments.app_name,
            lane=arguments.lane,
            expected_ledger_sha256=arguments.ledger_sha256,
            expected_function_id=arguments.expected_function_id,
            smoke=arguments.smoke,
            detached_ack=arguments.detached_ack,
        )
    except TerminalLeagueInvocationError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
