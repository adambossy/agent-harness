"""Unit tests for ``agent_harness.core.run_state``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import pytest

from agent_harness.core.errors import SchemaError
from agent_harness.core.models import Message, TextBlock, Usage
from agent_harness.core.run_state import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_HISTORY,
    Approval,
    ApprovalRequest,
    Interruption,
    PermissionMode,
    RunStateSnapshot,
)
from agent_harness.core.tools import ToolCall


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _make_snapshot(**overrides: object) -> RunStateSnapshot:
    base: dict[str, object] = {
        "run_id": "r1",
        "agent_name": "demo",
        "current_node": "ModelRequest",
        "messages": [],
        "pending_tool_calls": [],
        "pending_approvals": [],
        "usage": Usage(),
        "turn": 0,
        "created_at": _ts(),
    }
    base.update(overrides)
    # Rationale for the ignore below: ``base`` is dict[str, object] so each
    # value is widened to object; passing it as **kwargs gives mypy no way
    # to match each value against its declared field type. The dict is
    # constructed locally with spec-conformant values, so the runtime call
    # is safe.
    return RunStateSnapshot(**base)  # type: ignore[arg-type]


def test_schema_version_history_matches_current_version() -> None:
    """Every shipped version is documented; the latest line names the
    current version."""

    assert SCHEMA_VERSION_HISTORY, "history must not be empty"
    assert SCHEMA_VERSION_HISTORY[-1].startswith(f"{SCHEMA_VERSION}:")


def test_snapshot_round_trips_through_json() -> None:
    msg = Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())
    call = ToolCall(id="c1", name="search", arguments={"q": "x"})
    req = ApprovalRequest(
        tool_call_id="c1",
        tool_name="search",
        arguments={"q": "x"},
        requested_at=_ts(),
    )
    snap = _make_snapshot(
        messages=[msg],
        pending_tool_calls=[call],
        pending_approvals=[req],
        usage=Usage(input_tokens=5),
        turn=2,
        deps={"db": "url"},
        session_id="s1",
        permission_mode="plan",
        pre_plan_mode="default",
    )

    data = snap.to_json()
    restored = RunStateSnapshot.from_json(data)
    assert restored.run_id == "r1"
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.permission_mode == "plan"
    assert restored.pre_plan_mode == "default"
    assert restored.turn == 2
    assert restored.deps == {"db": "url"}
    assert len(restored.messages) == 1
    assert restored.messages[0].text == "hi"


def test_snapshot_defaults_for_optional_fields() -> None:
    snap = _make_snapshot()
    assert snap.permission_mode == "default"
    assert snap.pre_plan_mode is None
    assert snap.deps is None
    assert snap.session_id is None


def test_migrate_stub_returns_input_unchanged_for_current_version() -> None:
    raw = {"schema_version": SCHEMA_VERSION, "run_id": "r1"}
    assert RunStateSnapshot._migrate(raw) == raw


def test_migrate_handles_missing_version_field() -> None:
    """An old snapshot that omitted ``schema_version`` is treated as v1
    (the only version that has existed)."""

    raw: dict[str, object] = {"run_id": "r1"}
    out = RunStateSnapshot._migrate(raw)
    assert out is raw  # stub passthrough


def test_migrate_raises_on_unknown_future_version() -> None:
    """A snapshot from a newer harness version that this binary can't
    migrate from must raise :class:`SchemaError`, not silently pass through.

    Forward-compat is one of the documented purposes of the schema version
    (RS5); silently accepting an unknown version defeats it."""

    raw = {"schema_version": "999", "run_id": "r1"}
    with pytest.raises(SchemaError, match="unsupported snapshot schema version"):
        RunStateSnapshot._migrate(raw)


def test_permission_mode_membership_is_exhaustive() -> None:
    """The literal alias enumerates exactly the six documented modes."""

    members = set(get_args(PermissionMode))
    assert members == {
        "default",
        "plan",
        "accept_edits",
        "bypass",
        "dont_ask",
        "auto",
    }


def test_approval_request_and_decision_shapes() -> None:
    req = ApprovalRequest(
        tool_call_id="c1",
        tool_name="rm",
        arguments={"path": "/x"},
        requested_at=_ts(),
    )
    assert req.tool_name == "rm"

    yes = Approval(tool_call_id="c1", approve=True)
    assert yes.rationale is None

    no = Approval(tool_call_id="c1", approve=False, rationale="risky")
    assert no.rationale == "risky"


def test_interruption_carries_pending_and_snapshot() -> None:
    snap = _make_snapshot(current_node="ToolDispatch")
    req = ApprovalRequest(
        tool_call_id="c1",
        tool_name="rm",
        arguments={"path": "/x"},
        requested_at=_ts(),
    )
    pause = Interruption(pending_approvals=[req], run_state=snap)
    assert pause.run_state.current_node == "ToolDispatch"
    assert pause.pending_approvals[0].tool_name == "rm"


def test_invalid_permission_mode_rejected() -> None:
    with pytest.raises(ValueError):
        _make_snapshot(permission_mode="wizard")
