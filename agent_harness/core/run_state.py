"""Run-state snapshots, approvals, and permission modes.

Snapshots are captured at every node boundary and are versioned from day one
(OpenAI Agents SDK pattern). Approval is a value, not an exception — the
loop returns :class:`Interruption` from ``ToolDispatch`` when human input is
required.

Example:
    >>> from datetime import datetime, timezone
    >>> from agent_harness.core.models import Usage
    >>> snap = RunStateSnapshot(
    ...     run_id="r1",
    ...     agent_name="demo",
    ...     current_node="ModelRequest",
    ...     usage=Usage(),
    ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ... )
    >>> snap.schema_version
    '1'
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import SchemaError
from .models import Message, Usage
from .tools import ToolCall

SCHEMA_VERSION = "1"
"""Current snapshot schema version. Bump together with a migration."""

SCHEMA_VERSION_HISTORY: tuple[str, ...] = ("1: initial schema",)
"""Append a line for every shipped version. Used by the CI check that every
prior version migrates forward to current."""


PermissionMode = Literal[
    "default",
    "plan",
    "accept_edits",
    "bypass",
    "dont_ask",
    "auto",
]
"""Typed permission state.

* ``default``      — ask the user for any tool with ``needs_approval``.
* ``plan``         — read-only: only ``is_read_only`` tools allowed.
* ``accept_edits`` — auto-approve edits / writes.
* ``bypass``       — auto-approve everything (sandboxed runs only).
* ``dont_ask``     — bypass-equivalent with a different audit label.
* ``auto``         — heuristic: approve when confident, ask otherwise.
"""


# --- Approval value flow ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A pending request for the user to approve / reject a tool call.

    Example:
        >>> from datetime import datetime, timezone
        >>> ApprovalRequest(
        ...     tool_call_id="c1",
        ...     tool_name="rm",
        ...     arguments={},
        ...     requested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ... ).tool_name
        'rm'
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class Approval:
    """The user's decision on a single :class:`ApprovalRequest`.

    Example:
        >>> Approval(tool_call_id="c1", approve=False, rationale="risky").approve
        False
    """

    tool_call_id: str
    approve: bool
    rationale: str | None = None


# --- Snapshot ---------------------------------------------------------------


class RunStateSnapshot(BaseModel):
    """Round-trip-serialisable snapshot of a paused run.

    Captured at every node boundary. Stores the *minimum closure* needed to
    reconstruct the run together with its (still-extant) agent config —
    EventBus, Model, Provider, and Sandbox are re-attached from config on
    resume rather than snapshotted.

    Example:
        >>> from datetime import datetime, timezone
        >>> from agent_harness.core.models import Usage
        >>> snap = RunStateSnapshot(
        ...     run_id="r1",
        ...     agent_name="demo",
        ...     current_node="ModelRequest",
        ...     usage=Usage(),
        ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ... )
        >>> RunStateSnapshot.from_json(snap.to_json()).run_id
        'r1'
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    agent_name: str
    current_node: str
    messages: list[Message] = Field(default_factory=list)
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    pending_approvals: list[ApprovalRequest] = Field(default_factory=list)
    usage: Usage
    turn: int = 0
    deps: Any | None = None
    """JSON-serialisable user deps; non-serialisable deps are omitted and
    the caller re-supplies them on resume (RS7)."""

    session_id: str | None = None
    permission_mode: PermissionMode = "default"
    pre_plan_mode: PermissionMode | None = None
    """Mode to restore when leaving Plan Mode; None when not in Plan Mode."""

    created_at: datetime

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> RunStateSnapshot:
        """Deserialise, running :meth:`_migrate` if the version differs."""
        raw: dict[str, Any] = json.loads(data)
        return cls.model_validate(cls._migrate(raw))

    @classmethod
    def _migrate(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Apply migrations based on ``raw['schema_version']``.

        v1 is the initial schema. Future versions append a branch here AND a
        line to :data:`SCHEMA_VERSION_HISTORY`. An unknown / future version
        raises :class:`SchemaError` rather than silently passing through —
        forward-compat is one of the documented purposes of the schema
        version.

        Example:
            >>> RunStateSnapshot._migrate({"schema_version": "1"})["schema_version"]
            '1'
        """
        version = raw.get("schema_version", SCHEMA_VERSION)
        if version == SCHEMA_VERSION:
            return raw
        raise SchemaError(  # pragma: no cover - reserved for future schema bumps
            f"unsupported snapshot schema version {version!r}; "
            f"this harness supports up to {SCHEMA_VERSION!r}",
            context={"schema_version": version, "supported": SCHEMA_VERSION},
        )


# --- Interruption (defined after RunStateSnapshot so the annotation
# resolves naturally without a post-hoc ``__annotations__`` patch) -----------


@dataclass(frozen=True, slots=True)
class Interruption:
    """Returned by ``ToolDispatch`` when one or more approvals are pending.

    Example:
        >>> from datetime import datetime, timezone
        >>> from agent_harness.core.models import Usage
        >>> snap = RunStateSnapshot(
        ...     run_id="r1",
        ...     agent_name="d",
        ...     current_node="ToolDispatch",
        ...     usage=Usage(),
        ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ... )
        >>> Interruption(pending_approvals=[], run_state=snap).pending_approvals
        []
    """

    pending_approvals: list[ApprovalRequest]
    run_state: RunStateSnapshot
