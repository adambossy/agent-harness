"""Per-run mutable context + result + small helpers (Wave 4).

Extracted from :mod:`.agent` so the public surface stays small and the
per-file LOC budget (<500 LOC for any core file) is kept comfortable.
Imports are Protocol-only; no concrete provider / sandbox / session here.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .events import EventBus
from .memory import LongTermMemory
from .models import Message, Usage
from .run_state import (
    Approval,
    ApprovalRequest,
    Interruption,
    PermissionMode,
    RunStateSnapshot,
)
from .sandbox import Sandbox
from .tools import Tool, ToolCall

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .agent import Agent

Deps = TypeVar("Deps")
Out = TypeVar("Out")


# --- Parent-run ContextVar -------------------------------------------------
#
# Threaded by :class:`ToolDispatch` so that ``Agent.as_tool``'s wrapper can
# discover the parent's :class:`RunContext` (and thus its ``EventBus`` and
# agent name) at tool-call time, without requiring tool functions to declare
# a ``ctx`` parameter (which ``Toolset`` doesn't forward). See AT2 / AT3 in
# ``components/subagents.md``.
_current_run_ctx: contextvars.ContextVar[RunContext[Any] | None] = contextvars.ContextVar(
    "_agent_harness_current_run_ctx",
    default=None,
)


def get_current_run_ctx() -> RunContext[Any] | None:
    """Return the :class:`RunContext` currently being dispatched, or ``None``.

    Used by :func:`Agent.as_tool`'s wrapper to access the parent agent's
    :class:`EventBus` and name for subagent event republishing.

    Example:
        >>> get_current_run_ctx() is None
        True
    """
    return _current_run_ctx.get()


def set_current_run_ctx(ctx: RunContext[Any] | None) -> contextvars.Token[Any]:
    """Set the current :class:`RunContext` and return a reset-token.

    Callers must :func:`reset_current_run_ctx` with the returned token in a
    ``finally:`` block to maintain proper LIFO stacking.

    Example:
        >>> token = set_current_run_ctx(None)
        >>> reset_current_run_ctx(token)
    """
    return _current_run_ctx.set(ctx)


def reset_current_run_ctx(token: contextvars.Token[Any]) -> None:
    """Restore the previous :class:`RunContext` value (companion to set).

    Example:
        >>> token = set_current_run_ctx(None)
        >>> reset_current_run_ctx(token)
    """
    _current_run_ctx.reset(token)


@dataclass(slots=True)
class RunContext(Generic[Deps]):
    """Per-run mutable state — passed to every node as the graph state.

    ``messages`` and ``usage`` mutate inside the loop; the other fields are
    fixed once the run starts. Add a field with a typed Protocol or put it
    in ``deps`` rather than growing this surface (RC1 / RC2).
    """

    run_id: str
    agent: Agent[Any, Any] | None
    deps: Deps | None
    messages: list[Message]
    usage: Usage
    sandbox: Sandbox | None
    long_term_memory: LongTermMemory | None
    event_bus: EventBus
    turn: int = 0
    pending_tool_calls: list[ToolCall] = field(default_factory=list)
    pending_approvals: list[ApprovalRequest] = field(default_factory=list)
    permission_mode: PermissionMode = "default"
    pre_plan_mode: PermissionMode | None = None
    prompt_text: str = ""
    output: Any | None = None
    resolved_approvals: dict[str, Approval] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult(Generic[Out]):
    """Result of an :meth:`Agent.run` call.

    ``output`` is ``None`` while the run is paused on approval; the loop
    populates it once the model produces a final-output message.

    Example:
        >>> RunResult(
        ...     output="ok", messages=[], pending_approvals=[], run_state=None, usage=Usage()
        ... ).output
        'ok'
    """

    output: Out | None
    messages: list[Message]
    pending_approvals: list[ApprovalRequest]
    run_state: RunStateSnapshot | None
    usage: Usage


def is_tool_allowed_in_mode(tool: Tool, mode: PermissionMode) -> bool:
    """Per S18: in ``"plan"`` mode only read-only tools are allowed.

    Every other mode lets every tool through; mode-specific approval
    bypasses (``accept_edits`` / ``bypass`` / ``dont_ask``) are handled by
    the approval check in :class:`ToolDispatch`, not by visibility here.

    Example:
        >>> from agent_harness.core.tools import Tool, ToolPolicy
        >>> async def _f() -> None: ...
        >>> ro = Tool(name="r", description="", schema={}, policy=ToolPolicy(is_read_only=True), fn=_f)
        >>> is_tool_allowed_in_mode(ro, "plan")
        True
    """
    if mode == "plan":
        return tool.policy.is_read_only
    return True


def snapshot_from_ctx(ctx: RunContext[Any], *, current_node: str) -> RunStateSnapshot:
    """Build a :class:`RunStateSnapshot` from the current ``ctx`` (LP4)."""
    return RunStateSnapshot(
        run_id=ctx.run_id,
        agent_name=ctx.agent.name if ctx.agent is not None else "",
        current_node=current_node,
        messages=list(ctx.messages),
        pending_tool_calls=list(ctx.pending_tool_calls),
        pending_approvals=list(ctx.pending_approvals),
        usage=ctx.usage.model_copy(),
        turn=ctx.turn,
        deps=None,
        session_id=None,
        permission_mode=ctx.permission_mode,
        pre_plan_mode=ctx.pre_plan_mode,
        created_at=datetime.now(UTC),
    )


def make_interruption(ctx: RunContext[Any]) -> Interruption:
    """Build an :class:`Interruption` snapshotting at ``ToolDispatch``."""
    snap = snapshot_from_ctx(ctx, current_node="ToolDispatch")
    return Interruption(pending_approvals=list(ctx.pending_approvals), run_state=snap)


__all__ = [
    "RunContext",
    "RunResult",
    "get_current_run_ctx",
    "is_tool_allowed_in_mode",
    "make_interruption",
    "reset_current_run_ctx",
    "set_current_run_ctx",
    "snapshot_from_ctx",
]
