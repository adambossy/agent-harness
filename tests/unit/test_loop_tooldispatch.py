"""Unit tests for :class:`ToolDispatch` (W4).

Covers parallel batching, approval flow + Interruption, timeout handling,
and PermissionMode enforcement.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_harness.core.agent import Agent
from agent_harness.core.events import (
    ApprovalRequested,
    InMemoryEventBus,
    ToolExecEnd,
    ToolExecStart,
)
from agent_harness.core.run_state import Approval
from agent_harness.core.tools import Tool, ToolCall, ToolPolicy
from agent_harness.core.toolsets import StaticToolset
from tests.fakes import FakeTurn, make_model

# --- helpers --------------------------------------------------------------


def _make_tool(
    name: str,
    *,
    fn: Any,
    needs_approval: bool = False,
    is_concurrency_safe: bool = False,
    is_read_only: bool = False,
    timeout_seconds: float | None = None,
) -> Tool:
    return Tool(
        name=name,
        description="",
        schema={"type": "object", "properties": {}, "required": []},
        policy=ToolPolicy(
            needs_approval=needs_approval,
            is_concurrency_safe=is_concurrency_safe,
            is_read_only=is_read_only,
            timeout_seconds=timeout_seconds,
        ),
        fn=fn,
    )


# --- approval flow --------------------------------------------------------


async def test_tooldispatch_pauses_on_approval() -> None:
    """A needs_approval=True tool pauses the run via Interruption."""
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())

    async def dangerous() -> str:
        return "should not run"

    t = _make_tool("rm", fn=dangerous, needs_approval=True)
    ts = StaticToolset(name="t", tools=[t])
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="...",
                tool_calls=[ToolCall(id="c1", name="rm", arguments={})],
            ),
        ),
        toolsets=[ts],
    )
    result = await a.run("rm", event_bus=bus)
    await bus.close()
    await task

    assert len(result.pending_approvals) == 1
    assert result.pending_approvals[0].tool_call_id == "c1"
    assert any(isinstance(e, ApprovalRequested) for e in collected)
    # Tool never executed.
    assert not any(isinstance(e, ToolExecStart) for e in collected)


async def test_tooldispatch_resumes_after_approval() -> None:
    """A second call to ``run`` with ``approvals=`` resumes the paused run."""

    async def dangerous() -> str:
        return "removed"

    t = _make_tool("rm", fn=dangerous, needs_approval=True)
    ts = StaticToolset(name="t", tools=[t])
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="rm-ing",
                tool_calls=[ToolCall(id="c1", name="rm", arguments={})],
            ),
            FakeTurn(text="done"),
        ),
        toolsets=[ts],
    )
    paused = await a.run("rm")
    assert paused.run_state is not None
    resumed = await Agent.resume(
        paused.run_state,
        agent=a,
        approvals=[Approval(tool_call_id="c1", approve=True)],
    )
    assert resumed.output == "done"


async def test_tooldispatch_deny_approval_produces_error_result() -> None:
    """A denied approval becomes an errored tool result."""

    async def dangerous() -> str:
        return "removed"

    t = _make_tool("rm", fn=dangerous, needs_approval=True)
    ts = StaticToolset(name="t", tools=[t])
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="rm-ing",
                tool_calls=[ToolCall(id="c1", name="rm", arguments={})],
            ),
            FakeTurn(text="cannot remove"),
        ),
        toolsets=[ts],
    )
    paused = await a.run("rm")
    assert paused.run_state is not None
    resumed = await Agent.resume(
        paused.run_state,
        agent=a,
        approvals=[Approval(tool_call_id="c1", approve=False, rationale="no way")],
    )
    assert resumed.output == "cannot remove"


# --- concurrency ----------------------------------------------------------


async def test_tooldispatch_parallel_safe_tools() -> None:
    """Concurrency-safe tools run in parallel; unsafe serial."""
    started: list[str] = []

    async def safe_a() -> str:
        started.append("a-start")
        await asyncio.sleep(0.01)
        started.append("a-end")
        return "A"

    async def safe_b() -> str:
        started.append("b-start")
        await asyncio.sleep(0.01)
        started.append("b-end")
        return "B"

    ts = StaticToolset(
        name="t",
        tools=[
            _make_tool("a", fn=safe_a, is_concurrency_safe=True),
            _make_tool("b", fn=safe_b, is_concurrency_safe=True),
        ],
    )
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="running",
                tool_calls=[
                    ToolCall(id="c1", name="a", arguments={}),
                    ToolCall(id="c2", name="b", arguments={}),
                ],
            ),
            FakeTurn(text="both done"),
        ),
        toolsets=[ts],
    )
    await a.run("go")
    # In parallel both starts precede both ends.
    a_start = started.index("a-start")
    b_start = started.index("b-start")
    a_end = started.index("a-end")
    b_end = started.index("b-end")
    # Both started before either ended (parallel).
    assert max(a_start, b_start) < min(a_end, b_end)


async def test_tooldispatch_unsafe_tools_run_serially() -> None:
    """Tools marked unsafe (default) run one at a time."""
    started: list[str] = []

    async def unsafe_a() -> str:
        started.append("a-start")
        await asyncio.sleep(0.01)
        started.append("a-end")
        return "A"

    async def unsafe_b() -> str:
        started.append("b-start")
        await asyncio.sleep(0.01)
        started.append("b-end")
        return "B"

    ts = StaticToolset(
        name="t",
        tools=[
            _make_tool("a", fn=unsafe_a),
            _make_tool("b", fn=unsafe_b),
        ],
    )
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="running",
                tool_calls=[
                    ToolCall(id="c1", name="a", arguments={}),
                    ToolCall(id="c2", name="b", arguments={}),
                ],
            ),
            FakeTurn(text="done"),
        ),
        toolsets=[ts],
    )
    await a.run("go")
    # Serial: a finishes before b starts.
    assert started == ["a-start", "a-end", "b-start", "b-end"]


# --- timeout --------------------------------------------------------------


async def test_tooldispatch_tool_timeout() -> None:
    """A tool exceeding its timeout becomes an errored result."""

    async def slow() -> str:
        await asyncio.sleep(1.0)
        return "never"

    t = _make_tool("slow", fn=slow, timeout_seconds=0.05)
    ts = StaticToolset(name="t", tools=[t])
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="running",
                tool_calls=[ToolCall(id="c1", name="slow", arguments={})],
            ),
            FakeTurn(text="timed out"),
        ),
        toolsets=[ts],
    )
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    await a.run("slow", event_bus=bus)
    await bus.close()
    await task
    # The ToolExecEnd event reports an error.
    exec_ends = [e for e in collected if isinstance(e, ToolExecEnd)]
    assert exec_ends
    assert any(e.error is not None and "timed out" in e.error for e in exec_ends)


# --- permission mode ------------------------------------------------------


async def test_tooldispatch_plan_mode_rejects_writes() -> None:
    """In plan mode a non-read-only tool is rejected before dispatch."""

    invocations: list[str] = []

    async def write() -> str:
        invocations.append("called")
        return "wrote"

    t = _make_tool("write", fn=write)
    ts = StaticToolset(name="t", tools=[t])
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="writing",
                tool_calls=[ToolCall(id="c1", name="write", arguments={})],
            ),
            FakeTurn(text="cannot write"),
        ),
        toolsets=[ts],
    )
    # Construct a snapshot with permission_mode=plan and resume from it.
    from datetime import UTC, datetime

    from agent_harness.core.models import Usage
    from agent_harness.core.run_state import RunStateSnapshot

    snap = RunStateSnapshot(
        run_id="r1",
        agent_name=a.name,
        current_node="PrepareTurn",
        messages=[],
        pending_tool_calls=[],
        pending_approvals=[],
        usage=Usage(),
        turn=0,
        permission_mode="plan",
        created_at=datetime.now(UTC),
    )
    result = await a.run("write something", run_state=snap)
    # Tool never invoked.
    assert invocations == []
    assert result.output == "cannot write"
