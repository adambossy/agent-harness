"""Unit tests for :class:`DecideNext` (W4).

Covers the terminate-or-loop branch and history-processor invocation
(:class:`CompactionStart` / :class:`CompactionEnd`).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_harness.core.agent import Agent
from agent_harness.core.events import (
    CompactionEnd,
    CompactionStart,
    InMemoryEventBus,
    NodeEnter,
)
from agent_harness.core.models import Message
from agent_harness.core.tools import ToolCall
from agent_harness.core.toolsets import StaticToolset
from tests.fakes import FakeTurn, make_model


async def test_decidenext_terminates_on_final_assistant_text() -> None:
    """When the assistant emits text-only and output coerces, run ends."""
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(FakeTurn(text="all done")),
        toolsets=[],
    )
    result = await a.run("hi")
    assert result.output == "all done"


async def test_decidenext_loops_after_tool_call() -> None:
    """A tool call routes through ToolDispatch then back to ModelRequest."""
    from agent_harness.core.tools import Tool, ToolPolicy

    async def echo(text: str) -> str:
        return text

    t = Tool(
        name="echo",
        description="",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        policy=ToolPolicy(is_read_only=True),
        fn=echo,
    )
    ts = StaticToolset(name="t", tools=[t])
    bus = InMemoryEventBus()
    nodes: list[str] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            if isinstance(ev, NodeEnter):
                nodes.append(ev.node)

    task = asyncio.create_task(drain())
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="invoking",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            FakeTurn(text="done"),
        ),
        toolsets=[ts],
    )
    await a.run("go", event_bus=bus)
    await bus.close()
    await task
    assert "PrepareTurn" in nodes
    assert nodes.count("ModelRequest") == 2
    assert "ToolDispatch" in nodes
    assert nodes.count("DecideNext") == 2


async def test_decidenext_invokes_history_processors() -> None:
    """History processors are called between turns; events are emitted."""
    calls: list[str] = []

    def proc(msgs: list[Message]) -> list[Message]:
        calls.append("proc")
        return msgs

    from agent_harness.core.tools import Tool, ToolPolicy

    async def echo(text: str) -> str:
        return text

    t = Tool(
        name="echo",
        description="",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        policy=ToolPolicy(is_read_only=True),
        fn=echo,
    )
    ts = StaticToolset(name="t", tools=[t])
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(
            FakeTurn(
                text="step1",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
            ),
            FakeTurn(text="end"),
        ),
        toolsets=[ts],
        history_processors=[proc],
    )
    await a.run("go", event_bus=bus)
    await bus.close()
    await task

    assert calls  # processor was invoked
    starts = [e for e in collected if isinstance(e, CompactionStart)]
    ends = [e for e in collected if isinstance(e, CompactionEnd)]
    assert len(starts) == len(ends) == len(calls)


async def test_decidenext_no_processors_no_events() -> None:
    """When no processors are configured, no CompactionStart events fire."""
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    a: Agent[Any, Any] = Agent(name="d", model=make_model(FakeTurn(text="hi")), toolsets=[])
    await a.run("hi", event_bus=bus)
    await bus.close()
    await task

    assert not any(isinstance(e, CompactionStart) for e in collected)
