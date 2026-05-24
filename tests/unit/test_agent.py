"""Unit tests for :class:`Agent` and the public surface (W4).

Covers AG1 / AG2 / AG3 / AG4 / AG6 / AG7 / AG8 / AG9 plus the
``is_tool_allowed_in_mode`` helper.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agent_harness.core.agent import Agent, RunContext, RunResult, is_tool_allowed_in_mode
from agent_harness.core.errors import ConfigError
from agent_harness.core.events import (
    AgentEnd,
    AgentStart,
    InMemoryEventBus,
    MessageDelta,
    RunEnd,
    RunStart,
)
from agent_harness.core.models import Model
from agent_harness.core.tools import Tool, ToolCall, ToolPolicy
from agent_harness.sessions.inmemory import InMemorySession
from tests.fakes import FakeModel, FakeSandbox, FakeTurn


def _model(*turns: FakeTurn) -> Model:
    """Build a ``FakeModel`` cast to the ``Model`` Protocol.

    The cast is necessary because ``FakeModel.request`` is an async generator
    function while ``Model.request`` is declared ``async def -> AsyncIterator``;
    mypy can't bridge the two but runtime ``isinstance(.., Model)`` is True.
    """
    return cast(Model, FakeModel(script=list(turns)))


def _agent(**kwargs: Any) -> Agent[Any, Any]:
    """Helper for constructing test agents with sensible defaults."""
    return Agent(
        name=kwargs.pop("name", "demo"),
        model=kwargs.pop("model", _model(FakeTurn(text="hi"))),
        toolsets=kwargs.pop("toolsets", []),
        **kwargs,
    )


# --- construction ----------------------------------------------------------


def test_agent_construction_minimal() -> None:
    a = _agent()
    assert a.name == "demo"
    assert a.toolsets == []
    assert a.instructions == ""
    assert a.output_type is str


def test_agent_rejects_empty_name() -> None:
    with pytest.raises(ConfigError):
        Agent(name="", model=_model(), toolsets=[])


def test_agent_holds_no_mutable_state() -> None:
    """AG9: re-running an agent twice with different prompts must not leak."""
    a = _agent(model=_model(FakeTurn(text="one"), FakeTurn(text="two")))
    r1 = asyncio.run(a.run("first"))
    r2 = asyncio.run(a.run("second"))
    assert r1.output != r2.output
    # No mutable state on the agent itself other than config.
    assert "messages" not in vars(a)


# --- run ------------------------------------------------------------------


async def test_run_returns_typed_result() -> None:
    """AG1 / AG7: result has the expected shape."""
    a = _agent(model=_model(FakeTurn(text="hello world")))
    result = await a.run("ping")
    assert isinstance(result, RunResult)
    assert result.output == "hello world"
    assert result.pending_approvals == []
    assert result.usage.input_tokens > 0
    assert any(m.role == "user" for m in result.messages)


async def test_run_publishes_lifecycle_events() -> None:
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    a = _agent(model=_model(FakeTurn(text="ok")))
    await a.run("hi", event_bus=bus)
    await bus.close()
    await task

    types = {type(ev) for ev in collected}
    assert RunStart in types
    assert AgentStart in types
    assert AgentEnd in types
    assert RunEnd in types
    # MessageDelta carries the cumulative-partial invariant.
    deltas = [ev for ev in collected if isinstance(ev, MessageDelta)]
    assert deltas
    prev = ""
    for d in deltas:
        assert d.partial.text.startswith(prev)
        prev = d.partial.text


async def test_run_persists_messages_to_session() -> None:
    session = InMemorySession(session_id="s1")
    a = _agent(model=_model(FakeTurn(text="ok")), session=session)
    await a.run("hello")
    msgs = await session.get_messages()
    assert any(m.role == "user" for m in msgs)
    assert any(m.role == "assistant" for m in msgs)
    snaps = await session.get_run_states()
    # PrepareTurn -> ModelRequest -> DecideNext (no tool calls path).
    assert len(snaps) >= 3


# --- iter / stream --------------------------------------------------------


async def test_iter_yields_nodes() -> None:
    """AG2: ``iter`` exposes each step."""
    a = _agent(model=_model(FakeTurn(text="ok")))
    seen: list[str] = []
    async for step in a.iter("hi"):
        seen.append(type(step).__name__)
    # The 3 nodes for a no-tool-call path: PrepareTurn, ModelRequest, DecideNext + End.
    assert "PrepareTurn" in seen
    assert "ModelRequest" in seen
    assert "DecideNext" in seen
    assert "End" in seen


async def test_stream_yields_events() -> None:
    """AG3: ``stream`` exposes events with cumulative partials."""
    a = _agent(model=_model(FakeTurn(text="ok")))
    seen: list[Any] = []
    async for ev in a.stream("hi"):
        seen.append(ev)
    assert any(isinstance(e, RunStart) for e in seen)
    assert any(isinstance(e, RunEnd) for e in seen)


# --- as_tool --------------------------------------------------------------


def test_as_tool_produces_callable_tool() -> None:
    """AG4: ``as_tool`` wraps an agent as a :class:`Tool`."""
    child = _agent(model=_model(FakeTurn(text="child-result")))
    t = child.as_tool(name="run_child", description="invoke the child agent")
    assert isinstance(t, Tool)
    assert t.name == "run_child"
    assert "prompt" in t.schema.get("properties", {})


async def test_as_tool_invokes_child_run() -> None:
    """AG4: wrapped tool actually runs the child agent and returns its output."""
    child = _agent(model=_model(FakeTurn(text="child-result")))
    t = child.as_tool(name="ct", description="child")
    result = await t.fn(prompt="hello")
    assert result.error is None
    assert "child-result" in result.content[0].text


# --- resume ---------------------------------------------------------------


async def test_resume_round_trip_from_snapshot() -> None:
    """AG6: a paused run resumes from a snapshot."""

    # Set up a tool that needs approval.
    async def dangerous(path: str) -> str:
        return f"removed {path}"

    tool = Tool(
        name="rm",
        description="",
        schema={"type": "object", "properties": {"path": {"type": "string"}}},
        policy=ToolPolicy(needs_approval=True),
        fn=dangerous,
    )

    from agent_harness.core.toolsets import StaticToolset

    ts = StaticToolset(name="t", tools=[tool])
    model = _model(
        FakeTurn(
            text="removing...",
            tool_calls=[ToolCall(id="c1", name="rm", arguments={"path": "/tmp/foo"})],
        ),
        FakeTurn(text="done"),
    )
    a = _agent(model=model, toolsets=[ts])
    paused = await a.run("delete foo")
    assert paused.pending_approvals  # ran into approval gate
    assert paused.run_state is not None

    # Resume with approval.
    from agent_harness.core.run_state import Approval

    resumed = await Agent.resume(
        paused.run_state,
        agent=a,
        approvals=[Approval(tool_call_id="c1", approve=True)],
    )
    assert resumed.output == "done"
    # Tool result messages carry ToolResultBlock content (not TextBlock).
    tool_msgs = [m for m in resumed.messages if m.role == "tool"]
    assert tool_msgs
    from agent_harness.core.models import ToolResultBlock

    contents: list[Any] = [
        b.content for m in tool_msgs for b in m.content if isinstance(b, ToolResultBlock)
    ]
    assert any("removed" in str(c) for c in contents)


# --- helpers --------------------------------------------------------------


def test_is_tool_allowed_in_mode_plan_filters_writes() -> None:
    async def f() -> None:
        return None

    ro = Tool(
        name="r",
        description="",
        schema={},
        policy=ToolPolicy(is_read_only=True),
        fn=f,
    )
    rw = Tool(
        name="w",
        description="",
        schema={},
        policy=ToolPolicy(),
        fn=f,
    )
    assert is_tool_allowed_in_mode(ro, "plan")
    assert not is_tool_allowed_in_mode(rw, "plan")
    assert is_tool_allowed_in_mode(rw, "default")
    assert is_tool_allowed_in_mode(rw, "bypass")


def test_run_context_construction() -> None:
    from agent_harness.core.models import Usage

    bus = InMemoryEventBus()
    rc = RunContext[Any](
        run_id="r",
        agent=None,
        deps=None,
        messages=[],
        usage=Usage(),
        sandbox=None,
        long_term_memory=None,
        event_bus=bus,
    )
    assert rc.turn == 0


def test_run_with_sandbox_and_filesystem(tmp_path: Any) -> None:
    """Smoke that an agent + filesystem tools + sandbox composes."""
    from agent_harness.core.filesystem import FilesystemTools

    sandbox = FakeSandbox()

    async def setup() -> RunResult[Any]:
        await sandbox.write_file("x.txt", "value")
        model = _model(
            FakeTurn(
                text="reading",
                tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "x.txt"})],
            ),
            FakeTurn(text="the value"),
        )
        a = _agent(model=model, toolsets=[FilesystemTools(sandbox=sandbox)])
        return await a.run("what is x.txt?")

    result = asyncio.run(setup())
    assert result.output == "the value"
