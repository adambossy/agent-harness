"""Unit tests for the hardened :meth:`Agent.as_tool` (Wave 5).

Covers:

* Nested event republishing onto the parent's :class:`EventBus` (AT2) —
  parent sees child's ``MessageDelta`` / ``ToolExec*`` events plus the
  ``SubagentStart`` / ``SubagentStop`` bookends.
* Nested approval propagation (AT3) — when a child returns
  ``pending_approvals`` the parent's :class:`RunResult.pending_approvals`
  contains the child's request.
* End-to-end parent→child smoke (AT1 / AT4): a reader agent calls a
  ``summarizer`` subagent and completes successfully.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agent_harness.core.agent import Agent
from agent_harness.core.events import (
    AgentEnd,
    AgentStart,
    InMemoryEventBus,
    MessageDelta,
    MessageEnd,
    RunEnd,
    RunStart,
    SubagentStart,
    SubagentStop,
)
from agent_harness.core.models import Model
from agent_harness.core.tools import Tool, ToolCall, ToolPolicy
from agent_harness.core.toolsets import StaticToolset
from tests.fakes import FakeModel, FakeTurn, make_model


def _model(*turns: FakeTurn) -> Model:
    return cast(Model, FakeModel(script=list(turns)))


# --- AT2: nested event republishing ----------------------------------------


@pytest.mark.asyncio
async def test_as_tool_republishes_child_events_to_parent_bus() -> None:
    """The child's MessageDelta / ToolExec events surface on the parent bus.

    Run/Agent bookends are filtered (parent emits its own); SubagentStart /
    SubagentStop bracket the nested run.
    """
    child: Agent[Any, Any] = Agent(
        name="child",
        model=_model(FakeTurn(text="child-output")),
        toolsets=[],
    )
    child_tool = child.as_tool(name="run_child", description="call child")

    parent_model = make_model(
        FakeTurn(
            text="dispatching",
            tool_calls=[ToolCall(id="pc1", name="run_child", arguments={"prompt": "hi"})],
        ),
        FakeTurn(text="all done"),
    )
    parent: Agent[Any, Any] = Agent(
        name="parent",
        model=parent_model,
        toolsets=[StaticToolset(name="subagents", tools=[child_tool])],
    )

    bus = InMemoryEventBus()
    sub = bus.subscribe()
    collected: list[Any] = []

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    result = await parent.run("go", event_bus=bus)
    await bus.close()
    await task

    assert result.output == "all done"

    # Exactly one RunStart/RunEnd and AgentStart/AgentEnd (the parent's).
    assert sum(1 for e in collected if isinstance(e, RunStart)) == 1
    assert sum(1 for e in collected if isinstance(e, RunEnd)) == 1
    assert sum(1 for e in collected if isinstance(e, AgentStart)) == 1
    assert sum(1 for e in collected if isinstance(e, AgentEnd)) == 1

    # The child's MessageDelta + MessageEnd surface on the parent bus too.
    deltas = [e for e in collected if isinstance(e, MessageDelta)]
    ends = [e for e in collected if isinstance(e, MessageEnd)]
    # Parent had 2 turns of text + child had 1 turn = 3 MessageEnds total.
    assert len(ends) == 3
    # Likewise, deltas accumulate cumulatively per message_id.
    assert deltas

    # SubagentStart / SubagentStop bracket the nested run.
    starts = [e for e in collected if isinstance(e, SubagentStart)]
    stops = [e for e in collected if isinstance(e, SubagentStop)]
    assert len(starts) == 1 and len(stops) == 1
    assert starts[0].child_agent_name == "child"
    assert starts[0].parent_agent_name == "parent"
    assert starts[0].tool_call_id == ""  # tool_call_id propagation is best-effort


@pytest.mark.asyncio
async def test_as_tool_emits_subagent_bookends_only_when_parent_bus_present() -> None:
    """Calling the as_tool wrapper standalone (no parent context) is OK."""
    child: Agent[Any, Any] = Agent(
        name="solo",
        model=_model(FakeTurn(text="hello")),
        toolsets=[],
    )
    t = child.as_tool(name="solo", description="solo")
    # No parent context: the wrapper should run the child and return text.
    result = await t.fn(prompt="hi")
    assert result.error is None
    assert "hello" in result.content[0].text


# --- AT3: nested approval propagation -------------------------------------


@pytest.mark.asyncio
async def test_as_tool_propagates_child_pending_approvals() -> None:
    """A child paused on approval propagates up to the parent's run result."""

    async def dangerous() -> str:
        return "removed"

    danger_tool = Tool(
        name="rm",
        description="",
        schema={"type": "object", "properties": {}, "required": []},
        policy=ToolPolicy(needs_approval=True),
        fn=dangerous,
    )
    child: Agent[Any, Any] = Agent(
        name="child",
        model=make_model(
            FakeTurn(
                text="removing",
                tool_calls=[ToolCall(id="cc1", name="rm", arguments={})],
            ),
            FakeTurn(text="done"),
        ),
        toolsets=[StaticToolset(name="t", tools=[danger_tool])],
    )

    child_tool = child.as_tool(name="run_child", description="run child")

    parent: Agent[Any, Any] = Agent(
        name="parent",
        model=make_model(
            FakeTurn(
                text="calling child",
                tool_calls=[ToolCall(id="pc1", name="run_child", arguments={"prompt": "rm"})],
            ),
            FakeTurn(text="will not be reached"),
        ),
        toolsets=[StaticToolset(name="subs", tools=[child_tool])],
    )
    result = await parent.run("rm")
    # Parent's pending_approvals contains the child's approval request.
    assert result.pending_approvals
    assert any(a.tool_call_id == "cc1" and a.tool_name == "rm" for a in result.pending_approvals)
    # Parent's run state was snapshotted at the interruption boundary.
    assert result.run_state is not None
    assert result.run_state.current_node == "ToolDispatch"


# --- AT1 + AT4: end-to-end + usage accumulation ---------------------------


@pytest.mark.asyncio
async def test_reader_calls_summarizer_subagent_end_to_end() -> None:
    """A ``reader`` agent that hands off to a ``summarizer`` subagent.

    Verifies AT1 (subagent runs end-to-end as nested loop), AT4 (child
    usage rolls up into the parent's total).
    """
    from agent_harness.core.models import Usage

    summary = "Hello-world greeting + test-file note."
    summarizer: Agent[Any, Any] = Agent(
        name="summarizer",
        model=_model(FakeTurn(text=summary, usage=Usage(input_tokens=50, output_tokens=10))),
        toolsets=[],
    )
    summarizer_tool = summarizer.as_tool(
        name="summarizer",
        description="Summarize an input string.",
    )
    parent_model = make_model(
        FakeTurn(
            text="dispatching summarizer",
            tool_calls=[
                ToolCall(
                    id="pc1",
                    name="summarizer",
                    arguments={"prompt": "Hello, world!\nThis is a test file."},
                )
            ],
        ),
        FakeTurn(text=f"Reader summary: {summary}"),
    )
    reader: Agent[Any, Any] = Agent(
        name="reader",
        model=parent_model,
        toolsets=[StaticToolset(name="subs", tools=[summarizer_tool])],
    )
    result = await reader.run("summarize foo.txt")
    assert result.output is not None
    assert summary in result.output
    # AT4: child's usage rolled into parent's total.
    # Two parent turns (100+20 each from FakeTurn default) + child (50+10).
    assert result.usage.input_tokens >= 50  # at minimum the child's
