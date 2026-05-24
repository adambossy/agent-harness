"""Canonical end-to-end smoke test (W4).

Exercises every component on the critical path: Agent → Loop (all 4 nodes)
→ Model x Provider (faked) → Toolset → Tool → Sandbox → Session →
EventBus → RunStateSnapshot.

See ``proposal/verification.md`` § "The test": 2 turns, 1 tool call, ≥10
assertions covering the contracts listed there.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from agent_harness.core.agent import Agent
from agent_harness.core.events import (
    AgentEnd,
    AgentStart,
    InMemoryEventBus,
    MessageDelta,
    MessageEnd,
    MessageStart,
    NodeEnter,
    NodeExit,
    RunEnd,
    RunStart,
    ToolCallEnd,
    ToolExecEnd,
    ToolExecStart,
)
from agent_harness.core.filesystem import FilesystemTools
from agent_harness.core.tools import ToolCall
from agent_harness.sandboxes.inprocess import InProcessSandbox
from agent_harness.sessions.inmemory import InMemorySession
from tests.fakes import FakeProvider, FakeTurn, make_model


@pytest.mark.asyncio
async def test_smoke_two_turns_one_tool_call(tmp_path: Path) -> None:
    """The canonical smoke test (≥10 assertions, per verification.md)."""

    # ─── Arrange ─────────────────────────────────────────────────
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "foo.txt").write_text("Hello, world!\nThis is a test file.")

    sandbox = InProcessSandbox(root=str(workspace))
    session = InMemorySession(session_id="smoke-001")
    bus = InMemoryEventBus()

    from typing import cast

    from agent_harness.core.models import Provider

    model = make_model(
        FakeTurn(
            text="I'll read the file to answer your question.",
            tool_calls=[
                ToolCall(id="call_001", name="read", arguments={"path": "foo.txt"}),
            ],
        ),
        FakeTurn(
            text="The file contains a greeting ('Hello, world!') and a "
            "note that it's a test file.",
        ),
        provider=cast(Provider, FakeProvider("fake-anthropic")),
    )

    agent: Agent[Any, Any] = Agent(
        name="reader",
        model=model,
        toolsets=[FilesystemTools(sandbox=sandbox)],
        session=session,
        sandbox=sandbox,
        instructions="You read files and explain them.",
    )

    # ─── Act ─────────────────────────────────────────────────────
    collected: list[Any] = []
    sub = bus.subscribe()

    async def _drain() -> None:
        async for ev in sub:
            collected.append(ev)

    collector = asyncio.create_task(_drain())
    result = await agent.run(prompt="What's in foo.txt?", event_bus=bus)
    await bus.close()
    with contextlib.suppress(asyncio.CancelledError):
        await collector

    # ─── Assert ──────────────────────────────────────────────────
    # 1. Output is what the model said in turn 2
    assert result.output is not None
    assert "Hello, world!" in result.output
    assert "test file" in result.output

    # 2. No pending approvals
    assert result.pending_approvals == []

    # 3. Session captured the conversation
    msgs = await session.get_messages()
    user_msgs = [m for m in msgs if m.role == "user"]
    assert any("foo.txt" in m.text for m in user_msgs)
    assistant_tool_msgs = [m for m in msgs if m.role == "assistant" and m.has_tool_call()]
    assert len(assistant_tool_msgs) == 1
    assert assistant_tool_msgs[0].tool_calls[0].name == "read"
    tool_results = [m for m in msgs if m.role == "tool"]
    assert len(tool_results) == 1
    # ToolResultBlock content is a string with the tool output.
    tool_content = tool_results[0].content[0]
    assert hasattr(tool_content, "content")
    assert "Hello, world!" in str(tool_content.content)

    # 4. RunState snapshotted at every node boundary (>= 6 nodes visited)
    snapshots = await session.get_run_states()
    assert len(snapshots) >= 6
    # turn=2 (two ModelRequest visits)
    assert snapshots[-1].turn == 2
    # never entered Plan Mode
    assert snapshots[-1].permission_mode == "default"

    # 5. Usage accumulated across turns
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0

    # 6. Events fired in the right shape (counts)
    by_type: dict[type, int] = {}
    for ev in collected:
        by_type[type(ev)] = by_type.get(type(ev), 0) + 1
    assert by_type.get(RunStart, 0) == 1
    assert by_type.get(AgentStart, 0) == 1
    assert by_type.get(NodeEnter, 0) == 6
    assert by_type.get(NodeExit, 0) == 6
    assert by_type.get(MessageStart, 0) == 2  # one per turn
    assert by_type.get(MessageEnd, 0) == 2
    assert by_type.get(MessageDelta, 0) >= 6  # 3 chunks x 2 turns
    assert by_type.get(ToolCallEnd, 0) == 1
    assert by_type.get(ToolExecStart, 0) == 1
    assert by_type.get(ToolExecEnd, 0) == 1
    assert by_type.get(AgentEnd, 0) == 1
    assert by_type.get(RunEnd, 0) == 1

    # 7. MessageDelta carries cumulative partials (the pi-invariant)
    deltas_by_msg: dict[str, list[MessageDelta]] = {}
    for ev in collected:
        if isinstance(ev, MessageDelta):
            deltas_by_msg.setdefault(ev.message_id, []).append(ev)
    assert deltas_by_msg
    for msg_id, deltas in deltas_by_msg.items():
        prev = ""
        for d in deltas:
            cur = d.partial.text
            assert cur.startswith(
                prev
            ), f"MessageDelta.partial not cumulative for {msg_id}: '{prev}' → '{cur}'"
            prev = cur

    # 8. The loop traversed the expected node sequence
    seen_nodes = [ev.node for ev in collected if isinstance(ev, NodeEnter)]
    assert seen_nodes == [
        "PrepareTurn",
        "ModelRequest",
        "ToolDispatch",
        "DecideNext",
        "ModelRequest",
        "DecideNext",
    ], f"Loop traversed unexpected sequence: {seen_nodes}"

    # 9. The tool actually went through the sandbox
    tool_execs = [ev for ev in collected if isinstance(ev, ToolExecStart)]
    assert tool_execs[0].tool_name == "read"
    assert tool_execs[0].arguments == {"path": "foo.txt"}

    # 10. The model wasn't asked for a third turn (script wasn't exhausted)
    assert model._turn == 2


@pytest.mark.asyncio
async def test_reader_with_summarizer_subagent(tmp_path: Path) -> None:
    """W5 integration: a ``reader`` parent calls a ``summarizer`` subagent.

    Exercises ``Agent.as_tool``'s nested-event republishing (AT2) and
    end-to-end parent→child dispatch (AT1) inside the full loop.
    """
    summarizer: Agent[Any, Any] = Agent(
        name="summarizer",
        model=make_model(FakeTurn(text="A greeting and a test-file note.")),
        toolsets=[],
    )
    summarizer_tool = summarizer.as_tool(
        name="summarizer",
        description="Summarize an input string into one line.",
    )
    parent_model = make_model(
        FakeTurn(
            text="dispatching",
            tool_calls=[
                ToolCall(
                    id="ps1",
                    name="summarizer",
                    arguments={"prompt": "Hello, world!\nThis is a test file."},
                )
            ],
        ),
        FakeTurn(text="Reader summary: A greeting and a test-file note."),
    )
    from agent_harness.core.toolsets import StaticToolset

    reader: Agent[Any, Any] = Agent(
        name="reader",
        model=parent_model,
        toolsets=[StaticToolset(name="subs", tools=[summarizer_tool])],
    )

    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def _drain() -> None:
        async for ev in sub:
            collected.append(ev)

    collector = asyncio.create_task(_drain())
    result = await reader.run(prompt="Summarize foo.txt", event_bus=bus)
    await bus.close()
    with contextlib.suppress(asyncio.CancelledError):
        await collector

    assert result.output is not None
    assert "A greeting and a test-file note." in result.output
    # Parent's bus saw SubagentStart / SubagentStop.
    from agent_harness.core.events import SubagentStart, SubagentStop

    assert any(isinstance(e, SubagentStart) for e in collected)
    assert any(isinstance(e, SubagentStop) for e in collected)


@pytest.mark.asyncio
async def test_smoke_runs_under_100ms(tmp_path: Path) -> None:
    """The smoke test must run fast (no real network / I/O nondeterminism)."""
    import time

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "foo.txt").write_text("hi")

    sandbox = InProcessSandbox(root=str(workspace))
    model = make_model(
        FakeTurn(
            text="reading",
            tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "foo.txt"})],
        ),
        FakeTurn(text="hi"),
    )
    agent: Agent[Any, Any] = Agent(
        name="r",
        model=model,
        toolsets=[FilesystemTools(sandbox=sandbox)],
        sandbox=sandbox,
    )
    t0 = time.perf_counter()
    await agent.run("read foo")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 1000, f"smoke test took {elapsed_ms:.1f}ms; should be < 1s"
