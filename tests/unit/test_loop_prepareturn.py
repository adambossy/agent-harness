"""Unit tests for :class:`PrepareTurn` (W4).

Each node test runs the loop and inspects the node-boundary effects rather
than calling ``node.run`` directly, because pydantic-graph's GraphRunContext
isn't trivially mockable. We use a 1-turn FakeModel script so the loop
terminates after one PrepareTurn → ModelRequest → DecideNext path.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_harness.core.agent import Agent
from agent_harness.core.events import InMemoryEventBus, NodeEnter
from agent_harness.core.models import Message
from agent_harness.sessions.inmemory import InMemorySession
from tests.fakes import FakeTurn, make_model


async def test_prepareturn_appends_user_prompt_to_session() -> None:
    """The node appends the user prompt to the session messages."""
    session = InMemorySession(session_id="s1")
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(FakeTurn(text="hi")),
        toolsets=[],
        session=session,
    )
    await a.run("hello there")
    msgs = await session.get_messages()
    assert msgs[0].role == "user" or any(m.role == "user" for m in msgs)
    user_msgs = [m for m in msgs if m.role == "user"]
    assert any("hello there" in m.text for m in user_msgs)


async def test_prepareturn_applies_system_instructions() -> None:
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(FakeTurn(text="ok")),
        toolsets=[],
        instructions="You are concise.",
    )
    result = await a.run("hi")
    sys_msgs = [m for m in result.messages if m.role == "system"]
    assert any("concise" in m.text for m in sys_msgs)


async def test_prepareturn_publishes_node_enter_exit() -> None:
    """PrepareTurn emits a NodeEnter event."""
    bus = InMemoryEventBus()
    collected: list[NodeEnter] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            if isinstance(ev, NodeEnter):
                collected.append(ev)

    task = asyncio.create_task(drain())
    a: Agent[Any, Any] = Agent(name="d", model=make_model(FakeTurn(text="ok")), toolsets=[])
    await a.run("hi", event_bus=bus)
    await bus.close()
    await task

    assert collected[0].node == "PrepareTurn"


async def test_prepareturn_loads_existing_session_messages() -> None:
    """When the session already has messages, PrepareTurn loads them."""
    from datetime import UTC, datetime

    from agent_harness.core.models import TextBlock

    session = InMemorySession(session_id="s2")
    prior = Message(
        role="user",
        content=[TextBlock(text="prior context")],
        timestamp=datetime.now(UTC),
    )
    await session.add_messages([prior])
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(FakeTurn(text="ok")),
        toolsets=[],
        session=session,
    )
    result = await a.run("now")
    # Prior message is in the final messages list.
    assert any("prior context" in m.text for m in result.messages)


async def test_prepareturn_message_list_prompt() -> None:
    """A run with a list of Message objects is honored as the starting history."""
    from datetime import UTC, datetime

    from agent_harness.core.models import TextBlock

    msgs = [
        Message(role="user", content=[TextBlock(text="bootstrap")], timestamp=datetime.now(UTC)),
    ]
    a: Agent[Any, Any] = Agent(name="d", model=make_model(FakeTurn(text="ack")), toolsets=[])
    result = await a.run(msgs)
    # The bootstrap user msg is preserved; final assistant is "ack".
    assert any("bootstrap" in m.text for m in result.messages)
    assert result.output == "ack"
