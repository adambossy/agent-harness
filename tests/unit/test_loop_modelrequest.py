"""Unit tests for :class:`ModelRequest` (W4).

Covers the model-event republication, usage accumulation, and the
output-validation retry path (which exercises the :class:`ModelRetryRequest`
event and the ``ModelRequest -> ModelRequest`` edge).
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from agent_harness.core.agent import Agent
from agent_harness.core.events import (
    InMemoryEventBus,
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelRetryRequest,
    ModelStart,
)
from tests.fakes import FakeModel, FakeTurn, make_model


async def test_modelrequest_republishes_model_events() -> None:
    """The node forwards all ModelStart/MessageStart/Delta/End events."""
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    a: Agent[Any, Any] = Agent(
        name="d", model=make_model(FakeTurn(text="hello world")), toolsets=[]
    )
    await a.run("hi", event_bus=bus)
    await bus.close()
    await task

    types = {type(ev) for ev in collected}
    assert ModelStart in types
    assert MessageStart in types
    assert MessageEnd in types
    assert ModelEnd in types
    deltas = [ev for ev in collected if isinstance(ev, MessageDelta)]
    assert deltas
    # Cumulative-partial invariant.
    text = ""
    for d in deltas:
        cur = d.partial.text
        assert cur.startswith(text)
        text = cur


async def test_modelrequest_accumulates_usage() -> None:
    """Usage from each MessageEnd is summed into RunResult."""
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(FakeTurn(text="hi")),
        toolsets=[],
    )
    result = await a.run("hi")
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20


async def test_modelrequest_retry_on_output_validation_failure() -> None:
    """When output_type can't coerce, the node loops via ModelRetryRequest.

    The structured-output retry path. FakeModel turn 1 emits a non-integer;
    turn 2 emits a string that ``int()`` accepts.
    """
    bus = InMemoryEventBus()
    collected: list[Any] = []
    sub = bus.subscribe()

    async def drain() -> None:
        async for ev in sub:
            collected.append(ev)

    task = asyncio.create_task(drain())
    a: Agent[Any, int] = Agent(
        name="counter",
        model=make_model(
            FakeTurn(text="five words!"),  # not int-parseable
            FakeTurn(text="5"),
        ),
        toolsets=[],
        output_type=int,
    )
    result = await a.run("count words", event_bus=bus)
    await bus.close()
    await task

    assert result.output == 5
    retries = [ev for ev in collected if isinstance(ev, ModelRetryRequest)]
    assert retries, "expected at least one ModelRetryRequest"


async def test_modelrequest_string_output_default() -> None:
    """The default ``output_type=str`` accepts any text without retry."""
    a: Agent[Any, Any] = Agent(
        name="d",
        model=make_model(FakeTurn(text="anything goes")),
        toolsets=[],
    )
    result = await a.run("hi")
    assert result.output == "anything goes"


async def test_modelrequest_with_no_final_message_terminates() -> None:
    """An empty model script triggers an Error and graceful termination."""

    class _EmptyModel:
        name = "empty"
        provider = None
        capabilities = FakeModel.capabilities

        async def request(self, messages: Any, tools: Any, settings: Any) -> Any:
            return
            yield  # makes this an async generator

        async def compact_messages(self, msgs: Any) -> Any:
            return msgs

    a: Agent[Any, Any] = Agent(name="d", model=cast(Any, _EmptyModel()), toolsets=[])
    # The loop terminates without raising — the End() emits a RunResult.
    result = await a.run("hi")
    assert isinstance(result.usage.input_tokens, int)


async def test_modelrequest_increments_turn() -> None:
    """``ctx.turn`` is incremented at the start of each ModelRequest visit."""
    a: Agent[Any, int] = Agent(
        name="d",
        model=make_model(
            FakeTurn(text="five"),
            FakeTurn(text="5"),
        ),
        toolsets=[],
        output_type=int,
    )
    result = await a.run("count")
    # The terminal snapshot's turn should reflect 2 visits to ModelRequest.
    assert result.run_state is not None
    assert result.run_state.turn >= 2
