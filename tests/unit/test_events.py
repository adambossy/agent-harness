"""Unit tests for ``agent_harness.core.events``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import get_args

import pytest

from agent_harness.core.errors import BusClosedError, ConfigError
from agent_harness.core.events import (
    DEFAULT_QUEUE_MAXSIZE,
    AgentEnd,
    AgentStart,
    ApprovalRequested,
    ApprovalResolved,
    CompactionEnd,
    CompactionStart,
    ElicitationRequested,
    Error,
    Event,
    EventBus,
    InMemoryEventBus,
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelRetryRequest,
    ModelStart,
    ModelUsage,
    NodeEnter,
    NodeExit,
    RunEnd,
    RunStart,
    SubagentStart,
    SubagentStop,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolExecEnd,
    ToolExecStart,
)
from agent_harness.core.models import Message, TextBlock, Usage


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Union shape / exhaustiveness
# ---------------------------------------------------------------------------


EXPECTED_EVENT_MEMBERS = {
    RunStart,
    RunEnd,
    AgentStart,
    AgentEnd,
    NodeEnter,
    NodeExit,
    ModelStart,
    MessageStart,
    MessageDelta,
    MessageEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    ModelEnd,
    ModelUsage,
    ModelRetryRequest,
    ToolExecStart,
    ToolExecEnd,
    SubagentStart,
    SubagentStop,
    ApprovalRequested,
    ApprovalResolved,
    CompactionStart,
    CompactionEnd,
    ElicitationRequested,
    Error,
}


def test_event_union_is_closed_set() -> None:
    members = set(get_args(Event))
    assert members == EXPECTED_EVENT_MEMBERS


def test_no_handoff_events_in_union() -> None:
    """Per open-questions decision #4, Handoff was dropped from v1."""

    member_names = {cls.__name__ for cls in get_args(Event)}
    assert "HandoffStart" not in member_names
    assert "HandoffEnd" not in member_names


def test_subagent_events_present() -> None:
    member_names = {cls.__name__ for cls in get_args(Event)}
    assert "SubagentStart" in member_names
    assert "SubagentStop" in member_names


# ---------------------------------------------------------------------------
# InMemoryEventBus: basic publish / subscribe roundtrip
# ---------------------------------------------------------------------------


async def test_publish_then_subscribe_roundtrip() -> None:
    bus = InMemoryEventBus()
    iterator = bus.subscribe()
    await bus.publish(AgentStart(agent_name="root"))
    await bus.publish(AgentEnd(agent_name="root"))
    await bus.close()

    seen: list[Event] = []
    async for ev in iterator:
        seen.append(ev)

    assert len(seen) == 2
    assert isinstance(seen[0], AgentStart)
    assert isinstance(seen[1], AgentEnd)


async def test_single_subscriber_fifo_under_load() -> None:
    """EV6: single-subscriber FIFO is guaranteed."""

    bus = InMemoryEventBus()
    iterator = bus.subscribe()
    expected = [NodeEnter(node="ModelRequest", turn=i) for i in range(20)]
    for ev in expected:
        await bus.publish(ev)
    await bus.close()

    seen: list[Event] = [ev async for ev in iterator]
    assert [e.turn for e in seen if isinstance(e, NodeEnter)] == list(range(20))


async def test_multi_subscriber_independent_queues() -> None:
    """EV2: each subscriber gets its own queue."""

    bus = InMemoryEventBus()
    a = bus.subscribe()
    b = bus.subscribe()
    for i in range(5):
        await bus.publish(NodeEnter(node="X", turn=i))
    await bus.close()

    out_a: list[Event] = [ev async for ev in a]
    out_b: list[Event] = [ev async for ev in b]
    assert [e.turn for e in out_a if isinstance(e, NodeEnter)] == [0, 1, 2, 3, 4]
    assert [e.turn for e in out_b if isinstance(e, NodeEnter)] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Cumulative partial on MessageDelta
# ---------------------------------------------------------------------------


async def test_message_delta_carries_cumulative_partial() -> None:
    """EV5: a consumer should be able to read ``partial`` and never have to
    reconstruct it from prior deltas."""

    bus = InMemoryEventBus()
    iterator = bus.subscribe()

    base = Message(role="assistant", content=[], timestamp=_ts())
    deltas = ["Hel", "lo, ", "world"]
    cumulative_text = ""
    for d in deltas:
        cumulative_text += d
        partial = base.model_copy(update={"content": [TextBlock(text=cumulative_text)]})
        await bus.publish(MessageDelta(message_id="m1", delta=d, partial=partial))
    await bus.close()

    seen_partials: list[str] = []
    async for ev in iterator:
        if isinstance(ev, MessageDelta):
            seen_partials.append(ev.partial.text)

    assert seen_partials == ["Hel", "Hello, ", "Hello, world"]


# ---------------------------------------------------------------------------
# Back-pressure / overflow
# ---------------------------------------------------------------------------


async def test_overflow_drops_oldest_by_default() -> None:
    """EV3: when a subscriber's queue overflows, the oldest event is
    dropped and ``dropped_total`` increments."""

    bus = InMemoryEventBus(maxsize=2)
    iterator = bus.subscribe()
    # Publish 3 events into a queue that holds 2.
    await bus.publish(NodeEnter(node="A", turn=1))
    await bus.publish(NodeEnter(node="B", turn=2))
    await bus.publish(NodeEnter(node="C", turn=3))
    assert bus.dropped_total >= 1
    await bus.close()

    seen: list[Event] = [ev async for ev in iterator]
    # The oldest ("A") was dropped; the queue retains the two most recent
    # plus the close sentinel terminates iteration.
    turns = [e.turn for e in seen if isinstance(e, NodeEnter)]
    assert turns == [2, 3]


async def test_strict_mode_raises_on_overflow() -> None:
    """Strict mode propagates the QueueFull rather than dropping."""

    bus = InMemoryEventBus(maxsize=1, strict=True)
    bus.subscribe()
    await bus.publish(NodeEnter(node="A", turn=1))
    with pytest.raises(asyncio.QueueFull):
        await bus.publish(NodeEnter(node="B", turn=2))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_default_queue_maxsize_is_documented() -> None:
    assert DEFAULT_QUEUE_MAXSIZE == 1000
    assert InMemoryEventBus().maxsize == DEFAULT_QUEUE_MAXSIZE


def test_invalid_maxsize_rejected() -> None:
    with pytest.raises(ConfigError):
        InMemoryEventBus(maxsize=0)


async def test_publish_on_closed_bus_raises() -> None:
    bus = InMemoryEventBus()
    await bus.close()
    with pytest.raises(BusClosedError):
        await bus.publish(AgentStart(agent_name="x"))


async def test_close_is_idempotent() -> None:
    bus = InMemoryEventBus()
    iterator = bus.subscribe()
    await bus.close()
    await bus.close()  # second call must not raise
    # iterator terminates cleanly
    seen: list[Event] = [ev async for ev in iterator]
    assert seen == []


def test_inmemory_bus_is_an_event_bus() -> None:
    """InMemoryEventBus satisfies the runtime-checkable EventBus Protocol."""

    assert isinstance(InMemoryEventBus(), EventBus)


# ---------------------------------------------------------------------------
# Sample event spot-checks
# ---------------------------------------------------------------------------


def test_run_start_and_run_end_carry_payloads() -> None:
    start = RunStart(run_id="r1", agent_name="demo", prompt="hi")
    assert start.prompt == "hi"
    end = RunEnd(run_id="r1", result=None, usage=Usage(), duration_ms=42)
    assert end.duration_ms == 42


def test_error_event_optional_cause() -> None:
    err = Error(message="boom")
    assert err.cause is None
    assert err.recoverable is False
    err2 = Error(message="recoverable", cause=ValueError, recoverable=True)
    assert err2.cause is ValueError


def test_message_end_carries_final_message() -> None:
    msg = Message(role="assistant", content=[TextBlock(text="done")], timestamp=_ts())
    end = MessageEnd(message_id="m1", final=msg, usage=Usage(input_tokens=1))
    assert end.final.text == "done"
    assert end.usage.input_tokens == 1


def test_tool_exec_end_default_error_is_none() -> None:
    from agent_harness.core.tools import ToolResult

    ev = ToolExecEnd(tool_call_id="c1", result=ToolResult(content=[]))
    assert ev.error is None
