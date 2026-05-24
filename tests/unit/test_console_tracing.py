"""Unit tests for ``agent_harness.tracing.console``."""

from __future__ import annotations

import asyncio
import io
from datetime import UTC, datetime

import pytest

from agent_harness.core.events import (
    AgentStart,
    Error,
    InMemoryEventBus,
    MessageDelta,
    ModelRetryRequest,
    RunStart,
    ToolExecStart,
)
from agent_harness.core.models import Message, TextBlock
from agent_harness.tracing.console import ConsoleSubscriber


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


async def _drain(sub_task: asyncio.Task[None], bus: InMemoryEventBus) -> None:
    """Close the bus and await the subscriber task with a sane timeout."""
    await bus.close()
    await asyncio.wait_for(sub_task, timeout=1.0)


async def test_prints_run_start_line() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=False, timestamps=False)
    task = sub.start()
    await bus.publish(RunStart(run_id="r1", agent_name="demo", prompt="hi there"))
    await _drain(task, bus)
    out = buf.getvalue()
    assert "RunStart" in out
    assert "run_id='r1'" in out
    assert "agent_name='demo'" in out
    # Truncation/quote handling for the user prompt.
    assert "'hi there'" in out
    # No ANSI escape codes when color=False.
    assert "\x1b[" not in out


async def test_each_event_emits_one_line() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=False, timestamps=False)
    task = sub.start()
    await bus.publish(RunStart(run_id="r1", agent_name="demo", prompt="hi"))
    await bus.publish(AgentStart(agent_name="demo"))
    await bus.publish(ToolExecStart(tool_call_id="c1", tool_name="search", arguments={"q": "x"}))
    await _drain(task, bus)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3
    assert lines[0].startswith("RunStart(")
    assert lines[1].startswith("AgentStart(")
    assert lines[2].startswith("ToolExecStart(")
    # Arguments dict renders as the key list, not the full dict.
    assert "'q'" in lines[2]


async def test_message_delta_omits_partial_snapshot() -> None:
    """`partial` is intentionally suppressed to keep the dev printer terse."""
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=False, timestamps=False)
    task = sub.start()
    msg = Message(role="assistant", content=[TextBlock(text="hi")], timestamp=_ts())
    await bus.publish(MessageDelta(message_id="m1", delta="hi", partial=msg))
    await _drain(task, bus)
    out = buf.getvalue()
    assert "MessageDelta" in out
    assert "delta='hi'" in out
    assert "partial=" not in out


async def test_long_strings_get_truncated() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=False, timestamps=False)
    task = sub.start()
    long_prompt = "x" * 500
    await bus.publish(RunStart(run_id="r1", agent_name="demo", prompt=long_prompt))
    await _drain(task, bus)
    out = buf.getvalue()
    # The horizontal ellipsis character is what _truncate inserts.
    assert "…" in out
    # And of course the full 500-char string did NOT make it through.
    assert "x" * 500 not in out


async def test_color_emits_ansi_when_enabled() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=True, timestamps=False)
    task = sub.start()
    await bus.publish(RunStart(run_id="r1", agent_name="demo", prompt="hi"))
    await _drain(task, bus)
    out = buf.getvalue()
    # ANSI escape introducer present, and the reset sequence terminates.
    assert "\x1b[" in out
    assert "\x1b[0m" in out


async def test_color_tier_picks_warn_for_model_retry() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=True, timestamps=False)
    task = sub.start()
    await bus.publish(ModelRetryRequest(reason="rate-limited"))
    await _drain(task, bus)
    out = buf.getvalue()
    # \x1b[33m is the yellow / warn color for retries.
    assert "\x1b[33m" in out


async def test_color_tier_picks_red_for_error() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=True, timestamps=False)
    task = sub.start()
    await bus.publish(Error(message="boom", recoverable=False))
    await _drain(task, bus)
    out = buf.getvalue()
    assert "\x1b[31m" in out
    assert "Error" in out


async def test_timestamps_prefix_when_enabled() -> None:
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=False, timestamps=True)
    task = sub.start()
    await bus.publish(AgentStart(agent_name="demo"))
    await _drain(task, bus)
    line = buf.getvalue().strip()
    # First token looks like HH:MM:SS.mmm — at least 12 chars with two colons
    # and a dot.
    head, _, _rest = line.partition(" ")
    assert head.count(":") == 2
    assert "." in head


async def test_slow_subscriber_does_not_block_publish() -> None:
    """Subscriber drives itself; publish never awaits on stdio."""
    bus = InMemoryEventBus(maxsize=4)
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=False, timestamps=False)
    task = sub.start()
    # Publish more than `maxsize` events without yielding to the consumer.
    # If publish blocked, this loop would deadlock.
    for i in range(20):
        await bus.publish(AgentStart(agent_name=f"a{i}"))
    await _drain(task, bus)
    # Last published event must have made it through (drop-oldest semantics
    # mean older ones may have been dropped, but the *current* event is what
    # the user wants to see).
    out = buf.getvalue()
    assert "a19" in out


async def test_start_is_idempotent() -> None:
    bus = InMemoryEventBus()
    sub = ConsoleSubscriber(bus, stream=io.StringIO(), color=False, timestamps=False)
    t1 = sub.start()
    t2 = sub.start()
    assert t1 is t2
    await bus.close()
    await asyncio.wait_for(t1, timeout=1.0)


async def test_default_color_disabled_for_stringio() -> None:
    """StringIO has no ``isatty`` so color defaults off."""
    sub = ConsoleSubscriber(InMemoryEventBus(), stream=io.StringIO())
    assert sub.color is False


async def test_render_handles_missing_tier_gracefully() -> None:
    """Even if a future event class slips outside the tier tuples, color path
    must not crash — we just emit the bold name without a tier color."""
    bus = InMemoryEventBus()
    buf = io.StringIO()
    sub = ConsoleSubscriber(bus, stream=buf, color=True, timestamps=False)

    # Synthesise a frozen dataclass that isn't in the union; the renderer
    # works structurally so this exercises the "no color match" branch.
    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class _MysteryEvent:
        note: str

    rendered = sub._render(_MysteryEvent(note="hello"))  # type: ignore[arg-type]
    assert "MysteryEvent" in rendered

    await bus.close()
    # No task was started for this test; just ensure no leak.
    assert sub._task is None


def test_module_smoke_via_doctest_example() -> None:
    """The module docstring's runnable example actually returns True."""
    import doctest

    import agent_harness.tracing.console as mod

    result = doctest.testmod(mod, verbose=False)
    assert result.failed == 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    pytest.main([__file__, "-v"])
