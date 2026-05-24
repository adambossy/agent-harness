"""Unit tests for ``agent_harness.tracing.otel`` — skeleton surface only.

Per open-questions decision #6, OTEL ships as an *unimplemented skeleton*
in v1. These tests pin the public surface so the eventual v0.0.2
implementation can't silently break the contract.
"""

from __future__ import annotations

import inspect

import pytest

from agent_harness.core.events import (
    AgentEnd,
    AgentStart,
    Error,
    InMemoryEventBus,
    MessageEnd,
    MessageStart,
    NodeEnter,
    NodeExit,
    RunEnd,
    RunStart,
    ToolExecEnd,
    ToolExecStart,
)
from agent_harness.core.models import Message, TextBlock, Usage
from agent_harness.tracing import otel as otel_mod
from agent_harness.tracing.otel import OTELSubscriber


def test_module_imports() -> None:
    """The module imports cleanly without an ``opentelemetry`` dependency."""
    assert otel_mod.__doc__ is not None
    assert "SKELETON" in otel_mod.__doc__
    assert "open-questions" in otel_mod.__doc__.lower()


def test_subscriber_class_exposed() -> None:
    assert hasattr(otel_mod, "OTELSubscriber")
    assert inspect.isclass(OTELSubscriber)


def test_constructor_takes_bus_and_tracer() -> None:
    bus = InMemoryEventBus()
    sub = OTELSubscriber(bus, tracer=None)
    assert sub.bus is bus
    assert sub.tracer is None
    assert isinstance(sub.spans, dict)
    assert sub.spans == {}


async def test_run_raises_not_implemented() -> None:
    sub = OTELSubscriber(InMemoryEventBus(), tracer=None)
    with pytest.raises(NotImplementedError, match="skeleton"):
        await sub.run()


async def test_run_error_message_points_to_open_questions() -> None:
    sub = OTELSubscriber(InMemoryEventBus(), tracer=None)
    try:
        await sub.run()
    except NotImplementedError as exc:
        assert "open-questions.md" in str(exc).lower()
    else:  # pragma: no cover - run() must raise
        pytest.fail("OTELSubscriber.run() did not raise NotImplementedError")


def test_per_event_handler_methods_exist() -> None:
    """Every documented handler method is callable and returns None."""
    expected = {
        "on_run_start",
        "on_run_end",
        "on_agent_start",
        "on_agent_end",
        "on_node_enter",
        "on_node_exit",
        "on_message_start",
        "on_message_end",
        "on_tool_exec_start",
        "on_tool_exec_end",
        "on_error",
    }
    for name in expected:
        method = getattr(OTELSubscriber, name, None)
        assert method is not None, f"missing handler: {name}"
        assert callable(method)


def test_handler_stubs_are_safe_to_call() -> None:
    """Stubs must be safe to call — they document intent without emitting.

    Each handler is typed as ``-> None``; we exercise the body to keep
    coverage honest, and rely on the type annotation as the contract.
    """
    sub = OTELSubscriber(InMemoryEventBus(), tracer=None)
    usage = Usage()
    from datetime import UTC, datetime

    ts = datetime(2026, 1, 1, tzinfo=UTC)

    sub.on_run_start(RunStart(run_id="r1", agent_name="a", prompt="hi"))
    sub.on_run_end(RunEnd(run_id="r1", result=None, usage=usage, duration_ms=10))
    sub.on_agent_start(AgentStart(agent_name="a"))
    sub.on_agent_end(AgentEnd(agent_name="a"))
    sub.on_node_enter(NodeEnter(node="n", turn=0))
    sub.on_node_exit(NodeExit(node="n"))
    sub.on_message_start(MessageStart(message_id="m1"))
    final_msg = Message(role="assistant", content=[TextBlock(text="hi")], timestamp=ts)
    sub.on_message_end(MessageEnd(message_id="m1", final=final_msg, usage=usage))
    sub.on_tool_exec_start(ToolExecStart(tool_call_id="c1", tool_name="t", arguments={}))
    # ToolExecEnd needs a ToolResult; Layer 0 ToolResult is a dataclass.
    from agent_harness.core.tools import ToolResult

    sub.on_tool_exec_end(ToolExecEnd(tool_call_id="c1", result=ToolResult(content=[])))
    sub.on_error(Error(message="boom"))


def test_dispatch_routes_known_events() -> None:
    """``_dispatch`` exists and routes a sampling of events without raising."""
    sub = OTELSubscriber(InMemoryEventBus(), tracer=None)
    # `_dispatch` is the documented v0.0.2 entry point; calling it on the
    # current stub must be a no-op for every closed-set member.
    assert callable(sub._dispatch)
    sub._dispatch(RunStart(run_id="r1", agent_name="a", prompt="hi"))
    sub._dispatch(AgentStart(agent_name="a"))
    sub._dispatch(NodeEnter(node="n", turn=0))
    sub._dispatch(Error(message="boom"))


def test_v002_todo_marker_present() -> None:
    """A grep-friendly ``TODO(otel-v0.0.2)`` anchors the deferred work."""
    from pathlib import Path

    src = Path(otel_mod.__file__).read_text(encoding="utf-8")
    assert "TODO(otel-v0.0.2)" in src


if __name__ == "__main__":  # pragma: no cover - manual invocation
    pytest.main([__file__, "-v"])
