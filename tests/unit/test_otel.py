"""Unit tests for ``agent_harness.tracing.otel`` — the real OTEL subscriber.

Spans are captured with an in-memory OTEL exporter so we can assert the
emitted tree (names, GenAI semantic-convention attributes, parentage, usage,
error status) without a backend. Per decision #6, the subscriber is purely an
EventBus consumer (EV8): these tests pin that contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent_harness.core.events import (
    InMemoryEventBus,
    MessageEnd,
    MessageStart,
    ModelStart,
    RunEnd,
    RunStart,
    ToolExecEnd,
    ToolExecStart,
)
from agent_harness.core.models import Message, TextBlock, Usage
from agent_harness.core.tools import ToolResult
from agent_harness.tracing import otel as otel_mod
from agent_harness.tracing.otel import OTELSubscriber


@pytest.fixture
def tracer_exporter():
    """A fresh in-memory exporter + tracer per test."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    yield provider.get_tracer("test"), exp
    provider.shutdown()


def _assistant(text: str) -> Message:
    return Message(
        role="assistant",
        content=[TextBlock(text=text)],
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


class _RunResult:
    def __init__(self, output):
        self.output = output


def _by_name(exp):
    spans = exp.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    tree = {}
    for s in spans:
        parent = by_id.get(s.parent.span_id) if s.parent else None
        tree[s.name] = (s, parent.name if parent else None)
    return spans, tree


def test_module_imports_without_opentelemetry_at_top() -> None:
    """The module imports without requiring opentelemetry (lazy in __init__)."""
    assert otel_mod.__doc__ is not None
    assert hasattr(otel_mod, "OTELSubscriber")


async def test_builds_gen_ai_span_tree(tracer_exporter) -> None:
    tracer, exp = tracer_exporter
    bus = InMemoryEventBus()
    sub = OTELSubscriber(
        bus,
        tracer,
        root_name="penny-agent-run",
        root_attributes={"session.id": "conv-42", "langfuse.trace.tags": ["chat"]},
    )
    task = sub.start()

    await bus.publish(RunStart(run_id="r1", agent_name="penny", prompt="how much?"))
    await bus.publish(ModelStart(model_name="gemini-3.5-flash"))
    await bus.publish(MessageStart(message_id="m1"))
    await bus.publish(
        ToolExecStart(tool_call_id="c1", tool_name="run_sql", arguments={"sql": "select 1"})
    )
    await bus.publish(
        ToolExecEnd(
            tool_call_id="c1",
            result=ToolResult(content=[], structured_content={"rows": [[1]]}),
        )
    )
    await bus.publish(
        MessageEnd(
            message_id="m1",
            final=_assistant("You spent $42."),
            usage=Usage(input_tokens=100, output_tokens=12),
        )
    )
    await bus.publish(
        RunEnd(
            run_id="r1",
            result=_RunResult("You spent $42."),
            usage=Usage(input_tokens=100, output_tokens=12),
            duration_ms=1234,
        )
    )
    await bus.close()
    await task

    _, tree = _by_name(exp)
    assert "penny-agent-run" in tree
    assert "chat gemini-3.5-flash" in tree
    assert "execute_tool run_sql" in tree

    root, _ = tree["penny-agent-run"]
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["session.id"] == "conv-42"
    assert tuple(root.attributes["langfuse.trace.tags"]) == ("chat",)

    gen, gen_parent = tree["chat gemini-3.5-flash"]
    assert gen_parent == "penny-agent-run"
    assert gen.attributes["gen_ai.operation.name"] == "chat"
    assert gen.attributes["gen_ai.request.model"] == "gemini-3.5-flash"
    assert gen.attributes["gen_ai.usage.input_tokens"] == 100
    assert gen.attributes["gen_ai.usage.output_tokens"] == 12

    tool, tool_parent = tree["execute_tool run_sql"]
    assert tool_parent == "penny-agent-run"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "run_sql"


async def test_tool_error_sets_error_status(tracer_exporter) -> None:
    tracer, exp = tracer_exporter
    bus = InMemoryEventBus()
    sub = OTELSubscriber(bus, tracer)
    task = sub.start()
    await bus.publish(RunStart(run_id="r", agent_name="penny", prompt="x"))
    await bus.publish(ToolExecStart(tool_call_id="c1", tool_name="boom", arguments={}))
    await bus.publish(ToolExecEnd(tool_call_id="c1", result=ToolResult(content=[]), error="kaboom"))
    await bus.publish(RunEnd(run_id="r", result=_RunResult(None), usage=Usage(), duration_ms=1))
    await bus.close()
    await task

    _, tree = _by_name(exp)
    tool, _ = tree["execute_tool boom"]
    assert tool.status.status_code == StatusCode.ERROR


async def test_finalize_ends_dangling_spans_on_abort(tracer_exporter) -> None:
    """If the bus closes mid-run, open spans are still ended (and exported)."""
    tracer, exp = tracer_exporter
    bus = InMemoryEventBus()
    sub = OTELSubscriber(bus, tracer)
    task = sub.start()
    await bus.publish(RunStart(run_id="r", agent_name="penny", prompt="x"))
    await bus.publish(ModelStart(model_name="m"))
    await bus.publish(MessageStart(message_id="m1"))  # never ends
    await bus.close()
    await task

    names = {s.name for s in exp.get_finished_spans()}
    assert "invoke_agent penny" in names
    assert "chat m" in names  # dangling generation span ended in _finalize
