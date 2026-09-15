"""Unit tests for ``agent_harness.tracing.otel`` — the real OTEL subscriber.

Spans are captured with an in-memory OTEL exporter so we can assert the
emitted tree (names, GenAI semantic-convention attributes, parentage, usage,
error status) without a backend. Per decision #6, the subscriber is purely an
EventBus consumer (EV8): these tests pin that contract.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import UTC, datetime
from typing import cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode, Tracer

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
from agent_harness.core.models import Message, TextBlock, ThinkingBlock, ToolCallBlock, Usage
from agent_harness.core.tools import ToolResult
from agent_harness.tracing import otel as otel_mod
from agent_harness.tracing.otel import OTELSubscriber

TracerExporter = tuple[Tracer, InMemorySpanExporter]


@pytest.fixture
def tracer_exporter() -> Generator[TracerExporter]:
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


def _user(text: str) -> Message:
    return Message(
        role="user",
        content=[TextBlock(text=text)],
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


class _RunResult:
    def __init__(self, output: object) -> None:
        self.output = output


def _by_name(
    exp: InMemorySpanExporter,
) -> tuple[tuple[ReadableSpan, ...], dict[str, tuple[ReadableSpan, str | None]]]:
    spans = exp.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    tree = {}
    for s in spans:
        parent = by_id.get(s.parent.span_id) if s.parent else None
        tree[s.name] = (s, parent.name if parent else None)
    return spans, tree


def _attributes(span: ReadableSpan) -> dict[str, object]:
    """Materialize optional OTEL attributes into a test-friendly mapping."""
    return dict(span.attributes or {})


def test_module_imports_without_opentelemetry_at_top() -> None:
    """The module imports without requiring opentelemetry (lazy in __init__)."""
    assert otel_mod.__doc__ is not None
    assert hasattr(otel_mod, "OTELSubscriber")


async def test_builds_gen_ai_span_tree(tracer_exporter: TracerExporter) -> None:
    tracer, exp = tracer_exporter
    bus = InMemoryEventBus()
    sub = OTELSubscriber(
        bus,
        tracer,
        root_name="penny-agent-run",
        root_attributes={
            "session.id": "conv-42",
            "langfuse.trace.tags": ["chat"],
            "langfuse.observation.input": "how much?",
        },
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
    root_attributes = _attributes(root)
    assert root_attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root_attributes["session.id"] == "conv-42"
    assert tuple(cast(tuple[str, ...], root_attributes["langfuse.trace.tags"])) == ("chat",)
    assert root_attributes["gen_ai.completion"] == "You spent $42."
    assert root_attributes["output.value"] == "You spent $42."
    assert root_attributes["langfuse.observation.output"] == "You spent $42."

    gen, gen_parent = tree["chat gemini-3.5-flash"]
    assert gen_parent == "penny-agent-run"
    gen_attributes = _attributes(gen)
    assert gen_attributes["gen_ai.operation.name"] == "chat"
    assert gen_attributes["gen_ai.request.model"] == "gemini-3.5-flash"
    assert gen_attributes["gen_ai.usage.input_tokens"] == 100
    assert gen_attributes["gen_ai.usage.output_tokens"] == 12
    assert gen_attributes["gen_ai.completion"] == "You spent $42."

    tool, tool_parent = tree["execute_tool run_sql"]
    assert tool_parent == "penny-agent-run"
    tool_attributes = _attributes(tool)
    assert tool_attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_attributes["gen_ai.tool.name"] == "run_sql"


async def test_tool_error_sets_error_status(tracer_exporter: TracerExporter) -> None:
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


async def test_records_model_input_and_structured_assistant_output(
    tracer_exporter: TracerExporter,
) -> None:
    """Generation telemetry retains OpenRouter-style thinking and tool calls."""
    tracer, exp = tracer_exporter
    bus = InMemoryEventBus()
    sub = OTELSubscriber(bus, tracer)
    task = sub.start()
    await bus.publish(RunStart(run_id="r", agent_name="penny", prompt="categorize"))
    await bus.publish(ModelStart(model_name="z-ai/glm-5.3", messages=(_user("categorize ND NYC"),)))
    await bus.publish(MessageStart(message_id="m1"))
    await bus.publish(
        MessageEnd(
            message_id="m1",
            final=Message(
                role="assistant",
                content=[
                    ThinkingBlock(text="Use the merchant history.", signature="opaque-secret"),
                    ToolCallBlock(
                        id="call-1",
                        name="submit_categorization",
                        arguments={"category_key": "beauty", "reasoning": "Nail salon"},
                    ),
                ],
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            usage=Usage(input_tokens=100, output_tokens=20),
        )
    )
    await bus.publish(RunEnd(run_id="r", result=_RunResult(None), usage=Usage(), duration_ms=1))
    await bus.close()
    await task

    _, tree = _by_name(exp)
    generation, _ = tree["chat z-ai/glm-5.3"]
    generation_attributes = _attributes(generation)
    prompt = cast(str, generation_attributes["gen_ai.prompt"])
    assert json.loads(prompt) == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "categorize ND NYC"}],
        }
    ]
    completion = cast(str, generation_attributes["gen_ai.completion"])
    output = json.loads(completion)
    assert output == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "text": "Use the merchant history."},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "submit_categorization",
                "arguments": {"category_key": "beauty", "reasoning": "Nail salon"},
            },
        ],
    }
    assert "opaque-secret" not in completion

    root, _ = tree["invoke_agent penny"]
    root_attributes = _attributes(root)
    root_completion = cast(str, root_attributes["gen_ai.completion"])
    assert json.loads(root_completion) == output
    assert root_attributes["output.value"] == root_completion


async def test_finalize_ends_dangling_spans_on_abort(
    tracer_exporter: TracerExporter,
) -> None:
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
