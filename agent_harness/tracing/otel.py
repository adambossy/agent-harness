"""OpenTelemetry tracing subscriber.

Per open-questions decision #6, observability participates as an *EventBus
subscriber*, never as a hook (EV8: observing is never an intervention). This
module is the real OTEL subscriber: it consumes the typed :class:`Event`
stream and emits OpenTelemetry spans following the GenAI semantic
conventions (``gen_ai.*``), so the spans are meaningful in any OTEL backend
(Langfuse, Jaeger, Honeycomb, …) — the harness stays vendor-neutral and the
embedding app chooses the exporter.

Span tree per run:

* ``RunStart`` opens a root span (``gen_ai.operation.name = invoke_agent``).
* each assistant message opens a child ``generation`` span
  (``gen_ai.operation.name = chat``) carrying the model name and, on
  ``MessageEnd``, token usage and the completion text.
* each tool execution opens a child span
  (``gen_ai.operation.name = execute_tool``) carrying the arguments and,
  on ``ToolExecEnd``, the result (or an ERROR status).

``opentelemetry`` is imported lazily (inside the constructor), so importing
this module — and therefore :mod:`agent_harness.tracing` — never requires the
``[otel]`` extra. Instantiating :class:`OTELSubscriber` does; a missing
dependency raises a clear, actionable error.

Wiring (the app owns the exporter and the bus lifecycle)::

    from agent_harness.tracing import OTELSubscriber

    sub = OTELSubscriber(
        bus, root_name="my-agent-run", root_attributes={"session.id": conversation_id}
    )
    task = sub.start()  # spawn the consumer task
    await agent.run(prompt, event_bus=bus)
    await bus.close()
    await task  # drains + ends any open spans

Example:
    >>> from agent_harness.core.events import InMemoryEventBus
    >>> bus = InMemoryEventBus()
    >>> sub = OTELSubscriber(bus)  # doctest: +SKIP
    >>> isinstance(sub.spans, dict)  # doctest: +SKIP
    True
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from agent_harness.core.events import (
    Error,
    Event,
    EventBus,
    MessageEnd,
    MessageStart,
    ModelStart,
    RunEnd,
    RunStart,
    ToolExecEnd,
    ToolExecStart,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    import asyncio


# GenAI semantic-convention attribute keys (vendor-neutral).
_OP = "gen_ai.operation.name"
_AGENT_NAME = "gen_ai.agent.name"
_REQUEST_MODEL = "gen_ai.request.model"
_USAGE_IN = "gen_ai.usage.input_tokens"
_USAGE_OUT = "gen_ai.usage.output_tokens"
_USAGE_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
_USAGE_CACHE_WRITE = "gen_ai.usage.cache_creation.input_tokens"
_COMPLETION = "gen_ai.completion"
_TOOL_NAME = "gen_ai.tool.name"
_TOOL_ARGS = "gen_ai.tool.call.arguments"
_TOOL_RESULT = "gen_ai.tool.call.result"


def _json(value: Any) -> str:
    """Compact, never-raising JSON for span attributes."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


class OTELSubscriber:
    """Subscribe to an :class:`EventBus` and emit OpenTelemetry spans.

    Lifecycle mirrors :class:`~agent_harness.tracing.console.ConsoleSubscriber`:
    the subscription is acquired eagerly in ``__init__`` (so no event
    published between construction and the consumer task's first scheduling is
    missed), and :meth:`start` spawns the consumer task. Closing the bus
    drains the iterator and ends any still-open spans.

    Spans are nested by explicit context (the root span's context is the
    parent of every child), so correct parentage does not depend on the OTEL
    *current* context surviving across ``await`` points in the consumer loop.

    Args:
        bus: the event bus to subscribe to.
        tracer: an OTEL ``Tracer``; defaults to
            ``opentelemetry.trace.get_tracer("agent_harness")`` (the app's
            globally-configured provider).
        root_name: name for the root span. Defaults to
            ``"invoke_agent {agent_name}"`` from the ``RunStart`` event.
        root_attributes: extra attributes to set on the root span — the seam
            the app uses to attach backend-specific trace metadata (e.g.
            ``{"session.id": ..., "user.id": ..., "langfuse.trace.tags": [...]}``)
            without the harness knowing about any particular backend.
    """

    def __init__(
        self,
        bus: EventBus,
        tracer: Any = None,
        *,
        root_name: str | None = None,
        root_attributes: dict[str, Any] | None = None,
    ) -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - exercised via extra
            raise RuntimeError(
                "OTELSubscriber requires the 'opentelemetry' packages. "
                "Install the harness with the [otel] extra: "
                "pip install 'agent-harness[otel]'."
            ) from exc

        self._trace = trace
        self.bus = bus
        self.tracer = tracer if tracer is not None else trace.get_tracer("agent_harness")
        self.root_name = root_name
        self.root_attributes = root_attributes or {}

        # spans keyed by correlation id (message_id / tool_call_id); the root
        # is tracked separately. `spans` is part of the documented surface.
        self.spans: dict[str, Any] = {}
        self._root: Any = None
        self._root_ctx: Any = None
        self._model_name: str | None = None

        # Eager subscribe (synchronous) so early events aren't dropped.
        self._events = bus.subscribe()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        """Spawn the consumer task. Safe to call once per instance."""
        import asyncio

        if self._task is not None:
            return self._task
        self._task = asyncio.create_task(self._run(), name="otel-subscriber")
        return self._task

    async def run(self) -> None:
        """Consume events until the bus closes (alternative to :meth:`start`)."""
        await self._run()

    async def _run(self) -> None:
        try:
            async for event in self._events:
                # EV8: observation is never an intervention — a tracing bug
                # must never break the run.
                with contextlib.suppress(Exception):
                    self._dispatch(event)
        finally:
            self._finalize()

    def _dispatch(self, event: Event) -> None:
        if isinstance(event, RunStart):
            self.on_run_start(event)
        elif isinstance(event, ModelStart):
            self._model_name = event.model_name
        elif isinstance(event, MessageStart):
            self.on_message_start(event)
        elif isinstance(event, MessageEnd):
            self.on_message_end(event)
        elif isinstance(event, ToolExecStart):
            self.on_tool_exec_start(event)
        elif isinstance(event, ToolExecEnd):
            self.on_tool_exec_end(event)
        elif isinstance(event, RunEnd):
            self.on_run_end(event)
        elif isinstance(event, Error):
            self.on_error(event)

    # --- handlers -----------------------------------------------------------

    def on_run_start(self, event: RunStart) -> None:
        name = self.root_name or f"invoke_agent {event.agent_name}"
        attributes: dict[str, Any] = {
            _OP: "invoke_agent",
            _AGENT_NAME: event.agent_name,
            **self.root_attributes,
        }
        self._root = self.tracer.start_span(name, attributes=attributes)
        self._root_ctx = self._trace.set_span_in_context(self._root)

    def on_message_start(self, event: MessageStart) -> None:
        if self._root is None:
            return
        model = self._model_name or "model"
        span = self.tracer.start_span(
            f"chat {model}",
            context=self._root_ctx,
            attributes={_OP: "chat", _REQUEST_MODEL: model},
        )
        self.spans[event.message_id] = span

    def on_message_end(self, event: MessageEnd) -> None:
        span = self.spans.pop(event.message_id, None)
        if span is None:
            return
        usage = event.usage
        span.set_attribute(_USAGE_IN, usage.input_tokens)
        span.set_attribute(_USAGE_OUT, usage.output_tokens)
        if usage.cache_read_tokens:
            span.set_attribute(_USAGE_CACHE_READ, usage.cache_read_tokens)
        if usage.cache_write_tokens:
            span.set_attribute(_USAGE_CACHE_WRITE, usage.cache_write_tokens)
        with contextlib.suppress(Exception):
            span.set_attribute(_COMPLETION, event.final.text)
        span.end()

    def on_tool_exec_start(self, event: ToolExecStart) -> None:
        if self._root is None:
            return
        span = self.tracer.start_span(
            f"execute_tool {event.tool_name}",
            context=self._root_ctx,
            attributes={
                _OP: "execute_tool",
                _TOOL_NAME: event.tool_name,
                _TOOL_ARGS: _json(event.arguments),
            },
        )
        self.spans[event.tool_call_id] = span

    def on_tool_exec_end(self, event: ToolExecEnd) -> None:
        span = self.spans.pop(event.tool_call_id, None)
        if span is None:
            return
        result = event.result
        err = event.error or getattr(result, "error", None)
        structured = getattr(result, "structured_content", None)
        output = structured if structured is not None else getattr(result, "content", result)
        span.set_attribute(_TOOL_RESULT, _json(output))
        if err:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(err)))
        span.end()

    def on_run_end(self, event: RunEnd) -> None:
        if self._root is None:
            return
        usage = event.usage
        self._root.set_attribute(_USAGE_IN, usage.input_tokens)
        self._root.set_attribute(_USAGE_OUT, usage.output_tokens)
        self._root.set_attribute("agent_harness.duration_ms", event.duration_ms)
        with contextlib.suppress(Exception):
            output = getattr(event.result, "output", None)
            if output is not None:
                self._root.set_attribute(_COMPLETION, _json(output))
        self._root.end()
        self._root = None

    def on_error(self, event: Error) -> None:
        if self._root is None:
            return
        from opentelemetry.trace import Status, StatusCode

        self._root.set_status(Status(StatusCode.ERROR, event.message))

    def _finalize(self) -> None:
        """End any spans left open by an aborted/errored run."""
        for span in self.spans.values():
            with contextlib.suppress(Exception):
                span.end()
        self.spans.clear()
        if self._root is not None:
            with contextlib.suppress(Exception):
                self._root.end()
            self._root = None
