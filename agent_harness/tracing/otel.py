"""OpenTelemetry tracing subscriber — **v1 SKELETON**.

Per open-questions decision #6 ("Built-in observability — OpenTelemetry
only or also a built-in tracer?"), v1 ships OTEL as an *unimplemented
skeleton*: the subscriber interface and stub bodies are in place to
reserve the namespace and document the contract, but no real
``opentelemetry`` spans are emitted. The console pretty-printer at
:mod:`agent_harness.tracing.console` is fully implemented and is what
dev users should rely on until a real OTEL integration lands.

Why a skeleton rather than nothing?

* The shape of the subscriber is what consumers wire into their app
  startup; it should not move between v0.0.1 and v0.0.2.
* OTEL deserves a real integration (resource attribution, context
  propagation, semantic conventions), not a quick first pass. The
  skeleton makes the *gap* explicit so reviewers don't mistake a stub
  for the real thing.

# TODO(otel-v0.0.2): replace each stub with the real OTEL emit, gated on
# an ``opentelemetry`` import. The dependency belongs in the ``[otel]``
# extras section of ``pyproject.toml`` (already declared) so core
# remains import-light.

Example:
    >>> from agent_harness.core.events import InMemoryEventBus
    >>> bus = InMemoryEventBus()
    >>> sub = OTELSubscriber(bus, tracer=None)
    >>> sub.bus is bus
    True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_harness.core.events import (
    AgentEnd,
    AgentStart,
    Error,
    Event,
    EventBus,
    MessageEnd,
    MessageStart,
    NodeEnter,
    NodeExit,
    RunEnd,
    RunStart,
    ToolExecEnd,
    ToolExecStart,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    pass


class OTELSubscriber:
    """Subscribe to an :class:`EventBus` and emit OpenTelemetry spans.

    **v1 status: SKELETON.** :meth:`run` raises ``NotImplementedError``;
    the per-event handler methods document *what* would be emitted in
    v0.0.2 but do not call into ``opentelemetry``. Until this is filled
    in, use :class:`agent_harness.tracing.console.ConsoleSubscriber` for
    local diagnostics or wire your own subscriber for production
    tracing.

    The eventual implementation will hold a ``dict[str, Span]`` keyed by
    a stable correlation id (``agent_name``, ``message_id``,
    ``tool_call_id``) and start / end spans on the matching pairs.
    Context propagation across nested-loop subagents will piggyback on
    the same dict — the parent's ``AgentStart`` span becomes the parent
    span for the child's ``AgentStart``.

    Example:
        >>> from agent_harness.core.events import InMemoryEventBus
        >>> sub = OTELSubscriber(InMemoryEventBus(), tracer=None)
        >>> isinstance(sub.spans, dict)
        True
    """

    def __init__(self, bus: EventBus, tracer: Any) -> None:
        # `tracer` is typed `Any` deliberately — we do not import
        # ``opentelemetry`` from core/tracing in v1. Wave-2 OTEL integration
        # will refine the type to ``opentelemetry.trace.Tracer``.
        self.bus = bus
        self.tracer = tracer
        self.spans: dict[str, Any] = {}

    async def run(self) -> None:
        """Consume events from the bus and translate to OTEL spans.

        Not implemented in v1; see decision #6. The eventual shape::

            async for event in self.bus.subscribe():
                self._dispatch(event)
        """
        raise NotImplementedError(
            "tracing/otel.py is a v1 skeleton; see "
            "agent-harness-research/proposal/open-questions.md #6"
        )

    # --- per-event handlers (stubs) ----------------------------------------
    #
    # Each method documents the span / attribute the real implementation
    # would emit. Returning ``None`` is deliberate: callers can wire these
    # up today as no-op subscribers without crashing, and the contract
    # remains visible at code-search time.

    def _dispatch(self, event: Event) -> None:  # pragma: no cover - stub
        """Translate one event into a span action.

        # TODO(otel-v0.0.2): implement a structural dispatch matching the
        # method stubs below. The dispatch should be exhaustive on the
        # closed-set ``Event`` union to keep parity with EV4.
        """
        if isinstance(event, RunStart):
            self.on_run_start(event)
        elif isinstance(event, RunEnd):
            self.on_run_end(event)
        elif isinstance(event, AgentStart):
            self.on_agent_start(event)
        elif isinstance(event, AgentEnd):
            self.on_agent_end(event)
        elif isinstance(event, NodeEnter):
            self.on_node_enter(event)
        elif isinstance(event, NodeExit):
            self.on_node_exit(event)
        elif isinstance(event, MessageStart):
            self.on_message_start(event)
        elif isinstance(event, MessageEnd):
            self.on_message_end(event)
        elif isinstance(event, ToolExecStart):
            self.on_tool_exec_start(event)
        elif isinstance(event, ToolExecEnd):
            self.on_tool_exec_end(event)
        elif isinstance(event, Error):
            self.on_error(event)

    def on_run_start(self, event: RunStart) -> None:
        """Would start the root span named ``"run:{event.agent_name}"`` with
        attributes ``agent_harness.run_id`` and ``agent_harness.prompt.len``.
        """

    def on_run_end(self, event: RunEnd) -> None:
        """Would end the root span and set ``agent_harness.usage.*`` plus
        ``agent_harness.duration_ms`` attributes."""

    def on_agent_start(self, event: AgentStart) -> None:
        """Would start ``"agent:{event.agent_name}"`` as a child of the root
        (or of the parent agent's span when nested)."""

    def on_agent_end(self, event: AgentEnd) -> None:
        """Would end the matching agent span keyed by ``agent_name``."""

    def on_node_enter(self, event: NodeEnter) -> None:
        """Would start ``"node:{event.node}"`` with ``turn`` attribute."""

    def on_node_exit(self, event: NodeExit) -> None:
        """Would end the matching node span and set ``next`` /
        ``interrupted`` attributes."""

    def on_message_start(self, event: MessageStart) -> None:
        """Would start ``"message:{event.message_id}"`` under the current
        node span."""

    def on_message_end(self, event: MessageEnd) -> None:
        """Would end the message span and record ``usage.*`` and
        ``role=assistant`` attributes."""

    def on_tool_exec_start(self, event: ToolExecStart) -> None:
        """Would start ``"tool:{event.tool_name}"`` keyed by
        ``tool_call_id``."""

    def on_tool_exec_end(self, event: ToolExecEnd) -> None:
        """Would end the tool span and record ``error`` (if any) plus a
        truncated content summary."""

    def on_error(self, event: Error) -> None:
        """Would record ``event.message`` on the current span and set the
        span status to ``ERROR`` when ``recoverable=False``."""
