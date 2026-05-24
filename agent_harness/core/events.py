"""Typed event taxonomy + ``EventBus`` Protocol + ``InMemoryEventBus``.

The closed-set ``Event`` union is load-bearing (EV4); ``MessageDelta`` carries
a cumulative ``partial: Message`` (EV5). Per open-questions #4 there are no
``Handoff*`` events — nested subagents republish ``AgentStart`` /
``AgentEnd`` and the parent sees ``SubagentStart`` / ``SubagentStop``.

Example:
    >>> bus = InMemoryEventBus()
    >>> bus.maxsize > 0
    True
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .errors import BusClosedError, ConfigError
from .models import Message, Usage
from .run_state import ApprovalRequest

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .tools import ToolResult


# --- Lifecycle ---------------------------------------------------------------
# Each event below is a frozen-in-time fact emitted by the loop. The module
# docstring shows the canonical usage example; per-event docs stay short.


@dataclass(frozen=True, slots=True)
class RunStart:
    """Start of a run."""

    run_id: str
    agent_name: str
    prompt: str


@dataclass(frozen=True, slots=True)
class RunEnd:
    """End of a run; ``result`` is a forward-ref to Layer-3 ``RunResult``."""

    run_id: str
    result: Any
    usage: Usage
    duration_ms: int


@dataclass(frozen=True, slots=True)
class AgentStart:
    """An agent (root or subagent) has begun executing."""

    agent_name: str


@dataclass(frozen=True, slots=True)
class AgentEnd:
    """An agent has finished executing."""

    agent_name: str


# --- Graph nodes -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeEnter:
    """Entry to a graph node."""

    node: str
    turn: int


@dataclass(frozen=True, slots=True)
class NodeExit:
    """Exit from a graph node."""

    node: str
    next: str | None = None
    interrupted: bool = False


# --- Model events ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelStart:
    """Provider request is about to be issued."""

    model_name: str


@dataclass(frozen=True, slots=True)
class MessageStart:
    """An assistant message has begun streaming."""

    message_id: str


@dataclass(frozen=True, slots=True)
class MessageDelta:
    """One incremental chunk + the cumulative partial-so-far (EV5).

    Example:
        >>> from datetime import datetime, timezone
        >>> from agent_harness.core.models import Message
        >>> MessageDelta(
        ...     message_id="m1",
        ...     delta="hi",
        ...     partial=Message(
        ...         role="assistant", content=[], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
        ...     ),
        ... ).delta
        'hi'
    """

    message_id: str
    delta: str
    partial: Message


@dataclass(frozen=True, slots=True)
class MessageEnd:
    """An assistant message has completed."""

    message_id: str
    final: Message
    usage: Usage


@dataclass(frozen=True, slots=True)
class ThinkingStart:
    """Extended-thinking has begun."""

    message_id: str


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """One thinking increment + the cumulative partial string."""

    message_id: str
    delta: str
    partial: str


@dataclass(frozen=True, slots=True)
class ThinkingEnd:
    """Extended-thinking has completed."""

    message_id: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """The model has begun emitting a tool-call."""

    tool_call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One incremental chunk of a streaming tool-call's JSON arguments."""

    tool_call_id: str
    arguments_delta: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """A tool-call's arguments have fully arrived."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelEnd:
    """The model has finished its full streaming response."""

    message_id: str
    usage: Usage


@dataclass(frozen=True, slots=True)
class ModelRetryRequest:
    """The loop will retry the same model call (e.g. validation failure)."""

    reason: str


# --- Tool execution ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolExecStart:
    """A tool's body is about to run (post-approval)."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecEnd:
    """A tool's body has completed."""

    tool_call_id: str
    result: ToolResult
    error: str | None = None


# --- Subagents ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubagentStart:
    """A nested-loop subagent has begun. The nested run's own ``AgentStart``
    and node events are also republished on the parent's bus."""

    parent_agent_name: str
    child_agent_name: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class SubagentStop:
    """A nested-loop subagent has finished."""

    parent_agent_name: str
    child_agent_name: str
    tool_call_id: str


# --- Approvals / compaction / elicitation / error ----------------------------


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    """One or more tool-calls require user approval."""

    requests: list[ApprovalRequest]


@dataclass(frozen=True, slots=True)
class ApprovalResolved:
    """An approval decision has been received."""

    tool_call_id: str
    approved: bool


@dataclass(frozen=True, slots=True)
class CompactionStart:
    """A history-compaction stage is about to run."""

    processor_name: str
    messages_before: int


@dataclass(frozen=True, slots=True)
class CompactionEnd:
    """A history-compaction stage has completed."""

    processor_name: str
    messages_after: int
    usage_added: Usage


@dataclass(frozen=True, slots=True)
class ElicitationRequested:
    """An MCP server has requested a structured value from the user."""

    server_name: str
    prompt: str
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Error:
    """A surfaced error; ``recoverable=True`` means the loop will continue."""

    message: str
    cause: type[BaseException] | None = None
    recoverable: bool = False


# --- The closed-set Event union ---------------------------------------------


Event = (
    RunStart
    | RunEnd
    | AgentStart
    | AgentEnd
    | NodeEnter
    | NodeExit
    | ModelStart
    | MessageStart
    | MessageDelta
    | MessageEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | ModelEnd
    | ModelRetryRequest
    | ToolExecStart
    | ToolExecEnd
    | SubagentStart
    | SubagentStop
    | ApprovalRequested
    | ApprovalResolved
    | CompactionStart
    | CompactionEnd
    | ElicitationRequested
    | Error
)
"""Closed-set union. Adding a member is a versioned API change."""


# --- EventBus + InMemoryEventBus --------------------------------------------


@runtime_checkable
class EventBus(Protocol):
    """Publish / subscribe with one queue per subscriber (EV2).

    Example:
        >>> isinstance(InMemoryEventBus(), EventBus)
        True
    """

    async def publish(self, event: Event) -> None: ...
    def subscribe(self, *, from_event: int | None = None) -> AsyncIterator[Event]: ...
    async def close(self) -> None: ...


DEFAULT_QUEUE_MAXSIZE = 1000
"""Default per-subscriber queue size."""


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[Event]
    end: asyncio.Event
    closed: bool = False
    dropped: int = 0


class InMemoryEventBus:
    """Default ``EventBus`` — one ``asyncio.Queue`` per subscriber.

    Single-subscriber FIFO (EV6); cross-subscriber order not guaranteed.
    Overflow default drops the oldest event in that subscriber's queue (EV3);
    ``strict=True`` raises ``asyncio.QueueFull`` instead. ``from_event`` is
    silently ignored — replay belongs to :class:`SqliteEventBus` (future).

    Example:
        >>> InMemoryEventBus(maxsize=4).maxsize
        4
    """

    def __init__(self, *, maxsize: int = DEFAULT_QUEUE_MAXSIZE, strict: bool = False) -> None:
        if maxsize <= 0:
            raise ConfigError(
                "maxsize must be positive",
                context={"maxsize": maxsize},
            )
        self.maxsize = maxsize
        self.strict = strict
        self._subscribers: list[_Subscriber] = []
        self._closed = False
        self._dropped_total = 0

    @property
    def dropped_total(self) -> int:
        """Total events dropped across all subscribers (diagnostic)."""
        return self._dropped_total

    async def publish(self, event: Event) -> None:
        """Fan ``event`` out to every subscriber's queue (non-blocking)."""
        if self._closed:
            raise BusClosedError("publish on closed EventBus")
        for sub in self._subscribers:
            if sub.closed:
                continue
            self._enqueue(sub, event)

    def _drop_one(self, sub: _Subscriber) -> None:
        """Increment both the per-subscriber and bus-total drop counters.

        Invariant: every dropped event bumps both counters exactly once.
        """
        sub.dropped += 1
        self._dropped_total += 1

    def _enqueue(self, sub: _Subscriber, event: Event) -> None:
        # Fast path: room in the queue.
        try:
            sub.queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            if self.strict:
                raise
        # Overflow: drop-oldest semantics (EV3). Drain one slot, account for
        # the drop, then put the new event. If draining races and finds the
        # queue empty (a consumer just popped), no drop occurred — fall
        # through and put.
        try:
            sub.queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - race with consumer
            pass
        else:
            self._drop_one(sub)
        # Re-insert the new event. If even that fails the queue is in an
        # unrecoverable state for this subscriber; count the new event as
        # dropped so the diagnostic counter stays truthful.
        try:
            sub.queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - paranoia
            self._drop_one(sub)

    def subscribe(self, *, from_event: int | None = None) -> AsyncIterator[Event]:
        """Return a fresh async iterator of live events for this subscriber."""
        del from_event  # accepted for protocol parity; no-op here.
        sub = _Subscriber(queue=asyncio.Queue(maxsize=self.maxsize), end=asyncio.Event())
        self._subscribers.append(sub)
        return _SubscriberIterator(sub)

    async def close(self) -> None:
        """Signal end-of-stream. Already-queued events still drain."""
        if self._closed:
            return
        self._closed = True
        for sub in self._subscribers:
            sub.end.set()


@dataclass(slots=True)
class _SubscriberIterator:
    """Async iterator over one subscriber's queue.

    Yields buffered events first, then stops once the bus is closed AND the
    buffer is drained.
    """

    _sub: _Subscriber
    _done: bool = field(default=False)

    def __aiter__(self) -> _SubscriberIterator:
        return self

    async def __anext__(self) -> Event:
        if self._done:
            raise StopAsyncIteration
        try:
            return self._sub.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        end_wait = asyncio.create_task(self._sub.end.wait())
        get_wait = asyncio.create_task(self._sub.queue.get())
        try:
            await asyncio.wait(
                {end_wait, get_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # Cancel the loser AND await it so the task isn't left pending
            # (avoids "Task was destroyed but it is pending!" warnings).
            if not end_wait.done():
                end_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await end_wait
        if get_wait.done():
            return get_wait.result()
        get_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await get_wait
        try:
            return self._sub.queue.get_nowait()
        except asyncio.QueueEmpty:
            self._done = True
            self._sub.closed = True
            raise StopAsyncIteration from None
