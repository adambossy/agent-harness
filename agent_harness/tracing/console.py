"""Console pretty-printer subscriber for the :class:`EventBus`.

A development / local-diagnostics subscriber that consumes events from an
:class:`agent_harness.core.events.EventBus` and prints a human-readable line
per event to a ``TextIO`` stream (``sys.stdout`` by default). Implements
the EV8 contract: tracing participates as a *subscriber*, never as a hook
(observation is never an intervention).

Color is applied via stdlib ANSI escape codes when the destination is a TTY
and ``color=True`` (default). The subscriber drives itself from a single
``asyncio.Task`` so a slow downstream stream never blocks the run loop;
the bus's per-subscriber queue absorbs bursts and drop-oldest semantics
kick in if the printer falls behind (EV3).

Example:
    >>> import asyncio
    >>> from agent_harness.core.events import InMemoryEventBus, RunStart
    >>> async def demo() -> str:
    ...     bus = InMemoryEventBus()
    ...     import io
    ...
    ...     buf = io.StringIO()
    ...     sub = ConsoleSubscriber(bus, stream=buf, color=False)
    ...     task = sub.start()
    ...     await bus.publish(RunStart(run_id="r1", agent_name="demo", prompt="hi"))
    ...     await bus.close()
    ...     await task
    ...     return buf.getvalue()
    >>> "RunStart" in asyncio.run(demo())
    True
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, TextIO

from agent_harness.core.events import (
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
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelRetryRequest,
    ModelStart,
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

if TYPE_CHECKING:  # pragma: no cover - typing-only
    pass


# --- ANSI escapes -----------------------------------------------------------
#
# Tiny stdlib palette. Avoid adding ``rich`` as a dep — these are plenty
# for "events show up colored in a dev terminal".

_RESET: Final[str] = "\x1b[0m"
_BOLD: Final[str] = "\x1b[1m"
_DIM: Final[str] = "\x1b[2m"

# Tier-keyed colors. Lifecycle = cyan, semantic = green, raw = magenta,
# error = red.
_COLOR_LIFECYCLE: Final[str] = "\x1b[36m"
_COLOR_SEMANTIC: Final[str] = "\x1b[32m"
_COLOR_RAW: Final[str] = "\x1b[35m"
_COLOR_WARN: Final[str] = "\x1b[33m"
_COLOR_ERROR: Final[str] = "\x1b[31m"


_LIFECYCLE: Final[tuple[type, ...]] = (
    RunStart,
    RunEnd,
    AgentStart,
    AgentEnd,
    NodeEnter,
    NodeExit,
    SubagentStart,
    SubagentStop,
)
_SEMANTIC: Final[tuple[type, ...]] = (
    MessageStart,
    MessageDelta,
    MessageEnd,
    ToolExecStart,
    ToolExecEnd,
    CompactionStart,
    CompactionEnd,
    ApprovalRequested,
    ApprovalResolved,
    ElicitationRequested,
)
_RAW: Final[tuple[type, ...]] = (
    ModelStart,
    ModelEnd,
    ModelRetryRequest,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
)


def _tier_color(event: Event) -> str:
    """Pick the ANSI color for an event's tier; never raises."""
    if isinstance(event, Error):
        return _COLOR_ERROR
    if isinstance(event, ModelRetryRequest):
        return _COLOR_WARN
    if isinstance(event, _LIFECYCLE):
        return _COLOR_LIFECYCLE
    if isinstance(event, _SEMANTIC):
        return _COLOR_SEMANTIC
    if isinstance(event, _RAW):
        return _COLOR_RAW
    return ""


def _truncate(value: str, *, limit: int = 80) -> str:
    """Single-line, length-bounded display for a raw string field."""
    s = value.replace("\n", "\\n").replace("\r", "\\r")
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _short_repr(value: object) -> str:
    """Compact, human-readable rendering for an event field's value."""
    if isinstance(value, str):
        return repr(_truncate(value))
    if isinstance(value, dict):
        keys = list(value.keys())
        if not keys:
            return "{}"
        return "{" + ", ".join(repr(k) for k in keys[:4]) + (", ..." if len(keys) > 4 else "") + "}"
    if isinstance(value, list):
        return f"[{len(value)} item{'s' if len(value) != 1 else ''}]"
    return _truncate(repr(value))


def _format_event(event: Event) -> str:
    """Build the unstyled body for an event line. One line, no escapes.

    Every member of the closed-set ``Event`` union is a frozen dataclass, so
    we can iterate :func:`dataclasses.fields` without a runtime guard.
    """
    name = type(event).__name__
    parts: list[str] = []
    for f in fields(event):
        # Skip the cumulative `partial` snapshot on deltas — `delta` is enough
        # for a dev printer and the snapshot is noisy.
        if f.name == "partial":
            continue
        parts.append(f"{f.name}={_short_repr(getattr(event, f.name))}")
    if not parts:
        return name
    return f"{name}({', '.join(parts)})"


class ConsoleSubscriber:
    """Pretty-print every published :class:`Event` to ``stream``.

    The subscriber runs in its own ``asyncio.Task`` (started via
    :meth:`start`), so the loop's ``publish`` call never blocks on stdio.
    Color is auto-disabled when ``stream`` is not a TTY unless explicitly
    forced. Stop the subscriber by closing the bus — the iterator drains
    and the task completes.

    Example:
        >>> from agent_harness.core.events import InMemoryEventBus
        >>> bus = InMemoryEventBus()
        >>> sub = ConsoleSubscriber(bus)
        >>> sub.color  # auto-detected from sys.stdout
        False
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        stream: TextIO | None = None,
        color: bool | None = None,
        timestamps: bool = True,
    ) -> None:
        self._bus = bus
        self._stream: TextIO = stream if stream is not None else sys.stdout
        if color is None:
            # Default: color only when writing to a real terminal.
            isatty = getattr(self._stream, "isatty", None)
            self.color: bool = bool(isatty() if callable(isatty) else False)
        else:
            self.color = color
        self._timestamps = timestamps
        self._task: asyncio.Task[None] | None = None
        # Acquire the subscription eagerly (synchronously) so that any event
        # published between ``__init__`` and the consumer task's first
        # scheduling reaches us. ``EventBus.subscribe`` is synchronous and
        # the bus appends to its subscriber list before returning.
        self._events = bus.subscribe()

    def start(self) -> asyncio.Task[None]:
        """Spawn the consumer task. Safe to call once per instance."""
        if self._task is not None:
            return self._task
        self._task = asyncio.create_task(self._run(), name="console-subscriber")
        return self._task

    async def _run(self) -> None:
        async for event in self._events:
            self._emit(event)

    def _emit(self, event: Event) -> None:
        line = self._render(event)
        # `print` keeps newline handling consistent across StringIO / TTY /
        # pipes; flush=True keeps dev output snappy.
        print(line, file=self._stream, flush=True)

    def _render(self, event: Event) -> str:
        body = _format_event(event)
        if self.color:
            color = _tier_color(event)
            if color:
                body = (
                    f"{color}{_BOLD}{type(event).__name__}{_RESET}{color}"
                    + body[len(type(event).__name__) :]
                    + _RESET
                )
        if not self._timestamps:
            return body
        ts = datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]
        prefix = f"{_DIM}{ts}{_RESET} " if self.color else f"{ts} "
        return f"{prefix}{body}"
