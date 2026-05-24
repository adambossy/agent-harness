"""History processors — mechanical compaction primitives.

A :data:`HistoryProcessor` is any callable mapping ``list[Message]`` to a
new ``list[Message]``. The harness auto-detects the call shape via
:func:`inspect.signature` / :func:`inspect.iscoroutinefunction`, supporting
four signatures so users can drop in plain functions without boilerplate
(pydantic-ai's pattern, HP1):

* ``def proc(msgs)``
* ``def proc(msgs, ctx)``
* ``async def proc(msgs)``
* ``async def proc(msgs, ctx)``

This module ships only the *mechanical* built-ins — those that do not call
a model. LLM-driven processors (``MicroCompact``, ``ContextCollapse``,
``SummarizeOldTurns``, ``ProviderSideCompaction``) live in a later wave;
they need :class:`Model` / :class:`RunContext` access.

Each built-in is pure: same input ⇒ same output, no side effects except
the byte/token accounting the caller may inspect. That determinism is what
makes prompt-cache preservation possible (HP4): older transformed prefixes
stay byte-identical across turns.

Example:
    >>> from datetime import datetime, timezone
    >>> from agent_harness.core.models import Message, TextBlock
    >>> msgs = [
    ...     Message(
    ...         role="user",
    ...         content=[TextBlock(text="hi")],
    ...         timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ...     )
    ... ]
    >>> processor = HistorySnip()
    >>> processor(msgs) == msgs
    True
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .models import Message, ToolResultBlock

# --- Public type alias -------------------------------------------------------


HistoryProcessor = Callable[..., list[Message] | Awaitable[list[Message]]]
"""A processor is any callable matching one of four signatures:

* ``(msgs) -> list[Message]``
* ``(msgs, ctx) -> list[Message]``
* ``async (msgs) -> list[Message]``
* ``async (msgs, ctx) -> list[Message]``

Use :func:`apply_processor` to invoke one without caring which shape it has.

Example:
    >>> async def my_proc(msgs):
    ...     return msgs
    >>> import inspect
    >>> inspect.iscoroutinefunction(my_proc)
    True
"""


# --- Heuristics --------------------------------------------------------------


CHARS_PER_TOKEN = 4
"""Heuristic used when no real tokenizer is available.

Anthropic / OpenAI tokenizers average ~4 characters per English token; we
use that constant for budgeting decisions. Concrete adapters with a real
tokenizer should override their own counter — this module never imports
``tiktoken`` or ``anthropic``.
"""


TRUNCATED_MARKER = "[truncated by TokenBudgetCap]"
"""Marker text replacing tool-result bodies elided by :class:`TokenBudgetCap`.

The same constant string is reused on every turn so that older,
already-capped results stay byte-identical (HP4: cache preservation)."""


DEDUP_MARKER_FMT = "[deduped by DedupFileReads; superseded by a later read]"
"""Placeholder body left in earlier duplicate file-read results."""


# --- Arity detection + invocation -------------------------------------------


def _processor_takes_ctx(proc: HistoryProcessor) -> bool:
    """Return True iff ``proc`` accepts a second positional ``ctx`` argument.

    Falls back to *single-arg* on any ``ValueError`` from inspect (e.g. some
    builtins refuse signature inspection); the loop simply won't supply
    ``ctx``. Variadic ``*args`` is treated as multi-arg.
    """
    try:
        sig = inspect.signature(proc)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in positional):
        return True
    return len(positional) >= 2


async def apply_processor(
    proc: HistoryProcessor,
    msgs: list[Message],
    ctx: Any | None = None,
) -> list[Message]:
    """Invoke ``proc`` with the correct arity / awaitability.

    The loop calls this once per stage; it handles all four supported
    signatures transparently. ``ctx`` is forward-declared as :class:`Any`
    because :class:`RunContext` is a Layer-3 type the core does not yet
    know about.

    Example:
        >>> import asyncio
        >>> def double(msgs):
        ...     return msgs + msgs
        >>> asyncio.run(apply_processor(double, []))
        []
    """
    needs_ctx = _processor_takes_ctx(proc)
    args: tuple[Any, ...] = (msgs, ctx) if needs_ctx else (msgs,)
    result = proc(*args)
    if inspect.iscoroutine(result) or isinstance(result, Awaitable):
        # ``isinstance(result, Awaitable)`` covers third-party Future-like
        # objects; ``iscoroutine`` is the fast path.
        awaited = await asyncio.ensure_future(result)
        return list(awaited)
    # Sync return path — copy into a fresh list so callers can't mutate the
    # processor's internal state by appending to the result.
    return list(result)


# --- Helpers ----------------------------------------------------------------


def _block_byte_size(content: str | list[Any]) -> int:
    """Best-effort byte size of a :class:`ToolResultBlock`'s content."""
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    # Lists carry MCP-style block dicts; fall back to ``str(...)`` cost.
    return len(str(content).encode("utf-8"))


def _iter_tool_results(msgs: list[Message]) -> list[tuple[int, int, ToolResultBlock]]:
    """Yield ``(msg_idx, block_idx, block)`` for every tool-result block."""
    out: list[tuple[int, int, ToolResultBlock]] = []
    for mi, m in enumerate(msgs):
        for bi, b in enumerate(m.content):
            if isinstance(b, ToolResultBlock):
                out.append((mi, bi, b))
    return out


def _replace_block(msg: Message, block_idx: int, new_block: ToolResultBlock) -> Message:
    """Return a copy of ``msg`` with ``block_idx`` replaced by ``new_block``."""
    new_content = list(msg.content)
    new_content[block_idx] = new_block
    return msg.model_copy(update={"content": new_content})


# --- Built-in processors ----------------------------------------------------


class TokenBudgetCap:
    """Cap cumulative tool-result bytes at ``max_bytes`` (newest preserved).

    Walks tool results newest-to-oldest, accumulating their UTF-8 byte
    size; once the cumulative total exceeds ``max_bytes`` every *older*
    result is rewritten to the constant :data:`TRUNCATED_MARKER` string.
    Newer results that fit under the budget are untouched.

    The marker is a fixed constant so already-capped older results stay
    byte-identical across turns — required for prompt-cache preservation
    (HP4). Idempotent: running twice produces the same output as once.

    Example:
        >>> import asyncio
        >>> from datetime import datetime, timezone
        >>> from agent_harness.core.models import Message, ToolResultBlock
        >>> def _m(c):
        ...     return Message(
        ...         role="tool", content=[c], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
        ...     )
        >>> msgs = [
        ...     _m(ToolResultBlock(tool_call_id="c1", content="x" * 1000)),
        ...     _m(ToolResultBlock(tool_call_id="c2", content="y" * 200)),
        ... ]
        >>> capped = asyncio.run(apply_processor(TokenBudgetCap(max_bytes=500), msgs))
        >>> capped[0].content[0].content.startswith("[truncated")
        True
    """

    def __init__(self, max_bytes: int = 500_000) -> None:
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be >= 0; got {max_bytes!r}")
        self.max_bytes = max_bytes

    def __call__(self, msgs: list[Message]) -> list[Message]:
        results = _iter_tool_results(msgs)
        if not results:
            return list(msgs)
        cumulative = 0
        keep_until_idx = len(results)  # index into ``results``; default: keep all
        for i in range(len(results) - 1, -1, -1):
            _, _, blk = results[i]
            cumulative += _block_byte_size(blk.content)
            if cumulative > self.max_bytes:
                keep_until_idx = i
                break
        else:
            return list(msgs)
        out = list(msgs)
        for j in range(keep_until_idx + 1):
            mi, bi, blk = results[j]
            if isinstance(blk.content, str) and blk.content == TRUNCATED_MARKER:
                continue  # already capped; idempotency
            out[mi] = _replace_block(
                out[mi],
                bi,
                ToolResultBlock(tool_call_id=blk.tool_call_id, content=TRUNCATED_MARKER),
            )
        return out


class DedupFileReads:
    """Collapse duplicate file-read tool results into placeholders.

    Cline's optimization: when an agent reads the same file multiple times
    in one run, keep only the latest ``keep_latest_n`` reads verbatim and
    rewrite earlier reads to :data:`DEDUP_MARKER_FMT` so the model still
    sees that the read happened but does not pay tokens for stale content.

    Identification uses the originating tool-call's ``arguments['path']``
    or ``arguments['file_path']`` (the de-facto field names across Read /
    ReadFile tools). Calls without such an argument are ignored. The
    deterministic, content-free placeholder preserves the prompt cache.

    Example:
        >>> import asyncio
        >>> from datetime import datetime, timezone
        >>> from agent_harness.core.models import Message, ToolCallBlock, ToolResultBlock
        >>> ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        >>> msgs = [
        ...     Message(
        ...         role="assistant",
        ...         content=[ToolCallBlock(id="c1", name="Read", arguments={"path": "/a"})],
        ...         timestamp=ts,
        ...     ),
        ...     Message(
        ...         role="tool",
        ...         content=[ToolResultBlock(tool_call_id="c1", content="v1")],
        ...         timestamp=ts,
        ...     ),
        ...     Message(
        ...         role="assistant",
        ...         content=[ToolCallBlock(id="c2", name="Read", arguments={"path": "/a"})],
        ...         timestamp=ts,
        ...     ),
        ...     Message(
        ...         role="tool",
        ...         content=[ToolResultBlock(tool_call_id="c2", content="v2")],
        ...         timestamp=ts,
        ...     ),
        ... ]
        >>> out = asyncio.run(apply_processor(DedupFileReads(), msgs))
        >>> out[1].content[0].content.startswith("[deduped")
        True
        >>> out[3].content[0].content
        'v2'
    """

    READ_TOOL_NAMES = frozenset({"Read", "ReadFile", "read_file", "read"})
    PATH_ARG_NAMES = ("path", "file_path", "filepath")

    def __init__(self, keep_latest_n: int = 1) -> None:
        if keep_latest_n < 1:
            raise ValueError(f"keep_latest_n must be >= 1; got {keep_latest_n!r}")
        self.keep_latest_n = keep_latest_n

    def _path_from_call(self, args: dict[str, Any]) -> str | None:
        for k in self.PATH_ARG_NAMES:
            v = args.get(k)
            if isinstance(v, str):
                return v
        return None

    def __call__(self, msgs: list[Message]) -> list[Message]:
        # Pass 1: map tool_call_id -> file path (for read-shaped calls).
        path_by_call_id: dict[str, str] = {}
        for m in msgs:
            for b in m.content:
                if (
                    hasattr(b, "name")
                    and getattr(b, "name", None) in self.READ_TOOL_NAMES
                    and hasattr(b, "id")
                ):
                    args = getattr(b, "arguments", {}) or {}
                    path = self._path_from_call(args)
                    if path is not None:
                        path_by_call_id[b.id] = path
        if not path_by_call_id:
            return list(msgs)
        # Pass 2: index results by path in encounter order.
        results = _iter_tool_results(msgs)
        by_path: dict[str, list[int]] = {}
        for idx, (_, _, blk) in enumerate(results):
            path = path_by_call_id.get(blk.tool_call_id)
            if path is None:
                continue
            by_path.setdefault(path, []).append(idx)
        # Pass 3: rewrite all but the latest ``keep_latest_n`` per path.
        out = list(msgs)
        for indices in by_path.values():
            if len(indices) <= self.keep_latest_n:
                continue
            to_replace = indices[: -self.keep_latest_n]
            for ri in to_replace:
                mi, bi, blk = results[ri]
                if isinstance(blk.content, str) and blk.content == DEDUP_MARKER_FMT:
                    continue
                out[mi] = _replace_block(
                    out[mi],
                    bi,
                    ToolResultBlock(tool_call_id=blk.tool_call_id, content=DEDUP_MARKER_FMT),
                )
        return out


class HistorySnip:
    """Drop empty messages — cheap structural snip that costs nothing.

    The pipeline accumulates an assortment of low-signal messages over a
    long run: empty assistant turns left behind by streaming retries,
    tool-result envelopes whose content was emptied by an earlier capping
    stage, and so on. This processor removes any message whose ``content``
    list is empty, leaving everything else byte-identical (HP4).

    Pure / idempotent — running twice is the same as once. System messages
    are *never* dropped even when empty: they may carry metadata that the
    Provider injects downstream.

    Example:
        >>> import asyncio
        >>> from datetime import datetime, timezone
        >>> from agent_harness.core.models import Message, TextBlock
        >>> ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        >>> msgs = [
        ...     Message(role="user", content=[TextBlock(text="hi")], timestamp=ts),
        ...     Message(role="assistant", content=[], timestamp=ts),
        ... ]
        >>> len(asyncio.run(apply_processor(HistorySnip(), msgs)))
        1
    """

    def __call__(self, msgs: list[Message]) -> list[Message]:
        return [m for m in msgs if m.role == "system" or m.content]
