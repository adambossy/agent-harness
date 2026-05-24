"""LLM-driven (and deterministic-projection) ``HistoryProcessor`` built-ins.

This module ships the *expensive* compaction stages that the mechanical
processors in :mod:`agent_harness.core.history` are not allowed to express:

* :class:`MicroCompact` — per-tool-result LLM summarization, cache-preserving.
* :class:`ContextCollapse` — deterministic read-time projection of
  Read-Edit-Verify triplets into a single commit-shaped marker.
* :class:`SummarizeOldTurns` — full conversation summary via the model
  (Cline's 10-section template).
* :class:`ProviderSideCompaction` — delegate to
  ``ctx.agent.model.compact_messages``; no-op when the model lacks
  ``supports_compaction``.

Cache discipline (HP4) is enforced via content hashes embedded in every
LLM-emitted marker — re-running a processor against the same prefix
produces a byte-identical output, which keeps the Anthropic-style prompt
cache hot. Costs (input + output tokens) flow into ``ctx.usage`` via the
``+=`` operator (HP5).

Each processor is defensive about ``ctx`` because :class:`RunContext` is a
Layer-3 type the core does not yet own — every attribute lookup is wrapped
in :func:`getattr` with a safe fallback, so a Layer-2 caller can pass a
plain ``SimpleNamespace`` with just ``ctx.agent.model`` and ``ctx.usage``.

Example:
    >>> proc = MicroCompact(trigger_at_result_bytes=10_000)
    >>> proc.trigger_at_result_bytes
    10000
"""

from __future__ import annotations

import contextlib
import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .events import CompactionEnd, CompactionStart, ModelEnd
from .history import _block_byte_size, _iter_tool_results, _replace_block
from .models import (
    Message,
    ModelSettings,
    TextBlock,
    ToolResultBlock,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import asyncio
    from collections.abc import AsyncIterator


# Module-level task set keeping references to fire-and-forget event-publish
# tasks alive so they aren't GC'd before completion (RUF006).
_BG_TASKS: set[asyncio.Task[None]] = set()


# --- Public constants -------------------------------------------------------


DEFAULT_SUMMARY_TEMPLATE = """\
You are summarizing a long agent conversation so the agent can continue with
a tight context window. Produce a structured Markdown document with **exactly
these ten sections, in this order**, even if some are empty (write "(none)").

1. **Primary goal** — what the user asked for, in one sentence.
2. **Recent user intent** — what the user most recently steered toward.
3. **Key decisions** — choices the agent or user made and their rationale.
4. **Files touched** — paths and a one-line description per file.
5. **Commands run** — shell / tool invocations, with outcome (ok / failed).
6. **External facts gathered** — what the agent learned from the world.
7. **Open questions** — unresolved items the agent still owes the user.
8. **Errors and recoveries** — failures and how they were handled.
9. **Current state** — where the work stands right now.
10. **Next step** — the single concrete action the agent should take next.

Stay terse — prefer bullets to prose. Do not invent details that are not in
the conversation. Do not include verbatim file contents; reference them by
path. The output of this template will be used in place of the older turns.
"""
"""Cline-style 10-section structured summary template used by
:class:`SummarizeOldTurns`. Users override via the ``summary_template=``
kwarg without re-implementing the processor."""


MICROCOMPACT_MARKER_PREFIX = "[micro-compacted; sha256="
"""Identifies an already-LLM-summarized tool-result block.

Encoding: ``[micro-compacted; sha256=<64hex>] <summary text>``. The hash
is of the *original* content, so identical inputs ⇒ identical markers
(HP4: cache preservation)."""


SUMMARY_MARKER_PREFIX = "[summary; sha256="
"""Identifies an already-summarized older-turns block."""


COLLAPSE_MARKER_PREFIX = "[collapsed; sha256="
"""Identifies a deterministic Read-Edit-Verify collapse."""


# --- Helpers ----------------------------------------------------------------


def _bus(ctx: Any | None) -> Any | None:
    """Return ``ctx.event_bus`` if present, else ``None`` (defensive)."""
    return None if ctx is None else getattr(ctx, "event_bus", None)


async def _publish(bus: Any | None, event: Any) -> None:
    """Publish ``event`` on ``bus`` if it exposes an async ``publish``."""
    if bus is None:
        return
    publish = getattr(bus, "publish", None)
    if publish is not None:
        await publish(event)


def _content_hash(s: str | list[Any]) -> str:
    """SHA-256 of ``s`` rendered as UTF-8 bytes (hex)."""
    raw = s if isinstance(s, str) else str(s)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _is_micro_compacted(block: ToolResultBlock) -> bool:
    """True iff ``block`` was already summarized by :class:`MicroCompact`."""
    c = block.content
    return isinstance(c, str) and c.startswith(MICROCOMPACT_MARKER_PREFIX)


def _attribute_usage(ctx: Any | None, usage: Usage) -> None:
    """Add ``usage`` to ``ctx.usage`` (HP5).

    HP5 (cost attribution) is load-bearing for billing, so swallowing the
    write quietly is too risky. We tighten the suppress to
    ``AttributeError`` — the documented case where ``ctx.usage`` is not
    assignable (e.g. ``SimpleNamespace`` with a read-only descriptor or
    a frozen dataclass). On suppression we publish an ``Error`` event on
    ``ctx.event_bus`` (when present, ``recoverable=True``) so billing-side
    operators can see that attribution dropped on a turn rather than
    silently miss the line-item.

    Other exception types (e.g. ``TypeError`` from ``Usage.__add__``
    encountering a wrong operand) are deliberately *not* swallowed — they
    indicate a real bug at the call site and should surface.
    """
    if ctx is None:
        return
    existing = getattr(ctx, "usage", None)
    if existing is None:
        return
    try:
        ctx.usage = existing + usage
    except AttributeError as exc:
        # Documented case: ctx.usage is read-only. Surface via event bus.
        bus = _bus(ctx)
        if bus is None:
            return
        publish = getattr(bus, "publish", None)
        if publish is None:
            return
        from .events import Error  # local: avoid import cycle at module top

        event = Error(
            message=f"HP5 usage attribution dropped: {exc}",
            cause=type(exc),
            recoverable=True,
        )
        # Fire-and-forget: schedule on the running loop if any; otherwise
        # close the coro to avoid an "unawaited coroutine" warning.
        coro = publish(event)
        if hasattr(coro, "__await__"):
            import asyncio as _asyncio

            try:
                loop = _asyncio.get_running_loop()
                # Hold a reference so the GC doesn't reap a still-pending task.
                _BG_TASKS.add(task := loop.create_task(coro))
                task.add_done_callback(_BG_TASKS.discard)
            except RuntimeError:
                # No running loop — best we can do is close the coroutine
                # to avoid the unawaited-coro warning. Fail-soft.
                with contextlib.suppress(Exception):
                    coro.close()


def _model_from_ctx(ctx: Any | None) -> Any | None:
    """Return ``ctx.agent.model`` if reachable, else ``None``."""
    if ctx is None:
        return None
    agent = getattr(ctx, "agent", None)
    return None if agent is None else getattr(agent, "model", None)


async def _run_model_summary(
    model: Any,
    instruction: str,
    body: str,
    ctx: Any | None,
) -> str:
    """Drive ``model.request`` with a one-shot summary prompt.

    Returns the assistant's final text; attributes ``ModelEnd.usage`` to
    ``ctx.usage`` (HP5). Sends a tiny two-message slice so that providers
    treat it as a fresh turn (no tool catalog).
    """
    now = datetime.now(UTC)
    msgs: list[Message] = [
        Message(role="system", content=[TextBlock(text=instruction)], timestamp=now),
        Message(role="user", content=[TextBlock(text=body)], timestamp=now),
    ]
    final_text = ""
    stream: AsyncIterator[Any] = model.request(msgs, [], ModelSettings())
    async for ev in stream:
        final = getattr(ev, "final", None)
        if final is not None:
            text = getattr(final, "text", None)
            if isinstance(text, str) and text:
                final_text = text
        if isinstance(ev, ModelEnd):
            _attribute_usage(ctx, ev.usage)
    return final_text


# --- MicroCompact -----------------------------------------------------------


MICROCOMPACT_INSTRUCTION = (
    "Summarize the following tool result for an agent. Keep paths, identifiers, "
    "and error messages verbatim; drop verbose payload bodies; aim for under "
    "400 characters."
)


class MicroCompact:
    """LLM-driven per-tool-result summarization (cache-preserving).

    Rewrites every :class:`ToolResultBlock` whose UTF-8 byte size exceeds
    ``trigger_at_result_bytes`` *and* is not already summarized, replacing
    its content with ``[micro-compacted; sha256=<hash>] <summary>``. The
    hash is of the *original* body, so re-running on the same prefix
    produces byte-identical output (HP4). Already-summarized blocks are
    skipped — only newer items pay the LLM cost. Usage is attributed to
    ``ctx.usage`` (HP5).

    Example:
        >>> MicroCompact(trigger_at_result_bytes=42).trigger_at_result_bytes
        42
    """

    def __init__(self, trigger_at_result_bytes: int = 50_000) -> None:
        if trigger_at_result_bytes < 0:
            raise ValueError(
                f"trigger_at_result_bytes must be >= 0; got {trigger_at_result_bytes!r}"
            )
        self.trigger_at_result_bytes = trigger_at_result_bytes
        # original-hash -> summary marker; same input ⇒ same marker bytes.
        self._cache: dict[str, str] = {}

    async def __call__(self, msgs: list[Message], ctx: Any | None = None) -> list[Message]:
        model = _model_from_ctx(ctx)
        if model is None:
            return list(msgs)
        targets = [
            (mi, bi, blk)
            for (mi, bi, blk) in _iter_tool_results(msgs)
            if not _is_micro_compacted(blk)
            and _block_byte_size(blk.content) > self.trigger_at_result_bytes
        ]
        if not targets:
            return list(msgs)
        bus = _bus(ctx)
        await _publish(
            bus, CompactionStart(processor_name="MicroCompact", messages_before=len(msgs))
        )
        out = list(msgs)
        for mi, bi, blk in targets:
            content_str = blk.content if isinstance(blk.content, str) else str(blk.content)
            h = _content_hash(content_str)
            cached = self._cache.get(h)
            if cached is None:
                summary = await _run_model_summary(
                    model, MICROCOMPACT_INSTRUCTION, content_str, ctx
                )
                cached = f"{MICROCOMPACT_MARKER_PREFIX}{h}] {summary}"
                self._cache[h] = cached
            out[mi] = _replace_block(
                out[mi], bi, ToolResultBlock(tool_call_id=blk.tool_call_id, content=cached)
            )
        await _publish(
            bus,
            CompactionEnd(
                processor_name="MicroCompact", messages_after=len(out), usage_added=Usage()
            ),
        )
        return out


# --- ContextCollapse --------------------------------------------------------


def _single_named_tool_call(msg: Message, names: frozenset[str]) -> Any | None:
    """Return the single ``ToolCallBlock`` in ``msg`` whose name is in
    ``names`` — or ``None`` if ``msg`` has zero or more than one call."""
    from .models import ToolCallBlock

    calls = [b for b in msg.content if isinstance(b, ToolCallBlock)]
    if len(calls) != 1:
        return None
    return calls[0] if calls[0].name in names else None


def _is_tool_result(msg: Message) -> bool:
    return any(isinstance(b, ToolResultBlock) for b in msg.content)


def _path_arg(tc: Any) -> str | None:
    args = getattr(tc, "arguments", {}) or {}
    for k in ("path", "file_path", "filepath"):
        v = args.get(k)
        if isinstance(v, str):
            return v
    return None


class ContextCollapse:
    """Collapse Read-Edit(-Verify) message runs into one commit marker.

    Deterministic — no LLM. Matches the canonical edit-flow shape:

    1. assistant ``Read`` tool-call + tool-result reply,
    2. assistant ``Edit`` / ``Write`` / ``Patch`` tool-call + tool-result,
    3. optional verify pair (``Grep`` / ``RunTests``).

    Each matched run becomes one ``user``-role message containing
    ``[collapsed; sha256=<hash>] Read+Edit on <path>``. Identical inputs
    map to identical markers (HP4 cache safety, no LLM cost).

    Example:
        >>> isinstance(ContextCollapse(), ContextCollapse)
        True
    """

    READ_NAMES = frozenset({"Read", "ReadFile", "read_file", "read"})
    EDIT_NAMES = frozenset({"Edit", "Write", "Patch", "edit", "write", "patch"})
    VERIFY_NAMES = frozenset({"Grep", "RunTests", "grep", "run_tests"})

    async def __call__(self, msgs: list[Message], ctx: Any | None = None) -> list[Message]:
        spans: list[tuple[int, int, str]] = []
        i = 0
        while i + 3 < len(msgs):
            read_msg, read_res, edit_msg, edit_res = msgs[i : i + 4]
            read_tc = _single_named_tool_call(read_msg, self.READ_NAMES)
            edit_tc = _single_named_tool_call(edit_msg, self.EDIT_NAMES)
            if (
                read_tc is not None
                and edit_tc is not None
                and _is_tool_result(read_res)
                and _is_tool_result(edit_res)
            ):
                end = i + 4
                if (
                    end + 1 < len(msgs)
                    and _single_named_tool_call(msgs[end], self.VERIFY_NAMES) is not None
                    and _is_tool_result(msgs[end + 1])
                ):
                    end += 2
                path = _path_arg(read_tc) or _path_arg(edit_tc) or "<unknown>"
                spans.append((i, end, path))
                i = end
                continue
            i += 1
        if not spans:
            return list(msgs)
        bus = _bus(ctx)
        await _publish(
            bus, CompactionStart(processor_name="ContextCollapse", messages_before=len(msgs))
        )
        out: list[Message] = []
        cursor = 0
        for start, end, path in spans:
            out.extend(msgs[cursor:start])
            h = _content_hash("|".join(m.model_dump_json() for m in msgs[start:end]))
            out.append(
                Message(
                    role="user",
                    content=[TextBlock(text=f"{COLLAPSE_MARKER_PREFIX}{h}] Read+Edit on {path}")],
                    timestamp=msgs[start].timestamp,
                )
            )
            cursor = end
        out.extend(msgs[cursor:])
        await _publish(
            bus,
            CompactionEnd(
                processor_name="ContextCollapse", messages_after=len(out), usage_added=Usage()
            ),
        )
        return out


# --- SummarizeOldTurns ------------------------------------------------------


def _is_summary_message(msg: Message) -> bool:
    """True iff ``msg`` is a single-block marker emitted by SummarizeOldTurns."""
    if len(msg.content) != 1:
        return False
    block = msg.content[0]
    return isinstance(block, TextBlock) and block.text.startswith(SUMMARY_MARKER_PREFIX)


def _summary_input(msgs: list[Message]) -> str:
    """Render ``msgs`` into the deterministic text body used by SummarizeOldTurns.

    The output of this function is BOTH:

    1. the body shown to the model for summarization, AND
    2. the input to the SHA-256 that keys :class:`SummarizeOldTurns`'s cache
       (the marker preserved across turns is ``[summary; sha256=<hash>]``).

    Because the hash is load-bearing for HP4 (cache preservation across
    turns), the format must be stable. Specifically, it concatenates
    ``[<role>] <text>`` per message, joined by ``\\n\\n`` — using only
    ``Message.role`` and ``Message.text`` (which currently is the
    ``TextBlock`` text concatenated, *not* timestamps or tool-call IDs).

    If a future change adds non-text material to ``Message.text`` (e.g.
    folds in timestamps or tool-call IDs), the cache invalidates between
    turns even for the same logical prefix. Reviewers: keep this in sync
    with the ``Message.text`` accessor.
    """
    return "\n\n".join(f"[{m.role}] {m.text}" for m in msgs if m.text)


class SummarizeOldTurns:
    """LLM-summarize the oldest turns when estimated tokens exceed a threshold.

    When ``_estimate_tokens(msgs) > trigger_at_tokens`` and there are more
    than ``keep_recent_turns`` messages, keeps the most recent
    ``keep_recent_turns`` verbatim and folds the rest into one
    ``[summary; sha256=<hash>] <body>`` marker (HP4: hash of original text
    keys the cache, so identical prefixes yield identical bytes). Costs
    attributed to ``ctx.usage`` (HP5).

    Example:
        >>> SummarizeOldTurns(trigger_at_tokens=100).keep_recent_turns
        6
    """

    def __init__(
        self,
        trigger_at_tokens: int,
        keep_recent_turns: int = 6,
        summary_template: str = DEFAULT_SUMMARY_TEMPLATE,
    ) -> None:
        if trigger_at_tokens < 0:
            raise ValueError(f"trigger_at_tokens must be >= 0; got {trigger_at_tokens!r}")
        if keep_recent_turns < 0:
            raise ValueError(f"keep_recent_turns must be >= 0; got {keep_recent_turns!r}")
        self.trigger_at_tokens = trigger_at_tokens
        self.keep_recent_turns = keep_recent_turns
        self.summary_template = summary_template
        self._cache: dict[str, str] = {}

    def _estimate_tokens(self, msgs: list[Message]) -> int:
        # Heuristic: ~4 chars/token (matches ``history.CHARS_PER_TOKEN``).
        return sum(len(m.model_dump_json()) for m in msgs) // 4

    async def __call__(self, msgs: list[Message], ctx: Any | None = None) -> list[Message]:
        if self._estimate_tokens(msgs) <= self.trigger_at_tokens:
            return list(msgs)
        if len(msgs) <= self.keep_recent_turns:
            return list(msgs)
        model = _model_from_ctx(ctx)
        if model is None:
            return list(msgs)
        keep_start = len(msgs) - self.keep_recent_turns
        old, recent = msgs[:keep_start], msgs[keep_start:]
        if len(old) == 1 and _is_summary_message(old[0]):
            return list(msgs)
        body = _summary_input(old)
        h = _content_hash(body)
        cached = self._cache.get(h)
        bus = _bus(ctx)
        await _publish(
            bus, CompactionStart(processor_name="SummarizeOldTurns", messages_before=len(msgs))
        )
        if cached is None:
            summary = await _run_model_summary(model, self.summary_template, body, ctx)
            cached = f"{SUMMARY_MARKER_PREFIX}{h}] {summary}"
            self._cache[h] = cached
        out = [
            Message(role="user", content=[TextBlock(text=cached)], timestamp=old[0].timestamp),
            *recent,
        ]
        await _publish(
            bus,
            CompactionEnd(
                processor_name="SummarizeOldTurns", messages_after=len(out), usage_added=Usage()
            ),
        )
        return out


# --- ProviderSideCompaction -------------------------------------------------


class ProviderSideCompaction:
    """Delegate compaction to the active model (HP10).

    Calls ``ctx.agent.model.compact_messages(msgs)`` when
    ``model.capabilities.supports_compaction`` is True; otherwise a no-op
    that returns ``msgs`` unchanged. The model is responsible for its own
    cache discipline; this processor is a thin delegate.

    Example:
        >>> isinstance(ProviderSideCompaction(), ProviderSideCompaction)
        True
    """

    async def __call__(self, msgs: list[Message], ctx: Any | None = None) -> list[Message]:
        model = _model_from_ctx(ctx)
        if model is None:
            return list(msgs)
        caps = getattr(model, "capabilities", None)
        if caps is None or not getattr(caps, "supports_compaction", False):
            return list(msgs)
        bus = _bus(ctx)
        await _publish(
            bus,
            CompactionStart(processor_name="ProviderSideCompaction", messages_before=len(msgs)),
        )
        result = list(await model.compact_messages(msgs))
        await _publish(
            bus,
            CompactionEnd(
                processor_name="ProviderSideCompaction",
                messages_after=len(result),
                usage_added=Usage(),
            ),
        )
        return result
