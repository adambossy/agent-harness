"""Unit tests for the :data:`HistoryProcessor` callable protocol.

Verifies the four supported arities (sync/async, msgs-only/msgs+ctx) and
the :func:`apply_processor` dispatcher's behavior on edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from agent_harness.core.history import (
    HistoryProcessor,
    _processor_takes_ctx,
    apply_processor,
)
from agent_harness.core.models import Message, TextBlock


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _msgs(n: int = 1) -> list[Message]:
    return [
        Message(role="user", content=[TextBlock(text=f"m{i}")], timestamp=_ts()) for i in range(n)
    ]


# --- Arity detection --------------------------------------------------------


def test_arity_detection_sync_msgs_only() -> None:
    def proc(msgs: list[Message]) -> list[Message]:
        return msgs

    assert _processor_takes_ctx(proc) is False


def test_arity_detection_sync_msgs_and_ctx() -> None:
    def proc(msgs: list[Message], ctx: Any) -> list[Message]:
        return msgs

    assert _processor_takes_ctx(proc) is True


def test_arity_detection_async_msgs_only() -> None:
    async def proc(msgs: list[Message]) -> list[Message]:
        return msgs

    assert _processor_takes_ctx(proc) is False


def test_arity_detection_async_msgs_and_ctx() -> None:
    async def proc(msgs: list[Message], ctx: Any) -> list[Message]:
        return msgs

    assert _processor_takes_ctx(proc) is True


def test_arity_detection_handles_var_positional() -> None:
    def proc(*args: Any) -> list[Message]:
        return list(args[0])

    assert _processor_takes_ctx(proc) is True


def test_arity_detection_handles_unintrospectable_callable() -> None:
    # ``int`` rejects ``inspect.signature``; we fall back to msgs-only.
    assert _processor_takes_ctx(int) is False  # type: ignore[arg-type]


# --- apply_processor: each shape -------------------------------------------


async def test_apply_processor_sync_msgs_only() -> None:
    def proc(msgs: list[Message]) -> list[Message]:
        return msgs + msgs

    msgs = _msgs(2)
    out = await apply_processor(proc, msgs)
    assert len(out) == 4


async def test_apply_processor_sync_msgs_and_ctx() -> None:
    seen_ctx: list[Any] = []

    def proc(msgs: list[Message], ctx: Any) -> list[Message]:
        seen_ctx.append(ctx)
        return msgs

    sentinel = object()
    await apply_processor(proc, _msgs(), ctx=sentinel)
    assert seen_ctx == [sentinel]


async def test_apply_processor_async_msgs_only() -> None:
    async def proc(msgs: list[Message]) -> list[Message]:
        return list(reversed(msgs))

    msgs = _msgs(3)
    out = await apply_processor(proc, msgs)
    texts: list[str] = []
    for m in out:
        block = m.content[0]
        assert isinstance(block, TextBlock)
        texts.append(block.text)
    assert texts == ["m2", "m1", "m0"]


async def test_apply_processor_async_msgs_and_ctx() -> None:
    captured: dict[str, Any] = {}

    async def proc(msgs: list[Message], ctx: Any) -> list[Message]:
        captured["ctx"] = ctx
        captured["len"] = len(msgs)
        return msgs

    await apply_processor(proc, _msgs(2), ctx="ctx-value")
    assert captured == {"ctx": "ctx-value", "len": 2}


async def test_apply_processor_returns_a_copy_not_caller_list() -> None:
    """The returned list must not be the processor's internal one — callers
    that ``.append`` to the result must not corrupt processor state.
    """
    internal: list[Message] = _msgs(1)

    def proc(_msgs: list[Message]) -> list[Message]:
        return internal

    out = await apply_processor(proc, [])
    out.append(_msgs(1)[0])
    assert len(internal) == 1


async def test_apply_processor_ignores_ctx_when_processor_does_not_take_it() -> None:
    """Passing a ctx to a msgs-only processor is harmless — no TypeError."""

    def proc(msgs: list[Message]) -> list[Message]:
        return msgs

    out = await apply_processor(proc, _msgs(1), ctx="ignored")
    assert len(out) == 1


# --- Type alias is usable in annotations -----------------------------------


def test_history_processor_type_alias_accepts_all_shapes() -> None:
    """Smoke check that the alias is structurally satisfied by all 4 shapes."""

    def s1(msgs: list[Message]) -> list[Message]:
        return msgs

    def s2(msgs: list[Message], ctx: Any) -> list[Message]:
        return msgs

    async def s3(msgs: list[Message]) -> list[Message]:
        return msgs

    async def s4(msgs: list[Message], ctx: Any) -> list[Message]:
        return msgs

    procs: list[HistoryProcessor] = [s1, s2, s3, s4]
    assert len(procs) == 4


# --- Error path: a processor that raises must propagate --------------------


async def test_apply_processor_propagates_processor_exception() -> None:
    def proc(_msgs: list[Message]) -> list[Message]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await apply_processor(proc, _msgs(1))


async def test_apply_processor_propagates_async_processor_exception() -> None:
    async def proc(_msgs: list[Message]) -> list[Message]:
        raise ValueError("async boom")

    with pytest.raises(ValueError, match="async boom"):
        await apply_processor(proc, _msgs(1))
