"""Unit tests for ``tests.fakes``.

Verifies the test fakes implement their Protocols structurally and that
:class:`FakeModel` chunks text into 3 cumulative deltas correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_harness.core.events import (
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from agent_harness.core.models import Model, ModelSettings, Provider
from agent_harness.core.tools import ToolCall
from tests.fakes import FakeModel, FakeProvider, FakeSandbox, FakeTurn, _chunk_into

# ---------------------------------------------------------------------------
# Protocol structural checks
# ---------------------------------------------------------------------------


def test_fake_provider_satisfies_provider_protocol() -> None:
    """``FakeProvider`` has the ``name`` / ``base_url`` / ``request`` shape."""

    assert isinstance(FakeProvider(), Provider)


def test_fake_model_satisfies_model_protocol() -> None:
    """``FakeModel`` has the canonical ``Model`` attributes & methods."""

    assert isinstance(FakeModel(script=[FakeTurn()]), Model)


def test_fake_sandbox_has_nine_method_surface() -> None:
    """``FakeSandbox`` duck-types the 9-method ``Sandbox`` Protocol."""

    fs = FakeSandbox()
    required = {
        "read_file",
        "write_file",
        "stat",
        "readdir",
        "exists",
        "mkdir",
        "rm",
        "read_file_bytes",
        "exec",
    }
    for attr in required:
        assert callable(getattr(fs, attr)), f"FakeSandbox missing method: {attr}"
    # Required Protocol attributes
    assert fs.root == "/workspace"
    assert fs.name == "fake-sandbox"


# ---------------------------------------------------------------------------
# _chunk_into
# ---------------------------------------------------------------------------


def test_chunk_into_returns_exactly_n_chunks() -> None:
    chunks = _chunk_into("abcdefghij", 3)
    assert len(chunks) == 3
    assert "".join(chunks) == "abcdefghij"


def test_chunk_into_empty_string_yields_n_empty_chunks() -> None:
    """An empty text should still yield 3 empty deltas so the
    "always 3 deltas per turn" invariant holds."""

    chunks = _chunk_into("", 3)
    assert chunks == ["", "", ""]


def test_chunk_into_short_string_concatenates_to_original() -> None:
    """Even when the string is shorter than ``n``, concat must round-trip."""

    chunks = _chunk_into("ab", 3)
    assert len(chunks) == 3
    assert "".join(chunks) == "ab"


def test_chunk_into_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        _chunk_into("hi", 0)


# ---------------------------------------------------------------------------
# FakeModel streaming behavior
# ---------------------------------------------------------------------------


async def _collect(model: FakeModel) -> list[Any]:
    out: list[Any] = []
    async for ev in model.request([], [], ModelSettings()):
        out.append(ev)
    return out


async def test_fake_model_emits_lifecycle_events_in_order() -> None:
    model = FakeModel(script=[FakeTurn(text="hello world!")])
    events = await _collect(model)

    # ModelStart -> MessageStart -> 3x MessageDelta -> MessageEnd -> ModelEnd.
    assert isinstance(events[0], ModelStart)
    assert isinstance(events[1], MessageStart)
    deltas = [e for e in events if isinstance(e, MessageDelta)]
    assert len(deltas) == 3
    assert isinstance(events[-2], MessageEnd)
    assert isinstance(events[-1], ModelEnd)


async def test_fake_model_chunks_text_into_three_cumulative_deltas() -> None:
    """``MessageDelta.partial`` must be cumulative (the EV5 invariant)."""

    text = "The quick brown fox."
    model = FakeModel(script=[FakeTurn(text=text)])
    events = await _collect(model)
    deltas = [e for e in events if isinstance(e, MessageDelta)]

    assert len(deltas) == 3
    # The chunked deltas reconstruct the full text.
    assert "".join(d.delta for d in deltas) == text
    # Cumulative property: each .partial.text starts-with the previous.
    prev = ""
    for d in deltas:
        cur = d.partial.text
        assert cur.startswith(prev), f"non-cumulative: {prev!r} -> {cur!r}"
        prev = cur
    # Final partial equals the full text.
    assert deltas[-1].partial.text == text


async def test_fake_model_emits_tool_call_events_after_text() -> None:
    tc = ToolCall(id="c1", name="read", arguments={"path": "foo.txt"})
    model = FakeModel(script=[FakeTurn(text="reading", tool_calls=[tc])])
    events = await _collect(model)

    types = [type(e) for e in events]
    # ToolCallStart/Delta/End all appear after the last MessageDelta.
    last_delta = max(i for i, t in enumerate(types) if t is MessageDelta)
    first_tc = next(i for i, t in enumerate(types) if t is ToolCallStart)
    assert first_tc > last_delta

    tc_starts = [e for e in events if isinstance(e, ToolCallStart)]
    tc_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    tc_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tc_starts) == len(tc_deltas) == len(tc_ends) == 1
    assert tc_ends[0].tool_name == "read"
    assert tc_ends[0].arguments == {"path": "foo.txt"}


async def test_fake_model_advances_turn_counter() -> None:
    model = FakeModel(script=[FakeTurn(text="one"), FakeTurn(text="two")])
    await _collect(model)
    assert model._turn == 1
    await _collect(model)
    assert model._turn == 2


async def test_fake_model_raises_when_script_exhausted() -> None:
    """A loop that makes one more request than scripted must fail loudly."""

    model = FakeModel(script=[FakeTurn(text="only")])
    await _collect(model)
    with pytest.raises(AssertionError, match="script exhausted"):
        await _collect(model)


async def test_fake_model_message_end_carries_full_message() -> None:
    tc = ToolCall(id="c1", name="noop", arguments={})
    model = FakeModel(script=[FakeTurn(text="done", tool_calls=[tc])])
    events = await _collect(model)
    msg_end = next(e for e in events if isinstance(e, MessageEnd))

    assert msg_end.final.text == "done"
    assert msg_end.final.has_tool_call()
    assert msg_end.final.tool_calls[0].id == "c1"


# ---------------------------------------------------------------------------
# FakeSandbox basics
# ---------------------------------------------------------------------------


async def test_fake_sandbox_write_then_read_roundtrips() -> None:
    fs = FakeSandbox()
    await fs.write_file("foo.txt", "hello")
    assert await fs.read_file("foo.txt") == "hello"


async def test_fake_sandbox_read_missing_raises() -> None:
    fs = FakeSandbox()
    with pytest.raises(FileNotFoundError):
        await fs.read_file("nope.txt")


async def test_fake_sandbox_exists_and_rm() -> None:
    fs = FakeSandbox()
    await fs.write_file("a.txt", "x")
    assert await fs.exists("a.txt") is True
    await fs.rm("a.txt")
    assert await fs.exists("a.txt") is False
