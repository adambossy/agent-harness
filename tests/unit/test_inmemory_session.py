"""Unit tests for ``agent_harness.sessions.inmemory.InMemorySession``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from agent_harness.core.models import Message, TextBlock, Usage
from agent_harness.core.run_state import RunStateSnapshot
from agent_harness.sessions.inmemory import InMemorySession


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _msg(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)], timestamp=_ts())


def _snap(run_id: str = "r1", node: str = "ModelRequest") -> RunStateSnapshot:
    return RunStateSnapshot(
        run_id=run_id,
        agent_name="demo",
        current_node=node,
        usage=Usage(),
        created_at=_ts(),
    )


# --- session_id + initial state ---------------------------------------------


async def test_session_id_is_exposed() -> None:
    sess = InMemorySession(session_id="abc")
    assert sess.session_id == "abc"


async def test_initial_state_is_empty() -> None:
    sess = InMemorySession(session_id="s")
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
    assert await sess.get_latest_run_state() is None


# --- messages ---------------------------------------------------------------


async def test_add_and_get_messages_preserves_order() -> None:
    sess = InMemorySession(session_id="s")
    await sess.add_messages([_msg("a"), _msg("b")])
    await sess.add_messages([_msg("c")])
    out = await sess.get_messages()
    assert [m.text for m in out] == ["a", "b", "c"]


async def test_get_messages_returns_copy_not_internal_list() -> None:
    """Mutating the returned list must not affect later reads."""

    sess = InMemorySession(session_id="s")
    await sess.add_messages([_msg("a")])
    snapshot = await sess.get_messages()
    snapshot.append(_msg("INJECTED"))
    again = await sess.get_messages()
    assert [m.text for m in again] == ["a"]


async def test_add_messages_empty_list_is_noop() -> None:
    sess = InMemorySession(session_id="s")
    await sess.add_messages([])
    assert await sess.get_messages() == []


# --- snapshots --------------------------------------------------------------


async def test_add_run_state_and_get_latest() -> None:
    sess = InMemorySession(session_id="s")
    await sess.add_run_state(_snap("r1"))
    await sess.add_run_state(_snap("r2"))
    latest = await sess.get_latest_run_state()
    assert latest is not None
    assert latest.run_id == "r2"


async def test_get_run_states_returns_oldest_first() -> None:
    sess = InMemorySession(session_id="s")
    await sess.add_run_state(_snap("r1"))
    await sess.add_run_state(_snap("r2"))
    await sess.add_run_state(_snap("r3"))
    out = await sess.get_run_states()
    assert [s.run_id for s in out] == ["r1", "r2", "r3"]


async def test_get_run_states_limit_keeps_most_recent() -> None:
    sess = InMemorySession(session_id="s")
    for i in range(5):
        await sess.add_run_state(_snap(f"r{i}"))
    out = await sess.get_run_states(limit=2)
    assert [s.run_id for s in out] == ["r3", "r4"]


async def test_get_run_states_limit_zero_returns_empty() -> None:
    sess = InMemorySession(session_id="s")
    await sess.add_run_state(_snap("r1"))
    assert await sess.get_run_states(limit=0) == []


async def test_get_run_states_negative_limit_raises() -> None:
    sess = InMemorySession(session_id="s")
    with pytest.raises(ValueError, match="non-negative"):
        await sess.get_run_states(limit=-1)


async def test_get_run_states_returns_copy_not_internal_list() -> None:
    sess = InMemorySession(session_id="s")
    await sess.add_run_state(_snap("r1"))
    snap_list = await sess.get_run_states()
    snap_list.append(_snap("INJECTED"))
    again = await sess.get_run_states()
    assert [s.run_id for s in again] == ["r1"]


# --- clear -------------------------------------------------------------------


async def test_clear_drops_messages_and_snapshots() -> None:
    """SS2: a single ``clear()`` empties both halves of the session."""

    sess = InMemorySession(session_id="s")
    await sess.add_messages([_msg("a")])
    await sess.add_run_state(_snap("r1"))
    await sess.clear()
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
    assert await sess.get_latest_run_state() is None


# --- concurrency ------------------------------------------------------------


async def test_concurrent_writers_do_not_lose_entries() -> None:
    """Many parallel writers from a single event loop must all land.

    The lock serialises append-extend pairs so no message is dropped even
    when ``add_messages`` is awaited concurrently from many tasks.
    """

    sess = InMemorySession(session_id="s")

    async def writer(label: str, n: int) -> None:
        for i in range(n):
            await sess.add_messages([_msg(f"{label}-{i}")])

    await asyncio.gather(writer("a", 10), writer("b", 10), writer("c", 10))
    msgs = await sess.get_messages()
    assert len(msgs) == 30
    texts = {m.text for m in msgs}
    assert "a-0" in texts and "b-9" in texts and "c-5" in texts


async def test_concurrent_snapshot_writers_do_not_lose_entries() -> None:
    sess = InMemorySession(session_id="s")

    async def writer(prefix: str, n: int) -> None:
        for i in range(n):
            await sess.add_run_state(_snap(f"{prefix}-{i}"))

    await asyncio.gather(writer("a", 8), writer("b", 8))
    snaps = await sess.get_run_states()
    assert len(snaps) == 16
