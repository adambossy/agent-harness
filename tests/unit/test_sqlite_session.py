"""Unit tests for ``agent_harness.sessions.sqlite.SqliteSession``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_harness.core.models import Message, TextBlock, Usage
from agent_harness.core.run_state import RunStateSnapshot
from agent_harness.sessions.sqlite import SqliteSession


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


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "sessions.db"


# --- session_id + initial state ---------------------------------------------


async def test_session_id_is_exposed(db_path: Path) -> None:
    sess = SqliteSession(session_id="abc", path=db_path)
    assert sess.session_id == "abc"
    await sess.close()


async def test_initial_state_is_empty(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
    assert await sess.get_latest_run_state() is None
    await sess.close()


# --- messages ---------------------------------------------------------------


async def test_add_and_get_messages_preserves_order(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    await sess.add_messages([_msg("a"), _msg("b")])
    await sess.add_messages([_msg("c")])
    out = await sess.get_messages()
    assert [m.text for m in out] == ["a", "b", "c"]
    await sess.close()


async def test_add_messages_empty_list_is_noop(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    await sess.add_messages([])
    assert await sess.get_messages() == []
    await sess.close()


async def test_messages_persist_across_instances(db_path: Path) -> None:
    sess1 = SqliteSession(session_id="s", path=db_path)
    await sess1.add_messages([_msg("hello")])
    await sess1.close()

    sess2 = SqliteSession(session_id="s", path=db_path)
    out = await sess2.get_messages()
    assert [m.text for m in out] == ["hello"]
    await sess2.close()


async def test_sessions_are_isolated_by_session_id(db_path: Path) -> None:
    a = SqliteSession(session_id="A", path=db_path)
    b = SqliteSession(session_id="B", path=db_path)
    await a.add_messages([_msg("alpha")])
    await b.add_messages([_msg("beta")])
    assert [m.text for m in await a.get_messages()] == ["alpha"]
    assert [m.text for m in await b.get_messages()] == ["beta"]
    await a.close()
    await b.close()


# --- snapshots --------------------------------------------------------------


async def test_add_run_state_and_get_latest(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    await sess.add_run_state(_snap("r1"))
    await sess.add_run_state(_snap("r2"))
    latest = await sess.get_latest_run_state()
    assert latest is not None
    assert latest.run_id == "r2"
    await sess.close()


async def test_get_run_states_returns_oldest_first(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    await sess.add_run_state(_snap("r1"))
    await sess.add_run_state(_snap("r2"))
    await sess.add_run_state(_snap("r3"))
    out = await sess.get_run_states()
    assert [s.run_id for s in out] == ["r1", "r2", "r3"]
    await sess.close()


async def test_get_run_states_limit_keeps_most_recent(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    for i in range(5):
        await sess.add_run_state(_snap(f"r{i}"))
    out = await sess.get_run_states(limit=2)
    assert [s.run_id for s in out] == ["r3", "r4"]
    await sess.close()


async def test_get_run_states_limit_zero_returns_empty(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    await sess.add_run_state(_snap("r1"))
    assert await sess.get_run_states(limit=0) == []
    await sess.close()


async def test_get_run_states_negative_limit_raises(db_path: Path) -> None:
    sess = SqliteSession(session_id="s", path=db_path)
    with pytest.raises(ValueError, match="non-negative"):
        await sess.get_run_states(limit=-1)
    await sess.close()


# --- clear -------------------------------------------------------------------


async def test_clear_drops_messages_and_snapshots(db_path: Path) -> None:
    """SS2: a single ``clear()`` empties both halves of the session."""

    sess = SqliteSession(session_id="s", path=db_path)
    await sess.add_messages([_msg("a")])
    await sess.add_run_state(_snap("r1"))
    await sess.clear()
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
    assert await sess.get_latest_run_state() is None
    await sess.close()


async def test_clear_does_not_affect_other_sessions(db_path: Path) -> None:
    a = SqliteSession(session_id="A", path=db_path)
    b = SqliteSession(session_id="B", path=db_path)
    await a.add_messages([_msg("alpha")])
    await b.add_messages([_msg("beta")])
    await a.clear()
    assert await a.get_messages() == []
    assert [m.text for m in await b.get_messages()] == ["beta"]
    await a.close()
    await b.close()


# --- concurrency ------------------------------------------------------------


async def test_concurrent_writers_do_not_lose_entries(db_path: Path) -> None:
    """Many parallel writers must all land — the per-session lock serialises
    the read-max-then-insert sequence so ord values stay unique."""

    sess = SqliteSession(session_id="s", path=db_path)

    async def writer(label: str, n: int) -> None:
        for i in range(n):
            await sess.add_messages([_msg(f"{label}-{i}")])

    await asyncio.gather(writer("a", 10), writer("b", 10), writer("c", 10))
    msgs = await sess.get_messages()
    assert len(msgs) == 30
    texts = {m.text for m in msgs}
    assert "a-0" in texts and "b-9" in texts and "c-5" in texts
    await sess.close()
