"""Unit tests for ``agent_harness.sessions.redis.RedisSession``.

The ``redis`` package is an optional dependency. These tests exercise the
backend against a minimal in-process fake that mimics the subset of the
``redis.asyncio.Redis`` API the session uses (``lrange`` / ``rpush`` /
``delete``). This keeps the unit tests dependency-free while still
covering the real call sequence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from agent_harness.core.errors import NotSupportedError
from agent_harness.core.models import Message, TextBlock, Usage
from agent_harness.core.run_state import RunStateSnapshot
from agent_harness.sessions.redis import RedisSession


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _msg(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)], timestamp=_ts())


def _snap(run_id: str = "r1") -> RunStateSnapshot:
    return RunStateSnapshot(
        run_id=run_id,
        agent_name="demo",
        current_node="ModelRequest",
        usage=Usage(),
        created_at=_ts(),
    )


class _FakeAsyncRedis:
    """Minimal stand-in for ``redis.asyncio.Redis``.

    Stores list values keyed by str. Returns ``bytes`` from ``lrange``
    to match the default ``decode_responses=False`` behaviour, so the
    session's decode helper is exercised.
    """

    def __init__(self, *, decode_responses: bool = False) -> None:
        self._store: dict[str, list[Any]] = {}
        self._decode = decode_responses

    async def rpush(self, key: str, *values: Any) -> int:
        self._store.setdefault(key, []).extend(values)
        return len(self._store[key])

    async def lrange(self, key: str, start: int, end: int) -> list[Any]:
        values = self._store.get(key, [])
        # Redis end is inclusive.
        sliced = values[start : (end + 1) if end != -1 else None]
        if end == -1:
            sliced = values[start:]
        if self._decode:
            return [v if isinstance(v, str) else v.decode("utf-8") for v in sliced]
        return [v.encode("utf-8") if isinstance(v, str) else v for v in sliced]

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                n += 1
        return n


# --- construction errors ----------------------------------------------------


def test_missing_client_raises_not_supported() -> None:
    """No client passed -> NotSupportedError (typed framework error)."""

    with pytest.raises(NotSupportedError):
        RedisSession(session_id="s")


# --- happy-path against the fake -------------------------------------------


async def test_session_id_is_exposed() -> None:
    sess = RedisSession(session_id="abc", client=_FakeAsyncRedis())
    assert sess.session_id == "abc"


async def test_initial_state_is_empty() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
    assert await sess.get_latest_run_state() is None


async def test_add_and_get_messages_preserves_order() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    await sess.add_messages([_msg("a"), _msg("b")])
    await sess.add_messages([_msg("c")])
    out = await sess.get_messages()
    assert [m.text for m in out] == ["a", "b", "c"]


async def test_add_messages_empty_list_is_noop() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    await sess.add_messages([])
    assert await sess.get_messages() == []


async def test_add_run_state_and_get_latest() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    await sess.add_run_state(_snap("r1"))
    await sess.add_run_state(_snap("r2"))
    latest = await sess.get_latest_run_state()
    assert latest is not None
    assert latest.run_id == "r2"


async def test_get_run_states_returns_oldest_first() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    for r in ("r1", "r2", "r3"):
        await sess.add_run_state(_snap(r))
    out = await sess.get_run_states()
    assert [s.run_id for s in out] == ["r1", "r2", "r3"]


async def test_get_run_states_limit_keeps_most_recent() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    for i in range(5):
        await sess.add_run_state(_snap(f"r{i}"))
    out = await sess.get_run_states(limit=2)
    assert [s.run_id for s in out] == ["r3", "r4"]


async def test_get_run_states_limit_zero_returns_empty() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    await sess.add_run_state(_snap("r1"))
    assert await sess.get_run_states(limit=0) == []


async def test_get_run_states_negative_limit_raises() -> None:
    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    with pytest.raises(ValueError, match="non-negative"):
        await sess.get_run_states(limit=-1)


async def test_clear_drops_messages_and_snapshots() -> None:
    """SS2: a single ``clear()`` empties both halves of the session."""

    sess = RedisSession(session_id="s", client=_FakeAsyncRedis())
    await sess.add_messages([_msg("a")])
    await sess.add_run_state(_snap("r1"))
    await sess.clear()
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
    assert await sess.get_latest_run_state() is None


async def test_decode_responses_true_path() -> None:
    """Sanity-check the str-payload branch of ``_decode``."""

    sess = RedisSession(session_id="s", client=_FakeAsyncRedis(decode_responses=True))
    await sess.add_messages([_msg("a")])
    out = await sess.get_messages()
    assert out[0].text == "a"


async def test_sessions_are_isolated_by_session_id() -> None:
    shared = _FakeAsyncRedis()
    a = RedisSession(session_id="A", client=shared)
    b = RedisSession(session_id="B", client=shared)
    await a.add_messages([_msg("alpha")])
    await b.add_messages([_msg("beta")])
    assert [m.text for m in await a.get_messages()] == ["alpha"]
    assert [m.text for m in await b.get_messages()] == ["beta"]
