"""Redis :class:`Session` backend (SS3).

Multi-process / distributed deployments where one Redis instance is the
shared source of truth for both the conversation's message history and
its paused-run snapshots (SS1, SS2). The ``redis`` package is an
*optional* dependency (``pip install agent-harness[redis]``); when not
installed, instantiation raises :class:`NotSupportedError` rather than
``ImportError`` so the caller sees a typed framework error.

Keys (per ``session_id``):

* ``agent_harness:session:<sid>:messages``    — Redis list, head = oldest.
* ``agent_harness:session:<sid>:run_states``  — Redis list, head = oldest.

Both use ``RPUSH`` for append and ``LRANGE`` for reads so order is
preserved and the operations are atomic on the server.

Example:
    >>> from agent_harness.sessions.redis import RedisSession
    >>> # In practice, pass a `redis.asyncio.Redis` client:
    >>> # sess = RedisSession(session_id="s1", client=Redis.from_url(...))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_harness.core.errors import NotSupportedError
from agent_harness.core.memory import Session
from agent_harness.core.models import Message
from agent_harness.core.run_state import RunStateSnapshot

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


_KEY_PREFIX = "agent_harness:session"


def _messages_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:messages"


def _run_states_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:run_states"


class RedisSession(Session):
    """Async Redis-backed :class:`Session` implementation.

    Pass an already-configured ``redis.asyncio.Redis`` client; the
    session owns no connection pool of its own so callers can share a
    pool across many sessions.

    Example:
        >>> # async client is required:
        >>> # from redis.asyncio import Redis
        >>> # client = Redis.from_url("redis://localhost:6379/0",
        >>> #                         decode_responses=True)
        >>> # sess = RedisSession(session_id="s1", client=client)
    """

    session_id: str

    def __init__(self, session_id: str, client: Any | None = None) -> None:
        self.session_id = session_id
        if client is None:
            # Validate the optional dependency at construction time so a
            # missing install is reported with a typed error rather than
            # a cryptic AttributeError later.
            try:
                import redis.asyncio as _redis_async  # noqa: F401
            except ImportError as exc:
                raise NotSupportedError(
                    "redis is not installed; install with "
                    "`pip install agent-harness[redis]` or pass a "
                    "`redis.asyncio.Redis` client explicitly",
                    cause=exc,
                ) from exc
            raise NotSupportedError(
                "RedisSession requires an async redis client; pass "
                "`client=Redis.from_url(...)` from `redis.asyncio`"
            )
        self._client = client
        self._messages_key = _messages_key(session_id)
        self._run_states_key = _run_states_key(session_id)

    # --- messages -----------------------------------------------------------

    async def get_messages(self) -> list[Message]:
        payloads = await self._client.lrange(self._messages_key, 0, -1)
        return [Message.model_validate_json(_decode(p)) for p in payloads]

    async def add_messages(self, msgs: list[Message]) -> None:
        if not msgs:
            return
        payloads = [m.model_dump_json() for m in msgs]
        await self._client.rpush(self._messages_key, *payloads)

    # --- snapshots ----------------------------------------------------------

    async def get_run_states(self, *, limit: int | None = None) -> list[RunStateSnapshot]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        if limit is None:
            payloads = await self._client.lrange(self._run_states_key, 0, -1)
        else:
            # Tail ``limit`` items in oldest-first order.
            payloads = await self._client.lrange(self._run_states_key, -limit, -1)
        return [RunStateSnapshot.from_json(_decode(p)) for p in payloads]

    async def add_run_state(self, snap: RunStateSnapshot) -> None:
        await self._client.rpush(self._run_states_key, snap.to_json())

    async def get_latest_run_state(self) -> RunStateSnapshot | None:
        payloads = await self._client.lrange(self._run_states_key, -1, -1)
        if not payloads:
            return None
        return RunStateSnapshot.from_json(_decode(payloads[-1]))

    # --- clear --------------------------------------------------------------

    async def clear(self) -> None:
        await self._client.delete(self._messages_key, self._run_states_key)


def _decode(payload: Any) -> str:
    """Accept either ``str`` (decode_responses=True) or ``bytes``."""
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)
