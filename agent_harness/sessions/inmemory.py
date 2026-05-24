"""In-memory :class:`Session` backend (SS3, SS4).

For tests + ephemeral runs. Holds the message history and any stored
:class:`RunStateSnapshot`s in a single backend (SS2) and serialises mutating
access through an :class:`asyncio.Lock` so multiple subscribers / tools may
read and write concurrently without losing entries.

Example:
    >>> import asyncio
    >>> from datetime import UTC, datetime
    >>> from agent_harness.core.models import Message, TextBlock
    >>> sess = InMemorySession(session_id="s1")
    >>> msg = Message(
    ...     role="user", content=[TextBlock(text="hi")], timestamp=datetime(2026, 1, 1, tzinfo=UTC)
    ... )
    >>> asyncio.run(sess.add_messages([msg]))
    >>> asyncio.run(sess.get_messages())[0].text
    'hi'
"""

from __future__ import annotations

import asyncio

from agent_harness.core.memory import Session
from agent_harness.core.models import Message
from agent_harness.core.run_state import RunStateSnapshot


class InMemorySession(Session):
    """Process-local :class:`Session` implementation.

    Both messages and snapshots are kept in plain Python lists; a single
    :class:`asyncio.Lock` guards all mutation. Reads return *copies* so a
    caller iterating the result can't observe a later append.

    Example:
        >>> import asyncio
        >>> sess = InMemorySession(session_id="demo")
        >>> sess.session_id
        'demo'
        >>> asyncio.run(sess.get_latest_run_state()) is None
        True
    """

    session_id: str

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._messages: list[Message] = []
        self._snapshots: list[RunStateSnapshot] = []
        self._lock = asyncio.Lock()

    async def get_messages(self) -> list[Message]:
        async with self._lock:
            return list(self._messages)

    async def add_messages(self, msgs: list[Message]) -> None:
        async with self._lock:
            self._messages.extend(msgs)

    async def clear(self) -> None:
        async with self._lock:
            self._messages.clear()
            self._snapshots.clear()

    async def get_run_states(self, *, limit: int | None = None) -> list[RunStateSnapshot]:
        async with self._lock:
            if limit is None:
                return list(self._snapshots)
            if limit < 0:
                raise ValueError("limit must be non-negative")
            if limit == 0:
                return []
            return list(self._snapshots[-limit:])

    async def add_run_state(self, snap: RunStateSnapshot) -> None:
        async with self._lock:
            self._snapshots.append(snap)

    async def get_latest_run_state(self) -> RunStateSnapshot | None:
        async with self._lock:
            return self._snapshots[-1] if self._snapshots else None
