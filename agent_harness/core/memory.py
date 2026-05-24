"""Short-term and long-term memory Protocols (Layer 0).

Two Protocols and one record type:

- :class:`Session` — *short-term* memory; pluggable per-conversation backend
  that stores BOTH the message history AND any :class:`RunStateSnapshot`
  produced at a node boundary. A single backend keeps the two halves atomic
  (SS2): pause/resume never has to reconcile across stores.
- :class:`LongTermMemory` — *cross-session* memory; pluggable backend
  surfaced to the model via optional ``remember``/``recall`` tools.
- :class:`Memory` — the record :meth:`LongTermMemory.recall` /
  :meth:`LongTermMemory.list_memories` return.

Concrete adapters live under ``agent_harness.sessions`` /
``agent_harness.long_term``; core knows only the Protocol.

Example:
    >>> import asyncio
    >>> from agent_harness.sessions.inmemory import InMemorySession
    >>> sess = InMemorySession(session_id="s1")
    >>> asyncio.run(sess.get_messages())
    []
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .models import Message
    from .run_state import RunStateSnapshot


# --- Short-term: Session ----------------------------------------------------


@runtime_checkable
class Session(Protocol):
    """Short-term, per-conversation backend.

    Stores BOTH the message history AND any paused-run snapshots so the two
    halves stay atomic (SS1, SS2). Backends are intentionally small (SS6):
    list / append / clear messages, plus add / list / latest for snapshots.
    Compaction is *not* a Session concern (SS5).

    Example:
        >>> import asyncio
        >>> from agent_harness.sessions.inmemory import InMemorySession
        >>> sess = InMemorySession(session_id="demo")
        >>> sess.session_id
        'demo'
        >>> asyncio.run(sess.get_messages())
        []
    """

    session_id: str

    async def get_messages(self) -> list[Message]:
        """Return the full ordered message history."""
        ...

    async def add_messages(self, msgs: list[Message]) -> None:
        """Append ``msgs`` in order to the message history."""
        ...

    async def clear(self) -> None:
        """Drop every message and every stored snapshot."""
        ...

    async def get_run_states(self, *, limit: int | None = None) -> list[RunStateSnapshot]:
        """Return stored snapshots oldest-first; ``limit`` keeps only the
        most recent ``limit`` entries (still oldest-first)."""
        ...

    async def add_run_state(self, snap: RunStateSnapshot) -> None:
        """Append a snapshot to this session's history."""
        ...

    async def get_latest_run_state(self) -> RunStateSnapshot | None:
        """Return the most-recently appended snapshot, or ``None`` if none
        exists yet."""
        ...


# --- Long-term: cross-session memory ----------------------------------------


@dataclass(slots=True)
class Memory:
    """A single long-term memory record.

    ``score`` is set by :meth:`LongTermMemory.recall` when ranking; it is
    ``None`` for unranked listings (``list_memories``).

    Example:
        >>> from datetime import UTC, datetime
        >>> Memory(
        ...     id="m1",
        ...     content="prefers tabs",
        ...     metadata={"topic": "style"},
        ...     created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ... ).id
        'm1'
    """

    id: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@runtime_checkable
class LongTermMemory(Protocol):
    """Cross-session memory backend.

    Surfaced to the model via optional ``remember`` / ``recall`` built-in
    tools (LT2). The default in-tree backend is ``MemdirLongTermMemory``
    (Wave 3); the vector backend is an optional skeleton (LT7). Cross-agent
    sharing, automatic consolidation, and decay are out of scope for v1
    (LT8).

    Example:
        >>> import asyncio
        >>> from datetime import UTC, datetime
        >>> class _StubLTM:
        ...     async def remember(self, content, *, key=None, metadata=None):
        ...         return "m1"
        ...
        ...     async def recall(self, query, *, limit=5, filter=None):
        ...         return []
        ...
        ...     async def forget(self, memory_id):
        ...         return None
        ...
        ...     async def list_memories(self, *, limit=100):
        ...         return []
        >>> isinstance(_StubLTM(), LongTermMemory)
        True
    """

    async def remember(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store ``content``; return an opaque ``memory_id`` (LT1)."""
        ...

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        """Retrieve up to ``limit`` records matching ``query`` (LT1)."""
        ...

    async def forget(self, memory_id: str) -> None:
        """Drop the record identified by ``memory_id`` (LT1)."""
        ...

    async def list_memories(self, *, limit: int = 100) -> list[Memory]:
        """Return up to ``limit`` stored records (LT1)."""
        ...
