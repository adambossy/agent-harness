"""Unit tests for ``agent_harness.core.memory`` (Protocols + ``Memory``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_harness.core.memory import LongTermMemory, Memory, Session
from agent_harness.core.models import Message, TextBlock
from agent_harness.core.run_state import RunStateSnapshot


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# --- Memory dataclass --------------------------------------------------------


def test_memory_dataclass_defaults_and_fields() -> None:
    m = Memory(id="m1", content="hello", created_at=_ts())
    assert m.id == "m1"
    assert m.content == "hello"
    assert m.metadata == {}
    assert m.created_at == _ts()
    assert m.score is None


def test_memory_dataclass_carries_score_and_metadata() -> None:
    m = Memory(
        id="m1",
        content="prefers tabs",
        metadata={"topic": "style"},
        created_at=_ts(),
        score=0.42,
    )
    assert m.metadata == {"topic": "style"}
    assert m.created_at == _ts()
    assert m.score == 0.42


# --- Session Protocol --------------------------------------------------------


class _GoodSession:
    """Structural Session implementation used for the positive case."""

    session_id: str = "s"

    async def get_messages(self) -> list[Message]:
        return []

    async def add_messages(self, msgs: list[Message]) -> None:
        return None

    async def clear(self) -> None:
        return None

    async def get_run_states(self, *, limit: int | None = None) -> list[RunStateSnapshot]:
        return []

    async def add_run_state(self, snap: RunStateSnapshot) -> None:
        return None

    async def get_latest_run_state(self) -> RunStateSnapshot | None:
        return None


class _BadSession:
    """Missing ``get_run_states`` — should NOT satisfy Session."""

    session_id: str = "s"

    async def get_messages(self) -> list[Message]:
        return []

    async def add_messages(self, msgs: list[Message]) -> None:
        return None

    async def clear(self) -> None:
        return None

    async def add_run_state(self, snap: RunStateSnapshot) -> None:
        return None

    async def get_latest_run_state(self) -> RunStateSnapshot | None:
        return None


def test_session_protocol_accepts_full_shape() -> None:
    assert isinstance(_GoodSession(), Session)


def test_session_protocol_rejects_incomplete_shape() -> None:
    # Structural check: missing ``get_run_states`` => not a Session.
    assert not isinstance(_BadSession(), Session)


def test_session_protocol_runtime_checkable_for_inmemory_impl() -> None:
    """The shipping InMemorySession structurally satisfies the Protocol."""

    from agent_harness.sessions.inmemory import InMemorySession

    assert isinstance(InMemorySession(session_id="x"), Session)


# --- LongTermMemory Protocol -------------------------------------------------


class _GoodLTM:
    async def remember(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "m1"

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        return []

    async def forget(self, memory_id: str) -> None:
        return None

    async def list_memories(self, *, limit: int = 100) -> list[Memory]:
        return []


class _BadLTM:
    """Missing ``forget`` — should NOT satisfy LongTermMemory."""

    async def remember(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "m1"

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        return []

    async def list_memories(self, *, limit: int = 100) -> list[Memory]:
        return []


def test_long_term_memory_protocol_accepts_full_shape() -> None:
    assert isinstance(_GoodLTM(), LongTermMemory)


def test_long_term_memory_protocol_rejects_incomplete_shape() -> None:
    assert not isinstance(_BadLTM(), LongTermMemory)


async def test_long_term_memory_round_trip_through_stub() -> None:
    """Sanity-check the Protocol is invokable end-to-end via a stub."""

    class _Stub:
        def __init__(self) -> None:
            self.store: dict[str, Memory] = {}

        async def remember(
            self,
            content: str,
            *,
            key: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> str:
            mid = key or f"m{len(self.store)}"
            self.store[mid] = Memory(
                id=mid,
                content=content,
                metadata=metadata or {},
                created_at=_ts(),
            )
            return mid

        async def recall(
            self,
            query: str,
            *,
            limit: int = 5,
            filter: dict[str, Any] | None = None,
        ) -> list[Memory]:
            return [m for m in self.store.values() if query in m.content][:limit]

        async def forget(self, memory_id: str) -> None:
            self.store.pop(memory_id, None)

        async def list_memories(self, *, limit: int = 100) -> list[Memory]:
            return list(self.store.values())[:limit]

    ltm: LongTermMemory = _Stub()
    mid = await ltm.remember("user prefers tabs", key="prefs")
    assert mid == "prefs"
    hits = await ltm.recall("tabs")
    assert len(hits) == 1 and hits[0].id == "prefs"
    await ltm.forget("prefs")
    assert await ltm.list_memories() == []


# --- Cross-protocol sanity: an InMemorySession also stores RunStateSnapshots


async def test_session_protocol_messages_and_snapshots_share_backend() -> None:
    """SS2: messages and snapshots live in the same backend; clearing
    drops both."""

    from agent_harness.core.models import Usage
    from agent_harness.sessions.inmemory import InMemorySession

    sess = InMemorySession(session_id="s1")
    msg = Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())
    snap = RunStateSnapshot(
        run_id="r1",
        agent_name="demo",
        current_node="ModelRequest",
        usage=Usage(),
        created_at=_ts(),
    )
    await sess.add_messages([msg])
    await sess.add_run_state(snap)
    assert len(await sess.get_messages()) == 1
    assert (await sess.get_latest_run_state()) is not None

    await sess.clear()
    assert await sess.get_messages() == []
    assert await sess.get_run_states() == []
