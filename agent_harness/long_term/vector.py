"""Vector-backed :class:`LongTermMemory` — **v1 SKELETON** (LT7).

Per open-questions decision #2, semantic-vector recall is reserved as an
unimplemented skeleton in v1. The eventual implementation will target
**pgvector** (not chromadb) to avoid the heavy
``sentence-transformers + chromadb`` install. The shape lives here so
downstream code can refer to ``VectorLongTermMemory`` without changing
imports once the body lands.

# TODO(v0.0.2): implement the four LongTermMemory methods against
# pgvector. The recommended schema is a single ``memories`` table with
# ``id text primary key``, ``content text``, ``metadata jsonb``,
# ``embedding vector(N)`` and ``created_at timestamptz``; recall is an
# ANN search via the ``<->`` operator.

Example:
    >>> import asyncio
    >>> from agent_harness.core.memory import LongTermMemory
    >>> try:
    ...     v = VectorLongTermMemory(dsn="postgres://localhost/agent_harness")
    ...     isinstance(v, LongTermMemory)
    ... except Exception:
    ...     True  # pgvector may be absent in the dev venv
    True
"""

from __future__ import annotations

from typing import Any

from agent_harness.core.errors import NotSupportedError
from agent_harness.core.memory import LongTermMemory, Memory


class VectorLongTermMemory(LongTermMemory):
    """Vector-backed long-term memory — **skeleton only in v1**.

    Construction raises :class:`NotSupportedError` if the optional
    ``pgvector`` dependency is absent; otherwise the instance is built
    but each method raises :class:`NotImplementedError` pointing at
    ``# TODO(v0.0.2)``. This keeps the type usable in ``isinstance``
    checks and import-graph wiring without claiming functionality the
    backend doesn't yet have.

    Example:
        >>> try:
        ...     VectorLongTermMemory(dsn="postgres://localhost/x")
        ... except (NotSupportedError, NotImplementedError):
        ...     True
        True
    """

    def __init__(
        self,
        *,
        dsn: str,
        table: str = "memories",
        embedding_dim: int = 1536,
    ) -> None:
        # Lazy import so the skeleton can be referenced without the
        # optional dependency installed.
        try:
            import pgvector  # noqa: F401  # pragma: no cover - exercised only when installed
        except ImportError as exc:  # pragma: no cover - dev venv path
            raise NotSupportedError(
                "VectorLongTermMemory requires the 'vector' extra "
                "(`pip install agent-harness[vector]`); install pgvector "
                "to use this backend. See open-questions.md #2.",
                cause=exc,
            ) from exc
        self.dsn = dsn
        self.table = table
        self.embedding_dim = embedding_dim

    async def remember(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError(
            "VectorLongTermMemory.remember is a v1 skeleton; "
            "see # TODO(v0.0.2) in agent_harness/long_term/vector.py"
        )

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        raise NotImplementedError(
            "VectorLongTermMemory.recall is a v1 skeleton; "
            "see # TODO(v0.0.2) in agent_harness/long_term/vector.py"
        )

    async def forget(self, memory_id: str) -> None:
        raise NotImplementedError(
            "VectorLongTermMemory.forget is a v1 skeleton; "
            "see # TODO(v0.0.2) in agent_harness/long_term/vector.py"
        )

    async def list_memories(self, *, limit: int = 100) -> list[Memory]:
        raise NotImplementedError(
            "VectorLongTermMemory.list_memories is a v1 skeleton; "
            "see # TODO(v0.0.2) in agent_harness/long_term/vector.py"
        )
