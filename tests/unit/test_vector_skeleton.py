"""Unit tests for :mod:`agent_harness.long_term.vector` — skeleton only.

The vector backend is an unimplemented skeleton in v1 (open-questions #2).
These tests pin the public surface so the v0.0.2 implementation can't
silently break the contract.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from agent_harness.core.errors import NotSupportedError
from agent_harness.core.memory import LongTermMemory
from agent_harness.long_term import vector as vector_mod
from agent_harness.long_term.vector import VectorLongTermMemory

_HAS_PGVECTOR = importlib.util.find_spec("pgvector") is not None


def test_module_imports_without_pgvector() -> None:
    """The skeleton must import cleanly even without the optional dep."""
    assert vector_mod.__doc__ is not None
    assert "SKELETON" in vector_mod.__doc__
    assert "TODO(v0.0.2)" in Path(vector_mod.__file__).read_text(encoding="utf-8")


def test_class_exposed() -> None:
    assert inspect.isclass(VectorLongTermMemory)


def test_constructor_signature_matches_spec() -> None:
    """Constructor takes ``dsn``, optional ``table`` and ``embedding_dim``."""
    sig = inspect.signature(VectorLongTermMemory)
    params = sig.parameters
    assert "dsn" in params
    assert "table" in params
    assert "embedding_dim" in params


@pytest.mark.skipif(_HAS_PGVECTOR, reason="pgvector is installed; the absent path can't fire")
def test_constructor_raises_not_supported_without_pgvector() -> None:
    with pytest.raises(NotSupportedError, match="pgvector"):
        VectorLongTermMemory(dsn="postgres://localhost/x")


@pytest.mark.skipif(not _HAS_PGVECTOR, reason="pgvector not installed in this venv")
def test_instance_satisfies_protocol_when_pgvector_present() -> None:
    inst = VectorLongTermMemory(dsn="postgres://localhost/x")
    assert isinstance(inst, LongTermMemory)


@pytest.mark.skipif(not _HAS_PGVECTOR, reason="pgvector not installed in this venv")
async def test_methods_raise_not_implemented_v002() -> None:
    inst = VectorLongTermMemory(dsn="postgres://localhost/x")
    with pytest.raises(NotImplementedError, match=r"v0\.0\.2"):
        await inst.remember("x")
    with pytest.raises(NotImplementedError, match=r"v0\.0\.2"):
        await inst.recall("x")
    with pytest.raises(NotImplementedError, match=r"v0\.0\.2"):
        await inst.forget("x")
    with pytest.raises(NotImplementedError, match=r"v0\.0\.2"):
        await inst.list_memories()


def test_class_is_a_runtime_protocol_subtype() -> None:
    """Even without instantiation, the class shape must match the Protocol.

    We construct a structural attribute check rather than instantiating, so
    this assertion holds whether or not pgvector is installed.
    """
    required = {"remember", "recall", "forget", "list_memories"}
    for name in required:
        attr = getattr(VectorLongTermMemory, name, None)
        assert attr is not None and inspect.iscoroutinefunction(attr)


def test_todo_marker_present_in_source() -> None:
    src = Path(vector_mod.__file__).read_text(encoding="utf-8")
    assert "TODO(v0.0.2)" in src


if __name__ == "__main__":  # pragma: no cover - manual invocation
    pytest.main([__file__, "-v"])
