"""Unit tests for ``agent_harness.core.graph`` — the ``pydantic_graph`` adapter.

These tests pin the minimal surface Wave 4's Loop will consume, plus a
2-node toy graph round-trip to prove the adapter actually drives an engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent_harness.core import graph as graph_mod
from agent_harness.core.graph import (
    End,
    FullPersistence,
    Graph,
    GraphCtx,
    GraphRun,
    InMemoryPersistence,
    Node,
    StatePersistence,
    build_graph,
    iter_graph,
    run_graph,
)

# --- Surface assertions ------------------------------------------------------


def test_public_surface_is_minimal_and_named() -> None:
    """``__all__`` is the contract — Wave 4 imports nothing else from here.

    Adding to this list is a deliberate widening; this test fails loudly
    when someone exports a new symbol without thinking.
    """

    assert set(graph_mod.__all__) == {
        "End",
        "FullPersistence",
        "Graph",
        "GraphCtx",
        "GraphRun",
        "GraphRunResult",
        "InMemoryPersistence",
        "Node",
        "StatePersistence",
        "build_graph",
        "iter_graph",
        "run_graph",
    }


def test_node_is_aliased_to_pydantic_graph_basenode() -> None:
    """``Node`` is the upstream ``BaseNode`` so user subclassing works
    unchanged. We deliberately do not wrap it — the spec calls for a thin
    re-export."""

    from pydantic_graph import BaseNode

    assert Node is BaseNode


def test_graph_ctx_is_aliased_to_graphruncontext() -> None:
    from pydantic_graph import GraphRunContext

    assert GraphCtx is GraphRunContext


def test_state_persistence_aliases_resolve() -> None:
    from pydantic_graph.persistence import BaseStatePersistence
    from pydantic_graph.persistence.in_mem import (
        FullStatePersistence,
        SimpleStatePersistence,
    )

    assert StatePersistence is BaseStatePersistence
    assert InMemoryPersistence is SimpleStatePersistence
    assert FullPersistence is FullStatePersistence


# --- 2-node toy graph: round-trip end-to-end --------------------------------


@dataclass
class _State:
    """Toy state — counts increments; the 2-node graph exits at 2."""

    n: int = 0
    visited: list[str] = field(default_factory=list)


@dataclass
class _Inc(Node[_State, None, int]):
    """Increment node — always transitions to ``_Check``."""

    async def run(self, ctx: GraphCtx[_State, None]) -> _Check:
        ctx.state.n += 1
        ctx.state.visited.append("inc")
        return _Check()


@dataclass
class _Check(Node[_State, None, int]):
    """Terminal-or-loop node — returns ``End`` once state reaches 2."""

    async def run(self, ctx: GraphCtx[_State, None]) -> _Inc | End[int]:
        ctx.state.visited.append("check")
        if ctx.state.n >= 2:
            return End(ctx.state.n)
        return _Inc()


def test_build_graph_returns_typed_graph() -> None:
    g = build_graph([_Inc, _Check], name="toy")
    assert isinstance(g, Graph)
    assert g.name == "toy"
    # Node membership is what matters; order doesn't.
    assert set(g.get_nodes()) == {_Inc, _Check}


async def test_2_node_round_trip_to_end() -> None:
    """Happy path: graph loops until ``End``; ``state`` mutations and the
    return value are both observed."""

    g = build_graph([_Inc, _Check])
    state = _State()
    result = await run_graph(g, _Inc(), state=state)

    assert result.output == 2
    assert state.n == 2
    # Two full Inc->Check cycles before End — proves edges work.
    assert state.visited == ["inc", "check", "inc", "check"]


async def test_iter_yields_each_node_then_end() -> None:
    """``iter_graph`` yields each executed node — powers ``Agent.iter``."""

    g = build_graph([_Inc, _Check])
    seen: list[type] = []
    async with iter_graph(g, _Inc(), state=_State()) as run:
        assert isinstance(run, GraphRun)
        async for step in run:
            seen.append(type(step))

    # Four step transitions then End: Inc, Check, Inc, Check, End
    assert seen == [_Inc, _Check, _Inc, _Check, End]


async def test_persistence_records_snapshots_at_node_boundaries() -> None:
    """The persistence hook captures node-boundary state — what Wave 4's
    ``Session.add_run_state`` will hang off of (LP4 / RS1)."""

    g = build_graph([_Inc, _Check])
    persistence = FullPersistence()
    persistence.set_graph_types(g)
    await run_graph(g, _Inc(), state=_State(), persistence=persistence)

    history = await persistence.load_all()
    # 2x Inc + 2x Check + 1 End == 5 snapshots.
    assert len(history) == 5
    # First and last snapshots are predictable.
    assert type(history[0].node).__name__ == "_Inc"
    # The terminal snapshot is an EndSnapshot — class name suffices here.
    assert "End" in type(history[-1]).__name__


# --- Error path -------------------------------------------------------------


async def test_run_graph_propagates_node_errors() -> None:
    """A node that raises bubbles out unchanged — the adapter doesn't
    swallow exceptions (the loop owns recovery via ``Error`` events)."""

    @dataclass
    class _Boom(Node[None, None, int]):
        async def run(self, ctx: GraphCtx[None, None]) -> End[int]:
            raise RuntimeError("boom")

    g = build_graph([_Boom])
    with pytest.raises(RuntimeError, match="boom"):
        await run_graph(g, _Boom())


def test_build_graph_rejects_inconsistent_edges() -> None:
    """A node that names a successor missing from ``nodes`` is a wiring
    bug; the upstream engine raises ``GraphSetupError`` and we forward it
    rather than masking it (the loop owns nothing here — bad wiring is a
    developer error, not a runtime concern)."""

    from pydantic_graph import GraphSetupError

    # ``_Refs`` returns ``_Inc`` (defined at module scope so the forward
    # ref resolves) but we deliberately leave ``_Inc`` out of ``nodes``.
    @dataclass
    class _Refs(Node[_State, None, int]):
        async def run(self, ctx: GraphCtx[_State, None]) -> _Inc:
            return _Inc()

    with pytest.raises(GraphSetupError, match="not included in the graph"):
        build_graph([_Refs])
