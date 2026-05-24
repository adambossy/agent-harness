"""Thin adapter around ``pydantic_graph`` (per open-questions #1).

This module is the *boundary*: nothing else in ``agent_harness`` imports
``pydantic_graph`` directly. Wave 4's ``Loop`` (PrepareTurn / ModelRequest /
ToolDispatch / DecideNext) subclasses :class:`Node` here and calls
:func:`run_graph`; if the upstream package changes shape (or we swap
engines), the blast radius is contained to this file.

Vendored surface — only what the loop actually needs:

* :data:`Node` — alias for :class:`pydantic_graph.BaseNode` (the loop's
  4 typed nodes subclass this; their ``run`` return-type annotation is
  the edge set the engine reads).
* :data:`End` — terminal marker; a node returns ``End(result)`` to stop.
* :data:`GraphCtx` — alias for :class:`pydantic_graph.GraphRunContext`;
  loop nodes receive one of these as their ``ctx`` argument.
* :class:`Graph` / :class:`GraphRun` — the engine + iterable run handle.
* :func:`build_graph` — convenience constructor; pass the node *classes*
  (not instances) and get a typed :class:`Graph`.
* :func:`run_graph` — single-call helper that runs a graph to completion.
* :func:`iter_graph` — async iterator over each node as it executes
  (powers ``Agent.iter`` in Wave 4).
* :data:`StatePersistence` / :data:`InMemoryPersistence` — re-exports of
  the upstream persistence Protocol + default implementation, kept here
  so Wave 4 can snapshot at every node boundary without touching the
  vendored package directly.

Example:
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Tick(Node[int, None, int]):
    ...     async def run(self, ctx: GraphCtx[int, None]) -> "End[int]":
    ...         return End(ctx.state + 1)
    >>> g = build_graph([Tick])
    >>> g.name is None or isinstance(g.name, str)
    True
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

# Importing the BaseNode-based ``Graph`` engine emits a deprecation warning
# in ``pydantic_graph>=2`` (the project is moving to a builder-based API).
# We pin to the BaseNode pattern intentionally — open-questions #1 chose it
# for the typed-node-edges-from-return-types property the spec relies on —
# so we suppress *this* warning at import time. If/when we migrate, this
# block is the single edit point.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    # ``PydanticGraphDeprecationWarning`` is the precise class; importing
    # it conditionally keeps us decoupled from upstream's warning module
    # layout.
    try:
        from pydantic_graph import PydanticGraphDeprecationWarning

        warnings.simplefilter("ignore", category=PydanticGraphDeprecationWarning)
    except ImportError:  # pragma: no cover - older pydantic_graph
        pass

    from pydantic_graph import (
        BaseNode,
        End,
        Graph,
        GraphRun,
        GraphRunContext,
        GraphRunResult,
    )
    from pydantic_graph.persistence import BaseStatePersistence
    from pydantic_graph.persistence.in_mem import (
        FullStatePersistence,
        SimpleStatePersistence,
    )

# --- Re-exports / aliases ----------------------------------------------------

Node = BaseNode
"""Loop-node base class. Subclasses implement ``async def run(self, ctx)``;
the return-type annotation declares which nodes may follow."""

GraphCtx = GraphRunContext
"""Per-call context passed to :meth:`Node.run`. Exposes ``state`` and
``deps``."""

StatePersistence = BaseStatePersistence
"""Protocol implemented by every persistence backend (memory, sqlite, ...)."""

InMemoryPersistence = SimpleStatePersistence
"""Default ``StatePersistence`` — keeps only the latest snapshot in memory.
Wave 4 swaps this for ``FullStatePersistence`` (keeps every node-boundary
snapshot) when ``Session`` is configured."""

FullPersistence = FullStatePersistence
"""``StatePersistence`` that retains every node-boundary snapshot — used to
power ``agent.iter`` replay."""


# --- Typed builders ----------------------------------------------------------


def build_graph(
    nodes: Sequence[type[BaseNode[Any, Any, Any]]],
    *,
    name: str | None = None,
) -> Graph[Any, Any, Any]:
    """Construct a :class:`Graph` from the node *classes* of a loop.

    The engine reads each node's ``run`` return type to determine the edge
    set, so order in ``nodes`` is irrelevant — only membership matters.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Done(Node[None, None, str]):
        ...     async def run(self, ctx: GraphCtx[None, None]) -> "End[str]":
        ...         return End("done")
        >>> g = build_graph([Done], name="demo")
        >>> g.name
        'demo'
    """
    return Graph(nodes=tuple(nodes), name=name)


async def run_graph[StateT, DepsT, OutT](
    graph: Graph[StateT, DepsT, OutT],
    start: BaseNode[StateT, DepsT, OutT],
    *,
    state: StateT | None = None,
    deps: DepsT | None = None,
    persistence: BaseStatePersistence[StateT, OutT] | None = None,
) -> GraphRunResult[StateT, OutT]:
    """Run ``graph`` from ``start`` to an :class:`End` node.

    Returns the upstream :class:`GraphRunResult`; Wave 4 wraps that into
    the loop's ``RunResult``. The persistence handle, if supplied, holds
    every snapshot taken at each node boundary.

    ``state`` and ``deps`` are typed ``_StateT | None`` / ``_DepsT | None``
    in *our* signature (callers may have a graph whose state type already
    permits ``None``); the upstream signature uses ``StateT = None`` as a
    runtime default, and the cast below tells mypy not to worry. The
    runtime behaviour is identical.

    Example:
        >>> # Used by ``Agent.run`` in Wave 4 — see ``core/loop.py``.
        >>> callable(run_graph)
        True
    """
    return await graph.run(
        start,
        state=cast(StateT, state),
        deps=cast(DepsT, deps),
        persistence=persistence,
    )


def iter_graph[StateT, DepsT, OutT](
    graph: Graph[StateT, DepsT, OutT],
    start: BaseNode[StateT, DepsT, OutT],
    *,
    state: StateT | None = None,
    deps: DepsT | None = None,
    persistence: BaseStatePersistence[StateT, OutT] | None = None,
) -> AbstractAsyncContextManager[GraphRun[StateT, DepsT, OutT]]:
    """Async context manager yielding a :class:`GraphRun` over each node.

    Used as ``async with iter_graph(...) as run: async for node in run: ...``.
    Each iteration yields the node that *just executed* (or :class:`End`
    once the graph terminates). Wave 4 wires this into ``Agent.iter``
    (AG2: inspect-or-replace each step).

    Example:
        >>> callable(iter_graph)
        True
    """
    return graph.iter(
        start,
        state=cast(StateT, state),
        deps=cast(DepsT, deps),
        persistence=persistence,
    )


__all__ = [
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
]
