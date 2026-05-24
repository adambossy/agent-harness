"""Toolsets — composable groups of tools (Layer 1).

A :class:`Toolset` is a uniform interface for *any* collection of tools: a
fixed list (``StaticToolset``), an MCP server (Wave 2), a filtered view of
another set, an approval-gated wrapper, and so on. Everything composes by
wrapping. The loop never special-cases tool kinds — it just calls
``list_tools`` and ``call_tool``.

This module also ships :data:`TOOL_SEARCH` — the built-in ``ToolSearch``
standard tool that backs ``defer_loading`` (see S17 and the spec's "Deferred
tool loading" section). Given a query and an optional ``max_results``, it
returns JSON schemas for any *deferred* tool in the current run whose name
or description matches.

Example:
    >>> from agent_harness.core.tools import tool
    >>> @tool
    ... async def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> ts = StaticToolset(name="math", tools=[add])
    >>> ts.name
    'math'
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .errors import ToolError
from .models import TextBlock
from .tools import Tool, ToolCall, ToolPolicy, ToolResult, tool

# ---------------------------------------------------------------------------
# Toolset Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Toolset(Protocol):
    """A group of tools. Wrappers implement this same Protocol so everything
    composes.

    ``ctx`` is the Layer-3 ``RunContext`` (typed ``Any`` because it doesn't
    exist yet); subclasses may ignore it, or use it to gate visibility /
    dispatch dynamically (TS7).

    Example:
        >>> isinstance(StaticToolset(name="t", tools=[]), Toolset)
        True
    """

    name: str

    async def list_tools(self, ctx: Any) -> list[Tool]:
        """Return every tool this set offers for the current run."""
        ...

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        """Dispatch ``call`` to one of this set's tools."""
        ...


# ---------------------------------------------------------------------------
# Helpers shared by every concrete toolset.
# ---------------------------------------------------------------------------


def _find_tool(tools: list[Tool], name: str) -> Tool:
    """Return the tool named ``name`` from ``tools`` or raise ``ToolError``.

    Example:
        >>> from agent_harness.core.tools import Tool, ToolPolicy
        >>> async def f() -> None: ...
        >>> t = Tool(name="f", description="", schema={}, policy=ToolPolicy(), fn=f)
        >>> _find_tool([t], "f").name
        'f'
    """
    for t in tools:
        if t.name == name:
            return t
    raise ToolError(
        f"no tool named {name!r} in this toolset",
        context={"requested": name, "available": [t.name for t in tools]},
    )


async def _invoke(tool_obj: Tool, arguments: dict[str, Any]) -> ToolResult:
    """Invoke ``tool_obj.fn`` and wrap its return value as a :class:`ToolResult`.

    Sync functions are called directly; async functions are awaited. Bodies
    that already return a ``ToolResult`` are forwarded unchanged. Any other
    return value is wrapped in a single :class:`TextBlock` via ``str(...)``.
    Raised exceptions become an errored :class:`ToolResult` (the policy-level
    ``failure_error_function`` shapes the visible message).
    """
    fn = tool_obj.fn
    try:
        raw: Any = fn(**arguments)
        if isinstance(raw, Awaitable):
            raw = await raw
    except Exception as exc:
        formatter = tool_obj.policy.failure_error_function
        message = formatter(exc) if formatter is not None else f"{type(exc).__name__}: {exc}"
        return ToolResult(content=[TextBlock(text=message)], error=message)
    if isinstance(raw, ToolResult):
        return raw
    text = "" if raw is None else str(raw)
    return ToolResult(content=[TextBlock(text=text)])


# ---------------------------------------------------------------------------
# StaticToolset — the default.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StaticToolset:
    """Fixed list of ``@tool``-decorated functions. The 90% case.

    Tools whose ``policy.is_enabled`` is ``False`` (or whose callable
    predicate returns ``False`` for the current ``ctx``) are filtered out of
    :meth:`list_tools` per TS7. They remain dispatchable through
    :meth:`call_tool` so an in-flight call doesn't vanish mid-turn if the
    policy flips.

    Example:
        >>> from agent_harness.core.tools import tool
        >>> @tool
        ... async def ping() -> str:
        ...     '''No-op.'''
        ...     return "pong"
        >>> StaticToolset(name="t", tools=[ping]).name
        't'
    """

    name: str
    tools: list[Tool] = field(default_factory=list)

    async def list_tools(self, ctx: Any) -> list[Tool]:
        return [t for t in self.tools if _resolve_predicate(t.policy.is_enabled, ctx)]

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        del ctx  # static dispatch doesn't read ctx (subclasses may).
        target = _find_tool(self.tools, call.name)
        return await _invoke(target, call.arguments)


def _resolve_predicate(value: bool | Callable[..., bool], ctx: Any) -> bool:
    """Return ``value`` if it's a bool, else call it with ``ctx``.

    ``ToolPolicy`` predicates accept ``Callable[..., bool]`` because Layer 0
    can't pin the exact signature (RunContext lives in Layer 3).
    """
    if callable(value):
        try:
            return bool(value(ctx))
        except TypeError:
            # User passed a no-arg predicate; honor that.
            return bool(value())
    return bool(value)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PrefixedToolset:
    """Wrap another toolset; prepend ``f"{prefix}__"`` to every tool name.

    Useful for namespacing MCP servers (``"github__search_repos"``) so two
    sets with overlapping tool names can coexist on one agent.

    Example:
        >>> from agent_harness.core.tools import tool
        >>> @tool
        ... async def ping() -> str:
        ...     '''ping'''
        ...     return "pong"
        >>> inner = StaticToolset(name="x", tools=[ping])
        >>> p = PrefixedToolset(inner=inner, prefix="gh")
        >>> p.name
        'gh'
    """

    inner: Toolset
    prefix: str
    separator: str = "__"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        # ``name`` is the prefix itself — it's how the loop labels the group.
        self.name = self.prefix

    def _full(self, tool_name: str) -> str:
        return f"{self.prefix}{self.separator}{tool_name}"

    def _strip(self, full_name: str) -> str:
        head = f"{self.prefix}{self.separator}"
        if not full_name.startswith(head):
            raise ToolError(
                f"tool {full_name!r} is not prefixed with {head!r}",
                context={"name": full_name, "prefix": head},
            )
        return full_name[len(head) :]

    async def list_tools(self, ctx: Any) -> list[Tool]:
        return [
            Tool(
                name=self._full(t.name),
                description=t.description,
                schema=t.schema,
                policy=t.policy,
                fn=t.fn,
            )
            for t in await self.inner.list_tools(ctx)
        ]

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        inner_call = ToolCall(id=call.id, name=self._strip(call.name), arguments=call.arguments)
        return await self.inner.call_tool(ctx, inner_call)


@dataclass(slots=True)
class FilteredToolset:
    """Wrap another toolset; expose only tools satisfying ``predicate``.

    The predicate is re-evaluated on every ``list_tools`` call so it can
    depend on ``ctx`` (TS7). Calls to filtered-out tools still dispatch
    successfully — the filter is for *visibility*, not *security*; for
    access control wrap with :class:`ApprovalRequiredToolset` or use a
    sandbox policy.

    Example:
        >>> from agent_harness.core.tools import tool
        >>> @tool
        ... async def read() -> str:
        ...     '''read'''
        ...     return ""
        >>> @tool
        ... async def write(s: str) -> None:
        ...     '''write'''
        >>> inner = StaticToolset(name="fs", tools=[read, write])
        >>> safe = FilteredToolset(inner=inner, predicate=lambda t, _ctx: t.name == "read")
        >>> safe.name
        'fs'
    """

    inner: Toolset
    predicate: Callable[[Tool, Any], bool]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.inner.name

    async def list_tools(self, ctx: Any) -> list[Tool]:
        return [t for t in await self.inner.list_tools(ctx) if self.predicate(t, ctx)]

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        return await self.inner.call_tool(ctx, call)


@dataclass(slots=True)
class ApprovalRequiredToolset:
    """Wrap another toolset; force ``policy.needs_approval=True`` on every tool.

    Other policy fields (timeouts, guardrails, ...) are preserved. Useful to
    layer human-in-the-loop approval on top of an otherwise-unguarded set
    (e.g. an MCP server you don't fully trust).

    Example:
        >>> from agent_harness.core.tools import tool
        >>> @tool
        ... async def rm(path: str) -> None:
        ...     '''rm'''
        >>> guarded = ApprovalRequiredToolset(inner=StaticToolset(name="fs", tools=[rm]))
        >>> guarded.name
        'fs'
    """

    inner: Toolset
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.inner.name

    async def list_tools(self, ctx: Any) -> list[Tool]:
        return [
            Tool(
                name=t.name,
                description=t.description,
                schema=t.schema,
                policy=t.policy.model_copy(update={"needs_approval": True}),
                fn=t.fn,
            )
            for t in await self.inner.list_tools(ctx)
        ]

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        return await self.inner.call_tool(ctx, call)


@dataclass(slots=True)
class CachedToolset:
    """Wrap another toolset; cache ``list_tools`` for ``ttl_seconds`` seconds.

    Useful for remote toolsets (e.g. MCP) whose tool catalog rarely changes
    but is expensive to fetch. ``call_tool`` is always delegated unchanged
    (caching results is the loop's job, not the toolset's).

    Example:
        >>> from agent_harness.core.tools import tool
        >>> @tool
        ... async def noop() -> None:
        ...     '''noop'''
        >>> c = CachedToolset(inner=StaticToolset(name="t", tools=[noop]), ttl_seconds=60)
        >>> c.name
        't'
    """

    inner: Toolset
    ttl_seconds: float = 60.0
    name: str = field(init=False)
    _cache: list[Tool] | None = field(default=None, init=False, repr=False)
    _cached_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = self.inner.name

    async def list_tools(self, ctx: Any) -> list[Tool]:
        now = time.monotonic()
        if self._cache is None or now - self._cached_at >= self.ttl_seconds:
            self._cache = await self.inner.list_tools(ctx)
            self._cached_at = now
        return list(self._cache)

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        return await self.inner.call_tool(ctx, call)

    def invalidate(self) -> None:
        """Drop the cached listing — call when the upstream catalog changes
        (e.g. MCP ``list_changed``)."""
        self._cache = None


# ---------------------------------------------------------------------------
# Built-in ToolSearch standard tool.
# ---------------------------------------------------------------------------


_DEFERRED_TOOLS: list[Tool] = []
"""Process-wide registry of deferred tools the ``ToolSearch`` standard tool
may surface. Populated by :func:`register_deferred_tools`. Wave-3 will move
this onto ``RunContext`` so it's per-run rather than process-wide; this
module-level fallback keeps the unit tests self-contained.
"""


def register_deferred_tools(tools: list[Tool]) -> None:
    """Register tools as candidates for ``ToolSearch``.

    Idempotent. Callers (Wave-3 loop) will pass the deferred subset of all
    available tools for a given run; the module-level registry is a
    placeholder until ``RunContext`` exists.
    """
    seen = {t.name for t in _DEFERRED_TOOLS}
    for t in tools:
        if t.name not in seen:
            _DEFERRED_TOOLS.append(t)
            seen.add(t.name)


def clear_deferred_tools() -> None:
    """Empty the deferred-tools registry (tests use this for isolation)."""
    _DEFERRED_TOOLS.clear()


def _score(t: Tool, query: str) -> int:
    """Cheap relevance score: name-substring (3), word-boundary in
    description (2), substring elsewhere (1), miss (0). Higher is better."""
    q = query.lower()
    if not q:
        return 1  # match-everything-equally when query is empty
    name = t.name.lower()
    desc = t.description.lower()
    if q in name:
        return 3
    if re.search(rf"\b{re.escape(q)}\b", desc):
        return 2
    if q in desc:
        return 1
    return 0


@tool(
    name="ToolSearch",
    description=(
        "Fetch full JSON schemas for deferred tools matching a query. "
        "The returned schemas become callable in the next turn."
    ),
    policy=ToolPolicy(is_read_only=True, is_concurrency_safe=True, always_load=True),
)
async def tool_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Look up deferred tools by name / description.

    Args:
        query: Free-text query matched against deferred tool names and
            descriptions. An empty string returns the first ``max_results``
            deferred tools in registration order.
        max_results: Cap on returned schemas. Defaults to 5.

    Returns:
        A list of ``{name, description, schema}`` dicts, highest-scoring
        match first. Empty if no deferred tools match.
    """
    scored = [(t, _score(t, query)) for t in _DEFERRED_TOOLS]
    hits = [(t, s) for (t, s) in scored if s > 0]
    hits.sort(key=lambda pair: (-pair[1], pair[0].name))
    limit = max(0, max_results)
    return [
        {"name": t.name, "description": t.description, "schema": t.schema}
        for (t, _) in hits[:limit]
    ]


TOOL_SEARCH: Tool = tool_search
"""The built-in ``ToolSearch`` :class:`Tool`. Always-load by policy; the loop
surfaces this whenever any tool in the current run carries
``policy.defer_loading=True``."""
