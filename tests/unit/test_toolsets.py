"""Unit tests for ``agent_harness.core.toolsets``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_harness.core.errors import ToolError
from agent_harness.core.models import TextBlock
from agent_harness.core.tools import Tool, ToolCall, ToolPolicy, ToolResult, tool
from agent_harness.core.toolsets import (
    ApprovalRequiredToolset,
    CachedToolset,
    FilteredToolset,
    PrefixedToolset,
    StaticToolset,
    Toolset,
)


@tool
async def _read(path: str) -> str:
    """Read a file.

    Args:
        path: filesystem path.
    """
    return f"contents of {path}"


@tool
async def _write(path: str, contents: str) -> None:
    """Write contents to a file.

    Args:
        path: target path.
        contents: data to write.
    """


@tool
async def _boom() -> str:
    """Always raises."""
    raise RuntimeError("kaboom")


def test_static_toolset_satisfies_protocol() -> None:
    ts = StaticToolset(name="t", tools=[_read])
    assert isinstance(ts, Toolset)


async def test_static_toolset_lists_enabled_tools_only() -> None:
    disabled = ToolPolicy(is_enabled=False)
    hidden_tool = Tool(
        name="hidden",
        description="",
        schema={},
        policy=disabled,
        fn=lambda: None,
    )
    ts = StaticToolset(name="t", tools=[_read, hidden_tool])

    listed = await ts.list_tools(ctx=None)
    assert [t.name for t in listed] == ["_read"]


async def test_static_toolset_lists_with_predicate() -> None:
    """A callable ``is_enabled`` is invoked with the run ``ctx``."""

    policy = ToolPolicy(is_enabled=lambda ctx: bool(ctx and ctx.get("admin")))
    gated = Tool(
        name="gated",
        description="",
        schema={},
        policy=policy,
        fn=lambda: None,
    )
    ts = StaticToolset(name="t", tools=[gated])

    assert await ts.list_tools(ctx={"admin": False}) == []
    listed = await ts.list_tools(ctx={"admin": True})
    assert [t.name for t in listed] == ["gated"]


async def test_static_toolset_dispatches_tool_call() -> None:
    ts = StaticToolset(name="t", tools=[_read])
    result = await ts.call_tool(
        ctx=None, call=ToolCall(id="c1", name="_read", arguments={"path": "x"})
    )
    assert isinstance(result, ToolResult)
    assert result.error is None
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "contents of x"


async def test_static_toolset_call_unknown_tool_raises() -> None:
    ts = StaticToolset(name="t", tools=[_read])
    with pytest.raises(ToolError, match="no tool named"):
        await ts.call_tool(ctx=None, call=ToolCall(id="c1", name="missing", arguments={}))


async def test_call_tool_wraps_exception_as_errored_result() -> None:
    ts = StaticToolset(name="t", tools=[_boom])
    result = await ts.call_tool(ctx=None, call=ToolCall(id="c1", name="_boom"))
    assert result.error is not None
    assert "kaboom" in result.error


async def test_call_tool_uses_failure_error_function_when_set() -> None:
    def formatter(exc: Exception) -> str:
        return f"sanitised: {exc.__class__.__name__}"

    @tool(policy=ToolPolicy(failure_error_function=formatter))
    async def bad() -> str:
        """Bad."""
        raise ValueError("secret")

    ts = StaticToolset(name="t", tools=[bad])
    result = await ts.call_tool(ctx=None, call=ToolCall(id="c1", name="bad"))
    assert result.error == "sanitised: ValueError"


async def test_call_tool_forwards_existing_tool_result_unchanged() -> None:
    """A function that already returns a ``ToolResult`` is passed through."""

    @tool
    async def emit() -> ToolResult:
        """Emit a structured result."""
        return ToolResult(content=[TextBlock(text="custom")], metadata={"k": 1})

    ts = StaticToolset(name="t", tools=[emit])
    result = await ts.call_tool(ctx=None, call=ToolCall(id="c1", name="emit"))
    assert result.metadata == {"k": 1}
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "custom"


async def test_prefixed_toolset_prefixes_names_and_strips_on_dispatch() -> None:
    inner = StaticToolset(name="fs", tools=[_read])
    prefixed = PrefixedToolset(inner=inner, prefix="gh")

    listed = await prefixed.list_tools(ctx=None)
    assert [t.name for t in listed] == ["gh___read"]
    result = await prefixed.call_tool(
        ctx=None, call=ToolCall(id="c1", name="gh___read", arguments={"path": "y"})
    )
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "contents of y"


async def test_prefixed_toolset_call_with_unprefixed_name_raises() -> None:
    inner = StaticToolset(name="fs", tools=[_read])
    prefixed = PrefixedToolset(inner=inner, prefix="gh")
    with pytest.raises(ToolError, match="is not prefixed"):
        await prefixed.call_tool(ctx=None, call=ToolCall(id="c1", name="_read"))


async def test_filtered_toolset_drops_tools_failing_predicate() -> None:
    inner = StaticToolset(name="fs", tools=[_read, _write])
    only_read = FilteredToolset(inner=inner, predicate=lambda t, _ctx: t.name == "_read")
    listed = await only_read.list_tools(ctx=None)
    assert [t.name for t in listed] == ["_read"]
    # Dispatch still works for the included tool.
    res = await only_read.call_tool(
        ctx=None, call=ToolCall(id="c1", name="_read", arguments={"path": "p"})
    )
    block = res.content[0]
    assert isinstance(block, TextBlock)


async def test_approval_required_toolset_forces_needs_approval() -> None:
    inner = StaticToolset(name="fs", tools=[_read])
    guarded = ApprovalRequiredToolset(inner=inner)
    [t] = await guarded.list_tools(ctx=None)
    assert t.policy.needs_approval is True
    # Other fields are preserved.
    assert t.policy.is_destructive is False


async def test_cached_toolset_caches_listings_within_ttl() -> None:
    calls: list[int] = []

    class Counting:
        name = "c"

        async def list_tools(self, ctx: Any) -> list[Tool]:
            calls.append(1)
            return [_read]

        async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
            return ToolResult(content=[])

    cached = CachedToolset(inner=Counting(), ttl_seconds=60.0)
    await cached.list_tools(ctx=None)
    await cached.list_tools(ctx=None)
    await cached.list_tools(ctx=None)
    assert sum(calls) == 1
    cached.invalidate()
    await cached.list_tools(ctx=None)
    assert sum(calls) == 2


async def test_cached_toolset_returns_copy_so_caller_cannot_mutate_cache() -> None:
    inner = StaticToolset(name="fs", tools=[_read])
    cached = CachedToolset(inner=inner, ttl_seconds=60.0)
    first = await cached.list_tools(ctx=None)
    first.clear()
    second = await cached.list_tools(ctx=None)
    assert [t.name for t in second] == ["_read"]


async def test_wrappers_compose() -> None:
    """Filter → approve → cache → prefix stacks cleanly."""

    inner = StaticToolset(name="fs", tools=[_read, _write])
    filtered = FilteredToolset(inner=inner, predicate=lambda t, _: t.name == "_read")
    guarded = ApprovalRequiredToolset(inner=filtered)
    cached = CachedToolset(inner=guarded, ttl_seconds=60.0)
    prefixed = PrefixedToolset(inner=cached, prefix="ns")

    listed = await prefixed.list_tools(ctx=None)
    assert [t.name for t in listed] == ["ns___read"]
    assert listed[0].policy.needs_approval is True


def test_read_tool_function_is_coroutine() -> None:
    """Sanity check: the underlying ``fn`` for ``_read`` is an async function."""
    assert asyncio.iscoroutinefunction(_read.fn)
