"""Unit tests for :mod:`agent_harness.core.mcp` — the MCPServer Toolset.

These tests mock the underlying :mod:`mcp` SDK session so they exercise
the harness adapter logic without requiring a real MCP server. The
optional ``mcp`` dependency is installed in the dev venv, but the
harness still treats it as lazy-imported.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_harness.core.errors import NotSupportedError, ToolError
from agent_harness.core.mcp import (
    MCPServer,
    MCPServerHTTP,
    MCPServerSSE,
    MCPServerStdio,
    _convert_mcp_content,
    _mcp_tool_to_harness,
    _require_mcp,
)
from agent_harness.core.models import ImageBlock, TextBlock
from agent_harness.core.tools import ToolCall, ToolPolicy, ToolResult
from agent_harness.core.toolsets import (
    ApprovalRequiredToolset,
    CachedToolset,
    PrefixedToolset,
    Toolset,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _raw_tool(name: str, desc: str = "", schema: dict[str, Any] | None = None) -> SimpleNamespace:
    """Build an MCP-shaped raw Tool struct (matches ``mcp.types.Tool``)."""
    return SimpleNamespace(
        name=name,
        description=desc,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _raw_text_content(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _raw_image_content(data: str, mime: str = "image/png") -> SimpleNamespace:
    return SimpleNamespace(type="image", data=data, mimeType=mime)


class _FakeListTools:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools


class _FakeCallResult:
    def __init__(
        self,
        content: list[Any],
        *,
        is_error: bool = False,
        structured: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.isError = is_error
        self.structuredContent = structured


class _FakeSession:
    """Replacement for ``mcp.ClientSession`` — async-context-manager + RPC."""

    def __init__(
        self,
        tools: list[Any] | None = None,
        call_handler: Any | None = None,
        resources: list[Any] | None = None,
    ) -> None:
        self._tools = tools or []
        self._call_handler = call_handler
        self._resources = resources or []
        self.initialize_calls = 0
        self.list_tool_calls = 0
        self.tool_call_log: list[tuple[str, dict[str, Any] | None]] = []
        self.message_handler: Any = None

    async def initialize(self) -> _FakeListTools:
        self.initialize_calls += 1
        return _FakeListTools(self._tools)

    async def list_tools(self) -> _FakeListTools:
        self.list_tool_calls += 1
        return _FakeListTools(self._tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        self.tool_call_log.append((name, arguments))
        if self._call_handler is not None:
            return await self._call_handler(name, arguments)
        return _FakeCallResult([_raw_text_content("ok")])

    async def list_resources(self) -> SimpleNamespace:
        return SimpleNamespace(resources=self._resources)

    async def read_resource(self, uri: Any) -> SimpleNamespace:
        return SimpleNamespace(uri=uri)


def _patch_session_factory(monkeypatch: pytest.MonkeyPatch, fake: _FakeSession) -> None:
    """Patch ``mcp.ClientSession`` to return ``fake`` as an async-cm.

    ``ClientSession(read, write, message_handler=...)`` is used as an async
    context manager. We replace the class with a factory that captures the
    handler on ``fake`` and yields ``fake`` itself.
    """

    @asynccontextmanager
    async def _session_cm(*args: Any, **kwargs: Any) -> Any:
        fake.message_handler = kwargs.get("message_handler")
        yield fake

    class _SessionFactory:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            fake.message_handler = kwargs.get("message_handler")
            self._cm: Any = _session_cm(*args, **kwargs)

        async def __aenter__(self) -> _FakeSession:
            session: _FakeSession = await self._cm.__aenter__()
            return session

        async def __aexit__(
            self,
            exc_type: Any = None,
            exc: Any = None,
            tb: Any = None,
        ) -> None:
            await self._cm.__aexit__(exc_type, exc, tb)

    import mcp

    monkeypatch.setattr(mcp, "ClientSession", _SessionFactory)


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    cls: type[MCPServer],
    streams: tuple[Any, Any] = (object(), object()),
) -> None:
    """Replace ``cls._open_streams`` with an async-cm yielding stub streams."""

    @asynccontextmanager
    async def _streams_cm() -> Any:
        yield streams

    def _open_streams(self: Any) -> Any:
        return _streams_cm()

    monkeypatch.setattr(cls, "_open_streams", _open_streams)


# ---------------------------------------------------------------------------
# Lazy import + protocol shape
# ---------------------------------------------------------------------------


def test_require_mcp_returns_module() -> None:
    mod = _require_mcp()
    assert hasattr(mod, "ClientSession")


def test_require_mcp_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)  # ImportError on next import
    with pytest.raises(NotSupportedError, match="not installed"):
        _require_mcp()


def test_mcpserver_stdio_satisfies_toolset_protocol() -> None:
    srv = MCPServerStdio("local", command=["python"])
    assert isinstance(srv, Toolset)
    assert srv.name == "local"


def test_mcpserver_http_satisfies_toolset_protocol() -> None:
    srv = MCPServerHTTP("remote", url="https://example/mcp")
    assert isinstance(srv, Toolset)


def test_mcpserver_sse_satisfies_toolset_protocol() -> None:
    srv = MCPServerSSE("legacy", url="https://example/sse")
    assert isinstance(srv, Toolset)


# ---------------------------------------------------------------------------
# Content-block translation
# ---------------------------------------------------------------------------


def test_convert_mcp_content_text_passthrough() -> None:
    blocks = _convert_mcp_content([_raw_text_content("hi")])
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "hi"


def test_convert_mcp_content_image_passthrough() -> None:
    blocks = _convert_mcp_content([_raw_image_content("aGVsbG8=", mime="image/png")])
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].mime_type == "image/png"


def test_convert_mcp_content_unknown_falls_back_to_text() -> None:
    weird = SimpleNamespace(type="resource_link", uri="x")
    blocks = _convert_mcp_content([weird])
    assert isinstance(blocks[0], TextBlock)
    assert "resource_link" in blocks[0].text


def test_mcp_tool_to_harness_preserves_name_and_schema() -> None:
    raw = _raw_tool("search", desc="search the web", schema={"type": "object"})
    t = _mcp_tool_to_harness(raw)
    assert t.name == "search"
    assert t.description == "search the web"
    assert t.schema == {"type": "object"}
    assert isinstance(t.policy, ToolPolicy)


async def test_mcp_tool_to_harness_unreachable_fn_raises() -> None:
    """The placeholder ``fn`` must not be silently invokable — dispatch
    *must* route through ``MCPServer.call_tool``."""
    t = _mcp_tool_to_harness(_raw_tool("x"))
    with pytest.raises(ToolError, match="must be dispatched"):
        await t.fn()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_connect_initializes_session_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(tools=[_raw_tool("t1")])
    _patch_session_factory(monkeypatch, fake)
    srv = MCPServerStdio("s", command=["py"])
    _patch_transport(monkeypatch, MCPServerStdio)
    await srv.connect()
    await srv.connect()  # idempotent
    assert fake.initialize_calls == 1
    await srv.disconnect()


async def test_disconnect_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession()
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    await srv.disconnect()  # never connected: no-op
    await srv.connect()
    await srv.disconnect()
    await srv.disconnect()  # idempotent


async def test_async_context_manager_connects_and_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession()
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    async with srv as s2:
        assert s2 is srv
        assert srv._connected is True
    assert srv._connected is False


# ---------------------------------------------------------------------------
# list_tools + caching
# ---------------------------------------------------------------------------


async def test_list_tools_returns_translated_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession(tools=[_raw_tool("a"), _raw_tool("b")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    tools = await srv.list_tools(ctx=None)
    assert sorted(t.name for t in tools) == ["a", "b"]
    await srv.disconnect()


async def test_list_tools_caches_results(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession(tools=[_raw_tool("a")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    await srv.list_tools(ctx=None)
    await srv.list_tools(ctx=None)
    await srv.list_tools(ctx=None)
    assert fake.list_tool_calls == 1
    await srv.disconnect()


async def test_list_tools_returns_copy_so_caller_cannot_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(tools=[_raw_tool("a")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    first = await srv.list_tools(ctx=None)
    first.clear()
    second = await srv.list_tools(ctx=None)
    assert [t.name for t in second] == ["a"]
    await srv.disconnect()


async def test_list_tools_auto_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession(tools=[_raw_tool("a")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    pre: bool = srv._connected
    assert pre is False
    await srv.list_tools(ctx=None)
    post: bool = srv._connected
    assert post is True
    await srv.disconnect()


async def test_list_changed_notification_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(tools=[_raw_tool("a")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    await srv.list_tools(ctx=None)
    assert fake.list_tool_calls == 1

    # Simulate the server pushing a ``tools/list_changed`` notification.
    notif = SimpleNamespace(method="notifications/tools/list_changed")
    # The session_factory captured the handler on ``fake``.
    handler = fake.message_handler
    assert handler is not None
    await handler(SimpleNamespace(root=notif))

    await srv.list_tools(ctx=None)
    assert fake.list_tool_calls == 2
    await srv.disconnect()


async def test_unrelated_notification_does_not_invalidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(tools=[_raw_tool("a")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    await srv.list_tools(ctx=None)
    notif = SimpleNamespace(method="notifications/message")
    handler = fake.message_handler
    assert handler is not None
    await handler(SimpleNamespace(root=notif))
    await srv.list_tools(ctx=None)
    assert fake.list_tool_calls == 1
    await srv.disconnect()


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


async def test_call_tool_returns_text_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession(tools=[_raw_tool("echo")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    res = await srv.call_tool(ctx=None, call=ToolCall(id="c1", name="echo", arguments={"x": 1}))
    assert isinstance(res, ToolResult)
    assert res.error is None
    block = res.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "ok"
    assert fake.tool_call_log == [("echo", {"x": 1})]
    await srv.disconnect()


async def test_call_tool_with_no_arguments_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession()
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    await srv.call_tool(ctx=None, call=ToolCall(id="c1", name="echo"))
    assert fake.tool_call_log == [("echo", None)]
    await srv.disconnect()


async def test_call_tool_is_error_surfaces_in_result(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(_name: str, _args: Any) -> _FakeCallResult:
        return _FakeCallResult([_raw_text_content("rate limited")], is_error=True)

    fake = _FakeSession(call_handler=handler)
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    res = await srv.call_tool(ctx=None, call=ToolCall(id="c1", name="x"))
    assert res.error == "rate limited"
    await srv.disconnect()


async def test_call_tool_is_error_with_no_text_synthesizes_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(_name: str, _args: Any) -> _FakeCallResult:
        return _FakeCallResult([], is_error=True)

    fake = _FakeSession(call_handler=handler)
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    res = await srv.call_tool(ctx=None, call=ToolCall(id="c1", name="x"))
    assert res.error is not None
    assert "isError" in res.error
    await srv.disconnect()


async def test_call_tool_exception_becomes_errored_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(_name: str, _args: Any) -> _FakeCallResult:
        raise RuntimeError("boom")

    fake = _FakeSession(call_handler=handler)
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    res = await srv.call_tool(ctx=None, call=ToolCall(id="c1", name="x"))
    assert res.error is not None
    assert "boom" in res.error
    await srv.disconnect()


async def test_call_tool_structured_content_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(_name: str, _args: Any) -> _FakeCallResult:
        return _FakeCallResult([_raw_text_content("ok")], structured={"k": 1})

    fake = _FakeSession(call_handler=handler)
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    res = await srv.call_tool(ctx=None, call=ToolCall(id="c1", name="x"))
    assert res.metadata.get("structured") == {"k": 1}
    await srv.disconnect()


# ---------------------------------------------------------------------------
# Composition with wrappers (the core "MCPServer IS a Toolset" payoff)
# ---------------------------------------------------------------------------


async def test_mcpserver_composes_with_prefix_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession(tools=[_raw_tool("search")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("gh", command=["py"])
    prefixed = PrefixedToolset(inner=srv, prefix="gh")
    listed = await prefixed.list_tools(ctx=None)
    assert [t.name for t in listed] == ["gh__search"]
    # Dispatch through the wrapper strips the prefix.
    res = await prefixed.call_tool(
        ctx=None, call=ToolCall(id="c1", name="gh__search", arguments={"q": "x"})
    )
    assert isinstance(res, ToolResult)
    assert fake.tool_call_log == [("search", {"q": "x"})]
    await srv.disconnect()


async def test_mcpserver_composes_with_approval_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(tools=[_raw_tool("rm")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("fs", command=["py"])
    guarded = ApprovalRequiredToolset(inner=srv)
    [t] = await guarded.list_tools(ctx=None)
    assert t.policy.needs_approval is True
    await srv.disconnect()


async def test_mcpserver_composes_with_cached_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSession(tools=[_raw_tool("a")])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    cached = CachedToolset(inner=srv, ttl_seconds=60.0)
    await cached.list_tools(ctx=None)
    await cached.list_tools(ctx=None)
    # Inner cache + outer cache both serve subsequent calls.
    assert fake.list_tool_calls == 1
    await srv.disconnect()


# ---------------------------------------------------------------------------
# Optional capabilities (sampling/elicitation stubs + resources)
# ---------------------------------------------------------------------------


async def test_request_sampling_raises_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session_factory(monkeypatch, _FakeSession())
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    with pytest.raises(NotSupportedError, match="sampling"):
        await srv.request_sampling("hi")


async def test_request_elicitation_raises_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session_factory(monkeypatch, _FakeSession())
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    with pytest.raises(NotSupportedError, match="elicitation"):
        await srv.request_elicitation("prompt", {"type": "object"})


async def test_list_resources_delegates_to_session(monkeypatch: pytest.MonkeyPatch) -> None:
    res_obj = SimpleNamespace(uri="x://1", name="r1")
    fake = _FakeSession(resources=[res_obj])
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    listed = await srv.list_resources()
    assert listed == [res_obj]
    await srv.disconnect()


# ---------------------------------------------------------------------------
# Error paths: validation, connect failure
# ---------------------------------------------------------------------------


def test_mcpserver_stdio_rejects_empty_command() -> None:
    with pytest.raises(ToolError, match="non-empty"):
        MCPServerStdio("s", command=[])


async def test_connect_failure_closes_partial_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``initialize()`` raises, the AsyncExitStack must unwind."""
    fake = _FakeSession()
    fake.initialize = AsyncMock(side_effect=RuntimeError("init failed"))  # type: ignore[method-assign]
    _patch_session_factory(monkeypatch, fake)
    _patch_transport(monkeypatch, MCPServerStdio)
    srv = MCPServerStdio("s", command=["py"])
    with pytest.raises(RuntimeError, match="init failed"):
        await srv.connect()
    assert srv._connected is False
