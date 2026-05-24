"""Transport-specific tests for :mod:`agent_harness.core.mcp`.

These tests verify that the three transports (stdio, streamable HTTP,
legacy SSE) wire their parameters into the corresponding ``mcp`` SDK
client functions correctly. The actual streams/processes are mocked so
nothing real is spawned.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from agent_harness.core.mcp import (
    MCPServerHTTP,
    MCPServerSSE,
    MCPServerStdio,
    _LazyHTTPAdapter,
)

# ---------------------------------------------------------------------------
# Stdio transport: parameter construction
# ---------------------------------------------------------------------------


def test_stdio_constructor_stores_command_env_cwd() -> None:
    srv = MCPServerStdio("git", command=["uvx", "mcp-server-git"], env={"K": "v"}, cwd="/tmp")
    assert srv.transport == "stdio"
    assert srv.command == ["uvx", "mcp-server-git"]
    assert srv.env == {"K": "v"}
    assert srv.cwd == "/tmp"


async def test_stdio_open_streams_builds_stdio_server_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_open_streams`` should call ``stdio_client(StdioServerParameters(...))``."""
    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stdio_client(params: Any) -> Any:
        captured["params"] = params
        yield (object(), object())

    import mcp.client.stdio as stdio_mod

    monkeypatch.setattr(stdio_mod, "stdio_client", fake_stdio_client)

    srv = MCPServerStdio("g", command=["echo", "hi"], env={"X": "1"}, cwd="/var")
    cm = srv._open_streams()
    async with cm:
        pass
    params = captured["params"]
    assert params.command == "echo"
    assert params.args == ["hi"]
    assert params.env == {"X": "1"}


# ---------------------------------------------------------------------------
# HTTP transport: URL + auth + headers
# ---------------------------------------------------------------------------


def test_http_constructor_stores_url_auth_headers() -> None:
    async def auth() -> dict[str, str]:
        return {"Authorization": "Bearer t"}

    srv = MCPServerHTTP("h", url="https://e/mcp", auth=auth, headers={"X-K": "v"})
    assert srv.transport == "http"
    assert srv.url == "https://e/mcp"
    assert srv.auth is auth
    assert srv.headers == {"X-K": "v"}


async def test_http_resolve_headers_merges_static_and_auth() -> None:
    async def auth() -> dict[str, str]:
        return {"Authorization": "Bearer t"}

    srv = MCPServerHTTP("h", url="https://e", auth=auth, headers={"X-K": "v"})
    merged = await srv._resolve_headers()
    assert merged == {"X-K": "v", "Authorization": "Bearer t"}


async def test_http_resolve_headers_auth_overrides_static() -> None:
    """Auth result wins on key collisions — refreshed tokens beat stale ones."""

    async def auth() -> dict[str, str]:
        return {"Authorization": "Bearer NEW"}

    srv = MCPServerHTTP("h", url="https://e", auth=auth, headers={"Authorization": "Bearer OLD"})
    merged = await srv._resolve_headers()
    assert merged == {"Authorization": "Bearer NEW"}


async def test_http_resolve_headers_without_auth_returns_static() -> None:
    srv = MCPServerHTTP("h", url="https://e", headers={"X-K": "v"})
    merged = await srv._resolve_headers()
    assert merged == {"X-K": "v"}


# ---------------------------------------------------------------------------
# SSE transport: URL + auth + headers
# ---------------------------------------------------------------------------


def test_sse_constructor_stores_url_auth_headers() -> None:
    async def auth() -> dict[str, str]:
        return {"Authorization": "Bearer t"}

    srv = MCPServerSSE("s", url="https://e/sse", auth=auth, headers={"X-K": "v"})
    assert srv.transport == "sse"
    assert srv.url == "https://e/sse"
    assert srv.auth is auth
    assert srv.headers == {"X-K": "v"}


async def test_sse_resolve_headers_merges() -> None:
    async def auth() -> dict[str, str]:
        return {"Authorization": "Bearer t"}

    srv = MCPServerSSE("s", url="https://e", auth=auth, headers={"X-K": "v"})
    merged = await srv._resolve_headers()
    assert merged == {"X-K": "v", "Authorization": "Bearer t"}


# ---------------------------------------------------------------------------
# _LazyHTTPAdapter: defers transport open until headers resolve
# ---------------------------------------------------------------------------


async def test_lazy_http_adapter_calls_client_with_resolved_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headers are resolved *inside* ``__aenter__`` so OAuth refresh happens
    on each (re)connect rather than at construction."""
    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_client(url: str, *, headers: dict[str, str]) -> Any:
        captured["url"] = url
        captured["headers"] = headers
        yield (object(), object(), lambda: None)

    import mcp.client.streamable_http as http_mod

    monkeypatch.setattr(http_mod, "streamablehttp_client", fake_client)

    resolve_calls = 0

    async def headers_provider() -> dict[str, str]:
        nonlocal resolve_calls
        resolve_calls += 1
        return {"Authorization": "Bearer fresh"}

    adapter = _LazyHTTPAdapter(
        "mcp.client.streamable_http",
        "streamablehttp_client",
        "https://e/mcp",
        headers_provider,
    )
    async with adapter:
        pass

    assert captured["url"] == "https://e/mcp"
    assert captured["headers"] == {"Authorization": "Bearer fresh"}
    assert resolve_calls == 1


async def test_lazy_http_adapter_exit_without_enter_is_safe() -> None:
    """``__aexit__`` before ``__aenter__`` must not blow up."""

    async def hp() -> dict[str, str]:
        return {}

    adapter = _LazyHTTPAdapter("mcp.client.streamable_http", "streamablehttp_client", "u", hp)
    # Exit without ever entering should be a no-op.
    await adapter.__aexit__(None, None, None)


async def test_http_open_streams_uses_streamablehttp_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_client(url: str, *, headers: dict[str, str]) -> Any:
        captured["url"] = url
        captured["headers"] = headers
        yield (object(), object(), lambda: None)

    import mcp.client.streamable_http as http_mod

    monkeypatch.setattr(http_mod, "streamablehttp_client", fake_client)

    srv = MCPServerHTTP("h", url="https://e/mcp", headers={"X": "1"})
    cm = srv._open_streams()
    async with cm:
        pass
    assert captured["url"] == "https://e/mcp"
    assert captured["headers"] == {"X": "1"}


async def test_sse_open_streams_uses_sse_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_client(url: str, *, headers: dict[str, str]) -> Any:
        captured["url"] = url
        captured["headers"] = headers
        yield (object(), object())

    import mcp.client.sse as sse_mod

    monkeypatch.setattr(sse_mod, "sse_client", fake_client)

    srv = MCPServerSSE("s", url="https://e/sse", headers={"X": "1"})
    cm = srv._open_streams()
    async with cm:
        pass
    assert captured["url"] == "https://e/sse"
    assert captured["headers"] == {"X": "1"}


# ---------------------------------------------------------------------------
# Transport markers
# ---------------------------------------------------------------------------


def test_transport_class_attributes() -> None:
    assert MCPServerStdio.transport == "stdio"
    assert MCPServerHTTP.transport == "http"
    assert MCPServerSSE.transport == "sse"
