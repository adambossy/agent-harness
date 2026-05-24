"""MCP servers presented as :class:`Toolset`\\ s (Wave 3 — client only).

`MCPServer` *IS* a :class:`~agent_harness.core.toolsets.Toolset`. Because of
that, every existing wrapper — :class:`~agent_harness.core.toolsets.PrefixedToolset`,
:class:`~agent_harness.core.toolsets.FilteredToolset`,
:class:`~agent_harness.core.toolsets.ApprovalRequiredToolset`,
:class:`~agent_harness.core.toolsets.CachedToolset` — composes with MCP for
free (per the pydantic-ai framing).

Three transports ship in v0.0.1: stdio (subprocess), streamable HTTP, and
legacy SSE. The :mod:`mcp` SDK is an *optional* dependency; importing it is
deferred until an :class:`MCPServer` is constructed. If absent, the
constructor raises :class:`~agent_harness.core.errors.NotSupportedError`.

Per ``open-questions.md`` #3, v0.0.1 is **client-only**. There is no server
endpoint; ``from_agent`` is explicitly out of scope.

Example:
    >>> # Construction in test code (uses stdio):
    >>> from agent_harness.core.mcp import MCPServerStdio
    >>> # srv = MCPServerStdio("local", command=["python", "-m", "my_mcp"])
    >>> # async with srv:
    >>> #     tools = await srv.list_tools(ctx=None)
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Literal

from .errors import NotSupportedError, ToolError
from .models import ImageBlock, TextBlock
from .tools import Tool, ToolCall, ToolPolicy, ToolResult

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .models import ContentBlock


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# OAuth handler: returns headers to attach to *every* request. Called before
# each transport open so refreshed tokens surface naturally.
AuthHandler = Callable[[], Awaitable[dict[str, str]]]


def _require_mcp() -> Any:
    """Lazy-import the :mod:`mcp` SDK; raise NotSupportedError if absent.

    MCP is an *optional* dependency (``[project.optional-dependencies] mcp``).
    The harness only needs the SDK at construction time — importing it
    eagerly would force every user of ``agent_harness.core`` to install it.

    Example:
        >>> mcp_mod = _require_mcp()  # doctest: +SKIP
        >>> hasattr(mcp_mod, "ClientSession")  # doctest: +SKIP
        True
    """
    try:
        return importlib.import_module("mcp")
    except ImportError as exc:  # pragma: no cover - exercised only without mcp
        raise NotSupportedError(
            "The 'mcp' optional dependency is not installed. "
            "Install with: pip install 'agent-harness[mcp]'",
            cause=exc,
        ) from exc


def _convert_mcp_content(blocks: list[Any]) -> list[ContentBlock]:
    """Translate the MCP SDK's content-block list to harness blocks.

    MCP TextContent / ImageContent are pass-through; unknown shapes (audio,
    resource_link, embedded_resource) collapse to a TextBlock carrying a
    JSON-serialized payload so the model sees *something* rather than a
    silently-dropped block.
    """
    out: list[ContentBlock] = []
    for b in blocks:
        kind = getattr(b, "type", None)
        if kind == "text":
            out.append(TextBlock(text=getattr(b, "text", "")))
        elif kind == "image":
            out.append(
                ImageBlock(
                    data=getattr(b, "data", None),
                    mime_type=getattr(b, "mimeType", "image/png"),
                )
            )
        else:
            # Fall back: stringify so the model isn't silently shown nothing.
            out.append(TextBlock(text=str(b)))
    return out


def _mcp_tool_to_harness(raw: Any) -> Tool:
    """Map an MCP ``types.Tool`` to a harness :class:`Tool`.

    The MCP tool has no executable body on the client side — the function
    is a placeholder that should never run (dispatch always routes through
    :meth:`MCPServer.call_tool`). We attach an async stub that raises so
    bypassing the toolset is loud rather than silent.
    """
    name = str(raw.name)

    async def _unreachable(**_kwargs: Any) -> ToolResult:
        # Defensive: the loop dispatches via Toolset.call_tool, not Tool.fn.
        raise ToolError(
            f"MCP tool {name!r} must be dispatched via MCPServer.call_tool",
            context={"tool": name},
        )

    return Tool(
        name=name,
        description=getattr(raw, "description", "") or "",
        schema=getattr(raw, "inputSchema", None) or {"type": "object", "properties": {}},
        policy=ToolPolicy(),
        fn=_unreachable,
    )


# ---------------------------------------------------------------------------
# MCPServer base
# ---------------------------------------------------------------------------


class MCPServer:
    """An MCP server presented as a Toolset.

    Subclass-per-transport. The base implements the cache + lifecycle +
    sampling/elicitation stubs; subclasses (:class:`MCPServerStdio`,
    :class:`MCPServerHTTP`, :class:`MCPServerSSE`) supply ``_open_streams``.

    Because the structural :class:`~agent_harness.core.toolsets.Toolset`
    Protocol only requires ``name``, ``list_tools``, and ``call_tool``,
    ``MCPServer`` satisfies it without inheriting (Protocols are structural).

    Example:
        >>> from agent_harness.core.toolsets import Toolset
        >>> from agent_harness.core.mcp import MCPServerStdio  # doctest: +SKIP
        >>> srv = MCPServerStdio("local", command=["python"])  # doctest: +SKIP
        >>> isinstance(srv, Toolset)  # doctest: +SKIP
        True
    """

    transport: Literal["stdio", "http", "sse"]

    def __init__(self, name: str) -> None:
        # Force mcp import at construction; surfaces "not installed" loudly.
        _require_mcp()
        self.name = name
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._tools_cache: list[Tool] | None = None
        self._connected: bool = False

    # ---- Lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        """Open the transport, initialize the session, register notification
        handler for ``tools/list_changed`` cache invalidation.

        Idempotent: a second call while connected is a no-op.
        """
        if self._connected:
            return
        mcp_mod = _require_mcp()
        session_cls = mcp_mod.ClientSession
        stack = AsyncExitStack()
        try:
            streams = await stack.enter_async_context(self._open_streams())
            # Transports return (read, write) or (read, write, get_session_id).
            read_stream, write_stream = streams[0], streams[1]
            session = await stack.enter_async_context(
                session_cls(
                    read_stream,
                    write_stream,
                    message_handler=self._on_message,
                )
            )
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        self._connected = True

    async def disconnect(self) -> None:
        """Close the session and underlying transport. Idempotent."""
        if not self._connected:
            return
        self._connected = False
        self._tools_cache = None
        self._session = None
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    async def __aenter__(self) -> MCPServer:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.disconnect()

    def _open_streams(self) -> Any:
        """Return the transport-specific async context manager yielding
        ``(read_stream, write_stream[, get_session_id])``. Implemented by
        subclasses."""
        raise NotImplementedError  # pragma: no cover

    # ---- Toolset surface --------------------------------------------------

    async def list_tools(self, ctx: Any) -> list[Tool]:
        """Return the server's tool catalog. Cached until ``list_changed``.

        ``ctx`` is ignored — MCP catalogs aren't per-run, but the parameter
        is part of the Toolset Protocol.
        """
        del ctx
        if not self._connected:
            await self.connect()
        if self._tools_cache is not None:
            return list(self._tools_cache)
        result = await self._session.list_tools()
        self._tools_cache = [_mcp_tool_to_harness(t) for t in result.tools]
        return list(self._tools_cache)

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        """Dispatch ``call`` to the server. MCP errors surface as
        ``ToolResult(error=...)`` per MC7."""
        del ctx
        if not self._connected:
            await self.connect()
        try:
            raw = await self._session.call_tool(call.name, call.arguments or None)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return ToolResult(content=[TextBlock(text=message)], error=message)
        content = _convert_mcp_content(list(raw.content))
        error_msg: str | None = None
        if getattr(raw, "isError", False):
            # Use first text block's text as the error message if present.
            for blk in content:
                if isinstance(blk, TextBlock):
                    error_msg = blk.text
                    break
            if error_msg is None:
                error_msg = f"MCP tool {call.name!r} returned isError=True"
        metadata: dict[str, Any] = {}
        if getattr(raw, "structuredContent", None) is not None:
            metadata["structured"] = raw.structuredContent
        return ToolResult(content=content, error=error_msg, metadata=metadata)

    # ---- Optional MCP capabilities ---------------------------------------

    async def list_resources(self) -> list[Any]:
        """List server-exposed resources. Returns raw MCP ``Resource``\\ s."""
        if not self._connected:
            await self.connect()
        result = await self._session.list_resources()
        return list(result.resources)

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI. Returns the raw MCP result."""
        if not self._connected:
            await self.connect()
        mcp_mod = _require_mcp()
        any_url = mcp_mod.types.AnyUrl if hasattr(mcp_mod, "types") else None
        target: Any = any_url(uri) if any_url is not None else uri
        return await self._session.read_resource(target)

    async def request_sampling(self, prompt: str, **opts: Any) -> str:
        """MCP-server-initiated sampling against the host's LLM.

        v0.0.1 stub: optional capability is wired via callback at the
        :class:`mcp.ClientSession` boundary. Until Wave-4 integrates with
        the active model, this raises ``NotSupportedError``.
        """
        del prompt, opts
        raise NotSupportedError(
            "MCP sampling is not yet wired to the host model (v0.0.1 stub).",
            context={"server": self.name},
        )

    async def request_elicitation(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """MCP-server-initiated user elicitation. v0.0.1 stub.

        Once integrated, this will publish an ``ElicitationRequested`` event
        and await the user's structured reply.
        """
        del prompt, schema
        raise NotSupportedError(
            "MCP elicitation is not yet wired to the EventBus (v0.0.1 stub).",
            context={"server": self.name},
        )

    # ---- Notifications ----------------------------------------------------

    async def _on_message(self, message: Any) -> None:
        """Message handler installed on the underlying ``ClientSession``.

        Watches for ``notifications/tools/list_changed`` and invalidates the
        cached tool list. Other notifications are ignored (subscribers can
        observe them via Wave-4 event bridging).
        """
        # ServerNotification is a RootModel wrapping a union; reach through
        # ``root`` if present.
        inner = getattr(message, "root", message)
        method = getattr(inner, "method", None)
        if method == "notifications/tools/list_changed":
            self._tools_cache = None


# ---------------------------------------------------------------------------
# Concrete transports
# ---------------------------------------------------------------------------


class MCPServerStdio(MCPServer):
    """Run an MCP server as a subprocess; communicate over stdio.

    The most common transport for *local* MCP servers (filesystem, git, ...).

    Example:
        >>> srv = MCPServerStdio(  # doctest: +SKIP
        ...     "git",
        ...     command=["uvx", "mcp-server-git"],
        ...     env={"GIT_AUTHOR_NAME": "agent"},
        ... )
    """

    transport: Literal["stdio"] = "stdio"

    def __init__(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if not command:
            raise ToolError(
                "MCPServerStdio.command must be a non-empty list[str]",
                context={"server": name},
            )
        super().__init__(name)
        self.command = list(command)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd

    def _open_streams(self) -> Any:
        mcp_mod = _require_mcp()
        params = mcp_mod.StdioServerParameters(
            command=self.command[0],
            args=list(self.command[1:]),
            env=self.env,
            cwd=self.cwd,
        )
        stdio_mod = importlib.import_module("mcp.client.stdio")
        return stdio_mod.stdio_client(params)


class _LazyHTTPAdapter:
    """Defer transport open until headers can be resolved (incl. OAuth).

    The MCP HTTP / SSE clients are async context managers that need their
    headers at construction time. To support per-connect OAuth refresh, we
    wrap their entry in an adapter that resolves headers in ``__aenter__``.
    """

    def __init__(
        self,
        module_path: str,
        client_attr: str,
        url: str,
        headers_provider: Callable[[], Awaitable[dict[str, str]]],
    ) -> None:
        self._module_path = module_path
        self._client_attr = client_attr
        self._url = url
        self._headers_provider = headers_provider
        self._cm: Any = None

    async def __aenter__(self) -> Any:
        headers = await self._headers_provider()
        mod = importlib.import_module(self._module_path)
        client_fn = getattr(mod, self._client_attr)
        self._cm = client_fn(self._url, headers=headers)
        return await self._cm.__aenter__()

    async def __aexit__(self, *exc: object) -> Any:
        if self._cm is None:
            return None
        return await self._cm.__aexit__(*exc)


class _HTTPLike(MCPServer):
    """Shared HTTP / SSE transport scaffolding (URL + auth + headers)."""

    _module_path: str = ""
    _client_attr: str = ""

    def __init__(
        self,
        name: str,
        url: str,
        *,
        auth: AuthHandler | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name)
        self.url = url
        self.auth = auth
        self.headers = dict(headers) if headers is not None else {}

    async def _resolve_headers(self) -> dict[str, str]:
        merged = dict(self.headers)
        if self.auth is not None:
            merged.update(await self.auth())
        return merged

    def _open_streams(self) -> Any:
        return _LazyHTTPAdapter(
            self._module_path, self._client_attr, self.url, self._resolve_headers
        )


class MCPServerHTTP(_HTTPLike):
    """Streamable HTTP transport — the modern MCP transport.

    Bring-your-own ``auth`` callable returns headers per connect. Handles
    OAuth refresh, static API keys, and per-tenant credentials uniformly.

    Example:
        >>> async def my_auth() -> dict[str, str]:
        ...     return {"Authorization": "Bearer ..."}
        >>> srv = MCPServerHTTP("github", url="https://api/mcp", auth=my_auth)  # doctest: +SKIP
    """

    transport: Literal["http"] = "http"
    _module_path = "mcp.client.streamable_http"
    _client_attr = "streamablehttp_client"


class MCPServerSSE(_HTTPLike):
    """Legacy SSE transport. Prefer :class:`MCPServerHTTP` for new deployments.

    Example:
        >>> srv = MCPServerSSE("legacy", url="https://legacy/mcp")  # doctest: +SKIP
    """

    transport: Literal["sse"] = "sse"
    _module_path = "mcp.client.sse"
    _client_attr = "sse_client"


__all__ = [
    "AuthHandler",
    "MCPServer",
    "MCPServerHTTP",
    "MCPServerSSE",
    "MCPServerStdio",
]
