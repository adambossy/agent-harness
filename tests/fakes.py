"""Test fakes for the E2E smoke test and unit tests.

- :class:`FakeProvider` — no-op :class:`Provider` (bypassed by FakeModel).
- :class:`FakeTurn` — one scripted assistant response.
- :class:`FakeModel` — deterministic :class:`Model` yielding scripted turns
  as a typed event stream; chunks text into 3 cumulative deltas (EV5).
- :class:`FakeSandbox` — in-memory dict-backed sandbox; satisfies the
  9-method ``Sandbox`` Protocol structurally and returns the real
  ``FileStat`` / ``FileEntry`` / ``ExecResult`` dataclasses.

Example:
    >>> import asyncio
    >>> from agent_harness.core.models import ModelSettings
    >>> model = FakeModel(script=[FakeTurn(text="hi")])
    >>> async def _drain() -> int:
    ...     return sum(1 async for _ in model.request([], [], ModelSettings()))
    >>> asyncio.run(_drain()) > 0
    True
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from agent_harness.core.events import (
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from agent_harness.core.models import (
    Message,
    ModelCapabilities,
    ModelSettings,
    Provider,
    ProviderEvent,
    TextBlock,
    ToolCallBlock,
    Usage,
)
from agent_harness.core.sandbox import ExecResult, FileEntry, FileStat, SandboxConfig
from agent_harness.core.tools import ToolCall

# --- FakeProvider -----------------------------------------------------------


class FakeProvider:
    """No-op :class:`Provider` — :class:`FakeModel` bypasses it.

    Example:
        >>> from agent_harness.core.models import Provider
        >>> isinstance(FakeProvider(), Provider)
        True
    """

    def __init__(self, name: str = "fake-provider") -> None:
        self.name: str = name
        self.base_url: str | None = None

    async def request(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Never executed; the ``yield`` keeps this an async-generator."""
        del payload, stream, timeout
        _never: bool = False
        if _never:
            yield ProviderEvent(kind="raw")


# --- FakeTurn ---------------------------------------------------------------


@dataclass(slots=True)
class FakeTurn:
    """One scripted model response (= one ``Model.request`` invocation).

    Example:
        >>> FakeTurn(text="hi").text
        'hi'
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=100, output_tokens=20))


# --- FakeModel --------------------------------------------------------------


def _chunk_into(s: str, n: int) -> list[str]:
    """Split ``s`` into exactly ``n`` chunks whose concatenation equals ``s``.

    Example:
        >>> _chunk_into("abcdef", 3)
        ['ab', 'cd', 'ef']
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not s:
        return [""] * n
    size = max(1, len(s) // n)
    pieces = [s[i : i + size] for i in range(0, len(s), size)]
    return [*pieces[: n - 1], "".join(pieces[n - 1 :])]


class FakeModel:
    """Test :class:`Model` emitting scripted responses, chunking text into
    3 cumulative deltas (the EV5 invariant).

    Example:
        >>> from agent_harness.core.models import Model
        >>> isinstance(FakeModel(script=[FakeTurn(text="hi")]), Model)
        True
    """

    capabilities = ModelCapabilities(
        parallel_tool_calls=True,
        structured_output=True,
        context_window=200_000,
    )

    def __init__(
        self,
        *,
        script: list[FakeTurn],
        provider: Provider | None = None,
    ) -> None:
        self.name: str = "fake-model"
        # Rationale for the type-ignore: ``Provider`` is a runtime-checkable
        # Protocol; ``FakeProvider`` structurally satisfies it (verified by
        # the matching unit test). Mypy can't bridge the async-generator
        # / coroutine return-type mismatch between an ``async def`` function
        # and the Protocol's declared ``-> AsyncIterator`` shape.
        self.provider: Provider = provider if provider is not None else FakeProvider()  # type: ignore[assignment]
        self._script: list[FakeTurn] = script
        self._turn: int = 0

    async def request(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> AsyncIterator[Any]:
        """Yield the next scripted turn as a typed model-event stream."""
        del messages, tools, settings
        if self._turn >= len(self._script):
            raise AssertionError(
                f"FakeModel script exhausted at turn {self._turn}; "
                f"loop is making more requests than the script provides"
            )
        turn = self._script[self._turn]
        self._turn += 1

        msg_id = f"msg_{self._turn:03d}"
        yield ModelStart(model_name=self.name)
        yield MessageStart(message_id=msg_id)

        partial_text = ""
        for chunk in _chunk_into(turn.text, n=3):
            partial_text += chunk
            yield MessageDelta(
                message_id=msg_id,
                delta=chunk,
                partial=Message(
                    role="assistant",
                    content=[TextBlock(text=partial_text)],
                    timestamp=datetime.now(UTC),
                ),
            )

        for tc in turn.tool_calls:
            yield ToolCallStart(tool_call_id=tc.id, tool_name=tc.name)
            yield ToolCallDelta(tool_call_id=tc.id, arguments_delta=json.dumps(tc.arguments))
            yield ToolCallEnd(tool_call_id=tc.id, tool_name=tc.name, arguments=tc.arguments)

        final_content: list[Any] = [TextBlock(text=turn.text)]
        final_content.extend(
            ToolCallBlock(id=tc.id, name=tc.name, arguments=tc.arguments) for tc in turn.tool_calls
        )
        final = Message(
            role="assistant",
            content=final_content,
            timestamp=datetime.now(UTC),
        )
        yield MessageEnd(message_id=msg_id, final=final, usage=turn.usage)
        yield ModelEnd(message_id=msg_id, usage=turn.usage)

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        """No-op compaction — returns the input unchanged."""
        return list(msgs)


# --- FakeSandbox ------------------------------------------------------------


@dataclass(slots=True)
class _FileNode:
    content: bytes
    mtime: datetime


class FakeSandbox:
    """In-memory dict-backed ``Sandbox``.

    Structurally satisfies the 9-method :class:`Sandbox` Protocol from
    ``core/sandbox.py`` and returns the real :class:`FileStat`,
    :class:`FileEntry`, :class:`ExecResult` dataclasses.

    Example:
        >>> import asyncio
        >>> fs = FakeSandbox()
        >>> asyncio.run(fs.write_file("a.txt", "hi"))
        >>> asyncio.run(fs.read_file("a.txt"))
        'hi'
    """

    def __init__(self, root: str = "/workspace", name: str = "fake-sandbox") -> None:
        self.name: str = name
        self.root: str = root
        self.config: SandboxConfig = SandboxConfig()
        self._files: dict[str, _FileNode] = {}
        self._dirs: set[str] = {""}

    async def read_file(self, path: str, *, timeout: float | None = None) -> str:
        del timeout
        node = self._files.get(path)
        if node is None:
            raise FileNotFoundError(path)
        return node.content.decode("utf-8")

    async def write_file(self, path: str, content: str, *, timeout: float | None = None) -> None:
        del timeout
        self._files[path] = _FileNode(content=content.encode("utf-8"), mtime=datetime.now(UTC))

    async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
        del timeout
        node = self._files.get(path)
        if node is None:
            if path in self._dirs:
                return FileStat(
                    path=path,
                    size=0,
                    mtime=datetime.now(UTC),
                    is_dir=True,
                )
            raise FileNotFoundError(path)
        return FileStat(path=path, size=len(node.content), mtime=node.mtime, is_dir=False)

    async def readdir(self, path: str, *, timeout: float | None = None) -> list[FileEntry]:
        del timeout
        prefix = "" if path in {"", "/", "."} else path.rstrip("/") + "/"
        return [
            FileEntry(name=fp[len(prefix) :], is_dir=False)
            for fp in self._files
            if fp.startswith(prefix) and "/" not in fp[len(prefix) :]
        ]

    async def exists(self, path: str, *, timeout: float | None = None) -> bool:
        del timeout
        return path in self._files or path in self._dirs

    async def mkdir(
        self,
        path: str,
        *,
        parents: bool = False,
        timeout: float | None = None,
    ) -> None:
        del parents, timeout
        self._dirs.add(path)

    async def rm(
        self,
        path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        del recursive, timeout
        self._files.pop(path, None)
        self._dirs.discard(path)

    async def read_file_bytes(self, path: str, *, timeout: float | None = None) -> bytes:
        del timeout
        node = self._files.get(path)
        if node is None:
            raise FileNotFoundError(path)
        return node.content

    async def exec(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Stub: returns an :class:`ExecResult` with ``exit_code=0``."""
        del cmd, cwd, env, stdin, timeout
        return ExecResult(exit_code=0, stdout="", stderr="", timed_out=False)


# --- helpers ---------------------------------------------------------------


def make_model(*turns: FakeTurn, provider: Provider | None = None) -> Any:
    """Construct a :class:`FakeModel` typed as :class:`Model` for mypy.

    ``FakeModel.request`` is an async generator function whose runtime shape
    is ``AsyncIterator``; the :class:`Model` Protocol declares it as
    ``async def -> AsyncIterator``. The two are runtime-compatible but mypy
    can't bridge them, so we cast to ``Any`` (still satisfies the structural
    Protocol check at runtime).

    Example:
        >>> isinstance(make_model(FakeTurn(text="hi")), FakeModel)
        True
    """
    from agent_harness.core.models import Model

    return cast(Model, FakeModel(script=list(turns), provider=provider))
