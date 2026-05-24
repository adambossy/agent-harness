"""Model-facing filesystem tools (Layer 1).

:class:`FilesystemTools` is a built-in :class:`~agent_harness.core.toolsets.Toolset`
exposing ``read``, ``write``, ``edit``, ``grep``, ``glob``, and ``list_dir``
to the model (FT2). All I/O routes through the constructor-injected
:class:`~agent_harness.core.sandbox.Sandbox` so swapping the sandbox
automatically retargets every tool (FT1, FT3).

- **Truncation** (FT6): ``read`` caps at ``truncate_read_lines`` or
  ``truncate_read_bytes`` — whichever hits first. A marker is appended.
- **Unique-match edit** (FT7): ``edit`` requires its ``old_string`` to
  match exactly once by default (``expect_unique=True``).
- **Ignore-set** (FT4): an optional ``IgnoreSet`` (typed :data:`Any` here
  because the concrete class ships in ``agent_harness.extras.ignoreset``)
  duck-typed on ``matches(path) -> bool``.
- **Multi-root** (FT5): optional ``roots`` constrains which subtrees the
  tools may touch.

Example:
    >>> import asyncio
    >>> from tests.fakes import FakeSandbox
    >>> async def _demo() -> str:
    ...     ft = FilesystemTools(sandbox=FakeSandbox())
    ...     return ",".join(sorted(t.name for t in await ft.list_tools(None)))
    >>> asyncio.run(_demo())
    'edit,glob,grep,list_dir,read,write'
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .errors import ToolError
from .models import TextBlock
from .sandbox import FileEntry, Sandbox
from .tools import Tool, ToolCall, ToolPolicy, ToolResult, tool

_TRUNCATED_MARKER = "\n[... truncated by FilesystemTools ...]\n"

# IgnoreSet (FT4) ships in extras (Wave-3 peer); typed ``Any`` here. The
# Toolset only needs a ``matches(path: str) -> bool`` method.
IgnoreSetLike = Any


@dataclass(frozen=True, slots=True)
class GrepMatch:
    """A single ``grep`` hit.

    Example:
        >>> GrepMatch(path="a.txt", line_number=3, line="hit").line
        'hit'
    """

    path: str
    line_number: int
    line: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view returned to the model."""
        return {"path": self.path, "line_number": self.line_number, "line": self.line}


@dataclass(slots=True)
class FilesystemTools:
    """Built-in :class:`~agent_harness.core.toolsets.Toolset` exposing the
    standard filesystem surface to the model (FT2).

    Tool objects are constructed once at ``__post_init__`` so JSON schemas
    are stable; every body closes over ``self`` and routes through
    ``self.sandbox`` (FT1 / FT3).

    Example:
        >>> from tests.fakes import FakeSandbox
        >>> FilesystemTools(sandbox=FakeSandbox()).name
        'filesystem'
    """

    sandbox: Sandbox
    ignore: IgnoreSetLike | None = None
    roots: list[str] | None = None
    truncate_read_lines: int = 2000
    truncate_read_bytes: int = 50_000
    name: str = "filesystem"
    _tools: list[Tool] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._tools = [
            _build_read_tool(self),
            _build_write_tool(self),
            _build_edit_tool(self),
            _build_grep_tool(self),
            _build_glob_tool(self),
            _build_list_dir_tool(self),
        ]

    async def list_tools(self, ctx: Any) -> list[Tool]:
        """Return the six filesystem tools. ``ctx`` is unused."""
        del ctx
        return list(self._tools)

    async def call_tool(self, ctx: Any, call: ToolCall) -> ToolResult:
        """Dispatch ``call`` to one of the six filesystem tools."""
        del ctx
        for t in self._tools:
            if t.name == call.name:
                return await _invoke(t, call.arguments)
        raise ToolError(
            f"no tool named {call.name!r} in FilesystemTools",
            context={"requested": call.name, "available": [t.name for t in self._tools]},
        )

    def _check_path(self, path: str) -> None:
        """Reject paths blocked by the ignore-set or outside ``roots``.

        Raises :class:`ToolError`; :func:`_invoke` wraps it as an errored
        :class:`ToolResult` so the loop sees a normal failure.
        """
        if self.ignore is not None:
            matches = getattr(self.ignore, "matches", None)
            if callable(matches) and matches(path):
                raise ToolError(
                    f"path {path!r} is excluded by the ignore-set",
                    context={"path": path},
                )
        if self.roots is not None and not _within_any_root(path, self.roots):
            raise ToolError(
                f"path {path!r} is outside the configured roots",
                context={"path": path, "roots": list(self.roots)},
            )


# --- Helpers ---------------------------------------------------------------


def _within_any_root(path: str, roots: list[str]) -> bool:
    """Return True iff ``path`` lives under at least one of ``roots``.

    Example:
        >>> _within_any_root("src/a.py", ["src"])
        True
        >>> _within_any_root("/etc/passwd", ["src"])
        False
    """
    p = PurePosixPath(path)
    for root in roots:
        try:
            p.relative_to(PurePosixPath(root))
        except ValueError:
            continue
        return True
    return False


def _truncate(content: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    """Cut ``content`` at whichever of ``max_lines`` / ``max_bytes`` hits first.

    Example:
        >>> body = "\\n".join(str(i) for i in range(5))
        >>> text, cut = _truncate(body, max_lines=3, max_bytes=10_000)
        >>> cut
        True
    """
    lines = content.splitlines(keepends=True)
    truncated = False
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    text = "".join(lines)
    encoded = text.encode("utf-8")
    if max_bytes > 0 and len(encoded) > max_bytes:
        text = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True
    if truncated:
        text = text + _TRUNCATED_MARKER
    return text, truncated


async def _invoke(tool_obj: Tool, arguments: dict[str, Any]) -> ToolResult:
    """Run a tool body and wrap its return value as a :class:`ToolResult`."""
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
    text = "" if raw is None else raw if isinstance(raw, str) else json.dumps(raw, default=str)
    return ToolResult(content=[TextBlock(text=text)])


# --- Tool builders ---------------------------------------------------------


def _build_read_tool(ft: FilesystemTools) -> Tool:
    """Build the ``read`` tool bound to ``ft``."""

    async def read(
        path: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        """Read a UTF-8 text file (optionally restricted to a line range).

        Args:
            path: Sandbox-relative file path.
            line_start: 1-based inclusive first line; ``None`` = start.
            line_end: 1-based inclusive last line; ``None`` = EOF.
        """
        ft._check_path(path)
        content = await ft.sandbox.read_file(path)
        if line_start is not None or line_end is not None:
            lines = content.splitlines(keepends=True)
            start = max(1, line_start or 1) - 1
            end = len(lines) if line_end is None else max(start, line_end)
            content = "".join(lines[start:end])
        text, _ = _truncate(content, ft.truncate_read_lines, ft.truncate_read_bytes)
        return text

    return tool(
        name="read",
        description="Read a UTF-8 text file, optionally restricted to a line range.",
        policy=ToolPolicy(is_read_only=True, is_concurrency_safe=True),
    )(read)


def _build_write_tool(ft: FilesystemTools) -> Tool:
    """Build the ``write`` tool bound to ``ft``."""

    async def write(path: str, content: str) -> str:
        """Write a UTF-8 text file, replacing any existing contents.

        Args:
            path: Sandbox-relative file path.
            content: Text to write.
        """
        ft._check_path(path)
        await ft.sandbox.write_file(path, content)
        return f"wrote {len(content)} chars to {path}"

    return tool(
        name="write",
        description="Write text to a file, replacing any existing contents.",
        policy=ToolPolicy(is_destructive=True),
    )(write)


def _build_edit_tool(ft: FilesystemTools) -> Tool:
    """Build the ``edit`` tool bound to ``ft``."""

    async def edit(
        path: str,
        old_string: str,
        new_string: str,
        expect_unique: bool = True,
    ) -> str:
        """Replace ``old_string`` with ``new_string`` inside a file.

        Args:
            path: Sandbox-relative file path.
            old_string: Exact text to match (non-empty).
            new_string: Replacement text.
            expect_unique: If True (default, FT7), require a unique match.
        """
        if not old_string:
            raise ToolError("edit requires a non-empty old_string", context={"path": path})
        ft._check_path(path)
        content = await ft.sandbox.read_file(path)
        count = content.count(old_string)
        if count == 0:
            raise ToolError(f"old_string not found in {path!r}", context={"path": path})
        if expect_unique and count > 1:
            raise ToolError(
                (
                    f"old_string matched {count} times in {path!r}; "
                    "pass expect_unique=False or extend the snippet"
                ),
                context={"path": path, "match_count": count},
            )
        updated = content.replace(old_string, new_string, 1)
        await ft.sandbox.write_file(path, updated)
        plural = "s" if count != 1 else ""
        return f"edited {path} (1 of {count} occurrence{plural} replaced)"

    return tool(
        name="edit",
        description=(
            "Replace exact text inside a file. Requires the match to be unique "
            "by default (set expect_unique=False to relax)."
        ),
        policy=ToolPolicy(is_destructive=True),
    )(edit)


def _build_grep_tool(ft: FilesystemTools) -> Tool:
    """Build the ``grep`` tool bound to ``ft``."""

    async def grep(
        pattern: str,
        path: str | None = None,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Search a file (or every file under the active roots) for a pattern.

        Args:
            pattern: Substring or regex (``regex=True``).
            path: Optional single-file target; otherwise walks roots.
            regex: Treat ``pattern`` as a Python regex.
            case_sensitive: Default False — case is ignored unless True.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)
        targets = await _grep_targets(ft, path)
        results: list[dict[str, Any]] = []
        for target in targets:
            try:
                ft._check_path(target)
            except ToolError:
                continue
            try:
                body = await ft.sandbox.read_file(target)
            except FileNotFoundError:
                continue
            for i, line in enumerate(body.splitlines(), start=1):
                if compiled.search(line):
                    results.append(GrepMatch(target, i, line).as_dict())
        return results

    return tool(
        name="grep",
        description="Search files for a substring or regex; returns line-numbered matches.",
        policy=ToolPolicy(is_read_only=True, is_concurrency_safe=True),
    )(grep)


def _build_glob_tool(ft: FilesystemTools) -> Tool:
    """Build the ``glob`` tool bound to ``ft``."""

    async def glob(pattern: str, root: str | None = None) -> list[str]:
        """Return every path under ``root`` whose name matches ``pattern``.

        Args:
            pattern: A :meth:`pathlib.PurePath.match`-style glob.
            root: Starting directory; defaults to the sandbox root.
        """
        start = root if root is not None else ft.sandbox.root
        hits: list[str] = []
        async for fp in _walk(ft, start):
            if PurePosixPath(fp).match(pattern):
                try:
                    ft._check_path(fp)
                except ToolError:
                    continue
                hits.append(fp)
        hits.sort()
        return hits

    return tool(
        name="glob",
        description="List files matching a glob pattern (PurePath.match semantics).",
        policy=ToolPolicy(is_read_only=True, is_concurrency_safe=True),
    )(glob)


def _build_list_dir_tool(ft: FilesystemTools) -> Tool:
    """Build the ``list_dir`` tool bound to ``ft``."""

    async def list_dir(path: str) -> list[dict[str, Any]]:
        """List the immediate entries of ``path`` (non-recursive).

        Args:
            path: Sandbox-relative directory path.
        """
        ft._check_path(path)
        entries = await ft.sandbox.readdir(path)
        return [{"name": e.name, "is_dir": e.is_dir} for e in entries]

    return tool(
        name="list_dir",
        description="List the immediate entries of a directory (non-recursive).",
        policy=ToolPolicy(is_read_only=True, is_concurrency_safe=True),
    )(list_dir)


# --- Tree walking ----------------------------------------------------------


async def _grep_targets(ft: FilesystemTools, path: str | None) -> list[str]:
    """Return the list of files ``grep`` should scan."""
    if path is not None:
        return [path]
    roots = ft.roots if ft.roots else [ft.sandbox.root]
    out: list[str] = []
    for root in roots:
        async for fp in _walk(ft, root):
            out.append(fp)
    return out


async def _walk(ft: FilesystemTools, start: str) -> AsyncIterator[str]:
    """Yield every file path under ``start`` (recursive via ``readdir``)."""
    stack: list[str] = [start]
    while stack:
        cur = stack.pop()
        try:
            entries = await ft.sandbox.readdir(cur)
        except (FileNotFoundError, NotADirectoryError):
            continue
        for e in entries:
            assert isinstance(e, FileEntry)
            joined = _join(cur, e.name)
            if e.is_dir:
                stack.append(joined)
            else:
                yield joined


def _join(parent: str, name: str) -> str:
    """POSIX path join (every shipped sandbox uses forward slashes)."""
    if not parent or parent in {"/", "."}:
        return name
    return f"{parent.rstrip('/')}/{name}"


__all__ = ["FilesystemTools", "GrepMatch", "IgnoreSetLike"]
