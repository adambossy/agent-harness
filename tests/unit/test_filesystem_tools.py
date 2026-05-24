"""Unit tests for ``agent_harness.core.filesystem``."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_harness.core.filesystem import FilesystemTools, GrepMatch
from agent_harness.core.models import TextBlock
from agent_harness.core.tools import ToolCall, ToolResult
from agent_harness.core.toolsets import Toolset
from tests.fakes import FakeSandbox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed(sb: FakeSandbox, files: dict[str, str]) -> None:
    """Populate ``sb`` with ``{path: content}`` pairs."""
    for path, content in files.items():
        await sb.write_file(path, content)


def _text(result: ToolResult) -> str:
    """Return the concatenated text payload of a ToolResult."""
    return "".join(b.text for b in result.content if isinstance(b, TextBlock))


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_filesystem_tools_satisfies_toolset_protocol() -> None:
    ft = FilesystemTools(sandbox=FakeSandbox())
    assert isinstance(ft, Toolset)


async def test_list_tools_returns_exactly_the_six_filesystem_tools() -> None:
    ft = FilesystemTools(sandbox=FakeSandbox())
    listed = await ft.list_tools(ctx=None)
    names = sorted(t.name for t in listed)
    assert names == ["edit", "glob", "grep", "list_dir", "read", "write"]


async def test_call_tool_unknown_name_raises() -> None:
    """Dispatching a tool that doesn't exist surfaces as ``ToolError``."""

    from agent_harness.core.errors import ToolError

    ft = FilesystemTools(sandbox=FakeSandbox())
    with pytest.raises(ToolError):
        await ft.call_tool(ctx=None, call=ToolCall(id="c", name="nope", arguments={}))


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


async def test_read_returns_file_contents() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.txt": "hello"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "a.txt"})
    )
    assert _text(result) == "hello"
    assert result.error is None


async def test_read_truncates_at_line_limit() -> None:
    sb = FakeSandbox()
    body = "\n".join(str(i) for i in range(10))
    await _seed(sb, {"big.txt": body})
    ft = FilesystemTools(sandbox=sb, truncate_read_lines=3)

    result = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "big.txt"})
    )
    text = _text(result)
    # First three lines present.
    assert text.startswith("0\n1\n2")
    # Truncation marker appended.
    assert "truncated by FilesystemTools" in text
    # Line 5 (zero-indexed 5) is past the cap.
    assert "5" not in text.split("[")[0]


async def test_read_truncates_at_byte_limit_before_line_limit() -> None:
    """Whichever cap hits first wins (byte cap here)."""

    sb = FakeSandbox()
    payload = "x" * 200  # one long line, well over 50 bytes.
    await _seed(sb, {"a.txt": payload})
    ft = FilesystemTools(sandbox=sb, truncate_read_lines=2000, truncate_read_bytes=50)

    result = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "a.txt"})
    )
    text = _text(result)
    assert "truncated by FilesystemTools" in text
    body_only = text.split("\n[...")[0]
    assert len(body_only.encode("utf-8")) <= 50


async def test_read_with_line_range_returns_slice_only() -> None:
    sb = FakeSandbox()
    body = "\n".join(f"line {i}" for i in range(1, 11))
    await _seed(sb, {"a.txt": body})
    ft = FilesystemTools(sandbox=sb)

    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="r",
            name="read",
            arguments={"path": "a.txt", "line_start": 3, "line_end": 5},
        ),
    )
    text = _text(result)
    assert "line 3" in text
    assert "line 5" in text
    assert "line 1" not in text
    assert "line 6" not in text


async def test_read_propagates_file_not_found_as_errored_result() -> None:
    """Errors raised inside the body become an errored :class:`ToolResult`."""

    ft = FilesystemTools(sandbox=FakeSandbox())
    result = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "missing"})
    )
    assert result.error is not None
    assert "missing" in result.error or "FileNotFoundError" in result.error


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


async def test_write_routes_through_sandbox() -> None:
    sb = FakeSandbox()
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(id="w", name="write", arguments={"path": "new.txt", "content": "hi"}),
    )
    assert result.error is None
    assert await sb.read_file("new.txt") == "hi"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


async def test_edit_replaces_unique_match() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "foo = 1\nbar = 2\n"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="e",
            name="edit",
            arguments={"path": "a.py", "old_string": "foo = 1", "new_string": "foo = 42"},
        ),
    )
    assert result.error is None
    assert await sb.read_file("a.py") == "foo = 42\nbar = 2\n"


async def test_edit_rejects_multiple_matches_by_default() -> None:
    """FT7: ``expect_unique=True`` is the default; multi-match raises."""

    sb = FakeSandbox()
    await _seed(sb, {"a.py": "x = 1\nx = 1\n"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="e",
            name="edit",
            arguments={"path": "a.py", "old_string": "x = 1", "new_string": "y = 2"},
        ),
    )
    assert result.error is not None
    assert "matched 2 times" in result.error


async def test_edit_allows_first_match_when_expect_unique_false() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "x = 1\nx = 1\n"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="e",
            name="edit",
            arguments={
                "path": "a.py",
                "old_string": "x = 1",
                "new_string": "y = 2",
                "expect_unique": False,
            },
        ),
    )
    assert result.error is None
    assert await sb.read_file("a.py") == "y = 2\nx = 1\n"


async def test_edit_missing_match_raises() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "foo = 1\n"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="e",
            name="edit",
            arguments={"path": "a.py", "old_string": "bar", "new_string": "baz"},
        ),
    )
    assert result.error is not None
    assert "not found" in result.error


async def test_edit_empty_old_string_raises() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "x"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="e",
            name="edit",
            arguments={"path": "a.py", "old_string": "", "new_string": "y"},
        ),
    )
    assert result.error is not None


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


async def test_grep_returns_substring_matches() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "alpha\nBeta\ngamma\n"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(id="g", name="grep", arguments={"pattern": "beta", "path": "a.py"}),
    )
    payload = json.loads(_text(result))
    assert len(payload) == 1
    assert payload[0]["line_number"] == 2
    assert payload[0]["line"] == "Beta"


async def test_grep_regex_mode_with_case_sensitive() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "foo1\nfoo2\nbar3\n"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(
            id="g",
            name="grep",
            arguments={
                "pattern": r"foo\d",
                "path": "a.py",
                "regex": True,
                "case_sensitive": True,
            },
        ),
    )
    payload = json.loads(_text(result))
    assert {row["line"] for row in payload} == {"foo1", "foo2"}


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


async def test_glob_matches_by_purepath_semantics() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "1", "b.py": "2", "c.txt": "3"})
    ft = FilesystemTools(sandbox=sb, roots=[""])
    result = await ft.call_tool(
        ctx=None,
        call=ToolCall(id="g", name="glob", arguments={"pattern": "*.py", "root": ""}),
    )
    payload = json.loads(_text(result))
    assert sorted(payload) == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


async def test_list_dir_returns_entries() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"a.py": "1", "b.py": "2"})
    ft = FilesystemTools(sandbox=sb)
    result = await ft.call_tool(
        ctx=None, call=ToolCall(id="l", name="list_dir", arguments={"path": ""})
    )
    payload = json.loads(_text(result))
    names = sorted(row["name"] for row in payload)
    assert names == ["a.py", "b.py"]
    assert all(row["is_dir"] is False for row in payload)


# ---------------------------------------------------------------------------
# Ignore-set + multi-root gating
# ---------------------------------------------------------------------------


class _StubIgnore:
    """Minimal duck-typed ``IgnoreSet`` for tests."""

    def __init__(self, blocked: set[str]) -> None:
        self._blocked = blocked

    def matches(self, path: str) -> bool:
        return path in self._blocked


async def test_ignore_set_blocks_read() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"secret.env": "API_KEY=hunter2"})
    ft = FilesystemTools(sandbox=sb, ignore=_StubIgnore({"secret.env"}))
    result = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "secret.env"})
    )
    assert result.error is not None
    assert "excluded by the ignore-set" in result.error


async def test_roots_block_paths_outside_configured_subtrees() -> None:
    sb = FakeSandbox()
    await _seed(sb, {"src/main.py": "x", "etc/passwd": "y"})
    ft = FilesystemTools(sandbox=sb, roots=["src"])
    ok = await ft.call_tool(
        ctx=None,
        call=ToolCall(id="r", name="read", arguments={"path": "src/main.py"}),
    )
    assert ok.error is None and _text(ok) == "x"

    blocked = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "etc/passwd"})
    )
    assert blocked.error is not None
    assert "outside the configured roots" in blocked.error


# ---------------------------------------------------------------------------
# Sandbox swap (FT3)
# ---------------------------------------------------------------------------


async def test_sandbox_swap_retargets_every_tool() -> None:
    """FT3: swapping ``ft.sandbox`` rewires read/write transparently."""

    sb1 = FakeSandbox()
    sb2 = FakeSandbox()
    await sb1.write_file("a.txt", "from-sb1")
    await sb2.write_file("a.txt", "from-sb2")

    ft = FilesystemTools(sandbox=sb1)
    r1 = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "a.txt"})
    )
    assert _text(r1) == "from-sb1"

    ft.sandbox = sb2
    r2 = await ft.call_tool(
        ctx=None, call=ToolCall(id="r", name="read", arguments={"path": "a.txt"})
    )
    assert _text(r2) == "from-sb2"


# ---------------------------------------------------------------------------
# GrepMatch
# ---------------------------------------------------------------------------


def test_grep_match_as_dict_round_trip() -> None:
    m = GrepMatch(path="a.py", line_number=7, line="hit")
    assert m.as_dict() == {"path": "a.py", "line_number": 7, "line": "hit"}


# ---------------------------------------------------------------------------
# Tool schema sanity (the @tool decorator should infer descriptions correctly).
# ---------------------------------------------------------------------------


async def test_read_tool_schema_documents_parameters() -> None:
    ft = FilesystemTools(sandbox=FakeSandbox())
    tools = {t.name: t for t in await ft.list_tools(ctx=None)}
    schema: dict[str, Any] = tools["read"].schema
    props = schema["properties"]
    assert "path" in props
    assert props["path"].get("description")
    assert "line_start" in props
    assert "line_end" in props
    assert schema["required"] == ["path"]
