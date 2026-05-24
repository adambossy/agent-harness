"""Unit tests for the built-in ``ToolSearch`` standard tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Iterator
from typing import Any, cast

import pytest

from agent_harness.core.models import TextBlock
from agent_harness.core.tools import Tool, ToolCall, ToolPolicy, tool
from agent_harness.core.toolsets import (
    TOOL_SEARCH,
    StaticToolset,
    clear_deferred_tools,
    register_deferred_tools,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Each test starts with an empty deferred-tools registry."""

    clear_deferred_tools()
    try:
        yield
    finally:
        clear_deferred_tools()


@tool(policy=ToolPolicy(defer_loading=True))
async def fetch_url(url: str) -> str:
    """Fetch the contents of a URL.

    Args:
        url: HTTPS URL.
    """
    return ""


@tool(policy=ToolPolicy(defer_loading=True))
async def search_repos(query: str) -> list[str]:
    """Search GitHub repositories.

    Args:
        query: free text.
    """
    return []


@tool(policy=ToolPolicy(defer_loading=True))
async def delete_user(user_id: int) -> None:
    """Delete a user by id."""


async def _run_search(**kwargs: Any) -> list[dict[str, Any]]:
    """Invoke ``ToolSearch.fn`` directly. The decorator types the function
    as the broad ``ToolFn`` union (sync-or-async, arbitrary args) since
    ``Tool`` is a data carrier; this helper localises the cast."""

    raw = cast(Awaitable[list[dict[str, Any]]], TOOL_SEARCH.fn(**kwargs))
    return await raw


def test_tool_search_is_itself_a_tool_with_safe_policy() -> None:
    assert isinstance(TOOL_SEARCH, Tool)
    assert TOOL_SEARCH.name == "ToolSearch"
    assert TOOL_SEARCH.policy.is_read_only is True
    assert TOOL_SEARCH.policy.is_concurrency_safe is True
    # ``ToolSearch`` should always load so the model can discover deferred tools.
    assert TOOL_SEARCH.policy.always_load is True


def test_tool_search_schema_has_expected_inputs() -> None:
    props = TOOL_SEARCH.schema["properties"]
    assert props["query"]["type"] == "string"
    assert props["max_results"]["type"] == "integer"
    assert props["max_results"]["default"] == 5
    assert TOOL_SEARCH.schema["required"] == ["query"]


async def test_tool_search_returns_empty_when_registry_empty() -> None:
    out = await _run_search(query="anything")
    assert out == []


async def test_tool_search_returns_matching_schemas_by_name() -> None:
    register_deferred_tools([fetch_url, search_repos, delete_user])
    hits = await _run_search(query="search")
    assert isinstance(hits, list)
    names = [h["name"] for h in hits]
    # 'search_repos' is a name-substring hit (score 3); others miss / weak hit.
    assert names[0] == "search_repos"
    # Every entry carries name + description + schema.
    assert {"name", "description", "schema"} <= set(hits[0].keys())
    # And the schema is round-trippable JSON.
    json.dumps(hits[0]["schema"])


async def test_tool_search_matches_description_word_boundary() -> None:
    register_deferred_tools([fetch_url, search_repos, delete_user])
    hits = await _run_search(query="url")
    names = [h["name"] for h in hits]
    # 'fetch_url' has 'URL' in its description (and name) -> top hit.
    assert names[0] == "fetch_url"


async def test_tool_search_respects_max_results() -> None:
    register_deferred_tools([fetch_url, search_repos, delete_user])
    hits = await _run_search(query="a", max_results=1)
    assert len(hits) <= 1


async def test_tool_search_invocable_via_static_toolset() -> None:
    """``ToolSearch`` participates as a regular tool in any toolset."""

    register_deferred_tools([fetch_url, search_repos])
    ts = StaticToolset(name="builtin", tools=[TOOL_SEARCH])
    result = await ts.call_tool(
        ctx=None, call=ToolCall(id="c1", name="ToolSearch", arguments={"query": "search"})
    )
    assert result.error is None
    # The wrapped result content is text (str repr of the list); make sure it's
    # not empty and references the matched tool name.
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert "search_repos" in block.text


def test_register_deferred_tools_is_idempotent() -> None:
    register_deferred_tools([fetch_url])
    register_deferred_tools([fetch_url, fetch_url])
    # Internal accounting: a second call doesn't double-register.
    # Use the public surface (``ToolSearch``) to assert.
    hits = asyncio.run(_run_search(query="fetch"))
    assert len(hits) == 1
