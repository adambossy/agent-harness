"""Unit tests for the ``@tool`` decorator + ``Tool`` dataclass."""

from __future__ import annotations

from typing import Literal

import pytest

from agent_harness.core.errors import ConfigError
from agent_harness.core.tools import Tool, ToolPolicy, tool


def test_tool_decorator_bare_form_returns_tool_dataclass() -> None:
    """``@tool`` (no parens) wraps a function and produces a ``Tool``."""

    @tool
    async def ping() -> str:
        """Ping. Returns pong."""
        return "pong"

    assert isinstance(ping, Tool)
    assert ping.name == "ping"
    assert "Ping" in ping.description
    assert callable(ping.fn)
    assert isinstance(ping.policy, ToolPolicy)


def test_tool_decorator_keyword_form_overrides_name_description() -> None:
    @tool(name="Echo", description="Custom blurb.")
    async def echo(text: str) -> str:
        """Auto-generated description that should be overridden."""
        return text

    assert echo.name == "Echo"
    assert echo.description == "Custom blurb."


def test_schema_built_from_type_hints_with_defaults_and_required() -> None:
    @tool
    async def fetch(url: str, timeout: float = 10.0) -> str:
        """Fetch the contents of a URL.

        Args:
            url: HTTPS URL to fetch.
            timeout: Seconds before giving up.
        """
        return ""

    props = fetch.schema["properties"]
    assert props["url"]["type"] == "string"
    assert props["url"]["description"] == "HTTPS URL to fetch."
    assert props["timeout"]["type"] == "number"
    assert props["timeout"]["default"] == 10.0
    assert props["timeout"]["description"] == "Seconds before giving up."
    assert fetch.schema["required"] == ["url"]


def test_docstring_summary_becomes_tool_description() -> None:
    @tool
    async def add(a: int, b: int) -> int:
        """Add two integers and return the sum.

        Args:
            a: first
            b: second
        """
        return a + b

    assert add.description == "Add two integers and return the sum."


def test_decorator_accepts_explicit_policy() -> None:
    policy = ToolPolicy(is_read_only=True, is_concurrency_safe=True, timeout_seconds=5.0)

    @tool(policy=policy)
    def grep(pattern: str) -> str:
        """Grep stub."""
        return ""

    assert grep.policy is policy
    assert grep.policy.is_read_only is True


def test_ctx_first_parameter_is_stripped_from_schema() -> None:
    """The harness convention treats ``ctx`` as implicit; the model shouldn't
    see it in the JSON schema."""

    @tool
    async def write(ctx: object, path: str, contents: str) -> None:
        """Write contents to path.

        Args:
            path: target file
            contents: data to write
        """

    props = write.schema["properties"]
    assert "ctx" not in props
    assert set(props.keys()) == {"path", "contents"}
    assert write.schema["required"] == ["path", "contents"]


def test_sync_function_is_accepted() -> None:
    """``@tool`` accepts both sync and async functions (loop wraps sync)."""

    @tool
    def upper(text: str) -> str:
        """Uppercase text."""
        return text.upper()

    assert upper.name == "upper"
    assert upper.fn("hi") == "HI"


def test_literal_and_complex_annotations_flow_through_schema() -> None:
    @tool
    async def emit(mode: Literal["a", "b"], values: list[int]) -> str:
        """Emit something.

        Args:
            mode: which mode.
            values: a list of ints.
        """
        return mode

    props = emit.schema["properties"]
    assert props["mode"]["enum"] == ["a", "b"]
    assert props["values"]["type"] == "array"
    assert props["values"]["items"]["type"] == "integer"


def test_missing_type_hint_raises_config_error() -> None:
    """A model-visible parameter without a type annotation is unschematizable."""

    async def bad(x) -> None:  # type: ignore[no-untyped-def]
        """Bad signature."""

    with pytest.raises(ConfigError, match="missing a type hint"):
        tool(bad)


def test_blank_docstring_yields_humanized_default_description() -> None:
    @tool
    async def get_user_profile(user_id: int) -> str:
        return ""

    # No summary docstring → description falls back to humanized name.
    assert get_user_profile.description == "get user profile"
    # Schema is still generated; parameter just has no description.
    assert "user_id" in get_user_profile.schema["properties"]
    assert "description" not in get_user_profile.schema["properties"]["user_id"]


def test_numpy_style_docstring_is_recognized() -> None:
    @tool
    async def divide(a: float, b: float) -> float:
        """Divide a by b.

        Parameters
        ----------
        a
            numerator
        b
            denominator
        """
        return a / b

    props = divide.schema["properties"]
    # Griffe auto-detects numpy style and yields parameter descriptions.
    assert props["a"]["description"].strip() == "numerator"
    assert props["b"]["description"].strip() == "denominator"
