"""Unit tests for ``agent_harness.core.tools`` (types only)."""

from __future__ import annotations

import pytest

from agent_harness.core.errors import ConfigError
from agent_harness.core.models import TextBlock
from agent_harness.core.tools import ToolCall, ToolPolicy, ToolResult


def test_tool_call_dataclass_defaults() -> None:
    call = ToolCall(id="c1", name="search")
    assert call.arguments == {}
    assert call.id == "c1"


def test_tool_result_dataclass_defaults() -> None:
    block = TextBlock(text="ok")
    result = ToolResult(content=[block])
    assert result.error is None
    assert result.metadata == {}
    first = result.content[0]
    assert isinstance(first, TextBlock)
    assert first.text == "ok"


def test_tool_policy_default_values_match_spec() -> None:
    pol = ToolPolicy()
    assert pol.needs_approval is False
    assert pol.timeout_seconds is None
    assert pol.timeout_behavior == "error"
    assert pol.is_enabled is True
    assert pol.defer_loading is False
    assert pol.always_load is False
    assert pol.failure_error_function is None
    assert pol.tool_input_guardrails == []
    assert pol.tool_output_guardrails == []
    assert pol.is_read_only is False
    assert pol.is_destructive is False
    assert pol.is_concurrency_safe is False
    assert pol.max_result_size_chars is None
    assert pol.interrupt_behavior == "abort"


def test_tool_policy_accepts_callable_predicates() -> None:
    """``needs_approval`` / ``is_enabled`` / ``is_concurrency_safe`` may be
    either a static bool or a callable predicate."""

    def needs_approval(*args: object, **kwargs: object) -> bool:
        return True

    pol = ToolPolicy(
        needs_approval=needs_approval,
        is_enabled=lambda *_a, **_kw: False,
        is_concurrency_safe=lambda *_a, **_kw: True,
    )
    assert callable(pol.needs_approval)
    assert callable(pol.is_enabled)
    assert callable(pol.is_concurrency_safe)


def test_tool_policy_with_safety_flags_set() -> None:
    pol = ToolPolicy(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=50_000,
        timeout_seconds=30,
    )
    assert pol.is_read_only is True
    assert pol.is_concurrency_safe is True
    assert pol.max_result_size_chars == 50_000
    assert pol.timeout_seconds == 30


def test_tool_result_error_path() -> None:
    """An errored tool result carries content + non-None error."""

    result = ToolResult(content=[], error="execution failed", metadata={"trace_id": "abc"})
    assert result.error == "execution failed"
    assert result.metadata == {"trace_id": "abc"}


def test_tool_policy_defer_loading_only_is_allowed() -> None:
    """Either flag alone is fine; only the both-True combination is rejected."""

    deferred = ToolPolicy(defer_loading=True)
    assert deferred.defer_loading is True
    assert deferred.always_load is False

    forced = ToolPolicy(always_load=True)
    assert forced.always_load is True
    assert forced.defer_loading is False


def test_tool_policy_defer_loading_and_always_load_are_mutually_exclusive() -> None:
    """Setting both flags to ``True`` is a ``ConfigError`` (they're inverse
    intents; the spec lists them as siblings but the loader needs precedence
    enforced at construction time)."""

    with pytest.raises(ConfigError, match="mutually exclusive"):
        ToolPolicy(defer_loading=True, always_load=True)
