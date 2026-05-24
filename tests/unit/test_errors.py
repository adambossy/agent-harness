"""Unit tests for ``agent_harness.core.errors``."""

from __future__ import annotations

import pytest

from agent_harness.core.errors import (
    AgentHarnessError,
    BudgetExceededError,
    ConfigError,
    ModelError,
    NotSupportedError,
    SandboxError,
    SandboxTimeoutError,
    SchemaError,
    ToolError,
)


def test_all_subclasses_inherit_base() -> None:
    """Every concrete error category subclasses :class:`AgentHarnessError`."""

    for cls in (
        ConfigError,
        ModelError,
        ToolError,
        SandboxError,
        SandboxTimeoutError,
        NotSupportedError,
        SchemaError,
        BudgetExceededError,
    ):
        assert issubclass(cls, AgentHarnessError)


def test_sandbox_timeout_is_a_sandbox_error() -> None:
    assert issubclass(SandboxTimeoutError, SandboxError)


def test_raise_carries_message() -> None:
    with pytest.raises(ConfigError, match="missing model"):
        raise ConfigError("missing model")


def test_context_is_copied_and_defaults_to_empty() -> None:
    ctx = {"agent": "demo"}
    err = ConfigError("bad", context=ctx)
    assert err.context == {"agent": "demo"}
    # Mutating the input dict must not leak into the captured context.
    ctx["agent"] = "changed"
    assert err.context == {"agent": "demo"}

    bare = ToolError("oops")
    assert bare.context == {}


def test_cause_chains_via_keyword() -> None:
    underlying = ValueError("root")
    err = ToolError("upstream failed", cause=underlying)
    assert err.cause is underlying
    assert err.__cause__ is underlying


def test_raise_from_works() -> None:
    underlying = RuntimeError("io")
    with pytest.raises(SandboxError) as exc:
        try:
            raise underlying
        except RuntimeError as e:
            raise SandboxError("exec failed", cause=e) from e
    assert exc.value.cause is underlying
    assert exc.value.__cause__ is underlying


def test_not_supported_signals_optional_capability() -> None:
    err = NotSupportedError("compaction unavailable", context={"model": "x"})
    assert isinstance(err, AgentHarnessError)
    assert err.context["model"] == "x"


def test_budget_exceeded_and_schema_error() -> None:
    assert isinstance(BudgetExceededError("over"), AgentHarnessError)
    assert isinstance(SchemaError("bad arg"), AgentHarnessError)


def test_message_attribute_matches_str() -> None:
    err = ToolError("boom")
    assert err.message == "boom"
    assert str(err) == "boom"
