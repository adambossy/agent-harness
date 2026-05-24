"""Canonical exception hierarchy.

All errors raised by core / Layer 1-2 components derive from
:class:`AgentHarnessError`. Each carries an optional ``cause`` (the
underlying exception) and a ``context`` dict for structured diagnostics.

Example:
    >>> try:
    ...     raise ConfigError("missing model", context={"agent": "demo"})
    ... except AgentHarnessError as exc:
    ...     exc.context["agent"]
    'demo'
"""

from __future__ import annotations

from typing import Any


class AgentHarnessError(Exception):
    """Base class for every error raised by the agent harness.

    Example:
        >>> AgentHarnessError("boom", context={"k": 1}).context["k"]
        1
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.context: dict[str, Any] = dict(context) if context is not None else {}
        if cause is not None:
            # Preserve ``raise X from Y`` semantics for keyword usage too.
            self.__cause__ = cause


class ConfigError(AgentHarnessError):
    """Invalid agent / component configuration.

    Example:
        >>> isinstance(ConfigError("missing model"), AgentHarnessError)
        True
    """


class ModelError(AgentHarnessError):
    """The model returned malformed output or violated its contract.

    Example:
        >>> isinstance(ModelError("bad json"), AgentHarnessError)
        True
    """


class ToolError(AgentHarnessError):
    """Tool execution failed.

    Example:
        >>> isinstance(ToolError("oops"), AgentHarnessError)
        True
    """


class SandboxError(AgentHarnessError):
    """A sandbox operation failed.

    Example:
        >>> isinstance(SandboxError("exec failed"), AgentHarnessError)
        True
    """


class SandboxTimeoutError(SandboxError):
    """A sandbox operation exceeded its timeout (Flue's primary contract).

    Example:
        >>> isinstance(SandboxTimeoutError("30s"), SandboxError)
        True
    """


class NotSupportedError(AgentHarnessError):
    """The requested capability is not supported by this component.

    Example:
        >>> isinstance(NotSupportedError("no compaction"), AgentHarnessError)
        True
    """


class SchemaError(AgentHarnessError):
    """Schema validation failed (tool args, structured output, snapshot).

    Example:
        >>> isinstance(SchemaError("missing field"), AgentHarnessError)
        True
    """


class BudgetExceededError(AgentHarnessError):
    """A token / cost budget was exceeded.

    Example:
        >>> isinstance(BudgetExceededError("100k"), AgentHarnessError)
        True
    """


class BusClosedError(AgentHarnessError):
    """Raised when an :class:`EventBus` is used after ``close()``.

    Distinct from a generic ``RuntimeError`` so callers can ``except`` the
    closed-bus case without swallowing unrelated errors.

    Example:
        >>> isinstance(BusClosedError("bus closed"), AgentHarnessError)
        True
    """
