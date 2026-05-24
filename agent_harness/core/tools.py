"""Tool-related types (Layer 0).

Layer 0 owns only the *data shapes* of the tool surface:

- :class:`ToolCall` — the model's request to invoke a tool.
- :class:`ToolResult` — the canonical result envelope (MCP-shaped content
  blocks plus an optional error / metadata).
- :class:`ToolPolicy` — declarative per-tool behavior the loop reads.

The :func:`@tool <agent_harness.core.tools.tool>` decorator, the ``Tool``
dataclass, and the :class:`Toolset` Protocol live in Layer 1 and are added by
a later wave; this module deliberately does NOT define them.

Example:
    >>> call = ToolCall(id="c1", name="search", arguments={"q": "hi"})
    >>> call.name
    'search'
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .models import ContentBlock


# ---------------------------------------------------------------------------
# ToolCall / ToolResult
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    """A single tool-call request from the model.

    Example:
        >>> ToolCall(id="c1", name="search", arguments={"q": "hi"}).name
        'search'
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """Canonical tool result. Content blocks mirror MCP's shape.

    Example:
        >>> from agent_harness.core.models import TextBlock
        >>> r = ToolResult(content=[TextBlock(text="ok")])
        >>> r.error is None
        True
    """

    content: list[ContentBlock]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ToolPolicy
# ---------------------------------------------------------------------------


class ToolPolicy(BaseModel):
    """Declarative per-tool behavior. The loop reads these and never branches
    on tool kind.

    Every field has a safe default — tools opt in to additional surface as
    needed. Predicates accept either a static ``bool`` or a callable so
    policies can be context-sensitive without subclassing.

    Example:
        >>> ToolPolicy(needs_approval=True, timeout_seconds=30).needs_approval
        True
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    needs_approval: bool | Callable[..., bool] = False
    """If True (or predicate returns True), the call pauses for approval."""

    timeout_seconds: float | None = None
    """Maximum runtime for a single invocation; ``None`` means no timeout."""

    timeout_behavior: Literal["error", "soft"] = "error"
    """``error`` raises :class:`SandboxTimeoutError`; ``soft`` surfaces a marker
    result to the model and lets it decide what to do."""

    is_enabled: bool | Callable[..., bool] = True
    """If False (or predicate returns False), the tool is hidden from the
    model for this run."""

    defer_loading: bool = False
    """If True, only ``name + description`` is sent to the model up front;
    the full schema is fetched on demand via the built-in ``ToolSearch``."""

    always_load: bool = False
    """If True, the tool is always included up front regardless of catalog
    size — the inverse of ``defer_loading`` for must-have tools."""

    failure_error_function: Callable[[Exception], str] | None = None
    """Optional transformer for exceptions raised inside the tool body."""

    tool_input_guardrails: list[Any] = Field(default_factory=list)
    """Sequence of Guardrail instances applied to arguments pre-dispatch.
    The :class:`Guardrail` type lives in Layer 1; ``Any`` here is a
    forward-compatible placeholder."""

    tool_output_guardrails: list[Any] = Field(default_factory=list)
    """Sequence of Guardrail instances applied to results post-dispatch."""

    is_read_only: bool = False
    """Hint: the tool does not mutate outside state. Implies
    ``is_concurrency_safe`` unless explicitly overridden by the loop."""

    is_destructive: bool = False
    """Hint: the tool performs hard-to-undo work (rm, drop table, ...)."""

    is_concurrency_safe: bool | Callable[..., bool] = False
    """If True (or predicate returns True), the loop may run this call in
    parallel with sibling safe calls in the same turn."""

    max_result_size_chars: int | None = None
    """Optional cap on raw result size; the loop truncates and inserts a
    marker pointing at the full content. ``None`` means no cap."""

    interrupt_behavior: Literal["abort", "complete", "ask"] = "abort"
    """What happens if the run is interrupted mid-execution:
    ``abort`` cancels the task, ``complete`` lets it finish, ``ask`` defers
    to the caller's interrupt handler."""

    @model_validator(mode="after")
    def _defer_and_always_load_are_mutually_exclusive(self) -> Self:
        """``defer_loading`` and ``always_load`` cannot both be ``True``.

        They describe inverse intents (defer until needed vs. force-load up
        front). The spec lists both with ``False`` defaults; this validator
        enforces the implied precedence at construction time so the Wave-2
        loader doesn't have to invent one.
        """
        if self.defer_loading and self.always_load:
            raise ConfigError(
                "ToolPolicy.defer_loading and ToolPolicy.always_load are "
                "mutually exclusive; set at most one to True.",
                context={
                    "defer_loading": self.defer_loading,
                    "always_load": self.always_load,
                },
            )
        return self
