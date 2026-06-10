"""Tool-related types (Layer 0 + Layer 1).

Layer 0 owns the *data shapes*:

- :class:`ToolCall` — the model's request to invoke a tool.
- :class:`ToolResult` — the canonical result envelope (MCP-shaped content
  blocks plus an optional error / metadata).
- :class:`ToolPolicy` — declarative per-tool behavior the loop reads.

Layer 1 adds the ``Tool`` dataclass and the :func:`@tool
<agent_harness.core.tools.tool>` decorator that builds one from a plain
function via type-hints + Griffe docstring parsing. The :class:`Toolset`
Protocol and built-in wrappers live in :mod:`agent_harness.core.toolsets`.

Example:
    >>> call = ToolCall(id="c1", name="search", arguments={"q": "hi"})
    >>> call.name
    'search'
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast, get_type_hints, overload

import griffe
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

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

    ``structured_content`` mirrors MCP's ``structuredContent`` (spec rev
    2025-11-25): when a tool returns structured data (dict/list/scalar JSON
    values) it is preserved here verbatim, while a JSON-serialized copy is
    also placed in ``content`` as a ``TextBlock`` for the LLM and any
    legacy consumer that only reads ``content`` ("For backwards
    compatibility, a tool that returns structured content SHOULD also
    return the serialized JSON in a TextContent block.").

    Consumers that want the structured value (e.g. UIs, programmatic
    clients) read ``structured_content`` first and fall back to
    ``content`` only when it is ``None``.

    Example:
        >>> from agent_harness.core.models import TextBlock
        >>> r = ToolResult(content=[TextBlock(text="ok")])
        >>> r.error is None and r.structured_content is None
        True
    """

    content: list[ContentBlock]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    structured_content: Any | None = None


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


# ---------------------------------------------------------------------------
# Tool dataclass + @tool decorator (Layer 1)
# ---------------------------------------------------------------------------

# A tool's body may be sync or async, with arbitrary arg/return types. The
# decorator preserves the function on the ``Tool`` dataclass; the loop is
# responsible for awaiting if needed. Typed ``Any`` because the function shape
# is user-defined and Layer 0 cannot import :class:`RunContext` (Layer 3).
ToolFn = Callable[..., Awaitable[Any]] | Callable[..., Any]

# ``F`` keeps the decorator generic across sync/async callables.
F = TypeVar("F", bound=Callable[..., Any])

# Parameter names the harness treats as the implicit run-context argument and
# strips from the generated JSON schema (the model never supplies these).
_CTX_PARAM_NAMES: frozenset[str] = frozenset({"ctx", "context", "run_context"})


@dataclass(slots=True)
class Tool:
    """A callable tool: name + description + JSON schema + policy + function.

    A Tool is *just data*. There are no methods other than the function itself;
    dispatch happens through a :class:`~agent_harness.core.toolsets.Toolset`.
    The loop reads :attr:`policy` to make scheduling, approval, and result-
    capping decisions — it never branches on tool kind.

    Example:
        >>> async def echo(text: str) -> str:
        ...     return text
        >>> t = Tool(
        ...     name="echo",
        ...     description="Echo the given text.",
        ...     schema={"type": "object", "properties": {"text": {"type": "string"}}},
        ...     policy=ToolPolicy(is_read_only=True),
        ...     fn=echo,
        ... )
        >>> t.name
        'echo'
    """

    name: str
    description: str
    schema: dict[str, Any]
    policy: ToolPolicy
    fn: ToolFn


def _humanize(snake: str) -> str:
    """Turn ``snake_case`` into ``"snake case"`` for default descriptions."""
    return snake.replace("_", " ")


def _parse_param_descriptions(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Run Griffe's auto-parser over ``docstring``.

    Returns ``(summary, param_descriptions)``. The summary is the first
    text section (typically the one-line summary, but any preceding free
    text is concatenated). Missing-annotation warnings are silenced — they
    fire because the Docstring isn't bound to a real Griffe object, but we
    pull type info from the function signature, not the docstring.

    Example:
        >>> summary, params = _parse_param_descriptions("Hi.\\n\\nArgs:\\n    x: an integer\\n")
        >>> summary
        'Hi.'
        >>> params
        {'x': 'an integer'}
    """
    if not docstring:
        return "", {}
    ds = griffe.Docstring(inspect.cleandoc(docstring))
    # Suppress Griffe's "No type or annotation for parameter X" warnings.
    # They're noise here: we get types from the signature, not the docstring.
    griffe_logger = logging.getLogger("griffe")
    prev_level = griffe_logger.level
    griffe_logger.setLevel(logging.ERROR)
    try:
        sections = griffe.parse_auto(ds)
    finally:
        griffe_logger.setLevel(prev_level)
    text_parts: list[str] = []
    param_descs: dict[str, str] = {}
    for section in sections:
        if section.kind == griffe.DocstringSectionKind.text:
            value = section.value
            text = value if isinstance(value, str) else getattr(value, "value", "")
            if text:
                text_parts.append(text)
        elif section.kind == griffe.DocstringSectionKind.parameters:
            for p in section.value:
                if p.description:
                    param_descs[p.name] = p.description
    summary = "\n\n".join(text_parts).strip()
    return summary, param_descs


def _build_schema(fn: Callable[..., Any], param_descs: dict[str, str]) -> dict[str, Any]:
    """Construct a JSON Schema for ``fn``'s call arguments.

    Uses Pydantic ``create_model`` so list / dict / Optional / Union / Literal
    flow through naturally. Skips the conventional ``ctx`` / ``context`` /
    ``run_context`` first-parameter (Layer 3 feeds it implicitly; the model
    never supplies it).

    Example:
        >>> def f(x: int, y: str = "hi") -> str: ...
        >>> _build_schema(f, {"x": "an int"})["properties"]["x"]["description"]
        'an int'
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except (NameError, AttributeError, TypeError):
        # Forward refs or stringified annotations that can't be resolved at
        # decoration time fall back to whatever is on the signature object.
        # NameError: unresolved forward ref. AttributeError: stringified
        # annotation referencing a missing attribute. TypeError: callable
        # with unhintable signature (e.g. C-implemented).
        hints = {}
    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs aren't expressible in JSON-Schema cleanly.
            continue
        if name in _CTX_PARAM_NAMES:
            continue
        annotation: Any = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise ConfigError(
                f"Tool parameter {name!r} on {fn.__qualname__} is missing a type "
                f"hint; @tool requires annotations on every model-visible argument.",
                context={"function": fn.__qualname__, "parameter": name},
            )
        default: Any = ... if param.default is inspect.Parameter.empty else param.default
        description = param_descs.get(name)
        field_info = Field(default=default, description=description)
        fields[name] = (annotation, field_info)
    # ``model_name`` must be a valid Python identifier; mangle dotted qualnames.
    safe_name = re.sub(r"\W", "_", fn.__qualname__) + "Args"
    model_cls = create_model(safe_name, __base__=BaseModel, **fields)  # type: ignore[call-overload]
    schema = cast(dict[str, Any], model_cls.model_json_schema())
    # Drop Pydantic's auto-generated title noise; harness consumers don't need it.
    schema.pop("title", None)
    return schema


@overload
def tool(fn: F, /) -> Tool: ...  # TypeVar F shared across overloads


@overload
def tool(  # TypeVar F shared across overloads
    *,
    name: str | None = ...,
    description: str | None = ...,
    policy: ToolPolicy | None = ...,
) -> Callable[[F], Tool]: ...


def tool(  # using TypeVar (F) is clearer than PEP 695 across the three overloads
    fn: F | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    policy: ToolPolicy | None = None,
) -> Tool | Callable[[F], Tool]:
    """Decorator that wraps a plain function (sync or async) as a :class:`Tool`.

    The decorator pulls the input schema from the function's type hints (via
    Pydantic ``create_model``) and the parameter / function descriptions from
    the docstring (via Griffe, which auto-detects Google / numpy / sphinx
    style). The function itself is preserved on :attr:`Tool.fn` — dispatch
    happens through a :class:`~agent_harness.core.toolsets.Toolset`.

    Example:
        >>> @tool(policy=ToolPolicy(is_read_only=True))
        ... async def echo(text: str) -> str:
        ...     '''Echo the given text.
        ...
        ...     Args:
        ...         text: The string to echo back.
        ...     '''
        ...     return text
        >>> echo.name, echo.policy.is_read_only
        ('echo', True)
        >>> echo.schema["properties"]["text"]["description"]
        'The string to echo back.'

    Usable bare (``@tool``) or with keyword args (``@tool(policy=…)``).
    """

    def _wrap(target: Callable[..., Any]) -> Tool:
        if not callable(target):
            raise ConfigError(
                "@tool may only be applied to a callable.",
                context={"target": repr(target)},
            )
        raw_doc = inspect.getdoc(target)
        summary, param_descs = _parse_param_descriptions(raw_doc)
        resolved_name: str = name or getattr(target, "__name__", None) or "tool"
        resolved_description: str = description or summary or _humanize(resolved_name)
        schema = _build_schema(target, param_descs)
        return Tool(
            name=resolved_name,
            description=resolved_description,
            schema=schema,
            policy=policy or ToolPolicy(),
            fn=target,
        )

    if fn is not None:
        # Bare ``@tool`` form: ``fn`` is the decorated function.
        return _wrap(fn)
    return _wrap
