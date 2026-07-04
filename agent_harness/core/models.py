"""Canonical model / provider types and Protocols.

Owns the *types* that flow across the model boundary:

- :class:`Message` + the :data:`ContentBlock` discriminated union.
- :class:`Usage` (accumulating, summable via ``+``).
- :class:`ModelCapabilities` / :class:`ModelSettings` (declarative).
- :class:`Model` / :class:`Provider` Protocols (pydantic-ai's two-axis split).

Concrete adapters live under ``agent_harness.providers``; core knows the
shape only.

Example:
    >>> from datetime import datetime, timezone
    >>> Message(
    ...     role="user",
    ...     content=[TextBlock(text="hi")],
    ...     timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ... ).text
    'hi'
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from types import NotImplementedType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .errors import NotSupportedError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from collections.abc import AsyncIterator


# --- Content blocks ----------------------------------------------------------


class TextBlock(BaseModel):
    """Plain text.

    Example:
        >>> TextBlock(text="hello").text
        'hello'
    """

    type: Literal["text"] = "text"
    text: str


class ToolCallBlock(BaseModel):
    """Assistant-emitted tool-call request.

    Example:
        >>> ToolCallBlock(id="c1", name="search", arguments={"q": "x"}).name
        'search'
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """A tool result referenced back to its originating call.

    Example:
        >>> ToolResultBlock(tool_call_id="c1", content="ok").content
        'ok'
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str | list[Any]


class ImageBlock(BaseModel):
    """An image block; either ``data`` (base64) or ``url`` should be set.

    Example:
        >>> ImageBlock(data="aGVsbG8=", mime_type="image/png").mime_type
        'image/png'
    """

    type: Literal["image"] = "image"
    data: str | None = None
    mime_type: str
    url: str | None = None


class ThinkingBlock(BaseModel):
    """Extended-thinking trace from a reasoning-capable model.

    Example:
        >>> ThinkingBlock(text="step 1").text
        'step 1'
    """

    type: Literal["thinking"] = "thinking"
    text: str


ContentBlock = Annotated[
    TextBlock | ToolCallBlock | ToolResultBlock | ImageBlock | ThinkingBlock,
    Field(discriminator="type"),
]
"""Discriminated union of every content block kind the harness supports."""


# --- Message -----------------------------------------------------------------


class Message(BaseModel):
    """A canonical conversation message — provider-format-independent.

    Example:
        >>> from datetime import datetime, timezone
        >>> m = Message(
        ...     role="assistant",
        ...     content=[TextBlock(text="hi")],
        ...     timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ... )
        >>> m.text, m.has_tool_call()
        ('hi', False)
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentBlock] = Field(default_factory=list)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenate every :class:`TextBlock`'s text in order."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def has_tool_call(self) -> bool:
        """Return True iff any content block is a :class:`ToolCallBlock`."""
        return any(isinstance(b, ToolCallBlock) for b in self.content)

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        """Every :class:`ToolCallBlock` in this message, preserving order."""
        return [b for b in self.content if isinstance(b, ToolCallBlock)]


# --- Usage -------------------------------------------------------------------


class Usage(BaseModel):
    """Token / cache counters; sums via ``+``.

    Example:
        >>> (Usage(input_tokens=10) + Usage(input_tokens=2)).input_tokens
        12
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: object) -> Usage | NotImplementedType:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


# --- Cost --------------------------------------------------------------------


class Cost(BaseModel):
    """Per-category monetary cost of a :class:`Usage`, in a single currency.

    The harness never invents prices: a :data:`UsagePricer` supplied by the
    host turns tokens into a ``Cost``. Amounts are plain floats in whatever
    currency the pricer used (USD by convention). Sums via ``+`` so a run's
    total is ``sum(costs, Cost())``.

    Example:
        >>> (Cost(input_cost=0.01) + Cost(output_cost=0.02)).total
        0.03
    """

    model_config = ConfigDict(extra="forbid")

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0

    @property
    def total(self) -> float:
        """Sum of every cost category."""
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    def __add__(self, other: object) -> Cost | NotImplementedType:
        if not isinstance(other, Cost):
            return NotImplemented
        return Cost(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            cache_read_cost=self.cache_read_cost + other.cache_read_cost,
            cache_write_cost=self.cache_write_cost + other.cache_write_cost,
        )


UsagePricer = Callable[[str, Usage], Cost]
"""Host-supplied hook mapping ``(model_name, usage) -> Cost``.

Injected on :class:`~agent_harness.core.agent.Agent`; the loop calls it after
each model response and publishes the resulting cost as a
:class:`~agent_harness.core.events.ModelUsage` event. The harness ships no
price table of its own — see :mod:`agent_harness.usage.counting` for a
batteries-included implementation the host can construct and pass in.
"""


# --- Capabilities + settings -------------------------------------------------


class ModelCapabilities(BaseModel):
    """Declarative statement of what a model can do (the loop reads these,
    never the model's name).

    Example:
        >>> ModelCapabilities(context_window=200_000).parallel_tool_calls
        False
    """

    model_config = ConfigDict(extra="forbid")

    parallel_tool_calls: bool = False
    thinking: bool = False
    cache_control: bool = False
    vision: bool = False
    audio_input: bool = False
    audio_output: bool = False
    structured_output: bool = True
    context_window: int
    max_output_tokens: int | None = None
    supports_compaction: bool = False


class ModelSettings(BaseModel):
    """Per-call knobs; capability-gated values are silently dropped when the
    model can't honor them.

    Example:
        >>> ModelSettings(temperature=0.2).temperature
        0.2
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    parallel_tool_calls: bool | None = None
    thinking_budget: int | None = None
    # Provider-native ("built-in") tools appended to the wire tools list
    # alongside the function-declaration tools — e.g. web search:
    # OpenAI ``{"type": "web_search"}`` or Google ``{"google_search": {}}``.
    # These coexist with custom @tool functions; they do not replace them.
    builtin_tools: list[Any] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


# --- Provider + Model Protocols ---------------------------------------------


class ProviderEvent(BaseModel):
    """Provider-level transport event; refined by Wave-2 providers.

    Example:
        >>> ProviderEvent(kind="raw").kind
        'raw'
    """

    model_config = ConfigDict(extra="allow")

    kind: str


@runtime_checkable
class Provider(Protocol):
    """Auth + transport. Knows *how* to talk to a remote API, not *what* the
    API's wire shape looks like (that's :class:`Model`'s job).

    ``request`` is declared ``async def`` to match the spec
    (``model-and-provider.md``). Concrete implementations are typically
    async-generator functions (``async def`` + ``yield``); satisfying the
    Protocol at runtime requires only structural shape (``name``,
    ``base_url``, ``request``).

    Example:
        >>> class _StubProvider:
        ...     name = "stub"
        ...     base_url: str | None = None
        ...
        ...     async def request(self, payload, *, stream=False, timeout=None):
        ...         yield ProviderEvent(kind="raw")
        >>> isinstance(_StubProvider(), Provider)
        True
    """

    name: str
    base_url: str | None

    async def request(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...


@runtime_checkable
class Model(Protocol):
    """Canonical model API shape; the loop calls ``request`` with canonical
    types and consumes a typed event stream.

    ``request`` is declared ``async def`` to match the spec
    (``model-and-provider.md``). Wave-2 adapters implement it as an
    async-generator function.

    Example:
        >>> class _StubModel:
        ...     name = "stub-model"
        ...     provider = None  # any Provider; structural check ignores type
        ...     capabilities = ModelCapabilities(context_window=8000)
        ...
        ...     async def request(self, messages, tools, settings):
        ...         yield None
        ...
        ...     async def compact_messages(self, msgs):
        ...         return msgs
        >>> isinstance(_StubModel(), Model)
        True
    """

    name: str
    provider: Provider
    capabilities: ModelCapabilities

    async def request(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> AsyncIterator[Any]: ...

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        """Optional provider-side compaction. Default declines.

        Note: this body only fires for subclasses that explicitly call
        ``super().compact_messages(...)``. A class that *omits* the method
        will fail the structural ``isinstance(x, Model)`` check rather than
        inheriting this default — Protocols are structural, not nominal.
        """
        raise NotSupportedError("compact_messages is not supported by this model")


# Resolve forward refs in the discriminated content union.
Message.model_rebuild()
