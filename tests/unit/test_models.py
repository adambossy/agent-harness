"""Unit tests for ``agent_harness.core.models``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, get_args

import pytest

from agent_harness.core.errors import NotSupportedError
from agent_harness.core.models import (
    ContentBlock,
    ImageBlock,
    Message,
    Model,
    ModelCapabilities,
    ModelSettings,
    Provider,
    ProviderEvent,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_text_helper_concatenates_only_text_blocks() -> None:
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="Hello, "),
            ToolCallBlock(id="c1", name="noop", arguments={}),
            TextBlock(text="world!"),
            ThinkingBlock(text="(thought)"),
        ],
        timestamp=_ts(),
    )
    assert msg.text == "Hello, world!"


def test_has_tool_call_and_tool_calls_helpers() -> None:
    plain = Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())
    assert not plain.has_tool_call()
    assert plain.tool_calls == []

    with_call = Message(
        role="assistant",
        content=[
            TextBlock(text="ok"),
            ToolCallBlock(id="c1", name="f", arguments={"a": 1}),
            ToolCallBlock(id="c2", name="g", arguments={}),
        ],
        timestamp=_ts(),
    )
    assert with_call.has_tool_call()
    assert [c.id for c in with_call.tool_calls] == ["c1", "c2"]


def test_content_block_roundtrip_via_pydantic() -> None:
    """Every concrete block type survives JSON roundtripping."""

    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="hi"),
            ToolCallBlock(id="c1", name="search", arguments={"q": "x"}),
            ToolResultBlock(tool_call_id="c1", content="ok"),
            ImageBlock(mime_type="image/png", data="aGVsbG8="),
            ThinkingBlock(text="step 1"),
        ],
        timestamp=_ts(),
    )
    roundtripped = Message.model_validate_json(msg.model_dump_json())
    # The discriminated union should preserve concrete types.
    assert isinstance(roundtripped.content[0], TextBlock)
    assert isinstance(roundtripped.content[1], ToolCallBlock)
    assert isinstance(roundtripped.content[2], ToolResultBlock)
    assert isinstance(roundtripped.content[3], ImageBlock)
    assert isinstance(roundtripped.content[4], ThinkingBlock)
    assert roundtripped.content[1].arguments == {"q": "x"}


def test_content_block_union_membership() -> None:
    """The ContentBlock alias exposes all 5 concrete block types."""

    # ContentBlock is Annotated[Union[...], Field(discriminator=...)] —
    # ``get_args`` returns the Union, then we crack it open.
    union, _annotation = get_args(ContentBlock)
    members = set(get_args(union))
    assert members == {TextBlock, ToolCallBlock, ToolResultBlock, ImageBlock, ThinkingBlock}


def test_usage_addition_is_field_wise_and_pure() -> None:
    a = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=1, cache_write_tokens=2)
    b = Usage(input_tokens=3, output_tokens=4, cache_read_tokens=0, cache_write_tokens=7)
    c = a + b
    assert c.input_tokens == 13
    assert c.output_tokens == 9
    assert c.cache_read_tokens == 1
    assert c.cache_write_tokens == 9
    # Inputs are unmodified.
    assert a.input_tokens == 10


def test_usage_add_returns_notimplemented_for_other_types() -> None:
    """``Usage.__add__`` returns NotImplemented for unrelated types so
    Python falls back to the reflected operator."""

    result = Usage().__add__("nope")
    assert result is NotImplemented


def test_usage_defaults_are_zero() -> None:
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_read_tokens == 0
    assert u.cache_write_tokens == 0


def test_model_capabilities_defaults() -> None:
    caps = ModelCapabilities(context_window=200_000)
    assert caps.parallel_tool_calls is False
    assert caps.thinking is False
    assert caps.cache_control is False
    assert caps.vision is False
    assert caps.audio_input is False
    assert caps.audio_output is False
    assert caps.structured_output is True  # the only True-by-default
    assert caps.context_window == 200_000
    assert caps.max_output_tokens is None
    assert caps.supports_compaction is False


def test_model_capabilities_requires_context_window() -> None:
    with pytest.raises(ValueError):
        # Rationale for the ignore below: intentionally omitting the required
        # ``context_window`` argument to exercise Pydantic's missing-field
        # validation path at runtime.
        ModelCapabilities()  # type: ignore[call-arg]


def test_model_settings_defaults_are_all_none_or_empty() -> None:
    s = ModelSettings()
    assert s.temperature is None
    assert s.max_tokens is None
    assert s.top_p is None
    assert s.seed is None
    assert s.parallel_tool_calls is None
    assert s.thinking_budget is None
    assert s.extra == {}


def test_message_metadata_defaults_to_empty_dict() -> None:
    msg = Message(role="user", content=[], timestamp=_ts())
    assert msg.metadata == {}
    # mutate-default-free
    msg.metadata["k"] = "v"
    assert Message(role="user", content=[], timestamp=_ts()).metadata == {}


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        # Rationale for the ignore below: ``role`` is a Literal of the four
        # valid values; passing "wizard" intentionally violates that to
        # exercise Pydantic's rejection of out-of-set roles at runtime.
        Message(role="wizard", content=[], timestamp=_ts())  # type: ignore[arg-type]


class _DummyProvider:
    """Minimal Provider stand-in for runtime-checkable structural test."""

    name = "dummy"
    base_url: str | None = None

    async def request(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        yield ProviderEvent(kind="raw")


class _DummyModel:
    name = "dummy-model"
    provider = _DummyProvider()
    capabilities = ModelCapabilities(context_window=8_000)

    async def request(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> AsyncIterator[Any]:
        yield None

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        raise NotSupportedError("nope")


def test_provider_protocol_runtime_check() -> None:
    """Anything with ``name``, ``base_url``, and ``request`` satisfies Provider."""

    assert isinstance(_DummyProvider(), Provider)


def test_model_protocol_runtime_check() -> None:
    assert isinstance(_DummyModel(), Model)


async def test_default_compact_messages_raises_not_supported() -> None:
    model = _DummyModel()
    with pytest.raises(NotSupportedError):
        await model.compact_messages([])


def test_provider_event_accepts_extra_fields() -> None:
    """ProviderEvent is the Layer-0 placeholder; Wave-2 providers extend it."""

    # Rationale for the ignore below: ``payload`` is not a declared field on
    # the Layer-0 ``ProviderEvent`` placeholder, but
    # ``model_config = ConfigDict(extra="allow")`` accepts arbitrary extras
    # at runtime — exactly the contract this test asserts.
    ev = ProviderEvent(kind="raw_chunk", payload={"x": 1})  # type: ignore[call-arg]
    assert ev.kind == "raw_chunk"
