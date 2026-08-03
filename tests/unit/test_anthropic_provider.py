"""Unit tests for ``agent_harness.providers.anthropic``.

The ``anthropic`` SDK is an optional dependency; these tests mock it
entirely so they pass regardless of whether the SDK is installed in the
worktree's venv.
"""

from __future__ import annotations

import sys
import types
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_harness.core.errors import ConfigError, NotSupportedError
from agent_harness.core.events import (
    MessageDelta,
    ModelEnd,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallEnd,
)
from agent_harness.core.models import (
    Message,
    Model,
    ModelSettings,
    Provider,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from agent_harness.providers import anthropic as anthropic_mod
from agent_harness.providers.anthropic import (
    _CAPABILITIES_BY_MODEL,
    _CAPS_OPUS_4_7,
    EFFORT_LEVELS_BY_MODEL,
    OPUS_4_7,
    OPUS_4_8,
    OPUS_5,
    SONNET_4_6,
    SONNET_5,
    AnthropicMessagesModel,
    AnthropicProvider,
    _build_final_message,
    _parse_json_args,
    _usage_from,
    supported_effort_levels,
)
from tests.anthropic_sdk_fakes import (
    _build_fake_client,
    _FakeStreamEvent,
    _ts,
)

# --- tests: provider construction ------------------------------------------


def test_provider_accepts_injected_client() -> None:
    client = MagicMock()
    p = AnthropicProvider(client=client)
    assert p.client is client
    assert p.name == "anthropic"


def test_provider_raises_not_supported_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No injected client + no SDK module ⇒ NotSupportedError."""

    # Force the lazy import path to fail.
    sentinel = object()
    monkeypatch.setitem(sys.modules, "anthropic", sentinel)
    monkeypatch.delitem(sys.modules, "anthropic")

    def _raise_import(*_a: Any, **_kw: Any) -> Any:
        raise ImportError("no anthropic")

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _raise_import  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    with pytest.raises((NotSupportedError, ImportError)):
        # Either path is acceptable: NotSupportedError if _require_sdk fails,
        # or ImportError if AsyncAnthropic constructor surfaces the failure.
        AnthropicProvider(api_key="k")


# --- tests: protocol conformance -------------------------------------------


def test_model_and_provider_satisfy_protocols() -> None:
    p = AnthropicProvider(client=MagicMock())
    m = AnthropicMessagesModel(provider=p)
    # Cast through ``object`` so mypy doesn't try to prove subclass-method
    # compatibility for async-generator-vs-coroutine return types; the
    # runtime structural check is the real contract here.
    assert isinstance(cast(object, p), Provider)
    assert isinstance(cast(object, m), Model)
    assert m.name == OPUS_4_7
    assert m.capabilities.parallel_tool_calls is True
    assert m.capabilities.thinking is True
    assert m.capabilities.cache_control is True


# --- tests: payload translation --------------------------------------------


def test_messages_to_wire_extracts_system_and_blocks() -> None:
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")], timestamp=_ts()),
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(
            role="assistant",
            content=[
                TextBlock(text="here"),
                ToolCallBlock(id="c1", name="search", arguments={"q": "x"}),
            ],
            timestamp=_ts(),
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_call_id="c1", content="ok")],
            timestamp=_ts(),
        ),
    ]
    system, wire = AnthropicMessagesModel._messages_to_wire(msgs)
    assert system == "be brief"
    assert wire[0]["role"] == "user"
    assert wire[1]["role"] == "assistant"
    # The tool message becomes a user message carrying a tool_result block.
    assert wire[2]["role"] == "user"
    assert wire[2]["content"][0]["type"] == "tool_result"
    assert wire[2]["content"][0]["tool_use_id"] == "c1"


def test_build_payload_respects_settings_and_capabilities() -> None:
    p = AnthropicProvider(client=MagicMock())
    m = AnthropicMessagesModel(provider=p)
    msgs = [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())]
    settings = ModelSettings(
        temperature=0.3,
        max_tokens=512,
        top_p=0.95,
        parallel_tool_calls=False,
        thinking_budget=2_000,
        extra={"metadata": {"trace_id": "t1"}},
    )
    tools = [{"name": "search", "description": "", "input_schema": {"type": "object"}}]
    payload = m._build_payload(msgs, tools, settings)
    assert payload["model"] == OPUS_4_7
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.95
    assert payload["tools"] == tools
    assert payload["tool_choice"]["disable_parallel_tool_use"] is True
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["metadata"] == {"trace_id": "t1"}


def test_build_payload_uses_the_budget_shape_for_pre_4_7_models() -> None:
    """Older Claude models still take budget_tokens; only 4.7+ rejects it."""
    caps = _CAPS_OPUS_4_7.model_copy(update={"adaptive_thinking": False})
    m = AnthropicMessagesModel(provider=AnthropicProvider(client=MagicMock()), capabilities=caps)
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(thinking_budget=2_000),
    )
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2_000}


def test_build_payload_emits_adaptive_shape_without_a_budget() -> None:
    """Claude 4.7+ 400s on budget_tokens, and display must be opted in.

    The API default is display="omitted", which streams thinking blocks whose
    text is empty — reasoning that never reaches the caller.
    """
    m = AnthropicMessagesModel(provider=AnthropicProvider(client=MagicMock()))
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(thinking_budget=2_000),
    )
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "budget_tokens" not in payload["thinking"]


async def test_signature_only_thinking_block_survives_with_a_close_event() -> None:
    """The display="omitted" shape: a signature arrives, thinking text never does."""
    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_1")),
        _FakeStreamEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeStreamEvent(type="thinking", thinking="", signature=""),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="signature_delta", signature="sig-only"),
        ),
        _FakeStreamEvent(type="content_block_stop", index=0),
        _FakeStreamEvent(type="message_stop", message=None),
    ]
    model = AnthropicMessagesModel(provider=AnthropicProvider(client=_build_fake_client(events)))
    out = await _collect(model)
    # The block must survive: dropping it loses the signature the next turn replays.
    final = next(e for e in out if type(e).__name__ == "MessageEnd").final
    thoughts = [b for b in final.content if isinstance(b, ThinkingBlock)]
    assert [(b.text, b.signature) for b in thoughts] == [("", "sig-only")]
    # ...and a started thinking region must still be closed for consumers.
    assert len([e for e in out if isinstance(e, ThinkingStart)]) == 1
    assert len([e for e in out if isinstance(e, ThinkingEnd)]) == 1


def test_build_payload_drops_thinking_when_capability_off() -> None:
    p = AnthropicProvider(client=MagicMock())
    m = AnthropicMessagesModel(provider=p)
    m.capabilities = m.capabilities.model_copy(update={"thinking": False})
    msgs = [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())]
    payload = m._build_payload(msgs, [], ModelSettings(thinking_budget=1_000))
    assert "thinking" not in payload


def test_message_metadata_carries_cache_control() -> None:
    """``Message.metadata`` is the canonical carry-through for cache_control."""
    msg = Message(
        role="user",
        content=[TextBlock(text="big context")],
        timestamp=_ts(),
        metadata={"cache_control": {"type": "ephemeral"}},
    )
    _system, wire = AnthropicMessagesModel._messages_to_wire([msg])
    assert wire[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


# --- tests: streaming → ModelEvent translation -----------------------------


async def _final_message(model: AnthropicMessagesModel) -> Message:
    """The MessageEnd payload from a streamed turn — what the next turn replays."""
    for ev in await _collect(model):
        if type(ev).__name__ == "MessageEnd":
            return cast(Message, ev.final)
    raise AssertionError("stream produced no MessageEnd")


async def _collect(model: AnthropicMessagesModel) -> list[Any]:
    out: list[Any] = []
    async for ev in model.request([], [], ModelSettings()):
        out.append(ev)
    return out


async def test_request_emits_full_lifecycle_for_text_and_tool_call() -> None:
    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_1")),
        _FakeStreamEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeStreamEvent(type="text"),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="text_delta", text="Hello"),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="text_delta", text=" world"),
        ),
        _FakeStreamEvent(type="content_block_stop", index=0),
        _FakeStreamEvent(
            type="content_block_start",
            index=1,
            content_block=_FakeStreamEvent(type="tool_use", id="tc1", name="search"),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=1,
            delta=_FakeStreamEvent(type="input_json_delta", partial_json='{"q":'),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=1,
            delta=_FakeStreamEvent(type="input_json_delta", partial_json='"x"}'),
        ),
        _FakeStreamEvent(type="content_block_stop", index=1),
        _FakeStreamEvent(
            type="message_stop",
            message=_FakeStreamEvent(
                usage=_FakeStreamEvent(
                    input_tokens=10,
                    output_tokens=4,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                )
            ),
        ),
    ]
    client = _build_fake_client(events)
    p = AnthropicProvider(client=client)
    m = AnthropicMessagesModel(provider=p)

    out = await _collect(m)
    types_seen = [type(e).__name__ for e in out]
    assert types_seen[0] == "ModelStart"
    assert "MessageStart" in types_seen
    assert "MessageDelta" in types_seen
    assert "ToolCallStart" in types_seen
    assert "ToolCallDelta" in types_seen
    assert "ToolCallEnd" in types_seen
    assert "MessageEnd" in types_seen
    assert types_seen[-1] == "ModelEnd"

    # MessageDelta carries cumulative partial.
    deltas = [e for e in out if isinstance(e, MessageDelta)]
    assert deltas[-1].partial.text == "Hello world"

    # ToolCallEnd carries parsed arguments.
    tc_end = next(e for e in out if isinstance(e, ToolCallEnd))
    assert tc_end.tool_call_id == "tc1"
    assert tc_end.tool_name == "search"
    assert tc_end.arguments == {"q": "x"}

    # ModelEnd carries usage.
    me = next(e for e in out if isinstance(e, ModelEnd))
    assert me.usage.input_tokens == 10


async def test_request_emits_thinking_events() -> None:
    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_2")),
        _FakeStreamEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeStreamEvent(type="thinking"),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="thinking_delta", thinking="step 1"),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="thinking_delta", thinking=" then 2"),
        ),
        _FakeStreamEvent(type="content_block_stop", index=0),
        _FakeStreamEvent(type="message_stop", message=None),
    ]
    p = AnthropicProvider(client=_build_fake_client(events))
    m = AnthropicMessagesModel(provider=p)
    out = await _collect(m)
    starts = [e for e in out if isinstance(e, ThinkingStart)]
    deltas = [e for e in out if isinstance(e, ThinkingDelta)]
    ends = [e for e in out if isinstance(e, ThinkingEnd)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert deltas[-1].partial == "step 1 then 2"


async def test_request_wraps_sdk_error_in_model_error() -> None:
    class _BoomStream:
        async def __aenter__(self) -> _BoomStream:
            raise RuntimeError("network down")

        async def __aexit__(self, *_: Any) -> None:  # pragma: no cover
            return None

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.stream = MagicMock(return_value=_BoomStream())
    p = AnthropicProvider(client=client)
    m = AnthropicMessagesModel(provider=p)
    from agent_harness.core.errors import ModelError

    with pytest.raises(ModelError):
        async for _ev in m.request([], [], ModelSettings()):
            pass


# --- tests: compaction + provider.request transport ------------------------


async def test_compact_messages_raises_not_supported() -> None:
    p = AnthropicProvider(client=MagicMock())
    m = AnthropicMessagesModel(provider=p)
    with pytest.raises(NotSupportedError):
        await m.compact_messages([])


async def test_provider_request_non_stream_yields_response() -> None:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_FakeStreamEvent(id="r"))
    p = AnthropicProvider(client=client)
    got = [e async for e in p.request({"model": "x"}, stream=False)]
    assert len(got) == 1
    assert got[0].kind == "response"


# --- tests: helpers --------------------------------------------------------


def test_parse_json_args_handles_invalid_input() -> None:
    assert _parse_json_args("") == {}
    assert _parse_json_args("not-json") == {"_raw": "not-json"}
    assert _parse_json_args('"scalar"') == {"_value": "scalar"}
    assert _parse_json_args('{"a": 1}') == {"a": 1}


def test_build_final_message_orders_thinking_text_then_calls() -> None:
    msg = _build_final_message(
        text="hi",
        thinking=[("t", "")],
        tool_meta={0: {"id": "c1", "name": "f"}},
        tool_args={0: '{"a":1}'},
    )
    kinds = [type(b).__name__ for b in msg.content]
    assert kinds == ["ThinkingBlock", "TextBlock", "ToolCallBlock"]


def test_build_final_message_keeps_each_thinking_block_with_its_own_signature() -> None:
    """A turn can hold several thinking blocks (interleaved thinking).

    Anthropic verifies each signature against its own block's text, so
    collapsing them into one block would carry the wrong signature and 400 on
    the next turn.
    """
    msg = _build_final_message(
        text="answer",
        thinking=[("first", "sig-a"), ("second", "sig-b")],
        tool_meta={},
        tool_args={},
    )
    thoughts = [b for b in msg.content if isinstance(b, ThinkingBlock)]
    assert [(b.text, b.signature) for b in thoughts] == [
        ("first", "sig-a"),
        ("second", "sig-b"),
    ]


def test_build_final_message_skips_wholly_empty_thinking_blocks() -> None:
    msg = _build_final_message(text="answer", thinking=[("", "")], tool_meta={}, tool_args={})
    assert [type(b).__name__ for b in msg.content] == ["TextBlock"]


def test_usage_from_returns_none_for_none() -> None:
    assert _usage_from(None) is None
    assert _usage_from(_FakeStreamEvent(usage=None)) is None


# Confirm the module is importable without the SDK present.
def test_module_imports_without_sdk() -> None:
    assert anthropic_mod is not None


# --- tests: thinking-block signature round-trip -----------------------------


async def test_signature_delta_is_captured_into_the_thinking_block() -> None:
    """Claude models stream a real signature; it must survive into the block.

    GLM-5.2 and Kimi K3 cannot exercise this — they emit an empty signature and
    never send a signature_delta — so the non-empty path is pinned here.
    """
    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_1")),
        _FakeStreamEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeStreamEvent(type="thinking", thinking="", signature=""),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="thinking_delta", thinking="weighing it up"),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="signature_delta", signature="ErUBCkYIAxgCIkC0zzz"),
        ),
        _FakeStreamEvent(type="content_block_stop", index=0),
        _FakeStreamEvent(type="message_stop", message=None),
    ]
    model = AnthropicMessagesModel(provider=AnthropicProvider(client=_build_fake_client(events)))

    msg = await _final_message(model)
    blocks = [b for b in msg.content if isinstance(b, ThinkingBlock)]
    assert len(blocks) == 1
    assert blocks[0].text == "weighing it up"
    assert blocks[0].signature == "ErUBCkYIAxgCIkC0zzz"


async def test_signature_defaults_to_empty_when_the_stream_sends_none() -> None:
    # OpenRouter-served GLM-5.2 / Kimi K3 shape: signature "" on
    # content_block_start, no signature_delta ever.
    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_1")),
        _FakeStreamEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeStreamEvent(type="thinking", thinking="", signature=""),
        ),
        _FakeStreamEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeStreamEvent(type="thinking_delta", thinking="reasoning"),
        ),
        _FakeStreamEvent(type="content_block_stop", index=0),
        _FakeStreamEvent(type="message_stop", message=None),
    ]
    model = AnthropicMessagesModel(provider=AnthropicProvider(client=_build_fake_client(events)))

    msg = await _final_message(model)
    blocks = [b for b in msg.content if isinstance(b, ThinkingBlock)]
    assert blocks[0].signature == ""


def test_thinking_block_serializes_with_its_signature() -> None:
    # Omitting "signature" 400s with "expected string, received undefined"
    # whenever a thinking block is replayed in history.
    wire = AnthropicMessagesModel._block_to_wire(ThinkingBlock(text="t", signature="sig-abc"))
    assert wire == {"type": "thinking", "thinking": "t", "signature": "sig-abc"}


async def test_multiple_thinking_blocks_keep_their_own_text_and_signature() -> None:
    """End-to-end: interleaved thinking must not collapse into one block.

    Anthropic verifies each signature against its own block's text, so a
    concatenated block carrying the last signature 400s on the next turn.
    """

    def _thinking(idx: int, text: str, sig: str) -> list[_FakeStreamEvent]:
        return [
            _FakeStreamEvent(
                type="content_block_start",
                index=idx,
                content_block=_FakeStreamEvent(type="thinking", thinking="", signature=""),
            ),
            _FakeStreamEvent(
                type="content_block_delta",
                index=idx,
                delta=_FakeStreamEvent(type="thinking_delta", thinking=text),
            ),
            _FakeStreamEvent(
                type="content_block_delta",
                index=idx,
                delta=_FakeStreamEvent(type="signature_delta", signature=sig),
            ),
            _FakeStreamEvent(type="content_block_stop", index=idx),
        ]

    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_1")),
        *_thinking(0, "first", "sig-a"),
        *_thinking(1, "second", "sig-b"),
        _FakeStreamEvent(type="message_stop", message=None),
    ]
    model = AnthropicMessagesModel(provider=AnthropicProvider(client=_build_fake_client(events)))
    out = await _collect(model)

    final = next(e for e in out if type(e).__name__ == "MessageEnd").final
    thoughts = [b for b in final.content if isinstance(b, ThinkingBlock)]
    assert [(b.text, b.signature) for b in thoughts] == [("first", "sig-a"), ("second", "sig-b")]
    # Each block's deltas are cumulative for that block only, never global.
    assert [d.partial for d in out if isinstance(d, ThinkingDelta)] == ["first", "second"]
    assert len([e for e in out if isinstance(e, ThinkingStart)]) == 2
    assert len([e for e in out if isinstance(e, ThinkingEnd)]) == 2


# --- tests: model catalogue + effort ----------------------------------------

_ALL_CLAUDE_MODELS = (OPUS_5, OPUS_4_8, OPUS_4_7, SONNET_5, SONNET_4_6)


def _bare_model(name: str) -> AnthropicMessagesModel:
    return AnthropicMessagesModel(provider=AnthropicProvider(client=MagicMock()), name=name)


@pytest.mark.parametrize("name", _ALL_CLAUDE_MODELS)
def test_capabilities_resolve_from_a_bare_model_name(name: str) -> None:
    # Regression test: capabilities=None must resolve from the catalogue by
    # name, never silently hand out Opus 4.7's limits under another name.
    m = _bare_model(name)
    assert m.capabilities == _CAPS_OPUS_4_7
    assert m.capabilities.context_window == 1_000_000
    assert m.capabilities.max_output_tokens == 128_000
    assert m.capabilities.adaptive_thinking is True


def test_unknown_model_name_with_no_capabilities_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="claude-nonexistent"):
        _bare_model("claude-nonexistent")


def test_unknown_model_name_with_explicit_capabilities_constructs_fine() -> None:
    caps = _CAPS_OPUS_4_7.model_copy(update={"context_window": 200_000})
    m = AnthropicMessagesModel(
        provider=AnthropicProvider(client=MagicMock()),
        name="claude-nonexistent",
        capabilities=caps,
    )
    assert m.capabilities is caps


def test_catalogue_enumerates_every_supported_model() -> None:
    assert set(_CAPABILITIES_BY_MODEL) == set(_ALL_CLAUDE_MODELS)
    # The effort catalogue mirrors the capabilities catalogue key-for-key, so
    # a consumer can enumerate one table and trust the other.
    assert set(EFFORT_LEVELS_BY_MODEL) == set(_ALL_CLAUDE_MODELS)


def test_opus_4_7_max_output_matches_current_docs() -> None:
    # Was 64_000 in the code while the docs said 128k — pin the corrected value.
    assert _CAPS_OPUS_4_7.max_output_tokens == 128_000


def test_sonnet_4_6_effort_levels_exclude_xhigh() -> None:
    # xhigh is newer than max; Sonnet 4.6 has max but not xhigh. Offering
    # xhigh there would send a level the vendor documents as unsupported.
    assert supported_effort_levels(SONNET_4_6) == ("low", "medium", "high", "max")


@pytest.mark.parametrize("name", [OPUS_5, OPUS_4_8, OPUS_4_7, SONNET_5])
def test_every_other_claude_model_supports_all_five_levels(name: str) -> None:
    assert supported_effort_levels(name) == ("low", "medium", "high", "xhigh", "max")


def test_supported_effort_levels_raises_for_unknown_model() -> None:
    with pytest.raises(ConfigError, match="claude-nonexistent"):
        supported_effort_levels("claude-nonexistent")


def test_effort_lands_in_output_config() -> None:
    m = _bare_model(OPUS_5)
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(effort="xhigh"),
    )
    assert payload["output_config"] == {"effort": "xhigh"}


def test_effort_is_emitted_independently_of_thinking() -> None:
    # Anthropic documents effort as not requiring thinking to be enabled, so
    # it must not hide inside the thinking_budget guard.
    m = _bare_model(SONNET_4_6)
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(effort="max"),
    )
    assert "thinking" not in payload
    assert payload["output_config"] == {"effort": "max"}


def test_effort_and_thinking_budget_coexist() -> None:
    m = _bare_model(OPUS_4_7)
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(effort="low", thinking_budget=2_000),
    )
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "low"}


def test_no_effort_means_no_output_config() -> None:
    m = _bare_model(OPUS_4_7)
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(),
    )
    assert "output_config" not in payload
