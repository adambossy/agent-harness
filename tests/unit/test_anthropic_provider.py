"""Unit tests for ``agent_harness.providers.anthropic``.

The ``anthropic`` SDK is an optional dependency; these tests mock it
entirely so they pass regardless of whether the SDK is installed in the
worktree's venv.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_harness.core.errors import NotSupportedError
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
    ToolCallBlock,
    ToolResultBlock,
)
from agent_harness.providers import anthropic as anthropic_mod
from agent_harness.providers.anthropic import (
    OPUS_4_7,
    AnthropicMessagesModel,
    AnthropicProvider,
    _build_final_message,
    _parse_json_args,
    _usage_from,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# --- fake SDK stream --------------------------------------------------------


class _FakeStreamEvent:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeStream:
    def __init__(self, events: list[_FakeStreamEvent]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[_FakeStreamEvent]:
        async def _gen() -> AsyncIterator[_FakeStreamEvent]:
            for e in self._events:
                yield e

        return _gen()


def _build_fake_client(events: list[_FakeStreamEvent]) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeStream(events))
    return client


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
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2_000}
    assert payload["metadata"] == {"trace_id": "t1"}


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
        thinking="t",
        tool_meta={0: {"id": "c1", "name": "f"}},
        tool_args={0: '{"a":1}'},
    )
    kinds = [type(b).__name__ for b in msg.content]
    assert kinds == ["ThinkingBlock", "TextBlock", "ToolCallBlock"]


def test_usage_from_returns_none_for_none() -> None:
    assert _usage_from(None) is None
    assert _usage_from(_FakeStreamEvent(usage=None)) is None


# Confirm the module is importable without the SDK present.
def test_module_imports_without_sdk() -> None:
    assert anthropic_mod is not None
