"""Unit tests for ``agent_harness.providers.openai``.

The ``openai`` SDK is an optional dependency; these tests mock it
entirely so they pass regardless of whether the SDK is installed in the
worktree's venv.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_harness.core.errors import ModelError, NotSupportedError
from agent_harness.core.events import (
    MessageDelta,
    MessageEnd,
    ModelEnd,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
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
from agent_harness.providers import openai as openai_mod
from agent_harness.providers.openai import (
    GPT_5_5,
    OpenAIProvider,
    OpenAIResponsesModel,
    _budget_to_effort,
    _parse_json_args,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


class _FakeEvent:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeStream:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[_FakeEvent]:
        async def _gen() -> AsyncIterator[_FakeEvent]:
            for e in self._events:
                yield e

        return _gen()


def _build_fake_client(events: list[_FakeEvent]) -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.stream = MagicMock(return_value=_FakeStream(events))
    return client


# --- provider construction --------------------------------------------------


def test_provider_accepts_injected_client() -> None:
    client = MagicMock()
    p = OpenAIProvider(client=client)
    assert p.client is client
    assert p.name == "openai"


def test_provider_raises_not_supported_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_import(*_a: Any, **_kw: Any) -> Any:
        raise ImportError("no openai")

    fake_mod = types.ModuleType("openai")
    fake_mod.AsyncOpenAI = _raise_import  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    with pytest.raises((NotSupportedError, ImportError)):
        OpenAIProvider(api_key="k")


# --- protocol conformance ---------------------------------------------------


def test_model_and_provider_satisfy_protocols() -> None:
    p = OpenAIProvider(client=MagicMock())
    m = OpenAIResponsesModel(provider=p)
    # Cast through ``object`` so mypy doesn't try to prove method-signature
    # compatibility for async-generator return types.
    assert isinstance(cast(object, p), Provider)
    assert isinstance(cast(object, m), Model)
    assert m.name == GPT_5_5
    assert m.capabilities.parallel_tool_calls is True
    assert m.capabilities.thinking is True
    assert m.capabilities.cache_control is True


# --- payload translation ----------------------------------------------------


def test_messages_to_wire_emits_message_call_and_call_output_items() -> None:
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")], timestamp=_ts()),
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(
            role="assistant",
            content=[
                TextBlock(text="ok"),
                ToolCallBlock(id="c1", name="search", arguments={"q": "x"}),
            ],
            timestamp=_ts(),
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_call_id="c1", content="hits")],
            timestamp=_ts(),
        ),
    ]
    wire = OpenAIResponsesModel._messages_to_wire(msgs)
    types_in_order = [it["type"] for it in wire]
    # System and user text become message items; the assistant turn emits a
    # message + function_call; the tool message emits function_call_output.
    assert "function_call" in types_in_order
    assert "function_call_output" in types_in_order
    fc = next(it for it in wire if it["type"] == "function_call")
    assert fc["call_id"] == "c1"
    assert json.loads(fc["arguments"]) == {"q": "x"}
    fco = next(it for it in wire if it["type"] == "function_call_output")
    assert fco["call_id"] == "c1"
    assert fco["output"] == "hits"


def test_build_payload_maps_settings_and_extra() -> None:
    p = OpenAIProvider(client=MagicMock())
    m = OpenAIResponsesModel(provider=p)
    msgs = [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())]
    settings = ModelSettings(
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        parallel_tool_calls=True,
        thinking_budget=8_000,
        extra={"previous_response_id": "resp_123"},
    )
    payload = m._build_payload(msgs, [], settings)
    assert payload["model"] == GPT_5_5
    assert payload["max_output_tokens"] == 100
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.9
    assert payload["parallel_tool_calls"] is True
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["previous_response_id"] == "resp_123"


def test_build_payload_drops_thinking_when_capability_off() -> None:
    p = OpenAIProvider(client=MagicMock())
    m = OpenAIResponsesModel(provider=p)
    m.capabilities = m.capabilities.model_copy(update={"thinking": False})
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(thinking_budget=1_000),
    )
    assert "reasoning" not in payload


def test_tools_to_wire_uses_function_shape() -> None:
    tools = [{"type": "function", "name": "search", "description": "", "parameters": {}}]
    out = OpenAIResponsesModel._tools_to_wire(tools)
    assert out == tools


# --- streaming → ModelEvent translation -------------------------------------


async def _collect(model: OpenAIResponsesModel) -> list[Any]:
    out: list[Any] = []
    async for ev in model.request([], [], ModelSettings()):
        out.append(ev)
    return out


async def test_request_emits_full_lifecycle_for_text_and_tool_call() -> None:
    events = [
        _FakeEvent(type="response.created", response=_FakeEvent(id="resp_1")),
        _FakeEvent(type="response.output_text.delta", delta="Hello"),
        _FakeEvent(type="response.output_text.delta", delta=" world"),
        _FakeEvent(
            type="response.output_item.added",
            item=_FakeEvent(type="function_call", call_id="call_1", name="search"),
        ),
        _FakeEvent(
            type="response.function_call_arguments.delta",
            call_id="call_1",
            delta='{"q":',
        ),
        _FakeEvent(
            type="response.function_call_arguments.delta",
            call_id="call_1",
            delta='"x"}',
        ),
        _FakeEvent(
            type="response.function_call_arguments.done",
            call_id="call_1",
            arguments='{"q":"x"}',
        ),
        _FakeEvent(
            type="response.completed",
            response=_FakeEvent(
                usage=_FakeEvent(
                    input_tokens=12,
                    output_tokens=5,
                    input_tokens_details=_FakeEvent(cached_tokens=3),
                )
            ),
        ),
    ]
    p = OpenAIProvider(client=_build_fake_client(events))
    m = OpenAIResponsesModel(provider=p)
    out = await _collect(m)
    types_seen = [type(e).__name__ for e in out]
    assert types_seen[0] == "ModelStart"
    assert "MessageStart" in types_seen
    assert "ToolCallStart" in types_seen
    assert "ToolCallDelta" in types_seen
    assert "ToolCallEnd" in types_seen
    assert types_seen[-1] == "ModelEnd"

    deltas = [e for e in out if isinstance(e, MessageDelta)]
    assert deltas[-1].partial.text == "Hello world"

    tcs = [e for e in out if isinstance(e, ToolCallStart)]
    assert tcs[0].tool_call_id == "call_1"
    tcds = [e for e in out if isinstance(e, ToolCallDelta)]
    assert tcds[0].arguments_delta == '{"q":'
    tcend = next(e for e in out if isinstance(e, ToolCallEnd))
    assert tcend.arguments == {"q": "x"}

    me = next(e for e in out if isinstance(e, ModelEnd))
    assert me.usage.input_tokens == 12
    assert me.usage.cache_read_tokens == 3


async def test_request_emits_thinking_events() -> None:
    events = [
        _FakeEvent(type="response.created", response=_FakeEvent(id="r2")),
        _FakeEvent(type="response.reasoning_summary_text.delta", delta="think a"),
        _FakeEvent(type="response.reasoning_summary_text.delta", delta="; think b"),
        _FakeEvent(type="response.reasoning_summary_text.done"),
        _FakeEvent(type="response.output_text.delta", delta="done"),
        _FakeEvent(type="response.completed", response=_FakeEvent(usage=None)),
    ]
    p = OpenAIProvider(client=_build_fake_client(events))
    m = OpenAIResponsesModel(provider=p)
    out = await _collect(m)
    deltas = [e for e in out if isinstance(e, ThinkingDelta)]
    assert deltas[-1].partial == "think a; think b"
    # End-of-message structure: ThinkingEnd before any MessageDelta.
    msg_end = next(e for e in out if isinstance(e, MessageEnd))
    assert "done" in msg_end.final.text


async def test_request_wraps_sdk_error_in_model_error() -> None:
    class _Boom:
        async def __aenter__(self) -> _Boom:
            raise RuntimeError("upstream 500")

        async def __aexit__(self, *_: Any) -> None:  # pragma: no cover
            return None

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.stream = MagicMock(return_value=_Boom())
    p = OpenAIProvider(client=client)
    m = OpenAIResponsesModel(provider=p)
    with pytest.raises(ModelError):
        async for _ in m.request([], [], ModelSettings()):
            pass


# --- compaction + transport -------------------------------------------------


async def test_compact_messages_raises_not_supported() -> None:
    p = OpenAIProvider(client=MagicMock())
    m = OpenAIResponsesModel(provider=p)
    with pytest.raises(NotSupportedError):
        await m.compact_messages([])


async def test_provider_request_non_stream_yields_response() -> None:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=_FakeEvent(id="r"))
    p = OpenAIProvider(client=client)
    got = [e async for e in p.request({"model": "x"}, stream=False)]
    assert len(got) == 1
    assert got[0].kind == "response"


# --- helpers ----------------------------------------------------------------


def test_budget_to_effort_buckets() -> None:
    assert _budget_to_effort(0) == "minimal"
    assert _budget_to_effort(100) == "low"
    assert _budget_to_effort(5_000) == "medium"
    assert _budget_to_effort(50_000) == "high"


def test_parse_json_args_handles_invalid_input() -> None:
    assert _parse_json_args(None) == {}
    assert _parse_json_args("") == {}
    assert _parse_json_args("not-json") == {"_raw": "not-json"}
    assert _parse_json_args('"x"') == {"_value": "x"}
    assert _parse_json_args('{"a":1}') == {"a": 1}


def test_module_imports_without_sdk() -> None:
    assert openai_mod is not None
