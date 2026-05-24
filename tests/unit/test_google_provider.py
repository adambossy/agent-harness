"""Unit tests for ``agent_harness.providers.google``.

The ``google-genai`` SDK is an optional dependency; these tests mock it
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

from agent_harness.core.errors import ModelError, NotSupportedError
from agent_harness.core.events import (
    MessageDelta,
    ModelEnd,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallEnd,
    ToolCallStart,
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
from agent_harness.providers import google as google_mod
from agent_harness.providers.google import (
    GEMINI_3_5_FLASH,
    GeminiModel,
    GoogleProvider,
    _usage_from_meta,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


class _F:
    """Free-form fake namespace (attrgetter target)."""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _gen_iter(chunks: list[_F]) -> AsyncIterator[_F]:
    async def _gen() -> AsyncIterator[_F]:
        for c in chunks:
            yield c

    return _gen()


def _build_client(chunks: list[_F]) -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=_gen_iter(chunks))
    return client


# --- provider construction --------------------------------------------------


def test_provider_accepts_injected_client() -> None:
    client = MagicMock()
    p = GoogleProvider(client=client)
    assert p.client is client
    assert p.name == "google"


def test_provider_raises_not_supported_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force ``from google import genai`` to fail."""
    fake = types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", fake)
    # Ensure genai cannot be imported.
    if "google.genai" in sys.modules:
        monkeypatch.delitem(sys.modules, "google.genai")
    with pytest.raises((NotSupportedError, ImportError, AttributeError)):
        GoogleProvider(api_key="k")


# --- protocol conformance ---------------------------------------------------


def test_model_and_provider_satisfy_protocols() -> None:
    p = GoogleProvider(client=MagicMock())
    m = GeminiModel(provider=p)
    # Cast through ``object`` so mypy doesn't try to prove method-signature
    # compatibility for async-generator return types.
    assert isinstance(cast(object, p), Provider)
    assert isinstance(cast(object, m), Model)
    assert m.name == GEMINI_3_5_FLASH
    assert m.capabilities.parallel_tool_calls is True
    assert m.capabilities.thinking is True
    assert m.capabilities.cache_control is True
    assert m.capabilities.vision is True


# --- payload translation ----------------------------------------------------


def test_messages_to_wire_extracts_system_and_contents() -> None:
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")], timestamp=_ts()),
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text="t"),
                TextBlock(text="ok"),
                ToolCallBlock(id="c1", name="search", arguments={"q": "x"}),
            ],
            timestamp=_ts(),
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_call_id="search", content="hit")],
            timestamp=_ts(),
        ),
    ]
    system, contents = GeminiModel._messages_to_wire(msgs)
    assert system == "be brief"
    # user → role=user, assistant → role=model, tool → role=user.
    roles = [c["role"] for c in contents]
    assert roles == ["user", "model", "user"]
    # Assistant turn includes function_call part.
    fc_part = next(p for p in contents[1]["parts"] if "function_call" in p)
    assert fc_part["function_call"]["name"] == "search"
    assert fc_part["function_call"]["args"] == {"q": "x"}
    # Tool turn carries function_response.
    fr_part = next(p for p in contents[2]["parts"] if "function_response" in p)
    assert fr_part["function_response"]["name"] == "search"


def test_build_payload_maps_settings_and_extra() -> None:
    p = GoogleProvider(client=MagicMock())
    m = GeminiModel(provider=p)
    msgs = [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())]
    settings = ModelSettings(
        temperature=0.2,
        max_tokens=200,
        top_p=0.9,
        seed=42,
        thinking_budget=4_000,
        extra={"safety_settings": [{"category": "HARM"}]},
    )
    payload = m._build_payload(msgs, [], settings)
    cfg = payload["config"]
    assert payload["model"] == GEMINI_3_5_FLASH
    assert cfg["temperature"] == 0.2
    assert cfg["top_p"] == 0.9
    assert cfg["max_output_tokens"] == 200
    assert cfg["seed"] == 42
    assert cfg["thinking_config"] == {
        "include_thoughts": True,
        "thinking_budget": 4_000,
    }
    assert cfg["safety_settings"] == [{"category": "HARM"}]


def test_build_payload_drops_thinking_when_capability_off() -> None:
    p = GoogleProvider(client=MagicMock())
    m = GeminiModel(provider=p)
    m.capabilities = m.capabilities.model_copy(update={"thinking": False})
    payload = m._build_payload(
        [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())],
        [],
        ModelSettings(thinking_budget=1_000),
    )
    assert "thinking_config" not in payload.get("config", {})


def test_tools_to_wire_emits_function_declarations() -> None:
    tools = [{"name": "search", "description": "look up", "parameters": {"type": "object"}}]
    out = GeminiModel._tools_to_wire(tools)
    assert out == [{"function_declarations": tools}]


# --- streaming → ModelEvent translation -------------------------------------


async def _collect(model: GeminiModel) -> list[Any]:
    return [ev async for ev in model.request([], [], ModelSettings())]


async def test_request_emits_full_lifecycle_for_text_and_tool_call() -> None:
    chunks = [
        _F(
            candidates=[
                _F(content=_F(parts=[_F(text="Hello", thought=False, function_call=None)]))
            ],
            usage_metadata=None,
        ),
        _F(
            candidates=[
                _F(content=_F(parts=[_F(text=" world", thought=False, function_call=None)]))
            ],
            usage_metadata=None,
        ),
        _F(
            candidates=[
                _F(
                    content=_F(
                        parts=[
                            _F(
                                text=None,
                                thought=False,
                                function_call=_F(name="search", args={"q": "x"}, id="fc-1"),
                            )
                        ]
                    )
                )
            ],
            usage_metadata=_F(
                prompt_token_count=8,
                candidates_token_count=5,
                cached_content_token_count=2,
            ),
        ),
    ]
    p = GoogleProvider(client=_build_client(chunks))
    m = GeminiModel(provider=p)
    out = await _collect(m)
    types_seen = [type(e).__name__ for e in out]
    assert types_seen[0] == "ModelStart"
    assert "MessageStart" in types_seen
    assert "ToolCallStart" in types_seen
    assert "ToolCallEnd" in types_seen
    assert types_seen[-1] == "ModelEnd"

    deltas = [e for e in out if isinstance(e, MessageDelta)]
    assert deltas[-1].partial.text == "Hello world"

    tc_start = next(e for e in out if isinstance(e, ToolCallStart))
    assert tc_start.tool_name == "search"
    tc_end = next(e for e in out if isinstance(e, ToolCallEnd))
    assert tc_end.arguments == {"q": "x"}

    me = next(e for e in out if isinstance(e, ModelEnd))
    assert me.usage.input_tokens == 8
    assert me.usage.cache_read_tokens == 2


async def test_request_emits_thinking_events() -> None:
    chunks = [
        _F(
            candidates=[
                _F(content=_F(parts=[_F(text="step 1", thought=True, function_call=None)]))
            ],
            usage_metadata=None,
        ),
        _F(
            candidates=[
                _F(content=_F(parts=[_F(text=" then 2", thought=True, function_call=None)]))
            ],
            usage_metadata=None,
        ),
        _F(
            candidates=[_F(content=_F(parts=[_F(text="done", thought=False, function_call=None)]))],
            usage_metadata=None,
        ),
    ]
    p = GoogleProvider(client=_build_client(chunks))
    m = GeminiModel(provider=p)
    out = await _collect(m)
    starts = [e for e in out if isinstance(e, ThinkingStart)]
    ends = [e for e in out if isinstance(e, ThinkingEnd)]
    deltas = [e for e in out if isinstance(e, ThinkingDelta)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert deltas[-1].partial == "step 1 then 2"
    # A normal MessageDelta still surfaces post-thinking content.
    mdeltas = [e for e in out if isinstance(e, MessageDelta)]
    assert mdeltas[-1].partial.text == "done"


async def test_request_handles_empty_stream() -> None:
    p = GoogleProvider(client=_build_client([]))
    m = GeminiModel(provider=p)
    out = await _collect(m)
    types_seen = [type(e).__name__ for e in out]
    # Even with no chunks the model still emits a complete lifecycle.
    assert types_seen[0] == "ModelStart"
    assert "MessageStart" in types_seen
    assert types_seen[-1] == "ModelEnd"


async def test_request_wraps_sdk_error_in_model_error() -> None:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()

    async def _raise(**_kw: Any) -> Any:
        raise RuntimeError("network down")

    client.aio.models.generate_content_stream = _raise
    p = GoogleProvider(client=client)
    m = GeminiModel(provider=p)
    with pytest.raises(ModelError):
        async for _ in m.request([], [], ModelSettings()):
            pass


# --- compaction + transport -------------------------------------------------


async def test_compact_messages_raises_not_supported() -> None:
    p = GoogleProvider(client=MagicMock())
    m = GeminiModel(provider=p)
    with pytest.raises(NotSupportedError):
        await m.compact_messages([])


async def test_provider_request_non_stream_yields_response() -> None:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=_F(id="r"))
    p = GoogleProvider(client=client)
    got = [e async for e in p.request({"model": "x"}, stream=False)]
    assert len(got) == 1
    assert got[0].kind == "response"


# --- helpers ----------------------------------------------------------------


def test_usage_from_meta_handles_missing_fields() -> None:
    assert _usage_from_meta(None) is None
    u = _usage_from_meta(_F(prompt_token_count=1, candidates_token_count=2))
    assert u is not None
    assert u.input_tokens == 1
    assert u.output_tokens == 2


def test_module_imports_without_sdk() -> None:
    assert google_mod is not None
