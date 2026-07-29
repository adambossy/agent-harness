"""Unit tests for the OpenRouter provider.

The ``anthropic`` SDK is mocked entirely so these tests pass regardless of
whether it is installed.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agent_harness.core.credentials import ApiKeyCredential, OAuthCredential
from agent_harness.core.errors import ConfigError, NotSupportedError
from agent_harness.core.models import (
    Message,
    Model,
    ModelCapabilities,
    ModelSettings,
    Provider,
    TextBlock,
    ThinkingBlock,
)
from agent_harness.providers.openrouter import (
    CAPS_GLM_5_2,
    CAPS_KIMI_K3,
    GLM_5_2,
    KIMI_K3,
    MOONSHOT_DIRECT,
    OPENROUTER_BASE_URL,
    US_FP8_ZDR,
    OpenRouterModel,
    OpenRouterProvider,
    RoutingPolicy,
)

# NOTE: keep this import block minimal — ruff runs on commit, so an import
# added here before the task that uses it fails the gate as F401.


def test_routing_policy_to_wire_emits_openrouter_field_names() -> None:
    policy = RoutingPolicy(
        only=("baseten",),
        quantizations=("fp8",),
        zdr=True,
        data_collection="deny",
        require_parameters=True,
        sort="price",
        allow_fallbacks=True,
    )
    assert policy.to_wire() == {
        "only": ["baseten"],
        "quantizations": ["fp8"],
        "zdr": True,
        "data_collection": "deny",
        "require_parameters": True,
        "sort": "price",
        "allow_fallbacks": True,
    }


def test_routing_policy_omits_empty_collections() -> None:
    # An empty tuple means "no filter", which must be absent from the wire
    # object rather than sent as [] (which OpenRouter reads as "match nothing").
    wire = RoutingPolicy(only=(), quantizations=()).to_wire()
    assert "only" not in wire
    assert "quantizations" not in wire


def test_us_fp8_zdr_preset_matches_the_verified_allowlist() -> None:
    assert US_FP8_ZDR.only == (
        "novita",
        "decart",
        "atlas-cloud",
        "venice",
        "baseten",
        "io-net",
    )
    assert US_FP8_ZDR.quantizations == ("fp8", "bf16", "fp16", "fp32")
    assert US_FP8_ZDR.zdr is True
    assert US_FP8_ZDR.data_collection == "deny"
    assert US_FP8_ZDR.sort == "price"


def test_moonshot_direct_preset_allows_int4_single_provider() -> None:
    # Kimi K3 has exactly one endpoint (moonshotai/int4, Singapore). The
    # quantization floor and the US allowlist are deliberately dropped here.
    assert MOONSHOT_DIRECT.only == ("moonshotai",)
    assert MOONSHOT_DIRECT.quantizations == ()
    assert MOONSHOT_DIRECT.zdr is True


def _fake_anthropic_module() -> types.ModuleType:
    """A stand-in ``anthropic`` module recording AsyncAnthropic kwargs."""
    mod = types.ModuleType("anthropic")
    captured: dict[str, Any] = {}

    class _AsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    mod.AsyncAnthropic = _AsyncAnthropic  # type: ignore[attr-defined]
    mod.captured = captured  # type: ignore[attr-defined]
    return mod


def test_provider_name_is_openrouter() -> None:
    assert OpenRouterProvider.name == "openrouter"


def test_provider_defaults_to_openrouter_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    OpenRouterProvider(api_key="sk-or-test")
    assert mod.captured["base_url"] == OPENROUTER_BASE_URL
    assert mod.captured["api_key"] == "sk-or-test"


def test_base_url_omits_the_version_segment() -> None:
    # AsyncAnthropic appends "/v1/messages" to base_url. If OPENROUTER_BASE_URL
    # carried its own "/v1", every request would go to "/api/v1/v1/messages"
    # and get OpenRouter's HTML 404 instead of the API. This shipped green as
    # "https://openrouter.ai/api/v1" until a live call caught it, so pin it.
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api"
    assert not OPENROUTER_BASE_URL.rstrip("/").endswith("/v1")


def test_base_url_resolves_to_the_anthropic_messages_endpoint() -> None:
    # The real SDK is needed to pin the FULLY RESOLVED URL; mocked tests cannot
    # observe URL construction at all. Skipped when the extra is not installed.
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.AsyncAnthropic(api_key="sk-or-test", base_url=OPENROUTER_BASE_URL)
    resolved = str(client._prepare_url("/v1/messages"))
    assert resolved == "https://openrouter.ai/api/v1/messages"


def test_provider_honours_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    OpenRouterProvider(api_key="sk-or-test", base_url="https://proxy.example/v1")
    assert mod.captured["base_url"] == "https://proxy.example/v1"


def test_provider_accepts_openrouter_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    OpenRouterProvider(credential=ApiKeyCredential(provider="openrouter", key="sk-or-1"))
    assert mod.captured["api_key"] == "sk-or-1"


def test_provider_rejects_anthropic_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    with pytest.raises(ConfigError):
        OpenRouterProvider(credential=ApiKeyCredential(provider="anthropic", key="sk-ant-1"))


def test_provider_rejects_oauth_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    with pytest.raises(NotSupportedError):
        OpenRouterProvider(credential=OAuthCredential(provider="openrouter", access_token="t"))


def test_provider_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(NotSupportedError):
        OpenRouterProvider(api_key="sk-or-test")


def test_provider_satisfies_protocol() -> None:
    p = OpenRouterProvider(client=object())
    assert isinstance(cast(object, p), Provider)


def _model(**kw: Any) -> OpenRouterModel:
    return OpenRouterModel(provider=OpenRouterProvider(client=object()), **kw)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _msgs() -> list[Message]:
    return [Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts())]


def test_payload_carries_the_default_routing_policy() -> None:
    m = _model(name=GLM_5_2, capabilities=CAPS_GLM_5_2)
    payload = m._build_payload(_msgs(), [], ModelSettings())
    assert payload["model"] == "z-ai/glm-5.2"
    assert payload["extra_body"]["provider"] == US_FP8_ZDR.to_wire()


def test_explicit_routing_policy_overrides_the_default() -> None:
    m = _model(name=KIMI_K3, capabilities=CAPS_KIMI_K3, routing=MOONSHOT_DIRECT)
    payload = m._build_payload(_msgs(), [], ModelSettings())
    assert payload["extra_body"]["provider"]["only"] == ["moonshotai"]
    assert "quantizations" not in payload["extra_body"]["provider"]


def test_caller_supplied_extra_body_provider_wins() -> None:
    # ModelSettings.extra is merged by the parent _build_payload before we run,
    # so a caller who states a policy explicitly must not be overridden.
    m = _model(name=GLM_5_2, capabilities=CAPS_GLM_5_2)
    settings = ModelSettings(extra={"extra_body": {"provider": {"only": ["baseten"]}}})
    payload = m._build_payload(_msgs(), [], settings)
    assert payload["extra_body"]["provider"] == {"only": ["baseten"]}


def test_caller_extra_body_keys_are_preserved_alongside_routing() -> None:
    m = _model(name=GLM_5_2, capabilities=CAPS_GLM_5_2)
    settings = ModelSettings(extra={"extra_body": {"transforms": ["middle-out"]}})
    payload = m._build_payload(_msgs(), [], settings)
    assert payload["extra_body"]["transforms"] == ["middle-out"]
    assert payload["extra_body"]["provider"] == US_FP8_ZDR.to_wire()


def test_model_capability_constants_match_live_endpoint_limits() -> None:
    assert CAPS_GLM_5_2.context_window == 1_048_576
    assert CAPS_GLM_5_2.max_output_tokens == 131_072
    assert CAPS_KIMI_K3.context_window == 1_048_576


def test_model_satisfies_protocol() -> None:
    assert isinstance(cast(object, _model()), Model)


# --- tests: capability resolution (Finding 1) -------------------------------


def test_default_construction_resolves_glm_5_2_capabilities() -> None:
    # Regression test: capabilities=None must resolve from _CAPS_BY_MODEL by
    # name, never fall through to the parent's Anthropic-Opus default.
    m = _model()
    assert m.capabilities == CAPS_GLM_5_2
    assert m.capabilities.cache_control is False
    assert m.capabilities.context_window == 1_048_576
    assert m.capabilities.max_output_tokens == 131_072


def test_kimi_k3_with_no_explicit_capabilities_resolves_from_name() -> None:
    m = _model(name=KIMI_K3)
    assert m.capabilities == CAPS_KIMI_K3


def test_unknown_model_name_with_no_capabilities_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        _model(name="some/other-model")


def test_unknown_model_name_with_explicit_capabilities_constructs_fine() -> None:
    caps = ModelCapabilities(
        parallel_tool_calls=True,
        thinking=False,
        cache_control=False,
        vision=False,
        audio_input=False,
        audio_output=False,
        structured_output=True,
        context_window=32_768,
        max_output_tokens=8_192,
        supports_compaction=False,
    )
    m = _model(name="some/other-model", capabilities=caps)
    assert m.capabilities is caps


# --- tests: extra_body routing reaches the SDK call (Finding 2) ------------


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


async def test_request_passes_routing_policy_through_to_the_sdk_call() -> None:
    # test_payload_carries_the_default_routing_policy only proves _build_payload
    # is correct in isolation; this drives the full async request() path
    # against a fake client so a future change that filters payload keys or
    # swaps SDK methods before the call would fail here even if it left
    # _build_payload untouched.
    events = [
        _FakeStreamEvent(type="message_start", message=_FakeStreamEvent(id="msg_1")),
        _FakeStreamEvent(type="message_stop", message=None),
    ]
    client = _build_fake_client(events)
    provider = OpenRouterProvider(client=client)
    model = OpenRouterModel(provider=provider, name=GLM_5_2, capabilities=CAPS_GLM_5_2)

    async for _ in model.request([], [], ModelSettings()):
        pass

    assert client.messages.stream.called
    _, kwargs = client.messages.stream.call_args
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["provider"] == US_FP8_ZDR.to_wire()


def test_thinking_blocks_carry_a_signature_field() -> None:
    # The parent emits {"type": "thinking", ...} with no "signature", which the
    # Anthropic Messages schema requires. Echoing one back 400s on the second
    # turn of every tool-calling loop, because GLM/K3 reason on every turn.
    # Emitting the field (empty is accepted for these models) keeps the model
    # seeing its own prior reasoning rather than having it stripped.
    assert OpenRouterModel._block_to_wire(ThinkingBlock(text="deliberating")) == {
        "type": "thinking",
        "thinking": "deliberating",
        "signature": "",
    }


def test_non_thinking_blocks_still_serialize() -> None:
    wire = OpenRouterModel._block_to_wire(TextBlock(text="hello"))
    assert wire == {"type": "text", "text": "hello"}


def test_history_preserves_thinking_blocks() -> None:
    msgs = [
        Message(
            role="assistant",
            content=[ThinkingBlock(text="hmm"), TextBlock(text="answer")],
            timestamp=_ts(),
        )
    ]
    _system, wire = OpenRouterModel._messages_to_wire(msgs)
    assert wire == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": ""},
                {"type": "text", "text": "answer"},
            ],
        }
    ]
