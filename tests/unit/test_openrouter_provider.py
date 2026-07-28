"""Unit tests for the OpenRouter provider.

The ``anthropic`` SDK is mocked entirely so these tests pass regardless of
whether it is installed.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from agent_harness.core.credentials import ApiKeyCredential, OAuthCredential
from agent_harness.core.errors import ConfigError, NotSupportedError
from agent_harness.core.models import Message, Model, ModelSettings, Provider, TextBlock
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
