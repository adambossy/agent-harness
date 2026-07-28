"""Unit tests for the OpenRouter provider.

The ``anthropic`` SDK is mocked entirely so these tests pass regardless of
whether it is installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest

from agent_harness.core.credentials import ApiKeyCredential, OAuthCredential
from agent_harness.core.errors import ConfigError, NotSupportedError
from agent_harness.core.models import Provider
from agent_harness.providers.openrouter import (
    MOONSHOT_DIRECT,
    OPENROUTER_BASE_URL,
    US_FP8_ZDR,
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
