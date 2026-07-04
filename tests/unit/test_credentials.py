"""Unit tests for per-run credentials + the no-global-key-fallback rule.

Covers the pure resolver, the provider client-build path (the key actually
reaching the SDK client), and the ``Agent`` applying a resolved credential to
its model's provider for a run.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent_harness.core.agent import Agent
from agent_harness.core.credentials import (
    ApiKeyCredential,
    Credential,
    OAuthCredential,
    api_key_from_credential,
    resolve_credential,
)
from agent_harness.core.errors import ConfigError, NoCredentialError, NotSupportedError
from agent_harness.core.models import Model
from agent_harness.providers import anthropic as anthropic_mod
from agent_harness.providers.anthropic import AnthropicProvider
from tests.fakes import FakeTurn, make_model

# --- resolve_credential -----------------------------------------------------


def test_resolve_prefers_explicit_credential() -> None:
    cred = ApiKeyCredential(provider="anthropic", key="sk-explicit")
    other = ApiKeyCredential(provider="anthropic", key="sk-resolver")
    assert resolve_credential(credential=cred, credential_resolver=lambda: other) is cred


def test_resolve_falls_back_to_resolver() -> None:
    cred = ApiKeyCredential(provider="anthropic", key="sk-resolver")
    assert resolve_credential(credential_resolver=lambda: cred) is cred


def test_resolve_without_anything_raises_no_credential() -> None:
    with pytest.raises(NoCredentialError):
        resolve_credential()


# --- api_key_from_credential ------------------------------------------------


def test_api_key_from_matching_credential() -> None:
    cred = ApiKeyCredential(provider="anthropic", key="sk-x")
    assert api_key_from_credential(cred, expected_provider="anthropic") == "sk-x"


def test_api_key_provider_mismatch_raises_config_error() -> None:
    cred = ApiKeyCredential(provider="openai", key="sk-x")
    with pytest.raises(ConfigError):
        api_key_from_credential(cred, expected_provider="anthropic")


def test_api_key_from_oauth_raises_not_supported() -> None:
    cred = OAuthCredential(provider="anthropic", access_token="tok")
    with pytest.raises(NotSupportedError):
        api_key_from_credential(cred, expected_provider="anthropic")


# --- provider builds its client from the credential -------------------------


class _CapturingSDK:
    """Stand-in for the ``anthropic`` module capturing AsyncAnthropic kwargs."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] | None = None

    def AsyncAnthropic(self, **kwargs: Any) -> object:  # noqa: N802 - SDK name
        self.captured = kwargs
        return object()


def test_provider_builds_client_from_credential_key(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _CapturingSDK()
    monkeypatch.setattr(anthropic_mod, "_require_sdk", lambda: sdk)

    AnthropicProvider(credential=ApiKeyCredential(provider="anthropic", key="sk-run"))

    assert sdk.captured is not None
    assert sdk.captured["api_key"] == "sk-run"


def test_provider_uses_credential_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _CapturingSDK()
    monkeypatch.setattr(anthropic_mod, "_require_sdk", lambda: sdk)

    AnthropicProvider(
        credential_resolver=lambda: ApiKeyCredential(provider="anthropic", key="sk-lazy")
    )
    assert sdk.captured is not None
    assert sdk.captured["api_key"] == "sk-lazy"


def test_provider_without_key_or_credential_raises_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No api_key, no credential, no client ⇒ NoCredentialError — never a
    silent ANTHROPIC_API_KEY fallback."""
    monkeypatch.setattr(anthropic_mod, "_require_sdk", lambda: _CapturingSDK())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-should-be-ignored")

    with pytest.raises(NoCredentialError):
        AnthropicProvider()


# --- Agent applies the credential to the provider for a run -----------------


class _CapturingProvider:
    """Minimal provider that records the credential the agent applies."""

    name = "capture"
    base_url: str | None = None

    def __init__(self) -> None:
        self.applied: Credential | None = None

    def use_credential(self, credential: Credential) -> None:
        self.applied = credential

    async def request(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover - unused
        raise AssertionError("request should not be called by FakeModel")


class _PlainProvider:
    """Provider with no ``use_credential`` — cannot accept a per-run credential."""

    name = "plain"
    base_url: str | None = None

    async def request(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover - unused
        raise AssertionError("request should not be called by FakeModel")


async def test_agent_applies_credential_to_provider() -> None:
    provider = _CapturingProvider()
    model = make_model(FakeTurn(text="hi"), provider=cast(Any, provider))
    agent: Agent[None, str] = Agent(
        name="a",
        model=cast(Model, model),
        credential=ApiKeyCredential(provider="anthropic", key="sk-run"),
    )

    result = await agent.run("hello")

    assert result.output == "hi"
    assert isinstance(provider.applied, ApiKeyCredential)
    assert provider.applied.key == "sk-run"


async def test_agent_credential_resolver_applied_per_run() -> None:
    provider = _CapturingProvider()
    model = make_model(FakeTurn(text="hi"), provider=cast(Any, provider))
    agent: Agent[None, str] = Agent(
        name="a",
        model=cast(Model, model),
        credential_resolver=lambda: ApiKeyCredential(provider="anthropic", key="sk-lazy"),
    )

    await agent.run("hello")

    assert isinstance(provider.applied, ApiKeyCredential)
    assert provider.applied.key == "sk-lazy"


async def test_agent_without_credential_leaves_provider_untouched() -> None:
    provider = _CapturingProvider()
    model = make_model(FakeTurn(text="hi"), provider=cast(Any, provider))
    agent: Agent[None, str] = Agent(name="a", model=cast(Model, model))

    await agent.run("hello")

    assert provider.applied is None


async def test_agent_credential_with_incapable_provider_raises() -> None:
    provider = _PlainProvider()
    model = make_model(FakeTurn(text="hi"), provider=cast(Any, provider))
    agent: Agent[None, str] = Agent(
        name="a",
        model=cast(Model, model),
        credential=ApiKeyCredential(provider="anthropic", key="sk-run"),
    )

    with pytest.raises(ConfigError):
        await agent.run("hello")
