# OpenRouter Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `OpenRouterProvider` + `OpenRouterModel` to agent-harness that reaches GLM-5.2 and Kimi K3 through OpenRouter, with a preset US-only / FP8-or-better / zero-data-retention provider-routing policy.

**Architecture:** OpenRouter exposes an Anthropic Messages-compatible surface at `/api/v1/messages` (the "Anthropic Skin") that accepts the *same* `ProviderPreferences` object as `/chat/completions` — verified against `openrouter.ai/openapi.json`, where both paths `$ref` `#/components/schemas/ProviderPreferences`. That means we reuse the existing, already-tested `AnthropicMessagesModel` wire translation wholesale instead of writing a Chat-Completions adapter. The new module is thin: a frozen `RoutingPolicy` value type, a `OpenRouterProvider` subclassing `AnthropicProvider` (to change `name` and default `base_url`), and an `OpenRouterModel` subclassing `AnthropicMessagesModel` that injects the routing policy into `extra_body` in `_build_payload`.

**Tech Stack:** Python 3.13, `anthropic` SDK (already an optional extra), pytest, mypy --strict, ruff.

## Global Constraints

- Python 3.13; `mypy --strict` must pass; type annotations on everything.
- No bare `typing.Any` without a reason comment; no `# type: ignore` without a rationale comment.
- `Protocol` over `ABC`; records are `@dataclass(frozen=True, slots=True)` or pydantic `BaseModel`.
- Raise only errors from `agent_harness.core.errors` — do not invent a hierarchy.
- No logging library; publish typed events on the `EventBus`.
- **No new top-level dependency.** This work adds none — it rides the existing `anthropic` extra.
- Per-file target < 500 LOC. Line length 100 (ruff).
- Every public type needs a docstring **with a usage example** (`>>> # …  # doctest: +SKIP` for constructors).
- Imports: relative within a subpackage, absolute cross-subpackage. `core/` may not import from `providers/`.
- Gate: `uv run pre-commit run --all-files` and `uv run pytest` must both be clean.
- Tests must not require network or an installed SDK — mock everything, following `tests/unit/test_anthropic_provider.py`.

## Verified Facts (do not re-derive)

- `AnthropicProvider.name` is a **class attribute** (`anthropic.py:104`) read by `_resolve_key` → `api_key_from_credential(cred, expected_provider=self.name)` (`anthropic.py:163`). Overriding `name` in a subclass automatically changes the credential the provider will accept.
- `AnthropicProvider._build_client` passes `base_url` to `AsyncAnthropic` only when not `None` (`anthropic.py:148-149`).
- `AnthropicProvider.__init__` accepts a pre-built `client=` that bypasses credential resolution entirely (`anthropic.py:120-124`) — this is the test seam.
- `AnthropicMessagesModel._build_payload` (`anthropic.py:294-329`) builds the wire dict and, as its **last** step, merges `settings.extra` verbatim into the top-level payload. Overriding `_build_payload` and calling `super()` therefore sees any caller-supplied `extra_body` already in place.
- `payload` is splatted into `client.messages.stream(**payload)` (`anthropic.py:353`). The Anthropic SDK accepts an `extra_body` kwarg and merges it into the JSON request body — that is how OpenRouter-only fields reach the wire.
- `ModelCapabilities` fields: `parallel_tool_calls`, `thinking`, `cache_control`, `vision`, `audio_input`, `audio_output`, `structured_output`, `context_window`, `max_output_tokens`, `supports_compaction`.
- Live OpenRouter endpoint data (checked 2026-07-25): the six-provider US allowlist below yields 7 ZDR-flagged fp8 endpoints for `z-ai/glm-5.2`, cheapest `novita/fp8` at $0.70/$2.21 with 1,048,576 context and 131,072 max completion tokens. `moonshotai/kimi-k3` has exactly one endpoint — `moonshotai/int4`, Singapore, 1,048,576 context, ZDR-flagged.

## File Structure

- **Create** `agent_harness/providers/openrouter.py` — the whole feature: `RoutingPolicy`, the two preset policies, model-id + capability constants, `OpenRouterProvider`, `OpenRouterModel`. One cohesive module, ~230 LOC; these things change together so they live together.
- **Create** `tests/unit/test_openrouter_provider.py` — mirrors `tests/unit/test_anthropic_provider.py` structure.
- **Modify** `README.md` — add an OpenRouter usage snippet beside the existing provider examples.

No changes to `core/`, no new dependency, no registry wiring (agent-harness has none — hosts construct providers directly).

---

### Task 1: `RoutingPolicy` value type and preset policies

**Files:**
- Create: `agent_harness/providers/openrouter.py`
- Test: `tests/unit/test_openrouter_provider.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RoutingPolicy` (frozen dataclass) with `.to_wire() -> dict[str, Any]`; constants `US_FP8_ZDR: RoutingPolicy` and `MOONSHOT_DIRECT: RoutingPolicy`.

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for the OpenRouter provider.

The ``anthropic`` SDK is mocked entirely so these tests pass regardless of
whether it is installed.
"""

from __future__ import annotations

from agent_harness.providers.openrouter import (
    MOONSHOT_DIRECT,
    US_FP8_ZDR,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/agent-harness && uv run pytest tests/unit/test_openrouter_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_harness.providers.openrouter'`

- [ ] **Step 3: Write minimal implementation**

Create `agent_harness/providers/openrouter.py`:

```python
"""OpenRouter provider — GLM-5.2, Kimi K3, and anything else OpenRouter serves.

OpenRouter exposes an Anthropic Messages-compatible surface at
``/api/v1/messages`` (the "Anthropic Skin") that accepts the same
``ProviderPreferences`` routing object as its OpenAI-shaped endpoint. That
lets this module reuse :class:`~agent_harness.providers.anthropic.
AnthropicMessagesModel` wholesale and add only what is OpenRouter-specific:
which upstream inference providers a request may be routed to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MOONSHOT_DIRECT",
    "US_FP8_ZDR",
    "RoutingPolicy",
]


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Which upstream providers OpenRouter may route a request to.

    Renders to OpenRouter's ``provider`` preferences object. ``only``,
    ``quantizations``, ``zdr``, and ``data_collection`` are *hard* filters that
    define the candidate set; ``sort`` and ``allow_fallbacks`` only order and
    traverse it. Empty collections mean "no filter" and are omitted from the
    wire object — sending ``[]`` would match nothing.

    Example:
        >>> RoutingPolicy(only=("baseten",)).to_wire()["only"]
        ['baseten']
    """

    only: tuple[str, ...] = ()
    quantizations: tuple[str, ...] = ()
    zdr: bool = True
    data_collection: str = "deny"
    require_parameters: bool = True
    sort: str = "price"
    allow_fallbacks: bool = True

    def to_wire(self) -> dict[str, Any]:
        """Render as OpenRouter's ``provider`` preferences object."""
        wire: dict[str, Any] = {
            "zdr": self.zdr,
            "data_collection": self.data_collection,
            "require_parameters": self.require_parameters,
            "sort": self.sort,
            "allow_fallbacks": self.allow_fallbacks,
        }
        if self.only:
            wire["only"] = list(self.only)
        if self.quantizations:
            wire["quantizations"] = list(self.quantizations)
        return wire


US_FP8_ZDR = RoutingPolicy(
    only=("novita", "decart", "atlas-cloud", "venice", "baseten", "io-net"),
    quantizations=("fp8", "bf16", "fp16", "fp32"),
)
"""US-headquartered, FP8-or-better, zero-data-retention endpoints, cheapest first.

Verified against the live OpenRouter endpoint API on 2026-07-25: for
``z-ai/glm-5.2`` this matches 7 endpoints, cheapest ``novita/fp8`` at
$0.70/$2.21 per Mtok with the full 1M context. Because ``sort="price"``
orders the set, widening ``only`` adds fallback resilience without adding
cost — pricier endpoints serve only when cheaper ones fail.

Caveat: OpenRouter has no region field. "US" here means US-headquartered;
``datacenters`` is declared for only three of these six providers.
"""

MOONSHOT_DIRECT = RoutingPolicy(only=("moonshotai",))
"""Kimi K3's sole endpoint — ``moonshotai/int4``, Singapore-hosted, ZDR-flagged.

The quantization floor and US allowlist are deliberately absent: no other
provider serves K3 until its open weights land, so any stricter policy
matches zero endpoints and every request 404s.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/agent-harness && uv run pytest tests/unit/test_openrouter_provider.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/code/agent-harness
git add agent_harness/providers/openrouter.py tests/unit/test_openrouter_provider.py
git commit -m "feat(openrouter): RoutingPolicy value type and verified preset policies"
```

---

### Task 2: `OpenRouterProvider` (auth + transport)

**Files:**
- Modify: `agent_harness/providers/openrouter.py`
- Test: `tests/unit/test_openrouter_provider.py`

**Interfaces:**
- Consumes: `RoutingPolicy` from Task 1.
- Produces: `OpenRouterProvider(api_key=…, credential=…, credential_resolver=…, base_url=…, client=…, timeout=…, max_retries=…)` with class attribute `name = "openrouter"` and `OPENROUTER_BASE_URL: str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_openrouter_provider.py`:

```python
import sys
import types
from typing import Any, cast

import pytest

from agent_harness.core.credentials import ApiKeyCredential, OAuthCredential
from agent_harness.core.errors import ConfigError, NotSupportedError
from agent_harness.core.models import Provider
from agent_harness.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
)


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
    assert mod.captured["base_url"] == OPENROUTER_BASE_URL  # type: ignore[attr-defined]
    assert mod.captured["api_key"] == "sk-or-test"  # type: ignore[attr-defined]


def test_provider_honours_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    OpenRouterProvider(api_key="sk-or-test", base_url="https://proxy.example/v1")
    assert mod.captured["base_url"] == "https://proxy.example/v1"  # type: ignore[attr-defined]


def test_provider_accepts_openrouter_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    OpenRouterProvider(credential=ApiKeyCredential(provider="openrouter", key="sk-or-1"))
    assert mod.captured["api_key"] == "sk-or-1"  # type: ignore[attr-defined]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/agent-harness && uv run pytest tests/unit/test_openrouter_provider.py -v`
Expected: FAIL with `ImportError: cannot import name 'OpenRouterProvider'`

- [ ] **Step 3: Write minimal implementation**

Add to `agent_harness/providers/openrouter.py` (and extend `__all__` with `"OPENROUTER_BASE_URL"` and `"OpenRouterProvider"`):

```python
from agent_harness.core.credentials import Credential, CredentialResolver

from .anthropic import AnthropicProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
"""OpenRouter's API root. Its ``/messages`` path is Anthropic-compatible."""


class OpenRouterProvider(AnthropicProvider):
    """Auth + transport for OpenRouter's Anthropic-compatible surface.

    Identical to :class:`~agent_harness.providers.anthropic.AnthropicProvider`
    except that it defaults ``base_url`` to OpenRouter and identifies as
    ``"openrouter"`` — which makes the credential guard require an
    ``ApiKeyCredential(provider="openrouter", …)``, so an Anthropic key can
    never be wired here by accident.

    Example:
        >>> # OpenRouterProvider(api_key="sk-or-…")  # doctest: +SKIP
    """

    name: str = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credential: Credential | None = None,
        credential_resolver: CredentialResolver | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            api_key=api_key,
            credential=credential,
            credential_resolver=credential_resolver,
            base_url=base_url if base_url is not None else OPENROUTER_BASE_URL,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/agent-harness && uv run pytest tests/unit/test_openrouter_provider.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/code/agent-harness
git add agent_harness/providers/openrouter.py tests/unit/test_openrouter_provider.py
git commit -m "feat(openrouter): OpenRouterProvider with openrouter-scoped credentials"
```

---

### Task 3: `OpenRouterModel` — inject routing into the payload

**Files:**
- Modify: `agent_harness/providers/openrouter.py`
- Test: `tests/unit/test_openrouter_provider.py`

**Interfaces:**
- Consumes: `RoutingPolicy`, `US_FP8_ZDR`, `MOONSHOT_DIRECT`, `OpenRouterProvider`.
- Produces: `OpenRouterModel(provider=…, name=…, capabilities=…, routing=…)`; constants `GLM_5_2: str`, `KIMI_K3: str`, `CAPS_GLM_5_2: ModelCapabilities`, `CAPS_KIMI_K3: ModelCapabilities`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_openrouter_provider.py`:

```python
from agent_harness.core.models import Message, Model, ModelSettings, TextBlock
from agent_harness.providers.openrouter import (
    CAPS_GLM_5_2,
    CAPS_KIMI_K3,
    GLM_5_2,
    KIMI_K3,
    OpenRouterModel,
)


def _model(**kw: Any) -> OpenRouterModel:
    return OpenRouterModel(provider=OpenRouterProvider(client=object()), **kw)


def _msgs() -> list[Message]:
    return [Message(role="user", content=[TextBlock(text="hi")])]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/agent-harness && uv run pytest tests/unit/test_openrouter_provider.py -v`
Expected: FAIL with `ImportError: cannot import name 'OpenRouterModel'`

- [ ] **Step 3: Write minimal implementation**

Add to `agent_harness/providers/openrouter.py` (extend `__all__` with `"CAPS_GLM_5_2"`, `"CAPS_KIMI_K3"`, `"GLM_5_2"`, `"KIMI_K3"`, `"OpenRouterModel"`):

```python
from agent_harness.core.models import Message, ModelCapabilities, ModelSettings

from .anthropic import AnthropicMessagesModel

GLM_5_2 = "z-ai/glm-5.2"
"""Zhipu / Z.ai GLM-5.2 on OpenRouter."""

KIMI_K3 = "moonshotai/kimi-k3"
"""Moonshot AI Kimi K3 on OpenRouter."""

CAPS_GLM_5_2 = ModelCapabilities(
    parallel_tool_calls=True,
    thinking=True,
    cache_control=False,
    vision=False,
    audio_input=False,
    audio_output=False,
    structured_output=True,
    context_window=1_048_576,
    max_output_tokens=131_072,
    supports_compaction=False,
)
"""GLM-5.2 limits, from the cheapest routable endpoint (``novita/fp8``).

``cache_control`` is False: OpenRouter's automatic prompt caching on the
Anthropic Skin is documented for Claude models only.
"""

CAPS_KIMI_K3 = ModelCapabilities(
    parallel_tool_calls=True,
    thinking=True,
    cache_control=False,
    vision=True,
    audio_input=False,
    audio_output=False,
    structured_output=True,
    context_window=1_048_576,
    max_output_tokens=131_072,
    supports_compaction=False,
)
"""Kimi K3 limits. Multimodal input; reasoning is always on upstream."""


class OpenRouterModel(AnthropicMessagesModel):
    """An OpenRouter-served model driven over the Anthropic Messages format.

    Adds one thing to its parent: every request carries a
    :class:`RoutingPolicy` in ``extra_body.provider``, so an unconfigured call
    is still constrained to the vetted provider set. Without it OpenRouter
    load-balances by inverse-square price weighting, which pulls hard toward
    the cheapest — typically FP4-quantized — endpoint.

    A caller who puts their own ``provider`` object in
    ``ModelSettings.extra["extra_body"]`` wins; the policy is a default, not a
    cage.

    Example:
        >>> # OpenRouterModel(provider=p, name=GLM_5_2, capabilities=CAPS_GLM_5_2)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        provider: AnthropicProvider,
        name: str = GLM_5_2,
        capabilities: ModelCapabilities | None = None,
        routing: RoutingPolicy | None = None,
    ) -> None:
        super().__init__(provider=provider, name=name, capabilities=capabilities)
        self.routing = routing if routing is not None else US_FP8_ZDR

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> dict[str, Any]:
        payload = super()._build_payload(messages, tools, settings)
        # The parent merged settings.extra already, so anything the caller put
        # in extra_body is present here; setdefault leaves their policy intact.
        extra_body: dict[str, Any] = dict(payload.get("extra_body") or {})
        extra_body.setdefault("provider", self.routing.to_wire())
        payload["extra_body"] = extra_body
        return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/agent-harness && uv run pytest tests/unit/test_openrouter_provider.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Run the full gate**

Run: `cd ~/code/agent-harness && uv run pre-commit run --all-files && uv run pytest`
Expected: ruff, ruff-format, and mypy --strict clean; whole suite passes.

- [ ] **Step 6: Commit**

```bash
cd ~/code/agent-harness
git add agent_harness/providers/openrouter.py tests/unit/test_openrouter_provider.py
git commit -m "feat(openrouter): OpenRouterModel injecting the routing policy into every request"
```

---

### Task 4: README usage snippet

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add the snippet**

Add beside the existing provider examples in `README.md`:

````markdown
### OpenRouter (GLM-5.2, Kimi K3, …)

OpenRouter's `/api/v1/messages` endpoint is Anthropic-compatible, so the
Anthropic model adapter drives it directly. Every request carries a routing
policy restricting which upstream providers may serve it:

```python
from agent_harness.providers.openrouter import (
    CAPS_GLM_5_2, GLM_5_2, OpenRouterModel, OpenRouterProvider,
)

provider = OpenRouterProvider(api_key="sk-or-…")
model = OpenRouterModel(provider=provider, name=GLM_5_2, capabilities=CAPS_GLM_5_2)
agent = Agent(name="assistant", model=model)
```

The default `US_FP8_ZDR` policy routes only to US-headquartered,
FP8-or-better, zero-data-retention endpoints, cheapest first. For Kimi K3,
pass `routing=MOONSHOT_DIRECT` — it has a single upstream endpoint
(Singapore, int4), so the stricter default would match nothing.
````

- [ ] **Step 2: Commit**

```bash
cd ~/code/agent-harness
git add README.md
git commit -m "docs: OpenRouter provider usage"
```

---

## Deferred / follow-up (explicitly out of scope)

- **Live smoke test.** Every test here is mocked. Before trusting this in
  production, run one real request per model with a funded OpenRouter key and
  confirm (a) the Anthropic Skin serves *non-Claude* models correctly, and
  (b) `usage` accounting arrives. Model mapping is documented but unverified
  for GLM-5.2 and K3 specifically.
- **Thinking-budget mapping.** `reasoning` appears on OpenRouter's
  `/chat/completions` schema but **not** on `/messages`. The parent adapter
  sends Anthropic-style `thinking: {budget_tokens: N}`; how the Skin maps that
  onto GLM-5.2's High/Max effort levels (and K3's always-on reasoning) is
  unconfirmed. Verify with the smoke test before relying on `thinking_budget`.
- **A generic `ChatCompletionsModel`.** Deliberately not built. It would serve
  Groq/Together/vLLM directly, but no second caller exists yet — add the knob
  when one does.
- **Preflight policy validation.** Considered and declined: a helper that
  resolves a policy against the live endpoints + ZDR APIs would catch
  "no route matches" at construction instead of at request time.
- **Penny consumption.** Penny pins `agent-harness@v0.2.0`; bumping that pin
  to use this provider is a separate change in the Penny repo.
