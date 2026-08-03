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
from typing import Any, Literal

from agent_harness.core.credentials import Credential, CredentialResolver
from agent_harness.core.errors import ConfigError
from agent_harness.core.models import (
    Effort,
    Message,
    ModelCapabilities,
    ModelSettings,
)

from .anthropic import AnthropicMessagesModel, AnthropicProvider

__all__ = [
    "CAPS_GLM_5_2",
    "CAPS_KIMI_K3",
    "EFFORT_LEVELS_BY_MODEL",
    "GLM_5_2",
    "KIMI_K3",
    "MOONSHOT_DIRECT",
    "OPENROUTER_BASE_URL",
    "US_FP8_ZDR",
    "OpenRouterModel",
    "OpenRouterProvider",
    "RoutingPolicy",
    "supported_effort_levels",
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
    data_collection: Literal["deny", "allow"] = "deny"
    require_parameters: bool = True
    sort: Literal["price", "throughput", "latency"] = "price"
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
"""OpenRouter's API root, as the ``anthropic`` SDK wants it.

Deliberately **not** ``https://openrouter.ai/api/v1``: ``AsyncAnthropic``
appends ``/v1/messages`` to its ``base_url``, so the ``/v1`` must be omitted
here or requests go to ``/api/v1/v1/messages`` and get OpenRouter's HTML 404
page instead of the Anthropic-compatible endpoint. See
``test_base_url_resolves_to_the_anthropic_messages_endpoint``, which pins the
fully-resolved URL — mocked tests cannot catch this class of bug.
"""


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

_CAPABILITIES_BY_MODEL: dict[str, ModelCapabilities] = {
    GLM_5_2: CAPS_GLM_5_2,
    KIMI_K3: CAPS_KIMI_K3,
}
"""Known OpenRouter model ids → their verified capabilities.

Looked up in :meth:`OpenRouterModel.__init__` when the caller omits
``capabilities`` explicitly. Deliberately does *not* fall through to the
parent's Anthropic-Opus default — that fallback is exactly what once made an
unconfigured ``OpenRouterModel`` silently claim Opus context/output limits
and ``cache_control=True`` under a GLM name.
"""

EFFORT_LEVELS_BY_MODEL: dict[str, tuple[Effort, ...]] = {
    # OpenRouter's Anthropic-compatible endpoint documents the full
    # output_config.effort vocabulary for every model it serves; whether a
    # given upstream honours the level is up to that upstream, and an
    # unhonoured level fails soft (the request stays valid).
    GLM_5_2: ("low", "medium", "high", "xhigh", "max"),
    KIMI_K3: ("low", "medium", "high", "xhigh", "max"),
}
"""Effort levels each known model accepts, per OpenRouter's endpoint schema.

Keys mirror :data:`_CAPABILITIES_BY_MODEL` — the enumerable catalogue a
consumer lists models from instead of hand-maintaining a parallel table.
"""


def supported_effort_levels(model: str) -> tuple[Effort, ...]:
    """Effort levels ``model`` accepts, resolved from the catalogue.

    Raises :class:`ConfigError` naming the model when it is unknown, so a
    consumer can never offer a level the endpoint would reject.

    Example:
        >>> "xhigh" in supported_effort_levels(KIMI_K3)
        True
    """
    levels = EFFORT_LEVELS_BY_MODEL.get(model)
    if levels is None:
        raise ConfigError(
            f"no known effort levels for OpenRouter model {model!r}; "
            "see EFFORT_LEVELS_BY_MODEL for the supported ids"
        )
    return levels


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
        provider: OpenRouterProvider,
        name: str = GLM_5_2,
        capabilities: ModelCapabilities | None = None,
        routing: RoutingPolicy | None = None,
    ) -> None:
        if capabilities is None:
            capabilities = _CAPABILITIES_BY_MODEL.get(name)
            if capabilities is None:
                raise ConfigError(
                    f"no known capabilities for OpenRouter model {name!r}; pass "
                    "capabilities explicitly for models outside "
                    "_CAPABILITIES_BY_MODEL (GLM_5_2, KIMI_K3)"
                )
        super().__init__(provider=provider, name=name, capabilities=capabilities)
        self.routing = routing if routing is not None else US_FP8_ZDR

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> dict[str, Any]:
        payload = super()._build_payload(messages, tools, settings)
        # The parent merges settings.extra into the top-level payload (order
        # doesn't matter, only that it happens before we read it here), so
        # anything the caller put in extra_body is already present; setdefault
        # leaves their policy intact.
        # `or {}` would short-circuit here, silently coercing falsy non-dicts
        # ([], "", 0) instead of rejecting them; only None means "absent".
        raw_extra_body = payload.get("extra_body")
        if raw_extra_body is None:
            raw_extra_body = {}
        if not isinstance(raw_extra_body, dict):
            # dict("junk") raises a ValueError naming neither the field nor
            # this provider, three frames from where the caller went wrong.
            raise ConfigError(
                "ModelSettings.extra['extra_body'] must be a dict, got "
                f"{type(raw_extra_body).__name__}"
            )
        extra_body: dict[str, Any] = dict(raw_extra_body)
        extra_body.setdefault("provider", self.routing.to_wire())
        payload["extra_body"] = extra_body
        return payload
