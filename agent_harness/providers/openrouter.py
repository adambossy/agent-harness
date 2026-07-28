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
