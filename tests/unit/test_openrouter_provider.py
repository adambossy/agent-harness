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
