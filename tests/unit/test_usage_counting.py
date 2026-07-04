"""Unit tests for ``agent_harness.usage.counting`` + the ``ModelUsage`` event.

Covers the pure pricing arithmetic and the loop integration: an agent given a
``usage_pricer`` publishes a ``ModelUsage`` event with correct token/cost
fields; an agent without one publishes none.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import pytest

from agent_harness.core.agent import Agent
from agent_harness.core.events import Event, ModelUsage
from agent_harness.core.models import Cost, Model, Usage
from agent_harness.usage.counting import (
    ModelPrice,
    PriceTable,
    compute_cost,
    price_table_pricer,
)
from tests.fakes import FakeTurn, make_model

# --- pure pricing -----------------------------------------------------------


def test_cost_sums_and_totals() -> None:
    c = Cost(input_cost=0.01, output_cost=0.02) + Cost(cache_read_cost=0.03)
    assert c.total == pytest.approx(0.06)


def test_model_price_prices_every_token_category() -> None:
    price = ModelPrice(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_read_per_mtok=0.3,
        cache_write_per_mtok=3.75,
    )
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    cost = compute_cost(usage, price)
    assert cost.input_cost == pytest.approx(3.0)
    assert cost.output_cost == pytest.approx(15.0)
    assert cost.cache_read_cost == pytest.approx(0.3)
    assert cost.cache_write_cost == pytest.approx(3.75)
    assert cost.total == pytest.approx(22.05)


def test_cache_rates_default_to_input_rate() -> None:
    price = ModelPrice(input_per_mtok=3.0)
    cost = price.cost_of(Usage(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000))
    assert cost.cache_read_cost == pytest.approx(3.0)
    assert cost.cache_write_cost == pytest.approx(3.0)


def test_unknown_model_prices_at_zero() -> None:
    pricer = price_table_pricer(PriceTable({"known": ModelPrice(input_per_mtok=3.0)}))
    assert pricer("unknown", Usage(input_tokens=1_000_000)).total == 0.0


# --- loop integration -------------------------------------------------------


async def test_agent_with_pricer_publishes_model_usage() -> None:
    # FakeTurn defaults to Usage(input_tokens=100, output_tokens=20).
    prices = PriceTable({"fake-model": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)})
    model = make_model(FakeTurn(text="hi"))
    agent: Agent[None, str] = Agent(
        name="priced",
        model=cast(Model, model),
        usage_pricer=price_table_pricer(prices),
    )

    usages = [e async for e in _stream(agent, "hello") if isinstance(e, ModelUsage)]
    assert len(usages) == 1
    ev = usages[0]
    assert ev.model_name == "fake-model"
    assert ev.usage == Usage(input_tokens=100, output_tokens=20)
    # 100/1e6*3 + 20/1e6*15 = 0.0003 + 0.0003 = 0.0006
    assert ev.cost.total == pytest.approx(0.0006)


async def test_agent_without_pricer_publishes_no_model_usage() -> None:
    model = make_model(FakeTurn(text="hi"))
    agent: Agent[None, str] = Agent(name="unpriced", model=cast(Model, model))
    assert not [e async for e in _stream(agent, "hello") if isinstance(e, ModelUsage)]


async def _stream(agent: Agent[None, str], prompt: str) -> AsyncIterator[Event]:
    """Drive ``agent.stream`` (which owns + closes its own bus)."""
    async for ev in agent.stream(prompt):
        yield ev
