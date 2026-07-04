"""Token accounting + cost pricing (host-supplied prices).

The harness core carries no price table — it only knows the
:data:`~agent_harness.core.models.UsagePricer` shape ``(model_name, usage) ->
Cost``. This sub-package is the batteries-included implementation: a
:class:`PriceTable` the embedding app fills with its own per-model rates and a
:func:`price_table_pricer` factory that turns it into a ``UsagePricer`` to pass
to :class:`~agent_harness.core.agent.Agent`.

    from agent_harness.usage import ModelPrice, PriceTable, price_table_pricer

    prices = PriceTable({"claude-opus-4-7": ModelPrice(input_per_mtok=15.0, output_per_mtok=75.0)})
    agent = Agent(name="a", model=model, usage_pricer=price_table_pricer(prices))
"""

from __future__ import annotations

from .counting import ModelPrice, PriceTable, compute_cost, price_table_pricer

__all__ = [
    "ModelPrice",
    "PriceTable",
    "compute_cost",
    "price_table_pricer",
]
