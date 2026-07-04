"""Cost pricing for token :class:`~agent_harness.core.models.Usage`.

A deep, host-agnostic module: the app declares *what a model costs* as a
:class:`PriceTable`; this module turns tokens into a
:class:`~agent_harness.core.models.Cost` and packages the whole thing as a
:data:`~agent_harness.core.models.UsagePricer` the loop can call. Prices are
quoted the way vendors quote them — dollars per **million** tokens — so a
host can copy a pricing page verbatim.

No vendor prices are baked in: an empty :class:`PriceTable` prices every model
at zero, and an unknown model is treated the same way (see
:meth:`PriceTable.price_for`). The host owns the numbers; the harness owns the
arithmetic.

Example:
    >>> from agent_harness.core.models import Usage
    >>> prices = PriceTable({"m": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)})
    >>> pricer = price_table_pricer(prices)
    >>> round(pricer("m", Usage(input_tokens=1_000_000, output_tokens=1_000_000)).total, 2)
    18.0
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_harness.core.models import Cost, Usage, UsagePricer

TOKENS_PER_MILLION = 1_000_000
"""Prices are per-million-tokens; divide token counts by this to bill."""


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Per-million-token prices for a single model, in one currency (USD by
    convention). Cache rates default to the input rate when unset, matching
    the common vendor default that a cache *read* is billed like input and a
    cache *write* like input unless stated otherwise.

    Example:
        >>> ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0).output_per_mtok
        15.0
    """

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def _cache_read(self) -> float:
        return (
            self.cache_read_per_mtok
            if self.cache_read_per_mtok is not None
            else self.input_per_mtok
        )

    def _cache_write(self) -> float:
        return (
            self.cache_write_per_mtok
            if self.cache_write_per_mtok is not None
            else self.input_per_mtok
        )

    def cost_of(self, usage: Usage) -> Cost:
        """Price ``usage`` against these rates.

        Example:
            >>> from agent_harness.core.models import Usage
            >>> ModelPrice(input_per_mtok=3.0).cost_of(Usage(input_tokens=500_000)).total
            1.5
        """
        return Cost(
            input_cost=usage.input_tokens / TOKENS_PER_MILLION * self.input_per_mtok,
            output_cost=usage.output_tokens / TOKENS_PER_MILLION * self.output_per_mtok,
            cache_read_cost=usage.cache_read_tokens / TOKENS_PER_MILLION * self._cache_read(),
            cache_write_cost=usage.cache_write_tokens / TOKENS_PER_MILLION * self._cache_write(),
        )


@dataclass(frozen=True, slots=True)
class PriceTable:
    """Model-name → :class:`ModelPrice`, the app's whole pricing catalog.

    A missing model prices at zero rather than raising: pricing is a
    best-effort observation, never a gate on the run (define errors out of
    existence). A host that wants strictness can inspect
    :meth:`price_for` returning the zero default.

    Example:
        >>> pt = PriceTable({"m": ModelPrice(input_per_mtok=3.0)})
        >>> pt.price_for("m").input_per_mtok
        3.0
        >>> pt.price_for("unknown").input_per_mtok
        0.0
    """

    prices: dict[str, ModelPrice]

    def price_for(self, model_name: str) -> ModelPrice:
        """Return the price for ``model_name``, or the zero price if absent."""
        return self.prices.get(model_name, _ZERO_PRICE)


_ZERO_PRICE = ModelPrice()


def compute_cost(usage: Usage, price: ModelPrice) -> Cost:
    """Price ``usage`` against ``price`` — the free-function form of
    :meth:`ModelPrice.cost_of`.

    Example:
        >>> from agent_harness.core.models import Usage
        >>> compute_cost(Usage(output_tokens=2_000_000), ModelPrice(output_per_mtok=15.0)).total
        30.0
    """
    return price.cost_of(usage)


def price_table_pricer(price_table: PriceTable) -> UsagePricer:
    """Adapt a :class:`PriceTable` into a
    :data:`~agent_harness.core.models.UsagePricer` for
    :class:`~agent_harness.core.agent.Agent`.

    Example:
        >>> from agent_harness.core.models import Usage
        >>> pricer = price_table_pricer(PriceTable({"m": ModelPrice(input_per_mtok=3.0)}))
        >>> pricer("m", Usage(input_tokens=1_000_000)).input_cost
        3.0
    """

    def _price(model_name: str, usage: Usage) -> Cost:
        return price_table.price_for(model_name).cost_of(usage)

    return _price
