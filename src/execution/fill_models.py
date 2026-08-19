"""Fill models for the backtest executor.

The gap between the two is the strategy's execution-cost sensitivity, and both
are reported. ``InstantFillModel`` fills at the close in full for free —
optimistic by construction, a lower bound on cost rather than an estimate of
it. ``SpreadCrossFillModel`` crosses the spread and charges commission, which
is what a market order actually does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from src.execution.base import Fill, Order, Side

ZERO = Decimal(0)
TWO = Decimal(2)
BPS = Decimal(10_000)

#: Fallback half-spread when no quote is available, in basis points. One basis
#: point each side is generous for a large-cap ETF and thin for a small cap;
#: it is a parameter precisely because it is an assumption.
DEFAULT_SPREAD_BPS = Decimal("2")

#: Per-share commission, the retail-broker convention. Zero at most US brokers
#: today, so it exists mainly to be non-zero in sensitivity tests.
DEFAULT_COMMISSION_PER_SHARE = Decimal("0.005")


@runtime_checkable
class FillModel(Protocol):
    """How an order becomes fills against a price."""

    def fill(self, order: Order, price: Decimal, timestamp: datetime) -> list[Fill]: ...

    @property
    def name(self) -> str: ...


@dataclass(frozen=True, slots=True)
class InstantFillModel:
    """Fills in full at the reference price, free.

    **Optimistic by construction.** No spread, no commission, no market impact,
    and infinite liquidity at the close. Useful as a baseline and as the upper
    bound on what a strategy could earn; never as the headline result.
    """

    venue: str = "SIMULATED"

    @property
    def name(self) -> str:
        return "InstantFillModel"

    def fill(self, order: Order, price: Decimal, timestamp: datetime) -> list[Fill]:
        return [
            Fill(
                symbol=order.symbol,
                quantity=order.signed_quantity,
                price=price,
                timestamp=timestamp,
                venue=self.venue,
                commission=ZERO,
            )
        ]


@dataclass(frozen=True, slots=True)
class SpreadCrossFillModel:
    """Fills at the far side of the spread, plus commission.

    A buyer lifts the offer and a seller hits the bid, so both pay half the
    spread relative to the midpoint. Modelling the mid as the fill price — as
    the instant model effectively does — hands the strategy free money on every
    single trade, and the error compounds with turnover.
    """

    spread_bps: Decimal = DEFAULT_SPREAD_BPS
    commission_per_share: Decimal = DEFAULT_COMMISSION_PER_SHARE
    venue: str = "SIMULATED"

    @property
    def name(self) -> str:
        return "SpreadCrossFillModel"

    def fill(self, order: Order, price: Decimal, timestamp: datetime) -> list[Fill]:
        half_spread = price * (self.spread_bps / BPS) / TWO
        crossed = price + half_spread if order.side is Side.BUY else price - half_spread
        # A spread wide enough to drive the price non-positive is bad data, not
        # a free option.
        if crossed <= ZERO:
            crossed = price

        return [
            Fill(
                symbol=order.symbol,
                quantity=order.signed_quantity,
                price=crossed,
                timestamp=timestamp,
                venue=self.venue,
                commission=self.commission_per_share * Decimal(order.quantity),
            )
        ]


@dataclass(frozen=True, slots=True)
class QueuePositionFillModel:
    """Placeholder for the C++ limit-order-book simulator.

    Deliberately not implemented here. Queue position depends on the order book
    and on the behaviour of everyone else in it, which is the whole reason the
    separate engine exists.
    """

    @property
    def name(self) -> str:
        return "QueuePositionFillModel"

    def fill(self, order: Order, price: Decimal, timestamp: datetime) -> list[Fill]:
        raise NotImplementedError(
            "QueuePositionFillModel is backed by the C++ simulator, which is a "
            "separate project. Use SpreadCrossFillModel until it exists."
        )
