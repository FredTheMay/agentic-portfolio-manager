"""Backtest executor: applies a fill model and reports what happened.

Reports its own limitations rather than hiding them — it ignores participation
limits and urgency, and :meth:`SimulatedExecutor.capabilities` says so. An
executor that silently drops a constraint teaches the decision layer that the
constraint works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator

from src.decision.mandate import RebalanceMandate
from src.execution.base import (
    Account,
    Capabilities,
    ExecutionProvider,
    ExecutionReport,
    ExecutionUpdate,
    Fill,
    MarketSnapshot,
    PositionSnapshot,
    Rejection,
    RejectionCode,
    Side,
    implementation_shortfall_bps,
    size_orders,
)
from src.execution.fill_models import FillModel, InstantFillModel

ZERO = Decimal(0)
ONE = Decimal(1)

ENGINE_NAME = "SimulatedExecutor"
ENGINE_VERSION = "1.0.0"


@dataclass(slots=True)
class SimulatedExecutor(ExecutionProvider):
    """In-process executor for backtests.

    ``allow_short_cash`` is off: an order that would overdraw cash is rejected
    rather than silently creating leverage the IPS forbids. Rejecting is also
    the honest simulation — a real broker would decline it too.
    """

    fill_model: FillModel = field(default_factory=InstantFillModel)
    allow_short_cash: bool = False

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_name=f"{ENGINE_NAME}/{self.fill_model.name}",
            engine_version=ENGINE_VERSION,
            supports_intraday=False,
            # Stated plainly: this executor cannot respect a participation cap,
            # so the caller must treat that constraint as advisory.
            supports_participation_limits=False,
            supports_streaming_updates=True,
        )

    def execute(
        self,
        mandate: RebalanceMandate,
        account: Account,
        market: MarketSnapshot,
    ) -> Iterator[ExecutionUpdate]:
        orders, rejections = size_orders(mandate, account, market)

        starting_value = account.total_value(market.prices)
        positions = dict(account.positions)
        cash = account.cash
        fills: list[Fill] = []

        for order in orders:
            price = market.price(order.symbol)
            if price is None:
                continue

            produced = self.fill_model.fill(order, price, market.timestamp)
            cost = sum((f.notional for f in produced), ZERO)
            commission = sum((f.commission for f in produced), ZERO)

            if order.side is Side.BUY and not self.allow_short_cash:
                if cost + commission > cash:
                    rejection = Rejection(
                        order.symbol,
                        RejectionCode.INSUFFICIENT_CASH,
                        f"buy of {cost + commission} exceeds available cash {cash}",
                    )
                    rejections.append(rejection)
                    yield rejection
                    continue

            for produced_fill in produced:
                positions[order.symbol] = positions.get(order.symbol, 0) + produced_fill.quantity
                cash -= Decimal(produced_fill.quantity) * produced_fill.price
                cash -= produced_fill.commission
                fills.append(produced_fill)
                yield produced_fill

        positions = {symbol: qty for symbol, qty in positions.items() if qty != 0}
        ending_value = cash + sum(
            (Decimal(qty) * market.prices[s] for s, qty in positions.items() if s in market.prices),
            ZERO,
        )

        snapshots = tuple(
            PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                market_value=Decimal(quantity) * market.prices[symbol],
                weight=(
                    (Decimal(quantity) * market.prices[symbol]) / ending_value
                    if ending_value > ZERO
                    else ZERO
                ),
            )
            for symbol, quantity in sorted(positions.items())
            if symbol in market.prices
        )

        traded_notional = sum((f.notional for f in fills), ZERO)
        realized_turnover = (
            traded_notional / starting_value / Decimal(2) if starting_value > ZERO else ZERO
        )

        report = ExecutionReport(
            mandate_id=mandate.mandate_id,
            final_positions=snapshots,
            realized_turnover=realized_turnover,
            implementation_shortfall_bps=implementation_shortfall_bps(
                fills, dict(market.decision_prices) or dict(market.prices)
            ),
            total_commission=sum((f.commission for f in fills), ZERO),
            rejections=tuple(sorted(rejections, key=lambda r: (r.symbol, r.reason_code.value))),
            fills=tuple(fills),
            final_cash=cash,
        )
        yield report
