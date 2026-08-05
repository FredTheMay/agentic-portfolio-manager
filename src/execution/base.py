"""The execution boundary, Python side (SPEC §3).

Everything below the boundary lives in this package: sizing, orders, venues,
fills, brokers. The decision layer hands down a
:class:`~src.decision.mandate.RebalanceMandate` of target weights and gets back
an :class:`ExecutionReport`. That is the whole surface.

Sizing lives here rather than upstream because share counts are a function of
weights, prices, and portfolio value *at execution time*. A real algorithm
recomputes them as it works an order; a fixed share count decided minutes
earlier is already stale.

These types mirror ``proto/execution.proto`` one-for-one, so the future C++
engine implements the same service with no renegotiation. Swapping executors is
one config value (:func:`get_executor`).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Iterator, Mapping

from src.decision.mandate import RebalanceMandate
from src.time.clock import ensure_utc

ZERO = Decimal(0)
ONE = Decimal(1)
BPS = Decimal(10_000)

MONEY = Decimal("0.01")


class Side(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class RejectionCode(str, enum.Enum):
    """Why an order was not sent or not filled."""

    BELOW_MIN_NOTIONAL = "BELOW_MIN_NOTIONAL"
    NO_PRICE = "NO_PRICE"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    TURNOVER_EXCEEDED = "TURNOVER_EXCEEDED"
    NOT_SHORTABLE = "NOT_SHORTABLE"


class ExecutionError(RuntimeError):
    """Raised when a mandate cannot be executed at all."""


@dataclass(frozen=True, slots=True)
class Order:
    """An intent to trade a whole number of shares.

    Share counts are integral because fractional-share support varies by venue
    and rounding a fraction at fill time silently changes the position.
    """

    symbol: str
    side: Side
    quantity: int
    reference_price: Decimal

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ExecutionError(f"order quantity must be positive, got {self.quantity}")

    @property
    def notional(self) -> Decimal:
        return Decimal(self.quantity) * self.reference_price

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side is Side.BUY else -self.quantity


@dataclass(frozen=True, slots=True)
class Fill:
    """An execution. Mirrors ``Fill`` in the proto."""

    symbol: str
    quantity: int
    price: Decimal
    timestamp: datetime
    venue: str = "SIMULATED"
    commission: Decimal = ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def notional(self) -> Decimal:
        return Decimal(abs(self.quantity)) * self.price


@dataclass(frozen=True, slots=True)
class Rejection:
    """A trade that was not done, and why. Mirrors ``Rejection`` in the proto."""

    symbol: str
    reason_code: RejectionCode
    detail: str


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """A holding after execution. Mirrors ``PositionSnapshot`` in the proto."""

    symbol: str
    quantity: int
    market_value: Decimal
    weight: Decimal


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """The completion message. Mirrors ``ExecutionReport`` in the proto."""

    mandate_id: str
    final_positions: tuple[PositionSnapshot, ...]
    realized_turnover: Decimal
    implementation_shortfall_bps: Decimal
    total_commission: Decimal
    rejections: tuple[Rejection, ...]
    fills: tuple[Fill, ...] = ()
    final_cash: Decimal = ZERO

    @property
    def realized_weights(self) -> dict[str, Decimal]:
        return {p.symbol: p.weight for p in self.final_positions}


#: One streaming update. The proto models this as a oneof.
ExecutionUpdate = Fill | Rejection | ExecutionReport


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What an executor can actually honor (SPEC §3.2).

    Lets the decision layer degrade gracefully: an executor reporting
    ``supports_participation_limits = False`` means that constraint is
    advisory, and the caller must log it rather than assume it was respected.
    """

    engine_name: str
    engine_version: str
    supports_intraday: bool = False
    supports_participation_limits: bool = False
    supports_streaming_updates: bool = True


@dataclass(frozen=True, slots=True)
class Account:
    """Holdings and cash, in shares and currency — never weights.

    Weights are the decision layer's language. Below the boundary everything is
    share counts and cash, because that is what actually settles.
    """

    cash: Decimal
    positions: Mapping[str, int] = field(default_factory=dict)

    def market_value(self, prices: Mapping[str, Decimal]) -> Decimal:
        return sum(
            (Decimal(qty) * prices[symbol] for symbol, qty in self.positions.items() if symbol in prices),
            ZERO,
        )

    def total_value(self, prices: Mapping[str, Decimal]) -> Decimal:
        return self.cash + self.market_value(prices)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Prices at the moment of execution.

    ``decision_prices`` are the prices that prevailed when the mandate was
    decided, and exist solely to measure implementation shortfall against.
    """

    timestamp: datetime
    prices: Mapping[str, Decimal]
    spreads_bps: Mapping[str, Decimal] = field(default_factory=dict)
    decision_prices: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def price(self, symbol: str) -> Decimal | None:
        return self.prices.get(symbol)

    def decision_price(self, symbol: str) -> Decimal | None:
        return self.decision_prices.get(symbol, self.prices.get(symbol))


class ExecutionProvider(ABC):
    """A thing that can turn a mandate into fills.

    Implementations: :class:`~src.execution.simulated.SimulatedExecutor` (M5),
    ``NaiveExecutor`` against a paper broker (M8), and eventually a gRPC client
    pointing at the C++ engine. Adding the third requires zero changes outside
    this package.
    """

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """What this executor can honor."""

    @abstractmethod
    def execute(
        self,
        mandate: RebalanceMandate,
        account: Account,
        market: MarketSnapshot,
    ) -> Iterator[ExecutionUpdate]:
        """Work the mandate, streaming fills and finishing with a report."""

    def execute_to_completion(
        self,
        mandate: RebalanceMandate,
        account: Account,
        market: MarketSnapshot,
    ) -> ExecutionReport:
        """Drain :meth:`execute` and return the final report."""
        report: ExecutionReport | None = None
        for update in self.execute(mandate, account, market):
            if isinstance(update, ExecutionReport):
                report = update
        if report is None:
            raise ExecutionError("executor finished without emitting a completion report")
        return report


# ---------------------------------------------------------------------------
# Sizing — weights to share counts
# ---------------------------------------------------------------------------


def size_orders(
    mandate: RebalanceMandate,
    account: Account,
    market: MarketSnapshot,
) -> tuple[list[Order], list[Rejection]]:
    """Convert target weights into whole-share orders at current prices.

    This is the step the decision layer deliberately does not do. Portfolio
    value is recomputed here, from prices now, so a mandate decided minutes ago
    still sizes correctly.

    ``min_trade_notional`` is enforced here rather than upstream: it is a
    property of trading costs, and the decision layer has no notion of a trade.
    """
    orders: list[Order] = []
    rejections: list[Rejection] = []

    portfolio_value = account.total_value(market.prices)
    if portfolio_value <= ZERO:
        raise ExecutionError(f"portfolio value must be positive, got {portfolio_value}")

    for target in mandate.targets:
        price = market.price(target.symbol)
        if price is None or price <= ZERO:
            rejections.append(
                Rejection(
                    target.symbol,
                    RejectionCode.NO_PRICE,
                    "no usable price at execution time",
                )
            )
            continue

        desired_notional = target.target_weight * portfolio_value
        # Round toward zero: overshooting a target weight can breach a limit
        # the risk engine just finished enforcing.
        desired_shares = int((desired_notional / price).to_integral_value(rounding=ROUND_DOWN))
        delta = desired_shares - account.positions.get(target.symbol, 0)
        if delta == 0:
            continue

        notional = Decimal(abs(delta)) * price
        if notional < mandate.constraints.min_trade_notional:
            rejections.append(
                Rejection(
                    target.symbol,
                    RejectionCode.BELOW_MIN_NOTIONAL,
                    f"trade notional {notional.quantize(MONEY)} is below the "
                    f"{mandate.constraints.min_trade_notional} minimum",
                )
            )
            continue

        orders.append(
            Order(
                symbol=target.symbol,
                side=Side.BUY if delta > 0 else Side.SELL,
                quantity=abs(delta),
                reference_price=price,
            )
        )

    # Deterministic order: SPEC §9 requires identical runs to produce an
    # identical trade log, and dict iteration order is not a guarantee to rely on.
    orders.sort(key=lambda o: o.symbol)
    rejections.sort(key=lambda r: r.symbol)
    return orders, rejections


def implementation_shortfall_bps(
    fills: Mapping[str, list[Fill]] | list[Fill],
    decision_prices: Mapping[str, Decimal],
) -> Decimal:
    """Notional-weighted cost of trading away from the decision price.

    Positive means the trading cost money: buys filled above, or sells filled
    below, the price that prevailed when the decision was made. This is the
    number that separates a backtest from a claim — quoting returns without it
    assumes execution was free.
    """
    flat = fills if isinstance(fills, list) else [f for group in fills.values() for f in group]

    total_notional = ZERO
    weighted = ZERO
    for fill in flat:
        reference = decision_prices.get(fill.symbol)
        if reference is None or reference <= ZERO or fill.quantity == 0:
            continue
        # Sign by direction: paying more on a buy and receiving less on a sell
        # are both costs.
        direction = ONE if fill.quantity > 0 else -ONE
        slippage = direction * (fill.price - reference) / reference
        notional = fill.notional
        weighted += slippage * notional
        total_notional += notional

    if total_notional <= ZERO:
        return ZERO
    return (weighted / total_notional) * BPS
