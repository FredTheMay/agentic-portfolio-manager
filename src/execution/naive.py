"""Naive executor: whole-delta market orders against a paper broker.

No slicing, no scheduling, no participation control, and
:meth:`NaiveExecutor.capabilities` says so — an executor that silently drops a
constraint teaches the caller that the constraint works.

Paper only. :class:`AlpacaPaperBroker` refuses to construct against the live
endpoint.

Every order carries a ``client_order_id`` derived from the mandate id and
symbol, so a cycle retried after an ambiguous network failure presents the same
id and the broker rejects the duplicate rather than doubling the position.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import httpx

from src.audit.log import AuditEvent, AuditLog, Standard
from src.data.cache import redact
from src.decision.mandate import RebalanceMandate
from src.execution.base import (
    Account,
    Capabilities,
    ExecutionProvider,
    ExecutionReport,
    ExecutionUpdate,
    Fill,
    MarketSnapshot,
    Order,
    PositionSnapshot,
    Rejection,
    RejectionCode,
    Side,
    implementation_shortfall_bps,
    size_orders,
)
from src.time.clock import Clock, WallClock, ensure_utc

ZERO = Decimal(0)

#: Paper trading only. The live host is deliberately absent from this module.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_HOST_FRAGMENT = "api.alpaca.markets"

KEY_ENV = "ALPACA_API_KEY_ID"
SECRET_ENV = "ALPACA_API_SECRET_KEY"

ENGINE_NAME = "NaiveExecutor"
ENGINE_VERSION = "1.0.0"


class BrokerError(RuntimeError):
    """Raised when the broker refuses or fails a request."""


def client_order_id(mandate_id: str, symbol: str) -> str:
    """Deterministic idempotency key for one order within one mandate.

    Derived rather than random so that a retry after an ambiguous failure
    presents the same id and the broker can reject the duplicate.
    """
    return f"{mandate_id}-{symbol}"


@runtime_checkable
class Broker(Protocol):
    """The minimal broker surface this executor needs.

    Injected, so tests exercise the real sizing, idempotency, and reporting
    logic against a fake without a network or an API key.
    """

    def submit_market_order(
        self, symbol: str, side: str, quantity: int, client_order_id: str
    ) -> Mapping[str, Any]: ...

    def positions(self) -> Mapping[str, int]: ...

    def cash(self) -> Decimal: ...


@dataclass
class AlpacaPaperBroker:
    """Alpaca **paper** trading client.

    Refuses to point at the live endpoint. no live broker endpoint,
    ever — enforced here rather than left to a config review.
    """

    key_id: str | None = None
    secret_key: str | None = None
    base_url: str = PAPER_BASE_URL
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if LIVE_HOST_FRAGMENT in self.base_url and "paper" not in self.base_url:
            raise BrokerError(
                f"refusing to trade against {self.base_url}: this system is "
                "paper-trading only"
            )

    def _headers(self) -> dict[str, str]:
        key = self.key_id or os.environ.get(KEY_ENV)
        secret = self.secret_key or os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise BrokerError(
                f"no Alpaca paper credentials: set {KEY_ENV} and {SECRET_ENV}. "
                "Backtests run without them."
            )
        return {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise BrokerError(redact(f"{method} {path} failed: {exc}")) from exc
        if response.status_code >= 400:
            raise BrokerError(
                redact(f"{method} {path} returned {response.status_code}: {response.text[:200]}")
            )
        return response.json()

    def submit_market_order(
        self, symbol: str, side: str, quantity: int, client_order_id: str
    ) -> Mapping[str, Any]:
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        result = self._request("POST", "/v2/orders", payload)
        if not isinstance(result, Mapping):
            raise BrokerError("unexpected order response shape")
        return result

    def positions(self) -> Mapping[str, int]:
        result = self._request("GET", "/v2/positions")
        if not isinstance(result, list):
            raise BrokerError("unexpected positions response shape")
        return {str(p["symbol"]): int(float(p["qty"])) for p in result}

    def cash(self) -> Decimal:
        result = self._request("GET", "/v2/account")
        if not isinstance(result, Mapping):
            raise BrokerError("unexpected account response shape")
        return Decimal(str(result["cash"]))


@dataclass
class NaiveExecutor(ExecutionProvider):
    """Sends market orders for the whole delta, immediately.

    No slicing, no scheduling, no participation control. That is the point: it
    is the honest floor, and the C++ engine's value will be measured against it.
    """

    broker: Broker
    clock: Clock = field(default_factory=WallClock)
    audit: AuditLog | None = None
    #: Client order ids already sent, for in-process replay protection.
    submitted: set[str] = field(default_factory=set)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_name=ENGINE_NAME,
            engine_version=ENGINE_VERSION,
            supports_intraday=True,
            # Stated plainly rather than silently ignored.
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
        fills: list[Fill] = []
        positions = dict(account.positions)
        cash = account.cash
        timestamp = ensure_utc(self.clock.now())

        for order in orders:
            key = client_order_id(mandate.mandate_id, order.symbol)
            if key in self.submitted:
                rejection = Rejection(
                    order.symbol,
                    RejectionCode.BELOW_MIN_NOTIONAL,
                    f"duplicate order {key} suppressed by idempotency key",
                )
                rejections.append(rejection)
                yield rejection
                continue

            try:
                response = self.broker.submit_market_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=order.quantity,
                    client_order_id=key,
                )
            except BrokerError as exc:
                rejection = Rejection(order.symbol, RejectionCode.NO_PRICE, str(exc))
                rejections.append(rejection)
                yield rejection
                continue

            self.submitted.add(key)
            fill = _fill_from_response(order, response, timestamp)
            positions[order.symbol] = positions.get(order.symbol, 0) + fill.quantity
            cash -= Decimal(fill.quantity) * fill.price + fill.commission
            fills.append(fill)

            if self.audit is not None:
                self.audit.record(
                    AuditEvent(
                        timestamp=timestamp,
                        actor=ENGINE_NAME,
                        code="ORDER_SUBMITTED",
                        standard=Standard.III_A_LOYALTY,
                        symbol=order.symbol,
                        detail=(
                            f"{order.side.value} {order.quantity} at ~{fill.price} "
                            f"(mandate {mandate.mandate_id})"
                        ),
                    )
                )
            yield fill

        positions = {symbol: qty for symbol, qty in positions.items() if qty != 0}
        value = cash + sum(
            (Decimal(q) * market.prices[s] for s, q in positions.items() if s in market.prices),
            ZERO,
        )
        snapshots = tuple(
            PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                market_value=Decimal(quantity) * market.prices[symbol],
                weight=(Decimal(quantity) * market.prices[symbol]) / value if value > ZERO else ZERO,
            )
            for symbol, quantity in sorted(positions.items())
            if symbol in market.prices
        )

        starting = account.total_value(market.prices)
        traded = sum((f.notional for f in fills), ZERO)
        yield ExecutionReport(
            mandate_id=mandate.mandate_id,
            final_positions=snapshots,
            realized_turnover=traded / starting / Decimal(2) if starting > ZERO else ZERO,
            implementation_shortfall_bps=implementation_shortfall_bps(
                fills, dict(market.decision_prices) or dict(market.prices)
            ),
            total_commission=sum((f.commission for f in fills), ZERO),
            rejections=tuple(sorted(rejections, key=lambda r: (r.symbol, r.reason_code.value))),
            fills=tuple(fills),
            final_cash=cash,
        )


def _fill_from_response(
    order: Order,
    response: Mapping[str, Any],
    fallback_time: datetime,
) -> Fill:
    """Build a fill from a broker order response.

    A market order submitted to a paper endpoint may come back accepted but not
    yet filled. Falling back to the reference price is an approximation, and it
    is exactly the kind of approximation the implementation-shortfall figure
    exists to expose.
    """
    filled_price = response.get("filled_avg_price")
    price = Decimal(str(filled_price)) if filled_price else order.reference_price

    filled_qty = response.get("filled_qty")
    quantity = int(float(filled_qty)) if filled_qty else order.quantity
    signed = quantity if order.side is Side.BUY else -quantity

    stamp = response.get("filled_at") or response.get("submitted_at")
    timestamp = fallback_time
    if isinstance(stamp, str):
        try:
            timestamp = ensure_utc(datetime.fromisoformat(stamp.replace("Z", "+00:00")))
        except ValueError:
            timestamp = fallback_time

    return Fill(
        symbol=order.symbol,
        quantity=signed,
        price=price,
        timestamp=timestamp,
        venue="ALPACA_PAPER",
        commission=ZERO,
    )


def account_from_broker(broker: Broker) -> Account:
    """Read live holdings and cash into an :class:`Account`."""
    return Account(cash=broker.cash(), positions=dict(broker.positions()))


__all__ = [
    "AlpacaPaperBroker",
    "Broker",
    "BrokerError",
    "NaiveExecutor",
    "PAPER_BASE_URL",
    "account_from_broker",
    "client_order_id",
]
