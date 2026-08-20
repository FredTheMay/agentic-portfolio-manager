"""Event-driven market data model.

The engine consumes a stream of timestamped events, so a future streaming
source implements the same protocol and the engine loop is unchanged.
Timestamps are instants, never dates.

Prices are ``Decimal`` and floats are rejected at construction: binary floats
cannot represent decimal cash exactly, and the error accumulates silently
across a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterator, Literal, Protocol, runtime_checkable

from src.time.clock import ensure_utc

EventKind = Literal["BAR", "QUOTE", "TRADE", "CORPORATE_ACTION"]

ActionKind = Literal["SPLIT", "DIVIDEND"]


class MarketDataError(Exception):
    """Raised on a malformed market event."""


def _require_decimal(name: str, value: object) -> Decimal:
    """Return ``value`` as a ``Decimal``, rejecting floats.

    Takes ``object`` rather than ``Decimal`` deliberately: the annotations on
    these dataclasses are a promise, and this is the runtime check that the
    promise was kept. Callers routinely pass floats read from JSON.
    """
    if isinstance(value, float):
        raise MarketDataError(
            f"{name} must be Decimal, got float {value!r} — "
            "money and prices are never float"
        )
    if not isinstance(value, Decimal):
        raise MarketDataError(f"{name} must be Decimal, got {type(value).__name__}")
    return value


def _require_non_negative(name: str, value: Decimal) -> Decimal:
    if value < 0:
        raise MarketDataError(f"{name} must be non-negative, got {value}")
    return value


def _require_positive_decimal(name: str, value: object) -> Decimal:
    return _require_non_negative(name, _require_decimal(name, value))


def _require_share_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarketDataError(
            f"{name} must be an int, got {type(value).__name__} (share counts are integral)"
        )
    return value


@dataclass(frozen=True, slots=True)
class BarPayload:
    """An OHLCV bar.

    ``close`` is the **unadjusted** print, used for share-count arithmetic.
    ``adj_close`` is split- and dividend-adjusted, used for return
    calculation. They are separate fields precisely so the two can never be
    mixed by accident.
    """

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adj_close: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            _require_positive_decimal(name, getattr(self, name))
        if self.adj_close is not None:
            _require_positive_decimal("adj_close", self.adj_close)
        if _require_share_count("volume", self.volume) < 0:
            raise MarketDataError(f"volume must be non-negative, got {self.volume}")
        if self.low > self.high:
            raise MarketDataError(f"low {self.low} exceeds high {self.high}")


@dataclass(frozen=True, slots=True)
class QuotePayload:
    """Top-of-book quote. Feeds ``SpreadCrossFillModel``."""

    bid: Decimal
    ask: Decimal
    bid_size: int = 0
    ask_size: int = 0

    def __post_init__(self) -> None:
        _require_positive_decimal("bid", self.bid)
        _require_positive_decimal("ask", self.ask)
        _require_share_count("bid_size", self.bid_size)
        _require_share_count("ask_size", self.ask_size)
        if self.bid > self.ask:
            raise MarketDataError(f"crossed quote: bid {self.bid} > ask {self.ask}")


@dataclass(frozen=True, slots=True)
class TradePayload:
    """A single executed print on the tape."""

    price: Decimal
    size: int

    def __post_init__(self) -> None:
        _require_positive_decimal("price", self.price)
        if _require_share_count("size", self.size) <= 0:
            raise MarketDataError(f"trade size must be positive, got {self.size}")


@dataclass(frozen=True, slots=True)
class ActionPayload:
    """A corporate action.

    Splits and dividends are handled explicitly rather than being folded
    invisibly into an adjusted price series, because share-count arithmetic
    needs the raw event.

    ``split_ratio`` is new shares per old share (2 for a 2-for-1).
    ``cash_amount`` is per-share cash for a dividend.
    """

    kind: ActionKind
    split_ratio: Decimal | None = None
    cash_amount: Decimal | None = None

    def __post_init__(self) -> None:
        if self.kind == "SPLIT":
            if self.split_ratio is None:
                raise MarketDataError("SPLIT action requires split_ratio")
            if _require_decimal("split_ratio", self.split_ratio) <= 0:
                raise MarketDataError(f"split_ratio must be positive, got {self.split_ratio}")
        elif self.kind == "DIVIDEND":
            if self.cash_amount is None:
                raise MarketDataError("DIVIDEND action requires cash_amount")
            _require_positive_decimal("cash_amount", self.cash_amount)
        else:
            raise MarketDataError(f"unknown corporate action kind {self.kind!r}")


Payload = BarPayload | QuotePayload | TradePayload | ActionPayload

_PAYLOAD_FOR_KIND: dict[str, type] = {
    "BAR": BarPayload,
    "QUOTE": QuotePayload,
    "TRADE": TradePayload,
    "CORPORATE_ACTION": ActionPayload,
}


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """One timestamped observation about one symbol.

    ``timestamp`` is a tz-aware UTC instant and is normalized on construction.
    """

    timestamp: datetime
    symbol: str
    kind: EventKind
    payload: Payload

    def __post_init__(self) -> None:
        # Frozen dataclass: normalize through object.__setattr__.
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if not self.symbol:
            raise MarketDataError("symbol must be non-empty")
        expected = _PAYLOAD_FOR_KIND.get(self.kind)
        if expected is None:
            raise MarketDataError(f"unknown event kind {self.kind!r}")
        if not isinstance(self.payload, expected):
            raise MarketDataError(
                f"kind {self.kind!r} requires {expected.__name__}, "
                f"got {type(self.payload).__name__}"
            )


@runtime_checkable
class MarketDataSource(Protocol):
    """A replayable source of market events.

    Implementations must yield events in non-decreasing timestamp order — the
    backtest clock only moves forwards, and out-of-order events would let the
    system read data it could not have had.

    v1 ships ``DailyBarSource``. A ``StreamingQuoteSource`` implements the
    same protocol over a websocket with no change to the engine.
    """

    def stream(self, start: datetime, end: datetime) -> Iterator[MarketEvent]:
        """Yield events in ``[start, end]``, ordered by timestamp."""
        ...
