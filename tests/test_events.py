"""Unit tests for the event model (SPEC §4.2, §4.4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterator

import pytest

from src.data.events import (
    ActionPayload,
    BarPayload,
    MarketDataError,
    MarketDataSource,
    MarketEvent,
    QuotePayload,
    TradePayload,
)
from src.time.clock import UTC

D = Decimal
TS = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def a_bar(**overrides: object) -> BarPayload:
    kwargs: dict[str, object] = {
        "open": D("100.00"),
        "high": D("102.50"),
        "low": D("99.75"),
        "close": D("101.25"),
        "volume": 1_000_000,
    }
    kwargs.update(overrides)
    return BarPayload(**kwargs)  # type: ignore[arg-type]


def test_bar_accepts_decimals() -> None:
    bar = a_bar(adj_close=D("101.25"))
    assert bar.close == D("101.25")
    assert bar.adj_close == D("101.25")


def test_bar_rejects_float_prices() -> None:
    # SPEC §9: money is never float. Binary rounding error compounds silently
    # across a backtest, so it is refused at construction.
    with pytest.raises(MarketDataError, match="never float"):
        a_bar(close=101.25)


def test_bar_rejects_inconsistent_range() -> None:
    with pytest.raises(MarketDataError, match="exceeds high"):
        a_bar(low=D("110"), high=D("105"))


def test_bar_rejects_non_integer_volume() -> None:
    with pytest.raises(MarketDataError, match="volume must be an int"):
        a_bar(volume=D("1000"))


def test_bar_keeps_adjusted_and_unadjusted_separate() -> None:
    # SPEC §4.4: adjusted for returns, unadjusted for share arithmetic, never
    # mixed. Two fields make that mistake impossible to make silently.
    bar = a_bar(close=D("200.00"), adj_close=D("100.00"))
    assert bar.close != bar.adj_close
    assert a_bar().adj_close is None


def test_quote_rejects_crossed_market() -> None:
    with pytest.raises(MarketDataError, match="crossed quote"):
        QuotePayload(bid=D("10.05"), ask=D("10.00"))


def test_trade_requires_positive_size() -> None:
    with pytest.raises(MarketDataError):
        TradePayload(price=D("10"), size=0)


def test_split_action_requires_ratio() -> None:
    assert ActionPayload(kind="SPLIT", split_ratio=D("2")).split_ratio == D("2")
    with pytest.raises(MarketDataError, match="requires split_ratio"):
        ActionPayload(kind="SPLIT")


def test_dividend_action_requires_cash_amount() -> None:
    assert ActionPayload(kind="DIVIDEND", cash_amount=D("0.24")).cash_amount == D("0.24")
    with pytest.raises(MarketDataError, match="requires cash_amount"):
        ActionPayload(kind="DIVIDEND")


def test_event_normalizes_timestamp_to_utc() -> None:
    eastern = timezone(timedelta(hours=-5))
    event = MarketEvent(
        timestamp=datetime(2024, 1, 3, 16, 0, tzinfo=eastern),
        symbol="SPY",
        kind="BAR",
        payload=a_bar(),
    )
    assert event.timestamp == datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def test_event_rejects_naive_timestamp() -> None:
    from src.time.clock import ClockError

    with pytest.raises(ClockError):
        MarketEvent(timestamp=datetime(2024, 1, 3, 16, 0), symbol="SPY", kind="BAR", payload=a_bar())


def test_event_rejects_payload_kind_mismatch() -> None:
    with pytest.raises(MarketDataError, match="requires BarPayload"):
        MarketEvent(
            timestamp=TS, symbol="SPY", kind="BAR", payload=TradePayload(price=D("1"), size=1)
        )


def test_event_rejects_empty_symbol() -> None:
    with pytest.raises(MarketDataError, match="symbol"):
        MarketEvent(timestamp=TS, symbol="", kind="BAR", payload=a_bar())


def test_event_is_frozen_and_hashable() -> None:
    event = MarketEvent(timestamp=TS, symbol="SPY", kind="BAR", payload=a_bar())
    with pytest.raises(Exception):
        event.symbol = "QQQ"  # type: ignore[misc]
    assert hash(event) == hash(MarketEvent(timestamp=TS, symbol="SPY", kind="BAR", payload=a_bar()))


class _StubSource:
    """Minimal source, proving the protocol is implementable without inheritance."""

    def stream(self, start: datetime, end: datetime) -> Iterator[MarketEvent]:
        yield MarketEvent(timestamp=start, symbol="SPY", kind="BAR", payload=a_bar())


def test_market_data_source_protocol_is_structural() -> None:
    source = _StubSource()
    assert isinstance(source, MarketDataSource)
    events = list(source.stream(TS, TS))
    assert events[0].symbol == "SPY"
