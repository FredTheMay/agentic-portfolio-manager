"""Market data sources behind the ``MarketDataSource`` protocol (SPEC §4.2).

The engine consumes an ordered stream of :class:`~src.data.events.MarketEvent`.
It never asks where the events came from, which is what lets a live websocket
feed replace a daily bar file later without touching the engine loop.

**Ordering is a correctness property, not a convenience.** Events must arrive
in non-decreasing timestamp order: the simulation clock only moves forwards
(:class:`~src.time.clock.SimulationClock`), and an out-of-order event would
either crash the clock or, worse, let the system act on data from the future.
:class:`InMemoryEventSource` merges across symbols and validates the ordering
rather than trusting its input.

**Adjusted versus unadjusted prices** (SPEC §4.4). Bars carry both. ``close``
is the unadjusted print, used for share-count arithmetic; ``adj_close`` is
split- and dividend-adjusted, used for return calculation. Corporate actions
are additionally emitted as their own events so the portfolio can apply them
explicitly instead of inferring them from a jump in an adjusted series.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.data.cache import JsonFetcher
from src.data.events import BarPayload, MarketDataError, MarketEvent
from src.time.clock import UTC, ensure_utc

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"

#: Daily bars are stamped at the US equity close, 16:00 America/New_York.
#: Expressed in UTC as a fixed 21:00, which is correct during EST. See
#: `DailyBarSource` for why the DST caveat is acceptable for a daily strategy.
US_EQUITY_CLOSE_UTC_HOUR = 21


class SourceError(RuntimeError):
    """Raised on malformed market data or an out-of-order stream."""


def _parse_amount(value: Any, field: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise SourceError(f"{field} must be numeric, got a boolean")
    if isinstance(value, (int, str)):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise SourceError(f"{field} is not numeric: {value!r}") from exc
    if isinstance(value, float):
        # cache.loads decodes JSON floats as Decimal, so this is a fallback for
        # hand-built payloads. Convert through repr, never through arithmetic.
        return Decimal(repr(value))
    raise SourceError(f"{field} is not numeric: {value!r}")


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if not isinstance(value, str):
        raise SourceError(f"timestamp must be a string, got {value!r}")

    # Check date-only first: `datetime.fromisoformat` accepts "2024-01-03" and
    # returns midnight, which would silently place a daily bar seventeen hours
    # before the session it represents.
    if len(value) == 10 and "T" not in value:
        try:
            day = date.fromisoformat(value)
        except ValueError as exc:
            raise SourceError(f"unparseable timestamp {value!r}") from exc
        # A bare date is a session, stamped at that session's close.
        return datetime(day.year, day.month, day.day, US_EQUITY_CLOSE_UTC_HOUR, tzinfo=UTC)

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceError(f"unparseable timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return ensure_utc(parsed)


def bar_event(symbol: str, payload: Mapping[str, Any]) -> MarketEvent:
    """Build a ``BAR`` event from an Alpaca-shaped bar record.

    Alpaca uses ``o/h/l/c/v`` and stamps ``t`` in RFC-3339. ``adj_close`` is
    accepted under either name and left ``None`` when absent, because an
    absent adjustment must not be silently assumed equal to the raw close.
    """
    raw_adjusted = payload.get("adj_close", payload.get("ac"))
    try:
        bar = BarPayload(
            open=_parse_amount(payload["o"], "open"),
            high=_parse_amount(payload["h"], "high"),
            low=_parse_amount(payload["l"], "low"),
            close=_parse_amount(payload["c"], "close"),
            volume=int(payload.get("v", 0)),
            adj_close=None if raw_adjusted is None else _parse_amount(raw_adjusted, "adj_close"),
        )
    except KeyError as exc:
        raise SourceError(f"bar for {symbol} is missing field {exc}") from exc
    except MarketDataError as exc:
        raise SourceError(f"invalid bar for {symbol}: {exc}") from exc

    return MarketEvent(
        timestamp=_parse_timestamp(payload["t"]),
        symbol=symbol,
        kind="BAR",
        payload=bar,
    )


def events_from_alpaca_payload(payload: Mapping[str, Any]) -> list[MarketEvent]:
    """Flatten Alpaca's ``{"bars": {"SPY": [...], ...}}`` into a flat event list."""
    bars = payload.get("bars")
    if not isinstance(bars, Mapping):
        raise SourceError("Alpaca response contained no bars object")
    events: list[MarketEvent] = []
    for symbol, records in bars.items():
        if not isinstance(records, list):
            raise SourceError(f"bars for {symbol} were not a list")
        events.extend(bar_event(symbol, record) for record in records)
    return events


@dataclass(frozen=True, slots=True)
class InMemoryEventSource:
    """A ``MarketDataSource`` over an in-memory event list.

    Backs both offline replay and the test suite. Events are sorted on
    construction, so callers may supply them grouped by symbol — the natural
    shape of a vendor response — without having to merge them by hand.
    """

    events: tuple[MarketEvent, ...]

    @classmethod
    def from_events(cls, events: Iterable[MarketEvent]) -> InMemoryEventSource:
        ordered = tuple(sorted(events, key=lambda e: (e.timestamp, e.symbol)))
        return cls(events=ordered)

    @classmethod
    def from_alpaca_payload(cls, payload: Mapping[str, Any]) -> InMemoryEventSource:
        return cls.from_events(events_from_alpaca_payload(payload))

    def stream(self, start: datetime, end: datetime) -> Iterator[MarketEvent]:
        """Yield events in ``[start, end]``, ordered by timestamp."""
        first = ensure_utc(start)
        last = ensure_utc(end)
        if last < first:
            raise SourceError(f"end {last.isoformat()} precedes start {first.isoformat()}")
        for event in self.events:
            if first <= event.timestamp <= last:
                yield event

    def symbols(self) -> frozenset[str]:
        return frozenset(event.symbol for event in self.events)


def merge_ordered(streams: Sequence[Iterator[MarketEvent]]) -> Iterator[MarketEvent]:
    """Merge several already-ordered event streams into one ordered stream.

    Used to combine a bar feed with a corporate-action feed. Each input must
    already be ordered; ``heapq.merge`` preserves that across the merge without
    materializing everything, which matters once a stream is a live socket
    rather than a list.
    """
    return heapq.merge(*streams, key=lambda event: (event.timestamp, event.symbol))


def assert_ordered(events: Iterable[MarketEvent]) -> list[MarketEvent]:
    """Validate non-decreasing timestamps, raising on the first inversion.

    Called on anything arriving from outside. An out-of-order feed is a data
    bug that would otherwise surface much later as an inexplicable backtest
    result, or not at all.
    """
    materialized = list(events)
    for earlier, later in zip(materialized, materialized[1:]):
        if later.timestamp < earlier.timestamp:
            raise SourceError(
                f"events out of order: {later.symbol} at {later.timestamp.isoformat()} "
                f"follows {earlier.symbol} at {earlier.timestamp.isoformat()}"
            )
    return materialized


class AlpacaBarClient:
    """Daily bars from Alpaca's market data API, through a :class:`JsonFetcher`.

    Alpaca paper-trading keys are read by the caller and passed in; this class
    only shapes the request. With a caching fetcher the first run records the
    bars and every later run replays them offline.
    """

    def __init__(self, fetcher: JsonFetcher, feed: str = "iex") -> None:
        self._fetcher = fetcher
        self._feed = feed

    def daily_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> InMemoryEventSource:
        """Fetch adjusted daily bars for ``symbols`` over ``[start, end]``."""
        if not symbols:
            raise SourceError("at least one symbol is required")
        params = {
            "symbols": ",".join(sorted(symbols)),
            "timeframe": "1Day",
            "start": ensure_utc(start).date().isoformat(),
            "end": ensure_utc(end).date().isoformat(),
            # Split and dividend adjusted, so `adj_close` is meaningful.
            "adjustment": "all",
            "feed": self._feed,
        }
        payload = self._fetcher.get_json(ALPACA_BARS_URL, params)
        if not isinstance(payload, Mapping):
            raise SourceError("Alpaca response was not a JSON object")
        source = InMemoryEventSource.from_alpaca_payload(payload)
        assert_ordered(source.events)
        return source
