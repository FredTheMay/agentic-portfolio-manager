"""Deterministic synthetic market data (SPEC §4.2).

A real :class:`~src.data.events.MarketDataSource`, not a test fixture. It lives
in ``src/data/`` because production code uses it: with no API keys configured
there is nothing recorded to replay, so the scheduled Lambda cycle and the
dashboard both fall back to this and **say so** in their status output.

It began life under ``tests/``, which meant ``src/api/`` imported from the test
suite — a layering inversion that would have broken any deployment packaging
only ``src/``. ``tests/test_layer_isolation.py`` now fails the build if that
recurs.

Generated from a seeded PRNG rather than recorded from a vendor, so every run
produces byte-identical bars and the whole system runs offline. Prices follow a
geometric random walk around a single common market factor, scaled by each
symbol's beta — without that factor the symbols are independent walks, the
portfolio has zero beta against the benchmark by construction, and Treynor,
Jensen's alpha and R-squared are all meaningless.

**Synthetic prices have no earnings, no regimes, no crashes and no fat tails.**
Anything measured on them demonstrates that the pipeline works, never that a
strategy does.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from decimal import Decimal

from src.data.events import BarPayload, MarketEvent
from src.data.sources import InMemoryEventSource
from src.time.clock import UTC

D = Decimal

#: Symbol -> (starting price, annual drift, annual volatility, beta, sector)
#
# Sized deliberately: the IPS caps any name at 10%, so a fully invested
# portfolio needs at least ten holdings. A smaller universe makes the
# constrained frontier infeasible, which is a configuration error rather than
# a market condition.
UNIVERSE: dict[str, tuple[str, str, str, str, str]] = {
    "SPY": ("400.00", "0.08", "0.16", "1.00", "BROAD_EQUITY"),
    "AAA": ("100.00", "0.11", "0.26", "1.35", "TECH"),
    "BBB": ("80.00", "0.09", "0.22", "1.10", "TECH"),
    "CCC": ("50.00", "0.06", "0.15", "0.70", "HEALTH"),
    "DDD": ("120.00", "0.07", "0.18", "0.85", "ENERGY"),
    "EEE": ("60.00", "0.05", "0.13", "0.60", "UTILITIES"),
    "FFF": ("90.00", "0.10", "0.24", "1.20", "TECH"),
    "GGG": ("140.00", "0.07", "0.17", "0.90", "HEALTH"),
    "HHH": ("70.00", "0.06", "0.14", "0.65", "CONSUMER_STAPLES"),
    "III": ("110.00", "0.09", "0.20", "1.05", "FINANCIALS"),
    "JJJ": ("45.00", "0.08", "0.19", "0.95", "INDUSTRIALS"),
    "KKK": ("200.00", "0.05", "0.12", "0.55", "REAL_ESTATE"),
    "LLL": ("30.00", "0.09", "0.23", "1.15", "MATERIALS"),
}

SECTORS = {symbol: spec[4] for symbol, spec in UNIVERSE.items()}
BETAS = {symbol: D(spec[3]) for symbol, spec in UNIVERSE.items()}

TRADING_DAYS = 252
SESSION_CLOSE_HOUR = 21


def make_source(
    days: int = 400,
    start: datetime | None = None,
    seed: int = 20240603,
) -> InMemoryEventSource:
    """Build a deterministic daily-bar source over ``days`` sessions."""
    first = start or datetime(2022, 1, 3, SESSION_CLOSE_HOUR, tzinfo=UTC)
    rng = random.Random(seed)

    levels = {symbol: float(spec[0]) for symbol, spec in UNIVERSE.items()}
    events: list[MarketEvent] = []

    market_volatility = float(UNIVERSE["SPY"][2]) / (TRADING_DAYS**0.5)
    market_drift = float(UNIVERSE["SPY"][1]) / TRADING_DAYS

    for day in range(days):
        # Skip weekends so the calendar looks like a real session sequence.
        stamp = first + timedelta(days=day)
        if stamp.weekday() >= 5:
            continue

        # A single common factor drives every name, scaled by its beta, plus an
        # idiosyncratic shock. Without this the symbols are independent random
        # walks: the portfolio has zero beta against the benchmark by
        # construction, and Treynor, Jensen's alpha and R-squared all become
        # meaningless. A generator that cannot exercise CAPM is not testing it.
        market_shock = rng.gauss(market_drift, market_volatility)

        for symbol, spec in UNIVERSE.items():
            beta = float(spec[3])
            drift = float(spec[1]) / TRADING_DAYS
            total_volatility = float(spec[2]) / (TRADING_DAYS**0.5)
            # Split total risk into systematic and idiosyncratic so the
            # generated series actually has the beta it advertises.
            systematic = beta * market_volatility
            idiosyncratic = max(total_volatility**2 - systematic**2, 0.0) ** 0.5
            shock = (
                drift
                + beta * (market_shock - market_drift)
                + rng.gauss(0.0, idiosyncratic)
            )
            levels[symbol] = max(levels[symbol] * (1.0 + shock), 1.0)

            close = D(f"{levels[symbol]:.4f}")
            spread = close * D("0.004")
            events.append(
                MarketEvent(
                    timestamp=stamp,
                    symbol=symbol,
                    kind="BAR",
                    payload=BarPayload(
                        open=close,
                        high=close + spread,
                        low=close - spread,
                        close=close,
                        volume=1_000_000,
                        # No corporate actions in the synthetic set, so adjusted
                        # equals unadjusted. Kept explicit rather than left None
                        # so the return path is exercised.
                        adj_close=close,
                    ),
                )
            )

    return InMemoryEventSource.from_events(events)
