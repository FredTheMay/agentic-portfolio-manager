"""Assembling a real backtest from recorded market data (SPEC §4.2, §6.2).

The counterpart to :mod:`src.data.synthetic`. Where that generates bars, this
one replays what was actually fetched — and every function here reads through
the cache, so a backtest over real data needs credentials exactly once
(``scripts/backfill.py``) and is offline and byte-reproducible thereafter.

Two things are *estimated* here rather than assumed, because with real data
they are no longer free parameters:

**Betas.** The synthetic universe declares each symbol's beta. Real symbols do
not, so beta is estimated by regressing excess returns on the benchmark's
(SPEC §6.2 [CORRECTED]) — which also yields R² and the standard errors, unlike
the ``Cov/Var`` shortcut.

**The risk-free rate.** Taken from FRED's ``DGS3MO`` and converted from its
bank-discount quote to a bond-equivalent yield before use. Skipping that
conversion understates Rf and inflates every risk-adjusted metric downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from src.cfa.fixed_income import discount_to_bond_equivalent_yield
from src.cfa.returns import estimate_beta
from src.data.cache import CachingFetcher, ResponseCache
from src.data.events import BarPayload, MarketEvent
from src.data.fred import THREE_MONTH_TREASURY, FredClient
from src.data.sources import AlpacaBarClient, InMemoryEventSource, live_alpaca_fetcher
from src.data.universe import Universe
from src.time.clock import ensure_utc

ZERO = Decimal(0)
ONE = Decimal(1)

#: Where backfilled responses live. Gitignored.
DEFAULT_CACHE_ROOT = Path("data/cache")

#: FRED quotes DGS3MO as a percentage; the system works in decimal fractions.
PERCENT = Decimal(100)

#: T-bill tenor DGS3MO refers to, for the discount->BEY conversion.
BILL_DAYS = 91


class LiveDataError(RuntimeError):
    """Raised when recorded data is missing or unusable."""


def cached_fetcher(cache_root: Path | None = None, offline: bool = True) -> CachingFetcher:
    """A fetcher over the backfill cache.

    Defaults to ``offline=True``: a backtest must never silently reach the
    network mid-run, because that would make the result depend on the day it
    was run (SPEC §9). ``scripts/backfill.py`` is the one place that fetches.
    """
    root = cache_root or DEFAULT_CACHE_ROOT
    if offline:
        return CachingFetcher(ResponseCache(root=root), offline=True)
    return CachingFetcher(ResponseCache(root=root), inner=live_alpaca_fetcher())


def load_bars(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    cache_root: Path | None = None,
    offline: bool = True,
) -> InMemoryEventSource:
    """Replay recorded daily bars as a ``MarketDataSource``."""
    client = AlpacaBarClient(cached_fetcher(cache_root, offline))
    return client.daily_bars(list(symbols), ensure_utc(start), ensure_utc(end))


def adjusted_series(source: InMemoryEventSource) -> dict[str, list[Decimal]]:
    """Adjusted closes per symbol, in time order.

    Adjusted, because these become *returns* — SPEC §4.4 keeps adjusted and
    unadjusted apart and this is the return side.
    """
    series: dict[str, list[Decimal]] = {}
    for event in source.events:
        if not isinstance(event.payload, BarPayload):
            continue
        price = event.payload.adj_close or event.payload.close
        series.setdefault(event.symbol, []).append(price)
    return series


def to_returns(prices: Sequence[Decimal]) -> list[Decimal]:
    return [
        after / before - ONE
        for before, after in zip(prices, prices[1:])
        if before > ZERO
    ]


@dataclass(frozen=True, slots=True)
class BetaEstimate:
    """A regression-estimated beta, with the uncertainty that comes with it."""

    symbol: str
    beta: Decimal
    r_squared: Decimal
    observations: int


def estimate_betas(
    source: InMemoryEventSource,
    benchmark: str,
    risk_free_rate: Decimal,
    periods_per_year: int = 252,
) -> dict[str, BetaEstimate]:
    """Regress each symbol's excess returns on the benchmark's (SPEC §6.2).

    Symbols whose history does not line up with the benchmark's are skipped
    rather than given a default beta: a fabricated 1.0 would flow straight into
    the portfolio beta constraint and the CAPM expected return.
    """
    series = adjusted_series(source)
    if benchmark not in series:
        raise LiveDataError(f"no recorded history for the benchmark {benchmark}")

    market = to_returns(series[benchmark])
    periodic_rf = risk_free_rate / Decimal(periods_per_year)
    market_excess = [r - periodic_rf for r in market]

    estimates: dict[str, BetaEstimate] = {}
    for symbol, prices in sorted(series.items()):
        returns = to_returns(prices)
        if len(returns) != len(market) or len(returns) < 3:
            continue
        try:
            fit = estimate_beta([r - periodic_rf for r in returns], market_excess)
        except Exception:  # noqa: BLE001 - a degenerate series is not fatal
            continue
        estimates[symbol] = BetaEstimate(
            symbol=symbol,
            beta=fit.slope,
            r_squared=fit.r_squared,
            observations=fit.observations,
        )
    return estimates


def risk_free_rate(
    as_of: datetime,
    cache_root: Path | None = None,
    offline: bool = True,
    fallback: Decimal = Decimal("0.04"),
) -> Decimal:
    """Bond-equivalent yield from FRED ``DGS3MO``, as of an instant.

    ``DGS3MO`` is quoted on a **bank discount basis**: it divides the discount
    by face value rather than by the price actually paid, and annualizes on 360
    days. Both biases run low, so feeding it in raw understates Rf and inflates
    every excess return and Sharpe ratio computed from it (SPEC §6.2
    [CORRECTED]).

    Falls back to ``fallback`` when nothing has been recorded, so a run without
    FRED still completes — with a stated assumption rather than a crash.
    """
    try:
        series = FredClient(cached_fetcher(cache_root, offline)).series(THREE_MONTH_TREASURY)
    except Exception:  # noqa: BLE001 - no recorded FRED data is a normal state
        return fallback

    vintage = series.as_of(ensure_utc(as_of))
    if vintage is None:
        return fallback
    discount = vintage.value / PERCENT
    if discount <= ZERO:
        return fallback
    return discount_to_bond_equivalent_yield(discount, BILL_DAYS)


def market_return(betas: Mapping[str, BetaEstimate], risk_free: Decimal) -> Decimal:
    """Equity risk premium assumption, stated rather than buried.

    A long-run premium of 5% over the risk-free rate. This is an *assumption*,
    not an estimate — realized premia over any sample are far too noisy to
    estimate a forward-looking mean from, which is precisely why the optimizer
    takes CAPM expected returns rather than sample means (SPEC §6.2).
    """
    return risk_free + Decimal("0.05")


def universe_inputs(
    universe: Universe,
    source: InMemoryEventSource,
    as_of: datetime,
    cache_root: Path | None = None,
) -> tuple[dict[str, str], dict[str, Decimal], Decimal, Decimal]:
    """Sectors, betas, risk-free rate and market return for a real backtest."""
    rate = risk_free_rate(as_of, cache_root)
    estimates = estimate_betas(source, universe.benchmark_equity, rate)
    betas = {symbol: est.beta for symbol, est in estimates.items()}
    return universe.sectors, betas, rate, market_return(estimates, rate)
