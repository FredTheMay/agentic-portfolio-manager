"""Assembles a backtest from recorded market data.

Reads through the cache, so a run over real data needs credentials exactly once
(``scripts/backfill.py``) and is offline and reproducible thereafter.

Two quantities are estimated rather than assumed. Betas are regressed from
excess returns, since real symbols do not come with one. The risk-free rate
comes from ``DGS3MO`` converted from its bank-discount quote to a
bond-equivalent yield; skipping that conversion understates it and inflates
every risk-adjusted metric downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from src.time.clock import UTC, ensure_utc

ZERO = Decimal(0)
ONE = Decimal(1)

#: Where backfilled responses live. Gitignored.
DEFAULT_CACHE_ROOT = Path("data/cache")

#: Written by ``scripts/backfill.py``, read by everything that replays.
#:
#: A cache key is a hash of the full request, date range included, so a replay
#: that guesses a window even one day different from the recorded one misses
#: every entry and silently falls back to synthetic data. The manifest removes
#: the guess: the recorder states what it fetched and the reader replays exactly
#: that.
MANIFEST_NAME = "manifest.json"

#: FRED quotes DGS3MO as a percentage; the system works in decimal fractions.
PERCENT = Decimal(100)

#: T-bill tenor DGS3MO refers to, for the discount->BEY conversion.
BILL_DAYS = 91


class LiveDataError(RuntimeError):
    """Raised when recorded data is missing or unusable."""


@dataclass(frozen=True, slots=True)
class Manifest:
    """What was recorded, so a replay can ask for precisely the same thing."""

    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    recorded_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "symbols": list(self.symbols),
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "recorded_at": self.recorded_at.isoformat(),
            },
            indent=2,
            sort_keys=True,
        )


def write_manifest(
    root: Path, symbols: Sequence[str], start: datetime, end: datetime, recorded_at: datetime
) -> Manifest:
    manifest = Manifest(
        symbols=tuple(symbols),
        start=ensure_utc(start),
        end=ensure_utc(end),
        recorded_at=ensure_utc(recorded_at),
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def read_manifest(root: Path | None = None) -> Manifest | None:
    """The recorded window, or ``None`` when nothing has been backfilled."""
    path = (root or DEFAULT_CACHE_ROOT) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(
            symbols=tuple(raw["symbols"]),
            start=ensure_utc(datetime.fromisoformat(raw["start"])),
            end=ensure_utc(datetime.fromisoformat(raw["end"])),
            recorded_at=ensure_utc(datetime.fromisoformat(raw["recorded_at"])),
        )
    except (KeyError, ValueError, TypeError):
        return None


def cached_fetcher(cache_root: Path | None = None, offline: bool = True) -> CachingFetcher:
    """A fetcher over the backfill cache.

    Defaults to ``offline=True``: a backtest must never silently reach the
    network mid-run, because that would make the result depend on the day it
    was run. ``scripts/backfill.py`` is the one place that fetches.
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

    Adjusted, because these become *returns* keeps adjusted and
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
    """Regress each symbol's excess returns on the benchmark's.

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
    every excess return and Sharpe ratio computed from it.

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
    takes CAPM expected returns rather than sample means.
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


@dataclass(frozen=True, slots=True)
class BacktestSetup:
    """Everything a run needs, plus a label saying where the data came from.

    The label is not decoration. Three callers — ``make results``, the
    dashboard, and the scheduled Lambda — each need the same "real if recorded,
    else synthetic" decision, and three copies of it would eventually disagree
    about which one they were showing. Presenting synthetic numbers as market
    results is the single most misleading thing this project could do, so the
    choice is made once and carries its own provenance.
    """

    source: InMemoryEventSource
    symbols: tuple[str, ...]
    sectors: Mapping[str, str]
    betas: Mapping[str, Decimal]
    benchmark: str
    start: datetime
    end: datetime
    risk_free_rate: Decimal
    market_return: Decimal
    data_source: str
    is_real: bool


def _synthetic_setup() -> BacktestSetup:
    from src.data.synthetic import BETAS, SECTORS, make_source

    start = datetime(2022, 1, 3, 21, tzinfo=UTC)
    return BacktestSetup(
        source=make_source(days=760),
        symbols=(
            "AAA", "BBB", "CCC", "DDD", "EEE", "FFF",
            "GGG", "HHH", "III", "JJJ", "KKK", "LLL",
        ),
        sectors=SECTORS,
        betas=BETAS,
        benchmark="SPY",
        start=start,
        end=start + timedelta(days=730),
        risk_free_rate=Decimal("0.04"),
        market_return=Decimal("0.09"),
        data_source="synthetic (nothing recorded — run `make backfill`)",
        is_real=False,
    )


def resolve_setup(
    cache_root: Path | None = None,
    minimum_symbols: int = 10,
) -> BacktestSetup:
    """Recorded market data when available, synthetic otherwise.

    ``minimum_symbols`` defaults to ten because the IPS caps any name at 10%,
    so a fully invested portfolio needs at least that many holdings with usable
    betas; below that the constrained frontier is infeasible and the optimizer
    would raise on every cycle.
    """
    from src.data.universe import load_universe

    manifest = read_manifest(cache_root)
    if manifest is None:
        return _synthetic_setup()

    try:
        universe = load_universe()
        source = load_bars(
            manifest.symbols, manifest.start, manifest.end, cache_root, offline=True
        )
        if not source.events:
            return _synthetic_setup()

        as_of = source.events[-1].timestamp
        sectors, betas, rate, market = universe_inputs(universe, source, as_of, cache_root)
        symbols = tuple(s for s in universe.tradable() if s in betas)
        if len(symbols) < minimum_symbols:
            return _synthetic_setup()
    except Exception:  # noqa: BLE001 - absent or partial recording is a normal state
        return _synthetic_setup()

    return BacktestSetup(
        source=source,
        symbols=symbols,
        sectors=sectors,
        betas=betas,
        benchmark=universe.benchmark_equity,
        start=manifest.start,
        end=manifest.end,
        risk_free_rate=rate,
        market_return=market,
        data_source=f"real market data (recorded {manifest.recorded_at.date().isoformat()})",
        is_real=True,
    )
