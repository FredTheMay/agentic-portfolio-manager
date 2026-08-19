"""Walk-forward validation and fill-model sensitivity.

Estimate on a rolling window, trade the following period, roll forward. A
single backtest over the full history is curve fitting, and iterating on it
until it looks good is how noise gets fitted.

Walk-forward does not eliminate that — reusing the same data to pick a model
across folds still leaks — but it does ensure every trade uses only information
available before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Mapping, Sequence

from src.backtest.engine import (
    BacktestConfig,
    BacktestError,
    BacktestResult,
    result_digest,
    run_backtest,
)
from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.data.events import MarketDataSource
from src.execution.fill_models import InstantFillModel, SpreadCrossFillModel
from src.execution.simulated import SimulatedExecutor
from src.risk.ips import InvestmentPolicy
from src.time.clock import ensure_utc

ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class Window:
    """One walk-forward fold: estimate on one span, trade the next."""

    estimation_start: datetime
    estimation_end: datetime
    trade_start: datetime
    trade_end: datetime

    def __post_init__(self) -> None:
        for name in ("estimation_start", "estimation_end", "trade_start", "trade_end"):
            object.__setattr__(self, name, ensure_utc(getattr(self, name)))
        if self.estimation_end > self.trade_start:
            raise BacktestError(
                "estimation window overlaps the trading window: the fold would "
                "trade on data from its own future"
            )


def rolling_windows(
    start: datetime,
    end: datetime,
    estimation: timedelta,
    trade: timedelta,
) -> list[Window]:
    """Non-overlapping trading periods, each preceded by its own estimation span."""
    first = ensure_utc(start)
    last = ensure_utc(end)
    if estimation <= timedelta(0) or trade <= timedelta(0):
        raise BacktestError("estimation and trade spans must be positive")

    windows: list[Window] = []
    cursor = first + estimation
    while cursor < last:
        stop = min(cursor + trade, last)
        windows.append(
            Window(
                estimation_start=cursor - estimation,
                estimation_end=cursor,
                trade_start=cursor,
                trade_end=stop,
            )
        )
        cursor = stop
    return windows


@dataclass(frozen=True, slots=True)
class FillModelOutcome:
    """One run under one fill model."""

    fill_model: str
    metrics: PerformanceMetrics
    digest: str
    vetoed_cycles: int
    executed_cycles: int
    total_commission: Decimal
    mean_shortfall_bps: Decimal


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Results under both fill models, plus the execution-cost gap."""

    optimistic: FillModelOutcome
    realistic: FillModelOutcome

    @property
    def execution_cost_drag(self) -> Decimal:
        """Annualized return given up to execution costs.

        The number a strategy is tempted to leave out. Reporting only the
        optimistic figure is the classic amateur tell.
        """
        return self.optimistic.metrics.annualized_twr - self.realistic.metrics.annualized_twr


def _outcome(
    label: str,
    result: BacktestResult,
    risk_free_rate: Decimal,
    periods_per_year: int,
) -> FillModelOutcome:
    metrics = compute_metrics(
        result.equity_curve,
        result.benchmark_curve,
        risk_free_rate=risk_free_rate,
        cash_flows=list(result.cash_flows),
        periods_per_year=periods_per_year,
    )
    reports = [c.report for c in result.cycles if c.report is not None]
    commission = sum((r.total_commission for r in reports), ZERO)
    shortfall = (
        sum((r.implementation_shortfall_bps for r in reports), ZERO) / Decimal(len(reports))
        if reports
        else ZERO
    )
    return FillModelOutcome(
        fill_model=label,
        metrics=metrics,
        digest=result_digest(result),
        vetoed_cycles=len(result.vetoed),
        executed_cycles=len(result.executed),
        total_commission=commission,
        mean_shortfall_bps=shortfall,
    )


def run_under_both_fill_models(
    config: BacktestConfig,
    source: MarketDataSource,
    policy: InvestmentPolicy,
    sectors: Mapping[str, str],
    betas: Mapping[str, Decimal],
    spread_bps: Decimal = Decimal("2"),
) -> WalkForwardResult:
    """Run the same backtest twice, once per fill model, and report both."""
    optimistic = run_backtest(
        config,
        source,
        SimulatedExecutor(fill_model=InstantFillModel()),
        policy,
        sectors,
        betas,
    )
    realistic = run_backtest(
        config,
        source,
        SimulatedExecutor(fill_model=SpreadCrossFillModel(spread_bps=spread_bps)),
        policy,
        sectors,
        betas,
    )
    return WalkForwardResult(
        optimistic=_outcome(
            "InstantFillModel", optimistic, config.risk_free_rate, config.periods_per_year
        ),
        realistic=_outcome(
            "SpreadCrossFillModel", realistic, config.risk_free_rate, config.periods_per_year
        ),
    )


def run_walk_forward(
    config: BacktestConfig,
    source: MarketDataSource,
    policy: InvestmentPolicy,
    sectors: Mapping[str, str],
    betas: Mapping[str, Decimal],
    estimation: timedelta,
    trade: timedelta,
) -> list[tuple[Window, BacktestResult]]:
    """Run one backtest per fold, each trading only its own out-of-sample span."""
    windows = rolling_windows(config.start, config.end, estimation, trade)
    if not windows:
        raise BacktestError(
            "no walk-forward windows fit in the requested span; shorten the "
            "estimation or trade period"
        )

    folds: list[tuple[Window, BacktestResult]] = []
    for window in windows:
        fold_config = BacktestConfig(
            start=window.estimation_start,
            end=window.trade_end,
            initial_cash=config.initial_cash,
            symbols=config.symbols,
            benchmark_symbol=config.benchmark_symbol,
            rebalance_every=config.rebalance_every,
            estimation_window=config.estimation_window,
            market_return=config.market_return,
            risk_free_rate=config.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
        try:
            folds.append(
                (
                    window,
                    run_backtest(
                        fold_config,
                        source,
                        SimulatedExecutor(fill_model=SpreadCrossFillModel()),
                        policy,
                        sectors,
                        betas,
                    ),
                )
            )
        except BacktestError:
            # A fold with too little data is skipped, not fabricated.
            continue
    return folds
