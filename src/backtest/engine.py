"""The backtest engine (SPEC §4.2, M6).

An **event loop**, not ``for day in trading_days:``. Events arrive at instants
from a :class:`~src.data.events.MarketDataSource`, the
:class:`~src.time.clock.SimulationClock` is advanced to each one, and the
identical code path runs against a live feed. That is the whole reason the
clock and the event model were built first: a date-indexed loop would have to
be rewritten to add intraday, and this one does not.

One cycle, in order:

1. estimate inputs from the trailing window (shrunk covariance, CAPM returns)
2. optimize (:mod:`src.decision.optimizer`)
3. **risk engine** (:mod:`src.risk.engine`) — approve, modify, or veto
4. emit a mandate (:mod:`src.decision.mandate`)
5. execute across the boundary (:mod:`src.execution`)
6. **reconcile** realized against target and carry the drift forward

Step 6 is not optional. Realized weights never equal target weights, and a
system that assumes they do produces backtests that lie (SPEC §3.4).

Determinism: no wall clock, no randomness, no dict-order dependence. Two runs
over identical inputs produce byte-identical output, checked by
:func:`result_digest` (SPEC §9, §11).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Mapping, Sequence

from src.agents.pipeline import NoViews, ViewPipeline
from src.data.events import BarPayload, MarketDataSource, MarketEvent
from src.decision.mandate import RebalanceMandate, Reconciliation, build_mandate, reconcile
from src.decision.optimizer import OptimizerError, TargetPortfolio, estimate_inputs, optimize
from src.execution.base import (
    Account,
    ExecutionProvider,
    ExecutionReport,
    MarketSnapshot,
)
from src.risk.codes import Decision
from src.risk.engine import RiskAssessment, RiskContext, evaluate
from src.risk.ips import InvestmentPolicy
from src.time.clock import SimulationClock, ensure_utc

ZERO = Decimal(0)
ONE = Decimal(1)


class BacktestError(RuntimeError):
    """Raised when a backtest cannot proceed."""


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything that defines a run. Hashed into the result digest."""

    start: datetime
    end: datetime
    initial_cash: Decimal
    symbols: tuple[str, ...]
    benchmark_symbol: str
    #: Trading days between rebalance attempts. The corridor check (SPEC §7)
    #: still decides whether a proposed rebalance is worth doing.
    rebalance_every: int = 21
    #: Trailing observations used to estimate the covariance.
    estimation_window: int = 120
    market_return: Decimal = Decimal("0.09")
    risk_free_rate: Decimal = Decimal("0.04")
    periods_per_year: int = 252


@dataclass(frozen=True, slots=True)
class CycleRecord:
    """One decision cycle, kept whole for the audit trail and the dashboard."""

    timestamp: datetime
    assessment: RiskAssessment
    target: TargetPortfolio | None
    mandate: RebalanceMandate | None
    report: ExecutionReport | None
    reconciliation: Reconciliation | None
    #: Why no proposal was produced, when that is the case. A silent no-trade
    #: is indistinguishable from a working strategy that chose to hold, which
    #: is exactly how a misconfigured optimizer hides for a whole backtest.
    note: str = ""
    #: Numeric tilts the agents contributed, empty when views are disabled.
    tilts: Mapping[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Equity curve, every cycle, and the vetoes."""

    config: BacktestConfig
    timestamps: tuple[datetime, ...]
    equity_curve: tuple[Decimal, ...]
    benchmark_curve: tuple[Decimal, ...]
    cycles: tuple[CycleRecord, ...]
    cash_flows: tuple[tuple[datetime, Decimal], ...] = ()

    @property
    def vetoed(self) -> tuple[CycleRecord, ...]:
        """Cycles the risk engine refused — the dashboard's first panel."""
        return tuple(c for c in self.cycles if c.assessment.decision is Decision.REJECTED)

    @property
    def executed(self) -> tuple[CycleRecord, ...]:
        return tuple(c for c in self.cycles if c.report is not None)


def _close(event: MarketEvent) -> Decimal | None:
    """Unadjusted close, for share arithmetic (SPEC §4.4)."""
    if isinstance(event.payload, BarPayload):
        return event.payload.close
    return None


def _adjusted(event: MarketEvent) -> Decimal | None:
    """Adjusted close, for return calculation. Never mixed with the raw close."""
    if isinstance(event.payload, BarPayload):
        return event.payload.adj_close or event.payload.close
    return None


def run_backtest(
    config: BacktestConfig,
    source: MarketDataSource,
    executor: ExecutionProvider,
    policy: InvestmentPolicy,
    sectors: Mapping[str, str],
    betas: Mapping[str, Decimal],
    views: ViewPipeline | None = None,
) -> BacktestResult:
    """Run the event loop from ``config.start`` to ``config.end``.

    ``views`` supplies the qualitative half (SPEC §5.4). It defaults to
    :class:`~src.agents.pipeline.NoViews`, so the engine is pure quantitative
    construction unless a caller opts in — and swapping in an agent pipeline
    backed by ``NullProvider`` changes nothing about whether the cycle
    completes (SPEC §2.1(4)).
    """
    view_pipeline = views or NoViews()
    start = ensure_utc(config.start)
    end = ensure_utc(config.end)
    if end <= start:
        raise BacktestError("backtest end must follow its start")

    clock = SimulationClock(start)
    account = Account(cash=config.initial_cash, positions={})

    # Latest known price per symbol, and the adjusted history the estimator uses.
    prices: dict[str, Decimal] = {}
    history: dict[str, list[Decimal]] = {s: [] for s in config.symbols}

    timestamps: list[datetime] = []
    equity: list[Decimal] = []
    benchmark_curve: list[Decimal] = []
    cycles: list[CycleRecord] = []

    peak_equity = config.initial_cash
    drawdown = ZERO
    bars_seen = 0
    last_rebalance_index = -10**9
    benchmark_units: Decimal | None = None
    pending: list[MarketEvent] = []
    current_stamp: datetime | None = None

    def flush(stamp: datetime) -> None:
        """Apply one timestamp's events, then mark the book and maybe rebalance."""
        nonlocal bars_seen, last_rebalance_index, peak_equity, drawdown
        nonlocal account, benchmark_units

        clock.advance_to(stamp)
        for event in pending:
            price = _close(event)
            if price is not None:
                prices[event.symbol] = price
            adjusted = _adjusted(event)
            if adjusted is not None and event.symbol in history:
                history[event.symbol].append(adjusted)

        # A session only counts once every tracked symbol has printed, so the
        # estimation window is not skewed by a symbol that started late.
        if not all(history[s] for s in config.symbols):
            return
        bars_seen += 1

        value = account.total_value(prices)
        timestamps.append(stamp)
        equity.append(value)

        if benchmark_units is None and config.benchmark_symbol in prices:
            benchmark_units = config.initial_cash / prices[config.benchmark_symbol]
        benchmark_curve.append(
            benchmark_units * prices[config.benchmark_symbol]
            if benchmark_units is not None and config.benchmark_symbol in prices
            else config.initial_cash
        )

        if value > peak_equity:
            peak_equity = value
        drawdown = (peak_equity - value) / peak_equity if peak_equity > ZERO else ZERO

        # window + 1 prices are needed to form `window` returns.
        ready = min(len(history[s]) for s in config.symbols) > config.estimation_window
        due = bars_seen - last_rebalance_index >= config.rebalance_every
        if not (ready and due):
            return

        last_rebalance_index = bars_seen
        record = _rebalance(
            stamp=stamp,
            config=config,
            account=account,
            prices=prices,
            history=history,
            policy=policy,
            sectors=sectors,
            betas=betas,
            executor=executor,
            drawdown=drawdown,
            views=view_pipeline,
        )
        cycles.append(record)
        if record.report is not None:
            account = Account(
                cash=record.report.final_cash,
                positions={p.symbol: p.quantity for p in record.report.final_positions},
            )

    for event in source.stream(start, end):
        if current_stamp is not None and event.timestamp != current_stamp:
            flush(current_stamp)
            pending = []
        current_stamp = event.timestamp
        pending.append(event)

    if current_stamp is not None and pending:
        flush(current_stamp)

    if len(equity) < 2:
        raise BacktestError(
            "backtest produced fewer than two marks; check the data window and symbols"
        )

    return BacktestResult(
        config=config,
        timestamps=tuple(timestamps),
        equity_curve=tuple(equity),
        benchmark_curve=tuple(benchmark_curve),
        cycles=tuple(cycles),
        cash_flows=(
            (timestamps[0], -config.initial_cash),
            (timestamps[-1], equity[-1]),
        ),
    )


def _rebalance(
    *,
    stamp: datetime,
    config: BacktestConfig,
    account: Account,
    prices: Mapping[str, Decimal],
    history: Mapping[str, Sequence[Decimal]],
    policy: InvestmentPolicy,
    sectors: Mapping[str, str],
    betas: Mapping[str, Decimal],
    executor: ExecutionProvider,
    drawdown: Decimal,
    views: ViewPipeline,
) -> CycleRecord:
    """One decision cycle: views, estimate, optimize, check, mandate, execute, reconcile."""
    window = config.estimation_window
    symbols = tuple(s for s in config.symbols if len(history[s]) > window and s in prices)
    # `window` returns need `window + 1` prices; the readiness check upstream
    # guarantees it, and disagreeing on the boundary would silently skip every
    # rebalance without ever raising.
    if len(symbols) < 2:
        return _no_trade(
            stamp, account, prices, sectors, betas, policy, drawdown,
            note=f"only {len(symbols)} symbols had enough history to optimize over",
        )

    # Returns over the trailing window, from *adjusted* prices (SPEC §4.4).
    observations: list[list[Decimal]] = []
    for index in range(-window, 0):
        row: list[Decimal] = []
        for symbol in symbols:
            series = history[symbol]
            previous, current = series[index - 1], series[index]
            row.append(current / previous - ONE if previous > ZERO else ZERO)
        observations.append(row)

    portfolio_value = account.total_value(prices)
    current_weights = {
        symbol: (Decimal(quantity) * prices[symbol]) / portfolio_value
        for symbol, quantity in account.positions.items()
        if symbol in prices and portfolio_value > ZERO
    }

    # The qualitative layer enters here and nowhere else: as an additive
    # adjustment to the CAPM baseline, produced by table lookup (SPEC §5.4).
    tilts = views.tilts(symbols, stamp)

    try:
        inputs = estimate_inputs(
            symbols,
            observations,
            {s: betas[s] for s in symbols if s in betas},
            market_return=config.market_return,
            risk_free_rate=config.risk_free_rate,
            periods_per_year=config.periods_per_year,
            tilts=tilts,
        )
        target = optimize(inputs, max_position_weight=policy.max_position_weight)
    except (OptimizerError, KeyError) as exc:
        return _no_trade(
            stamp, account, prices, sectors, betas, policy, drawdown,
            note=f"optimizer could not produce a portfolio: {exc}",
        )

    covariance = {
        a: {b: inputs.covariance[i][j] for j, b in enumerate(symbols)}
        for i, a in enumerate(symbols)
    }
    context = RiskContext(
        as_of=stamp,
        current_weights=current_weights,
        sectors=sectors,
        betas=betas,
        covariance=covariance,
        expected_returns=dict(zip(symbols, inputs.expected_returns)),
        universe=frozenset(symbols),
        drawdown=drawdown,
        risk_free_rate=config.risk_free_rate,
    )
    assessment = evaluate(target.weights, context, policy)

    if assessment.decision is Decision.REJECTED:
        return CycleRecord(
            timestamp=stamp,
            assessment=assessment,
            target=target,
            mandate=None,
            report=None,
            reconciliation=None,
            note="vetoed by the risk engine",
            tilts=tilts,
        )

    mandate = build_mandate(
        decision_time=stamp,
        portfolio_value=portfolio_value,
        target_weights=dict(assessment.weights),
        current_weights=current_weights,
        min_trade_notional=assessment.min_trade_notional,
        max_turnover=policy.max_turnover,
    )
    snapshot = MarketSnapshot(
        timestamp=stamp, prices=dict(prices), decision_prices=dict(prices)
    )
    report = executor.execute_to_completion(mandate, account, snapshot)
    return CycleRecord(
        timestamp=stamp,
        assessment=assessment,
        target=target,
        mandate=mandate,
        report=report,
        reconciliation=reconcile(mandate, report.realized_weights),
        tilts=tilts,
    )


def _no_trade(
    stamp: datetime,
    account: Account,
    prices: Mapping[str, Decimal],
    sectors: Mapping[str, str],
    betas: Mapping[str, Decimal],
    policy: InvestmentPolicy,
    drawdown: Decimal,
    note: str = "",
) -> CycleRecord:
    """A cycle that produced no proposal — recorded, with the reason, not skipped."""
    context = RiskContext(
        as_of=stamp,
        current_weights={},
        sectors=sectors,
        betas=betas,
        covariance={},
        expected_returns={},
        universe=frozenset(),
        drawdown=drawdown,
    )
    return CycleRecord(
        timestamp=stamp,
        assessment=evaluate({}, context, policy),
        target=None,
        mandate=None,
        report=None,
        reconciliation=None,
        note=note,
    )


def result_digest(result: BacktestResult) -> str:
    """Stable hash of a run's output (SPEC §11: two runs, identical hashes).

    Covers the equity curve and every mandate id, so a change in either the
    marks or the decisions moves the digest.
    """
    parts: list[str] = [
        result.config.start.isoformat(),
        result.config.end.isoformat(),
        str(result.config.initial_cash),
        ",".join(result.config.symbols),
    ]
    for stamp, value in zip(result.timestamps, result.equity_curve):
        parts.append(f"{stamp.isoformat()}={value.quantize(Decimal('0.000001'))}")
    for cycle in result.cycles:
        parts.append(
            f"{cycle.timestamp.isoformat()}:{cycle.assessment.decision.value}:"
            f"{cycle.mandate.mandate_id if cycle.mandate else '-'}"
        )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
