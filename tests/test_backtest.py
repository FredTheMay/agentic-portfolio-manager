"""Backtest engine, metrics, and walk-forward validation (SPEC §4.2, §6.2, M6)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestError,
    result_digest,
    run_backtest,
)
from src.backtest.metrics import (
    MetricsError,
    annualize,
    annualized_volatility,
    compute_metrics,
    max_drawdown,
    periodic_returns,
    summarize,
)
from src.backtest.walkforward import (
    Window,
    rolling_windows,
    run_under_both_fill_models,
    run_walk_forward,
)
from src.execution.fill_models import InstantFillModel, SpreadCrossFillModel
from src.execution.simulated import SimulatedExecutor
from src.risk.ips import load_policy
from src.time.clock import UTC
from tests.synthetic import BETAS, SECTORS, make_source

D = Decimal
POLICY = load_policy()

START = datetime(2022, 1, 3, 21, tzinfo=UTC)


def config(**overrides: object) -> BacktestConfig:
    base: dict[str, object] = {
        "start": START,
        "end": START + timedelta(days=560),
        "initial_cash": D("100000.00"),
        "symbols": (
            "AAA", "BBB", "CCC", "DDD", "EEE", "FFF",
            "GGG", "HHH", "III", "JJJ", "KKK", "LLL",
        ),
        "benchmark_symbol": "SPY",
        "rebalance_every": 21,
        "estimation_window": 100,
        "market_return": D("0.09"),
        "risk_free_rate": D("0.04"),
        "periods_per_year": 252,
    }
    base.update(overrides)
    return BacktestConfig(**base)  # type: ignore[arg-type]


# ===========================================================================
# Metrics
# ===========================================================================


def test_periodic_returns() -> None:
    assert periodic_returns([D("100"), D("110"), D("99")]) == [D("0.1"), D("-0.1")]


def test_periodic_returns_of_a_single_point_is_empty() -> None:
    assert periodic_returns([D("100")]) == []


def test_periodic_returns_rejects_a_wiped_out_curve() -> None:
    with pytest.raises(MetricsError):
        periodic_returns([D("0"), D("100")])


def test_max_drawdown() -> None:
    # Peak 120, trough 90 -> (120 - 90) / 120 = 0.25
    assert max_drawdown([D("100"), D("120"), D("90"), D("110")]) == D("0.25")


def test_max_drawdown_of_a_rising_curve_is_zero() -> None:
    assert max_drawdown([D("100"), D("110"), D("120")]) == D("0")


def test_max_drawdown_measures_from_the_peak_not_the_start() -> None:
    # The investor's experience is peak-to-trough, which is what the IPS
    # circuit breaker is written against.
    assert max_drawdown([D("100"), D("200"), D("150")]) == D("0.25")


def test_annualize_scales_a_cumulative_return() -> None:
    # 21% over half a year (126 of 252 days) -> 1.21^2 - 1 = 0.4641
    result = annualize(D("0.21"), periods=126, periods_per_year=252)
    assert abs(result - D("0.4641")) < D("1e-9")


def test_annualize_handles_a_total_loss() -> None:
    assert annualize(D("-1"), periods=126) == D("-1")


def test_annualized_volatility_scales_by_root_time() -> None:
    # Daily sd of 0.01 -> annual 0.01 * sqrt(252) = 0.15874...
    daily = [D("0.01"), D("-0.01")] * 10
    result = annualized_volatility(daily, periods_per_year=252)
    assert D("0.15") < result < D("0.17")


def test_metrics_require_a_matching_benchmark() -> None:
    with pytest.raises(MetricsError, match="benchmark"):
        compute_metrics([D("100"), D("101")], [D("100")], risk_free_rate=D("0.04"))


def test_metrics_require_two_points() -> None:
    with pytest.raises(MetricsError):
        compute_metrics([D("100")], [D("100")], risk_free_rate=D("0.04"))


def test_metrics_report_twr_and_mwr_separately() -> None:
    # SPEC §6.1: TWR is the headline; MWR is reported alongside so the gap can
    # be explained rather than hidden.
    equity = [D("100"), D("105"), D("110"), D("115")]
    benchmark = [D("100"), D("102"), D("104"), D("106")]
    flows = [
        (datetime(2023, 1, 1, tzinfo=UTC), D("-100")),
        (datetime(2024, 1, 1, tzinfo=UTC), D("115")),
    ]
    metrics = compute_metrics(equity, benchmark, D("0.04"), cash_flows=flows)

    assert metrics.twr > D("0")
    assert metrics.mwr is not None
    assert metrics.periods == 3


def test_alpha_carries_a_t_statistic() -> None:
    # SPEC §6.1: a positive alpha without its t-stat is noise nobody ruled out.
    equity = [D(str(100 + i)) for i in range(30)]
    benchmark = [D(str(100 + i * 0.5)) for i in range(30)]
    metrics = compute_metrics(equity, benchmark, D("0.04"))

    assert metrics.regression is not None
    assert metrics.alpha_t_stat is not None
    assert isinstance(metrics.alpha_is_significant, bool)


def test_a_portfolio_tracking_its_benchmark_has_beta_near_one() -> None:
    curve = [D(str(100 * (1.001**i))) for i in range(40)]
    metrics = compute_metrics(curve, curve, D("0.04"))
    assert abs(metrics.beta - D("1")) < D("0.01")
    assert metrics.tracking_error == D("0")
    assert metrics.information_ratio is None


def test_mwr_is_none_without_a_sign_change() -> None:
    equity = [D("100"), D("110")]
    flows = [
        (datetime(2023, 1, 1, tzinfo=UTC), D("-100")),
        (datetime(2024, 1, 1, tzinfo=UTC), D("-10")),
    ]
    metrics = compute_metrics(equity, equity, D("0.04"), cash_flows=flows)
    assert metrics.mwr is None


def test_summarize_produces_display_strings() -> None:
    equity = [D(str(100 + i)) for i in range(30)]
    benchmark = [D(str(100 + i * 0.5)) for i in range(30)]
    summary = summarize(compute_metrics(equity, benchmark, D("0.04")))

    assert set(summary) >= {"annualized_twr", "sharpe", "max_drawdown", "alpha_t_stat"}
    assert all(isinstance(v, str) for v in summary.values())


# ===========================================================================
# The engine
# ===========================================================================


def test_backtest_runs_end_to_end() -> None:
    result = run_backtest(
        config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS
    )

    assert len(result.equity_curve) > 100
    assert len(result.equity_curve) == len(result.benchmark_curve)
    assert len(result.equity_curve) == len(result.timestamps)
    assert result.cycles, "the run must attempt at least one rebalance"


def test_backtest_is_deterministic() -> None:
    # SPEC §11: two identical runs produce identical output hashes.
    first = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    second = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)

    assert result_digest(first) == result_digest(second)
    assert first.equity_curve == second.equity_curve


def test_a_different_run_produces_a_different_digest() -> None:
    a = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    b = run_backtest(
        config(initial_cash=D("250000.00")),
        make_source(),
        SimulatedExecutor(),
        POLICY,
        SECTORS,
        BETAS,
    )
    assert result_digest(a) != result_digest(b)


def test_the_event_loop_advances_time_monotonically() -> None:
    result = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    stamps = list(result.timestamps)
    assert stamps == sorted(stamps)
    assert all(s.tzinfo is UTC for s in stamps)


def test_every_executed_cycle_is_reconciled() -> None:
    # SPEC §3.4: realized weights never equal targets, and the residual is
    # mandatory to record.
    result = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    for cycle in result.executed:
        assert cycle.reconciliation is not None
        assert cycle.mandate is not None


def test_risk_approved_weights_respect_the_position_cap() -> None:
    result = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    for cycle in result.cycles:
        if cycle.assessment.approved:
            for weight in cycle.assessment.weights.values():
                assert weight <= POLICY.max_position_weight + D("1e-9")


def test_vetoed_cycles_are_retained_for_the_dashboard() -> None:
    # SPEC §7: every rejection is persisted and surfaced.
    result = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    for cycle in result.vetoed:
        assert cycle.assessment.violations
        assert cycle.report is None


def test_backtest_rejects_an_inverted_window() -> None:
    with pytest.raises(BacktestError):
        run_backtest(
            config(end=START - timedelta(days=1)),
            make_source(),
            SimulatedExecutor(),
            POLICY,
            SECTORS,
            BETAS,
        )


def test_backtest_reports_too_little_data() -> None:
    with pytest.raises(BacktestError, match="fewer than two marks"):
        run_backtest(
            config(end=START + timedelta(hours=1)),  # a single session
            make_source(),
            SimulatedExecutor(),
            POLICY,
            SECTORS,
            BETAS,
        )


def test_metrics_can_be_computed_from_a_run() -> None:
    result = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    metrics = compute_metrics(
        result.equity_curve,
        result.benchmark_curve,
        risk_free_rate=D("0.04"),
        cash_flows=list(result.cash_flows),
    )
    assert metrics.periods == len(result.equity_curve) - 1
    assert metrics.max_drawdown >= D("0")


# ===========================================================================
# Walk-forward and fill-model sensitivity
# ===========================================================================


def test_rolling_windows_do_not_overlap() -> None:
    windows = rolling_windows(
        START, START + timedelta(days=400), timedelta(days=100), timedelta(days=50)
    )
    assert windows
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.trade_end <= later.trade_start


def test_a_window_never_trades_on_its_own_estimation_data() -> None:
    # The property that makes walk-forward mean anything.
    windows = rolling_windows(
        START, START + timedelta(days=400), timedelta(days=100), timedelta(days=50)
    )
    for window in windows:
        assert window.estimation_end <= window.trade_start


def test_a_window_that_overlaps_is_rejected() -> None:
    with pytest.raises(BacktestError, match="overlaps"):
        Window(
            estimation_start=START,
            estimation_end=START + timedelta(days=100),
            trade_start=START + timedelta(days=50),
            trade_end=START + timedelta(days=150),
        )


def test_rolling_windows_reject_non_positive_spans() -> None:
    with pytest.raises(BacktestError):
        rolling_windows(START, START + timedelta(days=10), timedelta(0), timedelta(days=5))


def test_results_are_reported_under_both_fill_models() -> None:
    # SPEC §4.3: reporting only the optimistic number is the amateur tell.
    result = run_under_both_fill_models(config(), make_source(), POLICY, SECTORS, BETAS)

    assert result.optimistic.fill_model == "InstantFillModel"
    assert result.realistic.fill_model == "SpreadCrossFillModel"
    assert result.optimistic.metrics.periods == result.realistic.metrics.periods


def test_the_realistic_fill_model_costs_money() -> None:
    result = run_under_both_fill_models(config(), make_source(), POLICY, SECTORS, BETAS)

    # Crossing the spread and paying commission cannot be free.
    assert result.realistic.total_commission > D("0")
    assert result.realistic.mean_shortfall_bps > D("0")
    assert result.optimistic.mean_shortfall_bps == D("0")


def test_execution_cost_drag_is_reported() -> None:
    result = run_under_both_fill_models(config(), make_source(), POLICY, SECTORS, BETAS)
    # The gap IS the execution-cost sensitivity of the strategy.
    assert result.execution_cost_drag == (
        result.optimistic.metrics.annualized_twr - result.realistic.metrics.annualized_twr
    )


def test_walk_forward_produces_folds() -> None:
    folds = run_walk_forward(
        config(end=START + timedelta(days=700)),
        make_source(days=700),
        POLICY,
        SECTORS,
        BETAS,
        estimation=timedelta(days=200),
        trade=timedelta(days=150),
    )
    assert folds
    for window, result in folds:
        assert window.estimation_end <= window.trade_start
        assert len(result.equity_curve) >= 2


def test_walk_forward_reports_an_impossible_span() -> None:
    with pytest.raises(BacktestError, match="no walk-forward windows"):
        run_walk_forward(
            config(),
            make_source(),
            POLICY,
            SECTORS,
            BETAS,
            estimation=timedelta(days=5000),
            trade=timedelta(days=50),
        )


def test_a_no_trade_cycle_records_why() -> None:
    # A silent no-trade is indistinguishable from a strategy that chose to
    # hold, which is exactly how a misconfigured optimizer hid for an entire
    # backtest during M6. With too few names the constrained frontier is
    # infeasible, and the cycle must say so.
    result = run_backtest(
        config(symbols=("AAA", "BBB", "CCC")),
        make_source(),
        SimulatedExecutor(),
        POLICY,
        SECTORS,
        BETAS,
    )
    no_trade = [c for c in result.cycles if c.target is None]
    assert no_trade, "expected the optimizer to be unable to build a portfolio"
    assert any("optimizer could not" in c.note for c in no_trade)


def test_the_backtest_actually_trades() -> None:
    # The regression test for that bug: a run that executes nothing is not a
    # passing backtest, it is a broken one.
    result = run_backtest(config(), make_source(), SimulatedExecutor(), POLICY, SECTORS, BETAS)
    assert result.executed, "no cycle reached the executor"
    assert result.equity_curve[-1] != result.equity_curve[0]
