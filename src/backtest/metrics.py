"""Performance and attribution metrics (SPEC §6.1, §6.2, §11).

Every figure here is computed by :mod:`src.cfa`; this module only assembles
them from an equity curve and a benchmark.

**TWR is the headline.** GIPS requires time-weighted returns because
chain-linking sub-period returns removes the effect of the *timing* of external
cash flows, isolating the manager's decisions from the client's. MWR is
reported alongside, and the gap between them is the cash-flow timing effect —
explaining that gap is the point of showing both (SPEC §6.1).

**Alpha comes with a t-statistic.** A positive Jensen's alpha is a point
estimate; whether it is distinguishable from zero is a different question, and
the t-stat on the regression intercept is the only honest way to answer it. A
strategy quoting alpha without it is quoting noise it has not ruled out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

from src.cfa.returns import (
    RegressionResult,
    arithmetic_mean_return,
    estimate_beta,
    money_weighted_return,
    sample_standard_deviation,
    time_weighted_return,
)
from src.cfa.portfolio import (
    information_ratio,
    jensens_alpha,
    m_squared,
    sharpe_ratio,
    treynor_ratio,
)

ZERO = Decimal(0)
ONE = Decimal(1)

#: US equity trading days per year.
TRADING_DAYS = 252


class MetricsError(ValueError):
    """Raised when a metric cannot be computed from the given series."""


def periodic_returns(equity_curve: Sequence[Decimal]) -> list[Decimal]:
    """Period-over-period returns from an equity curve."""
    if len(equity_curve) < 2:
        return []
    returns: list[Decimal] = []
    for before, after in zip(equity_curve, equity_curve[1:]):
        if before <= ZERO:
            raise MetricsError(f"non-positive equity {before} in curve")
        returns.append(after / before - ONE)
    return returns


def max_drawdown(equity_curve: Sequence[Decimal]) -> Decimal:
    """Largest peak-to-trough decline, as a positive fraction.

    Measured on the equity curve rather than on returns because that is what an
    investor actually experiences, and it is the quantity the IPS circuit
    breaker is written against (SPEC §7).
    """
    if not equity_curve:
        return ZERO
    peak = equity_curve[0]
    worst = ZERO
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > ZERO:
            decline = (peak - value) / peak
            if decline > worst:
                worst = decline
    return worst


def annualize(total_return: Decimal, periods: int, periods_per_year: int = TRADING_DAYS) -> Decimal:
    """Geometric annualization of a cumulative return."""
    if periods <= 0:
        return ZERO
    growth = ONE + total_return
    if growth <= ZERO:
        # A total loss cannot be annualized; reporting -100% is the honest answer.
        return Decimal(-1)
    years = Decimal(periods) / Decimal(periods_per_year)
    if years <= ZERO:
        return ZERO
    return growth ** (ONE / years) - ONE


def annualized_volatility(
    returns: Sequence[Decimal], periods_per_year: int = TRADING_DAYS
) -> Decimal:
    """Standard deviation of periodic returns, scaled by the square root of time."""
    if len(returns) < 2:
        return ZERO
    return sample_standard_deviation(returns) * Decimal(periods_per_year).sqrt()


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Everything SPEC §11's numeric checklist asks for."""

    periods: int
    total_return: Decimal
    twr: Decimal
    annualized_twr: Decimal
    mwr: Decimal | None
    benchmark_twr: Decimal
    annualized_benchmark_twr: Decimal
    annualized_volatility: Decimal
    max_drawdown: Decimal
    beta: Decimal
    r_squared: Decimal
    sharpe: Decimal | None
    treynor: Decimal | None
    jensens_alpha: Decimal
    alpha_t_stat: Decimal | None
    information_ratio: Decimal | None
    tracking_error: Decimal
    m_squared: Decimal | None
    regression: RegressionResult | None = None

    @property
    def alpha_is_significant(self) -> bool:
        """Two-sided t-test at roughly 5%, using the large-sample critical value.

        Deliberately conservative about what it claims: this says the intercept
        is distinguishable from zero *in sample*, which is not the same as the
        strategy having alpha.
        """
        if self.alpha_t_stat is None:
            return False
        return abs(self.alpha_t_stat) > Decimal("1.96")


def compute_metrics(
    equity_curve: Sequence[Decimal],
    benchmark_curve: Sequence[Decimal],
    risk_free_rate: Decimal,
    cash_flows: Sequence[tuple[datetime, Decimal]] | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceMetrics:
    """Assemble the full metric set from portfolio and benchmark equity curves.

    ``risk_free_rate`` must already be a **bond-equivalent yield**. FRED's
    ``DGS3MO`` is quoted on a discount basis and understates the rate until
    converted (SPEC §6.2 [CORRECTED]) — using it raw inflates every
    risk-adjusted figure below.
    """
    if len(equity_curve) < 2:
        raise MetricsError("need at least two equity points to measure performance")
    if len(benchmark_curve) != len(equity_curve):
        raise MetricsError(
            f"benchmark has {len(benchmark_curve)} points against {len(equity_curve)}"
        )

    portfolio = periodic_returns(equity_curve)
    benchmark = periodic_returns(benchmark_curve)
    periods = len(portfolio)

    twr = time_weighted_return(portfolio)
    benchmark_twr = time_weighted_return(benchmark)
    annual_twr = annualize(twr, periods, periods_per_year)
    annual_benchmark = annualize(benchmark_twr, periods, periods_per_year)

    volatility = annualized_volatility(portfolio, periods_per_year)

    # Per-period risk-free, for the excess-return regression.
    periodic_rf = risk_free_rate / Decimal(periods_per_year)
    excess_portfolio = [r - periodic_rf for r in portfolio]
    excess_benchmark = [r - periodic_rf for r in benchmark]

    regression: RegressionResult | None = None
    beta = ZERO
    r_squared = ZERO
    alpha_t: Decimal | None = None
    if periods >= 3:
        try:
            regression = estimate_beta(excess_portfolio, excess_benchmark)
            beta = regression.slope
            r_squared = regression.r_squared
            alpha_t = regression.t_stat_intercept
        except Exception:  # noqa: BLE001 - a degenerate window is not fatal
            regression = None

    alpha = jensens_alpha(annual_twr, risk_free_rate, beta, annual_benchmark)

    active = [p - b for p, b in zip(portfolio, benchmark)]
    tracking = annualized_volatility(active, periods_per_year)

    sharpe = sharpe_ratio(annual_twr, risk_free_rate, volatility) if volatility > ZERO else None
    treynor = treynor_ratio(annual_twr, risk_free_rate, beta) if beta != ZERO else None
    info = (
        information_ratio(annual_twr, annual_benchmark, tracking) if tracking > ZERO else None
    )
    benchmark_volatility = annualized_volatility(benchmark, periods_per_year)
    m2 = (
        m_squared(annual_twr, risk_free_rate, volatility, benchmark_volatility)
        if volatility > ZERO
        else None
    )

    mwr: Decimal | None = None
    if cash_flows and len(cash_flows) >= 2:
        try:
            mwr = money_weighted_return(cash_flows)
        except ValueError:
            # A series with no sign change has no IRR. Reporting None is
            # correct; inventing a number is not.
            mwr = None

    return PerformanceMetrics(
        periods=periods,
        total_return=twr,
        twr=twr,
        annualized_twr=annual_twr,
        mwr=mwr,
        benchmark_twr=benchmark_twr,
        annualized_benchmark_twr=annual_benchmark,
        annualized_volatility=volatility,
        max_drawdown=max_drawdown(equity_curve),
        beta=beta,
        r_squared=r_squared,
        sharpe=sharpe,
        treynor=treynor,
        jensens_alpha=alpha,
        alpha_t_stat=alpha_t,
        information_ratio=info,
        tracking_error=tracking,
        m_squared=m2,
        regression=regression,
    )


def summarize(metrics: PerformanceMetrics) -> dict[str, str]:
    """Flatten to display strings for the dashboard and the README checklist."""

    def fmt(value: Decimal | None, places: str = "0.0001") -> str:
        return "n/a" if value is None else str(value.quantize(Decimal(places)))

    return {
        "periods": str(metrics.periods),
        "annualized_twr": fmt(metrics.annualized_twr),
        "annualized_benchmark_twr": fmt(metrics.annualized_benchmark_twr),
        "mwr": fmt(metrics.mwr),
        "annualized_volatility": fmt(metrics.annualized_volatility),
        "max_drawdown": fmt(metrics.max_drawdown),
        "beta": fmt(metrics.beta, "0.01"),
        "r_squared": fmt(metrics.r_squared, "0.01"),
        "sharpe": fmt(metrics.sharpe, "0.01"),
        "treynor": fmt(metrics.treynor, "0.0001"),
        "jensens_alpha": fmt(metrics.jensens_alpha),
        "alpha_t_stat": fmt(metrics.alpha_t_stat, "0.01"),
        "information_ratio": fmt(metrics.information_ratio, "0.01"),
        "tracking_error": fmt(metrics.tracking_error),
        "alpha_significant": str(metrics.alpha_is_significant),
    }
