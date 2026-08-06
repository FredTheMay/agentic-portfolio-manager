"""Return and dispersion measures.

CFA Level I topic area: Quantitative Methods (SPEC §6.1).

Pure functions, zero I/O. ``Decimal`` at every public boundary; the regression
and IRR routines solve in float64 and convert back through
:mod:`src.cfa._numeric`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from src.cfa._numeric import (
    NumericError,
    require_min_length,
    require_same_length,
    to_decimal,
    to_float,
    to_float_array,
)
from src.time.clock import ensure_utc

ZERO = Decimal(0)
ONE = Decimal(1)

#: Actual/365 fixed, the XIRR convention.
DAYS_PER_YEAR = Decimal(365)
#: Year fractions are measured in seconds, not whole days. Truncating to days
#: would silently discard the time of an intraday cash flow, which is exactly
#: the assumption SPEC §4.2 exists to avoid.
SECONDS_PER_YEAR = DAYS_PER_YEAR * Decimal(86_400)

#: Widening brackets for the IRR solve. A short holding period annualizes to an
#: enormous rate — a 1% gain over twelve hours is ~1400x annualized — so a
#: bracket wide enough for daily data cannot span intraday. Rather than pick one
#: absurd bound, widen until the root is bracketed.
IRR_BRACKETS = (10.0, 1_000.0, 1e6)


def holding_period_return(
    begin_price: Decimal,
    end_price: Decimal,
    income: Decimal = ZERO,
) -> Decimal:
    """``HPR = (P1 - P0 + D1) / P0``.

    CFA Level I: Quantitative Methods — rates of return.
    """
    if begin_price == ZERO:
        raise ZeroDivisionError("holding period return is undefined for a zero beginning price")
    return (end_price - begin_price + income) / begin_price


def time_weighted_return(subperiod_returns: Sequence[Decimal]) -> Decimal:
    """``TWR = PROD(1 + HPRi) - 1``.

    The headline performance metric. GIPS requires time-weighted returns
    because chain-linking sub-period returns removes the effect of the *timing*
    of external cash flows, isolating the manager's decisions from the client's.

    CFA Level I: Quantitative Methods — time-weighted rate of return.
    """
    compounded = ONE
    for r in subperiod_returns:
        compounded *= ONE + r
    return compounded - ONE


def money_weighted_return(cash_flows: Sequence[tuple[datetime, Decimal]]) -> Decimal:
    """Annualized IRR of a dated cash-flow series (XIRR).

    Reported alongside TWR: the gap between them *is* the cash-flow timing
    effect, and explaining that gap is the point of showing both. Sign
    convention: contributions negative, distributions positive.

    CFA Level I: Quantitative Methods — money-weighted rate of return.
    """
    require_min_length("cash_flows", cash_flows, 2)

    dated = sorted(((ensure_utc(ts), amount) for ts, amount in cash_flows), key=lambda p: p[0])
    amounts = [amount for _, amount in dated]
    if not (any(a > ZERO for a in amounts) and any(a < ZERO for a in amounts)):
        raise ValueError("IRR requires at least one sign change in the cash flows")

    start = dated[0][0]
    # Year fractions on actual/365, measured to the second. Float is required
    # for the solve itself.
    years = [
        to_float(Decimal(str((ts - start).total_seconds())) / SECONDS_PER_YEAR)
        for ts, _ in dated
    ]
    values = [to_float(amount) for _, amount in dated]

    if years[-1] <= 0:
        raise ValueError("all cash flows share one instant; no rate of return is defined")

    def npv(rate: float) -> float:
        return float(sum(v / (1.0 + rate) ** t for v, t in zip(values, years)))

    # -0.9999 rather than -1 because the NPV pole sits at -1.
    low = -0.9999
    base = npv(low)
    for high in IRR_BRACKETS:
        if base * npv(high) <= 0:
            return to_decimal(
                float(brentq(npv, low, high, xtol=1e-12, rtol=1e-12, maxiter=200))
            )

    raise ValueError(
        f"could not bracket an IRR in (-99.99%, {IRR_BRACKETS[-1]:.0%}); "
        "the cash-flow series may have no real internal rate of return"
    )


def geometric_mean_return(returns: Sequence[Decimal]) -> Decimal:
    """``(PROD(1 + Ri))^(1/n) - 1``.

    The compound rate that reproduces the realized ending wealth, which is why
    it is the correct summary of a *past* multi-period record.

    CFA Level I: Quantitative Methods — geometric mean return.
    """
    require_min_length("returns", returns, 1)
    compounded = ONE
    for r in returns:
        if r <= -ONE:
            raise ValueError(f"return {r} wipes out the position; geometric mean is undefined")
        compounded *= ONE + r
    return compounded ** (ONE / Decimal(len(returns))) - ONE


def arithmetic_mean_return(returns: Sequence[Decimal]) -> Decimal:
    """``sum(Ri) / n``. The best estimate of a *single future* period's return.

    CFA Level I: Quantitative Methods — arithmetic mean.
    """
    require_min_length("returns", returns, 1)
    return sum(returns, ZERO) / Decimal(len(returns))


def sample_variance(returns: Sequence[Decimal]) -> Decimal:
    """``sum((Ri - Rbar)^2) / (n - 1)``.

    Divides by ``n - 1``: with the mean estimated from the same sample, the
    ``n`` divisor is biased downward.

    CFA Level I: Quantitative Methods — sample variance.
    """
    require_min_length("returns", returns, 2)
    mean = arithmetic_mean_return(returns)
    total = sum(((r - mean) ** 2 for r in returns), ZERO)
    return total / Decimal(len(returns) - 1)


def sample_standard_deviation(returns: Sequence[Decimal]) -> Decimal:
    """Square root of :func:`sample_variance`.

    CFA Level I: Quantitative Methods — sample standard deviation.
    """
    return sample_variance(returns).sqrt()


def covariance(x: Sequence[Decimal], y: Sequence[Decimal]) -> Decimal:
    """``sum((xi - xbar)(yi - ybar)) / (n - 1)``.

    CFA Level I: Quantitative Methods — covariance.
    """
    require_same_length("x", x, "y", y)
    require_min_length("x", x, 2)
    mean_x = arithmetic_mean_return(x)
    mean_y = arithmetic_mean_return(y)
    total = sum(((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)), ZERO)
    return total / Decimal(len(x) - 1)


def correlation(x: Sequence[Decimal], y: Sequence[Decimal]) -> Decimal:
    """``rho = Cov(x, y) / (sd_x * sd_y)``, in ``[-1, 1]``.

    CFA Level I: Quantitative Methods — correlation.
    """
    sd_x = sample_standard_deviation(x)
    sd_y = sample_standard_deviation(y)
    if sd_x == ZERO or sd_y == ZERO:
        raise NumericError("correlation is undefined when a series has zero variance")
    return covariance(x, y) / (sd_x * sd_y)


def coefficient_of_variation(returns: Sequence[Decimal]) -> Decimal:
    """``CV = sd / mean`` — risk per unit of return, so lower is better.

    CFA Level I: Quantitative Methods — coefficient of variation.
    """
    mean = arithmetic_mean_return(returns)
    if mean == ZERO:
        raise NumericError("coefficient of variation is undefined for a zero mean")
    return sample_standard_deviation(returns) / mean


def downside_deviation(
    returns: Sequence[Decimal],
    minimum_acceptable_return: Decimal,
) -> Decimal:
    """Target semideviation about a minimum acceptable return.

    Only shortfalls enter the numerator, but the denominator is ``n - 1`` over
    *all* observations — the CFA convention. Upside dispersion is not risk,
    which is the whole argument for preferring this to standard deviation.

    CFA Level I: Quantitative Methods — downside deviation.
    """
    require_min_length("returns", returns, 2)
    shortfalls = [
        (r - minimum_acceptable_return) ** 2 for r in returns if r < minimum_acceptable_return
    ]
    if not shortfalls:
        return ZERO
    return (sum(shortfalls, ZERO) / Decimal(len(returns) - 1)).sqrt()


def continuously_compounded_return(holding_period: Decimal) -> Decimal:
    """``R_cc = ln(1 + HPR)``.

    Log returns are additive across time, which is what makes them the right
    scale for aggregating and for distributional assumptions.

    CFA Level I: Quantitative Methods — continuously compounded returns.
    """
    growth = ONE + holding_period
    if growth <= ZERO:
        raise ValueError(f"ln is undefined for a growth factor of {growth}")
    return growth.ln()


def safety_first_ratio(
    expected_return: Decimal,
    threshold_return: Decimal,
    standard_deviation: Decimal,
) -> Decimal:
    """Roy's safety-first ratio ``(E(Rp) - R_L) / sigma_p``.

    **Maximized**, not minimized: the optimal portfolio is the one that
    minimizes the probability of returning less than the threshold, and under
    normality that is the one furthest above ``R_L`` in standard deviations.

    CFA Level I: Quantitative Methods — Roy's safety-first criterion.
    """
    if standard_deviation <= ZERO:
        raise NumericError("safety-first ratio requires a positive standard deviation")
    return (expected_return - threshold_return) / standard_deviation


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """A simple OLS fit, reported with its uncertainty.

    Values are ``Decimal``, converted once from the float64 least-squares fit.
    ``t_stat_intercept`` is the statistic that decides whether Jensen's alpha
    is distinguishable from zero.
    """

    slope: Decimal
    intercept: Decimal
    r_squared: Decimal
    se_slope: Decimal
    se_intercept: Decimal
    t_stat_slope: Decimal | None
    t_stat_intercept: Decimal | None
    observations: int


def ols_regression(y: Sequence[Decimal], x: Sequence[Decimal]) -> RegressionResult:
    """Least-squares fit of ``y`` on ``x``, with standard errors and R-squared.

    CFA Level I: Quantitative Methods — simple linear regression.
    """
    require_same_length("y", y, "x", x)
    # Two parameters are estimated, so n - 2 residual degrees of freedom
    # requires at least three points for the standard errors to exist.
    require_min_length("x", x, 3)

    y_arr = to_float_array(y)
    x_arr = to_float_array(x)
    n = len(x_arr)

    mean_x = float(np.mean(x_arr))
    mean_y = float(np.mean(y_arr))
    sxx = float(np.sum((x_arr - mean_x) ** 2))
    if sxx == 0.0:
        raise NumericError("regressor has zero variance; slope is undefined")

    sxy = float(np.sum((x_arr - mean_x) * (y_arr - mean_y)))
    syy = float(np.sum((y_arr - mean_y) ** 2))

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    residual_ss = float(np.sum((y_arr - (intercept + slope * x_arr)) ** 2))
    r_squared = 1.0 if syy == 0.0 else 1.0 - residual_ss / syy

    # Residual variance on n - 2 degrees of freedom.
    sigma_squared = residual_ss / (n - 2)
    se_slope = float(np.sqrt(sigma_squared / sxx))
    se_intercept = float(np.sqrt(sigma_squared * (1.0 / n + mean_x**2 / sxx)))

    # A perfect fit leaves no residual variance, so the t-statistics are not
    # defined. Reporting None is honest; reporting infinity is not.
    t_slope = to_decimal(slope / se_slope) if se_slope > 0.0 else None
    t_intercept = to_decimal(intercept / se_intercept) if se_intercept > 0.0 else None

    return RegressionResult(
        slope=to_decimal(slope),
        intercept=to_decimal(intercept),
        r_squared=to_decimal(r_squared),
        se_slope=to_decimal(se_slope),
        se_intercept=to_decimal(se_intercept),
        t_stat_slope=t_slope,
        t_stat_intercept=t_intercept,
        observations=n,
    )


def estimate_beta(
    asset_excess_returns: Sequence[Decimal],
    market_excess_returns: Sequence[Decimal],
) -> RegressionResult:
    """Estimate beta by regressing excess returns on excess market returns.

    Deliberately *not* the ``Cov/Var`` shortcut. That gives the point estimate
    and nothing else; the regression additionally yields R² — the systematic
    share of variance — plus the standard errors and the t-statistic on the
    intercept, without which "the strategy has positive alpha" is an unfalsifiable
    claim.

    CFA Level I: Portfolio Management — beta estimation (SPEC §6.1, §6.2).
    """
    return ols_regression(asset_excess_returns, market_excess_returns)
