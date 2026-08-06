"""Golden tests for the CFA Level I core (SPEC §6).

Every expected value here is hand-computed from the formula in the spec and
written as a literal. Nothing is asserted against the implementation's own
output — a test that computes the answer the same way the code does proves only
that the code is self-consistent.

Sections follow SPEC §6.1 through §6.8.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Callable

import pytest

from src.cfa import _numeric as num
from src.cfa import alternatives as alt
from src.cfa import derivatives as dv
from src.cfa import fixed_income as fi
from src.cfa import portfolio as pf
from src.cfa import ratios as rt
from src.cfa import returns as ret
from src.cfa import valuation as val
from src.time.clock import UTC

D = Decimal


def approx(actual: Decimal, expected: str, places: int = 10) -> None:
    """Assert ``actual`` matches a hand-computed literal to ``places``."""
    tolerance = D(10) ** -places
    assert abs(actual - D(expected)) < tolerance, f"{actual} != {expected}"


def at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# ===========================================================================
# §6.1 Quantitative Methods — src/cfa/returns.py
# ===========================================================================


def test_holding_period_return() -> None:
    # HPR = (P1 - P0 + D1) / P0 = (110 - 100 + 2) / 100 = 0.12
    assert ret.holding_period_return(D("100"), D("110"), D("2")) == D("0.12")


def test_holding_period_return_without_income() -> None:
    assert ret.holding_period_return(D("50"), D("45")) == D("-0.10")


def test_holding_period_return_rejects_zero_base() -> None:
    with pytest.raises(ZeroDivisionError):
        ret.holding_period_return(D("0"), D("10"))


def test_time_weighted_return() -> None:
    # TWR = PROD(1 + HPRi) - 1
    #     = 1.10 * 0.95 * 1.08 - 1 = 1.1286 - 1 = 0.1286
    assert ret.time_weighted_return([D("0.10"), D("-0.05"), D("0.08")]) == D("0.1286")


def test_time_weighted_return_of_empty_series_is_zero() -> None:
    assert ret.time_weighted_return([]) == D("0")


def test_money_weighted_return_single_period() -> None:
    # -100 at t0, +110 one year later (365 days) -> IRR = 10%
    flows = [(at(2023, 1, 1), D("-100")), (at(2024, 1, 1), D("110"))]
    approx(ret.money_weighted_return(flows), "0.10", places=8)


def test_money_weighted_return_multi_period() -> None:
    # -100, -100, +231 at one-year spacing.
    # NPV at 10%: -100 - 100/1.1 + 231/1.21 = -100 - 90.9091 + 190.9091 = 0
    flows = [
        (at(2021, 1, 1), D("-100")),
        (at(2022, 1, 1), D("-100")),
        (at(2023, 1, 1), D("231")),
    ]
    approx(ret.money_weighted_return(flows), "0.10", places=8)


def test_money_weighted_return_measures_time_to_the_second() -> None:
    # SPEC §4.2: timestamps are instants, never dates. Truncating year
    # fractions to whole days made these two series indistinguishable, which
    # silently assumed daily data in the one metric most exposed to timing.
    at_midnight = [(at(2023, 1, 1), D("-100")), (at(2023, 6, 30), D("110"))]
    at_noon = [
        (at(2023, 1, 1), D("-100")),
        (datetime(2023, 6, 30, 12, tzinfo=UTC), D("110")),
    ]

    earlier = ret.money_weighted_return(at_midnight)
    later = ret.money_weighted_return(at_noon)

    assert earlier != later, "time of day must affect an annualized rate"
    # The same gain taken half a day later annualizes to slightly less.
    assert later < earlier


def test_money_weighted_return_handles_an_intraday_round_trip() -> None:
    # A 1% gain over twelve hours annualizes to roughly 1400x. The bracket has
    # to be wide enough to contain it, or the whole intraday claim is untested.
    flows = [
        (at(2023, 1, 1), D("-100")),
        (datetime(2023, 1, 1, 12, tzinfo=UTC), D("101")),
    ]
    result = ret.money_weighted_return(flows)
    assert result > D("1000"), f"expected an enormous annualized rate, got {result}"


def test_money_weighted_return_rejects_a_zero_length_holding() -> None:
    # Every flow at one instant: there is no period to earn a return over.
    with pytest.raises(ValueError, match="one instant"):
        ret.money_weighted_return([(at(2023, 1, 1), D("-100")), (at(2023, 1, 1), D("110"))])


def test_money_weighted_return_requires_a_sign_change() -> None:
    flows = [(at(2023, 1, 1), D("-100")), (at(2024, 1, 1), D("-50"))]
    with pytest.raises(ValueError, match="sign change"):
        ret.money_weighted_return(flows)


def test_twr_and_mwr_diverge_on_mid_period_cash_flow() -> None:
    # SPEC §6.1: GIPS requires TWR because it isolates the strategy from the
    # timing of external cash flows. This is the case that shows why.
    #
    # +100% then -50%: TWR = 2.0 * 0.5 - 1 = 0. But contributing more money
    # just before the loss makes the investor's own experience negative.
    twr = ret.time_weighted_return([D("1.00"), D("-0.50")])
    assert twr == D("0")

    flows = [
        (at(2021, 1, 1), D("-100")),  # initial
        (at(2022, 1, 1), D("-1000")),  # contribution after the doubling
        (at(2023, 1, 1), D("600")),  # (100*2 + 1000) * 0.5
    ]
    mwr = ret.money_weighted_return(flows)
    assert mwr < D("0"), "MWR must reflect the badly timed contribution"


def test_geometric_mean_return() -> None:
    # (1.331)^(1/3) - 1 = 1.1 - 1 = 0.10
    approx(ret.geometric_mean_return([D("0.331"), D("0"), D("0")]), "0.10")


def test_geometric_mean_return_two_periods() -> None:
    # sqrt(1.44) - 1 = 1.2 - 1 = 0.20
    approx(ret.geometric_mean_return([D("0.44"), D("0")]), "0.20")


def test_arithmetic_mean_return() -> None:
    # (0.10 - 0.05 + 0.08 + 0.07) / 4 = 0.20 / 4 = 0.05
    assert ret.arithmetic_mean_return([D("0.10"), D("-0.05"), D("0.08"), D("0.07")]) == D("0.05")


def test_geometric_mean_never_exceeds_arithmetic_mean() -> None:
    series = [D("0.20"), D("-0.10"), D("0.15")]
    assert ret.geometric_mean_return(series) <= ret.arithmetic_mean_return(series)


def test_sample_variance_and_standard_deviation() -> None:
    # mean 0.20; deviations -0.1, 0, +0.1; squares sum to 0.02
    # sample variance = 0.02 / (3 - 1) = 0.01; sd = 0.10
    series = [D("0.10"), D("0.20"), D("0.30")]
    assert ret.sample_variance(series) == D("0.01")
    approx(ret.sample_standard_deviation(series), "0.10")


def test_sample_variance_uses_n_minus_1() -> None:
    # Population variance would be 0.02/3; the sample estimator divides by n-1.
    series = [D("0.10"), D("0.20"), D("0.30")]
    assert ret.sample_variance(series) == D("0.01")
    assert ret.sample_variance(series) != D("0.02") / D("3")


def test_sample_variance_needs_two_observations() -> None:
    with pytest.raises(Exception):
        ret.sample_variance([D("0.10")])


def test_covariance_and_correlation() -> None:
    # x = 1,2,3 (mean 2); y = 2,4,6 (mean 4)
    # cov = ((-1)(-2) + 0 + (1)(2)) / 2 = 4/2 = 2
    # sd_x = 1, sd_y = 2  ->  rho = 2 / (1 * 2) = 1.0 (perfectly linear)
    x = [D("1"), D("2"), D("3")]
    y = [D("2"), D("4"), D("6")]
    assert ret.covariance(x, y) == D("2")
    approx(ret.correlation(x, y), "1.0")


def test_correlation_of_inverse_series_is_negative_one() -> None:
    x = [D("1"), D("2"), D("3")]
    y = [D("6"), D("4"), D("2")]
    approx(ret.correlation(x, y), "-1.0")


def test_covariance_rejects_mismatched_lengths() -> None:
    with pytest.raises(Exception):
        ret.covariance([D("1"), D("2")], [D("1")])


def test_coefficient_of_variation() -> None:
    # CV = sd / mean = 0.10 / 0.20 = 0.5
    approx(ret.coefficient_of_variation([D("0.10"), D("0.20"), D("0.30")]), "0.5")


def test_downside_deviation() -> None:
    # Target semideviation about MAR = 0. Only returns below MAR contribute,
    # but the denominator is n - 1 over all observations (CFA convention).
    # deviations below: -0.10, -0.10 -> 0.01 + 0.01 = 0.02
    # 0.02 / (3 - 1) = 0.01 -> sqrt = 0.10
    approx(ret.downside_deviation([D("0.10"), D("-0.10"), D("-0.10")], D("0")), "0.10")


def test_downside_deviation_ignores_upside() -> None:
    # Two series with identical downside but different upside must agree.
    a = ret.downside_deviation([D("0.10"), D("-0.10"), D("-0.10")], D("0"))
    b = ret.downside_deviation([D("9.99"), D("-0.10"), D("-0.10")], D("0"))
    assert a == b


def test_downside_deviation_is_zero_when_nothing_is_below_target() -> None:
    assert ret.downside_deviation([D("0.10"), D("0.20")], D("0")) == D("0")


def test_continuously_compounded_return() -> None:
    # R_cc = ln(1 + HPR) = ln(1.10) = 0.0953101798...
    approx(ret.continuously_compounded_return(D("0.10")), "0.0953101798043")


def test_continuously_compounded_return_is_additive() -> None:
    # The property that makes log returns useful: they sum across periods.
    total = ret.time_weighted_return([D("0.10"), D("0.10")])
    summed = D("2") * ret.continuously_compounded_return(D("0.10"))
    approx(ret.continuously_compounded_return(total), str(summed), places=12)


def test_continuously_compounded_return_rejects_total_loss() -> None:
    with pytest.raises(ValueError):
        ret.continuously_compounded_return(D("-1"))


def test_safety_first_ratio() -> None:
    # SFRatio = (E(Rp) - R_L) / sigma_p = (0.12 - 0.03) / 0.18 = 0.5
    approx(ret.safety_first_ratio(D("0.12"), D("0.03"), D("0.18")), "0.5")


def test_safety_first_ratio_prefers_the_higher_value() -> None:
    # SPEC §6.1 [CORRECTED]: Roy's criterion is maximized, not minimized.
    safer = ret.safety_first_ratio(D("0.12"), D("0.03"), D("0.10"))
    riskier = ret.safety_first_ratio(D("0.12"), D("0.03"), D("0.30"))
    assert safer > riskier


def test_ols_regression_textbook_values() -> None:
    # x = 1..5, y = 2,4,5,4,5. Hand-computed:
    #   mean x = 3, mean y = 4
    #   Sxy = (-2)(-2) + (-1)(0) + (0)(1) + (1)(0) + (2)(1) = 6
    #   Sxx = 4 + 1 + 0 + 1 + 4 = 10        Syy = 4 + 0 + 1 + 0 + 1 = 6
    #   slope     = 6 / 10 = 0.6
    #   intercept = 4 - 0.6(3) = 2.2
    #   SSR = slope^2 * Sxx = 3.6   SSE = 6 - 3.6 = 2.4   R^2 = 3.6/6 = 0.6
    #   s^2 = SSE/(n-2) = 0.8
    #   SE(slope)     = sqrt(0.8/10)                = 0.2828427125
    #   SE(intercept) = sqrt(0.8 * (1/5 + 9/10))    = 0.9380831520
    #   t(intercept)  = 2.2 / 0.9380831520          = 2.3452078799
    y = [D("2"), D("4"), D("5"), D("4"), D("5")]
    x = [D("1"), D("2"), D("3"), D("4"), D("5")]
    fit = ret.ols_regression(y, x)

    approx(fit.slope, "0.6", places=9)
    approx(fit.intercept, "2.2", places=9)
    approx(fit.r_squared, "0.6", places=9)
    approx(fit.se_slope, "0.2828427125", places=9)
    approx(fit.se_intercept, "0.9380831520", places=9)
    assert fit.t_stat_intercept is not None and fit.t_stat_slope is not None
    approx(fit.t_stat_intercept, "2.3452078799", places=8)
    approx(fit.t_stat_slope, "2.1213203436", places=8)
    assert fit.observations == 5


def test_ols_regression_perfect_fit_has_unit_r_squared() -> None:
    y = [D("3"), D("5"), D("7"), D("9")]
    x = [D("1"), D("2"), D("3"), D("4")]
    fit = ret.ols_regression(y, x)
    approx(fit.slope, "2.0", places=9)
    approx(fit.intercept, "1.0", places=9)
    approx(fit.r_squared, "1.0", places=9)


def test_estimate_beta_regresses_excess_returns() -> None:
    # SPEC §6.2 [CORRECTED]: beta by regression, not Cov/Var, so R^2 and the
    # standard errors come with it. An asset that moves 1.5x the market with no
    # idiosyncratic noise has beta 1.5 and R^2 = 1.
    market = [D("0.02"), D("-0.01"), D("0.03"), D("0.00"), D("0.015")]
    asset = [D("0.03"), D("-0.015"), D("0.045"), D("0.00"), D("0.0225")]
    fit = ret.estimate_beta(asset, market)

    approx(fit.slope, "1.5", places=9)
    approx(fit.r_squared, "1.0", places=9)


def test_estimate_beta_reports_alpha_significance() -> None:
    # SPEC §6.1: the t-stat on the intercept is the only honest way to say
    # whether Jensen's alpha is distinguishable from zero.
    market = [D("0.02"), D("-0.01"), D("0.03"), D("0.00"), D("0.015")]
    asset = [D("0.03"), D("-0.015"), D("0.045"), D("0.005"), D("0.0225")]
    fit = ret.estimate_beta(asset, market)
    assert fit.t_stat_intercept is not None


def test_regression_needs_more_points_than_parameters() -> None:
    with pytest.raises(Exception):
        ret.ols_regression([D("1"), D("2")], [D("1"), D("2")])


# ===========================================================================
# §6.2 Portfolio Management — src/cfa/portfolio.py
# ===========================================================================

# Two-asset workhorse: sd 20% and 10%, uncorrelated.
#   Sigma = [[0.04, 0.00],
#            [0.00, 0.01]]
COV_2 = [[D("0.04"), D("0.00")], [D("0.00"), D("0.01")]]
MU_2 = [D("0.10"), D("0.05")]

# Same assets at rho = 0.5:  cov = 0.5 * 0.20 * 0.10 = 0.01
COV_2_CORRELATED = [[D("0.04"), D("0.01")], [D("0.01"), D("0.01")]]


def test_expected_portfolio_return() -> None:
    # E(Rp) = 0.6(0.10) + 0.4(0.05) = 0.06 + 0.02 = 0.08
    assert pf.expected_portfolio_return([D("0.6"), D("0.4")], MU_2) == D("0.08")


def test_expected_portfolio_return_rejects_length_mismatch() -> None:
    with pytest.raises(Exception):
        pf.expected_portfolio_return([D("1.0")], MU_2)


def test_two_asset_variance() -> None:
    # w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 s1 s2 rho
    # = 0.36(0.04) + 0.16(0.01) + 2(0.6)(0.4)(0.20)(0.10)(0.5)
    # = 0.0144 + 0.0016 + 0.0048 = 0.0208
    variance = pf.two_asset_variance(D("0.6"), D("0.4"), D("0.20"), D("0.10"), D("0.5"))
    approx(variance, "0.0208")


def test_portfolio_variance_matches_the_two_asset_formula() -> None:
    # SPEC §6.2 names the two-asset variance as the unit-test check on w'Sigma w.
    weights = [D("0.6"), D("0.4")]
    quadratic_form = pf.portfolio_variance(weights, COV_2_CORRELATED)
    closed_form = pf.two_asset_variance(D("0.6"), D("0.4"), D("0.20"), D("0.10"), D("0.5"))
    approx(quadratic_form, str(closed_form), places=12)
    approx(quadratic_form, "0.0208", places=12)


def test_portfolio_standard_deviation() -> None:
    # sqrt(0.0208) = 0.1442220510...
    sd = pf.portfolio_standard_deviation([D("0.6"), D("0.4")], COV_2_CORRELATED)
    approx(sd, "0.1442220510", places=9)


def test_diversification_reduces_risk_below_the_weighted_average() -> None:
    # The core result: with rho < 1, portfolio sd is strictly below the
    # weighted average of the component sds.
    weights = [D("0.6"), D("0.4")]
    sd = pf.portfolio_standard_deviation(weights, COV_2_CORRELATED)
    weighted_average = D("0.6") * D("0.20") + D("0.4") * D("0.10")
    assert sd < weighted_average


def test_portfolio_variance_rejects_non_square_matrix() -> None:
    with pytest.raises(Exception):
        pf.portfolio_variance([D("0.5"), D("0.5")], [[D("0.04"), D("0.0")]])


def test_minimum_variance_portfolio() -> None:
    # Uncorrelated: w1 = s2^2 / (s1^2 + s2^2) = 0.01 / 0.05 = 0.2
    weights = pf.minimum_variance_portfolio(COV_2)
    approx(weights[0], "0.2", places=9)
    approx(weights[1], "0.8", places=9)
    approx(sum(weights, D(0)), "1.0", places=9)


def test_minimum_variance_portfolio_variance() -> None:
    # 0.04(0.04) + 0.64(0.01) = 0.0016 + 0.0064 = 0.008
    weights = pf.minimum_variance_portfolio(COV_2)
    approx(pf.portfolio_variance(weights, COV_2), "0.008", places=9)


def test_minimum_variance_portfolio_is_the_global_minimum() -> None:
    mvp = pf.portfolio_variance(pf.minimum_variance_portfolio(COV_2), COV_2)
    for w1 in ("0.0", "0.1", "0.3", "0.5", "0.9", "1.0"):
        other = [D(w1), D("1") - D(w1)]
        assert mvp <= pf.portfolio_variance(other, COV_2) + D("1e-12")


def test_tangency_portfolio() -> None:
    # w proportional to Sigma^-1 (mu - rf 1)
    #   excess = [0.08, 0.03];  Sigma^-1 = [[25, 0], [0, 100]]
    #   Sigma^-1 excess = [2, 3];  normalize by 5 -> [0.4, 0.6]
    weights = pf.tangency_portfolio(MU_2, COV_2, D("0.02"))
    approx(weights[0], "0.4", places=9)
    approx(weights[1], "0.6", places=9)


def test_tangency_portfolio_maximizes_the_sharpe_ratio() -> None:
    # E(Rp) = 0.4(0.10) + 0.6(0.05) = 0.07;  var = 0.16(0.04) + 0.36(0.01) = 0.01
    # Sharpe = (0.07 - 0.02) / 0.10 = 0.5
    rf = D("0.02")
    weights = pf.tangency_portfolio(MU_2, COV_2, rf)
    best = pf.sharpe_ratio(
        pf.expected_portfolio_return(weights, MU_2),
        rf,
        pf.portfolio_standard_deviation(weights, COV_2),
    )
    approx(best, "0.5", places=9)

    for w1 in ("0.0", "0.2", "0.3", "0.5", "0.7", "1.0"):
        other = [D(w1), D("1") - D(w1)]
        rival = pf.sharpe_ratio(
            pf.expected_portfolio_return(other, MU_2),
            rf,
            pf.portfolio_standard_deviation(other, COV_2),
        )
        assert rival <= best + D("1e-9")


def test_efficient_frontier_is_ordered_and_starts_at_the_minimum_variance() -> None:
    frontier = pf.efficient_frontier(MU_2, COV_2, points=9)
    assert len(frontier) == 9

    returns = [p.expected_return for p in frontier]
    assert returns == sorted(returns), "frontier must be ordered by expected return"

    lowest_sd = min(p.standard_deviation for p in frontier)
    approx(lowest_sd, "0.0894427191", places=5)  # sqrt(0.008)

    for point in frontier:
        approx(sum(point.weights, D(0)), "1.0", places=6)
        assert all(w >= D("-1e-6") for w in point.weights), "long-only by default"


def test_efficient_frontier_dominates_interior_portfolios() -> None:
    # For any frontier point there is no long-only portfolio with the same
    # return and lower risk. Check against a grid.
    frontier = pf.efficient_frontier(MU_2, COV_2, points=5)
    for point in frontier:
        for w1 in ("0.0", "0.25", "0.5", "0.75", "1.0"):
            other = [D(w1), D("1") - D(w1)]
            if pf.expected_portfolio_return(other, MU_2) >= point.expected_return:
                assert (
                    pf.portfolio_standard_deviation(other, COV_2)
                    >= point.standard_deviation - D("1e-6")
                )


def test_capm_expected_return() -> None:
    # E(Ri) = Rf + beta(E(Rm) - Rf) = 0.03 + 1.2(0.10 - 0.03) = 0.114
    approx(pf.capm_expected_return(D("0.03"), D("1.2"), D("0.10")), "0.114")


def test_security_market_line_is_capm() -> None:
    # The SML is CAPM drawn against beta; a zero-beta asset earns Rf and the
    # market portfolio (beta = 1) earns E(Rm).
    approx(pf.security_market_line(D("0.03"), D("0"), D("0.10")), "0.03")
    approx(pf.security_market_line(D("0.03"), D("1"), D("0.10")), "0.10")


def test_portfolio_beta() -> None:
    # beta_p = 0.5(1.4) + 0.5(0.8) = 1.1
    approx(pf.portfolio_beta([D("0.5"), D("0.5")], [D("1.4"), D("0.8")]), "1.1")


def test_capital_allocation_line() -> None:
    # E(Rp) = Rf + [(E(Ri) - Rf)/sd_i] sd_p
    #       = 0.03 + [(0.13 - 0.03)/0.20](0.10) = 0.03 + 0.05 = 0.08
    approx(pf.capital_allocation_line(D("0.03"), D("0.13"), D("0.20"), D("0.10")), "0.08")


def test_capital_market_line() -> None:
    # The CAL drawn from the market portfolio: slope is the market Sharpe ratio.
    # 0.03 + [(0.10 - 0.03)/0.15](0.30) = 0.03 + 0.14 = 0.17
    approx(pf.capital_market_line(D("0.03"), D("0.10"), D("0.15"), D("0.30")), "0.17")


def test_jensens_alpha() -> None:
    # alpha = Rp - [Rf + beta(Rm - Rf)] = 0.13 - 0.114 = 0.016
    approx(pf.jensens_alpha(D("0.13"), D("0.03"), D("1.2"), D("0.10")), "0.016")


def test_jensens_alpha_is_zero_on_the_sml() -> None:
    fair = pf.capm_expected_return(D("0.03"), D("1.2"), D("0.10"))
    approx(pf.jensens_alpha(fair, D("0.03"), D("1.2"), D("0.10")), "0.0")


def test_sharpe_ratio() -> None:
    # (0.13 - 0.03) / 0.20 = 0.5
    approx(pf.sharpe_ratio(D("0.13"), D("0.03"), D("0.20")), "0.5")


def test_treynor_ratio() -> None:
    # (0.13 - 0.03) / 1.2 = 0.0833333333
    approx(pf.treynor_ratio(D("0.13"), D("0.03"), D("1.2")), "0.0833333333", places=9)


def test_m_squared() -> None:
    # (Rp - Rf)(sd_m / sd_p) + Rf = (0.10)(0.15/0.20) + 0.03 = 0.105
    approx(pf.m_squared(D("0.13"), D("0.03"), D("0.20"), D("0.15")), "0.105")


def test_m_squared_equals_market_return_for_a_market_matching_sharpe() -> None:
    # A portfolio with the market's Sharpe ratio levers to exactly E(Rm).
    approx(pf.m_squared(D("0.10"), D("0.03"), D("0.15"), D("0.15")), "0.10")


def test_information_ratio() -> None:
    # (Rp - Rb) / tracking error = (0.13 - 0.10) / 0.05 = 0.6
    approx(pf.information_ratio(D("0.13"), D("0.10"), D("0.05")), "0.6")


def test_risk_decomposition() -> None:
    # systematic  = beta^2 sd_m^2 = 1.44(0.0225) = 0.0324
    # total       = 0.20^2 = 0.04
    # unsystematic= 0.04 - 0.0324 = 0.0076
    decomposition = pf.risk_decomposition(D("1.2"), D("0.15"), D("0.20"))
    approx(decomposition.systematic_variance, "0.0324")
    approx(decomposition.unsystematic_variance, "0.0076")
    approx(decomposition.total_variance, "0.04")


def test_risk_decomposition_components_sum_to_total() -> None:
    decomposition = pf.risk_decomposition(D("0.9"), D("0.18"), D("0.25"))
    approx(
        decomposition.systematic_variance + decomposition.unsystematic_variance,
        str(decomposition.total_variance),
        places=12,
    )


def test_risk_decomposition_rejects_impossible_inputs() -> None:
    # Systematic variance cannot exceed total variance.
    with pytest.raises(Exception):
        pf.risk_decomposition(D("2.0"), D("0.20"), D("0.10"))


# --- Ledoit-Wolf shrinkage (SPEC §6.2 [CORRECTED]) -------------------------

# 6 observations of 4 assets: fewer observations than a stable 4x4 estimate
# needs, which is exactly the regime where the sample matrix misbehaves.
OBSERVATIONS = [
    [D("0.010"), D("0.020"), D("-0.005"), D("0.012")],
    [D("-0.008"), D("-0.015"), D("0.004"), D("-0.010")],
    [D("0.015"), D("0.030"), D("-0.002"), D("0.018")],
    [D("0.002"), D("0.005"), D("0.001"), D("0.003")],
    [D("-0.012"), D("-0.020"), D("0.006"), D("-0.014")],
    [D("0.006"), D("0.010"), D("-0.003"), D("0.007")],
]


def test_ledoit_wolf_returns_a_symmetric_matrix() -> None:
    result = pf.ledoit_wolf_covariance(OBSERVATIONS)
    matrix = result.covariance
    assert len(matrix) == 4
    for i in range(4):
        for j in range(4):
            approx(matrix[i][j], str(matrix[j][i]), places=15)


def test_ledoit_wolf_shrinkage_intensity_is_a_proportion() -> None:
    result = pf.ledoit_wolf_covariance(OBSERVATIONS)
    assert D("0") <= result.shrinkage <= D("1")


def test_ledoit_wolf_shrinks_correlations_toward_zero() -> None:
    # The whole point: the sample matrix overstates off-diagonal structure when
    # observations are scarce, and MVO is notoriously sensitive to exactly that.
    # Shrinkage pulls off-diagonals toward the diagonal target.
    shrunk = pf.ledoit_wolf_covariance(OBSERVATIONS).covariance
    sample = pf.sample_covariance_matrix(OBSERVATIONS)
    assert abs(shrunk[0][1]) < abs(sample[0][1])


def test_ledoit_wolf_output_is_positive_definite() -> None:
    # A covariance matrix that is not positive definite breaks the optimizer.
    result = pf.ledoit_wolf_covariance(OBSERVATIONS)
    weights = pf.minimum_variance_portfolio(result.covariance)
    assert pf.portfolio_variance(weights, result.covariance) > D("0")


def test_sample_covariance_matrix_matches_pairwise_covariance() -> None:
    matrix = pf.sample_covariance_matrix(OBSERVATIONS)
    column_0 = [row[0] for row in OBSERVATIONS]
    column_1 = [row[1] for row in OBSERVATIONS]
    approx(matrix[0][1], str(ret.covariance(column_0, column_1)), places=12)
    approx(matrix[0][0], str(ret.sample_variance(column_0)), places=12)


# ===========================================================================
# §6.4 Financial Statement Analysis — src/cfa/ratios.py
# ===========================================================================

# One consistent set of statements, used by every ratio test below.
#
#   Income statement            Balance sheet (averages)
#   -----------------------     ------------------------------
#   Revenue          1,000      Total assets           2,000
#   COGS               600      Total equity             800
#   Gross profit       400      Total debt               600
#   Operating exp.     200      Inventory                200
#   EBIT               200      Receivables              125
#   Interest            50      Cash                     100
#   EBT                150      ST investments            50
#   Tax                 45      Current assets           500
#   Net income         105      Current liabilities      250
#
#   CFO 180

REVENUE = D("1000")
COGS = D("600")
GROSS_PROFIT = D("400")
EBIT = D("200")
INTEREST = D("50")
EBT = D("150")
NET_INCOME = D("105")
CFO = D("180")

AVG_ASSETS = D("2000")
AVG_EQUITY = D("800")
TOTAL_DEBT = D("600")
AVG_INVENTORY = D("200")
AVG_RECEIVABLES = D("125")


def test_current_ratio() -> None:
    # 500 / 250 = 2.0
    assert rt.current_ratio(D("500"), D("250")) == D("2")


def test_quick_ratio_excludes_inventory() -> None:
    # (cash + short-term investments + receivables) / current liabilities
    # = (100 + 50 + 150) / 250 = 1.2
    # Below the current ratio of 2.0 precisely because inventory is dropped.
    approx(rt.quick_ratio(D("100"), D("50"), D("150"), D("250")), "1.2")


def test_debt_to_equity() -> None:
    # 600 / 800 = 0.75
    assert rt.debt_to_equity(TOTAL_DEBT, AVG_EQUITY) == D("0.75")


def test_interest_coverage() -> None:
    # EBIT / interest = 200 / 50 = 4.0
    assert rt.interest_coverage(EBIT, INTEREST) == D("4")


def test_interest_coverage_rejects_zero_interest() -> None:
    with pytest.raises(Exception):
        rt.interest_coverage(EBIT, D("0"))


def test_gross_profit_margin() -> None:
    # 400 / 1000 = 0.40
    assert rt.gross_profit_margin(GROSS_PROFIT, REVENUE) == D("0.4")


def test_operating_profit_margin() -> None:
    # 200 / 1000 = 0.20
    assert rt.operating_profit_margin(EBIT, REVENUE) == D("0.2")


def test_net_profit_margin() -> None:
    # 105 / 1000 = 0.105
    assert rt.net_profit_margin(NET_INCOME, REVENUE) == D("0.105")


def test_return_on_assets() -> None:
    # 105 / 2000 = 0.0525
    assert rt.return_on_assets(NET_INCOME, AVG_ASSETS) == D("0.0525")


def test_return_on_equity() -> None:
    # 105 / 800 = 0.13125
    assert rt.return_on_equity(NET_INCOME, AVG_EQUITY) == D("0.13125")


def test_equity_multiplier() -> None:
    # 2000 / 800 = 2.5
    assert rt.equity_multiplier(AVG_ASSETS, AVG_EQUITY) == D("2.5")


def test_inventory_turnover() -> None:
    # COGS / average inventory = 600 / 200 = 3.0
    assert rt.inventory_turnover(COGS, AVG_INVENTORY) == D("3")


def test_receivables_turnover() -> None:
    # revenue / average receivables = 1000 / 125 = 8.0
    assert rt.receivables_turnover(REVENUE, AVG_RECEIVABLES) == D("8")


def test_total_asset_turnover() -> None:
    # revenue / average total assets = 1000 / 2000 = 0.5
    assert rt.total_asset_turnover(REVENUE, AVG_ASSETS) == D("0.5")


def test_dupont_three_step_components() -> None:
    # ROE = net margin x asset turnover x equity multiplier
    #     = 0.105 x 0.5 x 2.5 = 0.13125
    dupont = rt.dupont_three_step(NET_INCOME, REVENUE, AVG_ASSETS, AVG_EQUITY)
    approx(dupont.net_profit_margin, "0.105")
    approx(dupont.asset_turnover, "0.5")
    approx(dupont.equity_multiplier, "2.5")
    approx(dupont.return_on_equity, "0.13125")


def test_dupont_three_step_reconciles_with_direct_roe() -> None:
    # The decomposition is only useful if it reproduces the number it explains.
    dupont = rt.dupont_three_step(NET_INCOME, REVENUE, AVG_ASSETS, AVG_EQUITY)
    approx(dupont.return_on_equity, str(rt.return_on_equity(NET_INCOME, AVG_EQUITY)), places=12)


def test_dupont_five_step_components() -> None:
    # tax burden     = NI / EBT   = 105 / 150 = 0.70
    # interest burden= EBT / EBIT = 150 / 200 = 0.75
    # EBIT margin    = EBIT / rev = 200 / 1000 = 0.20
    # asset turnover = 0.5        equity multiplier = 2.5
    # product = 0.70(0.75)(0.20)(0.5)(2.5) = 0.13125
    dupont = rt.dupont_five_step(NET_INCOME, EBT, EBIT, REVENUE, AVG_ASSETS, AVG_EQUITY)
    approx(dupont.tax_burden, "0.70")
    approx(dupont.interest_burden, "0.75")
    approx(dupont.operating_margin, "0.20")
    approx(dupont.asset_turnover, "0.5")
    approx(dupont.equity_multiplier, "2.5")
    approx(dupont.return_on_equity, "0.13125")


def test_dupont_five_step_agrees_with_three_step() -> None:
    three = rt.dupont_three_step(NET_INCOME, REVENUE, AVG_ASSETS, AVG_EQUITY)
    five = rt.dupont_five_step(NET_INCOME, EBT, EBIT, REVENUE, AVG_ASSETS, AVG_EQUITY)
    approx(five.return_on_equity, str(three.return_on_equity), places=12)


def test_accruals_ratio_is_negative_when_cash_exceeds_earnings() -> None:
    # SPEC §6.4 [CORRECTED]: (NI - CFO) / average total assets
    # = (105 - 180) / 2000 = -0.0375. Cash backs the earnings — good quality.
    approx(rt.accruals_ratio(NET_INCOME, CFO, AVG_ASSETS), "-0.0375")


def test_accruals_ratio_flags_earnings_unbacked_by_cash() -> None:
    # Same reported profit, far less cash: (105 - 30) / 2000 = +0.0375.
    # A positive accruals ratio is the earnings-quality red flag.
    approx(rt.accruals_ratio(NET_INCOME, D("30"), AVG_ASSETS), "0.0375")


def test_accruals_ratio_ranking_is_independent_of_reported_profit() -> None:
    high_quality = rt.accruals_ratio(NET_INCOME, CFO, AVG_ASSETS)
    low_quality = rt.accruals_ratio(NET_INCOME, D("30"), AVG_ASSETS)
    assert high_quality < low_quality


def test_ratios_reject_zero_denominators() -> None:
    calls: list[Callable[[], Decimal]] = [
        lambda: rt.current_ratio(D("500"), D("0")),
        lambda: rt.return_on_equity(NET_INCOME, D("0")),
        lambda: rt.total_asset_turnover(REVENUE, D("0")),
        lambda: rt.net_profit_margin(NET_INCOME, D("0")),
        lambda: rt.inventory_turnover(COGS, D("0")),
    ]
    for call in calls:
        with pytest.raises(rt.RatioError):
            call()


# ===========================================================================
# §6.5 Equity Investments — src/cfa/valuation.py
# ===========================================================================


def test_gordon_growth_value() -> None:
    # V0 = D1 / (r - g) = 2.00 / (0.10 - 0.05) = 2.00 / 0.05 = 40.00
    value = val.gordon_growth_value(D("2.00"), D("0.10"), D("0.05"))
    assert value is not None
    approx(value, "40.00")


def test_gordon_growth_returns_none_when_growth_reaches_the_discount_rate() -> None:
    # SPEC §6.5 guard. At g >= r the geometric series diverges: the model says
    # "infinite value", which is a modelling failure, not a buy signal.
    assert val.gordon_growth_value(D("2.00"), D("0.05"), D("0.05")) is None
    assert val.gordon_growth_value(D("2.00"), D("0.05"), D("0.06")) is None


def test_gordon_growth_composes_with_capm() -> None:
    # SPEC §6.5: "with r from CAPM".
    required = pf.capm_expected_return(D("0.03"), D("1.0"), D("0.10"))  # = 0.10
    value = val.gordon_growth_value(D("2.00"), required, D("0.05"))
    assert value is not None
    approx(value, "40.00")


def test_sustainable_growth_rate() -> None:
    # g = (1 - payout) x ROE = 0.60 x 0.15 = 0.09
    approx(val.sustainable_growth_rate(D("0.40"), D("0.15")), "0.09")


def test_sustainable_growth_is_zero_at_full_payout() -> None:
    # Retaining nothing means funding no new assets, so no internally financed growth.
    approx(val.sustainable_growth_rate(D("1.00"), D("0.15")), "0")


def test_justified_leading_pe() -> None:
    # SPEC §6.5 [CORRECTED]: P/E1 = payout / (r - g) = 0.40 / 0.05 = 8.0
    value = val.justified_leading_pe(D("0.40"), D("0.10"), D("0.05"))
    assert value is not None
    approx(value, "8.0")


def test_justified_trailing_pe() -> None:
    # P/E0 = [payout (1 + g)] / (r - g) = (0.40 x 1.05) / 0.05 = 8.4
    value = val.justified_trailing_pe(D("0.40"), D("0.10"), D("0.05"))
    assert value is not None
    approx(value, "8.4")


def test_trailing_pe_exceeds_leading_pe_by_the_growth_factor() -> None:
    # The relationship that catches a mix-up of the two forms.
    leading = val.justified_leading_pe(D("0.40"), D("0.10"), D("0.05"))
    trailing = val.justified_trailing_pe(D("0.40"), D("0.10"), D("0.05"))
    assert leading is not None and trailing is not None
    approx(trailing, str(leading * D("1.05")), places=12)


def test_justified_pe_guards_on_growth() -> None:
    assert val.justified_leading_pe(D("0.40"), D("0.05"), D("0.05")) is None
    assert val.justified_trailing_pe(D("0.40"), D("0.05"), D("0.06")) is None


def test_enterprise_value() -> None:
    # SPEC §6.5 [CORRECTED]: EV = market cap + total debt - cash
    # = 1000 + 400 - 150 = 1250
    assert val.enterprise_value(D("1000"), D("400"), D("150")) == D("1250")


def test_enterprise_value_subtracts_cash() -> None:
    # Cash is netted off because an acquirer gets it back immediately; it is
    # not part of what the operating business costs.
    with_cash = val.enterprise_value(D("1000"), D("400"), D("150"))
    without_cash = val.enterprise_value(D("1000"), D("400"), D("0"))
    assert with_cash < without_cash


def test_relative_multiple_premium() -> None:
    # 20 / 16 - 1 = 0.25 -> a 25% premium to the sector median
    approx(val.relative_multiple_premium(D("20"), D("16")), "0.25")


def test_relative_multiple_discount_is_negative() -> None:
    approx(val.relative_multiple_premium(D("12"), D("16")), "-0.25")


def test_free_cash_flow_to_equity() -> None:
    # FCFE = CFO - fixed capital investment + net borrowing
    #      = 500 - 300 + 100 = 300
    assert val.free_cash_flow_to_equity(D("500"), D("300"), D("100")) == D("300")


def test_fcfe_value() -> None:
    # V0 = FCFE1 / (r - g) = 300 / (0.10 - 0.05) = 6000
    value = val.fcfe_value(D("300"), D("0.10"), D("0.05"))
    assert value is not None
    approx(value, "6000")


def test_valuation_hierarchy_prefers_ddm_for_dividend_payers() -> None:
    result = val.value_equity(
        required_return=D("0.10"),
        growth_rate=D("0.05"),
        dividend_next=D("2.00"),
        fcfe_next=D("300"),
    )
    assert result.method == "DDM"
    assert result.value is not None
    approx(result.value, "40.00")


def test_valuation_hierarchy_falls_back_to_fcfe_for_non_payers() -> None:
    # SPEC §6.5 [CORRECTED]: most of a large-cap tech universe pays no
    # dividend, so DDM returns None for the majority of names. Without a
    # fallback the model would silently have no opinion on most of the universe.
    result = val.value_equity(
        required_return=D("0.10"),
        growth_rate=D("0.05"),
        dividend_next=None,
        fcfe_next=D("300"),
    )
    assert result.method == "FCFE"
    assert result.value is not None
    approx(result.value, "6000")


def test_valuation_hierarchy_treats_a_zero_dividend_as_no_dividend() -> None:
    result = val.value_equity(
        required_return=D("0.10"),
        growth_rate=D("0.05"),
        dividend_next=D("0"),
        fcfe_next=D("300"),
    )
    assert result.method == "FCFE"


def test_valuation_hierarchy_reports_when_it_cannot_value() -> None:
    result = val.value_equity(
        required_return=D("0.10"),
        growth_rate=D("0.05"),
        dividend_next=None,
        fcfe_next=None,
    )
    assert result.method == "NONE"
    assert result.value is None
    assert result.reason


def test_valuation_hierarchy_reports_when_growth_breaks_the_model() -> None:
    result = val.value_equity(
        required_return=D("0.05"),
        growth_rate=D("0.08"),
        dividend_next=D("2.00"),
        fcfe_next=None,
    )
    assert result.value is None
    assert "growth" in result.reason.lower()


# ===========================================================================
# §6.6 Fixed Income — src/cfa/fixed_income.py
# ===========================================================================

# Reference bond: 3-year, 6% annual coupon, priced at par (YTM = 6%).
#
#   t   CF      PV at 6%        t x PV          t(t+1) x PV
#   1     60     56.60377358     56.60377358      113.20754716
#   2     60     53.39978640    106.79957280      320.39871840
#   3   1060    889.99644002   2669.98932006    10679.95728024
#            -------------   --------------   ---------------
#              1000.00000000   2833.39266644    11113.56354580
#
#   Macaulay  = 2833.39266644 / 1000            = 2.83339267
#   Modified  = 2.83339267 / 1.06               = 2.67301195
#   Convexity = 11113.5635458 / (1000 x 1.1236) = 9.89103200


def test_bond_priced_at_par_when_coupon_equals_yield() -> None:
    # The definitional check: a bond yielding its coupon rate is worth par.
    price = fi.bond_price(D("1000"), D("0.06"), D("0.06"), 3)
    approx(price, "1000", places=8)


def test_zero_coupon_bond_price() -> None:
    # 1000 / 1.10^2 = 1000 / 1.21 = 826.44628099
    approx(fi.bond_price(D("1000"), D("0"), D("0.10"), 2), "826.44628099", places=8)


def test_bond_trades_at_a_discount_when_yield_exceeds_coupon() -> None:
    assert fi.bond_price(D("1000"), D("0.05"), D("0.07"), 3) < D("1000")


def test_bond_trades_at_a_premium_when_yield_is_below_coupon() -> None:
    assert fi.bond_price(D("1000"), D("0.07"), D("0.05"), 3) > D("1000")


def test_semiannual_coupons_are_discounted_per_period() -> None:
    # 6 periods of 30 at 3% per period, par at 6% annual.
    approx(fi.bond_price(D("1000"), D("0.06"), D("0.06"), 3, periods_per_year=2), "1000", places=8)


def test_yield_to_maturity_inverts_bond_price() -> None:
    price = fi.bond_price(D("1000"), D("0.05"), D("0.07"), 5)
    approx(fi.yield_to_maturity(price, D("1000"), D("0.05"), 5), "0.07", places=8)


def test_yield_to_maturity_of_a_par_bond_is_its_coupon() -> None:
    approx(fi.yield_to_maturity(D("1000"), D("1000"), D("0.06"), 3), "0.06", places=8)


def test_current_yield() -> None:
    # annual coupon / price = 60 / 800 = 0.075
    approx(fi.current_yield(D("60"), D("800")), "0.075")


def test_current_yield_equals_coupon_rate_at_par() -> None:
    approx(fi.current_yield(D("60"), D("1000")), "0.06")


def test_macaulay_duration() -> None:
    approx(fi.macaulay_duration(D("1000"), D("0.06"), D("0.06"), 3), "2.83339267", places=6)


def test_macaulay_duration_of_a_zero_coupon_bond_is_its_maturity() -> None:
    # No intermediate cash flows, so the weighted average time to payment is
    # exactly the maturity. The cleanest check on the weighting logic.
    approx(fi.macaulay_duration(D("1000"), D("0"), D("0.08"), 5), "5", places=8)


def test_modified_duration() -> None:
    # Macaulay / (1 + y/m) = 2.83339267 / 1.06 = 2.67301195
    macaulay = fi.macaulay_duration(D("1000"), D("0.06"), D("0.06"), 3)
    approx(fi.modified_duration(macaulay, D("0.06")), "2.67301195", places=6)


def test_modified_duration_is_below_macaulay_duration() -> None:
    macaulay = fi.macaulay_duration(D("1000"), D("0.06"), D("0.06"), 3)
    assert fi.modified_duration(macaulay, D("0.06")) < macaulay


def test_convexity() -> None:
    approx(fi.convexity(D("1000"), D("0.06"), D("0.06"), 3), "9.891032", places=4)


def test_price_change_percent() -> None:
    # -ModDur x dy + 0.5 x convexity x dy^2
    # = -2.67301195(0.01) + 0.5(9.8910320)(0.0001)
    # = -0.0267301195 + 0.0004945516 = -0.0262355679
    change = fi.price_change_percent(D("2.67301195"), D("9.8910320"), D("0.01"))
    approx(change, "-0.0262355679", places=9)


def test_convexity_makes_the_estimate_asymmetric() -> None:
    # The reason the second-order term exists: a yield fall helps more than an
    # equal yield rise hurts, so duration alone understates the gain and
    # overstates the loss.
    up = fi.price_change_percent(D("2.67301195"), D("9.8910320"), D("0.01"))
    down = fi.price_change_percent(D("2.67301195"), D("9.8910320"), D("-0.01"))
    assert down > abs(up)


def test_duration_only_estimate_ignores_convexity() -> None:
    linear = fi.price_change_percent(D("2.67301195"), D("0"), D("0.01"))
    approx(linear, "-0.0267301195", places=9)


def test_portfolio_duration_is_a_weighted_average() -> None:
    # 0.4(3) + 0.6(7) = 1.2 + 4.2 = 5.4
    approx(fi.portfolio_duration([D("0.4"), D("0.6")], [D("3"), D("7")]), "5.4")


# --- Money-market yield conversions (SPEC §6.6 [CORRECTED]) ----------------
#
# 180-day bill, face 100, price 98.


def test_bank_discount_yield() -> None:
    # (D / F)(360 / t) = (2 / 100)(360 / 180) = 0.04
    approx(fi.bank_discount_yield(D("100"), D("98"), 180), "0.04")


def test_holding_period_yield() -> None:
    # (P1 - P0 + D1) / P0 = (100 - 98) / 98 = 0.0204081633
    approx(fi.holding_period_yield(D("98"), D("100")), "0.0204081633", places=9)


def test_money_market_yield() -> None:
    # HPY x (360 / t) = 0.0204081633 x 2 = 0.0408163265
    approx(fi.money_market_yield(D("0.0204081633"), 180), "0.0408163266", places=9)


def test_effective_annual_yield() -> None:
    # (1 + HPY)^(365/t) - 1 = 1.0204081633^(365/180) - 1
    #   ln(1.0204081633)        = 0.0202027077
    #   x 365/180 (= 2.0277778) = 0.0409666017
    #   exp(0.0409666017)       = 1.0418173105
    approx(fi.effective_annual_yield(D("0.0204081633"), 180), "0.04181731", places=8)


def test_the_three_annualizations_disagree_and_rank_predictably() -> None:
    # Same 180-day bill, three conventions. Money-market yield is lowest (360
    # days, no compounding); EAY is highest (365 days, compounded). Quoting the
    # wrong one is a silent error of ~100bp on a 4% instrument.
    hpy = D("0.0204081633")
    mmy = fi.money_market_yield(hpy, 180)
    bey = fi.discount_to_bond_equivalent_yield(D("0.04"), 180)
    eay = fi.effective_annual_yield(hpy, 180)
    assert mmy < bey < eay


def test_discount_to_bond_equivalent_yield() -> None:
    # SPEC §6.2 [CORRECTED]: FRED's DGS3MO is quoted on a discount basis and
    # must be converted before it is used as Rf.
    #   price = 100(1 - 0.04 x 180/360) = 98
    #   BEY   = (2 / 98)(365 / 180) = 0.0413832187
    approx(fi.discount_to_bond_equivalent_yield(D("0.04"), 180), "0.0413832187", places=8)


def test_bond_equivalent_yield_exceeds_the_discount_yield() -> None:
    # This is why the conversion matters. The discount yield divides the gain
    # by *face* and annualizes on 360 days; BEY divides by the *price* actually
    # paid and annualizes on 365. Using the discount yield as Rf understates it,
    # which inflates every excess return and therefore every Sharpe ratio.
    discount = D("0.04")
    bey = fi.discount_to_bond_equivalent_yield(discount, 180)
    assert bey > discount


def test_discount_yield_conversion_rejects_impossible_inputs() -> None:
    with pytest.raises(Exception):
        fi.discount_to_bond_equivalent_yield(D("0.04"), 0)
    with pytest.raises(Exception):
        fi.bank_discount_yield(D("100"), D("98"), 0)


# ===========================================================================
# §6.7 Derivatives — src/cfa/derivatives.py
# ===========================================================================

# Reference option set: S0 = 100, X = 100, r = 5%, T = 1 year.
#   PV(X) = 100 / 1.05 = 95.2380952381


def test_present_value_of_strike() -> None:
    approx(dv.discount(D("100"), D("0.05"), D("1")), "95.2380952381", places=8)


def test_european_put_call_parity_holds_for_a_consistent_quote() -> None:
    # C + PV(X) = P + S0
    # With C = 10:  P = 10 + 95.2380952 - 100 = 5.2380952
    check = dv.european_put_call_parity(
        call_price=D("10"),
        put_price=D("5.2380952381"),
        spot=D("100"),
        strike=D("100"),
        risk_free_rate=D("0.05"),
        years=D("1"),
    )
    assert check.holds
    approx(check.difference, "0", places=8)


def test_european_put_call_parity_detects_an_arbitrage() -> None:
    # C = 10, P = 8:  left = 105.238, right = 108 -> off by -2.762
    check = dv.european_put_call_parity(
        call_price=D("10"),
        put_price=D("8"),
        spot=D("100"),
        strike=D("100"),
        risk_free_rate=D("0.05"),
        years=D("1"),
    )
    assert not check.holds
    approx(check.difference, "-2.7619047619", places=8)


def test_implied_put_price_from_parity() -> None:
    # P = C + PV(X) - S0 = 10 + 95.2380952 - 100
    approx(
        dv.implied_put_price(D("10"), D("100"), D("100"), D("0.05"), D("1")),
        "5.2380952381",
        places=8,
    )


def test_implied_call_price_from_parity() -> None:
    # C = P + S0 - PV(X)
    approx(
        dv.implied_call_price(D("5.2380952381"), D("100"), D("100"), D("0.05"), D("1")),
        "10",
        places=8,
    )


# --- American parity bounds (SPEC §6.7 [CORRECTED]) ------------------------


def test_american_parity_bounds() -> None:
    # S0 - X <= C - P <= S0 - PV(X)
    # lower = 100 - 100 = 0
    # upper = 100 - 95.2380952 = 4.7619048
    bounds = dv.american_put_call_parity_bounds(D("100"), D("100"), D("0.05"), D("1"))
    approx(bounds.lower, "0", places=8)
    approx(bounds.upper, "4.7619047619", places=8)


def test_american_bounds_accept_a_quote_that_strict_parity_rejects() -> None:
    # THE point of the correction. US listed equity options are American, and
    # the early-exercise right breaks strict parity. C = 10, P = 8 gives
    # C - P = 2, comfortably inside [0, 4.76] — a perfectly legitimate quote
    # that the v1 equality check would have flagged as an arbitrage.
    strict = dv.european_put_call_parity(
        call_price=D("10"),
        put_price=D("8"),
        spot=D("100"),
        strike=D("100"),
        risk_free_rate=D("0.05"),
        years=D("1"),
    )
    american = dv.american_parity_breach(
        call_price=D("10"),
        put_price=D("8"),
        spot=D("100"),
        strike=D("100"),
        risk_free_rate=D("0.05"),
        years=D("1"),
    )
    assert not strict.holds, "strict European parity fires on this quote"
    assert american.holds, "American bounds correctly accept it"


def test_american_bounds_still_catch_a_real_breach_above() -> None:
    # C - P = 9 is outside [0, 4.76]: no early-exercise story explains that.
    breach = dv.american_parity_breach(
        call_price=D("10"),
        put_price=D("1"),
        spot=D("100"),
        strike=D("100"),
        risk_free_rate=D("0.05"),
        years=D("1"),
    )
    assert not breach.holds


def test_american_bounds_still_catch_a_real_breach_below() -> None:
    # C - P = -3 is below the lower bound of 0.
    breach = dv.american_parity_breach(
        call_price=D("5"),
        put_price=D("8"),
        spot=D("100"),
        strike=D("100"),
        risk_free_rate=D("0.05"),
        years=D("1"),
    )
    assert not breach.holds


def test_dividends_lower_the_american_parity_bounds() -> None:
    # A dividend paid before expiry transfers value out of the stock, so the
    # call is worth less relative to the put.
    without = dv.american_put_call_parity_bounds(D("100"), D("100"), D("0.05"), D("1"))
    with_dividend = dv.american_put_call_parity_bounds(
        D("100"), D("100"), D("0.05"), D("1"), present_value_dividends=D("2")
    )
    assert with_dividend.lower < without.lower


# --- Forward pricing (SPEC §6.7 [CORRECTED]) -------------------------------


def test_forward_price_with_dividend_carry() -> None:
    # F0 = (S0 - PV(dividends))(1 + r)^T = (100 - 2)(1.05) = 102.90
    approx(dv.forward_price(D("100"), D("0.05"), D("1"), D("2")), "102.90", places=8)


def test_forward_price_without_dividends() -> None:
    # F0 = 100(1.05) = 105.00
    approx(dv.forward_price(D("100"), D("0.05"), D("1")), "105.00", places=8)


def test_ignoring_dividends_overstates_the_forward_price() -> None:
    # SPEC §6.7 [CORRECTED]: the v1 form S0(1+r)^T is wrong for any
    # dividend-paying equity, and wrong in a consistent direction.
    correct = dv.forward_price(D("100"), D("0.05"), D("1"), D("2"))
    naive = dv.forward_price(D("100"), D("0.05"), D("1"))
    assert naive > correct
    approx(naive - correct, "2.10", places=8)


# --- Intrinsic value, time value, moneyness --------------------------------


def test_call_intrinsic_value() -> None:
    assert dv.intrinsic_value(dv.OptionType.CALL, D("110"), D("100")) == D("10")
    assert dv.intrinsic_value(dv.OptionType.CALL, D("90"), D("100")) == D("0")


def test_put_intrinsic_value() -> None:
    assert dv.intrinsic_value(dv.OptionType.PUT, D("90"), D("100")) == D("10")
    assert dv.intrinsic_value(dv.OptionType.PUT, D("110"), D("100")) == D("0")


def test_time_value_is_premium_less_intrinsic() -> None:
    # A call at 12 with S = 110, X = 100: intrinsic 10, time value 2.
    intrinsic = dv.intrinsic_value(dv.OptionType.CALL, D("110"), D("100"))
    assert dv.time_value(D("12"), intrinsic) == D("2")


def test_time_value_cannot_be_negative() -> None:
    with pytest.raises(Exception):
        dv.time_value(D("8"), D("10"))


def test_call_moneyness() -> None:
    assert dv.moneyness(dv.OptionType.CALL, D("110"), D("100")) is dv.Moneyness.IN_THE_MONEY
    assert dv.moneyness(dv.OptionType.CALL, D("100"), D("100")) is dv.Moneyness.AT_THE_MONEY
    assert dv.moneyness(dv.OptionType.CALL, D("90"), D("100")) is dv.Moneyness.OUT_OF_THE_MONEY


def test_put_moneyness_is_the_mirror_of_the_call() -> None:
    assert dv.moneyness(dv.OptionType.PUT, D("90"), D("100")) is dv.Moneyness.IN_THE_MONEY
    assert dv.moneyness(dv.OptionType.PUT, D("100"), D("100")) is dv.Moneyness.AT_THE_MONEY
    assert dv.moneyness(dv.OptionType.PUT, D("110"), D("100")) is dv.Moneyness.OUT_OF_THE_MONEY


# --- The four basic positions ----------------------------------------------


def test_long_call_payoff_and_profit() -> None:
    # payoff = max(ST - X, 0); profit = payoff - premium
    assert dv.option_payoff(dv.Position.LONG_CALL, D("100"), D("115")) == D("15")
    assert dv.option_profit(dv.Position.LONG_CALL, D("100"), D("5"), D("115")) == D("10")
    assert dv.option_payoff(dv.Position.LONG_CALL, D("100"), D("90")) == D("0")
    assert dv.option_profit(dv.Position.LONG_CALL, D("100"), D("5"), D("90")) == D("-5")


def test_short_call_is_the_mirror_of_the_long_call() -> None:
    assert dv.option_payoff(dv.Position.SHORT_CALL, D("100"), D("115")) == D("-15")
    assert dv.option_profit(dv.Position.SHORT_CALL, D("100"), D("5"), D("115")) == D("-10")


def test_long_put_payoff_and_profit() -> None:
    assert dv.option_payoff(dv.Position.LONG_PUT, D("100"), D("85")) == D("15")
    assert dv.option_profit(dv.Position.LONG_PUT, D("100"), D("4"), D("85")) == D("11")


def test_short_put_is_the_mirror_of_the_long_put() -> None:
    assert dv.option_payoff(dv.Position.SHORT_PUT, D("100"), D("85")) == D("-15")
    assert dv.option_profit(dv.Position.SHORT_PUT, D("100"), D("4"), D("85")) == D("-11")


def test_options_are_zero_sum() -> None:
    # Every dollar the long makes, the short loses. Checks all four positions
    # against each other across the whole price range.
    for spot in ("70", "90", "100", "110", "130"):
        long_call = dv.option_profit(dv.Position.LONG_CALL, D("100"), D("5"), D(spot))
        short_call = dv.option_profit(dv.Position.SHORT_CALL, D("100"), D("5"), D(spot))
        assert long_call + short_call == D("0")

        long_put = dv.option_profit(dv.Position.LONG_PUT, D("100"), D("4"), D(spot))
        short_put = dv.option_profit(dv.Position.SHORT_PUT, D("100"), D("4"), D(spot))
        assert long_put + short_put == D("0")


def test_payoff_diagram_covers_the_requested_range() -> None:
    diagram = dv.payoff_diagram(
        dv.Position.LONG_CALL, D("100"), D("5"), [D("80"), D("100"), D("120")]
    )
    assert [point.spot for point in diagram] == [D("80"), D("100"), D("120")]
    assert diagram[0].payoff == D("0")
    assert diagram[2].payoff == D("20")
    assert diagram[2].profit == D("15")


# --- Covered call and protective put ---------------------------------------


def test_covered_call_payoff_is_capped_at_the_strike() -> None:
    # Long stock + short call: payoff = min(ST, X)
    assert dv.covered_call_payoff(D("110"), D("105")) == D("105")
    assert dv.covered_call_payoff(D("90"), D("105")) == D("90")


def test_covered_call_profit_and_breakeven() -> None:
    # Stock at 100, short the 105 call for 3.
    #   breakeven  = 100 - 3 = 97
    #   max profit = 105 - 100 + 3 = 8
    approx(dv.covered_call_breakeven(D("100"), D("3")), "97")
    approx(dv.covered_call_profit(D("110"), D("105"), D("100"), D("3")), "8")
    approx(dv.covered_call_profit(D("97"), D("105"), D("100"), D("3")), "0")
    approx(dv.covered_call_profit(D("90"), D("105"), D("100"), D("3")), "-7")


def test_covered_call_trades_upside_for_income() -> None:
    # Above the strike the position stops participating — the premium is
    # compensation for exactly that.
    at_110 = dv.covered_call_profit(D("110"), D("105"), D("100"), D("3"))
    at_200 = dv.covered_call_profit(D("200"), D("105"), D("100"), D("3"))
    assert at_110 == at_200


def test_protective_put_payoff_has_a_floor() -> None:
    # Long stock + long put: payoff = max(ST, X)
    assert dv.protective_put_payoff(D("90"), D("95")) == D("95")
    assert dv.protective_put_payoff(D("110"), D("95")) == D("110")


def test_protective_put_profit_and_breakeven() -> None:
    # Stock at 100, long the 95 put for 2.
    #   breakeven = 100 + 2 = 102
    #   max loss  = 100 - 95 + 2 = 7
    approx(dv.protective_put_breakeven(D("100"), D("2")), "102")
    approx(dv.protective_put_profit(D("110"), D("95"), D("100"), D("2")), "8")
    approx(dv.protective_put_profit(D("102"), D("95"), D("100"), D("2")), "0")
    approx(dv.protective_put_profit(D("90"), D("95"), D("100"), D("2")), "-7")


def test_protective_put_loss_is_bounded() -> None:
    # However far the stock falls, the loss stops at the max-loss figure. This
    # is the overlay SPEC §6.7 proposes when volatility breaches the IPS ceiling.
    at_90 = dv.protective_put_profit(D("90"), D("95"), D("100"), D("2"))
    at_10 = dv.protective_put_profit(D("10"), D("95"), D("100"), D("2"))
    assert at_90 == at_10 == D("-7")


def test_protective_put_cost_drag() -> None:
    # 2.00 of premium on a 100.00 position = 2% drag before any protection pays.
    approx(dv.protective_put_cost_drag(D("2"), D("100")), "0.02")


# ===========================================================================
# §6.8 Alternative Investments — src/cfa/alternatives.py
# ===========================================================================

# "2 and 20" on a fund that starts at 100 and grows to 120 gross.
#   management fee = 0.02 x 120                    = 2.40
#   value net of management fee = 120 - 2.40       = 117.60
#   gain over the 100 high-water mark              = 17.60
#   incentive fee = 0.20 x 17.60                   = 3.52
#   total fees                                     = 5.92
#   ending value net                               = 114.08


def test_hedge_fund_fees_two_and_twenty() -> None:
    fees = alt.hedge_fund_fees(
        beginning_value=D("100"),
        ending_value=D("120"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
    )
    approx(fees.management_fee, "2.40")
    approx(fees.incentive_fee, "3.52")
    approx(fees.total_fees, "5.92")
    approx(fees.ending_value_net, "114.08")


def test_fees_take_a_large_bite_out_of_the_gross_return() -> None:
    # Gross +20%, investor +14.08%. Nearly six points of a twenty point gain.
    fees = alt.hedge_fund_fees(
        beginning_value=D("100"),
        ending_value=D("120"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
    )
    approx(fees.investor_return, "0.1408")


def test_high_water_mark_rises_after_a_gain() -> None:
    fees = alt.hedge_fund_fees(
        beginning_value=D("100"),
        ending_value=D("120"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
    )
    approx(fees.high_water_mark, "114.08")


def test_no_incentive_fee_below_the_high_water_mark() -> None:
    # Year 2: the fund falls from 114.08 to 100 gross.
    #   management fee = 0.02 x 100 = 2.00; net = 98.00, below the mark.
    fees = alt.hedge_fund_fees(
        beginning_value=D("114.08"),
        ending_value=D("100"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
        high_water_mark=D("114.08"),
    )
    approx(fees.management_fee, "2.00")
    approx(fees.incentive_fee, "0")
    approx(fees.ending_value_net, "98.00")


def test_high_water_mark_does_not_fall_after_a_loss() -> None:
    # The whole point of the mark: losses must be earned back before the
    # manager is paid an incentive fee again.
    fees = alt.hedge_fund_fees(
        beginning_value=D("114.08"),
        ending_value=D("100"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
        high_water_mark=D("114.08"),
    )
    approx(fees.high_water_mark, "114.08")


def test_incentive_fee_only_applies_to_gains_above_the_mark() -> None:
    # Year 3: back to 120 gross against a 114.08 mark.
    #   management fee = 2.40; net = 117.60
    #   gain above mark = 117.60 - 114.08 = 3.52 -> incentive = 0.704
    fees = alt.hedge_fund_fees(
        beginning_value=D("98"),
        ending_value=D("120"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
        high_water_mark=D("114.08"),
    )
    approx(fees.incentive_fee, "0.704")


def test_hurdle_rate_raises_the_incentive_fee_threshold() -> None:
    # 8% hurdle on a 100 start: incentive applies only above 108.
    #   gain = 120 - 108 = 12 -> incentive = 0.20 x 12 = 2.40
    fees = alt.hedge_fund_fees(
        beginning_value=D("100"),
        ending_value=D("120"),
        management_fee_rate=D("0"),
        incentive_fee_rate=D("0.20"),
        hurdle_rate=D("0.08"),
    )
    approx(fees.incentive_fee, "2.40")


def test_management_fee_can_be_charged_on_beginning_assets() -> None:
    # 0.02 x 100 = 2.00 rather than 0.02 x 120 = 2.40. The basis is a real
    # term of the agreement, not a rounding detail.
    fees = alt.hedge_fund_fees(
        beginning_value=D("100"),
        ending_value=D("120"),
        management_fee_rate=D("0.02"),
        incentive_fee_rate=D("0.20"),
        management_fee_basis="BEGINNING",
    )
    approx(fees.management_fee, "2.00")


# --- Category distinctions -------------------------------------------------


def test_appraisal_valued_categories_are_flagged_for_smoothing() -> None:
    assert alt.has_smoothed_valuations(alt.AlternativeCategory.REAL_ESTATE)
    assert alt.has_smoothed_valuations(alt.AlternativeCategory.PRIVATE_EQUITY)


def test_exchange_traded_categories_are_not_smoothed() -> None:
    # Commodity futures mark to a live market every day; there is no appraisal
    # lag to unsmooth.
    assert not alt.has_smoothed_valuations(alt.AlternativeCategory.COMMODITIES)


def test_every_category_has_a_profile() -> None:
    for category in alt.AlternativeCategory:
        profile = alt.CATEGORY_PROFILES[category]
        assert profile.valuation_basis
        assert profile.typical_fee_structure


# --- Smoothed pricing and its effect on the covariance matrix --------------


def test_first_order_autocorrelation() -> None:
    # x = 1..5, mean 3, deviations -2,-1,0,1,2
    #   numerator   = (-1)(-2) + (0)(-1) + (1)(0) + (2)(1) = 4
    #   denominator = 4 + 1 + 0 + 1 + 4                    = 10
    #   rho1 = 0.4
    approx(alt.first_order_autocorrelation([D("1"), D("2"), D("3"), D("4"), D("5")]), "0.4")


def test_unsmooth_returns_applies_the_geltner_filter() -> None:
    # r_true(t) = (r_obs(t) - rho r_obs(t-1)) / (1 - rho), with rho = 0.4:
    #   (2 - 0.4)/0.6 = 2.6666667      (4 - 1.2)/0.6 = 4.6666667
    #   (3 - 0.8)/0.6 = 3.6666667      (5 - 1.6)/0.6 = 5.6666667
    unsmoothed = alt.unsmooth_returns(
        [D("1"), D("2"), D("3"), D("4"), D("5")], autocorrelation=D("0.4")
    )
    assert len(unsmoothed) == 4
    approx(unsmoothed[0], "2.6666666667", places=8)
    approx(unsmoothed[3], "5.6666666667", places=8)


def test_unsmoothing_raises_measured_volatility() -> None:
    # SPEC §6.8: appraisal-based and stale pricing damp reported volatility,
    # which flatters Sharpe ratios and — the part that matters for §6.2 —
    # understates the variances and covariances fed to the optimizer. An
    # optimizer handed smoothed inputs will overweight the illiquid sleeve
    # because it looks less risky than it is.
    smoothed = [
        D("0.010"), D("0.011"), D("0.012"), D("0.013"), D("0.012"),
        D("0.011"), D("0.010"), D("0.011"), D("0.012"), D("0.013"),
    ]
    rho = alt.first_order_autocorrelation(smoothed)
    assert rho > D("0"), "the series must actually be smoothed for the test to mean anything"

    observed_sd = ret.sample_standard_deviation(smoothed)
    unsmoothed_sd = ret.sample_standard_deviation(alt.unsmooth_returns(smoothed, rho))
    assert unsmoothed_sd > observed_sd


def test_unsmooth_rejects_an_autocorrelation_of_one() -> None:
    with pytest.raises(Exception):
        alt.unsmooth_returns([D("1"), D("2"), D("3")], autocorrelation=D("1"))


# ===========================================================================
# Guards — every refusal to return a number
# ===========================================================================
#
# These are as much a part of the specification as the formulas. A model that
# silently returns a plausible-looking number when its assumptions are broken
# is worse than one that raises, because the bad number reaches a portfolio
# weight and nothing ever flags it.


def test_numeric_boundary_rejects_non_finite_values() -> None:
    with pytest.raises(num.NumericError):
        num.to_float(D("Infinity"))
    with pytest.raises(num.NumericError):
        num.to_decimal(float("inf"))
    with pytest.raises(num.NumericError):
        num.to_decimal(float("nan"))
    with pytest.raises(num.NumericError):
        num.to_float_array([D("1"), D("NaN")])


def test_numeric_boundary_rejects_malformed_matrices() -> None:
    with pytest.raises(num.NumericError):
        num.to_float_matrix([])
    with pytest.raises(num.NumericError):
        num.to_float_matrix([[D("1"), D("2")], [D("3")]])  # ragged
    with pytest.raises(num.NumericError):
        num.to_float_matrix([[D("1"), D("2")]])  # not square


def test_return_measures_refuse_undefined_inputs() -> None:
    with pytest.raises(ValueError):
        # A -100% return wipes the position out; no geometric mean exists.
        ret.geometric_mean_return([D("-1"), D("0.5")])
    with pytest.raises(num.NumericError):
        ret.correlation([D("1"), D("1"), D("1")], [D("1"), D("2"), D("3")])
    with pytest.raises(num.NumericError):
        ret.coefficient_of_variation([D("-0.1"), D("0.1")])  # mean is zero
    with pytest.raises(num.NumericError):
        ret.safety_first_ratio(D("0.12"), D("0.03"), D("0"))
    with pytest.raises(num.NumericError):
        # A regressor with no variance cannot identify a slope.
        ret.ols_regression([D("1"), D("2"), D("3")], [D("5"), D("5"), D("5")])


def test_money_weighted_return_reports_an_unbracketable_series() -> None:
    with pytest.raises(ValueError):
        ret.money_weighted_return(
            [(at(2023, 1, 1), D("-100")), (at(2024, 1, 1), D("100000000000"))]
        )


def test_portfolio_measures_refuse_undefined_inputs() -> None:
    with pytest.raises(num.NumericError):
        pf.two_asset_variance(D("0.5"), D("0.5"), D("0.2"), D("0.1"), D("1.5"))  # rho > 1
    with pytest.raises(num.NumericError):
        pf.portfolio_variance([D("0.5")], COV_2)  # weight/matrix mismatch
    with pytest.raises(num.NumericError):
        pf.capital_allocation_line(D("0.03"), D("0.13"), D("0"), D("0.10"))
    with pytest.raises(num.NumericError):
        pf.sharpe_ratio(D("0.13"), D("0.03"), D("0"))
    with pytest.raises(num.NumericError):
        pf.treynor_ratio(D("0.13"), D("0.03"), D("0"))
    with pytest.raises(num.NumericError):
        pf.m_squared(D("0.13"), D("0.03"), D("0"), D("0.15"))
    with pytest.raises(num.NumericError):
        pf.information_ratio(D("0.13"), D("0.10"), D("0"))


def test_portfolio_standard_deviation_rejects_an_invalid_covariance_matrix() -> None:
    # A matrix that yields negative variance is not a covariance matrix.
    broken = [[D("-1"), D("0")], [D("0"), D("-1")]]
    with pytest.raises(num.NumericError):
        pf.portfolio_standard_deviation([D("0.5"), D("0.5")], broken)


def test_optimizers_refuse_a_singular_covariance_matrix() -> None:
    singular = [[D("0.04"), D("0.04")], [D("0.04"), D("0.04")]]
    with pytest.raises(num.NumericError):
        pf.minimum_variance_portfolio(singular)
    with pytest.raises(num.NumericError):
        pf.tangency_portfolio(MU_2, singular, D("0.02"))


def test_frontier_refuses_degenerate_problems() -> None:
    with pytest.raises(num.NumericError):
        pf.efficient_frontier(MU_2, COV_2, points=1)
    with pytest.raises(num.NumericError):
        # Identical expected returns: there is no frontier, only a point.
        pf.efficient_frontier([D("0.05"), D("0.05")], COV_2, points=5)


def test_observation_panel_validation() -> None:
    with pytest.raises(num.NumericError):
        pf.sample_covariance_matrix([[D("0.01")]])  # a single period
    with pytest.raises(num.NumericError):
        pf.ledoit_wolf_covariance([[D("0.01"), D("0.02")], [D("0.01")]])  # ragged
    with pytest.raises(num.NumericError):
        pf.ledoit_wolf_covariance([[], []])  # no assets


def test_fixed_income_refuses_impossible_bonds() -> None:
    with pytest.raises(num.NumericError):
        fi.bond_price(D("1000"), D("0.05"), D("0.05"), 0)  # zero maturity
    with pytest.raises(num.NumericError):
        fi.bond_price(D("1000"), D("0.05"), D("0.05"), 3, periods_per_year=0)
    with pytest.raises(num.NumericError):
        fi.bond_price(D("1000"), D("0.05"), D("-1.5"), 3)  # yield below -100%
    with pytest.raises(num.NumericError):
        fi.yield_to_maturity(D("0"), D("1000"), D("0.05"), 3)
    with pytest.raises(num.NumericError):
        fi.current_yield(D("60"), D("0"))
    with pytest.raises(num.NumericError):
        fi.modified_duration(D("2.8"), D("0.06"), periods_per_year=0)
    with pytest.raises(num.NumericError):
        fi.modified_duration(D("2.8"), D("-2"))  # yield below -100%


def test_money_market_conversions_refuse_impossible_inputs() -> None:
    with pytest.raises(num.NumericError):
        fi.bank_discount_yield(D("0"), D("98"), 180)  # zero face
    with pytest.raises(num.NumericError):
        fi.holding_period_yield(D("0"), D("100"))
    with pytest.raises(num.NumericError):
        fi.money_market_yield(D("0.02"), 0)
    with pytest.raises(num.NumericError):
        fi.effective_annual_yield(D("-2"), 180)  # growth factor below zero
    with pytest.raises(num.NumericError):
        # A 100% discount yield over a full year implies a zero price.
        fi.discount_to_bond_equivalent_yield(D("2.5"), 360)


def test_derivatives_refuse_impossible_inputs() -> None:
    with pytest.raises(dv.DerivativesError):
        dv.compound(D("100"), D("-1.5"), D("1"))
    with pytest.raises(dv.DerivativesError):
        dv.discount(D("100"), D("-1.5"), D("1"))
    with pytest.raises(dv.DerivativesError):
        dv.forward_price(D("100"), D("0.05"), D("1"), D("-2"))  # negative dividends
    with pytest.raises(dv.DerivativesError):
        dv.forward_price(D("100"), D("0.05"), D("1"), D("150"))  # dividends exceed spot
    with pytest.raises(dv.DerivativesError):
        dv.protective_put_cost_drag(D("2"), D("0"))


def test_valuation_refuses_impossible_inputs() -> None:
    with pytest.raises(ZeroDivisionError):
        val.relative_multiple_premium(D("20"), D("0"))
    assert val.fcfe_value(D("300"), D("0.05"), D("0.05")) is None


def test_alternatives_refuse_impossible_inputs() -> None:
    with pytest.raises(num.NumericError):
        alt.hedge_fund_fees(
            beginning_value=D("0"),
            ending_value=D("120"),
            management_fee_rate=D("0.02"),
            incentive_fee_rate=D("0.20"),
        )
    with pytest.raises(num.NumericError):
        alt.hedge_fund_fees(
            beginning_value=D("100"),
            ending_value=D("120"),
            management_fee_rate=D("-0.02"),
            incentive_fee_rate=D("0.20"),
        )
    with pytest.raises(num.NumericError):
        alt.first_order_autocorrelation([D("1"), D("1"), D("1")])  # constant series


def test_capped_frontier_stops_at_the_attainable_maximum() -> None:
    # With a 30% per-name cap, no portfolio can reach the best asset's own
    # return: the remaining 70% has to sit in worse names. Spanning to max(mu)
    # would ask the solver for a target that does not exist.
    #   0.30(0.10) + 0.30(0.05) = 0.045 over two assets, budget exhausted at 0.60
    #   ... so the cap must be loose enough to reach 100% invested.
    mu = [D("0.10"), D("0.08"), D("0.05")]
    cov = [
        [D("0.04"), D("0.00"), D("0.00")],
        [D("0.00"), D("0.02"), D("0.00")],
        [D("0.00"), D("0.00"), D("0.01")],
    ]
    frontier = pf.efficient_frontier(mu, cov, points=6, max_weight=D("0.50"))

    # Attainable maximum: 0.50(0.10) + 0.50(0.08) = 0.09, not 0.10.
    top = max(p.expected_return for p in frontier)
    approx(top, "0.09", places=6)
    for point in frontier:
        assert max(point.weights) <= D("0.50") + D("1e-6")


def test_frontier_rejects_a_cap_that_cannot_fund_the_portfolio() -> None:
    mu = [D("0.10"), D("0.05")]
    cov = [[D("0.04"), D("0.00")], [D("0.00"), D("0.01")]]
    with pytest.raises(Exception, match="cannot reach"):
        pf.efficient_frontier(mu, cov, points=5, max_weight=D("0.25"))
