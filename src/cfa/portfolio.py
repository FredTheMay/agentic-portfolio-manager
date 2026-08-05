"""Portfolio construction and risk-adjusted performance.

CFA Level I topic area: Portfolio Management (SPEC §6.2).

Pure functions, zero I/O. ``Decimal`` at every public boundary; matrix algebra
and constrained optimization run in float64 through :mod:`src.cfa._numeric`.

**On the risk-free rate.** Every ratio here that takes ``risk_free_rate``
expects a *bond-equivalent yield*. FRED's ``DGS3MO`` is quoted on a discount
basis and must be converted first — see
:func:`src.cfa.fixed_income.discount_to_bond_equivalent_yield`. Feeding the
discount yield straight in makes Sharpe, Treynor, CAPM, and the CAL all wrong
in the same direction, which is the kind of error that looks like alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf

from src.cfa._numeric import (
    NumericError,
    require_min_length,
    require_same_length,
    to_decimal,
    to_decimal_list,
    to_float,
    to_float_array,
    to_float_matrix,
)

ZERO = Decimal(0)
ONE = Decimal(1)


def expected_portfolio_return(
    weights: Sequence[Decimal],
    expected_returns: Sequence[Decimal],
) -> Decimal:
    """``E(Rp) = sum(wi E(Ri))`` — the weighted average of component returns.

    CFA Level I: Portfolio Management — portfolio expected return.
    """
    require_same_length("weights", weights, "expected_returns", expected_returns)
    require_min_length("weights", weights, 1)
    return sum((w * r for w, r in zip(weights, expected_returns)), ZERO)


def two_asset_variance(
    weight_a: Decimal,
    weight_b: Decimal,
    sd_a: Decimal,
    sd_b: Decimal,
    correlation: Decimal,
) -> Decimal:
    """``w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 s1 s2 rho``.

    Kept as its own closed form because it is the check on the general
    quadratic form: if ``w'Sigma w`` disagrees with this, the matrix code is
    wrong.

    CFA Level I: Portfolio Management — portfolio variance, two assets.
    """
    if not (-ONE <= correlation <= ONE):
        raise NumericError(f"correlation must lie in [-1, 1], got {correlation}")
    return (
        weight_a**2 * sd_a**2
        + weight_b**2 * sd_b**2
        + 2 * weight_a * weight_b * sd_a * sd_b * correlation
    )


def portfolio_variance(
    weights: Sequence[Decimal],
    covariance_matrix: Sequence[Sequence[Decimal]],
) -> Decimal:
    """``sigma_p^2 = w' Sigma w``.

    CFA Level I: Portfolio Management — portfolio variance.
    """
    w = to_float_array(weights)
    sigma = to_float_matrix(covariance_matrix)
    if len(w) != sigma.shape[0]:
        raise NumericError(f"{len(w)} weights against a {sigma.shape[0]}x{sigma.shape[1]} matrix")
    return to_decimal(float(w @ sigma @ w))


def portfolio_standard_deviation(
    weights: Sequence[Decimal],
    covariance_matrix: Sequence[Sequence[Decimal]],
) -> Decimal:
    """Square root of :func:`portfolio_variance`.

    CFA Level I: Portfolio Management — portfolio standard deviation.
    """
    variance = portfolio_variance(weights, covariance_matrix)
    if variance < ZERO:
        raise NumericError(f"negative portfolio variance {variance}; covariance matrix is invalid")
    return variance.sqrt()


def minimum_variance_portfolio(
    covariance_matrix: Sequence[Sequence[Decimal]],
) -> list[Decimal]:
    """Global minimum-variance portfolio: ``min w'Sigma w`` s.t. ``sum(w) = 1``.

    Closed form ``w = Sigma^-1 1 / (1' Sigma^-1 1)``. Unconstrained beyond the
    budget constraint, so weights may be negative; the IPS layer (SPEC §7) is
    what forbids shorting, not this function.

    CFA Level I: Portfolio Management — minimum-variance portfolio.
    """
    sigma = to_float_matrix(covariance_matrix)
    ones = np.ones(sigma.shape[0], dtype=np.float64)
    try:
        inverse_times_ones = np.linalg.solve(sigma, ones)
    except np.linalg.LinAlgError as exc:
        raise NumericError("covariance matrix is singular") from exc

    denominator = float(ones @ inverse_times_ones)
    if denominator == 0.0:
        raise NumericError("degenerate covariance matrix: weights do not normalize")
    return to_decimal_list(inverse_times_ones / denominator)


def tangency_portfolio(
    expected_returns: Sequence[Decimal],
    covariance_matrix: Sequence[Sequence[Decimal]],
    risk_free_rate: Decimal,
) -> list[Decimal]:
    """Maximum-Sharpe portfolio: ``max (E(Rp) - Rf) / sigma_p``.

    Closed form ``w proportional to Sigma^-1 (mu - Rf 1)``, normalized to sum
    to one. The point where the CAL is tangent to the efficient frontier.

    CFA Level I: Portfolio Management — optimal risky portfolio.
    """
    require_same_length("expected_returns", expected_returns, "covariance_matrix", covariance_matrix)
    sigma = to_float_matrix(covariance_matrix)
    mu = to_float_array(expected_returns)
    excess = mu - to_float(risk_free_rate)

    try:
        raw = np.linalg.solve(sigma, excess)
    except np.linalg.LinAlgError as exc:
        raise NumericError("covariance matrix is singular") from exc

    total = float(np.sum(raw))
    if total == 0.0:
        raise NumericError("no tangency portfolio: excess returns normalize to zero")
    return to_decimal_list(raw / total)


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One portfolio on the efficient frontier."""

    expected_return: Decimal
    standard_deviation: Decimal
    weights: list[Decimal]


def efficient_frontier(
    expected_returns: Sequence[Decimal],
    covariance_matrix: Sequence[Sequence[Decimal]],
    points: int = 20,
    long_only: bool = True,
) -> list[FrontierPoint]:
    """Minimum-variance portfolio at each of ``points`` target returns.

    The frontier runs from the minimum-variance portfolio up to the highest
    attainable return. It deliberately excludes the branch *below* the
    minimum-variance point: those portfolios exist, but every one of them is
    dominated by a portfolio on the upper branch with the same risk and a
    higher return, so they are not efficient and have no business being
    plotted as though they were.

    Long-only by default, because that is the mandate this system operates
    under (SPEC §7 ``NO_SHORTING``) and an unconstrained frontier would be
    advertising portfolios the risk engine will always veto. Weight constraints
    are also half the defense against mean-variance instability — the other
    half is shrinking the covariance estimate (:func:`ledoit_wolf_covariance`).

    CFA Level I: Portfolio Management — efficient frontier.
    """
    if points < 2:
        raise NumericError(f"frontier needs at least 2 points, got {points}")
    require_same_length("expected_returns", expected_returns, "covariance_matrix", covariance_matrix)

    sigma = to_float_matrix(covariance_matrix)
    mu = to_float_array(expected_returns)
    n = len(mu)

    if float(np.min(mu)) == float(np.max(mu)):
        raise NumericError("all expected returns are identical; the frontier is a single point")

    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    start = np.full(n, 1.0 / n, dtype=np.float64)
    budget = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

    # The frontier starts at the minimum-variance portfolio computed under the
    # *same* bounds — the unconstrained closed form would be infeasible here
    # whenever it wants a short position.
    anchor = minimize(
        lambda w: float(w @ sigma @ w),
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=[budget],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not anchor.success:
        raise NumericError(f"minimum-variance solve failed: {anchor.message}")

    lowest = float(anchor.x @ mu)
    highest = float(np.max(mu))
    if highest <= lowest:
        raise NumericError("no efficient frontier above the minimum-variance portfolio")

    frontier: list[FrontierPoint] = []
    for target in np.linspace(lowest, highest, points):
        constraints = [
            budget,
            {"type": "eq", "fun": (lambda w, t=float(target): float(w @ mu - t))},
        ]
        solution = minimize(
            lambda w: float(w @ sigma @ w),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not solution.success:
            raise NumericError(f"frontier solve failed at target {target}: {solution.message}")

        weights: NDArray[np.float64] = solution.x
        # Clamp before the square root: SLSQP can land a hair below zero on a
        # near-degenerate problem, and sqrt of -1e-18 is a crash, not a result.
        variance = max(float(weights @ sigma @ weights), 0.0)
        frontier.append(
            FrontierPoint(
                expected_return=to_decimal(float(weights @ mu)),
                standard_deviation=to_decimal(float(np.sqrt(variance))),
                weights=to_decimal_list(weights),
            )
        )
    return frontier


def capital_allocation_line(
    risk_free_rate: Decimal,
    risky_return: Decimal,
    risky_sd: Decimal,
    portfolio_sd: Decimal,
) -> Decimal:
    """``E(Rp) = Rf + [(E(Ri) - Rf) / sigma_i] sigma_p``.

    CFA Level I: Portfolio Management — capital allocation line.
    """
    if risky_sd <= ZERO:
        raise NumericError("the risky asset must have positive standard deviation")
    return risk_free_rate + ((risky_return - risk_free_rate) / risky_sd) * portfolio_sd


def capital_market_line(
    risk_free_rate: Decimal,
    market_return: Decimal,
    market_sd: Decimal,
    portfolio_sd: Decimal,
) -> Decimal:
    """The CAL drawn from the market portfolio; its slope is the market Sharpe ratio.

    CFA Level I: Portfolio Management — capital market line.
    """
    return capital_allocation_line(risk_free_rate, market_return, market_sd, portfolio_sd)


def capm_expected_return(
    risk_free_rate: Decimal,
    beta: Decimal,
    market_return: Decimal,
) -> Decimal:
    """``E(Ri) = Rf + beta_i (E(Rm) - Rf)``.

    CFA Level I: Portfolio Management — CAPM.
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)


def security_market_line(
    risk_free_rate: Decimal,
    beta: Decimal,
    market_return: Decimal,
) -> Decimal:
    """Required return for a given beta — CAPM plotted against beta.

    Identical arithmetic to :func:`capm_expected_return`, named separately
    because the SML is a distinct concept: it prices *systematic* risk only,
    and an asset plotting above it is mispriced rather than merely volatile.

    CFA Level I: Portfolio Management — security market line.
    """
    return capm_expected_return(risk_free_rate, beta, market_return)


def portfolio_beta(weights: Sequence[Decimal], betas: Sequence[Decimal]) -> Decimal:
    """``beta_p = sum(wi beta_i)`` — beta is linear in the weights.

    CFA Level I: Portfolio Management — portfolio beta.
    """
    require_same_length("weights", weights, "betas", betas)
    require_min_length("weights", weights, 1)
    return sum((w * b for w, b in zip(weights, betas)), ZERO)


def jensens_alpha(
    portfolio_return: Decimal,
    risk_free_rate: Decimal,
    beta: Decimal,
    market_return: Decimal,
) -> Decimal:
    """``alpha = Rp - [Rf + beta(Rm - Rf)]`` — return beyond CAPM's requirement.

    A point estimate. Whether it is distinguishable from zero is a separate
    question, answered by the t-statistic on the regression intercept in
    :func:`src.cfa.returns.estimate_beta`.

    CFA Level I: Portfolio Management — Jensen's alpha.
    """
    return portfolio_return - capm_expected_return(risk_free_rate, beta, market_return)


def sharpe_ratio(
    portfolio_return: Decimal,
    risk_free_rate: Decimal,
    portfolio_sd: Decimal,
) -> Decimal:
    """``(Rp - Rf) / sigma_p`` — excess return per unit of *total* risk.

    CFA Level I: Portfolio Management — Sharpe ratio.
    """
    if portfolio_sd <= ZERO:
        raise NumericError("Sharpe ratio requires a positive standard deviation")
    return (portfolio_return - risk_free_rate) / portfolio_sd


def treynor_ratio(
    portfolio_return: Decimal,
    risk_free_rate: Decimal,
    beta: Decimal,
) -> Decimal:
    """``(Rp - Rf) / beta_p`` — excess return per unit of *systematic* risk.

    The right measure when the portfolio is one sleeve of a diversified whole,
    since idiosyncratic risk is diversified away at the total-portfolio level.

    CFA Level I: Portfolio Management — Treynor ratio.
    """
    if beta == ZERO:
        raise NumericError("Treynor ratio is undefined at zero beta")
    return (portfolio_return - risk_free_rate) / beta


def m_squared(
    portfolio_return: Decimal,
    risk_free_rate: Decimal,
    portfolio_sd: Decimal,
    market_sd: Decimal,
) -> Decimal:
    """``M^2 = (Rp - Rf)(sigma_m / sigma_p) + Rf``.

    Sharpe expressed in percentage points: the return the portfolio would have
    earned levered to the market's volatility, directly comparable to ``E(Rm)``.

    CFA Level I: Portfolio Management — M-squared.
    """
    if portfolio_sd <= ZERO:
        raise NumericError("M-squared requires a positive portfolio standard deviation")
    return (portfolio_return - risk_free_rate) * (market_sd / portfolio_sd) + risk_free_rate


def information_ratio(
    portfolio_return: Decimal,
    benchmark_return: Decimal,
    tracking_error: Decimal,
) -> Decimal:
    """``(Rp - Rb) / tracking error`` — active return per unit of active risk.

    CFA Level I: Portfolio Management — information ratio.
    """
    if tracking_error <= ZERO:
        raise NumericError("information ratio requires positive tracking error")
    return (portfolio_return - benchmark_return) / tracking_error


@dataclass(frozen=True, slots=True)
class RiskDecomposition:
    """Total variance split into its systematic and diversifiable parts."""

    total_variance: Decimal
    systematic_variance: Decimal
    unsystematic_variance: Decimal


def risk_decomposition(
    beta: Decimal,
    market_sd: Decimal,
    portfolio_sd: Decimal,
) -> RiskDecomposition:
    """Split ``sigma_p^2`` into ``beta_p^2 sigma_m^2`` plus residual variance.

    The residual is the part diversification can remove and which, under CAPM,
    therefore earns no expected return.

    CFA Level I: Portfolio Management — systematic and unsystematic risk.
    """
    total = portfolio_sd**2
    systematic = beta**2 * market_sd**2
    unsystematic = total - systematic
    if unsystematic < ZERO:
        raise NumericError(
            f"systematic variance {systematic} exceeds total {total}; inputs are inconsistent"
        )
    return RiskDecomposition(
        total_variance=total,
        systematic_variance=systematic,
        unsystematic_variance=unsystematic,
    )


def sample_covariance_matrix(
    observations: Sequence[Sequence[Decimal]],
) -> list[list[Decimal]]:
    """Sample covariance matrix from ``observations[t][i]`` (rows are periods).

    Provided mainly as the comparison point for :func:`ledoit_wolf_covariance`.
    Prefer the shrunk estimate for anything that feeds an optimizer.

    CFA Level I: Quantitative Methods — sample covariance.
    """
    matrix = _observation_matrix(observations)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    return [to_decimal_list(row) for row in np.atleast_2d(covariance)]


@dataclass(frozen=True, slots=True)
class ShrunkCovariance:
    """A Ledoit-Wolf estimate and the intensity used to produce it."""

    covariance: list[list[Decimal]]
    #: 0 = pure sample estimate, 1 = pure diagonal target.
    shrinkage: Decimal


def ledoit_wolf_covariance(
    observations: Sequence[Sequence[Decimal]],
) -> ShrunkCovariance:
    """Ledoit-Wolf shrinkage estimate of the covariance matrix.

    Mean-variance optimization is notoriously sensitive to its inputs: it
    treats estimation error as signal and concentrates into whichever assets
    happen to look best, so small changes in the sample produce wildly
    different weights. The sample covariance matrix is the worst offender,
    especially when observations are scarce relative to assets, where it is
    ill-conditioned or outright singular.

    Shrinkage pulls the sample matrix toward a structured diagonal target by an
    analytically optimal amount, trading a little bias for a large variance
    reduction. Together with weight constraints (:func:`efficient_frontier`),
    that is this system's defense against MVO instability.

    CFA Level I: Portfolio Management — covariance estimation (SPEC §6.2).
    """
    matrix = _observation_matrix(observations)
    covariance, intensity = ledoit_wolf(matrix, assume_centered=False)
    return ShrunkCovariance(
        covariance=[to_decimal_list(row) for row in np.atleast_2d(covariance)],
        shrinkage=to_decimal(float(intensity)),
    )


def _observation_matrix(
    observations: Sequence[Sequence[Decimal]],
) -> NDArray[np.float64]:
    """Validate and convert an ``(n_periods, n_assets)`` observation panel."""
    require_min_length("observations", observations, 2)
    width = len(observations[0])
    if width == 0:
        raise NumericError("observations must contain at least one asset")
    if any(len(row) != width for row in observations):
        raise NumericError("ragged observation panel: periods differ in asset count")
    return np.array(
        [[to_float(value) for value in row] for row in observations],
        dtype=np.float64,
    )
