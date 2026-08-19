"""Portfolio construction: expected returns, shrunk covariance, target weights.

Mean-variance optimization treats estimation error as signal, so small changes
in the sample produce wildly different portfolios. Three defenses are applied:
Ledoit-Wolf shrinkage on the covariance, a long-only frontier with a per-name
cap, and CAPM expected returns rather than sample means — sample means being
the noisiest input MVO can be given.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Mapping, Sequence

from src.cfa._numeric import NumericError
from src.cfa.portfolio import (
    FrontierPoint,
    capm_expected_return,
    efficient_frontier,
    expected_portfolio_return,
    ledoit_wolf_covariance,
    portfolio_standard_deviation,
    sharpe_ratio,
)

ZERO = Decimal(0)
ONE = Decimal(1)

OptimizationMethod = Literal["MAX_SHARPE", "MINIMUM_VARIANCE"]


class OptimizerError(ValueError):
    """Raised when no portfolio can be constructed from the given inputs."""


@dataclass(frozen=True, slots=True)
class MarketInputs:
    """Estimated inputs to the optimizer, in a fixed symbol order.

    ``covariance`` is annualized and, by default, shrunk. Keeping the symbol
    order explicit rather than relying on dict iteration keeps the matrix and
    the vector aligned no matter how the caller built them.
    """

    symbols: tuple[str, ...]
    expected_returns: tuple[Decimal, ...]
    covariance: tuple[tuple[Decimal, ...], ...]
    risk_free_rate: Decimal
    shrinkage: Decimal | None = None

    def __post_init__(self) -> None:
        n = len(self.symbols)
        if n == 0:
            raise OptimizerError("no symbols to optimize over")
        if len(self.expected_returns) != n:
            raise OptimizerError("expected_returns does not match the symbol count")
        if len(self.covariance) != n or any(len(row) != n for row in self.covariance):
            raise OptimizerError("covariance is not square in the symbol count")


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """The decision layer's output: what to hold, and why it was chosen."""

    weights: Mapping[str, Decimal]
    expected_return: Decimal
    volatility: Decimal
    sharpe: Decimal
    method: OptimizationMethod
    #: The frontier the choice was made from, for the dashboard.
    frontier: tuple[FrontierPoint, ...] = ()


def capm_expected_returns(
    betas: Mapping[str, Decimal],
    market_return: Decimal,
    risk_free_rate: Decimal,
) -> dict[str, Decimal]:
    """Expected returns from CAPM rather than from sample means.

    Sample mean returns are the noisiest input MVO can be given: estimating a
    mean to useful precision needs decades of data, while estimating a
    covariance needs far less. CAPM collapses the problem to one market premium
    and a set of betas, which are comparatively stable.
    """
    return {
        symbol: capm_expected_return(risk_free_rate, beta, market_return)
        for symbol, beta in betas.items()
    }


def apply_view_tilts(
    expected_returns: Mapping[str, Decimal],
    tilts: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    """Add the aggregator's numeric tilts to the baseline expected returns.

    The tilts arrive already numeric, from a fixed table in
    ``config/view_mapping.yaml``. This function never sees a
    categorical view and never consults a model: an LLM's ``BULLISH`` becomes a
    number by auditable configuration, not by judgment.
    """
    tilted = dict(expected_returns)
    for symbol, tilt in tilts.items():
        if symbol in tilted:
            tilted[symbol] = tilted[symbol] + tilt
    return tilted


def estimate_inputs(
    symbols: Sequence[str],
    observations: Sequence[Sequence[Decimal]],
    betas: Mapping[str, Decimal],
    market_return: Decimal,
    risk_free_rate: Decimal,
    periods_per_year: int = 252,
    tilts: Mapping[str, Decimal] | None = None,
) -> MarketInputs:
    """Build optimizer inputs from a panel of periodic returns.

    ``observations`` is shaped ``(n_periods, n_assets)`` in ``symbols`` order.
    The covariance is shrunk and then annualized; expected returns come from
    CAPM plus any view tilts.
    """
    if not symbols:
        raise OptimizerError("no symbols to optimize over")
    if periods_per_year <= 0:
        raise OptimizerError("periods_per_year must be positive")

    shrunk = ledoit_wolf_covariance(observations)
    scale = Decimal(periods_per_year)
    annualized = tuple(tuple(value * scale for value in row) for row in shrunk.covariance)

    baseline = capm_expected_returns(betas, market_return, risk_free_rate)
    if tilts:
        baseline = apply_view_tilts(baseline, tilts)

    missing = [s for s in symbols if s not in baseline]
    if missing:
        raise OptimizerError(f"no expected return for {missing}")

    return MarketInputs(
        symbols=tuple(symbols),
        expected_returns=tuple(baseline[s] for s in symbols),
        covariance=annualized,
        risk_free_rate=risk_free_rate,
        shrinkage=shrunk.shrinkage,
    )


def optimize(
    inputs: MarketInputs,
    max_position_weight: Decimal | None = None,
    frontier_points: int = 25,
) -> TargetPortfolio:
    """Maximum-Sharpe portfolio on the long-only constrained frontier.

    The unconstrained tangency portfolio has a closed form, but it routinely
    wants short positions and 60% in one name, neither of which this mandate
    permits. So the frontier is solved subject to the real constraints and the
    best Sharpe ratio *on that frontier* is selected. It is the constrained
    tangency portfolio, and unlike the closed form it is always investable.

    Falls back to the minimum-variance point if no frontier portfolio has a
    positive Sharpe ratio — when nothing is expected to beat cash, the honest
    answer is the lowest-risk portfolio, not the least-bad gamble.
    """
    if frontier_points < 2:
        raise OptimizerError("frontier_points must be at least 2")

    expected = list(inputs.expected_returns)
    covariance = [list(row) for row in inputs.covariance]

    try:
        frontier = efficient_frontier(
            expected,
            covariance,
            points=frontier_points,
            long_only=True,
            max_weight=max_position_weight,
        )
    except NumericError as exc:
        raise OptimizerError(f"could not build a frontier: {exc}") from exc

    if not frontier:
        raise OptimizerError("frontier came back empty")

    best: FrontierPoint | None = None
    best_sharpe = Decimal("-Infinity")
    for point in frontier:
        if point.standard_deviation <= ZERO:
            continue
        ratio = sharpe_ratio(point.expected_return, inputs.risk_free_rate, point.standard_deviation)
        if ratio > best_sharpe:
            best_sharpe, best = ratio, point

    method: OptimizationMethod = "MAX_SHARPE"
    if best is None or best_sharpe <= ZERO:
        # Nothing on the frontier is expected to beat cash.
        best = min(frontier, key=lambda p: p.standard_deviation)
        method = "MINIMUM_VARIANCE"
        best_sharpe = (
            sharpe_ratio(best.expected_return, inputs.risk_free_rate, best.standard_deviation)
            if best.standard_deviation > ZERO
            else ZERO
        )

    weights = {
        symbol: weight
        for symbol, weight in zip(inputs.symbols, best.weights)
        # Drop dust: SLSQP leaves 1e-17 residuals that are not real positions.
        if weight > Decimal("1e-9")
    }

    return TargetPortfolio(
        weights=weights,
        expected_return=expected_portfolio_return(
            [weights.get(s, ZERO) for s in inputs.symbols], expected
        ),
        volatility=portfolio_standard_deviation(
            [weights.get(s, ZERO) for s in inputs.symbols], covariance
        ),
        sharpe=best_sharpe,
        method=method,
        frontier=tuple(frontier),
    )
