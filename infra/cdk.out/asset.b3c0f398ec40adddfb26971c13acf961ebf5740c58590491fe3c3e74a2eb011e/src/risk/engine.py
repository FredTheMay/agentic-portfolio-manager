"""The risk and IPS engine: approve, repair, or veto a set of target weights.

A pure function — no I/O, no LLM, no randomness, no clock — which is what makes
it testable at ten thousand cases and what makes the policy binding at runtime.

Repairs run in a fixed order, then *every* constraint is re-checked from
scratch on the result and anything still failing is rejected. That final gate
is load-bearing: repairs interact (capping a sector changes beta; blending for
turnover reinterpolates everything), and verifying the output is more reliable
than reasoning about whether the repair order is exhaustive.

Uniform scaling toward cash is safe because every constraint it targets is
monotone in the scale factor, so it can never undo an earlier repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Mapping, Sequence

from src.cfa._numeric import NumericError
from src.cfa.portfolio import minimum_variance_portfolio
from src.risk.codes import Decision, ReasonCode, Repair, Violation
from src.risk.ips import InvestmentPolicy
from src.time.clock import ensure_utc

ZERO = Decimal(0)
ONE = Decimal(1)

#: Slack for Decimal rounding when comparing against a limit. Tighter than any
#: limit in the IPS by many orders of magnitude.
TOLERANCE = Decimal("1e-12")


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Everything the engine needs to know that is not the proposal itself.

    ``covariance`` is **annualized**, matching ``max_annualized_volatility``.
    ``blocked_until`` maps a symbol to the instant its wash-sale window closes.
    """

    as_of: datetime
    current_weights: Mapping[str, Decimal]
    sectors: Mapping[str, str]
    betas: Mapping[str, Decimal]
    covariance: Mapping[str, Mapping[str, Decimal]]
    expected_returns: Mapping[str, Decimal]
    universe: frozenset[str]
    drawdown: Decimal
    risk_free_rate: Decimal = ZERO
    blocked_until: Mapping[str, datetime] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    """Measured properties of the portfolio the engine is returning."""

    gross_exposure: Decimal
    cash_weight: Decimal
    portfolio_beta: Decimal
    annualized_volatility: Decimal
    safety_first_ratio: Decimal | None
    turnover: Decimal
    max_drift: Decimal


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """The engine's verdict, with everything needed to audit it."""

    decision: Decision
    weights: Mapping[str, Decimal]
    cash_weight: Decimal
    violations: tuple[Violation, ...]
    repairs: tuple[Repair, ...]
    metrics: RiskMetrics
    #: Carried into the RebalanceMandate. Not enforced here: the
    #: decision layer does not size orders.
    min_trade_notional: Decimal

    @property
    def approved(self) -> bool:
        return self.decision in (Decision.APPROVED, Decision.MODIFIED)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def portfolio_beta(weights: Mapping[str, Decimal], betas: Mapping[str, Decimal]) -> Decimal:
    """``beta_p = sum(wi beta_i)``."""
    return sum((w * betas.get(s, ZERO) for s, w in weights.items()), ZERO)


def portfolio_volatility(
    weights: Mapping[str, Decimal],
    covariance: Mapping[str, Mapping[str, Decimal]],
) -> Decimal:
    """``sigma_p = sqrt(w' Sigma w)``, computed in exact decimal arithmetic.

    Done here rather than through :mod:`src.cfa.portfolio` because this runs
    inside a ten-thousand-case property test and the float64 round trip is
    both unnecessary for a quadratic form and a source of tolerance noise.
    """
    variance = ZERO
    for a, wa in weights.items():
        row = covariance.get(a, {})
        for b, wb in weights.items():
            variance += wa * wb * row.get(b, ZERO)
    if variance <= ZERO:
        return ZERO
    return variance.sqrt()


def sector_exposures(
    weights: Mapping[str, Decimal], sectors: Mapping[str, str]
) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {}
    for symbol, weight in weights.items():
        sector = sectors.get(symbol, "UNKNOWN")
        totals[sector] = totals.get(sector, ZERO) + weight
    return totals


def turnover(current: Mapping[str, Decimal], target: Mapping[str, Decimal]) -> Decimal:
    """One-way turnover: half the sum of absolute weight changes."""
    names = set(current) | set(target)
    traded = sum((abs(target.get(s, ZERO) - current.get(s, ZERO)) for s in names), ZERO)
    return traded / Decimal(2)


def max_drift(current: Mapping[str, Decimal], target: Mapping[str, Decimal]) -> Decimal:
    names = set(current) | set(target)
    if not names:
        return ZERO
    return max(abs(target.get(s, ZERO) - current.get(s, ZERO)) for s in names)


def safety_first_ratio(
    weights: Mapping[str, Decimal],
    context: RiskContext,
    policy: InvestmentPolicy,
) -> Decimal | None:
    """Roy's ratio for the portfolio, or ``None`` when it is undefined.

    A riskless portfolio has no shortfall distribution, so the ratio does not
    exist. That is not a failure — an all-cash book cannot breach a
    capital-preservation floor — so the caller treats ``None`` as passing.
    """
    cash = ONE - sum(weights.values(), ZERO)
    expected = sum(
        (w * context.expected_returns.get(s, ZERO) for s, w in weights.items()), ZERO
    )
    expected += cash * context.risk_free_rate

    volatility = portfolio_volatility(weights, context.covariance)
    if volatility <= ZERO:
        return None
    return (expected - policy.safety_first.threshold_return) / volatility


# ---------------------------------------------------------------------------
# verification — the gate every result must pass
# ---------------------------------------------------------------------------


def verify(
    weights: Mapping[str, Decimal],
    context: RiskContext,
    policy: InvestmentPolicy,
) -> list[Violation]:
    """Check every constraint against a finished portfolio.

    Independent of how the portfolio was produced. Nothing leaves this module
    without passing.
    """
    violations: list[Violation] = []

    for symbol, weight in sorted(weights.items()):
        if policy.long_only and weight < -TOLERANCE:
            violations.append(
                Violation(
                    ReasonCode.NO_SHORTING,
                    f"weight {weight} is negative",
                    symbol=symbol,
                    observed=weight,
                    limit=ZERO,
                )
            )
        if weight > policy.max_position_weight + TOLERANCE:
            violations.append(
                Violation(
                    ReasonCode.MAX_POSITION_WEIGHT,
                    f"weight {weight} exceeds the position cap",
                    symbol=symbol,
                    observed=weight,
                    limit=policy.max_position_weight,
                )
            )
        if symbol not in context.universe:
            violations.append(
                Violation(
                    ReasonCode.UNIVERSE_WHITELIST,
                    "not an approved instrument",
                    symbol=symbol,
                )
            )
        if weight > TOLERANCE and _is_blocked(symbol, context):
            violations.append(
                Violation(
                    ReasonCode.WASH_SALE_WINDOW,
                    "repurchase inside the wash-sale window",
                    symbol=symbol,
                    observed=weight,
                )
            )

    for sector, exposure in sorted(sector_exposures(weights, context.sectors).items()):
        if exposure > policy.max_sector_weight + TOLERANCE:
            violations.append(
                Violation(
                    ReasonCode.MAX_SECTOR_WEIGHT,
                    f"{sector} exposure {exposure} exceeds the sector cap",
                    observed=exposure,
                    limit=policy.max_sector_weight,
                )
            )

    gross = sum(weights.values(), ZERO)
    if gross > policy.max_gross_exposure + TOLERANCE:
        violations.append(
            Violation(
                ReasonCode.NO_LEVERAGE,
                f"gross exposure {gross} exceeds the ceiling",
                observed=gross,
                limit=policy.max_gross_exposure,
            )
        )

    cash = ONE - gross
    if cash < policy.min_cash_buffer - TOLERANCE:
        violations.append(
            Violation(
                ReasonCode.MIN_CASH_BUFFER,
                f"cash {cash} is below the liquidity floor",
                observed=cash,
                limit=policy.min_cash_buffer,
            )
        )

    beta = portfolio_beta(weights, context.betas)
    if beta > policy.max_portfolio_beta + TOLERANCE:
        violations.append(
            Violation(
                ReasonCode.MAX_PORTFOLIO_BETA,
                f"portfolio beta {beta} exceeds the ceiling",
                observed=beta,
                limit=policy.max_portfolio_beta,
            )
        )

    volatility = portfolio_volatility(weights, context.covariance)
    if volatility > policy.max_annualized_volatility + TOLERANCE:
        violations.append(
            Violation(
                ReasonCode.MAX_VOLATILITY,
                f"annualized volatility {volatility} exceeds the ceiling",
                observed=volatility,
                limit=policy.max_annualized_volatility,
            )
        )

    ratio = safety_first_ratio(weights, context, policy)
    if ratio is not None and ratio < policy.safety_first.minimum_ratio:
        violations.append(
            Violation(
                ReasonCode.SAFETY_FIRST_THRESHOLD,
                f"safety-first ratio {ratio} is below the floor",
                observed=ratio,
                limit=policy.safety_first.minimum_ratio,
            )
        )

    return violations


def _is_blocked(symbol: str, context: RiskContext) -> bool:
    until = context.blocked_until.get(symbol)
    if until is None:
        return False
    return ensure_utc(context.as_of) < ensure_utc(until)


# ---------------------------------------------------------------------------
# repairs
# ---------------------------------------------------------------------------


def _minimum_variance(context: RiskContext) -> dict[str, Decimal]:
    """Long-only minimum-variance weights over the investable universe.

    Falls back to all cash if the covariance matrix is singular. Refusing to
    hold anything is always a valid answer to "reduce risk"; guessing is not.
    """
    symbols = sorted(s for s in context.universe if s in context.covariance)
    if not symbols:
        return {}
    matrix = [[context.covariance[a].get(b, ZERO) for b in symbols] for a in symbols]
    try:
        raw = minimum_variance_portfolio(matrix)
    except (NumericError, ValueError):
        return {}
    # The closed form is unconstrained and may short; the repair pipeline
    # clamps and rescales it like any other proposal.
    return {symbol: weight for symbol, weight in zip(symbols, raw) if weight > ZERO}


def evaluate(
    proposed: Mapping[str, Decimal],
    context: RiskContext,
    policy: InvestmentPolicy,
) -> RiskAssessment:
    """Approve, repair, or reject a proposed set of target weights.

    Returns weights that are guaranteed to satisfy every constraint in
    , or ``REJECTED`` with the codes that could not be satisfied.
    """
    repairs: list[Repair] = []
    weights: dict[str, Decimal] = {s: w for s, w in proposed.items() if w != ZERO}

    # --- Circuit breaker: replaces the proposal outright ------------------
    breaker_tripped = context.drawdown > policy.max_drawdown
    if breaker_tripped:
        weights = _minimum_variance(context)
        repairs.append(
            Repair(
                ReasonCode.DRAWDOWN_CIRCUIT_BREAKER,
                f"drawdown {context.drawdown} exceeds {policy.max_drawdown}; "
                "forced to the minimum-variance portfolio and blocked risk increases",
            )
        )

    # --- Per-name repairs -------------------------------------------------
    for symbol in sorted(weights):
        if symbol not in context.universe:
            del weights[symbol]
            repairs.append(
                Repair(ReasonCode.UNIVERSE_WHITELIST, "removed: not investable", symbol=symbol)
            )

    for symbol in sorted(weights):
        if _is_blocked(symbol, context):
            del weights[symbol]
            repairs.append(
                Repair(
                    ReasonCode.WASH_SALE_WINDOW,
                    "removed: inside the 30-day wash-sale window",
                    symbol=symbol,
                )
            )

    if policy.long_only:
        for symbol in sorted(weights):
            if weights[symbol] < ZERO:
                repairs.append(
                    Repair(
                        ReasonCode.NO_SHORTING,
                        f"clamped {weights[symbol]} to zero",
                        symbol=symbol,
                    )
                )
                del weights[symbol]

    for symbol in sorted(weights):
        if weights[symbol] > policy.max_position_weight:
            repairs.append(
                Repair(
                    ReasonCode.MAX_POSITION_WEIGHT,
                    f"capped {weights[symbol]} at {policy.max_position_weight}",
                    symbol=symbol,
                )
            )
            weights[symbol] = policy.max_position_weight

    # --- Sector caps ------------------------------------------------------
    for sector, exposure in sorted(sector_exposures(weights, context.sectors).items()):
        if exposure > policy.max_sector_weight and exposure > ZERO:
            scale = policy.max_sector_weight / exposure
            for symbol in sorted(weights):
                if context.sectors.get(symbol, "UNKNOWN") == sector:
                    weights[symbol] *= scale
            repairs.append(
                Repair(
                    ReasonCode.MAX_SECTOR_WEIGHT,
                    f"{sector} scaled from {exposure} to {policy.max_sector_weight}",
                )
            )

    # --- Uniform scale toward cash ---------------------------------------
    # every constraint below is monotone in the scale factor, so one pass at
    # the tightest of them satisfies all of them at once.
    gross = sum(weights.values(), ZERO)
    if gross > ZERO:
        scale = ONE
        reasons: list[ReasonCode] = []

        exposure_ceiling = policy.effective_exposure_ceiling
        if gross > exposure_ceiling:
            scale = min(scale, exposure_ceiling / gross)
            reasons.append(
                ReasonCode.NO_LEVERAGE
                if policy.max_gross_exposure <= ONE - policy.min_cash_buffer
                else ReasonCode.MIN_CASH_BUFFER
            )

        beta = portfolio_beta(weights, context.betas)
        if beta > policy.max_portfolio_beta and beta > ZERO:
            scale = min(scale, policy.max_portfolio_beta / beta)
            reasons.append(ReasonCode.MAX_PORTFOLIO_BETA)

        volatility_ceiling = policy.max_annualized_volatility
        if breaker_tripped and context.current_weights:
            # "Halt risk increases": past the breaker the book may not become
            # riskier than it already is, even if the IPS ceiling would allow it.
            current_volatility = portfolio_volatility(
                context.current_weights, context.covariance
            )
            if current_volatility > ZERO:
                volatility_ceiling = min(volatility_ceiling, current_volatility)

        volatility = portfolio_volatility(weights, context.covariance)
        if volatility > volatility_ceiling and volatility > ZERO:
            scale = min(scale, volatility_ceiling / volatility)
            reasons.append(ReasonCode.MAX_VOLATILITY)

        if scale < ONE:
            for symbol in weights:
                weights[symbol] *= scale
            for code in dict.fromkeys(reasons):
                repairs.append(Repair(code, f"scaled all weights by {scale}"))

    # --- Rebalance corridor ----------------------------------------------
    # evaluated on the repaired target: whether a trade is worth doing depends
    # on the portfolio actually being proposed, not the raw request.
    drift = max_drift(context.current_weights, weights)
    if context.current_weights and drift <= policy.corridor_absolute:
        metrics = _metrics(weights, context, policy)
        return RiskAssessment(
            decision=Decision.REJECTED,
            weights=dict(context.current_weights),
            cash_weight=ONE - sum(context.current_weights.values(), ZERO),
            violations=(
                Violation(
                    ReasonCode.REBALANCE_CORRIDOR,
                    f"maximum drift {drift} is inside the {policy.corridor_absolute} corridor; "
                    "trading would pay spread and commission to correct noise",
                    observed=drift,
                    limit=policy.corridor_absolute,
                ),
            ),
            repairs=tuple(repairs),
            metrics=metrics,
            min_trade_notional=policy.min_trade_notional,
        )

    # --- Turnover ---------------------------------------------------------
    traded = turnover(context.current_weights, weights)
    if traded > policy.max_turnover and traded > ZERO:
        blend = policy.max_turnover / traded
        names = set(context.current_weights) | set(weights)
        blended: dict[str, Decimal] = {}
        for symbol in sorted(names):
            start = context.current_weights.get(symbol, ZERO)
            end = weights.get(symbol, ZERO)
            value = start + blend * (end - start)
            if value != ZERO:
                blended[symbol] = value
        weights = blended
        repairs.append(
            Repair(
                ReasonCode.MAX_TURNOVER,
                f"turnover {traded} exceeded {policy.max_turnover}; "
                f"moved {blend} of the way toward the target",
            )
        )

    # --- Final verification ----------------------------------------------
    violations = verify(weights, context, policy)
    metrics = _metrics(weights, context, policy)

    if violations:
        return RiskAssessment(
            decision=Decision.REJECTED,
            weights=dict(context.current_weights),
            cash_weight=ONE - sum(context.current_weights.values(), ZERO),
            violations=tuple(violations),
            repairs=tuple(repairs),
            metrics=metrics,
            min_trade_notional=policy.min_trade_notional,
        )

    return RiskAssessment(
        decision=Decision.MODIFIED if repairs else Decision.APPROVED,
        weights=weights,
        cash_weight=ONE - sum(weights.values(), ZERO),
        violations=(),
        repairs=tuple(repairs),
        metrics=metrics,
        min_trade_notional=policy.min_trade_notional,
    )


def _metrics(
    weights: Mapping[str, Decimal],
    context: RiskContext,
    policy: InvestmentPolicy,
) -> RiskMetrics:
    gross = sum(weights.values(), ZERO)
    return RiskMetrics(
        gross_exposure=gross,
        cash_weight=ONE - gross,
        portfolio_beta=portfolio_beta(weights, context.betas),
        annualized_volatility=portfolio_volatility(weights, context.covariance),
        safety_first_ratio=safety_first_ratio(weights, context, policy),
        turnover=turnover(context.current_weights, weights),
        max_drift=max_drift(context.current_weights, weights),
    )


def rejection_summary(assessments: Sequence[RiskAssessment]) -> dict[ReasonCode, int]:
    """Count vetoes by reason code, for the dashboard panel."""
    counts: dict[ReasonCode, int] = {}
    for assessment in assessments:
        for violation in assessment.violations:
            counts[violation.code] = counts.get(violation.code, 0) + 1
    return counts
