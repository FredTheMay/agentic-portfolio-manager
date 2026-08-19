"""The risk engine's central property, over 10,000 generated cases.

    For arbitrary portfolios and arbitrary proposed weights, no approved
    output ever violates any constraint.

Written before the engine, deliberately. Stating the property first forces it
to be about *outputs* rather than about the path the code happens to take, and
an engine written first tends to get a test that follows its own branches
around and proves only that it is self-consistent.

For the same reason every constraint below is re-derived here from the table rather than calling the engine's own verifier. A property test that
shares the implementation's checking code proves nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.risk.codes import Decision, ReasonCode
from src.risk.engine import RiskContext, evaluate
from src.risk.ips import InvestmentPolicy, RiskLevel, SafetyFirstPolicy
from src.time.clock import UTC

D = Decimal

# Tolerance for the Decimal arithmetic the engine does when rescaling.
EPS = D("1e-9")

AS_OF = datetime(2024, 6, 3, tzinfo=UTC)

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")

SECTORS = {
    "AAA": "TECH",
    "BBB": "TECH",
    "CCC": "TECH",
    "DDD": "HEALTH",
    "EEE": "HEALTH",
    "FFF": "ENERGY",
}

BETAS = {
    "AAA": D("1.4"),
    "BBB": D("1.1"),
    "CCC": D("1.6"),
    "DDD": D("0.8"),
    "EEE": D("0.7"),
    "FFF": D("1.0"),
}

EXPECTED_RETURNS = {
    "AAA": D("0.12"),
    "BBB": D("0.10"),
    "CCC": D("0.14"),
    "DDD": D("0.07"),
    "EEE": D("0.06"),
    "FFF": D("0.09"),
}

# Annualized covariance. Diagonal-dominant so it is positive definite, with
# enough off-diagonal structure that the volatility constraint can actually bind.
_VARIANCES = {
    "AAA": D("0.0625"),  # sd 25%
    "BBB": D("0.0400"),  # sd 20%
    "CCC": D("0.0900"),  # sd 30%
    "DDD": D("0.0256"),  # sd 16%
    "EEE": D("0.0196"),  # sd 14%
    "FFF": D("0.0441"),  # sd 21%
}


def _covariance() -> dict[str, dict[str, Decimal]]:
    matrix: dict[str, dict[str, Decimal]] = {}
    for a in SYMBOLS:
        matrix[a] = {}
        for b in SYMBOLS:
            if a == b:
                matrix[a][b] = _VARIANCES[a]
            else:
                # Same sector correlates at 0.6, cross-sector at 0.2.
                rho = D("0.6") if SECTORS[a] == SECTORS[b] else D("0.2")
                sd_a = _VARIANCES[a].sqrt()
                sd_b = _VARIANCES[b].sqrt()
                matrix[a][b] = rho * sd_a * sd_b
    return matrix


COVARIANCE = _covariance()

POLICY = InvestmentPolicy(
    target_nominal_annual=D("0.08"),
    benchmark={"SPY": D("0.60"), "AGG": D("0.40")},
    ability=RiskLevel.ABOVE_AVERAGE,
    willingness=RiskLevel.MODERATE,
    max_portfolio_beta=D("1.20"),
    max_annualized_volatility=D("0.18"),
    volatility_lookback_days=60,
    safety_first=SafetyFirstPolicy(threshold_return=D("0.00"), minimum_ratio=D("0.30")),
    min_cash_buffer=D("0.05"),
    long_only=True,
    max_gross_exposure=D("0.95"),
    short_term_holding_days=366,
    short_term_gain_penalty=D("0.15"),
    wash_sale_window_days=30,
    horizon_years=10,
    stages=1,
    max_position_weight=D("0.10"),
    max_sector_weight=D("0.30"),
    corridor_absolute=D("0.05"),
    max_turnover=D("0.20"),
    min_trade_notional=D("100.00"),
    max_drawdown=D("0.15"),
)


# ---------------------------------------------------------------------------
# independent constraint checks, re-derived from the table
# ---------------------------------------------------------------------------


def portfolio_beta(weights: dict[str, Decimal]) -> Decimal:
    return sum((w * BETAS[s] for s, w in weights.items()), D(0))


def portfolio_volatility(weights: dict[str, Decimal]) -> Decimal:
    variance = D(0)
    for a, wa in weights.items():
        for b, wb in weights.items():
            variance += wa * wb * COVARIANCE[a][b]
    if variance <= 0:
        return D(0)
    return variance.sqrt()


def constraint_failures(
    weights: dict[str, Decimal],
    cash: Decimal,
    universe: frozenset[str],
    blocked: frozenset[str],
) -> list[str]:
    """Every rule that the final portfolio breaks. Independent of the engine."""
    failures: list[str] = []

    for symbol, weight in weights.items():
        if weight < -EPS:
            failures.append(f"NO_SHORTING: {symbol} at {weight}")
        if weight > POLICY.max_position_weight + EPS:
            failures.append(f"MAX_POSITION_WEIGHT: {symbol} at {weight}")
        if symbol not in universe:
            failures.append(f"UNIVERSE_WHITELIST: {symbol} not investable")
        if symbol in blocked and weight > EPS:
            failures.append(f"WASH_SALE_WINDOW: {symbol} repurchased at {weight}")

    by_sector: dict[str, Decimal] = {}
    for symbol, weight in weights.items():
        by_sector[SECTORS[symbol]] = by_sector.get(SECTORS[symbol], D(0)) + weight
    for sector, exposure in by_sector.items():
        if exposure > POLICY.max_sector_weight + EPS:
            failures.append(f"MAX_SECTOR_WEIGHT: {sector} at {exposure}")

    gross = sum(weights.values(), D(0))
    if gross > POLICY.max_gross_exposure + EPS:
        failures.append(f"NO_LEVERAGE: gross exposure {gross}")
    if cash < POLICY.min_cash_buffer - EPS:
        failures.append(f"MIN_CASH_BUFFER: cash {cash}")
    if abs(gross + cash - D(1)) > EPS:
        failures.append(f"weights and cash must sum to 1, got {gross + cash}")

    beta = portfolio_beta(weights)
    if beta > POLICY.max_portfolio_beta + EPS:
        failures.append(f"MAX_PORTFOLIO_BETA: {beta}")

    volatility = portfolio_volatility(weights)
    if volatility > POLICY.max_annualized_volatility + EPS:
        failures.append(f"MAX_VOLATILITY: {volatility}")

    return failures


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

weight = st.decimals(
    min_value=D("-0.5"), max_value=D("1.5"), places=4, allow_nan=False, allow_infinity=False
)

#: Deliberately adversarial: negative weights, weights above the position cap,
#: and gross exposure far above 1. A generator that only produced plausible
#: portfolios would never exercise the repair paths.
proposals = st.dictionaries(
    keys=st.sampled_from(SYMBOLS), values=weight, min_size=0, max_size=len(SYMBOLS)
)

current_weights = st.dictionaries(
    keys=st.sampled_from(SYMBOLS),
    values=st.decimals(min_value=D(0), max_value=D("0.10"), places=4, allow_nan=False),
    min_size=0,
    max_size=len(SYMBOLS),
)

drawdowns = st.decimals(min_value=D(0), max_value=D("0.40"), places=4, allow_nan=False)

#: Drawdowns strictly past the circuit-breaker limit. Generated directly rather
#: than filtered with assume(): filtering out ~60% of inputs both slows
#: generation and skews the distribution away from what is being tested.
breached_drawdowns = st.decimals(
    min_value=D("0.1501"), max_value=D("0.60"), places=4, allow_nan=False
)

universes = st.sets(st.sampled_from(SYMBOLS), min_size=1, max_size=len(SYMBOLS))

blocked_symbols = st.sets(st.sampled_from(SYMBOLS), min_size=0, max_size=2)


def build_context(
    current: dict[str, Decimal],
    universe: frozenset[str],
    blocked: frozenset[str],
    drawdown: Decimal,
) -> RiskContext:
    return RiskContext(
        as_of=AS_OF,
        current_weights=current,
        sectors=SECTORS,
        betas=BETAS,
        covariance=COVARIANCE,
        expected_returns=EXPECTED_RETURNS,
        universe=universe,
        drawdown=drawdown,
        risk_free_rate=D("0.04"),
        blocked_until={s: AS_OF + timedelta(days=10) for s in blocked},
    )


# ---------------------------------------------------------------------------
# the property
# ---------------------------------------------------------------------------


@settings(
    max_examples=10_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(
    proposed=proposals,
    current=current_weights,
    universe=universes,
    blocked=blocked_symbols,
    drawdown=drawdowns,
)
def test_no_approved_output_ever_violates_a_constraint(
    proposed: dict[str, Decimal],
    current: dict[str, Decimal],
    universe: set[str],
    blocked: set[str],
    drawdown: Decimal,
) -> None:
    """'s headline property, at 10,000 cases."""
    investable = frozenset(universe)
    barred = frozenset(blocked)
    context = build_context(current, investable, barred, drawdown)

    assessment = evaluate(proposed, context, POLICY)

    if assessment.decision is Decision.REJECTED:
        # A rejection must say why. An unexplained veto is not auditable.
        assert assessment.violations, "REJECTED with no reason code"
        return

    failures = constraint_failures(
        dict(assessment.weights), assessment.cash_weight, investable, barred
    )
    assert not failures, (
        f"{assessment.decision.value} output violates \n"
        + "\n".join(failures)
        + f"\n  proposed={proposed}\n  weights={dict(assessment.weights)}"
    )


@settings(max_examples=2_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(proposed=proposals, universe=universes)
def test_approved_means_untouched(proposed: dict[str, Decimal], universe: set[str]) -> None:
    """APPROVED must mean the proposal was compliant as submitted.

    If the engine changed anything, the honest answer is MODIFIED — otherwise
    a caller cannot tell whether the weights it gets back are its own.
    """
    investable = frozenset(universe)
    context = build_context({}, investable, frozenset(), D(0))
    assessment = evaluate(proposed, context, POLICY)

    if assessment.decision is Decision.APPROVED:
        submitted = {s: w for s, w in proposed.items() if w != 0}
        assert dict(assessment.weights) == submitted
        assert not assessment.repairs


@settings(max_examples=2_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(proposed=proposals, current=current_weights, universe=universes)
def test_output_is_always_a_valid_allocation(
    proposed: dict[str, Decimal], current: dict[str, Decimal], universe: set[str]
) -> None:
    """Weights plus cash sum to exactly 1, whatever the decision."""
    context = build_context(current, frozenset(universe), frozenset(), D(0))
    assessment = evaluate(proposed, context, POLICY)

    total = sum(assessment.weights.values(), D(0)) + assessment.cash_weight
    assert abs(total - D(1)) <= EPS, f"allocation sums to {total}"
    assert assessment.cash_weight >= 0


@settings(max_examples=2_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(proposed=proposals, universe=universes, blocked=blocked_symbols)
def test_a_blocked_symbol_is_never_repurchased(
    proposed: dict[str, Decimal], universe: set[str], blocked: set[str]
) -> None:
    """Wash-sale window: a name sold at a loss cannot come back inside 30 days."""
    barred = frozenset(blocked)
    context = build_context({}, frozenset(universe), barred, D(0))
    assessment = evaluate(proposed, context, POLICY)

    if assessment.decision is not Decision.REJECTED:
        for symbol in barred:
            assert assessment.weights.get(symbol, D(0)) <= EPS


@settings(max_examples=2_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(proposed=proposals, universe=universes, drawdown=breached_drawdowns)
def test_the_circuit_breaker_is_unconditional(
    proposed: dict[str, Decimal], universe: set[str], drawdown: Decimal
) -> None:
    """Past the drawdown limit, risk may never increase.

    peak-to-trough beyond 15% forces the minimum-variance portfolio
    and halts risk increases. There is no proposal that overrides this.
    """
    assert drawdown > POLICY.max_drawdown
    context = build_context({}, frozenset(universe), frozenset(), drawdown)
    assessment = evaluate(proposed, context, POLICY)

    assert any(
        v.code is ReasonCode.DRAWDOWN_CIRCUIT_BREAKER for v in assessment.violations
    ) or any(r.code is ReasonCode.DRAWDOWN_CIRCUIT_BREAKER for r in assessment.repairs)

    if assessment.decision is not Decision.REJECTED:
        assert portfolio_volatility(dict(assessment.weights)) <= (
            POLICY.max_annualized_volatility + EPS
        )


@settings(max_examples=2_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(proposed=proposals, current=current_weights, universe=universes)
def test_turnover_never_exceeds_the_cap(
    proposed: dict[str, Decimal], current: dict[str, Decimal], universe: set[str]
) -> None:
    """One-way turnover against current weights stays within the IPS cap."""
    context = build_context(current, frozenset(universe), frozenset(), D(0))
    assessment = evaluate(proposed, context, POLICY)

    if assessment.decision is Decision.REJECTED:
        return

    names = set(current) | set(assessment.weights)
    traded = sum(
        (abs(assessment.weights.get(s, D(0)) - current.get(s, D(0))) for s in names), D(0)
    )
    assert traded / 2 <= POLICY.max_turnover + EPS, f"turnover {traded / 2}"


@settings(max_examples=2_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(proposed=proposals, universe=universes)
def test_evaluation_is_deterministic(proposed: dict[str, Decimal], universe: set[str]) -> None:
    """Determinism: identical inputs produce identical output, every time."""
    context = build_context({}, frozenset(universe), frozenset(), D(0))
    first = evaluate(proposed, context, POLICY)
    second = evaluate(proposed, context, POLICY)

    assert first.decision is second.decision
    assert dict(first.weights) == dict(second.weights)
    assert first.cash_weight == second.cash_weight
    assert [v.code for v in first.violations] == [v.code for v in second.violations]
