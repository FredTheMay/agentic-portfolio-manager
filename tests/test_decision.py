"""Optimizer and mandate emission (SPEC §3, §6.2, M4)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.cfa.portfolio import efficient_frontier, sharpe_ratio
from src.decision.mandate import (
    ExecutionConstraints,
    MandateError,
    RebalanceMandate,
    TargetWeight,
    Urgency,
    build_mandate,
    mandate_id,
    reconcile,
)
from src.decision.optimizer import (
    MarketInputs,
    OptimizerError,
    apply_view_tilts,
    capm_expected_returns,
    estimate_inputs,
    optimize,
)
from src.time.clock import UTC

D = Decimal
NOW = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)

SYMBOLS = ("AAA", "BBB", "CCC", "DDD")
BETAS = {"AAA": D("1.4"), "BBB": D("1.0"), "CCC": D("0.7"), "DDD": D("0.9")}

# Four assets, uncorrelated, with rising risk.
COV = [
    [D("0.0400"), D("0.0050"), D("0.0020"), D("0.0030")],
    [D("0.0050"), D("0.0250"), D("0.0015"), D("0.0025")],
    [D("0.0020"), D("0.0015"), D("0.0100"), D("0.0010")],
    [D("0.0030"), D("0.0025"), D("0.0010"), D("0.0196")],
]
MU = (D("0.12"), D("0.09"), D("0.06"), D("0.08"))


def inputs(risk_free: str = "0.03") -> MarketInputs:
    return MarketInputs(
        symbols=SYMBOLS,
        expected_returns=MU,
        covariance=tuple(tuple(row) for row in COV),
        risk_free_rate=D(risk_free),
    )


# ---------------------------------------------------------------------------
# Expected-return construction
# ---------------------------------------------------------------------------


def test_capm_expected_returns() -> None:
    # E(Ri) = Rf + beta(E(Rm) - Rf); beta 1.4 at Rf 3%, market 9% -> 0.114
    result = capm_expected_returns(BETAS, market_return=D("0.09"), risk_free_rate=D("0.03"))
    assert result["AAA"] == D("0.114")
    assert result["BBB"] == D("0.09")


def test_view_tilts_are_additive_and_only_touch_known_symbols() -> None:
    base = {"AAA": D("0.10"), "BBB": D("0.08")}
    tilted = apply_view_tilts(base, {"AAA": D("0.02"), "ZZZ": D("0.05")})

    assert tilted["AAA"] == D("0.12")
    assert tilted["BBB"] == D("0.08")
    assert "ZZZ" not in tilted


def test_view_tilts_do_not_mutate_the_baseline() -> None:
    base = {"AAA": D("0.10")}
    apply_view_tilts(base, {"AAA": D("0.02")})
    assert base == {"AAA": D("0.10")}


def test_estimate_inputs_shrinks_and_annualizes() -> None:
    observations = [
        [D("0.010"), D("0.008"), D("0.004"), D("0.006")],
        [D("-0.006"), D("-0.004"), D("0.002"), D("-0.003")],
        [D("0.012"), D("0.009"), D("-0.001"), D("0.007")],
        [D("0.001"), D("0.002"), D("0.003"), D("0.001")],
        [D("-0.009"), D("-0.007"), D("0.001"), D("-0.005")],
        [D("0.004"), D("0.003"), D("0.002"), D("0.002")],
    ]
    result = estimate_inputs(
        SYMBOLS,
        observations,
        BETAS,
        market_return=D("0.09"),
        risk_free_rate=D("0.03"),
        periods_per_year=252,
    )

    assert result.symbols == SYMBOLS
    assert result.shrinkage is not None and D(0) <= result.shrinkage <= D(1)
    # Annualized daily variance is far larger than the daily figure.
    assert result.covariance[0][0] > D("0.001")


def test_estimate_inputs_reports_a_symbol_without_a_beta() -> None:
    observations = [[D("0.01"), D("0.01")], [D("-0.01"), D("0.00")], [D("0.02"), D("0.01")]]
    with pytest.raises(OptimizerError, match="expected return"):
        estimate_inputs(
            ("AAA", "ZZZ"), observations, {"AAA": D("1.0")}, D("0.09"), D("0.03")
        )


def test_market_inputs_validate_their_shape() -> None:
    with pytest.raises(OptimizerError):
        MarketInputs(symbols=(), expected_returns=(), covariance=(), risk_free_rate=D("0.03"))
    with pytest.raises(OptimizerError, match="square"):
        MarketInputs(
            symbols=("A", "B"),
            expected_returns=(D("0.1"), D("0.1")),
            covariance=((D("0.04"),),),
            risk_free_rate=D("0.03"),
        )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_optimize_produces_an_investable_portfolio() -> None:
    result = optimize(inputs())

    assert result.weights
    assert all(w > 0 for w in result.weights.values())
    assert abs(sum(result.weights.values()) - D(1)) < D("1e-6")


def test_optimize_maximizes_sharpe_on_the_frontier() -> None:
    result = optimize(inputs(), frontier_points=25)
    assert result.method == "MAX_SHARPE"

    # No frontier point may beat the selected one.
    frontier = efficient_frontier(list(MU), [list(r) for r in COV], points=25, long_only=True)
    for point in frontier:
        if point.standard_deviation > 0:
            rival = sharpe_ratio(point.expected_return, D("0.03"), point.standard_deviation)
            assert rival <= result.sharpe + D("1e-9")


def test_position_cap_is_respected_during_optimization() -> None:
    # Applied as a bound in the solve, not by truncating the result afterwards.
    result = optimize(inputs(), max_position_weight=D("0.30"))
    assert max(result.weights.values()) <= D("0.30") + D("1e-6")


def test_an_infeasible_position_cap_is_reported() -> None:
    # Four assets capped at 10% each cannot reach a fully invested portfolio.
    with pytest.raises(OptimizerError):
        optimize(inputs(), max_position_weight=D("0.10"))


def test_optimizer_falls_back_to_minimum_variance_when_nothing_beats_cash() -> None:
    # Risk-free above every expected return: no positive-Sharpe portfolio exists,
    # so the honest answer is the lowest-risk one rather than the least-bad bet.
    result = optimize(inputs(risk_free="0.50"))
    assert result.method == "MINIMUM_VARIANCE"


def test_optimizer_drops_solver_dust() -> None:
    # SLSQP leaves 1e-17 residuals that are not real positions.
    result = optimize(inputs())
    assert all(w > D("1e-9") for w in result.weights.values())


def test_optimizer_rejects_a_degenerate_frontier_request() -> None:
    with pytest.raises(OptimizerError):
        optimize(inputs(), frontier_points=1)


def test_optimization_is_deterministic() -> None:
    # SPEC §9: identical inputs, identical output.
    first = optimize(inputs())
    second = optimize(inputs())
    assert dict(first.weights) == dict(second.weights)
    assert first.sharpe == second.sharpe


def test_frontier_is_returned_for_the_dashboard() -> None:
    result = optimize(inputs(), frontier_points=12)
    assert len(result.frontier) == 12


# ---------------------------------------------------------------------------
# Mandate
# ---------------------------------------------------------------------------


def test_mandate_carries_weights_not_orders() -> None:
    # SPEC §3.1. The wire form has no share count, no order type, no venue.
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.07")},
        current_weights={"AAA": D("0.04")},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    wire = mandate.to_wire()

    assert wire["targets"] == [
        {"symbol": "AAA", "target_weight": "0.0700", "current_weight": "0.0400"}
    ]
    flat = str(wire).lower()
    for forbidden in ("quantity", "shares", "order", "venue", "limit_price"):
        assert forbidden not in flat


def test_every_monetary_value_on_the_wire_is_a_string() -> None:
    # SPEC §3.2: never float. Binary rounding at the one place two languages
    # must agree exactly is the worst possible place for it.
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.005"),
        target_weights={"AAA": D("0.07")},
        current_weights={},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    wire = mandate.to_wire()
    assert isinstance(wire["portfolio_value"], str)
    assert isinstance(wire["constraints"]["min_trade_notional"], str)
    for target in wire["targets"]:
        assert isinstance(target["target_weight"], str)


def test_exits_are_emitted_with_a_zero_target() -> None:
    # Omitting them would leave the executor unable to distinguish "sell this"
    # from "no opinion", stranding the position.
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.07")},
        current_weights={"AAA": D("0.04"), "BBB": D("0.05")},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    exits = [t for t in mandate.targets if t.symbol == "BBB"]
    assert len(exits) == 1
    assert exits[0].target_weight == D("0")
    assert exits[0].current_weight == D("0.05")


def test_untouched_symbols_are_not_included() -> None:
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.07"), "CCC": D("0")},
        current_weights={},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    assert [t.symbol for t in mandate.targets] == ["AAA"]


def test_mandate_id_is_deterministic() -> None:
    # A UUID would break both idempotency and SPEC §9's identical-run promise.
    kwargs = dict(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.07"), "BBB": D("0.03")},
        current_weights={},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    assert build_mandate(**kwargs).mandate_id == build_mandate(**kwargs).mandate_id  # type: ignore[arg-type]


def test_mandate_id_changes_with_the_decision() -> None:
    base = dict(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        current_weights={},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    a = build_mandate(target_weights={"AAA": D("0.07")}, **base)  # type: ignore[arg-type]
    b = build_mandate(target_weights={"AAA": D("0.08")}, **base)  # type: ignore[arg-type]
    assert a.mandate_id != b.mandate_id


def test_mandate_id_ignores_symbol_ordering() -> None:
    targets = [
        TargetWeight("BBB", D("0.03"), D("0")),
        TargetWeight("AAA", D("0.07"), D("0")),
    ]
    assert mandate_id(NOW, D("100000.00"), targets) == mandate_id(
        NOW, D("100000.00"), list(reversed(targets))
    )


def test_mandate_rejects_a_non_positive_portfolio_value() -> None:
    with pytest.raises(MandateError):
        RebalanceMandate(
            mandate_id="x",
            decision_time=NOW,
            portfolio_value=D("0"),
            targets=(),
            constraints=ExecutionConstraints(D("100"), D("0.20")),
        )


def test_mandate_rejects_duplicate_symbols() -> None:
    with pytest.raises(MandateError, match="duplicate"):
        RebalanceMandate(
            mandate_id="x",
            decision_time=NOW,
            portfolio_value=D("1000"),
            targets=(
                TargetWeight("AAA", D("0.07"), D("0")),
                TargetWeight("AAA", D("0.03"), D("0")),
            ),
            constraints=ExecutionConstraints(D("100"), D("0.20")),
        )


def test_mandate_normalizes_the_decision_time_to_utc() -> None:
    eastern = datetime(2024, 6, 3, 16, 0, tzinfo=UTC) - timedelta(hours=0)
    mandate = build_mandate(
        decision_time=eastern,
        portfolio_value=D("1000"),
        target_weights={"AAA": D("0.5")},
        current_weights={},
        min_trade_notional=D("100"),
        max_turnover=D("0.20"),
    )
    assert mandate.decision_time.tzinfo is UTC


def test_implied_turnover() -> None:
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.10"), "BBB": D("0.00")},
        current_weights={"AAA": D("0.00"), "BBB": D("0.10")},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    # Buy 10 points of AAA, sell 10 of BBB -> one-way turnover 10%.
    assert mandate.implied_turnover == D("0.10")


def test_urgency_defaults_to_normal() -> None:
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("1000"),
        target_weights={"AAA": D("0.5")},
        current_weights={},
        min_trade_notional=D("100"),
        max_turnover=D("0.20"),
    )
    assert mandate.urgency is Urgency.NORMAL
    assert mandate.to_wire()["urgency"] == "NORMAL"


def test_deadline_is_serialized_when_present() -> None:
    constraints = ExecutionConstraints(D("100"), D("0.20"), deadline=NOW)
    assert "deadline" in constraints.to_wire()
    assert "deadline" not in ExecutionConstraints(D("100"), D("0.20")).to_wire()


# ---------------------------------------------------------------------------
# Reconciliation (SPEC §3.4)
# ---------------------------------------------------------------------------


def test_reconciliation_measures_drift_against_intent() -> None:
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.10"), "BBB": D("0.05")},
        current_weights={"AAA": D("0.00"), "BBB": D("0.00")},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    # The executor got close but not exact — which is the normal case.
    result = reconcile(mandate, {"AAA": D("0.098"), "BBB": D("0.051")})

    assert result.per_symbol_drift["AAA"] == D("-0.002")
    assert result.per_symbol_drift["BBB"] == D("0.001")
    assert result.total_absolute_drift == D("0.003")
    assert result.max_drift_symbol == "AAA"


def test_reconciliation_notices_an_unrequested_fill() -> None:
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.10")},
        current_weights={},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    result = reconcile(mandate, {"AAA": D("0.10"), "ZZZ": D("0.02")})
    assert result.per_symbol_drift["ZZZ"] == D("0.02")


def test_perfect_execution_reconciles_to_zero() -> None:
    mandate = build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights={"AAA": D("0.10")},
        current_weights={},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )
    result = reconcile(mandate, {"AAA": D("0.10")})
    assert result.total_absolute_drift == D("0")
    assert result.max_drift == D("0")


def test_reconciliation_of_an_empty_mandate() -> None:
    mandate = RebalanceMandate(
        mandate_id="x",
        decision_time=NOW,
        portfolio_value=D("1000"),
        targets=(),
        constraints=ExecutionConstraints(D("100"), D("0.20")),
    )
    result = reconcile(mandate, {})
    assert result.max_drift == D("0")
    assert result.max_drift_symbol is None
