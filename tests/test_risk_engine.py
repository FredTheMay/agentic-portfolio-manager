"""Per-constraint behaviour of the risk engine and the IPS loader.

The property test proves the engine never emits a violating portfolio. These
tests prove it enforces each rule for the *right reason* and with the right
verdict — a gate that rejected everything would pass the property test and be
useless.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from src.risk.codes import Decision, ReasonCode
from src.risk.engine import (
    RiskContext,
    evaluate,
    portfolio_beta,
    portfolio_volatility,
    rejection_summary,
    turnover,
    verify,
)
from src.risk.ips import (
    DEFAULT_IPS_PATH,
    InvestmentPolicy,
    PolicyError,
    RiskLevel,
    load_policy,
    policy_from_document,
)
from src.time.clock import UTC
from tests.test_risk_properties import BETAS, COVARIANCE, EXPECTED_RETURNS, POLICY, SECTORS

D = Decimal
AS_OF = datetime(2024, 6, 3, tzinfo=UTC)
UNIVERSE = frozenset(SECTORS)


def context(**overrides: object) -> RiskContext:
    base: dict[str, object] = {
        "as_of": AS_OF,
        "current_weights": {},
        "sectors": SECTORS,
        "betas": BETAS,
        "covariance": COVARIANCE,
        "expected_returns": EXPECTED_RETURNS,
        "universe": UNIVERSE,
        "drawdown": D("0"),
        "risk_free_rate": D("0.04"),
        "blocked_until": {},
    }
    base.update(overrides)
    return RiskContext(**base)  # type: ignore[arg-type]


def codes(assessment: object) -> set[ReasonCode]:
    violations = {v.code for v in assessment.violations}  # type: ignore[attr-defined]
    repairs = {r.code for r in assessment.repairs}  # type: ignore[attr-defined]
    return violations | repairs


# ---------------------------------------------------------------------------
# the IPS
# ---------------------------------------------------------------------------


def test_the_shipped_ips_loads() -> None:
    policy = load_policy()
    assert policy.max_portfolio_beta == D("1.20")
    assert policy.max_annualized_volatility == D("0.18")
    assert policy.min_cash_buffer == D("0.05")
    assert policy.max_position_weight == D("0.10")
    assert policy.max_sector_weight == D("0.30")
    assert policy.max_turnover == D("0.20")
    assert policy.max_drawdown == D("0.15")
    assert policy.wash_sale_window_days == 30


def test_the_lower_of_ability_and_willingness_binds() -> None:
    # Asks for this to be asserted and unit-tested by name.
    policy = load_policy()
    assert policy.ability is RiskLevel.ABOVE_AVERAGE
    assert policy.willingness is RiskLevel.MODERATE
    assert policy.binding_risk_tolerance is RiskLevel.MODERATE


def test_binding_tolerance_is_symmetric() -> None:
    # Whichever side is lower binds, not "willingness always".
    high_willingness = InvestmentPolicy(
        **{
            **{f.name: getattr(POLICY, f.name) for f in POLICY.__dataclass_fields__.values()},
            "ability": RiskLevel.BELOW_AVERAGE,
            "willingness": RiskLevel.ABOVE_AVERAGE,
        }
    )
    assert high_willingness.binding_risk_tolerance is RiskLevel.BELOW_AVERAGE


def test_ips_path_points_at_the_shipped_file() -> None:
    assert DEFAULT_IPS_PATH.is_file()


def test_the_cash_buffer_binds_tighter_than_the_leverage_ceiling() -> None:
    # NO_LEVERAGE at sum(w) <= 1.0 and MIN_CASH_BUFFER at 5%
    # independently. They are not in conflict: the liquidity floor simply binds
    # first, at 0.95. Naming the combination once keeps the two from being
    # applied inconsistently.
    policy = load_policy()
    assert policy.max_gross_exposure == D("1.00")
    assert policy.effective_exposure_ceiling == D("0.95")


def test_ips_rejects_a_cash_buffer_outside_zero_to_one() -> None:
    document = _valid_document()
    document["constraints"]["liquidity"]["min_cash_buffer"] = "1.50"
    with pytest.raises(PolicyError, match="min_cash_buffer"):
        policy_from_document(document)


def test_ips_rejects_a_sector_cap_below_the_position_cap() -> None:
    document = _valid_document()
    document["constraints"]["unique_circumstances"]["max_sector_weight"] = "0.05"
    document["constraints"]["unique_circumstances"]["max_position_weight"] = "0.10"
    with pytest.raises(PolicyError, match="no single position"):
        policy_from_document(document)


def test_ips_rejects_a_benchmark_that_does_not_sum_to_one() -> None:
    document = _valid_document()
    document["return_objective"]["benchmark"] = {"SPY": "0.60", "AGG": "0.30"}
    with pytest.raises(PolicyError, match="sum to 1"):
        policy_from_document(document)


def test_ips_reports_a_missing_section() -> None:
    document = _valid_document()
    del document["circuit_breaker"]
    with pytest.raises(PolicyError, match="circuit_breaker"):
        policy_from_document(document)


def test_ips_reports_an_unknown_risk_level() -> None:
    document = _valid_document()
    document["risk_objective"]["willingness"] = "ENTHUSIASTIC"
    with pytest.raises(PolicyError, match="ENTHUSIASTIC"):
        policy_from_document(document)


def test_load_policy_reports_a_missing_file(tmp_path: object) -> None:
    from pathlib import Path

    with pytest.raises(PolicyError, match="not found"):
        load_policy(Path(str(tmp_path)) / "nope.yaml")


def _valid_document() -> dict[str, Any]:
    import copy

    import yaml

    return copy.deepcopy(yaml.safe_load(DEFAULT_IPS_PATH.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# individual constraints
# ---------------------------------------------------------------------------


def test_a_compliant_proposal_is_approved_untouched() -> None:
    proposal = {"AAA": D("0.08"), "DDD": D("0.09"), "FFF": D("0.07")}
    assessment = evaluate(proposal, context(), POLICY)

    assert assessment.decision is Decision.APPROVED
    assert dict(assessment.weights) == proposal
    assert assessment.repairs == ()
    assert assessment.violations == ()


def test_position_cap_is_enforced_by_capping() -> None:
    assessment = evaluate({"AAA": D("0.40")}, context(), POLICY)

    assert assessment.decision is Decision.MODIFIED
    assert assessment.weights["AAA"] == D("0.10")
    assert ReasonCode.MAX_POSITION_WEIGHT in codes(assessment)


def test_sector_cap_scales_the_offending_sector_only() -> None:
    # TECH: AAA + BBB + CCC = 0.30 exactly at the cap. Push it over.
    proposal = {"AAA": D("0.10"), "BBB": D("0.10"), "CCC": D("0.10"), "DDD": D("0.10")}
    assessment = evaluate(proposal, context(), POLICY)

    tech = sum(assessment.weights[s] for s in ("AAA", "BBB", "CCC"))
    assert tech <= D("0.30") + D("1e-12")
    # HEALTH was never over its cap, so DDD must be untouched by the sector step.
    assert assessment.weights["DDD"] <= D("0.10")


def test_short_weights_are_clamped_away() -> None:
    assessment = evaluate({"AAA": D("0.08"), "BBB": D("-0.05")}, context(), POLICY)

    assert "BBB" not in assessment.weights
    assert ReasonCode.NO_SHORTING in codes(assessment)


def test_leverage_is_scaled_back_to_the_exposure_ceiling() -> None:
    proposal = {s: D("0.10") for s in SECTORS}  # gross 0.60, fine
    heavy = {**proposal, "AAA": D("0.10")}
    assessment = evaluate(heavy, context(), POLICY)
    assert sum(assessment.weights.values()) <= POLICY.max_gross_exposure + D("1e-12")


def test_cash_buffer_always_survives() -> None:
    # Ask for 100% invested; the liquidity floor must still hold.
    proposal = {s: D("0.10") for s in SECTORS}
    assessment = evaluate(proposal, context(), POLICY)
    assert assessment.cash_weight >= POLICY.min_cash_buffer


def test_uninvestable_names_are_removed() -> None:
    assessment = evaluate(
        {"AAA": D("0.08"), "ZZZ": D("0.05")},
        context(universe=frozenset({"AAA", "DDD", "FFF"})),
        POLICY,
    )
    assert "ZZZ" not in assessment.weights
    assert ReasonCode.UNIVERSE_WHITELIST in codes(assessment)


def test_wash_sale_window_blocks_a_repurchase() -> None:
    blocked = {"AAA": AS_OF + timedelta(days=10)}
    assessment = evaluate({"AAA": D("0.08"), "DDD": D("0.08")}, context(blocked_until=blocked), POLICY)

    assert "AAA" not in assessment.weights
    assert ReasonCode.WASH_SALE_WINDOW in codes(assessment)


def test_wash_sale_window_expires() -> None:
    expired = {"AAA": AS_OF - timedelta(days=1)}
    assessment = evaluate({"AAA": D("0.08")}, context(blocked_until=expired), POLICY)
    assert assessment.weights.get("AAA") == D("0.08")


def test_beta_ceiling_scales_the_portfolio_toward_cash() -> None:
    # CCC has beta 1.6; a concentrated high-beta book must be scaled down.
    high_beta = {"AAA": D("0.10"), "CCC": D("0.10"), "BBB": D("0.10")}
    assessment = evaluate(high_beta, context(), POLICY)
    assert portfolio_beta(assessment.weights, BETAS) <= POLICY.max_portfolio_beta + D("1e-12")


def test_volatility_ceiling_is_enforced() -> None:
    proposal = {"AAA": D("0.10"), "CCC": D("0.10"), "BBB": D("0.09")}
    assessment = evaluate(proposal, context(), POLICY)
    assert portfolio_volatility(assessment.weights, COVARIANCE) <= (
        POLICY.max_annualized_volatility + D("1e-12")
    )


def test_rebalance_corridor_vetoes_a_trivial_trade() -> None:
    current = {"AAA": D("0.08"), "DDD": D("0.08")}
    barely_different = {"AAA": D("0.09"), "DDD": D("0.08")}
    assessment = evaluate(barely_different, context(current_weights=current), POLICY)

    assert assessment.decision is Decision.REJECTED
    assert {v.code for v in assessment.violations} == {ReasonCode.REBALANCE_CORRIDOR}
    # A vetoed rebalance leaves the book where it was.
    assert dict(assessment.weights) == current


def test_rebalance_corridor_allows_a_real_move() -> None:
    current = {"AAA": D("0.02"), "DDD": D("0.02")}
    assessment = evaluate({"AAA": D("0.10"), "DDD": D("0.09")}, context(current_weights=current), POLICY)
    assert assessment.decision is not Decision.REJECTED


def test_corridor_does_not_block_initial_funding() -> None:
    # With no current holdings there is nothing to drift from.
    assessment = evaluate({"AAA": D("0.01")}, context(current_weights={}), POLICY)
    assert assessment.decision is not Decision.REJECTED


def test_turnover_is_capped_by_blending_toward_current() -> None:
    current = {"AAA": D("0.10"), "BBB": D("0.10"), "CCC": D("0.10")}
    target = {"DDD": D("0.10"), "EEE": D("0.10"), "FFF": D("0.10")}
    assessment = evaluate(target, context(current_weights=current), POLICY)

    if assessment.decision is not Decision.REJECTED:
        assert turnover(current, assessment.weights) <= POLICY.max_turnover + D("1e-12")
        assert ReasonCode.MAX_TURNOVER in codes(assessment)


def test_drawdown_breaker_overrides_any_proposal() -> None:
    reckless = {"CCC": D("0.10"), "AAA": D("0.10")}
    assessment = evaluate(reckless, context(drawdown=D("0.20")), POLICY)

    assert ReasonCode.DRAWDOWN_CIRCUIT_BREAKER in codes(assessment)
    # Whatever came back, it is not the aggressive book that was asked for.
    assert dict(assessment.weights) != reckless


def test_drawdown_breaker_does_not_fire_inside_the_limit() -> None:
    assessment = evaluate({"AAA": D("0.08")}, context(drawdown=D("0.10")), POLICY)
    assert ReasonCode.DRAWDOWN_CIRCUIT_BREAKER not in codes(assessment)


def test_safety_first_floor_can_reject() -> None:
    # A floor no portfolio can clear must produce a rejection, not a silent pass.
    strict = InvestmentPolicy(
        **{
            **{f.name: getattr(POLICY, f.name) for f in POLICY.__dataclass_fields__.values()},
            "safety_first": type(POLICY.safety_first)(
                threshold_return=D("0.00"), minimum_ratio=D("99")
            ),
        }
    )
    assessment = evaluate({"AAA": D("0.08"), "DDD": D("0.08")}, context(), strict)

    assert assessment.decision is Decision.REJECTED
    assert ReasonCode.SAFETY_FIRST_THRESHOLD in {v.code for v in assessment.violations}


def test_an_all_cash_book_has_no_safety_first_ratio() -> None:
    # No shortfall distribution exists, so the ratio is undefined rather than
    # failing — cash cannot breach a capital-preservation floor.
    assessment = evaluate({}, context(), POLICY)
    assert assessment.metrics.safety_first_ratio is None
    assert assessment.decision is Decision.APPROVED
    assert assessment.cash_weight == D("1")


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def test_min_trade_notional_is_passed_through_not_enforced() -> None:
    # This is a constraint for the executor. The decision layer does
    # not size orders, so it cannot enforce a notional here.
    assessment = evaluate({"AAA": D("0.08")}, context(), POLICY)
    assert assessment.min_trade_notional == POLICY.min_trade_notional


def test_metrics_are_reported_for_the_returned_portfolio() -> None:
    assessment = evaluate({"AAA": D("0.08"), "DDD": D("0.09")}, context(), POLICY)
    metrics = assessment.metrics

    assert metrics.gross_exposure == D("0.17")
    assert metrics.cash_weight == D("0.83")
    assert metrics.portfolio_beta == portfolio_beta(assessment.weights, BETAS)


def test_every_rejection_carries_a_code() -> None:
    # Every rejection is persisted and surfaced. An unexplained veto
    # is not auditable.
    current = {"AAA": D("0.08")}
    assessment = evaluate({"AAA": D("0.09")}, context(current_weights=current), POLICY)
    assert assessment.decision is Decision.REJECTED
    assert assessment.violations
    for violation in assessment.violations:
        assert violation.code in ReasonCode
        assert violation.detail


def test_rejection_summary_counts_by_code() -> None:
    current = {"AAA": D("0.08")}
    vetoed = [
        evaluate({"AAA": D("0.09")}, context(current_weights=current), POLICY) for _ in range(3)
    ]
    assert rejection_summary(vetoed) == {ReasonCode.REBALANCE_CORRIDOR: 3}


def test_verify_is_independent_of_the_repair_path() -> None:
    # Handed a portfolio the engine never produced, verify must still object.
    violations = verify({"AAA": D("1.50")}, context(), POLICY)
    found = {v.code for v in violations}
    assert ReasonCode.MAX_POSITION_WEIGHT in found
    assert ReasonCode.MAX_SECTOR_WEIGHT in found
    assert ReasonCode.NO_LEVERAGE in found
    assert ReasonCode.MIN_CASH_BUFFER in found


def test_verify_does_not_cry_leverage_below_the_ceiling() -> None:
    # 0.90 gross against a 0.95 effective ceiling breaches concentration and
    # risk limits, but not the leverage rule. Over-reporting codes would make
    # the vetoed-trades panel useless.
    found = {v.code for v in verify({"AAA": D("0.90")}, context(), POLICY)}
    assert ReasonCode.NO_LEVERAGE not in found
    assert ReasonCode.MIN_CASH_BUFFER not in found
    assert ReasonCode.MAX_POSITION_WEIGHT in found


def test_engine_does_not_mutate_its_inputs() -> None:
    # Pure function: the caller's dictionaries must come back unchanged.
    proposal = {"AAA": D("0.40")}
    current = {"BBB": D("0.05")}
    evaluate(proposal, context(current_weights=current), POLICY)

    assert proposal == {"AAA": D("0.40")}
    assert current == {"BBB": D("0.05")}
