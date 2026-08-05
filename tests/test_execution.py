"""Execution boundary: sizing, fill models, shortfall, reconciliation (SPEC §3, M5)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from src.decision.mandate import RebalanceMandate, build_mandate, reconcile
from src.execution import (
    Account,
    ExecutionError,
    ExecutionProvider,
    ExecutionReport,
    Fill,
    InstantFillModel,
    MarketSnapshot,
    Order,
    QueuePositionFillModel,
    RejectionCode,
    Side,
    SimulatedExecutor,
    SpreadCrossFillModel,
    get_executor,
    implementation_shortfall_bps,
    size_orders,
)
from src.time.clock import UTC

D = Decimal
NOW = datetime(2024, 6, 3, 21, 0, tzinfo=UTC)

PRICES = {"AAA": D("100.00"), "BBB": D("50.00"), "CCC": D("25.00")}


def market(**overrides: object) -> MarketSnapshot:
    base: dict[str, object] = {
        "timestamp": NOW,
        "prices": PRICES,
        "decision_prices": PRICES,
    }
    base.update(overrides)
    return MarketSnapshot(**base)  # type: ignore[arg-type]


def mandate(
    targets: dict[str, Decimal], current: dict[str, Decimal] | None = None
) -> RebalanceMandate:
    return build_mandate(
        decision_time=NOW,
        portfolio_value=D("100000.00"),
        target_weights=targets,
        current_weights=current or {},
        min_trade_notional=D("100.00"),
        max_turnover=D("0.20"),
    )


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_weights_become_whole_share_counts() -> None:
    # 10% of a 100,000 book at 100.00 = 100 shares.
    account = Account(cash=D("100000.00"))
    orders, _ = size_orders(mandate({"AAA": D("0.10")}), account, market())

    assert len(orders) == 1
    assert orders[0].symbol == "AAA"
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 100


def test_sizing_rounds_down_never_up() -> None:
    # 10% of 100,000 is 10,000; at 25.00 that is 400 shares exactly. Nudge the
    # price so the exact answer is fractional and check we never overshoot.
    account = Account(cash=D("100000.00"))
    prices = {**PRICES, "CCC": D("26.00")}
    orders, _ = size_orders(
        mandate({"CCC": D("0.10")}), account, market(prices=prices, decision_prices=prices)
    )
    # 10000 / 26 = 384.6 -> 384, not 385. Overshooting could breach a limit the
    # risk engine just finished enforcing.
    assert orders[0].quantity == 384


def test_sizing_recomputes_portfolio_value_from_current_prices() -> None:
    # The mandate says the book was worth 100,000 when decided. If prices moved,
    # sizing must use what it is worth *now*.
    account = Account(cash=D("50000.00"), positions={"AAA": 500})  # 100,000 total
    doubled = {"AAA": D("200.00"), "BBB": D("50.00"), "CCC": D("25.00")}
    orders, _ = size_orders(
        mandate({"BBB": D("0.10")}), account, market(prices=doubled, decision_prices=doubled)
    )
    # Book is now 50,000 cash + 500 x 200 = 150,000. 10% = 15,000 at 50 = 300.
    assert orders[0].quantity == 300


def test_selling_produces_a_sell_order() -> None:
    account = Account(cash=D("0"), positions={"AAA": 100})
    orders, _ = size_orders(mandate({"AAA": D("0.00")}, {"AAA": D("1.0")}), account, market())

    assert orders[0].side is Side.SELL
    assert orders[0].quantity == 100


def test_no_order_when_already_at_target() -> None:
    account = Account(cash=D("90000.00"), positions={"AAA": 100})
    orders, rejections = size_orders(mandate({"AAA": D("0.10")}), account, market())
    assert orders == []
    assert rejections == []


def test_min_trade_notional_is_enforced_below_the_boundary() -> None:
    # SPEC §7 passes this down as a constraint; the decision layer has no
    # notion of a trade, so it cannot enforce it.
    # 0.05% of 100,000 = 50.00, which at 25.00 is 2 shares worth 50 — a real
    # trade, but below the 100.00 minimum.
    account = Account(cash=D("100000.00"))
    orders, rejections = size_orders(mandate({"CCC": D("0.0005")}), account, market())

    assert orders == []
    assert rejections[0].reason_code is RejectionCode.BELOW_MIN_NOTIONAL


def test_a_target_below_one_share_produces_no_trade() -> None:
    # 0.05% of 100,000 at 100.00 is half a share. Rounding down means no trade
    # at all, which is correct: there is nothing to send.
    account = Account(cash=D("100000.00"))
    orders, rejections = size_orders(mandate({"AAA": D("0.0005")}), account, market())
    assert orders == []
    assert rejections == []


def test_a_missing_price_is_rejected_not_guessed() -> None:
    account = Account(cash=D("100000.00"))
    orders, rejections = size_orders(
        mandate({"ZZZ": D("0.10")}), account, market()
    )
    assert orders == []
    assert rejections[0].reason_code is RejectionCode.NO_PRICE


def test_sizing_is_deterministic_and_ordered() -> None:
    # SPEC §9: identical runs produce an identical trade log.
    account = Account(cash=D("100000.00"))
    request = mandate({"CCC": D("0.10"), "AAA": D("0.10"), "BBB": D("0.10")})
    first, _ = size_orders(request, account, market())
    second, _ = size_orders(request, account, market())

    assert [o.symbol for o in first] == ["AAA", "BBB", "CCC"]
    assert [o.symbol for o in first] == [o.symbol for o in second]


def test_sizing_refuses_a_worthless_account() -> None:
    with pytest.raises(ExecutionError):
        size_orders(mandate({"AAA": D("0.10")}), Account(cash=D("0")), market())


def test_orders_reject_a_non_positive_quantity() -> None:
    with pytest.raises(ExecutionError):
        Order("AAA", Side.BUY, 0, D("100"))


# ---------------------------------------------------------------------------
# Fill models
# ---------------------------------------------------------------------------


def test_instant_model_fills_at_the_reference_price_for_free() -> None:
    fills = InstantFillModel().fill(Order("AAA", Side.BUY, 100, D("100.00")), D("100.00"), NOW)
    assert fills[0].price == D("100.00")
    assert fills[0].commission == D("0")
    assert fills[0].quantity == 100


def test_spread_model_makes_a_buyer_pay_up() -> None:
    # 2bp spread on 100.00 -> half-spread 0.01, so a buy lifts at 100.01.
    model = SpreadCrossFillModel(spread_bps=D("2"), commission_per_share=D("0.005"))
    fills = model.fill(Order("AAA", Side.BUY, 100, D("100.00")), D("100.00"), NOW)

    assert fills[0].price == D("100.01")
    assert fills[0].commission == D("0.5")


def test_spread_model_makes_a_seller_receive_less() -> None:
    model = SpreadCrossFillModel(spread_bps=D("2"))
    fills = model.fill(Order("AAA", Side.SELL, 100, D("100.00")), D("100.00"), NOW)

    assert fills[0].price == D("99.99")
    assert fills[0].quantity == -100


def test_the_two_fill_models_disagree_and_that_gap_is_the_point() -> None:
    # SPEC §4.3: the gap between them IS the execution-cost sensitivity.
    order = Order("AAA", Side.BUY, 1000, D("100.00"))
    optimistic = InstantFillModel().fill(order, D("100.00"), NOW)[0]
    honest = SpreadCrossFillModel().fill(order, D("100.00"), NOW)[0]

    assert honest.price > optimistic.price
    assert honest.commission > optimistic.commission


def test_spread_model_refuses_to_produce_a_non_positive_price() -> None:
    model = SpreadCrossFillModel(spread_bps=D("100000"))
    fills = model.fill(Order("AAA", Side.SELL, 1, D("1.00")), D("1.00"), NOW)
    assert fills[0].price > D("0")


def test_queue_model_is_deliberately_not_implemented() -> None:
    # SPEC §4.3: backed by the C++ simulator, a separate project.
    with pytest.raises(NotImplementedError, match="C\\+\\+"):
        QueuePositionFillModel().fill(Order("AAA", Side.BUY, 1, D("1")), D("1"), NOW)


# ---------------------------------------------------------------------------
# Implementation shortfall
# ---------------------------------------------------------------------------


def test_buying_above_the_decision_price_is_a_positive_cost() -> None:
    fills = [Fill("AAA", 100, D("101.00"), NOW)]
    # 1% worse than the 100.00 decision price = 100bp of cost.
    assert implementation_shortfall_bps(fills, {"AAA": D("100.00")}) == D("100")


def test_selling_below_the_decision_price_is_also_a_cost() -> None:
    fills = [Fill("AAA", -100, D("99.00"), NOW)]
    assert implementation_shortfall_bps(fills, {"AAA": D("100.00")}) == D("100")


def test_filling_at_the_decision_price_costs_nothing() -> None:
    fills = [Fill("AAA", 100, D("100.00"), NOW)]
    assert implementation_shortfall_bps(fills, {"AAA": D("100.00")}) == D("0")


def test_shortfall_is_notional_weighted() -> None:
    # A big trade at 100bp and a tiny one at 0 should land near 100bp, not 50.
    fills = [
        Fill("AAA", 1000, D("101.00"), NOW),
        Fill("BBB", 1, D("50.00"), NOW),
    ]
    result = implementation_shortfall_bps(fills, {"AAA": D("100.00"), "BBB": D("50.00")})
    assert D("99") < result < D("100")


def test_shortfall_of_no_fills_is_zero() -> None:
    assert implementation_shortfall_bps([], {"AAA": D("100.00")}) == D("0")


# ---------------------------------------------------------------------------
# The simulated executor
# ---------------------------------------------------------------------------


def test_executor_reports_what_it_cannot_honor() -> None:
    # SPEC §3.2: an executor that silently drops a constraint teaches the
    # decision layer that the constraint works.
    capabilities = SimulatedExecutor().capabilities()
    assert capabilities.supports_participation_limits is False
    assert capabilities.supports_intraday is False
    assert capabilities.supports_streaming_updates is True


def test_executor_streams_fills_then_a_report() -> None:
    account = Account(cash=D("100000.00"))
    updates = list(SimulatedExecutor().execute(mandate({"AAA": D("0.10")}), account, market()))

    assert isinstance(updates[-1], ExecutionReport)
    assert any(isinstance(u, Fill) for u in updates)


def test_execution_moves_the_book_to_the_target() -> None:
    account = Account(cash=D("100000.00"))
    report = SimulatedExecutor().execute_to_completion(
        mandate({"AAA": D("0.10"), "BBB": D("0.05")}), account, market()
    )

    weights = report.realized_weights
    assert abs(weights["AAA"] - D("0.10")) < D("0.001")
    assert abs(weights["BBB"] - D("0.05")) < D("0.001")


def test_realized_weights_never_exactly_equal_targets_under_costs() -> None:
    # SPEC §3.4: this is why post-trade reconciliation is mandatory.
    account = Account(cash=D("100000.00"))
    request = mandate({"AAA": D("0.10")})
    report = SimulatedExecutor(fill_model=SpreadCrossFillModel()).execute_to_completion(
        request, account, market()
    )
    drift = reconcile(request, report.realized_weights)
    assert drift.total_absolute_drift > D("0")


def test_spread_model_produces_a_worse_shortfall_than_instant() -> None:
    account = Account(cash=D("100000.00"))
    request = mandate({"AAA": D("0.10"), "BBB": D("0.05")})

    optimistic = SimulatedExecutor(fill_model=InstantFillModel()).execute_to_completion(
        request, account, market()
    )
    honest = SimulatedExecutor(fill_model=SpreadCrossFillModel()).execute_to_completion(
        request, Account(cash=D("100000.00")), market()
    )

    assert optimistic.implementation_shortfall_bps == D("0")
    assert honest.implementation_shortfall_bps > D("0")
    assert honest.total_commission > optimistic.total_commission


def test_executor_refuses_to_overdraw_cash() -> None:
    # Rejecting is both the honest simulation and the only way to avoid
    # creating leverage the IPS forbids.
    account = Account(cash=D("500.00"))
    report = SimulatedExecutor().execute_to_completion(
        build_mandate(
            decision_time=NOW,
            portfolio_value=D("500.00"),
            target_weights={"AAA": D("1.0")},
            current_weights={},
            min_trade_notional=D("1.00"),
            max_turnover=D("1.0"),
        ),
        account,
        market(),
    )
    assert report.final_cash >= D("0")


def test_report_carries_the_mandate_id_for_idempotency() -> None:
    account = Account(cash=D("100000.00"))
    request = mandate({"AAA": D("0.10")})
    report = SimulatedExecutor().execute_to_completion(request, account, market())
    assert report.mandate_id == request.mandate_id


def test_execution_is_deterministic() -> None:
    # SPEC §9: two identical runs produce identical output.
    request = mandate({"AAA": D("0.10"), "BBB": D("0.05")})
    a = SimulatedExecutor().execute_to_completion(request, Account(cash=D("100000.00")), market())
    b = SimulatedExecutor().execute_to_completion(request, Account(cash=D("100000.00")), market())

    assert a.realized_weights == b.realized_weights
    assert a.implementation_shortfall_bps == b.implementation_shortfall_bps
    assert [f.price for f in a.fills] == [f.price for f in b.fills]


def test_realized_turnover_is_reported() -> None:
    account = Account(cash=D("100000.00"))
    report = SimulatedExecutor().execute_to_completion(mandate({"AAA": D("0.10")}), account, market())
    # Bought 10% of the book one way -> 5% two-way turnover measure.
    assert report.realized_turnover > D("0")


# ---------------------------------------------------------------------------
# Executor selection (SPEC §2.2: exactly one config value)
# ---------------------------------------------------------------------------


def test_executor_is_selected_by_a_single_name() -> None:
    assert isinstance(get_executor("simulated"), SimulatedExecutor)
    spread = get_executor("simulated_spread")
    assert isinstance(spread, SimulatedExecutor)
    assert spread.fill_model.name == "SpreadCrossFillModel"


def test_unknown_executor_is_reported() -> None:
    with pytest.raises(ExecutionError, match="unknown executor"):
        get_executor("teleporter")


def test_grpc_executor_is_a_stub() -> None:
    # SPEC §12: the C++ engine is a separate project. A stub that raises is the
    # correct implementation, and its existence proves the seam works.
    executor = get_executor("grpc")
    assert isinstance(executor, ExecutionProvider)
    with pytest.raises(NotImplementedError, match="separate project"):
        executor.capabilities()


def test_executor_selection_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTOR", "simulated_spread")
    executor = get_executor()
    assert isinstance(executor, SimulatedExecutor)
    assert executor.fill_model.name == "SpreadCrossFillModel"
