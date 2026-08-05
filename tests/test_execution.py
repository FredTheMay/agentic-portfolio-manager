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


# ---------------------------------------------------------------------------
# Naive executor against a paper broker (SPEC §3.3, M8)
# ---------------------------------------------------------------------------


class FakeBroker:
    """In-memory stand-in for the paper broker."""

    def __init__(self, cash_amount: Decimal = D("100000.00")) -> None:
        self._cash = cash_amount
        self._positions: dict[str, int] = {}
        self.submitted: list[dict[str, object]] = []
        self.seen_ids: set[str] = set()
        self.fail_next = False

    def submit_market_order(
        self, symbol: str, side: str, quantity: int, client_order_id: str
    ) -> dict[str, object]:
        from src.execution.naive import BrokerError

        if self.fail_next:
            self.fail_next = False
            raise BrokerError("simulated broker outage")
        if client_order_id in self.seen_ids:
            raise BrokerError(f"duplicate client_order_id {client_order_id}")
        self.seen_ids.add(client_order_id)
        self.submitted.append(
            {"symbol": symbol, "side": side, "qty": quantity, "id": client_order_id}
        )
        signed = quantity if side == "BUY" else -quantity
        self._positions[symbol] = self._positions.get(symbol, 0) + signed
        return {"filled_qty": str(quantity), "filled_avg_price": "100.00", "status": "filled"}

    def positions(self) -> dict[str, int]:
        return dict(self._positions)

    def cash(self) -> Decimal:
        return self._cash


def test_naive_executor_submits_market_orders() -> None:
    from src.execution.naive import NaiveExecutor

    broker = FakeBroker()
    executor = NaiveExecutor(broker=broker)
    report = executor.execute_to_completion(
        mandate({"AAA": D("0.10")}), Account(cash=D("100000.00")), market()
    )

    assert len(broker.submitted) == 1
    assert broker.submitted[0]["symbol"] == "AAA"
    assert report.fills[0].venue == "ALPACA_PAPER"


def test_client_order_id_is_deterministic() -> None:
    from src.execution.naive import client_order_id

    request = mandate({"AAA": D("0.10")})
    assert client_order_id(request.mandate_id, "AAA") == client_order_id(
        request.mandate_id, "AAA"
    )
    assert client_order_id(request.mandate_id, "AAA") != client_order_id(
        request.mandate_id, "BBB"
    )


def test_replaying_a_mandate_does_not_double_the_position() -> None:
    # The case that matters: a cycle retried after an ambiguous network failure,
    # where you genuinely do not know whether the order landed.
    from src.execution.naive import NaiveExecutor

    broker = FakeBroker()
    executor = NaiveExecutor(broker=broker)
    request = mandate({"AAA": D("0.10")})

    executor.execute_to_completion(request, Account(cash=D("100000.00")), market())
    second = executor.execute_to_completion(request, Account(cash=D("100000.00")), market())

    assert len(broker.submitted) == 1, "the replay must not reach the broker"
    assert any("idempotency" in r.detail for r in second.rejections)


def test_a_broker_failure_becomes_a_rejection_not_a_crash() -> None:
    from src.execution.naive import NaiveExecutor

    broker = FakeBroker()
    broker.fail_next = True
    executor = NaiveExecutor(broker=broker)
    report = executor.execute_to_completion(
        mandate({"AAA": D("0.10")}), Account(cash=D("100000.00")), market()
    )

    assert report.rejections
    assert report.fills == ()


def test_naive_executor_reports_what_it_cannot_honor() -> None:
    from src.execution.naive import NaiveExecutor

    capabilities = NaiveExecutor(broker=FakeBroker()).capabilities()
    assert capabilities.supports_participation_limits is False
    assert capabilities.supports_intraday is True


def test_naive_executor_writes_to_the_audit_log() -> None:
    from src.audit.log import AuditLog
    from src.execution.naive import NaiveExecutor

    audit = AuditLog()
    NaiveExecutor(broker=FakeBroker(), audit=audit).execute_to_completion(
        mandate({"AAA": D("0.10")}), Account(cash=D("100000.00")), market()
    )
    assert audit.by_code("ORDER_SUBMITTED")


def test_the_paper_broker_refuses_the_live_endpoint() -> None:
    # SPEC §1: no live broker endpoint, ever. Enforced in code rather than
    # left to a config review.
    from src.execution.naive import AlpacaPaperBroker, BrokerError

    with pytest.raises(BrokerError, match="paper-trading only"):
        AlpacaPaperBroker(base_url="https://api.alpaca.markets")


def test_the_paper_broker_needs_credentials() -> None:
    from src.execution.naive import AlpacaPaperBroker, BrokerError

    broker = AlpacaPaperBroker(key_id=None, secret_key=None)
    import os

    saved = {k: os.environ.pop(k, None) for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")}
    try:
        with pytest.raises(BrokerError, match="no Alpaca paper credentials"):
            broker.cash()
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_account_can_be_read_from_the_broker() -> None:
    from src.execution.naive import account_from_broker

    broker = FakeBroker()
    broker.submit_market_order("AAA", "BUY", 10, "seed")
    account = account_from_broker(broker)
    assert account.positions["AAA"] == 10
    assert account.cash == D("100000.00")
