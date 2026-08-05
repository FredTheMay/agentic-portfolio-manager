"""Dashboard API (SPEC §9, M9)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from src.api.routes import DashboardState, create_app
from src.api.schemas import DISCLAIMER, SURVIVORSHIP_NOTICE
from src.audit.log import AuditEvent, AuditLog, Standard
from src.backtest.engine import BacktestConfig, run_backtest
from src.execution.fill_models import SpreadCrossFillModel
from src.execution.simulated import SimulatedExecutor
from src.risk.ips import load_policy
from src.time.clock import UTC
from tests.synthetic import BETAS, SECTORS, make_source

D = Decimal
START = datetime(2022, 1, 3, 21, tzinfo=UTC)
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL")


@pytest.fixture(scope="module")
def client() -> TestClient:
    config = BacktestConfig(
        start=START,
        end=START + timedelta(days=560),
        initial_cash=D("100000.00"),
        symbols=SYMBOLS,
        benchmark_symbol="SPY",
        estimation_window=100,
    )
    executor = SimulatedExecutor(fill_model=SpreadCrossFillModel())
    result = run_backtest(config, make_source(), executor, load_policy(), SECTORS, BETAS)

    audit = AuditLog()
    audit.record(
        AuditEvent(START, "research", "MISSING_CITATION", Standard.V_A_DILIGENCE, "no source")
    )
    state = DashboardState.from_result(
        result, capabilities=executor.capabilities(), audit=audit, sectors=SECTORS
    )
    return TestClient(create_app(state))


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"


def test_every_response_carries_the_disclaimer(client: TestClient) -> None:
    # SPEC §1: the banner is a required field, not a template detail, so no
    # screen can render without it.
    for path in ("/api/portfolio", "/api/performance", "/api/vetoes", "/api/status"):
        assert client.get(path).json()["disclaimer"] == DISCLAIMER


def test_status_states_the_survivorship_limitation(client: TestClient) -> None:
    # SPEC §4.4 requires this in the dashboard footer, not only the README.
    assert client.get("/api/status").json()["survivorship_notice"] == SURVIVORSHIP_NOTICE


def test_status_says_what_it_is_running(client: TestClient) -> None:
    body = client.get("/api/status").json()
    assert body["cycles"] >= 1
    assert "llm_provider" in body and "executor" in body


def test_portfolio_returns_holdings(client: TestClient) -> None:
    body = client.get("/api/portfolio").json()
    assert Decimal(body["total_value"]) > 0
    for holding in body["holdings"]:
        assert Decimal(holding["weight"]) >= 0
        assert holding["sector"]


def test_monetary_values_cross_the_wire_as_strings(client: TestClient) -> None:
    # SPEC §3.2's rule applies here too: a weight round-tripped through a JSON
    # double is no longer the weight the risk engine approved.
    body = client.get("/api/portfolio").json()
    assert isinstance(body["total_value"], str)
    assert isinstance(body["cash_weight"], str)
    for holding in body["holdings"]:
        assert isinstance(holding["weight"], str)


def test_performance_reports_twr_and_mwr(client: TestClient) -> None:
    body = client.get("/api/performance").json()
    assert isinstance(body["annualized_twr"], str)
    assert "annualized_benchmark_twr" in body
    assert "mwr" in body
    assert len(body["equity_curve"]) == len(body["timestamps"])


def test_performance_exposes_alpha_significance(client: TestClient) -> None:
    # A positive alpha with an insignificant t-stat must not read as a result.
    body = client.get("/api/performance").json()
    assert isinstance(body["alpha_is_significant"], bool)
    assert "alpha_t_stat" in body


def test_frontier_is_served_with_the_selected_portfolio(client: TestClient) -> None:
    body = client.get("/api/frontier").json()
    assert body["points"]
    assert body["selected"] is not None
    assert body["method"] in {"MAX_SHARPE", "MINIMUM_VARIANCE"}


def test_vetoes_are_grouped_by_code(client: TestClient) -> None:
    # SPEC §7 calls this the screen to demo first.
    body = client.get("/api/vetoes").json()
    assert body["total"] == len(body["vetoes"])
    for veto in body["vetoes"]:
        assert veto["code"]
        assert veto["detail"]


def test_attribution_splits_systematic_from_diversifiable(client: TestClient) -> None:
    body = client.get("/api/attribution").json()
    total = Decimal(body["total_variance"])
    systematic = Decimal(body["systematic_variance"])
    unsystematic = Decimal(body["unsystematic_variance"])
    assert abs((systematic + unsystematic) - total) < Decimal("1e-6")


def test_audit_trail_is_served(client: TestClient) -> None:
    body = client.get("/api/audit").json()
    assert body["total"] >= 1
    assert body["entries"][0]["standard"].startswith("V(A)")


def test_capabilities_name_the_advisory_constraints(client: TestClient) -> None:
    # SPEC §3.2: a constraint the executor cannot honor is advisory, and the
    # operator should see that without reading code.
    body = client.get("/api/capabilities").json()
    assert body["supports_participation_limits"] is False
    assert "max_participation_rate" in body["advisory_constraints"]


def test_cycles_expose_the_decision_trail(client: TestClient) -> None:
    body = client.get("/api/cycles").json()
    assert body
    for cycle in body:
        assert cycle["decision"] in {"APPROVED", "MODIFIED", "REJECTED"}


def test_the_api_is_read_only(client: TestClient) -> None:
    # CFA Standard III(A): the IPS binds at runtime. A route that could relax
    # it would make that untrue whatever the YAML said.
    for path in ("/api/portfolio", "/api/vetoes", "/api/capabilities"):
        assert client.post(path).status_code in (404, 405)
        assert client.delete(path).status_code in (404, 405)
