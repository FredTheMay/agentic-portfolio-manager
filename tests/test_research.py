"""Per-symbol research service and endpoints (M11)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.research import ResearchService
from src.api.routes import DashboardState, create_app
from src.api.schemas import RATIO_FAMILIES
from src.backtest.engine import BacktestConfig, run_backtest
from src.data.live import resolve_setup
from src.data.universe import load_universe
from src.execution.simulated import SimulatedExecutor
from src.risk.ips import load_policy

D = Decimal


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> ResearchService:
    # An empty cache root forces the synthetic fallback, so this exercises the
    # service without depending on anyone having run `make backfill`.
    setup = resolve_setup(cache_root=tmp_path_factory.mktemp("empty"))
    return ResearchService(
        setup=setup,
        universe=load_universe(),
        current_weights={"AAPL": D("0.07")},
        veto_codes={"AAPL": ("MAX_SECTOR_WEIGHT",)},
        cache_root=Path("/nonexistent"),
    )


def test_the_screen_covers_the_whole_universe(service: ResearchService) -> None:
    cards = service.screen()
    assert len(cards) == len(load_universe().instruments)
    assert {c.symbol for c in cards} == set(load_universe().symbols)


def test_a_profile_reports_sector_and_category(service: ResearchService) -> None:
    card = service.profile("AAPL")
    assert card.sector
    assert card.category in {"EQUITY", "ETF"}


def test_current_weight_is_carried_through(service: ResearchService) -> None:
    assert service.profile("AAPL").current_weight == D("0.07")
    assert service.profile("MSFT").current_weight == D("0")


def test_research_reports_veto_history(service: ResearchService) -> None:
    # A name the risk engine has objected to should say so on its own page.
    assert "MAX_SECTOR_WEIGHT" in service.research("AAPL").veto_codes
    assert service.research("MSFT").veto_codes == ()


def test_an_instrument_without_filings_says_so_rather_than_faking_it(
    service: ResearchService,
) -> None:
    # No EDGAR cache is available here, so every name lands in this state —
    # which is the same state a REIT ETF is legitimately in.
    found = service.research("AAPL")
    assert found.ratios == {}
    assert found.valuation is None
    assert any("No SEC filings" in note for note in found.notes)


def test_research_never_invents_a_price(service: ResearchService) -> None:
    # A symbol in the universe with no recorded bars must report None, not zero.
    card = service.profile("VNQ")
    assert card.latest_price is None or card.latest_price > D("0")


def test_every_mapped_ratio_has_a_family() -> None:
    # A ratio with no family falls into "Other" in the UI, which is a silent
    # way to lose a whole group from the display.
    from src.agents.fundamental import ratio_table
    from src.data.edgar import Fundamentals
    from src.time.clock import UTC

    complete = Fundamentals(
        symbol="X",
        as_of=datetime(2024, 6, 1, tzinfo=UTC),
        period_end=datetime(2023, 12, 31, tzinfo=UTC),
        revenue=D("1000"), cost_of_goods_sold=D("600"), gross_profit=D("400"),
        operating_income=D("200"), interest_expense=D("50"), net_income=D("105"),
        cash_flow_operations=D("180"), total_assets=D("2000"), total_equity=D("800"),
        total_liabilities=D("1200"), current_assets=D("500"), current_liabilities=D("250"),
        inventory=D("200"), receivables=D("125"), cash=D("100"), long_term_debt=D("600"),
    )
    unmapped = set(ratio_table(complete)) - set(RATIO_FAMILIES)
    assert not unmapped, f"ratios with no UI family: {sorted(unmapped)}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    setup = resolve_setup(cache_root=tmp_path_factory.mktemp("empty2"))
    config = BacktestConfig(
        start=setup.start,
        end=setup.start + timedelta(days=560),
        initial_cash=D("100000.00"),
        symbols=setup.symbols,
        benchmark_symbol=setup.benchmark,
        estimation_window=100,
        market_return=setup.market_return,
        risk_free_rate=setup.risk_free_rate,
    )
    executor = SimulatedExecutor()
    result = run_backtest(
        config, setup.source, executor, load_policy(), setup.sectors, setup.betas
    )
    state = DashboardState.from_result(
        result,
        capabilities=executor.capabilities(),
        sectors=setup.sectors,
        research=ResearchService(
            setup=setup, universe=load_universe(), cache_root=Path("/nonexistent")
        ),
    )
    return TestClient(create_app(state))


def test_screen_endpoint(client: TestClient) -> None:
    body = client.get("/api/screen").json()
    assert body["count"] > 0
    assert body["sectors"]
    assert body["disclaimer"]


def test_research_endpoint(client: TestClient) -> None:
    body = client.get("/api/research/AAPL").json()
    assert body["profile"]["symbol"] == "AAPL"
    assert "notes" in body


def test_research_is_case_insensitive(client: TestClient) -> None:
    assert client.get("/api/research/aapl").status_code == 200


def test_an_unknown_symbol_is_a_404_not_an_empty_page(client: TestClient) -> None:
    response = client.get("/api/research/NOTREAL")
    assert response.status_code == 404
    assert "universe" in response.json()["detail"]


def test_research_endpoints_are_read_only(client: TestClient) -> None:
    for path in ("/api/screen", "/api/research/AAPL"):
        assert client.post(path).status_code in (404, 405)
        assert client.delete(path).status_code in (404, 405)


def test_research_values_cross_the_wire_as_strings(client: TestClient) -> None:
    body = client.get("/api/research/AAPL").json()
    profile = body["profile"]
    for key in ("current_weight", "beta", "latest_price"):
        assert profile[key] is None or isinstance(profile[key], str)
