"""Universe loading and real-data assembly (SPEC §4.2, §6.2, §6.3).

`config/universe.yaml` was written at M2 as the record of the survivorship
limitation and then went unused for eight milestones. These tests cover it and
the module that turns recorded market data into a backtest.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.data.events import BarPayload, MarketEvent
from src.data.live import (
    BetaEstimate,
    adjusted_series,
    estimate_betas,
    market_return,
    risk_free_rate,
    to_returns,
)
from src.data.sources import InMemoryEventSource
from src.data.universe import (
    DEFAULT_UNIVERSE_PATH,
    UniverseError,
    equity_symbols,
    load_universe,
    universe_from_document,
)
from src.time.clock import UTC

D = Decimal
NOW = datetime(2024, 6, 3, 21, tzinfo=UTC)


def document() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(DEFAULT_UNIVERSE_PATH.read_text(encoding="utf-8")))


# ===========================================================================
# Universe
# ===========================================================================


def test_the_shipped_universe_loads() -> None:
    universe = load_universe()
    assert len(universe.instruments) >= 10
    assert universe.benchmark_equity == "SPY"
    assert universe.benchmark_bonds == "AGG"


def test_the_universe_is_large_enough_for_the_position_cap() -> None:
    # The IPS caps any name at 10%, so a fully invested portfolio needs at
    # least ten holdings. A smaller universe makes the constrained frontier
    # infeasible — which cost an entire silent backtest at M6.
    assert len(load_universe().tradable()) >= 10


def test_the_benchmark_is_not_itself_a_holding() -> None:
    universe = load_universe()
    assert universe.benchmark_equity not in universe.tradable()
    # ...but it still needs price history, to measure against.
    assert universe.benchmark_equity in universe.fetch_list()


def test_survivorship_bias_is_carried_on_the_universe() -> None:
    # SPEC §4.4 requires this stated, not discovered.
    universe = load_universe()
    assert universe.survivorship_biased is True
    assert universe.survivorship_reason


def test_equity_symbols_excludes_etfs() -> None:
    # EDGAR has filings for operating companies, not for index ETFs.
    universe = load_universe()
    equities = set(equity_symbols(universe))
    assert "SPY" not in equities
    assert "AAPL" in equities


def test_sectors_cover_every_instrument() -> None:
    universe = load_universe()
    assert set(universe.sectors) == set(universe.symbols)


def test_a_symbol_cannot_be_listed_and_excluded() -> None:
    doc = document()
    doc["exclusions"].append({"symbol": "AAPL", "reason": "contradiction"})
    with pytest.raises(UniverseError, match="listed and excluded"):
        universe_from_document(doc)


def test_a_duplicate_listing_is_rejected() -> None:
    doc = document()
    doc["equities"].append({"symbol": "AAPL", "sector": "INFORMATION_TECHNOLOGY"})
    with pytest.raises(UniverseError, match="listed twice"):
        universe_from_document(doc)


def test_benchmark_weights_must_sum_to_one() -> None:
    doc = document()
    doc["benchmark"]["bond_weight"] = "0.30"
    with pytest.raises(UniverseError, match="sum to 1"):
        universe_from_document(doc)


def test_an_empty_universe_is_rejected() -> None:
    doc = document()
    doc["equities"], doc["etfs"] = [], []
    with pytest.raises(UniverseError, match="no instruments"):
        universe_from_document(doc)


def test_a_missing_file_is_reported() -> None:
    with pytest.raises(UniverseError, match="not found"):
        load_universe(Path("/nonexistent/universe.yaml"))


# ===========================================================================
# Live data assembly
# ===========================================================================


def bars(symbol: str, prices: list[str]) -> list[MarketEvent]:
    return [
        MarketEvent(
            timestamp=NOW + timedelta(days=index),
            symbol=symbol,
            kind="BAR",
            payload=BarPayload(
                open=D(price), high=D(price), low=D(price), close=D(price),
                volume=1000, adj_close=D(price),
            ),
        )
        for index, price in enumerate(prices)
    ]


def test_returns_are_computed_from_adjusted_prices() -> None:
    # SPEC §4.4: adjusted for returns, unadjusted for share arithmetic.
    source = InMemoryEventSource.from_events(bars("AAA", ["100", "110", "99"]))
    series = adjusted_series(source)
    assert to_returns(series["AAA"]) == [D("0.1"), D("-0.1")]


def test_beta_is_estimated_by_regression() -> None:
    # A symbol moving exactly 1.5x the benchmark has beta 1.5 and R-squared 1.
    market = ["100", "102", "101", "103", "102", "104"]
    levered = ["100", "103", "101.5", "104.5", "103", "106"]
    source = InMemoryEventSource.from_events(bars("SPY", market) + bars("AAA", levered))

    estimates = estimate_betas(source, "SPY", risk_free_rate=D("0"))
    assert "AAA" in estimates
    assert abs(estimates["AAA"].beta - D("1.5")) < D("0.05")
    assert estimates["AAA"].r_squared > D("0.99")


def test_a_symbol_with_mismatched_history_gets_no_beta() -> None:
    # A fabricated 1.0 would flow straight into the beta constraint and the
    # CAPM expected return. Skipping is the honest answer.
    source = InMemoryEventSource.from_events(
        bars("SPY", ["100", "101", "102", "103"]) + bars("AAA", ["50", "51"])
    )
    assert "AAA" not in estimate_betas(source, "SPY", D("0"))


def test_estimating_betas_needs_the_benchmark() -> None:
    from src.data.live import LiveDataError

    source = InMemoryEventSource.from_events(bars("AAA", ["100", "101", "102"]))
    with pytest.raises(LiveDataError, match="benchmark"):
        estimate_betas(source, "SPY", D("0"))


def test_the_risk_free_rate_falls_back_when_nothing_is_recorded() -> None:
    # A run without FRED completes on a stated assumption rather than crashing.
    rate = risk_free_rate(NOW, cache_root=Path("/nonexistent"), fallback=D("0.04"))
    assert rate == D("0.04")


def test_the_market_return_assumption_is_a_premium_over_the_risk_free_rate() -> None:
    # Stated, not buried: realized premia are far too noisy to estimate a
    # forward-looking mean from, which is why CAPM inputs are used at all.
    estimates: dict[str, BetaEstimate] = {}
    assert market_return(estimates, D("0.04")) == D("0.09")
    assert market_return(estimates, D("0.02")) == D("0.07")


# ===========================================================================
# The shared setup resolver
# ===========================================================================


def test_the_resolver_falls_back_to_synthetic_and_labels_it(tmp_path: Path) -> None:
    # Three callers need the same "real if recorded, else synthetic" decision.
    # Three copies of it would eventually disagree about which they were showing.
    from src.data.live import resolve_setup

    setup = resolve_setup(cache_root=tmp_path)
    assert setup.is_real is False
    assert "synthetic" in setup.data_source
    assert len(setup.symbols) >= 10, "the 10% position cap needs ten holdings"


def test_the_synthetic_fallback_is_self_consistent(tmp_path: Path) -> None:
    from src.data.live import resolve_setup

    setup = resolve_setup(cache_root=tmp_path)
    assert set(setup.symbols) <= set(setup.betas)
    assert set(setup.symbols) <= set(setup.sectors)
    assert setup.end > setup.start


def test_credentials_never_decide_a_cache_key() -> None:
    # A replay under a different (or absent) key must hit the same entries.
    # Before this, every keyless replay missed and silently fell back to
    # synthetic data while reporting real.
    from src.data.cache import cache_key

    with_key = cache_key("https://api.test/x", {"series_id": "DGS3MO", "api_key": "SECRET"})
    without = cache_key("https://api.test/x", {"series_id": "DGS3MO"})
    other_key = cache_key("https://api.test/x", {"series_id": "DGS3MO", "api_key": "OTHER"})

    assert with_key == without == other_key


def test_a_credential_never_appears_in_an_error_message() -> None:
    # The canonical request form is quoted verbatim in OfflineError, which ends
    # up in logs, tracebacks and CI output.
    from src.data.cache import CachingFetcher, OfflineError, ResponseCache
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fetcher = CachingFetcher(ResponseCache(root=Path(tmp)), offline=True)
        try:
            fetcher.get_json("https://api.test/x", {"api_key": "SUPERSECRET", "id": "A"})
        except OfflineError as exc:
            assert "SUPERSECRET" not in str(exc)
            assert "id=A" in str(exc), "non-sensitive params should still be shown"
        else:
            raise AssertionError("expected an OfflineError")
