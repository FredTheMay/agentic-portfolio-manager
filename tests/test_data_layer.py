"""Data layer: caching, offline replay, EDGAR, FRED, and market data sources (M2).

The whole suite runs with no network and no API keys. Every client takes its
fetcher by injection, so these tests exercise the real parsing and
point-in-time logic against recorded payload shapes rather than mocking it away.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.data import edgar, fred, sources
from src.data.cache import (
    CachingFetcher,
    FetchError,
    OfflineError,
    OfflineFetcher,
    ResponseCache,
    StubFetcher,
    cache_key,
    dumps,
    loads,
)
from src.data.events import BarPayload, MarketEvent
from src.time.clock import UTC

D = Decimal


def at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# ===========================================================================
# Cache and the network boundary
# ===========================================================================


def test_json_numbers_are_parsed_as_decimal() -> None:
    # SPEC §9: a price arriving as the JSON literal 0.1 must be exactly 0.1,
    # not the nearest binary double.
    parsed = loads('{"price": 0.1}')
    assert isinstance(parsed["price"], Decimal)
    assert parsed["price"] == D("0.1")
    # The value a naive float parse would have produced is a different number.
    assert parsed["price"] != D(0.1)


def test_decimals_survive_a_cache_round_trip_as_numbers() -> None:
    # Serializing a Decimal as a quoted string would make the replayed value a
    # different type from the live one, and the two paths would diverge.
    encoded = dumps({"price": D("0.1"), "volume": 1000})
    assert encoded == '{"price":0.1,"volume":1000}'
    assert loads(encoded) == {"price": D("0.1"), "volume": 1000}


def test_decimal_exponent_notation_is_normalized() -> None:
    assert dumps({"v": D("1E+2")}) == '{"v":100}'


def test_json_integers_stay_integers() -> None:
    parsed = loads('{"volume": 1000000}')
    assert isinstance(parsed["volume"], int)


def test_dumps_is_canonical() -> None:
    # Key order must not depend on dict insertion order, so a diff on a cache
    # file shows a data change rather than a reshuffle.
    assert dumps({"b": 1, "a": 2}) == dumps({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_cache_key_is_stable_across_param_order() -> None:
    assert cache_key("https://x/y", {"a": "1", "b": "2"}) == cache_key(
        "https://x/y", {"b": "2", "a": "1"}
    )


def test_cache_key_distinguishes_different_requests() -> None:
    assert cache_key("https://x/y", {"a": "1"}) != cache_key("https://x/y", {"a": "2"})
    assert cache_key("https://x/y") != cache_key("https://x/z")


def test_cache_round_trips_a_payload(tmp_path: Path) -> None:
    cache = ResponseCache(root=tmp_path)
    key = cache_key("https://example.test/data")

    assert cache.get(key) is None
    assert not cache.has(key)

    cache.put(key, {"value": D("1.25")})
    assert cache.has(key)
    assert cache.get(key) == {"value": D("1.25")}


def test_cache_leaves_no_temporary_files(tmp_path: Path) -> None:
    # Write-then-rename: an interrupted run must not leave a truncated file
    # that later parses as valid but incomplete data.
    cache = ResponseCache(root=tmp_path)
    cache.put(cache_key("https://example.test/a"), {"a": 1})
    assert not list(tmp_path.rglob("*.tmp"))


def test_offline_fetcher_refuses_everything() -> None:
    with pytest.raises(OfflineError):
        OfflineFetcher().get_json("https://example.test/data")


def test_caching_fetcher_records_then_replays(tmp_path: Path) -> None:
    live = StubFetcher({"https://example.test/data": {"value": 1}})
    fetcher = CachingFetcher(ResponseCache(root=tmp_path), inner=live)

    first = fetcher.get_json("https://example.test/data")
    second = fetcher.get_json("https://example.test/data")

    assert first == second == {"value": 1}
    assert len(live.calls) == 1, "the second call must be served from cache"
    assert (fetcher.hits, fetcher.misses) == (1, 1)


def test_a_recorded_cache_replays_with_no_network(tmp_path: Path) -> None:
    # SPEC §9: two identical runs produce identical output. Record once...
    cache = ResponseCache(root=tmp_path)
    recorder = CachingFetcher(cache, inner=StubFetcher({"https://example.test/d": {"v": 2}}))
    recorder.get_json("https://example.test/d")

    # ...then replay with no live fetcher at all.
    replay = CachingFetcher(cache, offline=True)
    assert replay.get_json("https://example.test/d") == {"v": 2}


def test_offline_mode_fails_loudly_on_a_cache_miss(tmp_path: Path) -> None:
    # A missing input must not become a silent gap the optimizer interpolates.
    replay = CachingFetcher(ResponseCache(root=tmp_path), offline=True)
    with pytest.raises(OfflineError, match="not cached"):
        replay.get_json("https://example.test/never-fetched")


def test_caching_fetcher_requires_a_source_of_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CachingFetcher(ResponseCache(root=tmp_path))


def test_stub_fetcher_reports_unknown_urls() -> None:
    with pytest.raises(FetchError):
        StubFetcher({}).get_json("https://example.test/missing")


def test_malformed_json_is_reported_as_a_fetch_error() -> None:
    with pytest.raises(FetchError):
        loads("{not json")


# ===========================================================================
# EDGAR
# ===========================================================================

TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

# One tag, two annual filings. FY2023 describes a period ending 2023-12-31 but
# is not filed until 2024-02-15.
COMPANY_FACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2022-12-31", "val": 90, "filed": "2023-02-14", "form": "10-K"},
                        {"end": "2023-12-31", "val": 105, "filed": "2024-02-15", "form": "10-K"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2023-12-31", "val": 2000, "filed": "2024-02-15", "form": "10-K"},
                    ]
                }
            },
        }
    },
}


def edgar_client() -> edgar.EdgarClient:
    return edgar.EdgarClient(
        StubFetcher(
            {
                edgar.TICKER_MAP_URL: TICKER_MAP,
                edgar.COMPANY_FACTS_URL.format(cik=320193): COMPANY_FACTS,
            }
        )
    )


def test_edgar_resolves_a_ticker_to_a_cik() -> None:
    assert edgar_client().resolve_cik("aapl") == 320193


def test_edgar_reports_an_unknown_ticker() -> None:
    with pytest.raises(edgar.EdgarError, match="no CIK"):
        edgar_client().resolve_cik("NOTREAL")


def test_edgar_indexes_by_filing_date_not_period_end() -> None:
    # The M2 restatement of the §4.4 rule, now through the real parser.
    client = edgar_client()

    january = client.get_fundamentals("AAPL", at(2024, 1, 15))
    assert january is not None
    assert january.net_income == D("90"), "FY2023 was not filed until February"

    march = client.get_fundamentals("AAPL", at(2024, 3, 1))
    assert march is not None
    assert march.net_income == D("105")


def test_edgar_applies_a_publication_lag() -> None:
    # A filing dated 2024-02-15 is treated as public at the *end* of that day,
    # so it is not visible at midnight when the US market has not yet opened.
    client = edgar_client()

    same_day = client.get_fundamentals("AAPL", at(2024, 2, 15))
    assert same_day is not None
    assert same_day.net_income == D("90")

    next_day = client.get_fundamentals("AAPL", at(2024, 2, 16))
    assert next_day is not None
    assert next_day.net_income == D("105")


def test_edgar_publication_lag_is_configurable() -> None:
    eager = edgar.EdgarClient(
        StubFetcher(
            {
                edgar.TICKER_MAP_URL: TICKER_MAP,
                edgar.COMPANY_FACTS_URL.format(cik=320193): COMPANY_FACTS,
            }
        ),
        publication_lag=timedelta(0),
    )
    same_day = eager.get_fundamentals("AAPL", at(2024, 2, 15))
    assert same_day is not None and same_day.net_income == D("105")


def test_edgar_returns_none_before_any_filing() -> None:
    assert edgar_client().get_fundamentals("AAPL", at(2019, 1, 1)) is None


def test_edgar_reports_the_latest_visible_period() -> None:
    snapshot = edgar_client().get_fundamentals("AAPL", at(2024, 3, 1))
    assert snapshot is not None
    assert snapshot.period_end == at(2023, 12, 31)


def test_edgar_leaves_untagged_fields_as_none() -> None:
    # A missing figure must stay distinguishable from a zero.
    snapshot = edgar_client().get_fundamentals("AAPL", at(2024, 3, 1))
    assert snapshot is not None
    assert snapshot.revenue is None
    assert snapshot.total_assets == D("2000")


def test_edgar_skips_facts_without_a_filing_date() -> None:
    # No filing date means no evidence of when it became public, so it cannot
    # be shown to have been available and must not be usable.
    facts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {"USD": [{"end": "2023-12-31", "val": 105}]}
                }
            }
        }
    }
    series = edgar.concept_series(facts, "NetIncomeLoss")
    assert len(series) == 0


def test_edgar_concept_series_is_empty_for_an_absent_tag() -> None:
    assert len(edgar.concept_series(COMPANY_FACTS, "NoSuchTag")) == 0


def test_edgar_amounts_never_pass_through_binary_float() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "X": {"units": {"USD": [{"end": "2023-12-31", "val": D("0.1"), "filed": "2024-02-15"}]}}
            }
        }
    }
    vintage = edgar.concept_series(facts, "X").as_of(at(2024, 3, 1))
    assert vintage is not None
    assert vintage.value == D("0.1")


def test_edgar_average_requires_both_periods() -> None:
    assert edgar.average(D("100"), D("200")) == D("150")
    assert edgar.average(D("100"), None) is None
    assert edgar.average(None, None) is None


def test_edgar_total_debt_prefers_long_term_debt() -> None:
    with_debt = edgar.Fundamentals(
        symbol="X", as_of=at(2024, 1, 1), period_end=None,
        long_term_debt=D("600"), total_liabilities=D("900"),
    )
    without = edgar.Fundamentals(
        symbol="X", as_of=at(2024, 1, 1), period_end=None, total_liabilities=D("900"),
    )
    assert edgar.total_debt(with_debt) == D("600")
    assert edgar.total_debt(without) == D("900")


# ===========================================================================
# FRED
# ===========================================================================

# Q1 2024 GDP: released 2024-04-25 at 2.1, revised 2024-05-30 down to 1.6.
GDP_VINTAGES = {
    "observations": [
        {"realtime_start": "2024-04-25", "date": "2024-01-01", "value": "2.1"},
        {"realtime_start": "2024-05-30", "date": "2024-01-01", "value": "1.6"},
    ]
}


def test_fred_revision_does_not_leak_backwards() -> None:
    series = fred.parse_observations(GDP_VINTAGES)

    april = series.as_of(at(2024, 4, 30))
    assert april is not None and april.value == D("2.1")

    june = series.as_of(at(2024, 6, 1))
    assert june is not None and june.value == D("1.6")


def test_fred_drops_missing_observations() -> None:
    # "." is FRED's sentinel. It must not become a zero a regression believes.
    payload = {
        "observations": [
            {"realtime_start": "2024-01-05", "date": "2024-01-01", "value": "."},
            {"realtime_start": "2024-02-05", "date": "2024-02-01", "value": "3.5"},
        ]
    }
    series = fred.parse_observations(payload)
    assert len(series) == 1
    visible = series.as_of(at(2024, 3, 1))
    assert visible is not None and visible.value == D("3.5")


def test_fred_skips_observations_without_a_realtime_date() -> None:
    payload = {"observations": [{"date": "2024-01-01", "value": "3.5"}]}
    assert len(fred.parse_observations(payload)) == 0


def test_fred_rejects_a_response_without_observations() -> None:
    with pytest.raises(fred.FredError):
        fred.parse_observations({"error": "bad key"})


def test_fred_client_reads_a_value_as_of() -> None:
    # UNRATE is in REVISED_SERIES, so it is requested across the full real-time
    # window and each revision arrives as its own row.
    canonical = (
        f"{fred.OBSERVATIONS_URL}?file_type=json&output_type=1"
        f"&realtime_end={fred.ALL_REALTIME_END}"
        f"&realtime_start={fred.ALL_REALTIME_START}&series_id=UNRATE"
    )
    client = fred.FredClient(StubFetcher({canonical: GDP_VINTAGES}), api_key=None)
    assert client.value_as_of(fred.UNEMPLOYMENT, at(2024, 4, 30)) == D("2.1")
    assert client.value_as_of(fred.UNEMPLOYMENT, at(2024, 6, 1)) == D("1.6")


def test_fred_api_key_is_not_part_of_the_request_when_absent() -> None:
    # A cached series must replay with no key at all.
    canonical = (
        f"{fred.OBSERVATIONS_URL}?file_type=json&output_type=1"
        f"&realtime_end={fred.ALL_REALTIME_END}"
        f"&realtime_start={fred.ALL_REALTIME_START}&series_id=UNRATE"
    )
    stub = StubFetcher(
        {
            canonical: {
                "observations": [
                    {"realtime_start": "2024-02-02", "date": "2024-01-01", "value": "3.7"}
                ]
            }
        }
    )
    client = fred.FredClient(stub, api_key=None)
    assert client.value_as_of(fred.UNEMPLOYMENT, at(2024, 3, 1)) == D("3.7")


def test_year_over_year_change_needs_a_full_year_of_visible_history() -> None:
    payload = {
        "observations": [
            {"realtime_start": f"2024-{month:02d}-05", "date": f"2024-{month:02d}-01", "value": "100"}
            for month in range(1, 7)
        ]
    }
    series = fred.parse_observations(payload)
    assert fred.year_over_year_change(series, at(2024, 7, 1)) is None


def test_year_over_year_change_computes_from_visible_observations() -> None:
    records = [
        {"realtime_start": "2023-01-05", "date": "2023-01-01", "value": "100"},
    ]
    records += [
        {"realtime_start": f"2023-{month:02d}-05", "date": f"2023-{month:02d}-01", "value": "100"}
        for month in range(2, 13)
    ]
    records.append({"realtime_start": "2024-01-05", "date": "2024-01-01", "value": "103"})
    series = fred.parse_observations({"observations": records})

    change = fred.year_over_year_change(series, at(2024, 2, 1))
    assert change is not None
    assert change == D("0.03")


# ===========================================================================
# Market data sources
# ===========================================================================

ALPACA_PAYLOAD = {
    "bars": {
        "SPY": [
            {"t": "2024-01-03T21:00:00Z", "o": 470, "h": 472, "l": 469, "c": 471, "v": 1000},
            {"t": "2024-01-04T21:00:00Z", "o": 471, "h": 474, "l": 470, "c": 473, "v": 1100},
        ],
        "AGG": [
            {"t": "2024-01-03T21:00:00Z", "o": 97, "h": 98, "l": 96, "c": 97, "v": 500},
        ],
    }
}


def test_alpaca_payload_becomes_ordered_events() -> None:
    source = sources.InMemoryEventSource.from_alpaca_payload(ALPACA_PAYLOAD)
    stamps = [event.timestamp for event in source.events]
    assert stamps == sorted(stamps)
    assert source.symbols() == {"SPY", "AGG"}


def test_events_from_separate_symbols_are_interleaved_by_time() -> None:
    # The engine consumes one ordered stream, not one stream per symbol.
    source = sources.InMemoryEventSource.from_alpaca_payload(ALPACA_PAYLOAD)
    first_day = [e for e in source.events if e.timestamp == at(2024, 1, 3).replace(hour=21)]
    assert {e.symbol for e in first_day} == {"SPY", "AGG"}


def test_stream_respects_its_window() -> None:
    source = sources.InMemoryEventSource.from_alpaca_payload(ALPACA_PAYLOAD)
    window = list(source.stream(at(2024, 1, 4), at(2024, 1, 5)))
    assert len(window) == 1
    assert window[0].symbol == "SPY"


def test_stream_rejects_an_inverted_window() -> None:
    source = sources.InMemoryEventSource.from_alpaca_payload(ALPACA_PAYLOAD)
    with pytest.raises(sources.SourceError):
        list(source.stream(at(2024, 1, 5), at(2024, 1, 1)))


def test_source_satisfies_the_market_data_protocol() -> None:
    from src.data.events import MarketDataSource

    assert isinstance(sources.InMemoryEventSource.from_events([]), MarketDataSource)


def test_bare_dates_are_stamped_at_the_session_close() -> None:
    event = sources.bar_event("SPY", {"t": "2024-01-03", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1})
    assert event.timestamp == datetime(2024, 1, 3, 21, tzinfo=UTC)


def test_adjusted_close_is_absent_unless_supplied() -> None:
    # SPEC §4.4: an absent adjustment must not be assumed equal to the raw close.
    plain = sources.bar_event("SPY", {"t": "2024-01-03", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1})
    assert isinstance(plain.payload, BarPayload)
    assert plain.payload.adj_close is None

    adjusted = sources.bar_event(
        "SPY", {"t": "2024-01-03", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1, "ac": "1.5"}
    )
    assert isinstance(adjusted.payload, BarPayload)
    assert adjusted.payload.adj_close == D("1.5")


def test_bar_prices_are_decimal_not_float() -> None:
    event = sources.bar_event("SPY", {"t": "2024-01-03", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1})
    assert isinstance(event.payload, BarPayload)
    assert isinstance(event.payload.close, Decimal)


def test_malformed_bars_are_rejected() -> None:
    with pytest.raises(sources.SourceError, match="missing field"):
        sources.bar_event("SPY", {"t": "2024-01-03", "o": 1})
    with pytest.raises(sources.SourceError):
        # low above high is not a bar.
        sources.bar_event("SPY", {"t": "2024-01-03", "o": 1, "h": 1, "l": 9, "c": 1, "v": 1})
    with pytest.raises(sources.SourceError):
        sources.bar_event("SPY", {"t": "nonsense", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1})


def test_alpaca_response_shape_is_validated() -> None:
    with pytest.raises(sources.SourceError):
        sources.events_from_alpaca_payload({})
    with pytest.raises(sources.SourceError):
        sources.events_from_alpaca_payload({"bars": {"SPY": "not a list"}})


def test_assert_ordered_catches_an_inverted_feed() -> None:
    def bar(day: int) -> MarketEvent:
        return sources.bar_event(
            "SPY", {"t": f"2024-01-{day:02d}", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1}
        )

    assert len(sources.assert_ordered([bar(3), bar(4)])) == 2
    with pytest.raises(sources.SourceError, match="out of order"):
        sources.assert_ordered([bar(4), bar(3)])


def test_merge_ordered_preserves_ordering() -> None:
    left = sources.InMemoryEventSource.from_alpaca_payload({"bars": {"SPY": ALPACA_PAYLOAD["bars"]["SPY"]}})
    right = sources.InMemoryEventSource.from_alpaca_payload({"bars": {"AGG": ALPACA_PAYLOAD["bars"]["AGG"]}})

    merged = list(
        sources.merge_ordered(
            [left.stream(at(2024, 1, 1), at(2024, 2, 1)), right.stream(at(2024, 1, 1), at(2024, 2, 1))]
        )
    )
    stamps = [event.timestamp for event in merged]
    assert stamps == sorted(stamps)
    assert len(merged) == 3


def test_alpaca_client_requests_adjusted_bars() -> None:
    stub = StubFetcher(
        {
            f"{sources.ALPACA_BARS_URL}?adjustment=all&end=2024-01-05&feed=iex"
            f"&limit={sources.ALPACA_PAGE_LIMIT}&start=2024-01-01"
            f"&symbols=AGG,SPY&timeframe=1Day": ALPACA_PAYLOAD
        }
    )
    client = sources.AlpacaBarClient(stub)
    source = client.daily_bars(["SPY", "AGG"], at(2024, 1, 1), at(2024, 1, 5))
    assert len(source.events) == 3


def test_alpaca_client_requires_symbols() -> None:
    with pytest.raises(sources.SourceError):
        sources.AlpacaBarClient(StubFetcher({})).daily_bars([], at(2024, 1, 1), at(2024, 1, 5))


# ===========================================================================
# Offline replay end to end
# ===========================================================================


def test_the_whole_data_layer_replays_offline(tmp_path: Path) -> None:
    # Record once against a stub standing in for the live APIs...
    cache = ResponseCache(root=tmp_path)
    live = StubFetcher(
        {
            edgar.TICKER_MAP_URL: TICKER_MAP,
            edgar.COMPANY_FACTS_URL.format(cik=320193): COMPANY_FACTS,
        }
    )
    recording = CachingFetcher(cache, inner=live)
    recorded = edgar.EdgarClient(recording).get_fundamentals("AAPL", at(2024, 3, 1))

    # ...then replay with no live fetcher and no keys.
    replayed = edgar.EdgarClient(CachingFetcher(cache, offline=True)).get_fundamentals(
        "AAPL", at(2024, 3, 1)
    )

    assert recorded is not None and replayed is not None
    assert recorded == replayed, "SPEC §9: identical inputs, identical output"


def test_edgar_contact_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    # SEC EDGAR's fair-access policy requires a real contact address and
    # throttles traffic without one. Hardcoding a placeholder meant every live
    # run would have been throttled with no way to fix it but editing source.
    from src.data.cache import HttpxFetcher, default_user_agent, user_agent_is_configured

    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    assert not user_agent_is_configured()
    assert "set EDGAR_USER_AGENT" in default_user_agent()

    monkeypatch.setenv("EDGAR_USER_AGENT", "Jane Doe jane@example.com")
    assert user_agent_is_configured()
    assert default_user_agent() == "Jane Doe jane@example.com"
    assert HttpxFetcher()._headers["User-Agent"] == "Jane Doe jane@example.com"


def test_alpaca_requests_carry_authentication_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Alpaca authenticates market data by header. EDGAR identifies callers by
    # User-Agent and FRED takes a query parameter, so this was the one vendor
    # the shared fetcher could not reach — every live call returned 401.
    from src.data.cache import HttpxFetcher
    from src.data.sources import SourceError, alpaca_headers

    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(SourceError, match="no Alpaca credentials"):
        alpaca_headers()

    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKTEST")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    headers = alpaca_headers()
    assert headers == {"APCA-API-KEY-ID": "PKTEST", "APCA-API-SECRET-KEY": "secret"}

    fetcher = HttpxFetcher(extra_headers=headers)
    assert fetcher._headers["APCA-API-KEY-ID"] == "PKTEST"
    # The EDGAR contact must survive alongside the vendor headers.
    assert "User-Agent" in fetcher._headers


def test_revised_and_unrevised_series_are_fetched_differently() -> None:
    # FRED caps a response at ~2000 vintage dates. A daily series has one per
    # business day and is rejected with a 400 if the full real-time window is
    # requested, so daily rates are fetched at current real time and dated by
    # observation. Revised monthly series get the full window.
    from src.data.fred import REVISED_SERIES, CPI, THREE_MONTH_TREASURY, FredClient

    assert CPI in REVISED_SERIES
    assert THREE_MONTH_TREASURY not in REVISED_SERIES

    seen: list[dict[str, str]] = []

    class RecordingFetcher:
        def get_json(self, url: str, params: dict[str, str] | None = None) -> object:
            seen.append(dict(params or {}))
            return {"observations": []}

    client = FredClient(RecordingFetcher(), api_key=None)  # type: ignore[arg-type]
    client.series(CPI)
    client.series(THREE_MONTH_TREASURY)

    assert "realtime_start" in seen[0], "a revised series needs the full vintage window"
    assert "realtime_start" not in seen[1], "a daily series must not request it (400)"


def test_an_unrevised_series_is_dated_by_observation() -> None:
    # The response for these carries today's real-time window, not the original
    # publication date, so realtime_start would wrongly read "published today"
    # and every historical value would look invisible until now.
    from datetime import timedelta as _td

    from src.data.fred import parse_observations

    payload = {
        "observations": [
            {"realtime_start": "2026-08-18", "realtime_end": "2026-08-18",
             "date": "2020-06-01", "value": "0.14"},
        ]
    }
    series = parse_observations(payload, publication_lag=_td(days=1))
    assert series.as_of(at(2020, 6, 3)) is not None, "must be visible shortly after its date"
    assert series.as_of(at(2020, 5, 1)) is None, "must not be visible before its date"


def test_alpaca_bars_are_paged_to_completion() -> None:
    # Alpaca caps a page at 1000 bars and returns next_page_token instead of
    # the rest. Ignoring the token truncates a multi-symbol multi-year request
    # to the first symbol or two — which reads as "this universe has no data".
    # Cache keys canonicalize by sorting params, so page_token sorts between
    # `limit` and `start` rather than being appended.
    def url(token: str | None = None) -> str:
        parts = {
            "adjustment": "all",
            "end": "2024-01-05",
            "feed": "iex",
            "limit": str(sources.ALPACA_PAGE_LIMIT),
            "start": "2024-01-01",
            "symbols": "AGG,SPY",
            "timeframe": "1Day",
        }
        if token:
            parts["page_token"] = token
        query = "&".join(f"{k}={parts[k]}" for k in sorted(parts))
        return f"{sources.ALPACA_BARS_URL}?{query}"

    page_one = {
        "bars": {"SPY": [ALPACA_PAYLOAD["bars"]["SPY"][0]]},
        "next_page_token": "TOKEN2",
    }
    page_two = {
        "bars": {"AGG": ALPACA_PAYLOAD["bars"]["AGG"]},
        "next_page_token": None,
    }
    stub = StubFetcher({url(): page_one, url("TOKEN2"): page_two})

    source = sources.AlpacaBarClient(stub).daily_bars(
        ["SPY", "AGG"], at(2024, 1, 1), at(2024, 1, 5)
    )
    assert source.symbols() == {"SPY", "AGG"}, "the second page must be fetched"
    assert len(stub.calls) == 2


def test_alpaca_paging_terminates() -> None:
    # A token loop that never ends must fail loudly rather than hang.
    looping = {"bars": {}, "next_page_token": "SAME"}

    class LoopingFetcher:
        def get_json(self, url: str, params: dict[str, str] | None = None) -> object:
            return looping

    with pytest.raises(sources.SourceError, match="did not terminate"):
        sources.AlpacaBarClient(LoopingFetcher()).daily_bars(  # type: ignore[arg-type]
            ["SPY"], at(2024, 1, 1), at(2024, 1, 5)
        )


def test_sec_and_alpaca_disagree_on_class_share_tickers() -> None:
    # SEC's ticker file has BRK-B where Alpaca sends BRK.B. Same instrument.
    client = edgar.EdgarClient(
        StubFetcher(
            {
                edgar.TICKER_MAP_URL: {
                    "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"}
                }
            }
        )
    )
    assert client.resolve_cik("BRK.B") == 1067983
    assert client.resolve_cik("BRK-B") == 1067983
