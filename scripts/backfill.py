"""Record real market data into the cache. THE only script that fetches.

Everything else in the system reads through the cache offline, so this runs
once (and again whenever you extend the window). Re-running is cheap: cached
responses are served from disk and never refetched.

    make backfill

Requires EDGAR_USER_AGENT, and ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY for
prices. FRED_API_KEY is optional — without it the risk-free rate falls back to
a stated assumption rather than the real curve.

Partial success is the expected outcome on a first run and is reported per
service rather than aborting: EDGAR tag coverage in particular varies by filer,
and knowing *which* symbols resolved is what makes the gap fixable.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data.cache import CachingFetcher, HttpxFetcher, ResponseCache, user_agent_is_configured
from src.data.edgar import CONCEPT_TAGS, EdgarClient
from src.data.fred import CPI, FED_FUNDS, TERM_SPREAD, THREE_MONTH_TREASURY, UNEMPLOYMENT
from src.data.fred import FredClient
from src.data.live import DEFAULT_CACHE_ROOT
from src.data.sources import AlpacaBarClient, alpaca_headers, live_alpaca_fetcher
from src.data.universe import equity_symbols, load_universe

#: Enough history for a 100-session estimation window plus a walk-forward span.
DEFAULT_YEARS = 3


def recording_fetcher(root: Path, extra_headers: dict[str, str] | None = None) -> CachingFetcher:
    return CachingFetcher(
        ResponseCache(root=root), inner=HttpxFetcher(extra_headers=extra_headers)
    )


def backfill_prices(root: Path, years: int) -> bool:
    universe = load_universe()
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=365 * years + 30)

    try:
        alpaca_headers()
    except Exception as exc:  # noqa: BLE001
        print(f"  prices  SKIPPED  {exc}")
        return False

    try:
        source = AlpacaBarClient(live_alpaca_fetcher(cache_root=root)).daily_bars(
            list(universe.fetch_list()), start, end
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  prices  FAILED   {type(exc).__name__}: {exc}")
        return False

    got = source.symbols()
    missing = sorted(set(universe.fetch_list()) - got)
    print(f"  prices  OK       {len(source.events)} bars, {len(got)} symbols")
    if missing:
        print(f"                   no data for: {', '.join(missing)}")
    return True


def backfill_fundamentals(root: Path) -> bool:
    if not user_agent_is_configured():
        print("  edgar   SKIPPED  set EDGAR_USER_AGENT to 'Name email@example.com'")
        return False

    universe = load_universe()
    client = EdgarClient(recording_fetcher(root))
    as_of = datetime.now(timezone.utc)

    resolved, failures, coverage = 0, [], []
    for symbol in equity_symbols(universe):
        try:
            fundamentals = client.get_fundamentals(symbol, as_of)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{symbol} ({type(exc).__name__})")
            continue
        if fundamentals is None:
            failures.append(f"{symbol} (no visible filings)")
            continue
        resolved += 1
        populated = sum(
            1 for field in CONCEPT_TAGS if getattr(fundamentals, field, None) is not None
        )
        coverage.append(populated)

    total = len(equity_symbols(universe))
    print(f"  edgar   OK       {resolved}/{total} symbols resolved")
    if coverage:
        # Low coverage means CONCEPT_TAGS is missing tags these filers use —
        # the most likely first failure with real data.
        print(
            f"                   fields populated: min {min(coverage)}, "
            f"median {sorted(coverage)[len(coverage) // 2]} of {len(CONCEPT_TAGS)}"
        )
    if failures:
        print(f"                   unresolved: {', '.join(failures[:8])}")
    return resolved > 0


def backfill_macro(root: Path) -> bool:
    if not os.environ.get("FRED_API_KEY"):
        print("  fred    SKIPPED  set FRED_API_KEY (risk-free rate falls back to 4%)")
        return False

    client = FredClient(recording_fetcher(root))
    ok = 0
    for series_id in (THREE_MONTH_TREASURY, TERM_SPREAD, UNEMPLOYMENT, CPI, FED_FUNDS):
        try:
            series = client.series(series_id)
            print(f"  fred    OK       {series_id}: {len(series)} vintages")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  fred    FAILED   {series_id}: {type(exc).__name__}: {exc}")
    return ok > 0


def main() -> int:
    root = Path(os.environ.get("CACHE_ROOT", str(DEFAULT_CACHE_ROOT)))
    years = int(os.environ.get("BACKFILL_YEARS", DEFAULT_YEARS))
    root.mkdir(parents=True, exist_ok=True)

    print(f"Recording into {root} ({years}y of history). Re-runs are served from cache.\n")
    prices = backfill_prices(root, years)
    fundamentals = backfill_fundamentals(root)
    backfill_macro(root)

    print()
    if prices:
        print("Prices recorded — `make results` will now use real data.")
    else:
        # Missing credentials on a first run is an expected state, not a
        # failure. Exiting non-zero here would make `make backfill` look broken
        # to someone who simply has not added keys yet.
        print("No prices recorded — `make results` stays on synthetic data.")
        print("Add credentials to .env (see .env.example), then re-run.")
    if not fundamentals:
        print("No fundamentals — the fundamental agent will return NEUTRAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
