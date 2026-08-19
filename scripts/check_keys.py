"""Verify each credential against its live API, one at a time.

Run before a first live fetch. Checking each service independently means a
failure names the service and the reason, rather than surfacing as a backtest
that mysteriously produced nothing.

Nothing here is required to run the system: with no credentials at all the
pipeline, backtest, dashboard and tests run offline on synthetic data.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

OK, MISSING, FAILED = "  OK   ", "MISSING", "FAILED "


def report(service: str, status: str, detail: str) -> None:
    print(f"[{status}] {service:<22} {detail}")


def check_edgar() -> None:
    from src.data.cache import HttpxFetcher, user_agent_is_configured
    from src.data.edgar import EdgarClient

    if not user_agent_is_configured():
        report("SEC EDGAR", MISSING, "set EDGAR_USER_AGENT to 'Name email@example.com'")
        return
    try:
        cik = EdgarClient(HttpxFetcher()).resolve_cik("AAPL")
        report("SEC EDGAR", OK, f"resolved AAPL -> CIK {cik}")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not itself crash
        report("SEC EDGAR", FAILED, f"{type(exc).__name__}: {exc}")


def check_fred() -> None:
    from src.data.cache import HttpxFetcher
    from src.data.fred import THREE_MONTH_TREASURY, FredClient

    if not os.environ.get("FRED_API_KEY"):
        report("FRED", MISSING, "set FRED_API_KEY")
        return
    try:
        series = FredClient(HttpxFetcher()).series(THREE_MONTH_TREASURY)
        report("FRED", OK, f"{THREE_MONTH_TREASURY}: {len(series)} vintages")
    except Exception as exc:  # noqa: BLE001
        report("FRED", FAILED, f"{type(exc).__name__}: {exc}")


def check_alpaca_data() -> None:
    from src.data.sources import AlpacaBarClient, live_alpaca_fetcher

    if not (os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY")):
        report("Alpaca market data", MISSING, "set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        return
    try:
        end = datetime.now(timezone.utc) - timedelta(days=2)
        source = AlpacaBarClient(live_alpaca_fetcher()).daily_bars(
            ["SPY"], end - timedelta(days=10), end
        )
        report("Alpaca market data", OK, f"SPY: {len(source.events)} daily bars")
    except Exception as exc:  # noqa: BLE001
        report("Alpaca market data", FAILED, f"{type(exc).__name__}: {exc}")


def check_alpaca_paper() -> None:
    from src.execution.naive import AlpacaPaperBroker

    if not (os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY")):
        report("Alpaca paper account", MISSING, "same credentials as market data")
        return
    try:
        cash = AlpacaPaperBroker().cash()
        report("Alpaca paper account", OK, f"buying power {cash}")
    except Exception as exc:  # noqa: BLE001
        report("Alpaca paper account", FAILED, f"{type(exc).__name__}: {exc}")


def check_llm() -> None:
    from src.agents.schemas import ResearchView
    from src.llm import get_provider

    name = os.environ.get("LLM_PROVIDER", "null")
    if name == "null":
        report("LLM", OK, "LLM_PROVIDER=null — every agent returns NEUTRAL (this is fine)")
        return
    try:
        view = get_provider(name).complete(
            "Return NEUTRAL.", "Ticker: SPY. No headlines.", ResearchView
        )
        report("LLM", OK, f"{name} answered {view.stance.value}")
    except Exception as exc:  # noqa: BLE001
        report("LLM", FAILED, f"{type(exc).__name__}: {exc}")


def main() -> None:
    print("Checking credentials. Every one is optional — with none set, the")
    print("system runs offline on synthetic data.\n")
    for check in (check_edgar, check_fred, check_alpaca_data, check_alpaca_paper, check_llm):
        check()
    print("\nMISSING is fine. FAILED means the credential exists but did not work.")


if __name__ == "__main__":
    main()
