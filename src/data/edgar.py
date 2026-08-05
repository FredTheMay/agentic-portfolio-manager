"""SEC EDGAR fundamentals, indexed by filing date (SPEC §4.4).

EDGAR's ``companyfacts`` API returns, for each XBRL tag, a list of facts that
each carry both ``end`` (the fiscal period the number describes) and ``filed``
(the date the filing hit the wire). This module keys visibility on **filed**,
never on ``end`` — see :mod:`src.data.pit` for why that distinction is the
whole ballgame.

Two further conservatisms:

**Publication lag.** EDGAR reports ``filed`` as a date with no time. Treating a
filing as public at 00:00 UTC on that date would make it visible before the US
market opened, which is a small lookahead. Filings are instead treated as
public at the *end* of the filing day (``PUBLICATION_LAG``, default 1 day). If
the effect is wrong it is wrong in the safe direction.

**Restatements are kept.** A later filing that restates an earlier period is
stored as a separate vintage rather than overwriting it, so a query in the
intervening window still returns the figure the market actually had.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from src.data.cache import FetchError, JsonFetcher
from src.data.pit import PointInTimeSeries, Vintage
from src.time.clock import UTC

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

#: Filings become visible at the end of the day they were filed, not the start.
PUBLICATION_LAG = timedelta(days=1)


class EdgarError(RuntimeError):
    """Raised on malformed or unusable EDGAR data."""


#: XBRL tags vary between filers, so each field lists candidates in preference
#: order. The first tag that yields a value visible at ``as_of`` wins.
CONCEPT_TAGS: Mapping[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "cost_of_goods_sold": ("CostOfGoodsAndServicesSold", "CostOfRevenue"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "interest_expense": ("InterestExpense", "InterestExpenseDebt"),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "cash_flow_operations": ("NetCashProvidedByUsedInOperatingActivities",),
    "total_assets": ("Assets",),
    "total_equity": ("StockholdersEquity",),
    "total_liabilities": ("Liabilities",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "inventory": ("InventoryNet",),
    "receivables": ("AccountsReceivableNetCurrent",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "dividends_paid": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
}


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Point-in-time financial statement figures for one company.

    Every field is optional: filers use different tags, and a missing figure is
    a normal outcome that must stay distinguishable from zero. Downstream code
    treats ``None`` as "no view", never as a zero to compute with.

    ``as_of`` records the instant this snapshot was taken, and ``period_end``
    the latest fiscal period visible at that instant.
    """

    symbol: str
    as_of: datetime
    period_end: datetime | None
    revenue: Decimal | None = None
    cost_of_goods_sold: Decimal | None = None
    gross_profit: Decimal | None = None
    operating_income: Decimal | None = None
    interest_expense: Decimal | None = None
    pretax_income: Decimal | None = None
    net_income: Decimal | None = None
    cash_flow_operations: Decimal | None = None
    total_assets: Decimal | None = None
    total_equity: Decimal | None = None
    total_liabilities: Decimal | None = None
    current_assets: Decimal | None = None
    current_liabilities: Decimal | None = None
    inventory: Decimal | None = None
    receivables: Decimal | None = None
    cash: Decimal | None = None
    long_term_debt: Decimal | None = None
    dividends_paid: Decimal | None = None


def _parse_date(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise EdgarError(f"{field} must be a date string, got {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EdgarError(f"{field} is not an ISO date: {value!r}") from exc
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def _parse_amount(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise EdgarError("boolean is not a monetary amount")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise EdgarError(f"not a numeric amount: {value!r}") from exc
    if isinstance(value, float):
        # Should not happen: cache.loads parses JSON floats as Decimal. Convert
        # through repr rather than dropping the datum, but never through
        # float arithmetic.
        return Decimal(repr(value))
    raise EdgarError(f"not a numeric amount: {value!r}")


def concept_series(
    facts: Mapping[str, Any],
    tag: str,
    unit: str = "USD",
    taxonomy: str = "us-gaap",
    publication_lag: timedelta = PUBLICATION_LAG,
) -> PointInTimeSeries[Decimal]:
    """Build a point-in-time series for one XBRL tag from a companyfacts document.

    Facts without an ``end`` or a ``filed`` date are skipped: a datum whose
    publication date is unknown cannot be shown to have been public, so it must
    not be usable.
    """
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if node is None:
        return PointInTimeSeries([])

    entries = node.get("units", {}).get(unit)
    if not entries:
        return PointInTimeSeries([])

    vintages: list[Vintage[Decimal]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw_end, raw_filed, raw_value = entry.get("end"), entry.get("filed"), entry.get("val")
        if raw_end is None or raw_filed is None or raw_value is None:
            continue
        period_end = _parse_date(raw_end, "end")
        published = _parse_date(raw_filed, "filed") + publication_lag
        vintages.append(
            Vintage(period_end=period_end, published=published, value=_parse_amount(raw_value))
        )
    return PointInTimeSeries(vintages)


class EdgarClient:
    """Reads company fundamentals from EDGAR through a :class:`JsonFetcher`.

    Takes its fetcher by injection, so the same client serves live requests, a
    cached replay, or a test stub with no code change.
    """

    def __init__(
        self,
        fetcher: JsonFetcher,
        publication_lag: timedelta = PUBLICATION_LAG,
    ) -> None:
        self._fetcher = fetcher
        self._publication_lag = publication_lag
        self._ticker_to_cik: dict[str, int] | None = None

    def resolve_cik(self, symbol: str) -> int:
        """Map a ticker to its SEC CIK number."""
        if self._ticker_to_cik is None:
            self._ticker_to_cik = self._load_ticker_map()
        cik = self._ticker_to_cik.get(symbol.upper())
        if cik is None:
            raise EdgarError(f"no CIK on file for ticker {symbol!r}")
        return cik

    def _load_ticker_map(self) -> dict[str, int]:
        payload = self._fetcher.get_json(TICKER_MAP_URL)
        if not isinstance(payload, Mapping):
            raise EdgarError("ticker map response was not a JSON object")
        mapping: dict[str, int] = {}
        for entry in payload.values():
            if not isinstance(entry, Mapping):
                continue
            ticker, cik = entry.get("ticker"), entry.get("cik_str")
            if isinstance(ticker, str) and cik is not None:
                mapping[ticker.upper()] = int(cik)
        if not mapping:
            raise EdgarError("ticker map contained no usable entries")
        return mapping

    def company_facts(self, cik: int) -> Mapping[str, Any]:
        """Raw companyfacts document for a CIK."""
        payload = self._fetcher.get_json(COMPANY_FACTS_URL.format(cik=cik))
        if not isinstance(payload, Mapping):
            raise EdgarError(f"companyfacts for CIK {cik} was not a JSON object")
        return payload

    def get_fundamentals(self, symbol: str, as_of: datetime) -> Fundamentals | None:
        """Financial statement figures for ``symbol`` as they stood at ``as_of``.

        Returns ``None`` when the company had filed nothing visible at that
        instant — a young or newly listed company, not an error.

        Each field is resolved independently against ``as_of``, so a snapshot
        can legitimately mix periods: a balance-sheet item restated in a later
        filing carries its restated value only from the restatement forward.
        """
        try:
            cik = self.resolve_cik(symbol)
            facts = self.company_facts(cik)
        except FetchError:
            raise

        values: dict[str, Decimal | None] = {}
        latest_period: datetime | None = None

        for field, tags in CONCEPT_TAGS.items():
            resolved: Decimal | None = None
            for tag in tags:
                series = concept_series(
                    facts, tag, publication_lag=self._publication_lag
                )
                vintage = series.as_of(as_of)
                if vintage is not None:
                    resolved = vintage.value
                    if latest_period is None or vintage.period_end > latest_period:
                        latest_period = vintage.period_end
                    break
            values[field] = resolved

        if all(value is None for value in values.values()):
            return None

        return Fundamentals(symbol=symbol.upper(), as_of=as_of, period_end=latest_period, **values)


def total_debt(fundamentals: Fundamentals) -> Decimal | None:
    """Best available debt figure for the solvency ratios.

    Prefers reported long-term debt. Falls back to total liabilities with the
    caveat that liabilities include payables and accruals, which are not
    borrowings — a materially different and larger number, so callers that care
    about the distinction should check ``long_term_debt`` themselves.
    """
    if fundamentals.long_term_debt is not None:
        return fundamentals.long_term_debt
    return fundamentals.total_liabilities


def average(current: Decimal | None, prior: Decimal | None) -> Decimal | None:
    """Average of two balance-sheet figures, for ratios that divide a flow by a stock.

    Returns ``None`` if either side is missing rather than silently using a
    single point-in-time balance, which would bias turnover and return ratios
    whenever the balance sheet grew or shrank over the period.
    """
    if current is None or prior is None:
        return None
    return (current + prior) / Decimal(2)


def visible_periods(
    facts: Mapping[str, Any],
    tag: str,
    as_of: datetime,
    publication_lag: timedelta = PUBLICATION_LAG,
) -> Sequence[datetime]:
    """Fiscal periods for ``tag`` that were public at ``as_of``. Diagnostic helper."""
    series = concept_series(facts, tag, publication_lag=publication_lag)
    return [vintage.period_end for vintage in series.visible_at(as_of)]


def iter_tags(facts: Mapping[str, Any], taxonomy: str = "us-gaap") -> list[str]:
    """Every XBRL tag present in a companyfacts document. Diagnostic helper."""
    node = facts.get("facts", {}).get(taxonomy, {})
    if not isinstance(node, Mapping):
        return []
    return sorted(str(tag) for tag in node)
