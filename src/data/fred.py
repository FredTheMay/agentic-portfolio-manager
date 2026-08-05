"""FRED macro series, vintage-aware (SPEC §4.4, §5.3).

Macro data is **revised**, often substantially. Q1 GDP released in April is
restated in May and again in June. A backtest that reads today's final value
for a date in April is trading on a number nobody had.

FRED exposes this through ALFRED's real-time dimension: each observation
carries ``realtime_start``, the date that value became the published figure.
This module requests all vintages (``output_type=4``) and keys visibility on
``realtime_start``, so a query in late April returns the April release and the
May revision stays invisible until May.

Series used by the Macro/Regime agent (SPEC §5.3):

===========  ==================================================
``T10Y3M``   10-year minus 3-month term spread; inversion signal
``UNRATE``   unemployment rate
``CPIAUCSL`` CPI, for the year-over-year inflation rate
``DFF``      effective fed funds rate, for policy direction
``DGS3MO``   3-month Treasury — **discount basis**, see below
===========  ==================================================

``DGS3MO`` is the natural risk-free rate and is quoted on a bank discount
basis. It must go through
:func:`src.cfa.fixed_income.discount_to_bond_equivalent_yield` before it is
used in Sharpe, Treynor, CAPM, or the CAL (SPEC §6.2 [CORRECTED]).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from src.data.cache import JsonFetcher
from src.data.pit import PointInTimeSeries, Vintage
from src.time.clock import UTC

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

#: Environment variable holding a FRED API key. Absent is fine when the cache
#: already has what a run needs.
API_KEY_ENV = "FRED_API_KEY"

#: FRED's sentinel for a missing observation.
MISSING = "."

TERM_SPREAD = "T10Y3M"
UNEMPLOYMENT = "UNRATE"
CPI = "CPIAUCSL"
FED_FUNDS = "DFF"
THREE_MONTH_TREASURY = "DGS3MO"


class FredError(RuntimeError):
    """Raised on malformed or unusable FRED data."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One FRED observation as published at one moment."""

    #: FRED dates an observation at the *start* of the period it covers.
    #: Visibility is governed entirely by ``published``, so this field
    #: identifies the period and never gates access.
    period: datetime
    published: datetime
    value: Decimal


def _parse_date(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise FredError(f"{field} must be a date string, got {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise FredError(f"{field} is not an ISO date: {value!r}") from exc
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def parse_observations(payload: Mapping[str, Any]) -> PointInTimeSeries[Decimal]:
    """Build a point-in-time series from a FRED observations document.

    Observations whose value is FRED's ``"."`` sentinel are dropped: a missing
    reading must stay missing rather than becoming a zero that a regression
    would treat as a real data point.
    """
    raw = payload.get("observations")
    if not isinstance(raw, list):
        raise FredError("FRED response contained no observations array")

    vintages: list[Vintage[Decimal]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("value")
        if value is None or value == MISSING:
            continue
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            continue

        period = _parse_date(entry.get("date"), "date")
        published_raw = entry.get("realtime_start")
        # Without a real-time date there is no evidence of when this was
        # public, so it cannot be used point-in-time.
        if published_raw is None:
            continue
        published = _parse_date(published_raw, "realtime_start")

        # FRED dates observations at period start, so a release can legitimately
        # predate the "period end" the store expects. Anchor the period at the
        # publication date when that happens; visibility is unaffected either way.
        vintages.append(
            Vintage(period_end=min(period, published), published=published, value=amount)
        )
    return PointInTimeSeries(vintages)


class FredClient:
    """Reads FRED series through a :class:`JsonFetcher`.

    ``api_key`` falls back to ``$FRED_API_KEY``. The key is *not* part of the
    cache key, so a cached series replays regardless of which key recorded it —
    and a backtest runs with no key at all.
    """

    def __init__(self, fetcher: JsonFetcher, api_key: str | None = None) -> None:
        self._fetcher = fetcher
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)

    def series(self, series_id: str) -> PointInTimeSeries[Decimal]:
        """Every vintage of ``series_id``, ready to query at any instant."""
        params = {
            "series_id": series_id,
            "file_type": "json",
            # All vintages, initial release and every revision.
            "output_type": "4",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        payload = self._fetcher.get_json(OBSERVATIONS_URL, params)
        if not isinstance(payload, Mapping):
            raise FredError(f"FRED response for {series_id} was not a JSON object")
        return parse_observations(payload)

    def value_as_of(self, series_id: str, as_of: datetime) -> Decimal | None:
        """Latest value of ``series_id`` that was published by ``as_of``."""
        vintage = self.series(series_id).as_of(as_of)
        return vintage.value if vintage is not None else None


def year_over_year_change(
    series: PointInTimeSeries[Decimal],
    as_of: datetime,
    periods_per_year: int = 12,
) -> Decimal | None:
    """Year-over-year rate of change, computed only from visible observations.

    Used for CPI inflation (SPEC §5.3). Returns ``None`` when a full year of
    history was not yet public — the honest answer, rather than an annualized
    figure derived from a partial window.
    """
    visible = series.visible_at(as_of)
    if len(visible) <= periods_per_year:
        return None
    latest = visible[-1].value
    year_ago = visible[-1 - periods_per_year].value
    if year_ago == 0:
        return None
    return latest / year_ago - Decimal(1)
