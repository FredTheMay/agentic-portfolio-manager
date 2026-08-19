"""FRED macroeconomic series, read point-in-time.

Macro data is revised, often substantially: ``CPIAUCSL`` carries roughly four
revisions per period. Visibility is keyed on ALFRED's ``realtime_start``, so a
query in late April sees the April release and the May revision stays invisible
until May.

Two fetch strategies, because FRED caps a response at ~2000 vintage dates.
Revised series are requested across the full real-time window. A *daily* series
has one vintage date per business day and exceeds that cap within a decade —
FRED answers 400 — so daily market rates are fetched at current real time and
dated by observation, which is correct because they are published once and
never revised. See :data:`REVISED_SERIES`.

``DGS3MO`` is quoted on a bank discount basis and must go through
:func:`src.cfa.fixed_income.discount_to_bond_equivalent_yield` before use as a
risk-free rate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from src.data.cache import JsonFetcher
from src.data.pit import PointInTimeSeries, Vintage
from src.time.clock import UTC

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

#: FRED's real-time parameters default to *today*, which returns only the
#: currently-effective value. Point-in-time work needs the whole history of
#: what was believed when, so revised series are requested across all real time.
ALL_REALTIME_START = "1776-07-04"
ALL_REALTIME_END = "9999-12-31"

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

#: Series FRED revises after first publication. These are fetched across the
#: full real-time window so every vintage is recorded — CPIAUCSL alone carries
#: roughly four revisions per period, and reading the final value at a date
#: before it existed is precisely the lookahead forbids.
REVISED_SERIES: frozenset[str] = frozenset({CPI, UNEMPLOYMENT})

#: Daily market-rate series are published once and never revised. FRED caps a
#: request at ~2000 vintage dates and a daily series has one per business day,
#: so the full real-time window is not merely unnecessary for these — it is
#: rejected with a 400. Their publication instant is derived from the
#: observation date instead.
DAILY_PUBLICATION_LAG = timedelta(days=1)


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


def parse_observations(
    payload: Mapping[str, Any],
    publication_lag: timedelta | None = None,
) -> PointInTimeSeries[Decimal]:
    """Build a point-in-time series from a FRED observations document.

    Observations whose value is FRED's ``"."`` sentinel are dropped: a missing
    reading must stay missing rather than becoming a zero that a regression
    would treat as a real data point.

    ``publication_lag`` handles unrevised series, where the response carries
    today's real-time window rather than the original publication date. The
    observation date plus the lag is used instead — correct for a series that
    is published once, and conservative in the safe direction.
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

        if publication_lag is not None:
            # Unrevised series: the response reports the *current* real-time
            # window, not when the value first appeared, so realtime_start
            # would wrongly read as "published today".
            published = period + publication_lag
        else:
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

    def series(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> PointInTimeSeries[Decimal]:
        """Every vintage of ``series_id``, ready to query at any instant.

        Revised series (:data:`REVISED_SERIES`) are requested across the whole
        real-time window, so each revision arrives as its own row with the date
        it became the published figure. Unrevised daily series cannot be
        requested that way — FRED caps a response at ~2000 vintage dates and a
        daily series exceeds that within a decade — so they are fetched at
        current real time and dated by observation instead.
        """
        revised = series_id in REVISED_SERIES
        params = {
            "series_id": series_id,
            "file_type": "json",
            "output_type": "1",
        }
        if revised:
            params["realtime_start"] = ALL_REALTIME_START
            params["realtime_end"] = ALL_REALTIME_END
        if observation_start:
            params["observation_start"] = observation_start
        if self._api_key:
            params["api_key"] = self._api_key

        payload = self._fetcher.get_json(OBSERVATIONS_URL, params)
        if not isinstance(payload, Mapping):
            raise FredError(f"FRED response for {series_id} was not a JSON object")
        return parse_observations(
            payload, publication_lag=None if revised else DAILY_PUBLICATION_LAG
        )

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

    Used for CPI inflation. Returns ``None`` when a full year of
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
