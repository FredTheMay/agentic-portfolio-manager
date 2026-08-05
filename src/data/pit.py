"""Point-in-time data access (SPEC §4.4).

The mechanism that makes lookahead bias structurally impossible rather than
merely discouraged.

Every datum carries two instants:

``period_end``
    what the datum *describes* — a fiscal quarter, a reference month.
``published``
    when it *became public* — a filing date, a release timestamp.

**Only ``published`` decides visibility.** Indexing fundamentals by fiscal
period end is the classic backtest error: FY2023 figures describe a period
ending 31 December but are not filed until February, so a period-indexed store
serves them to a January query and the backtest trades six weeks of hindsight.

The same applies to revisions. Macro series are restated, and the restatement
must not leak backwards: a query in late April must see the April release of a
Q1 number, not the May revision, because that is what the world believed at the
time.

This module is generic over the value type. EDGAR fundamentals
(:mod:`src.data.edgar`) and FRED series (:mod:`src.data.fred`) both store their
observations here rather than reimplementing the visibility rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Iterable, TypeVar

from src.time.clock import ensure_utc

T = TypeVar("T")


class PointInTimeError(ValueError):
    """Raised on a record that cannot be point-in-time correct."""


@dataclass(frozen=True, slots=True)
class Vintage(Generic[T]):
    """One observation, as published at one moment.

    Several vintages may share a ``period_end``: that is a revision, and each
    one is kept so the store can answer what was believed at any past instant.
    """

    period_end: datetime
    published: datetime
    value: T

    def __post_init__(self) -> None:
        # frozen dataclass: normalize through object.__setattr__.
        period_end = ensure_utc(self.period_end)
        published = ensure_utc(self.published)
        if published < period_end:
            raise PointInTimeError(
                f"published {published.isoformat()} precedes period_end "
                f"{period_end.isoformat()}: a period cannot be reported before it ends"
            )
        object.__setattr__(self, "period_end", period_end)
        object.__setattr__(self, "published", published)


class PointInTimeSeries(Generic[T]):
    """An append-only history of vintages, queryable as of any instant.

    Construction copies the input, so a caller mutating their list afterwards
    cannot retroactively change what the backtest could see.
    """

    __slots__ = ("_vintages",)

    def __init__(self, vintages: Iterable[Vintage[T]]) -> None:
        # Sort by period, then by publication, so "latest revision of the most
        # recent period" is the last visible element.
        self._vintages: tuple[Vintage[T], ...] = tuple(
            sorted(vintages, key=lambda v: (v.period_end, v.published))
        )

    def __len__(self) -> int:
        return len(self._vintages)

    def __repr__(self) -> str:
        return f"PointInTimeSeries({len(self._vintages)} vintages)"

    def history(self) -> tuple[Vintage[T], ...]:
        """Every vintage ever recorded, ignoring visibility. For audit only."""
        return self._vintages

    def visible_at(self, as_of: datetime) -> list[Vintage[T]]:
        """Every vintage published at or before ``as_of``, ordered by period.

        Where a period has been revised, only the latest revision *visible at
        that instant* is returned.
        """
        moment = ensure_utc(as_of)
        latest_by_period: dict[datetime, Vintage[T]] = {}
        for vintage in self._vintages:
            if vintage.published <= moment:
                # Sorted by (period, published), so a later match for the same
                # period is always the newer revision.
                latest_by_period[vintage.period_end] = vintage
        return [latest_by_period[period] for period in sorted(latest_by_period)]

    def as_of(self, as_of: datetime) -> Vintage[T] | None:
        """The most recent *period* known at ``as_of``, at its then-current value.

        Returns ``None`` when nothing had been published yet — a normal state
        for a young company, not an error.

        Note this ranks by ``period_end``, not by ``published``: a revision to
        an old period does not displace a newer period's first release.
        """
        visible = self.visible_at(as_of)
        return visible[-1] if visible else None

    def latest_for_period(self, period_end: datetime, as_of: datetime) -> Vintage[T] | None:
        """The value believed for one specific period, as of an instant.

        Use this to reconstruct what a figure looked like at the time, rather
        than what it was eventually restated to.
        """
        period = ensure_utc(period_end)
        moment = ensure_utc(as_of)
        candidates = [
            v for v in self._vintages if v.period_end == period and v.published <= moment
        ]
        return candidates[-1] if candidates else None
