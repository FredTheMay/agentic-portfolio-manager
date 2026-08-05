"""Lookahead bias is impossible by construction (SPEC §4.4).

The single most important test file in the repo. A backtest that can see a
filing before it was published does not have a bug in its returns — it has
returns that mean nothing at all, and the number will always look good.

Every data accessor takes an ``as_of`` instant and may only return what was
public at that instant. These tests pin that property from both directions:
the obvious case (a future filing is invisible) and the subtle one (a *revised*
figure must not leak backwards into the period before the revision).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from src.data.pit import PointInTimeSeries, PointInTimeError, Vintage
from src.time.clock import UTC

D = Decimal


def at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# A company with a December fiscal year end. The Q4/FY2023 figures describe a
# period ending 2023-12-31 but are not filed with the SEC until 2024-02-15.
FY2023 = Vintage(period_end=at(2023, 12, 31), published=at(2024, 2, 15), value=D("105"))
FY2022 = Vintage(period_end=at(2022, 12, 31), published=at(2023, 2, 14), value=D("90"))


def annual_filings() -> PointInTimeSeries[Decimal]:
    return PointInTimeSeries([FY2022, FY2023])


# ---------------------------------------------------------------------------
# The spec's named case
# ---------------------------------------------------------------------------


def test_a_q4_filing_published_in_february_is_invisible_in_january() -> None:
    # SPEC §4.4, verbatim. The period ended 2023-12-31, so a system indexing by
    # fiscal period end would happily serve it on 2024-01-15 — six weeks before
    # anyone outside the company could have seen it.
    series = annual_filings()

    in_january = series.as_of(at(2024, 1, 15))
    assert in_january is not None
    assert in_january.value == D("90"), "January must still see only the FY2022 filing"

    after_filing = series.as_of(at(2024, 2, 16))
    assert after_filing is not None
    assert after_filing.value == D("105")


def test_visibility_flips_exactly_on_the_publication_instant() -> None:
    series = annual_filings()

    day_before = series.as_of(at(2024, 2, 14))
    assert day_before is not None and day_before.value == D("90")

    # Inclusive at the publication instant: once it is out, it is public.
    on_the_day = series.as_of(at(2024, 2, 15))
    assert on_the_day is not None and on_the_day.value == D("105")


def test_nothing_is_visible_before_the_first_publication() -> None:
    assert annual_filings().as_of(at(2020, 1, 1)) is None


def test_period_end_is_never_used_for_visibility() -> None:
    # The property that makes the whole thing work: asking at any instant
    # between period end and publication returns the *previous* filing.
    series = annual_filings()
    for day in (1, 10, 20, 31):
        result = series.as_of(at(2024, 1, day))
        assert result is not None
        assert result.period_end == at(2022, 12, 31)


# ---------------------------------------------------------------------------
# Revisions — the subtle half
# ---------------------------------------------------------------------------

# Macro series are revised. GDP for Q1 is released in April and revised in May.
# At an as_of in late April you must see the April number, not the May one.
FIRST_RELEASE = Vintage(period_end=at(2024, 3, 31), published=at(2024, 4, 25), value=D("2.1"))
REVISION = Vintage(period_end=at(2024, 3, 31), published=at(2024, 5, 30), value=D("1.6"))


def revised_series() -> PointInTimeSeries[Decimal]:
    return PointInTimeSeries([FIRST_RELEASE, REVISION])


def test_a_revision_does_not_leak_backwards() -> None:
    # In late April the world believed 2.1. A backtest that shows 1.6 there is
    # trading on information that did not exist.
    in_april = revised_series().as_of(at(2024, 4, 30))
    assert in_april is not None
    assert in_april.value == D("2.1")


def test_the_revision_applies_once_it_is_published() -> None:
    in_june = revised_series().as_of(at(2024, 6, 1))
    assert in_june is not None
    assert in_june.value == D("1.6")


def test_latest_for_period_respects_as_of() -> None:
    series = revised_series()
    early = series.latest_for_period(at(2024, 3, 31), as_of=at(2024, 4, 30))
    late = series.latest_for_period(at(2024, 3, 31), as_of=at(2024, 6, 1))
    assert early is not None and early.value == D("2.1")
    assert late is not None and late.value == D("1.6")


def test_latest_for_period_returns_none_for_an_unknown_period() -> None:
    assert revised_series().latest_for_period(at(2019, 3, 31), as_of=at(2024, 6, 1)) is None


def test_as_of_prefers_the_most_recent_period_not_the_most_recent_release() -> None:
    # A revision to an *old* period, published after a newer period's first
    # release, must not displace the newer period.
    old_revision = Vintage(
        period_end=at(2022, 12, 31), published=at(2024, 3, 1), value=D("91")
    )
    series = PointInTimeSeries([FY2022, FY2023, old_revision])

    current = series.as_of(at(2024, 3, 2))
    assert current is not None
    assert current.period_end == at(2023, 12, 31)
    assert current.value == D("105")


def test_visible_at_returns_only_published_records() -> None:
    visible = annual_filings().visible_at(at(2024, 1, 15))
    assert [v.value for v in visible] == [D("90")]


def test_visible_at_is_ordered_by_period() -> None:
    visible = annual_filings().visible_at(at(2024, 3, 1))
    assert [v.period_end for v in visible] == [at(2022, 12, 31), at(2023, 12, 31)]


# ---------------------------------------------------------------------------
# Construction-time guards
# ---------------------------------------------------------------------------


def test_a_record_published_before_its_period_ends_is_rejected() -> None:
    # Data that claims to describe a period that has not finished is either a
    # forecast or a date-parsing bug. Either way it must not enter the store.
    with pytest.raises(PointInTimeError, match="published"):
        Vintage(period_end=at(2024, 12, 31), published=at(2024, 6, 30), value=D("1"))


def test_naive_timestamps_are_rejected() -> None:
    from src.time.clock import ClockError

    with pytest.raises(ClockError):
        Vintage(period_end=datetime(2023, 12, 31), published=at(2024, 2, 15), value=D("1"))
    with pytest.raises(ClockError):
        Vintage(period_end=at(2023, 12, 31), published=datetime(2024, 2, 15), value=D("1"))


def test_as_of_rejects_a_naive_query() -> None:
    from src.time.clock import ClockError

    with pytest.raises(ClockError):
        annual_filings().as_of(datetime(2024, 1, 15))


def test_timestamps_are_normalized_to_utc() -> None:
    from datetime import timedelta, timezone

    eastern = timezone(timedelta(hours=-5))
    vintage = Vintage(
        period_end=datetime(2023, 12, 31, 19, 0, tzinfo=eastern),  # = 2024-01-01 00:00 UTC
        published=datetime(2024, 2, 15, 9, 0, tzinfo=eastern),
        value=D("1"),
    )
    assert vintage.period_end == at(2024, 1, 1)
    assert vintage.published == datetime(2024, 2, 15, 14, 0, tzinfo=UTC)


def test_empty_series_is_permitted_and_answers_none() -> None:
    # A company with no filings yet is a normal state, not an error.
    empty: PointInTimeSeries[Decimal] = PointInTimeSeries([])
    assert empty.as_of(at(2024, 1, 1)) is None
    assert empty.visible_at(at(2024, 1, 1)) == []


def test_series_is_immutable_after_construction() -> None:
    records = [FY2022]
    series = PointInTimeSeries(records)
    records.append(FY2023)  # mutating the caller's list must not affect the store
    assert series.as_of(at(2024, 3, 1)) is not None
    assert series.as_of(at(2024, 3, 1)).value == D("90")  # type: ignore[union-attr]
