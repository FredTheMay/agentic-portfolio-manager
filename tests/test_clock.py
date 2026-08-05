"""Unit tests for the clock abstraction (SPEC §4.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.time.clock import UTC, Clock, ClockError, SimulationClock, WallClock, ensure_utc


def test_both_clocks_satisfy_the_protocol() -> None:
    assert isinstance(WallClock(), Clock)
    assert isinstance(SimulationClock(datetime(2024, 1, 1, tzinfo=UTC)), Clock)


def test_wall_clock_returns_tz_aware_utc() -> None:
    now = WallClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_ensure_utc_rejects_naive_datetimes() -> None:
    with pytest.raises(ClockError, match="naive datetime"):
        ensure_utc(datetime(2024, 1, 1, 12, 0))


def test_ensure_utc_converts_other_zones() -> None:
    eastern = timezone(timedelta(hours=-5))
    # 09:30 New York on a winter day is 14:30 UTC — the market open as an instant.
    converted = ensure_utc(datetime(2024, 1, 3, 9, 30, tzinfo=eastern))
    assert converted == datetime(2024, 1, 3, 14, 30, tzinfo=UTC)
    assert converted.tzinfo is UTC


def test_simulation_clock_normalizes_start_to_utc() -> None:
    eastern = timezone(timedelta(hours=-5))
    clock = SimulationClock(datetime(2024, 1, 3, 9, 30, tzinfo=eastern))
    assert clock.now() == datetime(2024, 1, 3, 14, 30, tzinfo=UTC)


def test_simulation_clock_rejects_naive_start() -> None:
    with pytest.raises(ClockError):
        SimulationClock(datetime(2024, 1, 1))


def test_simulation_clock_advances_forwards() -> None:
    clock = SimulationClock(datetime(2024, 1, 1, tzinfo=UTC))
    clock.advance_to(datetime(2024, 1, 2, tzinfo=UTC))
    assert clock.now() == datetime(2024, 1, 2, tzinfo=UTC)

    clock.advance_by(timedelta(hours=6))
    assert clock.now() == datetime(2024, 1, 2, 6, tzinfo=UTC)


def test_simulation_clock_allows_standing_still() -> None:
    # Several events can share one instant; that is not an ordering violation.
    start = datetime(2024, 1, 1, tzinfo=UTC)
    clock = SimulationClock(start)
    clock.advance_to(start)
    assert clock.now() == start


def test_simulation_clock_refuses_to_move_backwards() -> None:
    # Out-of-order events would let the system read data it could not have had.
    clock = SimulationClock(datetime(2024, 6, 1, tzinfo=UTC))
    with pytest.raises(ClockError, match="backwards"):
        clock.advance_to(datetime(2024, 5, 31, tzinfo=UTC))
    with pytest.raises(ClockError, match="backwards"):
        clock.advance_by(timedelta(seconds=-1))
    assert clock.now() == datetime(2024, 6, 1, tzinfo=UTC)


def test_simulation_clock_is_deterministic() -> None:
    # SPEC §9: identical inputs, identical output.
    a = SimulationClock(datetime(2024, 1, 1, tzinfo=UTC))
    b = SimulationClock(datetime(2024, 1, 1, tzinfo=UTC))
    a.advance_by(timedelta(days=3))
    b.advance_by(timedelta(days=3))
    assert a.now() == b.now()
