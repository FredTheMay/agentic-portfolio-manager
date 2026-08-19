"""Clock abstraction: the only module permitted to read the wall clock.

Every other module takes a :class:`Clock`, which is what lets one code path
serve both a backtest and a live cycle. ``tests/test_no_wall_clock.py`` fails
the build on any direct call elsewhere.

Every timestamp is a tz-aware UTC instant, never a date.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

UTC = timezone.utc


class ClockError(Exception):
    """Raised on naive timestamps or on a simulation clock moving backwards."""


def ensure_utc(ts: datetime) -> datetime:
    """Normalize ``ts`` to UTC, rejecting naive datetimes.

    A naive datetime is always a bug here: it means some caller lost the
    timezone, and the resulting comparison against a UTC instant would be
    silently wrong rather than loud.
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ClockError(
            f"naive datetime {ts!r}: all timestamps must be tz-aware"
        )
    return ts.astimezone(UTC)


@runtime_checkable
class Clock(Protocol):
    """Source of the current instant."""

    def now(self) -> datetime:
        """Return the current instant, tz-aware UTC."""
        ...


class WallClock:
    """Real time. Used for live paper trading.

    The single permitted ``datetime.now()`` call in the codebase lives here.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)

    def __repr__(self) -> str:
        return "WallClock()"


class SimulationClock:
    """Backtest time, advanced explicitly by the event loop.

    Time moves only when the engine moves it, and only forwards. A backwards
    step means events were fed out of order, which would let the system read
    data it could not have had — so it raises rather than silently reordering.
    """

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        """Move the clock to ``ts``. Must be at or after the current instant."""
        target = ensure_utc(ts)
        if target < self._now:
            raise ClockError(
                f"clock cannot move backwards: {self._now.isoformat()} -> {target.isoformat()}"
            )
        self._now = target

    def advance_by(self, delta: timedelta) -> None:
        """Move the clock forward by ``delta``."""
        if delta < timedelta(0):
            raise ClockError(f"clock cannot move backwards: negative delta {delta!r}")
        self._now = self._now + delta

    def __repr__(self) -> str:
        return f"SimulationClock({self._now.isoformat()})"
