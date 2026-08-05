"""Time control (SPEC §4.1).

The **only** package permitted to read the wall clock. Every other module takes
a :class:`~src.time.clock.Clock`. This is what lets one code path serve both a
three-year backtest and a live daily cycle.
"""

from src.time.clock import (
    UTC,
    Clock,
    ClockError,
    SimulationClock,
    WallClock,
    ensure_utc,
)

__all__ = [
    "UTC",
    "Clock",
    "ClockError",
    "SimulationClock",
    "WallClock",
    "ensure_utc",
]
