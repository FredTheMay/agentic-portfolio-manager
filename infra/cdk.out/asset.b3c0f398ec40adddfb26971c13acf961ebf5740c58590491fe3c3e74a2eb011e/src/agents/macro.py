"""Macro agent: business-cycle phase by rule, narrative by model.

The phase is classified by :func:`classify_phase`, twelve lines you can read
and disagree with. The model writes prose around a classification it was
handed and cannot change it.

Signals are read point-in-time, so a backtest sees only what was published at
the time. :func:`classify_phase` returns NEUTRAL when signals disagree rather
than picking the closest match.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.agents.schemas import CyclePhase, MacroNarrative
from src.data.fred import (
    CPI,
    FED_FUNDS,
    TERM_SPREAD,
    UNEMPLOYMENT,
    year_over_year_change,
)
from src.data.pit import PointInTimeSeries
from src.llm.base import LLMProvider, Stance
from src.time.clock import ensure_utc

ZERO = Decimal(0)

#: A term spread below this is treated as inverted.
INVERSION_THRESHOLD = Decimal("0")
#: Year-over-year CPI above this counts as elevated inflation.
HIGH_INFLATION = Decimal("0.03")
#: Rise in the unemployment rate, in percentage points, that counts as a trend.
UNEMPLOYMENT_TREND = Decimal("0.5")

SYSTEM_PROMPT = """You are a macroeconomic commentator.

You will be given macro indicators and a business-cycle phase that has ALREADY
been determined by a deterministic rule.

Rules you must follow:
- Do NOT dispute or re-derive the phase. It was decided by rule.
- Output BULLISH, NEUTRAL, or BEARISH for broad equity risk, plus a conviction
  from 1 to 5.
- You may NOT output any number. Refer to indicators by name and direction.
- Write for a reader who will see the indicator values alongside your text."""


@dataclass(frozen=True, slots=True)
class MacroSignals:
    """Point-in-time macro readings, all optional.

    A missing signal is normal — a series may not have been released yet at the
    ``as_of`` instant — and must stay distinguishable from a reading of zero.
    """

    as_of: datetime
    term_spread: Decimal | None = None
    unemployment: Decimal | None = None
    unemployment_change: Decimal | None = None
    inflation_yoy: Decimal | None = None
    fed_funds: Decimal | None = None
    fed_funds_change: Decimal | None = None

    def render(self) -> str:
        def show(name: str, value: Decimal | None) -> str:
            return f"  {name}: {'not yet published' if value is None else value}"

        return "\n".join(
            [
                show("term_spread_10y_3m", self.term_spread),
                show("unemployment_rate", self.unemployment),
                show("unemployment_change", self.unemployment_change),
                show("cpi_yoy", self.inflation_yoy),
                show("fed_funds_rate", self.fed_funds),
                show("fed_funds_change", self.fed_funds_change),
            ]
        )


def classify_phase(signals: MacroSignals) -> CyclePhase:
    """Classify the business cycle by rule.

    Deliberately simple and deliberately legible. The ordering encodes which
    signal dominates: an inverted curve with rising unemployment is a
    contraction regardless of what inflation is doing, because by then the
    labour market has already turned.

    Returns ``NEUTRAL`` when the signals do not agree, rather than picking the
    closest match. A phase the data does not support is worse than no phase.
    """
    inverted = signals.term_spread is not None and signals.term_spread < INVERSION_THRESHOLD
    rising_unemployment = (
        signals.unemployment_change is not None
        and signals.unemployment_change >= UNEMPLOYMENT_TREND
    )
    falling_unemployment = (
        signals.unemployment_change is not None
        and signals.unemployment_change <= -UNEMPLOYMENT_TREND
    )
    hot = signals.inflation_yoy is not None and signals.inflation_yoy > HIGH_INFLATION
    tightening = signals.fed_funds_change is not None and signals.fed_funds_change > ZERO

    if inverted and rising_unemployment:
        return CyclePhase.CONTRACTION
    if inverted and (hot or tightening):
        # Curve inverted and policy still tight: late cycle, not yet recession.
        return CyclePhase.PEAK
    if falling_unemployment and not inverted:
        return CyclePhase.EXPANSION
    if rising_unemployment and not inverted:
        return CyclePhase.PEAK
    if not inverted and signals.term_spread is not None and not hot and falling_unemployment:
        return CyclePhase.TROUGH
    return CyclePhase.NEUTRAL


def read_signals(
    as_of: datetime,
    term_spread: PointInTimeSeries[Decimal] | None = None,
    unemployment: PointInTimeSeries[Decimal] | None = None,
    cpi: PointInTimeSeries[Decimal] | None = None,
    fed_funds: PointInTimeSeries[Decimal] | None = None,
) -> MacroSignals:
    """Read every series as of one instant, never later."""
    moment = ensure_utc(as_of)

    def latest(series: PointInTimeSeries[Decimal] | None) -> Decimal | None:
        if series is None:
            return None
        vintage = series.as_of(moment)
        return vintage.value if vintage is not None else None

    def change(series: PointInTimeSeries[Decimal] | None, lag: int = 12) -> Decimal | None:
        if series is None:
            return None
        visible = series.visible_at(moment)
        if len(visible) <= lag:
            return None
        return visible[-1].value - visible[-1 - lag].value

    return MacroSignals(
        as_of=moment,
        term_spread=latest(term_spread),
        unemployment=latest(unemployment),
        unemployment_change=change(unemployment),
        inflation_yoy=year_over_year_change(cpi, moment) if cpi is not None else None,
        fed_funds=latest(fed_funds),
        fed_funds_change=change(fed_funds),
    )


@dataclass(frozen=True, slots=True)
class MacroView:
    """The rule's phase plus the model's narrative around it."""

    phase: CyclePhase
    signals: MacroSignals
    narrative: MacroNarrative


@dataclass(slots=True)
class MacroAgent:
    """Classifies by rule, narrates by model."""

    provider: LLMProvider

    def run(self, signals: MacroSignals) -> MacroView:
        phase = classify_phase(signals)
        user = (
            f"As of: {signals.as_of.isoformat()}\n"
            f"Business-cycle phase (determined by rule): {phase.value}\n\n"
            f"Indicators:\n{signals.render()}"
        )
        narrative = self.provider.complete(SYSTEM_PROMPT, user, MacroNarrative)
        return MacroView(phase=phase, signals=signals, narrative=narrative)


#: Convenience for callers assembling series from FRED.
SERIES_IDS = {
    "term_spread": TERM_SPREAD,
    "unemployment": UNEMPLOYMENT,
    "cpi": CPI,
    "fed_funds": FED_FUNDS,
}


__all__ = [
    "CyclePhase",
    "MacroAgent",
    "MacroSignals",
    "MacroView",
    "SERIES_IDS",
    "classify_phase",
    "read_signals",
]
