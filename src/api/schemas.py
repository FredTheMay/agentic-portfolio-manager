"""API response models (SPEC §9, M9).

Every monetary and ratio value crosses the wire as a **decimal string**, for
the same reason the execution contract does (SPEC §3.2): JSON numbers are
IEEE-754 doubles, and a weight that round-trips through one is no longer the
weight the risk engine approved.

The disclaimer is a required field on the root document rather than a template
detail, so no screen can render without it (SPEC §1).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from pydantic import BaseModel, Field

#: SPEC §1. Carried on every response, not just the landing page.
DISCLAIMER = "Educational paper-trading simulation. Not investment advice."

#: SPEC §4.4 requires this stated in the dashboard footer, not only the README.
SURVIVORSHIP_NOTICE = (
    "Backtest universe is a fixed current list, not point-in-time index "
    "membership. Absolute returns are overstated; treat them as an upper bound."
)


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def ratio(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001")))


class Holding(BaseModel):
    symbol: str
    quantity: int
    market_value: str
    weight: str
    sector: str | None = None


class PortfolioResponse(BaseModel):
    as_of: str
    total_value: str
    cash: str
    cash_weight: str
    holdings: list[Holding]
    disclaimer: str = DISCLAIMER


class PerformanceResponse(BaseModel):
    """TWR is the headline; MWR sits beside it so the gap can be explained."""

    periods: int
    annualized_twr: str
    annualized_benchmark_twr: str
    mwr: str | None
    annualized_volatility: str
    max_drawdown: str
    sharpe: str | None
    treynor: str | None
    information_ratio: str | None
    jensens_alpha: str
    alpha_t_stat: str | None
    #: False is a legitimate and common answer. Displayed prominently.
    alpha_is_significant: bool
    beta: str
    r_squared: str
    tracking_error: str
    equity_curve: list[str]
    benchmark_curve: list[str]
    timestamps: list[str]
    disclaimer: str = DISCLAIMER


class FrontierPointResponse(BaseModel):
    expected_return: str
    standard_deviation: str
    weights: dict[str, str]


class FrontierResponse(BaseModel):
    points: list[FrontierPointResponse]
    selected: FrontierPointResponse | None
    method: str
    disclaimer: str = DISCLAIMER


class VetoResponse(BaseModel):
    """One vetoed trade. SPEC §7 calls this the screen to demo first."""

    timestamp: str
    code: str
    symbol: str | None
    detail: str
    observed: str | None
    limit: str | None


class VetoesResponse(BaseModel):
    total: int
    by_code: dict[str, int]
    vetoes: list[VetoResponse]
    disclaimer: str = DISCLAIMER


class AttributionResponse(BaseModel):
    """Systematic versus diversifiable risk (SPEC §6.2)."""

    total_variance: str
    systematic_variance: str
    unsystematic_variance: str
    systematic_share: str
    beta: str
    disclaimer: str = DISCLAIMER


class AuditEntryResponse(BaseModel):
    timestamp: str
    actor: str
    code: str
    standard: str
    symbol: str | None
    detail: str


class AuditResponse(BaseModel):
    total: int
    by_code: dict[str, int]
    entries: list[AuditEntryResponse]
    disclaimer: str = DISCLAIMER


class CapabilitiesResponse(BaseModel):
    """What the configured executor can honor (SPEC §3.2).

    Surfaced in the UI because a constraint the executor cannot respect is
    advisory, and the operator should be able to see that without reading code.
    """

    engine_name: str
    engine_version: str
    supports_intraday: bool
    supports_participation_limits: bool
    supports_streaming_updates: bool
    advisory_constraints: list[str] = Field(default_factory=list)


class CycleSummary(BaseModel):
    timestamp: str
    decision: str
    note: str
    veto_codes: list[str]
    repair_codes: list[str]
    turnover: str | None
    shortfall_bps: str | None


class SystemStatus(BaseModel):
    """Everything a viewer needs to judge what they are looking at."""

    llm_provider: str
    executor: str
    cycles: int
    executed: int
    vetoed: int
    data_source: str
    disclaimer: str = DISCLAIMER
    survivorship_notice: str = SURVIVORSHIP_NOTICE


def format_weights(weights: dict[str, Decimal]) -> dict[str, str]:
    return {symbol: ratio(weight) for symbol, weight in sorted(weights.items())}


def format_series(values: Sequence[Decimal]) -> list[str]:
    return [money(value) for value in values]
