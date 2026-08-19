"""API response models.

Every monetary and ratio value crosses the wire as a decimal string. JSON
numbers are IEEE-754 doubles, and a weight round-tripped through one is no
longer the weight the risk engine approved.

The disclaimer is a required field rather than a template detail, so no screen
can render without it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from pydantic import BaseModel, Field

#: . Carried on every response, not just the landing page.
DISCLAIMER = "Educational paper-trading simulation. Not investment advice."

#: this belongs in the dashboard footer, not only the README.
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
    """One vetoed trade. this is the screen to demo first."""

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
    """Systematic versus diversifiable risk."""

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
    """What the configured executor can honor.

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


# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------


class SymbolCard(BaseModel):
    """One instrument's headline figures, for the screener grid."""

    symbol: str
    sector: str
    category: str
    beta: str | None
    current_weight: str
    latest_price: str | None
    change_1d: str | None
    change_ytd: str | None
    volatility: str | None
    has_fundamentals: bool


class ScreenResponse(BaseModel):
    as_of: str
    data_source: str
    count: int
    symbols: list[SymbolCard]
    sectors: list[str]
    disclaimer: str = DISCLAIMER


class PricePointResponse(BaseModel):
    t: str
    close: str
    adjusted: str


class RatioRow(BaseModel):
    name: str
    value: str
    #: Which CFA family the ratio belongs to, for grouping in the UI.
    family: str


class ValuationResponse(BaseModel):
    method: str
    value: str | None
    reason: str


class ResearchResponse(BaseModel):
    """Everything the system knows about one instrument."""

    profile: SymbolCard
    as_of: str
    prices: list[PricePointResponse]
    ratios: list[RatioRow]
    valuation: ValuationResponse | None
    enterprise_value: str | None
    capm_required_return: str | None
    fundamentals_period: str | None
    veto_codes: list[str]
    #: Caveats about *this* instrument's data — an unmapped tag, a model that
    #: does not converge. Surfaced rather than silently degrading the numbers.
    notes: list[str]
    disclaimer: str = DISCLAIMER


#: Ratio name -> CFA family, for grouping. Anything unlisted is "Other".
RATIO_FAMILIES: dict[str, str] = {
    "current_ratio": "Liquidity",
    "quick_ratio": "Liquidity",
    "debt_to_equity": "Solvency",
    "interest_coverage": "Solvency",
    "gross_margin": "Profitability",
    "operating_margin": "Profitability",
    "net_margin": "Profitability",
    "return_on_assets": "Profitability",
    "return_on_equity": "Profitability",
    "inventory_turnover": "Activity",
    "receivables_turnover": "Activity",
    "total_asset_turnover": "Activity",
    "accruals_ratio": "Earnings quality",
    "dupont_net_margin": "DuPont",
    "dupont_asset_turnover": "DuPont",
    "dupont_equity_multiplier": "DuPont",
}
