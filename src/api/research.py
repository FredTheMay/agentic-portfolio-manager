"""Per-symbol research service (M11).

Assembles everything the system knows about one instrument into a single
response: price history, point-in-time fundamentals, the CFA ratio table,
valuation under the SPEC §6.5 hierarchy, the regression-estimated beta, and
whether the risk engine has ever vetoed a trade in it.

Every number here is computed by :mod:`src.cfa` from recorded data. The service
assembles and formats; it does not calculate, and it never asks a model for a
figure (SPEC §2.1).

Fundamentals are read ``as_of`` the end of the recorded window rather than
"now", so a research page shows what was actually knowable at the point the
backtest ended — the same point-in-time discipline the backtest runs under
(SPEC §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

from src.agents.fundamental import ratio_table
from src.cfa.portfolio import capm_expected_return
from src.cfa.valuation import (
    ValuationResult,
    enterprise_value,
    sustainable_growth_rate,
    value_equity,
)
from src.data.cache import CachingFetcher, ResponseCache
from src.data.edgar import EdgarClient, Fundamentals, total_debt
from src.data.events import BarPayload
from src.data.live import BacktestSetup, DEFAULT_CACHE_ROOT
from src.data.sources import InMemoryEventSource
from src.data.universe import Instrument, Universe

ZERO = Decimal(0)
ONE = Decimal(1)

#: Payout assumption when a filer reports no dividend. Most large-cap tech pays
#: none, which is exactly why SPEC §6.5 requires an FCFE fallback.
DEFAULT_PAYOUT = Decimal("0.30")


@dataclass(frozen=True, slots=True)
class PricePoint:
    timestamp: datetime
    close: Decimal
    adjusted: Decimal


@dataclass(frozen=True, slots=True)
class SymbolProfile:
    """The headline card for one instrument."""

    symbol: str
    sector: str
    category: str
    beta: Decimal | None
    r_squared: Decimal | None
    current_weight: Decimal
    latest_price: Decimal | None
    change_1d: Decimal | None
    change_ytd: Decimal | None
    volatility: Decimal | None
    has_fundamentals: bool


@dataclass(frozen=True, slots=True)
class SymbolResearch:
    """Everything known about one instrument."""

    profile: SymbolProfile
    prices: tuple[PricePoint, ...]
    ratios: Mapping[str, Decimal]
    fundamentals: Fundamentals | None
    valuation: ValuationResult | None
    enterprise_value: Decimal | None
    capm_required_return: Decimal | None
    veto_codes: tuple[str, ...]
    notes: tuple[str, ...]


def _returns(prices: Sequence[Decimal]) -> list[Decimal]:
    return [b / a - ONE for a, b in zip(prices, prices[1:]) if a > ZERO]


def _stdev(values: Sequence[Decimal]) -> Decimal | None:
    if len(values) < 2:
        return None
    mean = sum(values, ZERO) / Decimal(len(values))
    variance = sum(((v - mean) ** 2 for v in values), ZERO) / Decimal(len(values) - 1)
    return variance.sqrt() if variance > ZERO else ZERO


class ResearchService:
    """Assembles research responses from recorded data.

    Price series and fundamentals are loaded once and held, because a research
    page is read far more often than the underlying recording changes and
    re-parsing a companyfacts document per request would dominate the response
    time.
    """

    def __init__(
        self,
        setup: BacktestSetup,
        universe: Universe,
        current_weights: Mapping[str, Decimal] | None = None,
        veto_codes: Mapping[str, tuple[str, ...]] | None = None,
        cache_root: Path | None = None,
    ) -> None:
        self._setup = setup
        self._universe = universe
        self._weights = dict(current_weights or {})
        self._vetoes = dict(veto_codes or {})
        self._cache_root = cache_root or DEFAULT_CACHE_ROOT
        self._series = self._build_series(setup.source)
        self._fundamentals: dict[str, Fundamentals | None] = {}
        self._edgar: EdgarClient | None = None

    @staticmethod
    def _build_series(source: InMemoryEventSource) -> dict[str, list[PricePoint]]:
        series: dict[str, list[PricePoint]] = {}
        for event in source.events:
            if not isinstance(event.payload, BarPayload):
                continue
            series.setdefault(event.symbol, []).append(
                PricePoint(
                    timestamp=event.timestamp,
                    close=event.payload.close,
                    adjusted=event.payload.adj_close or event.payload.close,
                )
            )
        return series

    @property
    def as_of(self) -> datetime:
        return self._setup.end

    def instruments(self) -> tuple[Instrument, ...]:
        return self._universe.instruments

    def _edgar_client(self) -> EdgarClient | None:
        """A cache-only EDGAR client, or ``None`` when nothing was recorded."""
        if self._edgar is None:
            root = self._cache_root
            if not root.exists():
                return None
            self._edgar = EdgarClient(
                CachingFetcher(ResponseCache(root=root), offline=True)
            )
        return self._edgar

    def fundamentals_for(self, symbol: str) -> Fundamentals | None:
        if symbol in self._fundamentals:
            return self._fundamentals[symbol]

        client = self._edgar_client()
        result: Fundamentals | None = None
        if client is not None:
            try:
                result = client.get_fundamentals(symbol, self.as_of)
            except Exception:  # noqa: BLE001 - an ETF has no filings; not an error
                result = None
        self._fundamentals[symbol] = result
        return result

    def profile(self, symbol: str) -> SymbolProfile:
        points = self._series.get(symbol, [])
        closes = [p.adjusted for p in points]
        sectors = self._universe.sectors
        instrument = next(
            (i for i in self._universe.instruments if i.symbol == symbol), None
        )

        change_1d = None
        if len(closes) >= 2 and closes[-2] > ZERO:
            change_1d = closes[-1] / closes[-2] - ONE

        change_ytd = None
        if points:
            year = points[-1].timestamp.year
            start = next((p.adjusted for p in points if p.timestamp.year == year), None)
            if start and start > ZERO:
                change_ytd = closes[-1] / start - ONE

        window = _returns(closes[-252:]) if len(closes) > 2 else []
        sd = _stdev(window)
        volatility = sd * Decimal(252).sqrt() if sd is not None else None

        beta = self._setup.betas.get(symbol)
        return SymbolProfile(
            symbol=symbol,
            sector=sectors.get(symbol, "UNKNOWN"),
            category=instrument.category if instrument else "UNKNOWN",
            beta=beta,
            r_squared=None,
            current_weight=self._weights.get(symbol, ZERO),
            latest_price=points[-1].close if points else None,
            change_1d=change_1d,
            change_ytd=change_ytd,
            volatility=volatility,
            has_fundamentals=self.fundamentals_for(symbol) is not None,
        )

    def research(self, symbol: str, price_points: int = 400) -> SymbolResearch:
        """Assemble the full research view for one instrument."""
        profile = self.profile(symbol)
        points = tuple(self._series.get(symbol, [])[-price_points:])
        fundamentals = self.fundamentals_for(symbol)
        notes: list[str] = []

        ratios: Mapping[str, Decimal] = {}
        valuation: ValuationResult | None = None
        ev: Decimal | None = None

        if fundamentals is not None:
            ratios = ratio_table(fundamentals)
            if not ratios:
                notes.append("Filings are visible but too sparse to compute ratios.")
        else:
            notes.append(
                "No SEC filings recorded for this instrument — expected for an ETF."
            )

        required = None
        if profile.beta is not None:
            required = capm_expected_return(
                self._setup.risk_free_rate, profile.beta, self._setup.market_return
            )

        if fundamentals is not None and required is not None:
            roe = ratios.get("return_on_equity")
            growth = (
                sustainable_growth_rate(DEFAULT_PAYOUT, roe) if roe is not None else None
            )
            if growth is not None and growth < required:
                fcfe = None
                if (
                    fundamentals.cash_flow_operations is not None
                    and fundamentals.net_income is not None
                ):
                    # No capex tag is mapped, so FCFE is approximated by CFO.
                    # Stated as a note rather than presented as a clean figure.
                    fcfe = fundamentals.cash_flow_operations
                    notes.append(
                        "FCFE approximated by cash flow from operations: no capital "
                        "expenditure tag is mapped, so reinvestment is not deducted."
                    )
                valuation = value_equity(
                    required_return=required,
                    growth_rate=growth,
                    dividend_next=None,
                    fcfe_next=fcfe,
                )
            elif growth is not None:
                notes.append(
                    f"Sustainable growth ({growth:.2%}) is not below the required return "
                    f"({required:.2%}), so constant-growth models do not converge."
                )

            if (
                profile.latest_price is not None
                and fundamentals.total_assets is not None
                and fundamentals.cash is not None
            ):
                debt = total_debt(fundamentals)
                if debt is not None:
                    # Market cap is unavailable without a share count, so EV is
                    # reported on book equity and labelled as such.
                    equity_book = fundamentals.total_equity
                    if equity_book is not None:
                        ev = enterprise_value(equity_book, debt, fundamentals.cash)
                        notes.append(
                            "Enterprise value uses book equity, not market "
                            "capitalisation: no shares-outstanding tag is mapped."
                        )

        return SymbolResearch(
            profile=profile,
            prices=points,
            ratios=ratios,
            fundamentals=fundamentals,
            valuation=valuation,
            enterprise_value=ev,
            capm_required_return=required,
            veto_codes=self._vetoes.get(symbol, ()),
            notes=tuple(notes),
        )

    def screen(self) -> list[SymbolProfile]:
        """Every instrument's headline card, for the browse view."""
        return [self.profile(i.symbol) for i in self._universe.instruments]


def parse_decimal(value: str, field: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
