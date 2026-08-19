"""Read-only FastAPI surface for the dashboard.

There is no endpoint that places a trade, changes the policy, or overrides a
veto. The policy binds at runtime, and an HTTP route able to relax it would
make that untrue whatever the configuration said.

The app serves a snapshot from a completed run: the trading loop writes state,
the API reads it, and neither can block the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    DISCLAIMER,
    RATIO_FAMILIES,
    PricePointResponse,
    RatioRow,
    ResearchResponse,
    ScreenResponse,
    SymbolCard,
    ValuationResponse,
    AttributionResponse,
    AuditEntryResponse,
    AuditResponse,
    CapabilitiesResponse,
    CycleSummary,
    FrontierPointResponse,
    FrontierResponse,
    Holding,
    PerformanceResponse,
    PortfolioResponse,
    SystemStatus,
    VetoResponse,
    VetoesResponse,
    format_weights,
    format_series,
    money,
    ratio,
)
from src.api.research import ResearchService, SymbolProfile
from src.audit.log import AuditLog
from src.backtest.engine import BacktestResult
from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.cfa.portfolio import risk_decomposition
from src.execution.base import Capabilities
from src.risk.codes import Decision

ZERO = Decimal(0)


@dataclass(slots=True)
class DashboardState:
    """Everything the dashboard renders, assembled once from a finished run."""

    result: BacktestResult
    metrics: PerformanceMetrics
    capabilities: Capabilities
    audit: AuditLog = field(default_factory=AuditLog)
    sectors: Mapping[str, str] = field(default_factory=dict)
    llm_provider: str = "null"
    executor: str = "simulated"
    data_source: str = "synthetic"
    #: Present when the run was built from a resolved setup; drives the
    #: research endpoints. Absent for a bare backtest, which still serves the
    #: portfolio and performance views.
    research: ResearchService | None = None

    @classmethod
    def from_result(
        cls,
        result: BacktestResult,
        capabilities: Capabilities,
        audit: AuditLog | None = None,
        sectors: Mapping[str, str] | None = None,
        research: ResearchService | None = None,
        **labels: str,
    ) -> DashboardState:
        metrics = compute_metrics(
            result.equity_curve,
            result.benchmark_curve,
            risk_free_rate=result.config.risk_free_rate,
            cash_flows=list(result.cash_flows),
            periods_per_year=result.config.periods_per_year,
        )
        return cls(
            result=result,
            metrics=metrics,
            capabilities=capabilities,
            audit=audit or AuditLog(),
            sectors=sectors or {},
            research=research,
            **labels,
        )


def create_app(state: DashboardState) -> FastAPI:
    """Build the read-only dashboard API over a finished run."""
    app = FastAPI(
        title="Agentic Portfolio Manager",
        description=DISCLAIMER,
        version="1.0.0",
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "disclaimer": DISCLAIMER}

    @app.get("/api/status", response_model=SystemStatus)
    def status() -> SystemStatus:
        return SystemStatus(
            llm_provider=state.llm_provider,
            executor=state.executor,
            cycles=len(state.result.cycles),
            executed=len(state.result.executed),
            vetoed=len(state.result.vetoed),
            data_source=state.data_source,
        )

    @app.get("/api/portfolio", response_model=PortfolioResponse)
    def portfolio() -> PortfolioResponse:
        executed = state.result.executed
        if not executed:
            return PortfolioResponse(
                as_of=state.result.timestamps[-1].isoformat(),
                total_value=money(state.result.equity_curve[-1]),
                cash=money(state.result.equity_curve[-1]),
                cash_weight="1.000000",
                holdings=[],
            )

        report = executed[-1].report
        assert report is not None
        invested = sum((p.weight for p in report.final_positions), ZERO)
        return PortfolioResponse(
            as_of=executed[-1].timestamp.isoformat(),
            total_value=money(state.result.equity_curve[-1]),
            cash=money(report.final_cash),
            cash_weight=ratio(Decimal(1) - invested),
            holdings=[
                Holding(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    market_value=money(position.market_value),
                    weight=ratio(position.weight),
                    sector=state.sectors.get(position.symbol),
                )
                for position in report.final_positions
            ],
        )

    @app.get("/api/performance", response_model=PerformanceResponse)
    def performance() -> PerformanceResponse:
        m = state.metrics
        return PerformanceResponse(
            periods=m.periods,
            annualized_twr=ratio(m.annualized_twr),
            annualized_benchmark_twr=ratio(m.annualized_benchmark_twr),
            mwr=ratio(m.mwr) if m.mwr is not None else None,
            annualized_volatility=ratio(m.annualized_volatility),
            max_drawdown=ratio(m.max_drawdown),
            sharpe=ratio(m.sharpe) if m.sharpe is not None else None,
            treynor=ratio(m.treynor) if m.treynor is not None else None,
            information_ratio=(
                ratio(m.information_ratio) if m.information_ratio is not None else None
            ),
            jensens_alpha=ratio(m.jensens_alpha),
            alpha_t_stat=ratio(m.alpha_t_stat) if m.alpha_t_stat is not None else None,
            alpha_is_significant=m.alpha_is_significant,
            beta=ratio(m.beta),
            r_squared=ratio(m.r_squared),
            tracking_error=ratio(m.tracking_error),
            equity_curve=format_series(state.result.equity_curve),
            benchmark_curve=format_series(state.result.benchmark_curve),
            timestamps=[t.isoformat() for t in state.result.timestamps],
        )

    @app.get("/api/frontier", response_model=FrontierResponse)
    def frontier() -> FrontierResponse:
        latest = next(
            (c for c in reversed(state.result.cycles) if c.target is not None), None
        )
        if latest is None or latest.target is None:
            return FrontierResponse(points=[], selected=None, method="none")

        target = latest.target
        symbols = sorted(target.weights)
        points = [
            FrontierPointResponse(
                expected_return=ratio(point.expected_return),
                standard_deviation=ratio(point.standard_deviation),
                weights={},
            )
            for point in target.frontier
        ]
        selected = FrontierPointResponse(
            expected_return=ratio(target.expected_return),
            standard_deviation=ratio(target.volatility),
            weights=format_weights(dict(target.weights)),
        )
        return FrontierResponse(points=points, selected=selected, method=target.method)

    @app.get("/api/vetoes", response_model=VetoesResponse)
    def vetoes() -> VetoesResponse:
        entries: list[VetoResponse] = []
        by_code: dict[str, int] = {}
        for cycle in state.result.cycles:
            for violation in cycle.assessment.violations:
                by_code[violation.code.value] = by_code.get(violation.code.value, 0) + 1
                entries.append(
                    VetoResponse(
                        timestamp=cycle.timestamp.isoformat(),
                        code=violation.code.value,
                        symbol=violation.symbol,
                        detail=violation.detail,
                        observed=ratio(violation.observed)
                        if violation.observed is not None
                        else None,
                        limit=ratio(violation.limit) if violation.limit is not None else None,
                    )
                )
        return VetoesResponse(total=len(entries), by_code=by_code, vetoes=entries)

    @app.get("/api/attribution", response_model=AttributionResponse)
    def attribution() -> AttributionResponse:
        m = state.metrics
        benchmark_volatility = m.annualized_volatility
        try:
            decomposition = risk_decomposition(
                m.beta, benchmark_volatility, m.annualized_volatility
            )
        except Exception as exc:  # noqa: BLE001 - degenerate inputs are not fatal
            raise HTTPException(status_code=422, detail=f"attribution unavailable: {exc}")

        share = (
            decomposition.systematic_variance / decomposition.total_variance
            if decomposition.total_variance > ZERO
            else ZERO
        )
        return AttributionResponse(
            total_variance=ratio(decomposition.total_variance),
            systematic_variance=ratio(decomposition.systematic_variance),
            unsystematic_variance=ratio(decomposition.unsystematic_variance),
            systematic_share=ratio(share),
            beta=ratio(m.beta),
        )

    @app.get("/api/audit", response_model=AuditResponse)
    def audit() -> AuditResponse:
        entries = [
            AuditEntryResponse(
                timestamp=event.timestamp.isoformat(),
                actor=event.actor,
                code=event.code,
                standard=event.standard.value,
                symbol=event.symbol,
                detail=event.detail,
            )
            for event in sorted(state.audit.events, key=lambda e: e.timestamp)
        ]
        return AuditResponse(
            total=len(entries), by_code=state.audit.counts(), entries=entries
        )

    @app.get("/api/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        advisory: list[str] = []
        if not state.capabilities.supports_participation_limits:
            advisory.append("max_participation_rate")
        if not state.capabilities.supports_intraday:
            advisory.append("intraday_deadline")
        return CapabilitiesResponse(
            engine_name=state.capabilities.engine_name,
            engine_version=state.capabilities.engine_version,
            supports_intraday=state.capabilities.supports_intraday,
            supports_participation_limits=state.capabilities.supports_participation_limits,
            supports_streaming_updates=state.capabilities.supports_streaming_updates,
            advisory_constraints=advisory,
        )

    def _card(profile: SymbolProfile) -> SymbolCard:
        return SymbolCard(
            symbol=profile.symbol,
            sector=profile.sector,
            category=profile.category,
            beta=ratio(profile.beta) if profile.beta is not None else None,
            current_weight=ratio(profile.current_weight),
            latest_price=money(profile.latest_price)
            if profile.latest_price is not None
            else None,
            change_1d=ratio(profile.change_1d) if profile.change_1d is not None else None,
            change_ytd=ratio(profile.change_ytd) if profile.change_ytd is not None else None,
            volatility=ratio(profile.volatility) if profile.volatility is not None else None,
            has_fundamentals=profile.has_fundamentals,
        )

    @app.get("/api/screen", response_model=ScreenResponse)
    def screen() -> ScreenResponse:
        """Every investable instrument, with headline figures."""
        if state.research is None:
            raise HTTPException(status_code=503, detail="research data is not loaded")
        cards = [_card(p) for p in state.research.screen()]
        return ScreenResponse(
            as_of=state.research.as_of.isoformat(),
            data_source=state.data_source,
            count=len(cards),
            symbols=cards,
            sectors=sorted({c.sector for c in cards}),
        )

    @app.get("/api/research/{symbol}", response_model=ResearchResponse)
    def research(symbol: str) -> ResearchResponse:
        """Deep dive on one instrument."""
        if state.research is None:
            raise HTTPException(status_code=503, detail="research data is not loaded")

        known = {i.symbol for i in state.research.instruments()}
        wanted = symbol.upper()
        if wanted not in known:
            raise HTTPException(
                status_code=404,
                detail=f"{wanted} is not in the investable universe",
            )

        found = state.research.research(wanted)
        return ResearchResponse(
            profile=_card(found.profile),
            as_of=state.research.as_of.isoformat(),
            prices=[
                PricePointResponse(
                    t=point.timestamp.date().isoformat(),
                    close=money(point.close),
                    adjusted=money(point.adjusted),
                )
                for point in found.prices
            ],
            ratios=[
                RatioRow(
                    name=name,
                    value=ratio(value),
                    family=RATIO_FAMILIES.get(name, "Other"),
                )
                for name, value in sorted(found.ratios.items())
            ],
            valuation=(
                ValuationResponse(
                    method=found.valuation.method,
                    value=money(found.valuation.value)
                    if found.valuation.value is not None
                    else None,
                    reason=found.valuation.reason,
                )
                if found.valuation is not None
                else None
            ),
            enterprise_value=money(found.enterprise_value)
            if found.enterprise_value is not None
            else None,
            capm_required_return=ratio(found.capm_required_return)
            if found.capm_required_return is not None
            else None,
            fundamentals_period=(
                found.fundamentals.period_end.date().isoformat()
                if found.fundamentals is not None and found.fundamentals.period_end
                else None
            ),
            veto_codes=list(found.veto_codes),
            notes=list(found.notes),
        )

    @app.get("/api/cycles", response_model=list[CycleSummary])
    def cycles() -> list[CycleSummary]:
        return [
            CycleSummary(
                timestamp=cycle.timestamp.isoformat(),
                decision=cycle.assessment.decision.value,
                note=cycle.note,
                veto_codes=[v.code.value for v in cycle.assessment.violations],
                repair_codes=[r.code.value for r in cycle.assessment.repairs],
                turnover=ratio(cycle.report.realized_turnover) if cycle.report else None,
                shortfall_bps=(
                    ratio(cycle.report.implementation_shortfall_bps) if cycle.report else None
                ),
            )
            for cycle in state.result.cycles
        ]

    return app


def app_from_environment() -> FastAPI:
    """Build an app over the best data available.

    Real recorded market data when ``make backfill`` has run, synthetic
    otherwise — the same decision the results script and the scheduled Lambda
    make, taken once in :func:`src.data.live.resolve_setup` so the three cannot
    disagree about what they are displaying. ``/api/status`` reports which.
    """
    from src.backtest.engine import BacktestConfig, run_backtest
    from src.data.live import resolve_setup
    from src.data.universe import load_universe
    from src.execution.fill_models import SpreadCrossFillModel
    from src.execution.simulated import SimulatedExecutor
    from src.risk.ips import load_policy

    setup = resolve_setup()
    config = BacktestConfig(
        start=setup.start,
        end=setup.end,
        initial_cash=Decimal(os.environ.get("INITIAL_CASH", "100000.00")),
        symbols=setup.symbols,
        benchmark_symbol=setup.benchmark,
        estimation_window=100,
        market_return=setup.market_return,
        risk_free_rate=setup.risk_free_rate,
    )
    executor = SimulatedExecutor(fill_model=SpreadCrossFillModel())
    result = run_backtest(
        config, setup.source, executor, load_policy(), setup.sectors, setup.betas
    )

    # Realized weights from the last executed cycle, and every symbol the risk
    # engine has ever objected to — both feed the research view.
    weights: dict[str, Decimal] = {}
    if result.executed and result.executed[-1].report is not None:
        weights = dict(result.executed[-1].report.realized_weights)
    vetoes: dict[str, list[str]] = {}
    for cycle in result.cycles:
        for violation in cycle.assessment.violations:
            if violation.symbol:
                vetoes.setdefault(violation.symbol, []).append(violation.code.value)

    return create_app(
        DashboardState.from_result(
            result,
            capabilities=executor.capabilities(),
            sectors=setup.sectors,
            research=ResearchService(
                setup=setup,
                universe=load_universe(),
                current_weights=weights,
                veto_codes={k: tuple(v) for k, v in vetoes.items()},
            ),
            llm_provider=os.environ.get("LLM_PROVIDER", "null"),
            executor=os.environ.get("EXECUTOR", "simulated_spread"),
            data_source=setup.data_source,
        )
    )
