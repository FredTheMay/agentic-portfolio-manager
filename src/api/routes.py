"""FastAPI surface for the dashboard (SPEC §9, M9).

Read-only. There is no endpoint that places a trade, changes the IPS, or
overrides a veto — CFA Standard III(A) requires the policy to bind at runtime,
and an HTTP route that could relax it would make that untrue no matter what the
YAML said.

The app serves a snapshot produced by a completed run
(:class:`DashboardState`). It is deliberately not wired to a live trading loop:
the loop writes state, the API reads it, and neither can block the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    DISCLAIMER,
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

    @classmethod
    def from_result(
        cls,
        result: BacktestResult,
        capabilities: Capabilities,
        audit: AuditLog | None = None,
        sectors: Mapping[str, str] | None = None,
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
    """Build an app over a freshly computed synthetic run.

    Used by ``make serve`` and the Lambda handler. With no API keys configured
    there is nothing recorded to replay, so the dashboard shows a synthetic run
    and labels it as such in ``/api/status``.
    """
    from datetime import timedelta

    from src.backtest.engine import BacktestConfig, run_backtest
    from src.execution.fill_models import SpreadCrossFillModel
    from src.execution.simulated import SimulatedExecutor
    from src.risk.ips import load_policy
    from src.time.clock import UTC
    from src.data.synthetic import BETAS, SECTORS, make_source

    from datetime import datetime

    start = datetime(2022, 1, 3, 21, tzinfo=UTC)
    config = BacktestConfig(
        start=start,
        end=start + timedelta(days=730),
        initial_cash=Decimal("100000.00"),
        symbols=(
            "AAA", "BBB", "CCC", "DDD", "EEE", "FFF",
            "GGG", "HHH", "III", "JJJ", "KKK", "LLL",
        ),
        benchmark_symbol="SPY",
        estimation_window=100,
    )
    executor = SimulatedExecutor(fill_model=SpreadCrossFillModel())
    result = run_backtest(
        config, make_source(days=760), executor, load_policy(), SECTORS, BETAS
    )
    return create_app(
        DashboardState.from_result(
            result,
            capabilities=executor.capabilities(),
            sectors=SECTORS,
            llm_provider=os.environ.get("LLM_PROVIDER", "null"),
            executor=os.environ.get("EXECUTOR", "simulated_spread"),
            data_source="synthetic (no API keys configured)",
        )
    )
