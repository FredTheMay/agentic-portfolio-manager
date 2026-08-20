"""Lambda entry points: the scheduled decision cycle and the dashboard API.

EventBridge delivers at least once, and a Lambda retry after a timeout is
indistinguishable from a fresh invocation. Three independent defenses cover
that: content-hashed mandate ids, broker-side rejection of a duplicate
``client_order_id``, and writing a cycle as a ``put_item`` on that id so a
replay overwrites rather than appending.
"""

from __future__ import annotations

import os
from datetime import timedelta
from decimal import Decimal
from typing import Any, Mapping

from src.agents.pipeline import AgentViewPipeline, ViewPipeline
from src.api.store import REGION_ENV, TABLE_ENV, StateStore, build_store
from src.audit.log import AuditEvent, AuditLog, Standard
from src.data.live import BacktestSetup
from src.time.clock import Clock, WallClock

ZERO = Decimal(0)

__all__ = [
    "REGION_ENV",
    "TABLE_ENV",
    "api_handler",
    "build_store",
    "run_cycle",
    "scheduled_cycle",
]


def _build_view_pipeline(setup: BacktestSetup, audit: AuditLog) -> AgentViewPipeline:
    """Real Fundamental + Macro agents over already-recorded data.

    Research stays NEUTRAL: no headline source exists anywhere in this
    codebase, and ResearchAgent's own design already treats "no headlines" as
    a legitimate state, not a failure — fabricating one would be worse than
    leaving it unset. Wrapped in ResilientProvider so a malformed response or
    a rate limit degrades one symbol's view to NEUTRAL rather than raising
    partway through ~500 calls (23 symbols x 21 cycles) and losing the cycle.
    """
    from src.agents.fundamental import FundamentalAgent
    from src.agents.macro import MacroAgent, read_signals
    from src.agents.research import ResearchAgent
    from src.data.cache import CachingFetcher, ResponseCache
    from src.data.edgar import EdgarClient, Fundamentals
    from src.data.fred import CPI, FED_FUNDS, TERM_SPREAD, UNEMPLOYMENT, FredClient
    from src.data.live import DEFAULT_CACHE_ROOT
    from src.llm import get_provider
    from src.llm.cache import ResilientProvider

    provider = ResilientProvider(inner=get_provider())
    root = DEFAULT_CACHE_ROOT

    fundamentals: dict[str, Fundamentals] = {}
    macro_signals = None
    if root.exists():
        edgar = EdgarClient(CachingFetcher(ResponseCache(root=root), offline=True))
        for symbol in setup.symbols:
            try:
                current = edgar.get_fundamentals(symbol, setup.end)
            except Exception:  # noqa: BLE001 - an ETF has no filings; not an error
                current = None
            if current is not None:
                fundamentals[symbol] = current

        fred = FredClient(CachingFetcher(ResponseCache(root=root), offline=True))
        try:
            macro_signals = read_signals(
                setup.end,
                term_spread=fred.series(TERM_SPREAD),
                unemployment=fred.series(UNEMPLOYMENT),
                cpi=fred.series(CPI),
                fed_funds=fred.series(FED_FUNDS),
            )
        except Exception:  # noqa: BLE001 - degenerate/missing series is not fatal
            macro_signals = None

    return AgentViewPipeline(
        research=ResearchAgent(provider=provider, audit=audit),
        fundamental=FundamentalAgent(provider=provider, audit=audit),
        macro=MacroAgent(provider=provider),
        fundamentals=fundamentals,
        macro_signals=macro_signals,
        audit=audit,
    )


def run_cycle(
    store: StateStore | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Run one decision cycle and persist the outcome.

    Uses the synthetic source when no market data has been recorded, and says
    so in the returned payload. A cycle that silently invents data would be far
    worse than one that reports it is running on a simulation.
    """
    from src.api.routes import (
        _dashboard_state_from_result,
        render_research_snapshot,
        render_snapshot,
    )
    from src.backtest.engine import BacktestConfig, run_backtest
    from src.data.live import resolve_setup
    from src.execution import get_executor
    from src.execution.simulated import SimulatedExecutor
    from src.risk.ips import load_policy

    active_store = store or build_store()
    now = (clock or WallClock()).now()
    audit = AuditLog()

    executor = get_executor()
    setup = resolve_setup()
    used_executor = executor if isinstance(executor, SimulatedExecutor) else SimulatedExecutor()
    view_pipeline = _build_view_pipeline(setup, audit)
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
    result = run_backtest(
        config,
        setup.source,
        used_executor,
        load_policy(),
        setup.sectors,
        setup.betas,
        views=view_pipeline,
    )

    latest = result.cycles[-1] if result.cycles else None
    cycle_id = (
        latest.mandate.mandate_id
        if latest is not None and latest.mandate is not None
        else f"no-trade-{now.isoformat()}"
    )

    payload: dict[str, Any] = {
        "cycle_id": cycle_id,
        "as_of": now.isoformat(),
        "decision": latest.assessment.decision.value if latest else "NO_CYCLE",
        "note": latest.note if latest else "no rebalance was due",
        "equity": result.equity_curve[-1],
        "cycles": len(result.cycles),
        "executed": len(result.executed),
        "vetoed": len(result.vetoed),
        "data_source": setup.data_source,
    }

    audit.record(
        AuditEvent(
            timestamp=now,
            actor="scheduled_cycle",
            code="CYCLE_COMPLETED",
            standard=Standard.III_A_LOYALTY,
            detail=(
                f"{payload['decision']} — {payload['executed']} executed, "
                f"{payload['vetoed']} vetoed"
            ),
        )
    )

    # put_item on the mandate id: a replayed schedule overwrites rather than
    # appending a second record of the same decision.
    active_store.put_cycle(cycle_id, payload)

    # Pre-render every dashboard route now, while the result is already in
    # hand, so the API Lambda serves a DynamoDB read instead of replaying this
    # same backtest on every cold container. Kept out of the per-cycle record
    # above — that history accumulates indefinitely, and only the single
    # latest snapshot needs to carry the full rendered payload.
    state = _dashboard_state_from_result(result, used_executor, setup, audit=audit)
    active_store.put_snapshot({**payload, "routes": render_snapshot(state)})
    for symbol, research_payload in render_research_snapshot(state).items():
        active_store.put_research(symbol, research_payload)
    active_store.append_audit(list(audit))
    return payload


def scheduled_cycle(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
    """EventBridge entry point."""
    payload = run_cycle()
    return {"statusCode": 200, "body": payload}


def api_handler(event: Mapping[str, Any], context: Any) -> Any:
    """Lambda Function URL entry point, adapted to the FastAPI app via Mangum.

    Built fresh every invocation, deliberately not cached module-level for a
    warm container's whole lifetime. app_from_environment() already reads the
    persisted snapshot first, which is a single fast DynamoDB read — freezing
    the built app at cold start would mean a warm container keeps serving
    whatever snapshot existed at its first request forever, never noticing a
    newer one (a scheduled cycle's fresh result, or a corrected replay) until
    that container happens to recycle.
    """
    from mangum import Mangum

    from src.api.routes import app_from_environment

    return Mangum(app_from_environment())(event, context)
