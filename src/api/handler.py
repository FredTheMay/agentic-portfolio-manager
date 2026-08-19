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

from src.api.store import InMemoryStateStore, StateStore
from src.audit.log import AuditEvent, AuditLog, Standard
from src.time.clock import Clock, WallClock

ZERO = Decimal(0)

TABLE_ENV = "STATE_TABLE"
REGION_ENV = "AWS_REGION"


def build_store() -> StateStore:
    """DynamoDB when a table is configured, memory otherwise.

    Falling back to memory rather than failing means a misconfigured deployment
    degrades to "runs but does not persist", which is visible in the dashboard,
    instead of a Lambda that cannot start.
    """
    table = os.environ.get(TABLE_ENV)
    if not table:
        return InMemoryStateStore()
    from src.api.store import DynamoStateStore

    return DynamoStateStore(table_name=table, region=os.environ.get(REGION_ENV))


def run_cycle(
    store: StateStore | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Run one decision cycle and persist the outcome.

    Uses the synthetic source when no market data has been recorded, and says
    so in the returned payload. A cycle that silently invents data would be far
    worse than one that reports it is running on a simulation.
    """
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
        executor if isinstance(executor, SimulatedExecutor) else SimulatedExecutor(),
        load_policy(),
        setup.sectors,
        setup.betas,
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
    active_store.put_snapshot(payload)
    active_store.append_audit(list(audit))
    return payload


def scheduled_cycle(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
    """EventBridge entry point."""
    payload = run_cycle()
    return {"statusCode": 200, "body": payload}


def api_handler() -> Any:
    """API Gateway entry point, wrapped by Mangum in the Lambda image."""
    from src.api.routes import app_from_environment

    return app_from_environment()
