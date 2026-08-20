"""Lambda handlers, state store, and the WebSocket broadcaster."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

import pytest

from fastapi.testclient import TestClient

import src.api.routes as routes_module
from src.api.handler import run_cycle
from src.api.routes import CACHEABLE_ROUTES, app_from_environment, create_cached_app
from src.api.store import InMemoryStateStore, decode, encode
from src.api.ws import Broadcaster
from src.audit.log import AuditEvent, AuditLog, Standard
from src.time.clock import UTC, SimulationClock

D = Decimal
NOW = datetime(2024, 6, 3, 21, tzinfo=UTC)


# ---------------------------------------------------------------------------
# state store
# ---------------------------------------------------------------------------


def test_decimals_are_stored_as_strings() -> None:
    # DynamoDB's N type is arbitrary precision, but the boto3 round trip goes
    # through float unless you fight it. A weight stored as
    # 0.06999999999999999 is not the weight the risk engine approved.
    stored = encode({"weight": D("0.07"), "nested": [{"value": D("1.005")}]})
    assert stored == {"weight": "0.07", "nested": [{"value": "1.005"}]}


def test_exponent_notation_is_normalized() -> None:
    assert encode({"v": D("1E+2")}) == {"v": "100"}


def test_decode_undoes_boto3_reconstructing_ints_as_decimal() -> None:
    # boto3's resource API hands back every DynamoDB N-type attribute as
    # Decimal, including a plain int a caller wrote — json.dumps chokes on
    # that later (as a live 500 confirmed), so a read must convert it back.
    stored = {"cycles": D("21"), "by_code": {"MAX_SECTOR_WEIGHT": D("3")}, "symbol": "AAPL"}
    decoded = decode(stored)
    assert decoded == {"cycles": 21, "by_code": {"MAX_SECTOR_WEIGHT": 3}, "symbol": "AAPL"}
    assert type(decoded["cycles"]) is int


def test_in_memory_store_round_trips() -> None:
    store = InMemoryStateStore()
    store.put_cycle("mandate-abc", {"decision": "APPROVED", "equity": D("100.5")})

    assert store.get_cycle("mandate-abc") == {"decision": "APPROVED", "equity": "100.5"}
    assert store.get_cycle("missing") is None


def test_writing_the_same_cycle_twice_overwrites() -> None:
    # EventBridge delivers at least once. A replayed schedule must not produce
    # two records of the same decision.
    store = InMemoryStateStore()
    store.put_cycle("mandate-abc", {"decision": "APPROVED"})
    store.put_cycle("mandate-abc", {"decision": "APPROVED"})

    assert len(store.cycles) == 1


def test_audit_events_are_appended() -> None:
    store = InMemoryStateStore()
    store.append_audit(
        [AuditEvent(NOW, "risk", "MAX_SECTOR_WEIGHT", Standard.III_C_SUITABILITY, "capped")]
    )
    assert store.audit[0]["code"] == "MAX_SECTOR_WEIGHT"


# ---------------------------------------------------------------------------
# scheduled cycle
# ---------------------------------------------------------------------------


def test_a_scheduled_cycle_persists_its_result() -> None:
    store = InMemoryStateStore()
    payload = run_cycle(store=store, clock=SimulationClock(NOW))

    assert payload["decision"]
    assert store.latest_snapshot() is not None
    assert store.cycles


def test_a_scheduled_cycle_says_what_data_it_used() -> None:
    # A cycle that silently invented data would be far worse than one that
    # reports it is running on a simulation. The claim must match reality
    # whichever source is available, so it is compared against the resolver
    # rather than hardcoded to either answer.
    from src.data.live import resolve_setup

    payload = run_cycle(store=InMemoryStateStore(), clock=SimulationClock(NOW))
    assert payload["data_source"] == resolve_setup().data_source
    assert payload["data_source"], "the cycle must always state its data source"


def test_replaying_a_cycle_is_idempotent() -> None:
    store = InMemoryStateStore()
    clock = SimulationClock(NOW)
    first = run_cycle(store=store, clock=clock)
    second = run_cycle(store=store, clock=clock)

    assert first["cycle_id"] == second["cycle_id"]
    assert len(store.cycles) == 1, "the replay must overwrite, not append"


def test_a_cycle_uses_the_injected_clock() -> None:
    # No module reads the wall clock, including the Lambda handler.
    payload = run_cycle(store=InMemoryStateStore(), clock=SimulationClock(NOW))
    assert payload["as_of"] == NOW.isoformat()


def test_build_view_pipeline_uses_recorded_data_when_present() -> None:
    from src.api.handler import _build_view_pipeline
    from src.data.live import DEFAULT_CACHE_ROOT, resolve_setup
    from src.llm.cache import ResilientProvider

    setup = resolve_setup()
    pipeline = _build_view_pipeline(setup, AuditLog())

    assert isinstance(pipeline.research.provider, ResilientProvider)
    assert isinstance(pipeline.fundamental.provider, ResilientProvider)
    assert isinstance(pipeline.macro.provider, ResilientProvider)
    # Real when make backfill has run on this machine; empty (never
    # fabricated) otherwise — either way, never left unwired.
    if DEFAULT_CACHE_ROOT.exists():
        assert pipeline.fundamentals or pipeline.macro_signals is not None
    else:
        assert pipeline.fundamentals == {}
        assert pipeline.macro_signals is None


def test_build_store_falls_back_to_memory_without_a_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # A misconfigured deploy should degrade to "runs but does not persist",
    # which is visible, rather than to a Lambda that cannot start.
    from src.api.handler import build_store

    monkeypatch.delenv("STATE_TABLE", raising=False)
    assert isinstance(build_store(), InMemoryStateStore)


# ---------------------------------------------------------------------------
# snapshot cache: the API should not replay the backtest once a cycle exists
# ---------------------------------------------------------------------------


def _persisted_routes(store: InMemoryStateStore) -> Mapping[str, Any]:
    snapshot = store.latest_snapshot()
    assert snapshot is not None
    return snapshot["routes"]  # type: ignore[no-any-return]


def test_a_cycle_persists_a_rendered_snapshot() -> None:
    store = InMemoryStateStore()
    run_cycle(store=store, clock=SimulationClock(NOW))

    snapshot = store.latest_snapshot()
    assert snapshot is not None
    routes = snapshot["routes"]
    assert set(routes) == set(CACHEABLE_ROUTES)
    # Every cached payload is what a live call to that route actually
    # returned — not a placeholder — so /api/status should report real counts.
    assert routes["/api/status"]["cycles"] > 0


def test_cached_routes_never_touch_the_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryStateStore()
    run_cycle(store=store, clock=SimulationClock(NOW))
    routes = _persisted_routes(store)

    def _boom() -> None:
        raise AssertionError("a cached route recomputed the backtest")

    monkeypatch.setattr(routes_module, "build_dashboard_state", _boom)
    client = TestClient(create_cached_app(routes, store))

    for path in CACHEABLE_ROUTES:
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == routes[path]


def test_a_cycle_persists_research_for_every_universe_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.universe import load_universe

    store = InMemoryStateStore()
    run_cycle(store=store, clock=SimulationClock(NOW))

    universe_symbols = {i.symbol for i in load_universe().instruments}
    assert set(store.research) == universe_symbols

    def _boom() -> None:
        raise AssertionError("research recomputed the backtest instead of reading the cache")

    monkeypatch.setattr(routes_module, "build_dashboard_state", _boom)
    routes = _persisted_routes(store)
    client = TestClient(create_cached_app(routes, store))

    aapl = client.get("/api/research/AAPL")
    assert aapl.status_code == 200
    assert aapl.json()["profile"]["symbol"] == "AAPL"
    # Case-insensitive, matching create_app's live behavior.
    assert client.get("/api/research/aapl").status_code == 200


def test_cached_app_404s_an_unknown_symbol_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStateStore()
    run_cycle(store=store, clock=SimulationClock(NOW))

    def _boom() -> None:
        raise AssertionError("an unknown symbol triggered a live recompute")

    monkeypatch.setattr(routes_module, "build_dashboard_state", _boom)
    routes = _persisted_routes(store)
    client = TestClient(create_cached_app(routes, store))

    assert client.get("/api/research/NOTREAL").status_code == 404


def test_app_from_environment_prefers_a_persisted_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStateStore()
    run_cycle(store=store, clock=SimulationClock(NOW))

    monkeypatch.setattr(routes_module, "build_store", lambda: store)
    monkeypatch.setattr(
        routes_module,
        "build_dashboard_state",
        lambda: (_ for _ in ()).throw(AssertionError("recomputed despite a snapshot")),
    )

    client = TestClient(app_from_environment())
    assert client.get("/api/status").status_code == 200


def test_app_from_environment_falls_back_live_with_no_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "build_store", lambda: InMemoryStateStore())
    client = TestClient(app_from_environment())
    assert client.get("/api/status").status_code == 200


# ---------------------------------------------------------------------------
# webSocket broadcaster
# ---------------------------------------------------------------------------


def test_broadcaster_fans_out_to_subscribers() -> None:
    broadcaster = Broadcaster()
    a = broadcaster.subscribe()
    b = broadcaster.subscribe()

    assert broadcaster.publish("cycle", {"decision": "APPROVED"}) == 2
    assert "APPROVED" in a.get_nowait()
    assert "APPROVED" in b.get_nowait()


def test_every_pushed_message_carries_the_disclaimer() -> None:
    broadcaster = Broadcaster()
    queue = broadcaster.subscribe()
    broadcaster.publish("cycle", {})
    assert "Not investment advice" in queue.get_nowait()


def test_a_stalled_subscriber_cannot_block_the_publisher() -> None:
    # Blocking the trading loop to wait on a dashboard would be the wrong
    # trade every time.
    broadcaster = Broadcaster()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    broadcaster.subscribers.append(queue)

    assert broadcaster.publish("a", {}) == 1
    assert broadcaster.publish("b", {}) == 0  # dropped for that client alone


def test_unsubscribe_removes_the_queue() -> None:
    broadcaster = Broadcaster()
    queue = broadcaster.subscribe()
    assert broadcaster.subscriber_count == 1
    broadcaster.unsubscribe(queue)
    assert broadcaster.subscriber_count == 0
