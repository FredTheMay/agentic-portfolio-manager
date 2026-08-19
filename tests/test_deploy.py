"""Lambda handlers, state store, and the WebSocket broadcaster."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal

import pytest

from src.api.handler import run_cycle
from src.api.store import InMemoryStateStore, encode
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


def test_build_store_falls_back_to_memory_without_a_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # A misconfigured deploy should degrade to "runs but does not persist",
    # which is visible, rather than to a Lambda that cannot start.
    from src.api.handler import build_store

    monkeypatch.delenv("STATE_TABLE", raising=False)
    assert isinstance(build_store(), InMemoryStateStore)


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
