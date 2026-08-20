"""Persistence for cycle state and the audit trail.

Everything is stored as a decimal string. DynamoDB's ``N`` type is
arbitrary-precision, but the boto3 round trip goes through ``float`` unless you
fight it, and a weight stored as 0.06999999999999999 is not the weight the risk
engine approved.

``boto3`` is imported inside the constructor so the core package neither
depends on it nor pays to import it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from src.audit.log import AuditEvent

#: Partition key prefix for one rebalance cycle.
CYCLE_PREFIX = "CYCLE#"
#: Partition key for the most recent portfolio snapshot.
SNAPSHOT_KEY = "SNAPSHOT#latest"
AUDIT_PREFIX = "AUDIT#"
#: One item per symbol, not folded into SNAPSHOT_KEY: 400 price points across
#: a ~28-symbol universe is roughly 700KB combined, over DynamoDB's 400KB
#: single-item limit.
RESEARCH_PREFIX = "RESEARCH#"

TABLE_ENV = "STATE_TABLE"
REGION_ENV = "AWS_REGION"


def encode(value: Any) -> Any:
    """Convert a payload for storage, turning every Decimal into a string."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value: Any) -> Any:
    """Undo boto3's reconstruction of DynamoDB's N type as Decimal.

    encode() turns every real Decimal into a string before a write, so the
    only field that reaches DynamoDB's N type is a plain int (a count, a
    total). boto3's resource API deserializes *every* N-type attribute back
    as Decimal regardless of what was originally written — there is no way to
    ask it not to — so a read has to convert it back, or a consumer expecting
    a plain int (json.dumps, most of all) breaks on something that round
    tripped through storage rather than something the caller ever produced.
    """
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, Mapping):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


@runtime_checkable
class StateStore(Protocol):
    """Where a scheduled cycle leaves its results for the dashboard to read."""

    def put_cycle(self, cycle_id: str, payload: Mapping[str, Any]) -> None: ...

    def get_cycle(self, cycle_id: str) -> Mapping[str, Any] | None: ...

    def latest_snapshot(self) -> Mapping[str, Any] | None: ...

    def put_snapshot(self, payload: Mapping[str, Any]) -> None: ...

    def get_research(self, symbol: str) -> Mapping[str, Any] | None: ...

    def put_research(self, symbol: str, payload: Mapping[str, Any]) -> None: ...

    def append_audit(self, events: Sequence[AuditEvent]) -> None: ...


@dataclass
class InMemoryStateStore:
    """Non-persistent store. The default everywhere except AWS."""

    cycles: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    snapshot: Mapping[str, Any] | None = None
    research: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    audit: list[Mapping[str, Any]] = field(default_factory=list)

    def put_cycle(self, cycle_id: str, payload: Mapping[str, Any]) -> None:
        self.cycles[cycle_id] = encode(payload)

    def get_cycle(self, cycle_id: str) -> Mapping[str, Any] | None:
        return self.cycles.get(cycle_id)

    def latest_snapshot(self) -> Mapping[str, Any] | None:
        return self.snapshot

    def put_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.snapshot = encode(payload)

    def get_research(self, symbol: str) -> Mapping[str, Any] | None:
        return self.research.get(symbol)

    def put_research(self, symbol: str, payload: Mapping[str, Any]) -> None:
        self.research[symbol] = encode(payload)

    def append_audit(self, events: Sequence[AuditEvent]) -> None:
        self.audit.extend(json.loads(event.to_json()) for event in events)


@dataclass
class DynamoStateStore:
    """DynamoDB-backed store, single table, partition key ``pk``.

    Idempotent by design: writing the same cycle id twice overwrites rather than
    duplicating, which matters because EventBridge guarantees *at least* once
    delivery, not exactly once. A retried schedule must not produce two records
    of the same decision.
    """

    table_name: str
    region: str | None = None

    def __post_init__(self) -> None:
        try:
            import boto3  # noqa: PLC0415 - optional, AWS-only dependency
        except ImportError as exc:  # pragma: no cover - exercised only on AWS
            raise RuntimeError(
                "DynamoStateStore needs boto3: pip install -r infra/requirements.txt. "
                "Use InMemoryStateStore off AWS."
            ) from exc
        self._table = boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)

    def put_cycle(self, cycle_id: str, payload: Mapping[str, Any]) -> None:
        self._table.put_item(Item={"pk": f"{CYCLE_PREFIX}{cycle_id}", **encode(payload)})

    def get_cycle(self, cycle_id: str) -> Mapping[str, Any] | None:
        response = self._table.get_item(Key={"pk": f"{CYCLE_PREFIX}{cycle_id}"})
        item = response.get("Item")
        return decode(item) if isinstance(item, Mapping) else None

    def latest_snapshot(self) -> Mapping[str, Any] | None:
        response = self._table.get_item(Key={"pk": SNAPSHOT_KEY})
        item = response.get("Item")
        return decode(item) if isinstance(item, Mapping) else None

    def put_snapshot(self, payload: Mapping[str, Any]) -> None:
        self._table.put_item(Item={"pk": SNAPSHOT_KEY, **encode(payload)})

    def get_research(self, symbol: str) -> Mapping[str, Any] | None:
        response = self._table.get_item(Key={"pk": f"{RESEARCH_PREFIX}{symbol}"})
        item = response.get("Item")
        return decode(item) if isinstance(item, Mapping) else None

    def put_research(self, symbol: str, payload: Mapping[str, Any]) -> None:
        self._table.put_item(Item={"pk": f"{RESEARCH_PREFIX}{symbol}", **encode(payload)})

    def append_audit(self, events: Sequence[AuditEvent]) -> None:
        with self._table.batch_writer() as batch:
            for index, event in enumerate(events):
                batch.put_item(
                    Item={
                        "pk": f"{AUDIT_PREFIX}{event.timestamp.isoformat()}#{index}",
                        **encode(json.loads(event.to_json())),
                    }
                )


def build_store() -> StateStore:
    """DynamoDB when a table is configured, memory otherwise.

    Falling back to memory rather than failing means a misconfigured deployment
    degrades to "runs but does not persist", which is visible in the dashboard,
    instead of a Lambda that cannot start.
    """
    table = os.environ.get(TABLE_ENV)
    if not table:
        return InMemoryStateStore()
    return DynamoStateStore(table_name=table, region=os.environ.get(REGION_ENV))
