"""Audit log of consequential decisions, tagged with the CFA Standard each implements.

The point is reviewability after the fact, by someone who was not there.
Events carry a code and a detail, both facts; a narrator's interpretation is
kept in a separate field.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from src.time.clock import ensure_utc


class Standard(str, enum.Enum):
    """The CFA Standards this system implements in code."""

    #: I(C) Misrepresentation — the HALLUCINATED_FIGURE and fabricated-citation checks.
    I_C_MISREPRESENTATION = "I(C) Misrepresentation"
    #: III(A) Loyalty, prudence and care — the IPS binds, with no runtime override.
    III_A_LOYALTY = "III(A) Loyalty, Prudence and Care"
    #: III(C) Suitability — the IPS check *is* the suitability test.
    III_C_SUITABILITY = "III(C) Suitability"
    #: V(A) Diligence and reasonable basis — the citation requirement.
    V_A_DILIGENCE = "V(A) Diligence and Reasonable Basis"
    #: V(B) Communication with clients — fact and opinion kept separate.
    V_B_COMMUNICATION = "V(B) Communication with Clients"
    #: GIPS — the reason TWR is the headline metric.
    GIPS = "GIPS Composite Reporting"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One recorded act."""

    timestamp: datetime
    actor: str
    code: str
    standard: Standard
    detail: str
    symbol: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "actor": self.actor,
                "code": self.code,
                "standard": self.standard.value,
                "symbol": self.symbol,
                "detail": self.detail,
            },
            sort_keys=True,
        )


@dataclass(slots=True)
class AuditLog:
    """Append-only record of audit events.

    In-memory by default so the backtest can carry one per run without touching
    disk; :meth:`write` persists it when a run finishes.
    """

    events: list[AuditEvent] = field(default_factory=list)

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    def extend(self, events: Iterable[AuditEvent]) -> None:
        self.events.extend(events)

    def __iter__(self) -> Iterator[AuditEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def by_code(self, code: str) -> list[AuditEvent]:
        return [e for e in self.events if e.code == code]

    def by_standard(self, standard: Standard) -> list[AuditEvent]:
        return [e for e in self.events if e.standard is standard]

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for event in self.events:
            tally[event.code] = tally.get(event.code, 0) + 1
        return tally

    def write(self, path: Path) -> None:
        """Persist as JSON Lines, ordered by time.

        One event per line so the file appends cleanly and a long run does not
        have to be held in memory to be read back.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.events, key=lambda e: (e.timestamp, e.code, e.symbol or ""))
        path.write_text(
            "\n".join(event.to_json() for event in ordered) + "\n", encoding="utf-8"
        )
