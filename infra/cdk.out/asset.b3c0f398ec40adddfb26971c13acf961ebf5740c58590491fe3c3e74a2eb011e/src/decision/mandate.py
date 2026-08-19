"""The rebalance mandate: the entire decision-to-execution surface.

The decision layer emits **target weights, not orders**. Weights are the actual
decision; share counts are a function of weights, prices and portfolio value at
execution time, and a real execution algorithm recomputes them as it works the
order. Weights also keep the mandate valid while it is being worked.

``to_wire`` emits decimal strings — a float at the one place two languages must
agree exactly is the worst place for binary rounding.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

from src.time.clock import ensure_utc

ZERO = Decimal(0)

#: Weights are emitted to four decimal places: one basis point of resolution,
#: which is finer than any executor can act on and keeps the wire form stable.
WEIGHT_PLACES = Decimal("0.0001")
MONEY_PLACES = Decimal("0.01")


class Urgency(str, enum.Enum):
    """How hard the executor should push. Advisory to a naive executor."""

    PATIENT = "PATIENT"
    NORMAL = "NORMAL"
    AGGRESSIVE = "AGGRESSIVE"


class MandateError(ValueError):
    """Raised on a mandate that could not be executed as stated."""


def _weight(value: Decimal) -> str:
    return str(value.quantize(WEIGHT_PLACES))


def _money(value: Decimal) -> str:
    return str(value.quantize(MONEY_PLACES))


@dataclass(frozen=True, slots=True)
class TargetWeight:
    """One position's intent: where it is now, where it should be."""

    symbol: str
    target_weight: Decimal
    current_weight: Decimal

    def to_wire(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "target_weight": _weight(self.target_weight),
            "current_weight": _weight(self.current_weight),
        }


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    """Limits the executor must respect, or report that it cannot.

    ``max_participation_rate`` is ignored by the naive executor. That is not a
    silent failure: the caller checks
    :attr:`Capabilities.supports_participation_limits` and logs the constraint
    as advisory rather than assuming it was honored.
    """

    min_trade_notional: Decimal
    max_turnover: Decimal
    max_participation_rate: Decimal = Decimal("0.10")
    allow_partial: bool = True
    deadline: datetime | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "min_trade_notional": _money(self.min_trade_notional),
            "max_turnover": _weight(self.max_turnover),
            "max_participation_rate": _weight(self.max_participation_rate),
            "allow_partial": self.allow_partial,
        }
        if self.deadline is not None:
            wire["deadline"] = ensure_utc(self.deadline).isoformat()
        return wire


@dataclass(frozen=True, slots=True)
class RebalanceMandate:
    """What crosses the execution boundary. Weights and constraints, nothing else."""

    mandate_id: str
    decision_time: datetime
    portfolio_value: Decimal
    targets: tuple[TargetWeight, ...]
    constraints: ExecutionConstraints
    urgency: Urgency = Urgency.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", ensure_utc(self.decision_time))
        if self.portfolio_value <= ZERO:
            raise MandateError(f"portfolio value must be positive, got {self.portfolio_value}")
        symbols = [t.symbol for t in self.targets]
        if len(symbols) != len(set(symbols)):
            raise MandateError("duplicate symbol in mandate targets")

    @property
    def implied_turnover(self) -> Decimal:
        """One-way turnover the mandate implies, before any executor limits."""
        return (
            sum((abs(t.target_weight - t.current_weight) for t in self.targets), ZERO)
            / Decimal(2)
        )

    def to_wire(self) -> dict[str, Any]:
        """Map to ``proto/execution.proto``'s ``RebalanceMandate``.

        Every monetary and ratio value is a decimal string. A float here would
        put binary rounding at the one place two languages have to agree
        exactly.
        """
        return {
            "mandate_id": self.mandate_id,
            "decision_time": self.decision_time.isoformat(),
            "portfolio_value": _money(self.portfolio_value),
            "targets": [t.to_wire() for t in self.targets],
            "constraints": self.constraints.to_wire(),
            "urgency": self.urgency.value,
        }


def mandate_id(
    decision_time: datetime,
    portfolio_value: Decimal,
    targets: Sequence[TargetWeight],
) -> str:
    """Deterministic idempotency key for a mandate.

    Derived from the content rather than randomly generated, so replaying an
    identical decision produces an identical id and the executor can recognise
    and drop the duplicate. That also keeps 's promise that two runs
    over identical inputs produce a byte-identical trade log — a UUID here
    would break both properties at once.
    """
    parts = [
        ensure_utc(decision_time).isoformat(),
        _money(portfolio_value),
    ]
    for target in sorted(targets, key=lambda t: t.symbol):
        parts.append(f"{target.symbol}:{_weight(target.target_weight)}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"mandate-{digest[:32]}"


def build_mandate(
    *,
    decision_time: datetime,
    portfolio_value: Decimal,
    target_weights: Mapping[str, Decimal],
    current_weights: Mapping[str, Decimal],
    min_trade_notional: Decimal,
    max_turnover: Decimal,
    max_participation_rate: Decimal = Decimal("0.10"),
    urgency: Urgency = Urgency.NORMAL,
    deadline: datetime | None = None,
) -> RebalanceMandate:
    """Assemble a mandate from approved weights.

    Positions being exited are included with a target of zero. Omitting them
    would leave the executor unable to tell "sell this" from "no opinion", and
    the position would simply be stranded.
    """
    symbols = sorted(set(target_weights) | set(current_weights))
    targets = tuple(
        TargetWeight(
            symbol=symbol,
            target_weight=target_weights.get(symbol, ZERO),
            current_weight=current_weights.get(symbol, ZERO),
        )
        for symbol in symbols
        # A position that is absent from both sides is not a decision.
        if target_weights.get(symbol, ZERO) != ZERO or current_weights.get(symbol, ZERO) != ZERO
    )

    return RebalanceMandate(
        mandate_id=mandate_id(decision_time, portfolio_value, targets),
        decision_time=decision_time,
        portfolio_value=portfolio_value,
        targets=targets,
        constraints=ExecutionConstraints(
            min_trade_notional=min_trade_notional,
            max_turnover=max_turnover,
            max_participation_rate=max_participation_rate,
            deadline=deadline,
        ),
        urgency=urgency,
    )


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Post-trade drift between what was decided and what was achieved.

    Realized weights never equal target weights. Recording the residual is
    mandatory: it feeds the next cycle's corridor check, and a system that
    assumes perfect execution produces backtests that lie.
    """

    per_symbol_drift: Mapping[str, Decimal]
    total_absolute_drift: Decimal
    realized_turnover: Decimal
    max_drift_symbol: str | None = None

    @property
    def max_drift(self) -> Decimal:
        if not self.per_symbol_drift:
            return ZERO
        return max(abs(v) for v in self.per_symbol_drift.values())


def reconcile(
    mandate: RebalanceMandate,
    realized_weights: Mapping[str, Decimal],
) -> Reconciliation:
    """Compare achieved weights against the mandate's intent."""
    drift: dict[str, Decimal] = {}
    for target in mandate.targets:
        drift[target.symbol] = realized_weights.get(target.symbol, ZERO) - target.target_weight
    for symbol, weight in realized_weights.items():
        if symbol not in drift:
            # The executor filled something the mandate never asked for.
            drift[symbol] = weight

    total = sum((abs(v) for v in drift.values()), ZERO)
    realized = sum(
        (
            abs(realized_weights.get(t.symbol, ZERO) - t.current_weight)
            for t in mandate.targets
        ),
        ZERO,
    ) / Decimal(2)

    worst = max(drift, key=lambda s: abs(drift[s])) if drift else None
    return Reconciliation(
        per_symbol_drift=drift,
        total_absolute_drift=total,
        realized_turnover=realized,
        max_drift_symbol=worst,
    )
