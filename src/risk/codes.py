"""Reason codes and outcomes for the risk engine (SPEC §7).

Every veto carries a code. Codes are a closed enumeration rather than free text
because they are persisted, counted, and surfaced in the dashboard's
vetoed-trades panel — the screen SPEC §7 says to demo first. "The trade was
rejected" is not an audit trail; ``MAX_SECTOR_WEIGHT`` on a named sector is.

CFA Standard V(B), communication with clients: the assessment separates the
*fact* of a violation (code, measured value, limit) from any narrative about
it. The Narrator (SPEC §5.5) may phrase these for a human; it may not change
them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal


class Decision(str, enum.Enum):
    """What the risk engine did with a proposal."""

    #: Proposal satisfied every constraint as submitted.
    APPROVED = "APPROVED"
    #: Proposal was repaired into a compliant portfolio; weights changed.
    MODIFIED = "MODIFIED"
    #: Proposal could not be made compliant, or is vetoed outright.
    REJECTED = "REJECTED"


class ReasonCode(str, enum.Enum):
    """The constraint table of SPEC §7, one code per rule."""

    MAX_POSITION_WEIGHT = "MAX_POSITION_WEIGHT"
    MAX_SECTOR_WEIGHT = "MAX_SECTOR_WEIGHT"
    MIN_CASH_BUFFER = "MIN_CASH_BUFFER"
    MAX_PORTFOLIO_BETA = "MAX_PORTFOLIO_BETA"
    MAX_VOLATILITY = "MAX_VOLATILITY"
    SAFETY_FIRST_THRESHOLD = "SAFETY_FIRST_THRESHOLD"
    NO_LEVERAGE = "NO_LEVERAGE"
    NO_SHORTING = "NO_SHORTING"
    REBALANCE_CORRIDOR = "REBALANCE_CORRIDOR"
    MAX_TURNOVER = "MAX_TURNOVER"
    MIN_TRADE_NOTIONAL = "MIN_TRADE_NOTIONAL"
    UNIVERSE_WHITELIST = "UNIVERSE_WHITELIST"
    DRAWDOWN_CIRCUIT_BREAKER = "DRAWDOWN_CIRCUIT_BREAKER"
    WASH_SALE_WINDOW = "WASH_SALE_WINDOW"


#: Codes that can never be repaired by adjusting weights, only refused.
#: Everything else has a deterministic repair in the engine.
TERMINAL_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.SAFETY_FIRST_THRESHOLD,
        ReasonCode.REBALANCE_CORRIDOR,
    }
)


@dataclass(frozen=True, slots=True)
class Violation:
    """One constraint breach: which rule, where, and by how much.

    ``observed`` and ``limit`` are recorded so the audit log can show the
    margin rather than only the verdict — "beta 1.31 against a 1.20 ceiling"
    is reviewable; "beta too high" is not.
    """

    code: ReasonCode
    detail: str
    symbol: str | None = None
    observed: Decimal | None = None
    limit: Decimal | None = None

    def __str__(self) -> str:
        where = f" [{self.symbol}]" if self.symbol else ""
        return f"{self.code.value}{where}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Repair:
    """A change the engine made to bring a proposal into compliance.

    Recorded separately from violations: a repair means the constraint was
    *enforced*, not that the portfolio ended up breaching it. The distinction
    matters when counting vetoes for the dashboard.
    """

    code: ReasonCode
    detail: str
    symbol: str | None = None

    def __str__(self) -> str:
        where = f" [{self.symbol}]" if self.symbol else ""
        return f"{self.code.value}{where}: {self.detail}"
