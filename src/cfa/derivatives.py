"""Options and forwards: payoffs, parity, and carry.

CFA Level I topic area: Derivatives (SPEC §6.7).

Pure functions, zero I/O, ``Decimal`` throughout. Discrete compounding
``(1 + r)^T`` is used everywhere, matching the Level I convention.

**Nothing here trades.** The protective-put overlay (SPEC §6.7) is simulated:
the system displays the payoff diagram and the cost drag when portfolio
volatility breaches the IPS ceiling, and never sends an option order.

Which parity check this system uses, and why
--------------------------------------------
``C + PV(X) = P + S0`` is an **arbitrage identity for European options only**.
It is enforced by a replication argument that assumes neither side can be
exercised early.

US listed equity options are American. The right to exercise early has value,
so the identity becomes a pair of inequalities and a strict equality check
fires constantly on perfectly legitimate quotes — it flags the early-exercise
premium as though it were free money.

This module therefore ships both, and the system uses them differently:

* :func:`european_put_call_parity` — strict equality. Valid **only** for
  European-style contracts, i.e. cash-settled index options such as SPX.
* :func:`american_parity_breach` — the bounds
  ``S0 - D - X <= C - P <= S0 - PV(X)``. This is what runs against US listed
  equity options, and only a breach *outside* the bounds is reported.

Forwards versus futures
-----------------------
Both lock in a price today for delivery later, and the pricing formula here is
the forward's. They differ in credit and cash-flow profile: a forward is a
bilateral OTC contract whose entire gain or loss accrues as counterparty
exposure until settlement, whereas a future is exchange-traded and marked to
market daily through a clearinghouse, so exposure resets each day and margin
calls arrive as real cash flows. When rates are correlated with the underlying
those interim flows are not value-neutral, which is why forward and futures
prices need not be identical.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

ZERO = Decimal(0)
ONE = Decimal(1)

#: Tolerance for the European equality check, in price units.
DEFAULT_PARITY_TOLERANCE = Decimal("0.01")


class OptionType(str, enum.Enum):
    CALL = "CALL"
    PUT = "PUT"


class Position(str, enum.Enum):
    """The four basic option positions."""

    LONG_CALL = "LONG_CALL"
    SHORT_CALL = "SHORT_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"


class Moneyness(str, enum.Enum):
    IN_THE_MONEY = "IN_THE_MONEY"
    AT_THE_MONEY = "AT_THE_MONEY"
    OUT_OF_THE_MONEY = "OUT_OF_THE_MONEY"


class DerivativesError(ValueError):
    """Raised on economically impossible derivative inputs."""


def compound(value: Decimal, risk_free_rate: Decimal, years: Decimal) -> Decimal:
    """``value x (1 + r)^T``.

    CFA Level I: Derivatives — cost of carry.
    """
    growth = ONE + risk_free_rate
    if growth <= ZERO:
        raise DerivativesError(f"growth factor {growth} must be positive")
    return value * growth**years


def discount(value: Decimal, risk_free_rate: Decimal, years: Decimal) -> Decimal:
    """``value / (1 + r)^T`` — present value at the risk-free rate.

    CFA Level I: Derivatives — cost of carry.
    """
    growth = ONE + risk_free_rate
    if growth <= ZERO:
        raise DerivativesError(f"growth factor {growth} must be positive")
    return value / growth**years


def forward_price(
    spot: Decimal,
    risk_free_rate: Decimal,
    years: Decimal,
    present_value_dividends: Decimal = ZERO,
) -> Decimal:
    """``F0 = (S0 - PV(dividends))(1 + r)^T``.

    SPEC §6.7 [CORRECTED]. The naive ``S0(1 + r)^T`` ignores the carry benefit
    of holding the stock: a dividend paid before delivery goes to the *holder*,
    not the forward buyer, so it must be stripped out of the spot before
    compounding. Omitting it overstates the forward price on every
    dividend-paying equity.

    CFA Level I: Derivatives — pricing forward contracts.
    """
    if present_value_dividends < ZERO:
        raise DerivativesError("present value of dividends cannot be negative")
    if present_value_dividends > spot:
        raise DerivativesError("dividends cannot exceed the spot price")
    return compound(spot - present_value_dividends, risk_free_rate, years)


@dataclass(frozen=True, slots=True)
class ParityCheck:
    """Result of a put-call parity test."""

    left: Decimal
    right: Decimal
    difference: Decimal
    holds: bool
    detail: str


def european_put_call_parity(
    *,
    call_price: Decimal,
    put_price: Decimal,
    spot: Decimal,
    strike: Decimal,
    risk_free_rate: Decimal,
    years: Decimal,
    tolerance: Decimal = DEFAULT_PARITY_TOLERANCE,
) -> ParityCheck:
    """Check ``C + PV(X) = P + S0``.

    **European options only.** Applying this to American-style contracts
    produces false positives, because early exercise breaks the replication
    argument. Use :func:`american_parity_breach` for US listed equity options.

    CFA Level I: Derivatives — put-call parity.
    """
    left = call_price + discount(strike, risk_free_rate, years)
    right = put_price + spot
    difference = left - right
    holds = abs(difference) <= tolerance
    return ParityCheck(
        left=left,
        right=right,
        difference=difference,
        holds=holds,
        detail=(
            "European parity holds within tolerance"
            if holds
            else f"European parity violated by {difference}"
        ),
    )


def implied_put_price(
    call_price: Decimal,
    spot: Decimal,
    strike: Decimal,
    risk_free_rate: Decimal,
    years: Decimal,
) -> Decimal:
    """``P = C + PV(X) - S0``, rearranged from European parity.

    CFA Level I: Derivatives — put-call parity.
    """
    return call_price + discount(strike, risk_free_rate, years) - spot


def implied_call_price(
    put_price: Decimal,
    spot: Decimal,
    strike: Decimal,
    risk_free_rate: Decimal,
    years: Decimal,
) -> Decimal:
    """``C = P + S0 - PV(X)``, rearranged from European parity.

    CFA Level I: Derivatives — put-call parity.
    """
    return put_price + spot - discount(strike, risk_free_rate, years)


@dataclass(frozen=True, slots=True)
class ParityBounds:
    """Admissible range for ``C - P`` on American-style options."""

    lower: Decimal
    upper: Decimal

    def contains(self, call_minus_put: Decimal) -> bool:
        return self.lower <= call_minus_put <= self.upper


def american_put_call_parity_bounds(
    spot: Decimal,
    strike: Decimal,
    risk_free_rate: Decimal,
    years: Decimal,
    present_value_dividends: Decimal = ZERO,
) -> ParityBounds:
    """``S0 - D - X <= C - P <= S0 - PV(X)`` for American options.

    The lower bound comes from the possibility of early exercise on the call
    (which is why the dividend enters undiscounted), the upper from the put's.
    The gap between them is the early-exercise premium — real value that strict
    parity has no way to express.

    CFA Level I: Derivatives — put-call parity, American options.
    """
    return ParityBounds(
        lower=spot - present_value_dividends - strike,
        upper=spot - discount(strike, risk_free_rate, years),
    )


def american_parity_breach(
    *,
    call_price: Decimal,
    put_price: Decimal,
    spot: Decimal,
    strike: Decimal,
    risk_free_rate: Decimal,
    years: Decimal,
    present_value_dividends: Decimal = ZERO,
) -> ParityCheck:
    """Flag a quote only when ``C - P`` falls outside the American bounds.

    This is the check the system runs against US listed equity options
    (SPEC §6.7 [CORRECTED]).

    CFA Level I: Derivatives — put-call parity, American options.
    """
    bounds = american_put_call_parity_bounds(
        spot, strike, risk_free_rate, years, present_value_dividends
    )
    spread = call_price - put_price
    holds = bounds.contains(spread)

    if holds:
        detail = f"C - P = {spread} lies within [{bounds.lower}, {bounds.upper}]"
        difference = ZERO
    elif spread > bounds.upper:
        difference = spread - bounds.upper
        detail = f"C - P = {spread} exceeds the upper bound {bounds.upper}"
    else:
        difference = spread - bounds.lower
        detail = f"C - P = {spread} falls below the lower bound {bounds.lower}"

    return ParityCheck(
        left=bounds.lower,
        right=bounds.upper,
        difference=difference,
        holds=holds,
        detail=detail,
    )


# --- Intrinsic value, time value, moneyness --------------------------------


def intrinsic_value(option_type: OptionType, spot: Decimal, strike: Decimal) -> Decimal:
    """``max(S - X, 0)`` for a call, ``max(X - S, 0)`` for a put.

    CFA Level I: Derivatives — option value components.
    """
    if option_type is OptionType.CALL:
        return max(spot - strike, ZERO)
    return max(strike - spot, ZERO)


def time_value(premium: Decimal, intrinsic: Decimal) -> Decimal:
    """``premium - intrinsic value``.

    Cannot be negative: an option trading below intrinsic value would be a
    riskless arbitrage, so this raises rather than returning a negative number
    that would propagate as though it meant something.

    CFA Level I: Derivatives — option value components.
    """
    value = premium - intrinsic
    if value < ZERO:
        raise DerivativesError(
            f"premium {premium} is below intrinsic value {intrinsic}: "
            "negative time value is an arbitrage, not a quote"
        )
    return value


def moneyness(option_type: OptionType, spot: Decimal, strike: Decimal) -> Moneyness:
    """Classify an option as in, at, or out of the money.

    CFA Level I: Derivatives — moneyness.
    """
    if spot == strike:
        return Moneyness.AT_THE_MONEY
    if option_type is OptionType.CALL:
        return Moneyness.IN_THE_MONEY if spot > strike else Moneyness.OUT_OF_THE_MONEY
    return Moneyness.IN_THE_MONEY if spot < strike else Moneyness.OUT_OF_THE_MONEY


# --- The four basic positions ----------------------------------------------


def option_payoff(position: Position, strike: Decimal, spot_at_expiry: Decimal) -> Decimal:
    """Payoff at expiry, before premium.

    CFA Level I: Derivatives — option payoffs.
    """
    call_payoff = max(spot_at_expiry - strike, ZERO)
    put_payoff = max(strike - spot_at_expiry, ZERO)
    match position:
        case Position.LONG_CALL:
            return call_payoff
        case Position.SHORT_CALL:
            return -call_payoff
        case Position.LONG_PUT:
            return put_payoff
        case Position.SHORT_PUT:
            return -put_payoff


def option_profit(
    position: Position,
    strike: Decimal,
    premium: Decimal,
    spot_at_expiry: Decimal,
) -> Decimal:
    """Payoff net of the premium paid or received.

    The long pays the premium; the short receives it. Summing a long and a
    short of the same contract gives zero at every price — options transfer
    risk, they do not create value.

    CFA Level I: Derivatives — option profit.
    """
    payoff = option_payoff(position, strike, spot_at_expiry)
    if position in (Position.LONG_CALL, Position.LONG_PUT):
        return payoff - premium
    return payoff + premium


@dataclass(frozen=True, slots=True)
class PayoffPoint:
    """One point on a payoff diagram."""

    spot: Decimal
    payoff: Decimal
    profit: Decimal


def payoff_diagram(
    position: Position,
    strike: Decimal,
    premium: Decimal,
    spot_prices: Sequence[Decimal],
) -> list[PayoffPoint]:
    """Payoff and profit across a range of expiry prices, for plotting.

    CFA Level I: Derivatives — option payoff diagrams.
    """
    return [
        PayoffPoint(
            spot=spot,
            payoff=option_payoff(position, strike, spot),
            profit=option_profit(position, strike, premium, spot),
        )
        for spot in spot_prices
    ]


# --- Covered call ----------------------------------------------------------


def covered_call_payoff(spot_at_expiry: Decimal, strike: Decimal) -> Decimal:
    """Long stock plus short call: ``min(ST, X)``.

    CFA Level I: Derivatives — covered call.
    """
    return min(spot_at_expiry, strike)


def covered_call_profit(
    spot_at_expiry: Decimal,
    strike: Decimal,
    stock_cost: Decimal,
    call_premium: Decimal,
) -> Decimal:
    """``min(ST, X) - S0 + premium``.

    Income now in exchange for capped upside: above the strike the position
    stops participating entirely, which is the trade the premium pays for.

    CFA Level I: Derivatives — covered call.
    """
    return covered_call_payoff(spot_at_expiry, strike) - stock_cost + call_premium


def covered_call_breakeven(stock_cost: Decimal, call_premium: Decimal) -> Decimal:
    """``S0 - premium`` — the premium cushions the first of any decline.

    CFA Level I: Derivatives — covered call.
    """
    return stock_cost - call_premium


# --- Protective put --------------------------------------------------------


def protective_put_payoff(spot_at_expiry: Decimal, strike: Decimal) -> Decimal:
    """Long stock plus long put: ``max(ST, X)``.

    CFA Level I: Derivatives — protective put.
    """
    return max(spot_at_expiry, strike)


def protective_put_profit(
    spot_at_expiry: Decimal,
    strike: Decimal,
    stock_cost: Decimal,
    put_premium: Decimal,
) -> Decimal:
    """``max(ST, X) - S0 - premium``.

    Insurance: the loss is bounded below at ``S0 - X + premium`` however far
    the stock falls, and the premium is the deductible.

    CFA Level I: Derivatives — protective put.
    """
    return protective_put_payoff(spot_at_expiry, strike) - stock_cost - put_premium


def protective_put_breakeven(stock_cost: Decimal, put_premium: Decimal) -> Decimal:
    """``S0 + premium`` — the stock must clear the premium before profit starts.

    CFA Level I: Derivatives — protective put.
    """
    return stock_cost + put_premium


def protective_put_cost_drag(put_premium: Decimal, portfolio_value: Decimal) -> Decimal:
    """Premium as a fraction of the position it protects.

    Displayed alongside the payoff diagram whenever the overlay is proposed
    (SPEC §6.7), because the drag is certain and the protection is not.

    CFA Level I: Derivatives — protective put.
    """
    if portfolio_value <= ZERO:
        raise DerivativesError("portfolio value must be positive")
    return put_premium / portfolio_value
