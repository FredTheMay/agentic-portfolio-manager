"""Bond pricing, interest-rate risk, and money-market yield conventions.

CFA Level I topic area: Fixed Income.

Pure functions, zero I/O. ``Decimal`` at every public boundary; only
:func:`yield_to_maturity` leaves it, because inverting the price equation
requires an iterative solve.

**Why the yield conversions matter beyond this module.** FRED's ``DGS3MO`` —
the natural source for a risk-free rate — is quoted on a *bank discount basis*.
That convention divides the discount by face value rather than by the price
actually paid, and annualizes on a 360-day year. Both choices bias it low.
Feeding it directly into Sharpe, Treynor, CAPM, or the CAL overstates every
excess return, which is why :func:`discount_to_bond_equivalent_yield` first.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from scipy.optimize import brentq

from src.cfa._numeric import (
    NumericError,
    require_min_length,
    require_same_length,
    to_decimal,
    to_float,
)

ZERO = Decimal(0)
ONE = Decimal(1)

#: Money-market discount conventions annualize on 360 days.
DISCOUNT_YEAR = Decimal(360)
#: Bond-equivalent and effective yields annualize on 365.
BOND_YEAR = Decimal(365)


def _periods(years_to_maturity: int, periods_per_year: int) -> int:
    if years_to_maturity <= 0:
        raise NumericError(f"years_to_maturity must be positive, got {years_to_maturity}")
    if periods_per_year <= 0:
        raise NumericError(f"periods_per_year must be positive, got {periods_per_year}")
    return years_to_maturity * periods_per_year


def bond_price(
    face_value: Decimal,
    coupon_rate: Decimal,
    yield_to_maturity: Decimal,
    years_to_maturity: int,
    periods_per_year: int = 1,
) -> Decimal:
    """Present value of the coupon stream plus principal.

    Coupon and yield are annual rates; both are divided by ``periods_per_year``
    to get the per-period figures actually discounted.

    CFA Level I: Fixed Income — bond valuation.
    """
    n = _periods(years_to_maturity, periods_per_year)
    frequency = Decimal(periods_per_year)
    coupon = face_value * coupon_rate / frequency
    rate = yield_to_maturity / frequency
    if rate <= -ONE:
        raise NumericError(f"per-period yield {rate} is below -100%")

    price = ZERO
    for period in range(1, n + 1):
        discount = (ONE + rate) ** period
        price += coupon / discount
    price += face_value / (ONE + rate) ** n
    return price


def yield_to_maturity(
    price: Decimal,
    face_value: Decimal,
    coupon_rate: Decimal,
    years_to_maturity: int,
    periods_per_year: int = 1,
) -> Decimal:
    """The discount rate that equates the cash flows to ``price``.

    No closed form exists, so this is a bracketed root-find in float64 (see
    :mod:`src.cfa._numeric`). Reported as an annual rate.

    CFA Level I: Fixed Income — yield measures.
    """
    if price <= ZERO:
        raise NumericError(f"price must be positive, got {price}")
    n = _periods(years_to_maturity, periods_per_year)

    target = to_float(price)
    face = to_float(face_value)
    coupon = to_float(face_value * coupon_rate / Decimal(periods_per_year))
    frequency = periods_per_year

    def residual(annual_yield: float) -> float:
        rate = annual_yield / frequency
        value = sum(coupon / (1.0 + rate) ** period for period in range(1, n + 1))
        value += face / (1.0 + rate) ** n
        return value - target

    low, high = -0.99, 10.0
    if residual(low) * residual(high) > 0:
        raise NumericError("could not bracket a yield to maturity in (-99%, 1000%)")
    return to_decimal(float(brentq(residual, low, high, xtol=1e-14, rtol=1e-14, maxiter=200)))


def current_yield(annual_coupon: Decimal, price: Decimal) -> Decimal:
    """``annual coupon / price``.

    Income only: it ignores the pull-to-par capital gain or loss, which is why
    it overstates the return on a premium bond and understates it on a discount
    bond.

    CFA Level I: Fixed Income — yield measures.
    """
    if price <= ZERO:
        raise NumericError(f"price must be positive, got {price}")
    return annual_coupon / price


def macaulay_duration(
    face_value: Decimal,
    coupon_rate: Decimal,
    yield_to_maturity: Decimal,
    years_to_maturity: int,
    periods_per_year: int = 1,
) -> Decimal:
    """PV-weighted average time to receipt of the bond's cash flows, in years.

    CFA Level I: Fixed Income — duration.
    """
    n = _periods(years_to_maturity, periods_per_year)
    frequency = Decimal(periods_per_year)
    coupon = face_value * coupon_rate / frequency
    rate = yield_to_maturity / frequency

    weighted = ZERO
    price = ZERO
    for period in range(1, n + 1):
        cash_flow = coupon + (face_value if period == n else ZERO)
        present_value = cash_flow / (ONE + rate) ** period
        price += present_value
        weighted += Decimal(period) * present_value

    if price == ZERO:
        raise NumericError("bond has zero present value; duration is undefined")
    # Periods -> years.
    return weighted / price / frequency


def modified_duration(
    macaulay: Decimal,
    yield_to_maturity: Decimal,
    periods_per_year: int = 1,
) -> Decimal:
    """``Macaulay / (1 + y/m)`` — the actual price sensitivity to a yield move.

    Always below Macaulay duration, because a yield rise both reduces each cash
    flow's present value and discounts it more steeply.

    CFA Level I: Fixed Income — modified duration.
    """
    if periods_per_year <= 0:
        raise NumericError(f"periods_per_year must be positive, got {periods_per_year}")
    denominator = ONE + yield_to_maturity / Decimal(periods_per_year)
    if denominator <= ZERO:
        raise NumericError("modified duration is undefined at a yield below -100%")
    return macaulay / denominator


def convexity(
    face_value: Decimal,
    coupon_rate: Decimal,
    yield_to_maturity: Decimal,
    years_to_maturity: int,
    periods_per_year: int = 1,
) -> Decimal:
    """Second-order sensitivity of price to yield.

    ``sum(t(t+1) PV(CFt)) / (price (1+y)^2)``, expressed in years-squared.

    CFA Level I: Fixed Income — convexity.
    """
    n = _periods(years_to_maturity, periods_per_year)
    frequency = Decimal(periods_per_year)
    coupon = face_value * coupon_rate / frequency
    rate = yield_to_maturity / frequency

    weighted = ZERO
    price = ZERO
    for period in range(1, n + 1):
        cash_flow = coupon + (face_value if period == n else ZERO)
        present_value = cash_flow / (ONE + rate) ** period
        price += present_value
        weighted += Decimal(period) * Decimal(period + 1) * present_value

    if price == ZERO:
        raise NumericError("bond has zero present value; convexity is undefined")
    return weighted / (price * (ONE + rate) ** 2) / frequency**2


def price_change_percent(
    modified_duration: Decimal,
    convexity_measure: Decimal,
    yield_change: Decimal,
) -> Decimal:
    """``%dP ~= -ModDur x dy + 0.5 x convexity x dy^2``.

    The convexity term is what makes the estimate asymmetric: for a long bond,
    a yield fall gains more than an equal yield rise loses. Duration alone is a
    straight line through a curve and is wrong in the same direction on both
    sides.

    CFA Level I: Fixed Income — duration and convexity price estimates.
    """
    return -modified_duration * yield_change + (
        Decimal("0.5") * convexity_measure * yield_change**2
    )


def portfolio_duration(weights: Sequence[Decimal], durations: Sequence[Decimal]) -> Decimal:
    """Market-value-weighted average of component durations.

    An approximation: it assumes a parallel shift in the yield curve, so it
    says nothing about steepening or flattening.

    CFA Level I: Fixed Income — portfolio duration.
    """
    require_same_length("weights", weights, "durations", durations)
    require_min_length("weights", weights, 1)
    return sum((w * d for w, d in zip(weights, durations)), ZERO)


# --- Money-market yield conventions ----------------------------------------


def _require_days(days_to_maturity: int) -> Decimal:
    if days_to_maturity <= 0:
        raise NumericError(f"days_to_maturity must be positive, got {days_to_maturity}")
    return Decimal(days_to_maturity)


def bank_discount_yield(
    face_value: Decimal,
    price: Decimal,
    days_to_maturity: int,
) -> Decimal:
    """``(D / F)(360 / t)`` where ``D = face - price``.

    The convention US T-bills are quoted on, and a poor measure of return: it
    divides the gain by face value rather than by the money actually invested,
    and annualizes on a 360-day year without compounding.

    CFA Level I: Fixed Income — money-market yields.
    """
    days = _require_days(days_to_maturity)
    if face_value <= ZERO:
        raise NumericError("face value must be positive")
    return ((face_value - price) / face_value) * (DISCOUNT_YEAR / days)


def holding_period_yield(
    purchase_price: Decimal,
    sale_price: Decimal,
    income: Decimal = ZERO,
) -> Decimal:
    """``(P1 - P0 + D1) / P0`` — unannualized return over the holding period.

    CFA Level I: Fixed Income — money-market yields.
    """
    if purchase_price <= ZERO:
        raise NumericError("purchase price must be positive")
    return (sale_price - purchase_price + income) / purchase_price


def money_market_yield(holding_period: Decimal, days_to_maturity: int) -> Decimal:
    """``HPY x (360 / t)`` — CD-equivalent yield.

    Fixes the discount yield's denominator (it is based on price paid) but
    keeps the 360-day year and still does not compound.

    CFA Level I: Fixed Income — money-market yields.
    """
    days = _require_days(days_to_maturity)
    return holding_period * (DISCOUNT_YEAR / days)


def effective_annual_yield(holding_period: Decimal, days_to_maturity: int) -> Decimal:
    """``(1 + HPY)^(365/t) - 1`` — compounded, on a 365-day year.

    The only one of these conventions that is directly comparable across
    instruments with different maturities.

    CFA Level I: Fixed Income — money-market yields.
    """
    days = _require_days(days_to_maturity)
    growth = ONE + holding_period
    if growth <= ZERO:
        raise NumericError(f"growth factor {growth} must be positive")
    return growth ** (BOND_YEAR / days) - ONE


def discount_to_bond_equivalent_yield(
    discount_yield: Decimal,
    days_to_maturity: int,
) -> Decimal:
    """Convert a bank discount yield to a bond-equivalent yield.

    Recovers the implied price from the quoted discount, then restates the
    return on the money actually invested and on a 365-day year::

        price = F(1 - BDY x t/360)
        BEY   = (F - price)/price x 365/t

    this before ``DGS3MO`` is used as the
    risk-free rate anywhere. BEY is always above the discount yield it came
    from, so skipping the conversion understates Rf and inflates every
    risk-adjusted metric in the system.

    CFA Level I: Fixed Income — money-market yield conversions.
    """
    days = _require_days(days_to_maturity)
    # Face value cancels out of the ratio, so work in units of face.
    price = ONE - discount_yield * (days / DISCOUNT_YEAR)
    if price <= ZERO:
        raise NumericError(
            f"discount yield {discount_yield} over {days_to_maturity} days implies "
            "a non-positive price"
        )
    return ((ONE - price) / price) * (BOND_YEAR / days)
