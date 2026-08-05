"""Alternative investments: fee conventions and the smoothed-pricing problem.

CFA Level I topic area: Alternative Investments (SPEC §6.8).

Pure functions, zero I/O, ``Decimal`` throughout.

Why this module exists in a portfolio manager
---------------------------------------------
The universe holds REIT and commodity ETFs, so two things have to be handled
honestly.

**Fees compound against the investor.** A "2 and 20" fund that returns 20%
gross hands the investor about 14%. The high-water mark is the term that stops
a manager being paid twice for the same gains after a drawdown, and it is the
piece most often left out of a naive fee model.

**Smoothed pricing corrupts the covariance matrix.** Real estate and private
equity are valued by periodic appraisal, not continuous trading. Appraisals lag
and anchor on the previous mark, so reported returns are a moving average of
true returns. That autocorrelation damps measured volatility *and* measured
correlation with everything else.

The consequence is direct and bad: an optimizer fed those inputs (SPEC §6.2)
sees an asset that appears both low-risk and uncorrelated, and concentrates
into it. The apparent diversification is a measurement artifact. Where a
smoothed series must be used, :func:`unsmooth_returns` recovers a more honest
volatility first.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Sequence

from src.cfa._numeric import NumericError, require_min_length

ZERO = Decimal(0)
ONE = Decimal(1)

ManagementFeeBasis = Literal["BEGINNING", "ENDING"]


class AlternativeCategory(str, enum.Enum):
    """The Level I taxonomy of alternative investments."""

    HEDGE_FUND = "HEDGE_FUND"
    PRIVATE_EQUITY = "PRIVATE_EQUITY"
    REAL_ESTATE = "REAL_ESTATE"
    COMMODITIES = "COMMODITIES"
    INFRASTRUCTURE = "INFRASTRUCTURE"


@dataclass(frozen=True, slots=True)
class CategoryProfile:
    """What distinguishes a category for portfolio-construction purposes."""

    category: AlternativeCategory
    typical_fee_structure: str
    valuation_basis: str
    #: True when returns come from periodic appraisal rather than continuous
    #: trading, and are therefore autocorrelated and volatility-damped.
    smoothed_valuations: bool


CATEGORY_PROFILES: dict[AlternativeCategory, CategoryProfile] = {
    AlternativeCategory.HEDGE_FUND: CategoryProfile(
        category=AlternativeCategory.HEDGE_FUND,
        typical_fee_structure="management fee plus incentive fee, high-water mark",
        valuation_basis="mostly marked to market; illiquid sleeves may be appraised",
        smoothed_valuations=False,
    ),
    AlternativeCategory.PRIVATE_EQUITY: CategoryProfile(
        category=AlternativeCategory.PRIVATE_EQUITY,
        typical_fee_structure="management fee on committed capital plus carried interest",
        valuation_basis="periodic appraisal of unlisted holdings",
        smoothed_valuations=True,
    ),
    AlternativeCategory.REAL_ESTATE: CategoryProfile(
        category=AlternativeCategory.REAL_ESTATE,
        typical_fee_structure="management fee plus performance fee; REITs charge an expense ratio",
        valuation_basis="periodic appraisal; listed REITs trade continuously",
        smoothed_valuations=True,
    ),
    AlternativeCategory.COMMODITIES: CategoryProfile(
        category=AlternativeCategory.COMMODITIES,
        typical_fee_structure="expense ratio on an ETF; futures roll costs dominate",
        valuation_basis="exchange-traded futures, marked to market daily",
        smoothed_valuations=False,
    ),
    AlternativeCategory.INFRASTRUCTURE: CategoryProfile(
        category=AlternativeCategory.INFRASTRUCTURE,
        typical_fee_structure="management fee plus performance fee, long lock-up",
        valuation_basis="periodic appraisal of long-lived assets",
        smoothed_valuations=True,
    ),
}


def has_smoothed_valuations(category: AlternativeCategory) -> bool:
    """Whether this category's reported returns need unsmoothing before use.

    CFA Level I: Alternative Investments — categories and valuation.
    """
    return CATEGORY_PROFILES[category].smoothed_valuations


@dataclass(frozen=True, slots=True)
class FeeResult:
    """One period's fees, the investor's net outcome, and the updated mark."""

    management_fee: Decimal
    incentive_fee: Decimal
    total_fees: Decimal
    ending_value_net: Decimal
    #: Carried into the next period; never falls (SPEC §6.8).
    high_water_mark: Decimal
    #: Net-of-fee return to the investor.
    investor_return: Decimal


def hedge_fund_fees(
    *,
    beginning_value: Decimal,
    ending_value: Decimal,
    management_fee_rate: Decimal,
    incentive_fee_rate: Decimal,
    high_water_mark: Decimal | None = None,
    hurdle_rate: Decimal = ZERO,
    management_fee_basis: ManagementFeeBasis = "ENDING",
    incentive_net_of_management_fee: bool = True,
) -> FeeResult:
    """Management fee plus incentive fee, subject to a high-water mark.

    The incentive fee applies only to value above the higher of the high-water
    mark and any hurdle. The mark then ratchets up to the new net value but
    **never down**, so a manager who loses 20% and regains it earns no
    incentive fee on the recovery.

    Conventions vary by agreement and are therefore explicit parameters rather
    than assumptions: which assets the management fee is charged on, and
    whether the incentive fee is computed before or after it.

    CFA Level I: Alternative Investments — fee structures.
    """
    if beginning_value <= ZERO:
        raise NumericError("beginning value must be positive")
    for name, rate in (
        ("management_fee_rate", management_fee_rate),
        ("incentive_fee_rate", incentive_fee_rate),
    ):
        if rate < ZERO:
            raise NumericError(f"{name} cannot be negative")

    mark = beginning_value if high_water_mark is None else high_water_mark

    basis = beginning_value if management_fee_basis == "BEGINNING" else ending_value
    management_fee = management_fee_rate * basis

    incentive_base = ending_value - management_fee if incentive_net_of_management_fee else ending_value
    threshold = max(mark, beginning_value * (ONE + hurdle_rate))
    gain_above_threshold = max(incentive_base - threshold, ZERO)
    incentive_fee = incentive_fee_rate * gain_above_threshold

    total_fees = management_fee + incentive_fee
    ending_value_net = ending_value - total_fees

    return FeeResult(
        management_fee=management_fee,
        incentive_fee=incentive_fee,
        total_fees=total_fees,
        ending_value_net=ending_value_net,
        high_water_mark=max(mark, ending_value_net),
        investor_return=ending_value_net / beginning_value - ONE,
    )


def first_order_autocorrelation(returns: Sequence[Decimal]) -> Decimal:
    """Lag-1 autocorrelation of a return series.

    A materially positive value in a return series is the signature of stale or
    appraisal-based pricing: genuine period-to-period returns on a liquid asset
    are close to uncorrelated.

    CFA Level I: Alternative Investments — return smoothing.
    """
    require_min_length("returns", returns, 3)
    mean = sum(returns, ZERO) / Decimal(len(returns))
    deviations = [r - mean for r in returns]

    numerator = sum(
        (deviations[t] * deviations[t - 1] for t in range(1, len(deviations))),
        ZERO,
    )
    denominator = sum((d**2 for d in deviations), ZERO)
    if denominator == ZERO:
        raise NumericError("autocorrelation is undefined for a constant series")
    return numerator / denominator


def unsmooth_returns(
    returns: Sequence[Decimal],
    autocorrelation: Decimal,
) -> list[Decimal]:
    """Geltner first-order unsmoothing: ``r(t) = (r_obs(t) - rho r_obs(t-1)) / (1 - rho)``.

    Inverts the moving-average filter that appraisal-based valuation imposes,
    recovering a return series with more realistic volatility. Returns one
    fewer observation than it was given, since the first period has no
    predecessor.

    Use this before an illiquid sleeve's returns enter a covariance estimate.
    Feeding smoothed returns straight into the optimizer understates both its
    variance and its correlation with everything else, and the optimizer will
    happily overweight it on the strength of that error.

    CFA Level I: Alternative Investments — return smoothing.
    """
    require_min_length("returns", returns, 2)
    if autocorrelation >= ONE or autocorrelation <= -ONE:
        raise NumericError(
            f"autocorrelation must lie strictly within (-1, 1), got {autocorrelation}"
        )

    divisor = ONE - autocorrelation
    return [
        (returns[t] - autocorrelation * returns[t - 1]) / divisor for t in range(1, len(returns))
    ]
