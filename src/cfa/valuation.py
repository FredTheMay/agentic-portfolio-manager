"""Equity valuation: dividend discount, FCFE, and justified multiples.

CFA Level I topic area: Equity Investments.

The system assumes semi-strong-form efficiency, so these models exist to
produce a defensible required return and a sanity check on what is priced in,
not to find mispricings.

:func:`value_equity` applies the hierarchy explicitly: DDM where a dividend
exists, FCFE otherwise, multiples as a cross-check. Every model returns
``None`` rather than a number when its assumptions break — a ``None`` that
becomes "no view" is correct; a fabricated number becomes a portfolio weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

ZERO = Decimal(0)
ONE = Decimal(1)

ValuationMethod = Literal["DDM", "FCFE", "NONE"]


def gordon_growth_value(
    dividend_next_period: Decimal,
    required_return: Decimal,
    growth_rate: Decimal,
) -> Decimal | None:
    """``V0 = D1 / (r - g)`` — the constant-growth dividend discount model.

    Returns ``None`` when ``g >= r``. The closed form is the sum of a geometric
    series that only converges for ``g < r``; outside that range the model does
    not say the stock is infinitely valuable, it says the model does not apply.

    ``r`` is normally CAPM's required return
    (:func:`src.cfa.portfolio.capm_expected_return`).

    CFA Level I: Equity Investments — dividend discount models.
    """
    if growth_rate >= required_return:
        return None
    return dividend_next_period / (required_return - growth_rate)


def sustainable_growth_rate(payout_ratio: Decimal, return_on_equity: Decimal) -> Decimal:
    """``g = (1 - payout) x ROE`` — growth funded from retained earnings alone.

    CFA Level I: Equity Investments — sustainable growth rate.
    """
    return (ONE - payout_ratio) * return_on_equity


def justified_leading_pe(
    payout_ratio: Decimal,
    required_return: Decimal,
    growth_rate: Decimal,
) -> Decimal | None:
    """``P/E1 = payout / (r - g)`` — justified P/E on *next* year's earnings.

    the leading form has no ``(1 + g)`` factor. That
    factor belongs to the trailing form, which is stated on earnings already
    reported.

    CFA Level I: Equity Investments — price multiples.
    """
    if growth_rate >= required_return:
        return None
    return payout_ratio / (required_return - growth_rate)


def justified_trailing_pe(
    payout_ratio: Decimal,
    required_return: Decimal,
    growth_rate: Decimal,
) -> Decimal | None:
    """``P/E0 = [payout (1 + g)] / (r - g)`` — justified P/E on trailing earnings.

    CFA Level I: Equity Investments — price multiples.
    """
    if growth_rate >= required_return:
        return None
    return (payout_ratio * (ONE + growth_rate)) / (required_return - growth_rate)


def enterprise_value(
    market_capitalization: Decimal,
    total_debt: Decimal,
    cash_and_equivalents: Decimal,
) -> Decimal:
    """``EV = market cap + total debt - cash``.

    cash is **subtracted**. EV is the cost of acquiring
    the operating business; the acquirer assumes the debt but immediately
    recovers the cash, so only net debt is a real cost.

    CFA Level I: Equity Investments — enterprise value.
    """
    return market_capitalization + total_debt - cash_and_equivalents


def relative_multiple_premium(multiple: Decimal, sector_median: Decimal) -> Decimal:
    """``multiple / sector median - 1`` — premium (+) or discount (-) to peers.

    The cross-check in the hierarchy, not the primary estimate: it is silent on
    whether the sector itself is fairly priced.

    CFA Level I: Equity Investments — relative valuation.
    """
    if sector_median == ZERO:
        raise ZeroDivisionError("relative multiple is undefined against a zero sector median")
    return multiple / sector_median - ONE


def free_cash_flow_to_equity(
    cash_flow_operations: Decimal,
    fixed_capital_investment: Decimal,
    net_borrowing: Decimal,
) -> Decimal:
    """``FCFE = CFO - FCInv + net borrowing``.

    Cash available to shareholders after funding operations and reinvestment,
    whether or not management chooses to pay it out — which is exactly why it
    works where DDM cannot.

    CFA Level I: Equity Investments — free cash flow valuation.
    """
    return cash_flow_operations - fixed_capital_investment + net_borrowing


def fcfe_value(
    fcfe_next_period: Decimal,
    required_return: Decimal,
    growth_rate: Decimal,
) -> Decimal | None:
    """``V0 = FCFE1 / (r - g)`` — constant-growth FCFE model.

    Same convergence guard as :func:`gordon_growth_value`.

    CFA Level I: Equity Investments — free cash flow valuation.
    """
    if growth_rate >= required_return:
        return None
    return fcfe_next_period / (required_return - growth_rate)


@dataclass(frozen=True, slots=True)
class ValuationResult:
    """An intrinsic value plus which model produced it, and why.

    ``method`` and ``reason`` exist so the audit log can record *how* a name
    was valued. "No view" is a legitimate outcome and must be distinguishable
    from "valued at zero".
    """

    value: Decimal | None
    method: ValuationMethod
    reason: str


def value_equity(
    *,
    required_return: Decimal,
    growth_rate: Decimal,
    dividend_next: Decimal | None = None,
    fcfe_next: Decimal | None = None,
) -> ValuationResult:
    """Apply the valuation hierarchy: DDM where dividends exist, else FCFE.

    A zero dividend counts as no dividend — a non-payer valued by DDM would
    come out at zero, which is a wrong answer rather than a missing one.

    CFA Level I: Equity Investments — valuation model selection.
    """
    if growth_rate >= required_return:
        return ValuationResult(
            value=None,
            method="NONE",
            reason=(
                f"growth rate {growth_rate} is not below the required return "
                f"{required_return}; constant-growth models do not converge"
            ),
        )

    if dividend_next is not None and dividend_next > ZERO:
        return ValuationResult(
            value=gordon_growth_value(dividend_next, required_return, growth_rate),
            method="DDM",
            reason="dividend-paying: valued on the constant-growth DDM",
        )

    if fcfe_next is not None:
        return ValuationResult(
            value=fcfe_value(fcfe_next, required_return, growth_rate),
            method="FCFE",
            reason="no dividend: valued on constant-growth FCFE",
        )

    return ValuationResult(
        value=None,
        method="NONE",
        reason="neither a dividend nor an FCFE estimate is available",
    )
