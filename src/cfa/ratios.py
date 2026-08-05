"""Financial statement ratio analysis.

CFA Level I topic area: Financial Statement Analysis (SPEC §6.4).

Pure functions, zero I/O, ``Decimal`` throughout — no float appears anywhere in
this module, because every quantity here is an exact ratio of reported
currency amounts.

Balance-sheet inputs should be **averages** of beginning and ending balances
where they are divided into a flow measure (income, revenue, COGS). Mixing a
point-in-time balance with a full-year flow overstates turnover and return
measures, and the argument names say which is expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

ZERO = Decimal(0)


class RatioError(ZeroDivisionError):
    """Raised when a ratio's denominator is zero."""


def _divide(numerator: Decimal, denominator: Decimal, label: str) -> Decimal:
    if denominator == ZERO:
        raise RatioError(f"{label} is undefined: denominator is zero")
    return numerator / denominator


# --- Liquidity -------------------------------------------------------------


def current_ratio(current_assets: Decimal, current_liabilities: Decimal) -> Decimal:
    """``current assets / current liabilities``.

    CFA Level I: Financial Statement Analysis — liquidity ratios.
    """
    return _divide(current_assets, current_liabilities, "current ratio")


def quick_ratio(
    cash: Decimal,
    short_term_investments: Decimal,
    receivables: Decimal,
    current_liabilities: Decimal,
) -> Decimal:
    """``(cash + short-term investments + receivables) / current liabilities``.

    The acid test: inventory is excluded because converting it to cash requires
    finding a buyer, which is exactly what a firm under liquidity stress cannot
    reliably do.

    CFA Level I: Financial Statement Analysis — liquidity ratios.
    """
    return _divide(
        cash + short_term_investments + receivables,
        current_liabilities,
        "quick ratio",
    )


# --- Solvency --------------------------------------------------------------


def debt_to_equity(total_debt: Decimal, total_equity: Decimal) -> Decimal:
    """``total debt / total equity``.

    CFA Level I: Financial Statement Analysis — solvency ratios.
    """
    return _divide(total_debt, total_equity, "debt-to-equity")


def interest_coverage(ebit: Decimal, interest_expense: Decimal) -> Decimal:
    """``EBIT / interest expense`` — times interest earned.

    CFA Level I: Financial Statement Analysis — solvency ratios.
    """
    return _divide(ebit, interest_expense, "interest coverage")


def equity_multiplier(average_total_assets: Decimal, average_equity: Decimal) -> Decimal:
    """``average total assets / average equity`` — the leverage term in DuPont.

    CFA Level I: Financial Statement Analysis — DuPont analysis.
    """
    return _divide(average_total_assets, average_equity, "equity multiplier")


# --- Profitability ---------------------------------------------------------


def gross_profit_margin(gross_profit: Decimal, revenue: Decimal) -> Decimal:
    """``gross profit / revenue``.

    CFA Level I: Financial Statement Analysis — profitability ratios.
    """
    return _divide(gross_profit, revenue, "gross profit margin")


def operating_profit_margin(ebit: Decimal, revenue: Decimal) -> Decimal:
    """``EBIT / revenue``.

    CFA Level I: Financial Statement Analysis — profitability ratios.
    """
    return _divide(ebit, revenue, "operating profit margin")


def net_profit_margin(net_income: Decimal, revenue: Decimal) -> Decimal:
    """``net income / revenue``.

    CFA Level I: Financial Statement Analysis — profitability ratios.
    """
    return _divide(net_income, revenue, "net profit margin")


def return_on_assets(net_income: Decimal, average_total_assets: Decimal) -> Decimal:
    """``net income / average total assets``.

    CFA Level I: Financial Statement Analysis — profitability ratios.
    """
    return _divide(net_income, average_total_assets, "return on assets")


def return_on_equity(net_income: Decimal, average_equity: Decimal) -> Decimal:
    """``net income / average equity``.

    CFA Level I: Financial Statement Analysis — profitability ratios.
    """
    return _divide(net_income, average_equity, "return on equity")


# --- Activity --------------------------------------------------------------


def inventory_turnover(cogs: Decimal, average_inventory: Decimal) -> Decimal:
    """``COGS / average inventory``.

    Uses COGS rather than revenue: inventory is carried at cost, so revenue in
    the numerator would mix a marked-up flow with an at-cost balance.

    CFA Level I: Financial Statement Analysis — activity ratios.
    """
    return _divide(cogs, average_inventory, "inventory turnover")


def receivables_turnover(revenue: Decimal, average_receivables: Decimal) -> Decimal:
    """``revenue / average receivables``.

    CFA Level I: Financial Statement Analysis — activity ratios.
    """
    return _divide(revenue, average_receivables, "receivables turnover")


def total_asset_turnover(revenue: Decimal, average_total_assets: Decimal) -> Decimal:
    """``revenue / average total assets``.

    CFA Level I: Financial Statement Analysis — activity ratios.
    """
    return _divide(revenue, average_total_assets, "total asset turnover")


# --- DuPont ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DuPontThreeStep:
    """ROE decomposed into margin, efficiency, and leverage."""

    net_profit_margin: Decimal
    asset_turnover: Decimal
    equity_multiplier: Decimal

    @property
    def return_on_equity(self) -> Decimal:
        return self.net_profit_margin * self.asset_turnover * self.equity_multiplier


def dupont_three_step(
    net_income: Decimal,
    revenue: Decimal,
    average_total_assets: Decimal,
    average_equity: Decimal,
) -> DuPontThreeStep:
    """``ROE = net margin x asset turnover x equity multiplier``.

    Separates *how* a firm earns its ROE. Two firms at 15% ROE are not
    equivalent if one gets there on margin and the other on leverage — the
    second is far more fragile to a downturn.

    CFA Level I: Financial Statement Analysis — DuPont analysis.
    """
    return DuPontThreeStep(
        net_profit_margin=net_profit_margin(net_income, revenue),
        asset_turnover=total_asset_turnover(revenue, average_total_assets),
        equity_multiplier=equity_multiplier(average_total_assets, average_equity),
    )


@dataclass(frozen=True, slots=True)
class DuPontFiveStep:
    """ROE decomposed further, separating tax and financing effects."""

    tax_burden: Decimal
    interest_burden: Decimal
    operating_margin: Decimal
    asset_turnover: Decimal
    equity_multiplier: Decimal

    @property
    def return_on_equity(self) -> Decimal:
        return (
            self.tax_burden
            * self.interest_burden
            * self.operating_margin
            * self.asset_turnover
            * self.equity_multiplier
        )


def dupont_five_step(
    net_income: Decimal,
    ebt: Decimal,
    ebit: Decimal,
    revenue: Decimal,
    average_total_assets: Decimal,
    average_equity: Decimal,
) -> DuPontFiveStep:
    """``ROE = tax burden x interest burden x EBIT margin x turnover x leverage``.

    Splits the three-step net margin into tax burden (``NI/EBT``), interest
    burden (``EBT/EBIT``), and operating margin, which is what lets you see
    whether leverage is actually paying: more debt raises the equity multiplier
    but lowers the interest burden, and the five-step form shows which effect
    won.

    CFA Level I: Financial Statement Analysis — DuPont analysis.
    """
    return DuPontFiveStep(
        tax_burden=_divide(net_income, ebt, "tax burden"),
        interest_burden=_divide(ebt, ebit, "interest burden"),
        operating_margin=operating_profit_margin(ebit, revenue),
        asset_turnover=total_asset_turnover(revenue, average_total_assets),
        equity_multiplier=equity_multiplier(average_total_assets, average_equity),
    )


# --- Earnings quality ------------------------------------------------------


def accruals_ratio(
    net_income: Decimal,
    cash_flow_operations: Decimal,
    average_total_assets: Decimal,
) -> Decimal:
    """``(net income - CFO) / average total assets``.

    The balance-sheet-free earnings-quality screen. Accrual accounting lets
    reported profit run ahead of cash collection; a persistently positive
    accruals ratio means the gap is widening, which historically predicts both
    earnings reversals and restatements. Negative is good — cash exceeds
    reported profit.

    Used as a deterministic pre-screen alongside the solvency thresholds
    (SPEC §6.4), never as an LLM judgment call.

    CFA Level I: Financial Statement Analysis — earnings quality.
    """
    return _divide(net_income - cash_flow_operations, average_total_assets, "accruals ratio")
