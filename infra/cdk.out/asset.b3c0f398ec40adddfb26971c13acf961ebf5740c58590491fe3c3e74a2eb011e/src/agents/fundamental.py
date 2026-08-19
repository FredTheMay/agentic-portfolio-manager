"""Fundamental analyst: interprets a ratio table computed in Python.

The model receives finished ratios and returns a categorical view. Its
arithmetic is never trusted because it is not doing any.

``hallucinated_figures`` flags numeric tokens in the rationale that are absent
from the input table. The check is deliberately crude because the failure it
catches is crude: a model inventing a plausible figure it was never given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Mapping

from src.agents.schemas import FundamentalView
from src.audit.log import AuditEvent, AuditLog, Standard
from src.cfa import ratios as rt
from src.data.edgar import Fundamentals, average, total_debt
from src.llm.base import LLMProvider, Stance
from src.time.clock import ensure_utc

SYSTEM_PROMPT = """You are a fundamental analyst.

You will be given a table of financial ratios that have ALREADY been computed.
Your job is to interpret them, not to recompute them.

Rules you must follow:
- Output BULLISH, NEUTRAL, or BEARISH, plus a conviction from 1 to 5.
- You may NOT introduce any figure that is not in the table above. Do not
  estimate, extrapolate, or recall figures from memory.
- List the ratio names you relied on in figures_cited.
- If the table is too sparse to support a view, return NEUTRAL."""

#: Matches numeric tokens in prose, including negatives and decimals.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: Small integers are ordinary English ("the top 3 lines", "over 5 years") and
#: flagging them would drown the real signal in noise.
_IGNORED = {"0", "1", "2", "3", "4", "5", "10", "100"}


def ratio_table(current: Fundamentals, prior: Fundamentals | None = None) -> dict[str, Decimal]:
    """Compute every available ratio from point-in-time fundamentals.

    Ratios whose inputs are missing are omitted rather than zero-filled: a
    missing figure must stay distinguishable from a figure that is zero.
    Balance-sheet items are averaged with the prior period where one is
    available, since dividing a full-year flow by a period-end stock overstates
    turnover and return measures.
    """
    table: dict[str, Decimal] = {}

    assets = average(current.total_assets, prior.total_assets if prior else None)
    equity = average(current.total_equity, prior.total_equity if prior else None)
    inventory = average(current.inventory, prior.inventory if prior else None)
    receivables = average(current.receivables, prior.receivables if prior else None)
    assets = assets if assets is not None else current.total_assets
    equity = equity if equity is not None else current.total_equity
    inventory = inventory if inventory is not None else current.inventory
    receivables = receivables if receivables is not None else current.receivables

    revenue = current.revenue
    net_income = current.net_income
    ebit = current.operating_income
    debt = total_debt(current)

    try:
        if current.current_assets is not None and current.current_liabilities is not None:
            table["current_ratio"] = rt.current_ratio(
                current.current_assets, current.current_liabilities
            )
        if (
            current.cash is not None
            and current.receivables is not None
            and current.current_liabilities is not None
        ):
            table["quick_ratio"] = rt.quick_ratio(
                current.cash, Decimal(0), current.receivables, current.current_liabilities
            )
        if debt is not None and equity is not None:
            table["debt_to_equity"] = rt.debt_to_equity(debt, equity)
        if ebit is not None and current.interest_expense is not None:
            table["interest_coverage"] = rt.interest_coverage(ebit, current.interest_expense)
        if current.gross_profit is not None and revenue is not None:
            table["gross_margin"] = rt.gross_profit_margin(current.gross_profit, revenue)
        if ebit is not None and revenue is not None:
            table["operating_margin"] = rt.operating_profit_margin(ebit, revenue)
        if net_income is not None and revenue is not None:
            table["net_margin"] = rt.net_profit_margin(net_income, revenue)
        if net_income is not None and assets is not None:
            table["return_on_assets"] = rt.return_on_assets(net_income, assets)
        if net_income is not None and equity is not None:
            table["return_on_equity"] = rt.return_on_equity(net_income, equity)
        if current.cost_of_goods_sold is not None and inventory is not None:
            table["inventory_turnover"] = rt.inventory_turnover(
                current.cost_of_goods_sold, inventory
            )
        if revenue is not None and receivables is not None:
            table["receivables_turnover"] = rt.receivables_turnover(revenue, receivables)
        if revenue is not None and assets is not None:
            table["total_asset_turnover"] = rt.total_asset_turnover(revenue, assets)
        if (
            net_income is not None
            and current.cash_flow_operations is not None
            and assets is not None
        ):
            table["accruals_ratio"] = rt.accruals_ratio(
                net_income, current.cash_flow_operations, assets
            )
        if (
            net_income is not None
            and revenue is not None
            and assets is not None
            and equity is not None
        ):
            dupont = rt.dupont_three_step(net_income, revenue, assets, equity)
            table["dupont_net_margin"] = dupont.net_profit_margin
            table["dupont_asset_turnover"] = dupont.asset_turnover
            table["dupont_equity_multiplier"] = dupont.equity_multiplier
    except rt.RatioError:
        # A zero denominator in one filing is a data quality problem, not a
        # crash. Whatever was computed before it stands.
        pass

    return table


def render_table(table: Mapping[str, Decimal]) -> str:
    """Format the ratio table for the prompt, four decimal places."""
    return "\n".join(
        f"  {name}: {value.quantize(Decimal('0.0001'))}" for name, value in sorted(table.items())
    )


def hallucinated_figures(rationale: str, table: Mapping[str, Decimal]) -> list[str]:
    """Numeric tokens in the rationale that do not appear in the input table.

    CFA Standard I(C). Compares against the rendered table rather than against
    the raw Decimals, so a model quoting "0.1234" is matched against what it
    was actually shown.
    """
    shown = render_table(table)
    found: list[str] = []
    for token in _NUMBER.findall(rationale):
        if token in _IGNORED:
            continue
        try:
            Decimal(token)
        except InvalidOperation:
            continue
        if token not in shown:
            found.append(token)
    return found


@dataclass(slots=True)
class FundamentalAgent:
    """Interprets a deterministically computed ratio table."""

    provider: LLMProvider
    audit: AuditLog | None = None

    def run(
        self,
        current: Fundamentals,
        as_of: datetime,
        prior: Fundamentals | None = None,
    ) -> FundamentalView:
        table = ratio_table(current, prior)
        if not table:
            return FundamentalView(
                ticker=current.symbol,
                stance=Stance.NEUTRAL,
                conviction=1,
                rationale="no ratios could be computed from the filings visible at this date",
                figures_cited=[],
            )

        user = (
            f"Company: {current.symbol}\n"
            f"Latest fiscal period visible: "
            f"{current.period_end.date().isoformat() if current.period_end else 'unknown'}\n\n"
            f"Computed ratios:\n{render_table(table)}"
        )
        view = self.provider.complete(SYSTEM_PROMPT, user, FundamentalView)

        invented = hallucinated_figures(view.rationale, table)
        if invented and self.audit is not None:
            self.audit.record(
                AuditEvent(
                    timestamp=ensure_utc(as_of),
                    actor=f"fundamental:{self.provider.name}",
                    code="HALLUCINATED_FIGURE",
                    standard=Standard.I_C_MISREPRESENTATION,
                    symbol=current.symbol,
                    detail=f"rationale cites figures absent from the input table: {invented}",
                )
            )
        return view
