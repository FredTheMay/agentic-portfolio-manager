"""Wiring the agents into the decision cycle (SPEC §5.4).

This is the seam between the qualitative half of the system and the
quantitative one. Agents produce categorical views; the aggregator turns them
into numeric tilts by table lookup; the tilts adjust the CAPM baseline the
optimizer starts from. Nothing else crosses.

The contract is deliberately narrow — :meth:`ViewPipeline.tilts` takes symbols
and an instant and returns a plain ``symbol -> Decimal`` map — so the backtest
engine has no idea whether views came from three LLM agents or from nowhere at
all. That is what makes SPEC §2.1(4) testable end to end: swap
:class:`AgentViewPipeline` for :class:`NoViews`, or point it at
``NullProvider``, and the identical cycle runs.

Failure policy: an agent that raises is recorded and treated as NEUTRAL. A
research call that times out should cost the portfolio its *opinion*, not its
rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Mapping, Protocol, Sequence, runtime_checkable

from src.agents.aggregator import (
    AgentOpinion,
    AggregatedView,
    ViewMapping,
    aggregate,
    load_mapping,
    tilts_for_optimizer,
)
from src.agents.fundamental import FundamentalAgent
from src.agents.macro import MacroAgent, MacroSignals, MacroView
from src.agents.research import Headline, ResearchAgent
from src.audit.log import AuditEvent, AuditLog, Standard
from src.data.edgar import Fundamentals
from src.llm.base import LLMError, Stance
from src.time.clock import ensure_utc

ZERO = Decimal(0)


@runtime_checkable
class ViewPipeline(Protocol):
    """Produces per-symbol expected-return tilts for one decision cycle."""

    def tilts(self, symbols: Sequence[str], as_of: datetime) -> dict[str, Decimal]: ...


@dataclass(frozen=True, slots=True)
class NoViews:
    """Produces no tilts at all.

    The default. With this the system is pure quantitative construction: CAPM
    expected returns, shrunk covariance, constrained optimization, IPS. Every
    result in RESULTS.md was produced this way, which is worth stating —
    the LLM is not doing the work.
    """

    def tilts(self, symbols: Sequence[str], as_of: datetime) -> dict[str, Decimal]:
        return {}


@dataclass(frozen=True, slots=True)
class CycleViews:
    """Everything the agents concluded in one cycle, kept for the audit trail."""

    as_of: datetime
    aggregated: Mapping[str, AggregatedView]
    macro: MacroView | None
    opinions: tuple[AgentOpinion, ...]

    @property
    def tilts(self) -> dict[str, Decimal]:
        return tilts_for_optimizer(self.aggregated)


@dataclass(slots=True)
class AgentViewPipeline:
    """Runs the three agents and aggregates them into tilts (SPEC §5).

    All three agents share one provider, so a single ``LLM_PROVIDER=null``
    disables the qualitative layer entirely and the cycle still completes.
    """

    research: ResearchAgent
    fundamental: FundamentalAgent
    macro: MacroAgent
    mapping: ViewMapping = field(default_factory=load_mapping)
    audit: AuditLog | None = None

    #: Point-in-time inputs, supplied by the caller. Empty is a legitimate
    #: state — a company with no filings yet, or a day with no news.
    headlines: Mapping[str, Sequence[Headline]] = field(default_factory=dict)
    fundamentals: Mapping[str, Fundamentals] = field(default_factory=dict)
    descriptions: Mapping[str, str] = field(default_factory=dict)
    macro_signals: MacroSignals | None = None

    def run(self, symbols: Sequence[str], as_of: datetime) -> CycleViews:
        """Collect every agent's view of every symbol."""
        moment = ensure_utc(as_of)
        opinions: list[AgentOpinion] = []

        macro_view: MacroView | None = None
        macro_stance, macro_conviction = Stance.NEUTRAL, 1
        if self.macro_signals is not None:
            macro_view = self._safe_macro(self.macro_signals, moment)
            if macro_view is not None:
                macro_stance = macro_view.narrative.stance
                macro_conviction = macro_view.narrative.conviction

        for symbol in sorted(symbols):
            opinions.append(self._research_opinion(symbol, moment))
            opinions.append(self._fundamental_opinion(symbol, moment))
            # Macro is a portfolio-level signal applied identically to every
            # name; the aggregator's weights decide how much it matters.
            opinions.append(AgentOpinion("macro", symbol, macro_stance, macro_conviction))

        return CycleViews(
            as_of=moment,
            aggregated=aggregate(opinions, self.mapping),
            macro=macro_view,
            opinions=tuple(opinions),
        )

    def tilts(self, symbols: Sequence[str], as_of: datetime) -> dict[str, Decimal]:
        return self.run(symbols, as_of).tilts

    # -- Per-agent calls, each degrading to NEUTRAL rather than failing -----

    def _research_opinion(self, symbol: str, as_of: datetime) -> AgentOpinion:
        try:
            view = self.research.run(
                symbol,
                self.descriptions.get(symbol, ""),
                self.headlines.get(symbol, ()),
                as_of,
            )
        except LLMError as exc:
            self._record_failure("research", symbol, as_of, exc)
            return AgentOpinion("research", symbol, Stance.NEUTRAL, 1)
        return AgentOpinion("research", symbol, view.stance, view.conviction)

    def _fundamental_opinion(self, symbol: str, as_of: datetime) -> AgentOpinion:
        current = self.fundamentals.get(symbol)
        if current is None:
            # No filing visible at this instant is a normal state, not a
            # failure, and it must not become a fabricated view.
            return AgentOpinion("fundamental", symbol, Stance.NEUTRAL, 1)
        try:
            view = self.fundamental.run(current, as_of)
        except LLMError as exc:
            self._record_failure("fundamental", symbol, as_of, exc)
            return AgentOpinion("fundamental", symbol, Stance.NEUTRAL, 1)
        return AgentOpinion("fundamental", symbol, view.stance, view.conviction)

    def _safe_macro(self, signals: MacroSignals, as_of: datetime) -> MacroView | None:
        try:
            return self.macro.run(signals)
        except LLMError as exc:
            self._record_failure("macro", None, as_of, exc)
            return None

    def _record_failure(
        self, agent: str, symbol: str | None, as_of: datetime, exc: Exception
    ) -> None:
        if self.audit is None:
            return
        self.audit.record(
            AuditEvent(
                timestamp=as_of,
                actor=f"pipeline:{agent}",
                code="AGENT_UNAVAILABLE",
                standard=Standard.V_A_DILIGENCE,
                symbol=symbol,
                detail=(
                    f"{agent} agent failed ({exc}); defaulted to NEUTRAL so the "
                    "cycle could continue"
                ),
            )
        )
