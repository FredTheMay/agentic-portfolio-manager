"""Research Agent (SPEC §5.1).

Reads recent headlines and forms a categorical view. Implements CFA Standard
V(A), reasonable basis, mechanically: **a view without at least one dated
citation is discarded to NEUTRAL.** Not warned about, not logged and kept —
discarded. A recommendation the model cannot point at a source for is exactly
the output that is most confident and least grounded.

Headlines are filtered by ``as_of`` before they reach the prompt. An agent that
reads tomorrow's news is a lookahead bug wearing a costume, and it would be
invisible in the backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from src.agents.schemas import ResearchView
from src.audit.log import AuditEvent, AuditLog, Standard
from src.llm.base import LLMProvider, Stance
from src.time.clock import ensure_utc

#: SPEC §5.1: headlines from the last 14 days.
LOOKBACK = timedelta(days=14)

SYSTEM_PROMPT = """You are an equity research analyst.

You will be given a company, a description, and recent headlines. Return a
categorical view only.

Rules you must follow:
- Output BULLISH, NEUTRAL, or BEARISH, plus a conviction from 1 to 5.
- You may NOT output price targets, valuations, percentages, or any number
  other than the conviction ordinal. Numeric analysis is done elsewhere.
- Every view must cite at least one of the headlines you were given, with its
  URL and publication date. If the headlines do not support a directional
  view, return NEUTRAL.
- Do not use knowledge beyond the headlines provided."""


@dataclass(frozen=True, slots=True)
class Headline:
    """One dated news item."""

    published: datetime
    title: str
    url: str
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "published", ensure_utc(self.published))


def visible_headlines(
    headlines: Sequence[Headline],
    as_of: datetime,
    lookback: timedelta = LOOKBACK,
) -> list[Headline]:
    """Headlines published in ``[as_of - lookback, as_of]``.

    The upper bound is the point: a headline published after ``as_of`` did not
    exist yet and must never reach the prompt.
    """
    moment = ensure_utc(as_of)
    earliest = moment - lookback
    return sorted(
        (h for h in headlines if earliest <= h.published <= moment),
        key=lambda h: (h.published, h.url),
    )


def _neutral(ticker: str, reason: str) -> ResearchView:
    return ResearchView(
        ticker=ticker,
        stance=Stance.NEUTRAL,
        conviction=1,
        rationale=reason,
        citations=[],
    )


@dataclass(slots=True)
class ResearchAgent:
    """Turns headlines into a categorical, cited view."""

    provider: LLMProvider
    audit: AuditLog | None = None
    minimum_citations: int = 1

    def run(
        self,
        ticker: str,
        description: str,
        headlines: Sequence[Headline],
        as_of: datetime,
    ) -> ResearchView:
        visible = visible_headlines(headlines, as_of)
        if not visible:
            return _neutral(ticker, "no headlines in the lookback window")

        rendered = "\n".join(
            f"- [{h.published.date().isoformat()}] {h.title} ({h.url})"
            f"{(' — ' + h.summary) if h.summary else ''}"
            for h in visible
        )
        user = (
            f"Company: {ticker}\nDescription: {description}\n\n"
            f"Headlines available as of {as_of.isoformat()}:\n{rendered}"
        )

        view = self.provider.complete(SYSTEM_PROMPT, user, ResearchView)

        if len(view.citations) < self.minimum_citations:
            # CFA Standard V(A): no reasonable basis, so no recommendation.
            if self.audit is not None:
                self.audit.record(
                    AuditEvent(
                        timestamp=ensure_utc(as_of),
                        actor=f"research:{self.provider.name}",
                        code="MISSING_CITATION",
                        standard=Standard.V_A_DILIGENCE,
                        symbol=ticker,
                        detail=(
                            f"view {view.stance.value} discarded to NEUTRAL: "
                            f"{len(view.citations)} citations, {self.minimum_citations} required"
                        ),
                    )
                )
            return _neutral(ticker, "discarded to NEUTRAL: no dated citation supplied")

        allowed = {h.url for h in visible}
        invented = [c.url for c in view.citations if c.url not in allowed]
        if invented:
            # A citation to a source that was never supplied is fabricated.
            if self.audit is not None:
                self.audit.record(
                    AuditEvent(
                        timestamp=ensure_utc(as_of),
                        actor=f"research:{self.provider.name}",
                        code="FABRICATED_CITATION",
                        standard=Standard.I_C_MISREPRESENTATION,
                        symbol=ticker,
                        detail=f"cited sources not provided in the prompt: {invented}",
                    )
                )
            return _neutral(ticker, "discarded to NEUTRAL: cited an unsupplied source")

        return view
