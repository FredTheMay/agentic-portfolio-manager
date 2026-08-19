"""Agent response schemas.

Each is validated by :func:`~src.llm.schema_guard.validate_llm_schema` before a
request is made, so none may contain a numeric field. The only integer
permitted is :data:`~src.llm.base.Conviction`, a 1-5 ordinal that becomes a
number by table lookup rather than by arithmetic.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

from src.llm.base import Conviction, Stance


class Citation(BaseModel):
    """A dated source with a URL.

    CFA Standard V(A), reasonable basis: a recommendation without support is
    not a recommendation. :class:`ResearchView` instances lacking one are
    discarded to NEUTRAL.
    """

    url: str = Field(description="Direct link to the source")
    published: str = Field(description="Publication date, ISO-8601")
    quote: str = Field(description="The sentence this view rests on")


class ResearchView(BaseModel):
    """Output of the Research Agent."""

    ticker: str
    stance: Stance
    conviction: Conviction
    rationale: str
    citations: list[Citation] = Field(default_factory=list)


class FundamentalView(BaseModel):
    """Output of the Fundamental Analyst.

    The model receives a finished ratio table computed in Python and returns an
    interpretation. ``figures_cited`` exists so the ``HALLUCINATED_FIGURE``
    check can compare what the model claimed against what it was given.
    """

    ticker: str
    stance: Stance
    conviction: Conviction
    rationale: str
    #: Ratio names the model says it relied on, e.g. "return_on_equity".
    figures_cited: list[str] = Field(default_factory=list)


class CyclePhase(str, enum.Enum):
    """Business-cycle phase.

    Classified by **rule** in :mod:`src.agents.macro`, never by the model. It appears in a schema only so the narrative can be told which
    phase the rule chose.
    """

    EXPANSION = "EXPANSION"
    PEAK = "PEAK"
    CONTRACTION = "CONTRACTION"
    TROUGH = "TROUGH"
    NEUTRAL = "NEUTRAL"


class MacroNarrative(BaseModel):
    """Output of the Macro/Regime Agent.

    Narrative only. The phase is decided by rule and passed *in*; the model
    writes prose about it and expresses a categorical stance.
    """

    stance: Stance
    conviction: Conviction
    commentary: str
    caveat: str | None = None


class DecisionNarrative(BaseModel):
    """Output of the Narrator.

    A formatter, not a decision maker. It receives decisions and metrics that
    are already final and must echo every number verbatim.
    """

    headline: str
    summary: str
    #: Statements of fact, traceable to inputs.
    facts: list[str] = Field(default_factory=list)
    #: Interpretation. Separated from facts per CFA Standard V(B).
    opinions: list[str] = Field(default_factory=list)
