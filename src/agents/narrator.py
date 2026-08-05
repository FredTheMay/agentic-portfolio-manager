"""Narrator (SPEC §5.5). A formatter, not a decision maker.

Receives decisions and metrics that are already final and phrases them for the
dashboard. It cannot change anything, and :func:`verify_numbers_echoed` checks
that it did not: any figure appearing in the narrative that was not in the
input is flagged as ``NARRATOR_ALTERED_FIGURE``.

CFA Standard V(B), communication with clients, is enforced structurally rather
than by instruction: the schema has separate ``facts`` and ``opinions`` lists,
so a reader can tell which is which without trusting the model to have kept
them apart in prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from src.agents.schemas import DecisionNarrative
from src.audit.log import AuditEvent, AuditLog, Standard
from src.llm.base import LLMProvider
from src.time.clock import ensure_utc

SYSTEM_PROMPT = """You are writing a plain-language summary of decisions that
have already been made.

Rules you must follow:
- Every number you mention must be copied EXACTLY from the input. Do not round,
  rescale, convert, or recompute anything.
- Do not introduce any figure that is not in the input.
- Do not second-guess the decisions. They are final.
- Put statements of fact in `facts` and interpretation in `opinions`. Keep them
  strictly separate."""

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_IGNORED = {"0", "1", "2", "3", "4", "5", "10", "100"}


def render_context(
    decisions: Mapping[str, Decimal],
    metrics: Mapping[str, str],
) -> str:
    """Format final weights and metrics for the prompt."""
    weights = "\n".join(
        f"  {symbol}: {weight.quantize(Decimal('0.0001'))}"
        for symbol, weight in sorted(decisions.items())
    )
    figures = "\n".join(f"  {name}: {value}" for name, value in sorted(metrics.items()))
    return f"Final target weights:\n{weights or '  (all cash)'}\n\nMetrics:\n{figures}"


def verify_numbers_echoed(narrative: DecisionNarrative, context: str) -> list[str]:
    """Numeric tokens in the narrative that were not in its input.

    The narrator is a formatter; a figure it produced that nobody gave it is
    either a recomputation or an invention, and both are CFA Standard I(C)
    problems.
    """
    text = " ".join(
        [narrative.headline, narrative.summary, *narrative.facts, *narrative.opinions]
    )
    return [
        token
        for token in _NUMBER.findall(text)
        if token not in _IGNORED and token not in context
    ]


@dataclass(slots=True)
class Narrator:
    """Formats final decisions for display."""

    provider: LLMProvider
    audit: AuditLog | None = None

    def run(
        self,
        decisions: Mapping[str, Decimal],
        metrics: Mapping[str, str],
        as_of: datetime,
    ) -> DecisionNarrative:
        context = render_context(decisions, metrics)
        narrative = self.provider.complete(SYSTEM_PROMPT, context, DecisionNarrative)

        altered = verify_numbers_echoed(narrative, context)
        if altered and self.audit is not None:
            self.audit.record(
                AuditEvent(
                    timestamp=ensure_utc(as_of),
                    actor=f"narrator:{self.provider.name}",
                    code="NARRATOR_ALTERED_FIGURE",
                    standard=Standard.I_C_MISREPRESENTATION,
                    detail=f"narrative contains figures absent from its input: {altered}",
                )
            )
        return narrative
