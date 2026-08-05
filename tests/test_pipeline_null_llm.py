"""Structural test 3 — SPEC §2.1(4): the system works with the LLM disabled.

At M0 there is no pipeline yet, so this exercises the contract every future
stage depends on: ``NullProvider`` answers any agent schema neutrally, without
network access and without any agent registering a default.

**This file grows with each milestone.** By M7 it runs the full research →
aggregate → optimize → risk → mandate cycle under ``NullProvider`` and asserts
the cycle completes.
"""

from __future__ import annotations

import enum
from typing import Any

import pytest
from pydantic import BaseModel, Field

from src.llm.base import Conviction, LLMProvider, Stance
from src.llm.null import NULL_CONVICTION, NULL_RATIONALE, NullProvider, NullProviderError


class Citation(BaseModel):
    """A dated source. SPEC §5.1 discards any view lacking one."""

    url: str
    published: str
    quote: str


class ResearchView(BaseModel):
    """Shape of the M7 Research Agent output (SPEC §5.1)."""

    ticker: str
    stance: Stance
    conviction: Conviction
    rationale: str
    citations: list[Citation] = Field(default_factory=list)


class MacroNarrative(BaseModel):
    """Narrative-only output; the cycle phase is classified by rule (SPEC §5.3)."""

    stance: Stance
    commentary: str
    caveat: str | None = None


def test_null_provider_is_an_llm_provider() -> None:
    assert isinstance(NullProvider(), LLMProvider)
    assert NullProvider().name == "null"


def test_null_provider_returns_neutral_research_view() -> None:
    view = NullProvider().complete("system", "user", ResearchView)

    assert isinstance(view, ResearchView)
    assert view.stance is Stance.NEUTRAL
    assert view.conviction == NULL_CONVICTION
    assert view.citations == []
    # The rationale states why the view is neutral rather than leaving a blank
    # that reads like considered judgment in the audit log.
    assert view.rationale == NULL_RATIONALE


def test_null_provider_handles_optionals_and_nested_models() -> None:
    narrative = NullProvider().complete("system", "user", MacroNarrative)

    assert narrative.stance is Stance.NEUTRAL
    assert narrative.caveat is None


def test_null_provider_is_deterministic() -> None:
    # SPEC §9: identical inputs produce identical output.
    provider = NullProvider()
    first = provider.complete("s", "u", ResearchView)
    second = provider.complete("s", "u", ResearchView)
    assert first == second


def test_null_provider_ignores_prompt_content() -> None:
    # No network, no prompt sensitivity — the whole point of the null path.
    provider = NullProvider()
    bullish = provider.complete("be bullish", "AAPL is going to the moon", ResearchView)
    assert bullish.stance is Stance.NEUTRAL


# --- Neutral-value derivation for the shapes future agents will use ---------


class RequiredCollections(BaseModel):
    """Collections without defaults still have to resolve to something empty."""

    stance: Stance
    citations: list[Citation]
    tags: set[str]
    scores: dict[str, str]
    variadic: tuple[str, ...]
    pair: tuple[str, Stance]
    flagged: bool


class Nested(BaseModel):
    stance: Stance
    detail: MacroNarrative


def test_null_provider_fills_required_collections_empty() -> None:
    result = NullProvider().complete("s", "u", RequiredCollections)

    assert result.citations == []
    assert result.tags == set()
    assert result.scores == {}
    assert result.variadic == ()
    # A fixed-length tuple cannot be empty, so every slot gets its own neutral
    # value rather than the whole field collapsing to ().
    assert result.pair == (NULL_RATIONALE, Stance.NEUTRAL)
    assert result.flagged is False


def test_null_provider_recurses_into_nested_models() -> None:
    result = NullProvider().complete("s", "u", Nested)

    assert result.stance is Stance.NEUTRAL
    assert result.detail.stance is Stance.NEUTRAL
    assert result.detail.caveat is None


def test_null_provider_respects_explicit_defaults() -> None:
    class WithDefault(BaseModel):
        stance: Stance
        source: str = "preset"

    assert NullProvider().complete("s", "u", WithDefault).source == "preset"


class Phase(str, enum.Enum):
    """A str enum with no NEUTRAL member — nothing here is a neutral answer."""

    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"


class PhaseView(BaseModel):
    phase: Phase


def test_enum_without_neutral_member_fails_loudly() -> None:
    # Silently picking the first member would put an unearned macro call into
    # the portfolio. Refusing is the correct failure.
    with pytest.raises(NullProviderError, match="no NEUTRAL member"):
        NullProvider().complete("s", "u", PhaseView)


class Untyped(BaseModel):
    stance: Stance
    payload: Any


def test_underivable_field_fails_loudly() -> None:
    with pytest.raises(NullProviderError, match="no neutral value"):
        NullProvider().complete("s", "u", Untyped)
