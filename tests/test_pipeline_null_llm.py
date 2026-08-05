"""Structural test 3 — SPEC §2.1(4): the system works with the LLM disabled.

At M0 there is no pipeline yet, so this exercises the contract every future
stage depends on: ``NullProvider`` answers any agent schema neutrally, without
network access and without any agent registering a default.

The bottom of this file is the acceptance criterion itself: the full
research → aggregate → optimize → risk → mandate → execute → reconcile cycle,
running end to end with the LLM switched off.
"""

from __future__ import annotations

import enum
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel, Field

from src.agents.aggregator import load_mapping
from src.agents.fundamental import FundamentalAgent
from src.agents.macro import MacroAgent, MacroSignals
from src.agents.pipeline import AgentViewPipeline, NoViews
from src.agents.research import Headline, ResearchAgent
from src.audit.log import AuditLog
from src.backtest.engine import BacktestConfig, result_digest, run_backtest
from src.data.edgar import Fundamentals
from src.decision.optimizer import estimate_inputs
from src.execution.simulated import SimulatedExecutor
from src.llm.base import Conviction, LLMError, LLMProvider, Stance
from src.llm.null import NULL_CONVICTION, NULL_RATIONALE, NullProvider, NullProviderError
from src.risk.ips import load_policy
from src.time.clock import UTC
from src.data.synthetic import BETAS, SECTORS, make_source

M = TypeVar("M", bound=BaseModel)
D = Decimal
PIPELINE_START = datetime(2022, 1, 3, 21, tzinfo=UTC)
PIPELINE_SYMBOLS = (
    "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL",
)


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


# ===========================================================================
# The acceptance criterion (SPEC §11): the full pipeline, LLM disabled
# ===========================================================================
#
# Everything above tests the provider contract. This section runs the actual
# decision cycle — agents, aggregator, optimizer, risk engine, mandate,
# executor, reconciliation — against NullProvider, and asserts it completes and
# produces a portfolio.
#
# The point is not that the answers are good. With the LLM disabled every view
# is NEUTRAL and every tilt is zero, so the portfolio is pure quantitative
# construction. The point is that nothing in the chain *requires* the LLM.

def null_agent_pipeline(audit: AuditLog | None = None) -> AgentViewPipeline:
    """Every agent, wired to NullProvider, with realistic point-in-time inputs."""
    provider = NullProvider()
    return AgentViewPipeline(
        research=ResearchAgent(provider=provider, audit=audit),
        fundamental=FundamentalAgent(provider=provider, audit=audit),
        macro=MacroAgent(provider=provider),
        mapping=load_mapping(),
        audit=audit,
        headlines={
            "AAA": [
                Headline(
                    PIPELINE_START + timedelta(days=200),
                    "Bookings accelerated",
                    "https://news.test/aaa",
                )
            ]
        },
        fundamentals={
            "AAA": Fundamentals(
                symbol="AAA",
                as_of=PIPELINE_START,
                period_end=datetime(2021, 12, 31, tzinfo=UTC),
                revenue=D("1000"),
                net_income=D("105"),
                total_assets=D("2000"),
                total_equity=D("800"),
                cash_flow_operations=D("180"),
            )
        },
        macro_signals=MacroSignals(
            as_of=PIPELINE_START,
            term_spread=D("1.2"),
            unemployment=D("3.8"),
            unemployment_change=D("-0.6"),
            inflation_yoy=D("0.02"),
        ),
    )


def pipeline_config() -> BacktestConfig:
    return BacktestConfig(
        start=PIPELINE_START,
        end=PIPELINE_START + timedelta(days=560),
        initial_cash=D("100000.00"),
        symbols=PIPELINE_SYMBOLS,
        benchmark_symbol="SPY",
        estimation_window=100,
    )


def test_the_full_cycle_runs_with_the_llm_disabled() -> None:
    """SPEC §11: 'Full pipeline runs with NullProvider (no LLM)'."""
    audit = AuditLog()
    result = run_backtest(
        pipeline_config(),
        make_source(),
        SimulatedExecutor(),
        load_policy(),
        SECTORS,
        BETAS,
        views=null_agent_pipeline(audit),
    )

    assert result.cycles, "no decision cycle ran"
    assert result.executed, "no cycle reached the executor"
    assert len(result.equity_curve) > 100

    # The cycle completed all the way through reconciliation.
    for cycle in result.executed:
        assert cycle.mandate is not None
        assert cycle.report is not None
        assert cycle.reconciliation is not None


def test_every_agent_view_is_neutral_and_every_tilt_is_zero() -> None:
    # With the LLM off there is no qualitative opinion, and "no opinion" must
    # not quietly become a tilt.
    views = null_agent_pipeline().run(PIPELINE_SYMBOLS, PIPELINE_START)

    assert all(o.stance is Stance.NEUTRAL for o in views.opinions)
    assert all(tilt == D("0") for tilt in views.tilts.values())


def test_disabling_the_llm_changes_nothing_about_the_result() -> None:
    # The strongest form of SPEC §2.1(4): with NullProvider the agent pipeline
    # and no pipeline at all must produce byte-identical output. If they
    # differed, the LLM would be influencing the portfolio while switched off.
    config = pipeline_config()
    policy = load_policy()

    with_agents = run_backtest(
        config, make_source(), SimulatedExecutor(), policy, SECTORS, BETAS,
        views=null_agent_pipeline(),
    )
    without = run_backtest(
        config, make_source(), SimulatedExecutor(), policy, SECTORS, BETAS, views=NoViews()
    )

    assert result_digest(with_agents) == result_digest(without)


def test_the_pipeline_survives_an_agent_that_fails() -> None:
    # A research call that times out should cost the portfolio its opinion,
    # not its rebalance.
    class BrokenProvider(LLMProvider):
        name = "broken"

        def _complete(self, system: str, user: str, schema: type[M]) -> M:
            raise LLMError("provider unavailable")

    audit = AuditLog()
    broken = BrokenProvider()
    pipeline = AgentViewPipeline(
        research=ResearchAgent(provider=broken, audit=audit),
        fundamental=FundamentalAgent(provider=broken, audit=audit),
        macro=MacroAgent(provider=broken),
        mapping=load_mapping(),
        audit=audit,
        headlines={"AAA": [Headline(PIPELINE_START, "News", "https://news.test/a")]},
    )

    views = pipeline.run(("AAA", "BBB"), PIPELINE_START)

    assert all(tilt == D("0") for tilt in views.tilts.values())
    assert audit.by_code("AGENT_UNAVAILABLE"), "the failure must be recorded, not swallowed"


def test_the_tilts_actually_reach_the_optimizer() -> None:
    # Guards against the pipeline being wired up but inert — the failure mode
    # where every agent runs, produces views, and nothing consumes them.
    observations = [
        [D("0.010"), D("0.008")],
        [D("-0.006"), D("-0.004")],
        [D("0.012"), D("0.009")],
        [D("0.001"), D("0.002")],
    ]
    betas = {"AAA": D("1.2"), "BBB": D("0.9")}

    baseline = estimate_inputs(
        ("AAA", "BBB"), observations, betas, D("0.09"), D("0.03")
    )
    tilted = estimate_inputs(
        ("AAA", "BBB"), observations, betas, D("0.09"), D("0.03"),
        tilts={"AAA": D("0.02")},
    )

    assert tilted.expected_returns[0] == baseline.expected_returns[0] + D("0.02")
    assert tilted.expected_returns[1] == baseline.expected_returns[1]
