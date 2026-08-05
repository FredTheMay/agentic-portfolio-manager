"""LLM agents, provider resilience, and the aggregator (SPEC §2.1, §5, §8, M7)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from src.agents.aggregator import (
    AgentOpinion,
    MappingError,
    aggregate,
    load_mapping,
    mapping_from_document,
    tilts_for_optimizer,
)
from src.agents.fundamental import (
    FundamentalAgent,
    hallucinated_figures,
    ratio_table,
    render_table,
)
from src.agents.macro import (
    CyclePhase,
    MacroAgent,
    MacroSignals,
    classify_phase,
    read_signals,
)
from src.agents.narrator import Narrator, render_context, verify_numbers_echoed
from src.agents.research import Headline, ResearchAgent, visible_headlines
from src.agents.schemas import DecisionNarrative, FundamentalView, ResearchView
from src.audit.log import AuditEvent, AuditLog, Standard
from src.data.edgar import Fundamentals
from src.data.pit import PointInTimeSeries, Vintage
from src.llm.base import InvalidResponseError, LLMError, LLMProvider, Stance, TokenBucket
from src.llm.cache import (
    CachingProvider,
    FailoverProvider,
    LLMResponseCache,
    ResilientProvider,
    cache_key,
)
from src.llm.null import NullProvider
from src.time.clock import UTC, SimulationClock

D = Decimal
M = TypeVar("M", bound=BaseModel)
NOW = datetime(2024, 6, 3, tzinfo=UTC)


class ScriptedProvider(LLMProvider):
    """Returns a prepared response, or raises a prepared error."""

    name = "scripted"

    def __init__(self, response: BaseModel | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response  # type: ignore[return-value]


def bullish(ticker: str = "AAA", citations: list[dict[str, str]] | None = None) -> ResearchView:
    return ResearchView.model_validate(
        {
            "ticker": ticker,
            "stance": "BULLISH",
            "conviction": 4,
            "rationale": "Order book commentary was constructive.",
            "citations": citations
            if citations is not None
            else [
                {
                    "url": "https://news.test/a",
                    "published": "2024-06-01",
                    "quote": "Bookings accelerated.",
                }
            ],
        }
    )


# ===========================================================================
# Provider resilience (SPEC §8)
# ===========================================================================


def test_token_bucket_limits_then_refills() -> None:
    clock = SimulationClock(NOW)
    bucket = TokenBucket(capacity=2, refill_per_second=D("1"), clock=clock)

    assert bucket.take() and bucket.take()
    assert not bucket.take()

    clock.advance_by(timedelta(seconds=1))
    assert bucket.take()


def test_token_bucket_uses_the_injected_clock_not_the_wall_clock() -> None:
    # SPEC §4.1. Also means a test can exhaust and refill without sleeping.
    clock = SimulationClock(NOW)
    bucket = TokenBucket(capacity=1, refill_per_second=D("1"), clock=clock)
    bucket.take()
    assert not bucket.take()
    clock.advance_by(timedelta(hours=1))
    assert bucket.available == D("1")


def test_token_bucket_validates_its_configuration() -> None:
    clock = SimulationClock(NOW)
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, refill_per_second=D("1"), clock=clock)
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, refill_per_second=D("0"), clock=clock)


def test_cache_key_is_stable_and_discriminating() -> None:
    a = cache_key("gemini", "sys", "user", ResearchView)
    assert a == cache_key("gemini", "sys", "user", ResearchView)
    assert a != cache_key("gemini", "sys", "other", ResearchView)
    assert a != cache_key("groq", "sys", "user", ResearchView)
    # A different schema must not be served a cached response from another shape.
    assert a != cache_key("gemini", "sys", "user", FundamentalView)


def test_caching_provider_serves_the_second_call_from_disk(tmp_path: Path) -> None:
    inner = ScriptedProvider(response=bullish())
    provider = CachingProvider(inner=inner, cache=LLMResponseCache(root=tmp_path))

    first = provider.complete("sys", "user", ResearchView)
    second = provider.complete("sys", "user", ResearchView)

    assert first == second
    assert inner.calls == 1
    assert (provider.hits, provider.misses) == (1, 1)


def test_failover_moves_on_when_a_provider_fails() -> None:
    broken = ScriptedProvider(error=LLMError("quota exhausted"))
    working = ScriptedProvider(response=bullish())
    provider = FailoverProvider(providers=[broken, working])

    assert provider.complete("sys", "user", ResearchView).stance is Stance.BULLISH
    assert broken.calls == 1 and working.calls == 1


def test_failover_reports_when_everything_fails() -> None:
    provider = FailoverProvider(
        providers=[ScriptedProvider(error=LLMError("a")), ScriptedProvider(error=LLMError("b"))]
    )
    with pytest.raises(LLMError, match="every provider failed"):
        provider.complete("sys", "user", ResearchView)


def test_failover_needs_at_least_one_provider() -> None:
    with pytest.raises(ValueError):
        FailoverProvider(providers=[])


def test_resilient_provider_falls_back_to_neutral() -> None:
    # SPEC §2.1(3): two reparse attempts, then NEUTRAL and continue. A cycle
    # must not die because one model returned malformed JSON.
    inner = ScriptedProvider(error=InvalidResponseError("not JSON"))
    provider = ResilientProvider(inner=inner, attempts=2)

    view = provider.complete("sys", "user", ResearchView)
    assert view.stance is Stance.NEUTRAL
    assert inner.calls == 2
    assert provider.fallbacks == 1


def test_resilient_provider_returns_a_good_response_untouched() -> None:
    provider = ResilientProvider(inner=ScriptedProvider(response=bullish()))
    assert provider.complete("sys", "user", ResearchView).stance is Stance.BULLISH
    assert provider.fallbacks == 0


def test_resilient_provider_answers_neutral_when_out_of_budget() -> None:
    clock = SimulationClock(NOW)
    bucket = TokenBucket(capacity=1, refill_per_second=D("1"), clock=clock)
    inner = ScriptedProvider(response=bullish())
    provider = ResilientProvider(inner=inner, bucket=bucket)

    assert provider.complete("s", "u", ResearchView).stance is Stance.BULLISH
    # Budget exhausted: answer neutrally rather than burn quota or block.
    assert provider.complete("s", "u", ResearchView).stance is Stance.NEUTRAL
    assert inner.calls == 1


def test_the_whole_stack_composes(tmp_path: Path) -> None:
    stack = ResilientProvider(
        inner=CachingProvider(
            inner=FailoverProvider(
                providers=[ScriptedProvider(error=LLMError("down")), NullProvider()]
            ),
            cache=LLMResponseCache(root=tmp_path),
        )
    )
    assert stack.complete("s", "u", ResearchView).stance is Stance.NEUTRAL


# ===========================================================================
# Research agent (SPEC §5.1)
# ===========================================================================


def headlines() -> list[Headline]:
    return [
        Headline(NOW - timedelta(days=2), "Bookings accelerated", "https://news.test/a"),
        Headline(NOW - timedelta(days=40), "Old news", "https://news.test/old"),
        Headline(NOW + timedelta(days=2), "Tomorrow's news", "https://news.test/future"),
    ]


def test_headlines_from_the_future_are_never_visible() -> None:
    # A research agent that reads tomorrow's news is a lookahead bug in a
    # costume, and it would be invisible in the backtest results.
    visible = visible_headlines(headlines(), NOW)
    assert [h.url for h in visible] == ["https://news.test/a"]


def test_headlines_outside_the_lookback_are_dropped() -> None:
    visible = visible_headlines(headlines(), NOW, lookback=timedelta(days=90))
    assert "https://news.test/old" in {h.url for h in visible}
    assert "https://news.test/future" not in {h.url for h in visible}


def test_research_view_without_a_citation_is_discarded_to_neutral() -> None:
    # CFA Standard V(A): a recommendation with no reasonable basis is not a
    # recommendation. Discarded, not merely logged.
    audit = AuditLog()
    agent = ResearchAgent(provider=ScriptedProvider(response=bullish(citations=[])), audit=audit)

    view = agent.run("AAA", "A company", headlines(), NOW)

    assert view.stance is Stance.NEUTRAL
    assert audit.by_code("MISSING_CITATION")
    assert audit.events[0].standard is Standard.V_A_DILIGENCE


def test_research_view_citing_an_unsupplied_source_is_discarded() -> None:
    audit = AuditLog()
    fabricated = bullish(
        citations=[
            {"url": "https://invented.test/x", "published": "2024-06-01", "quote": "..."}
        ]
    )
    agent = ResearchAgent(provider=ScriptedProvider(response=fabricated), audit=audit)

    view = agent.run("AAA", "A company", headlines(), NOW)

    assert view.stance is Stance.NEUTRAL
    assert audit.by_code("FABRICATED_CITATION")


def test_a_properly_cited_view_survives() -> None:
    agent = ResearchAgent(provider=ScriptedProvider(response=bullish()))
    view = agent.run("AAA", "A company", headlines(), NOW)
    assert view.stance is Stance.BULLISH
    assert view.conviction == 4


def test_no_headlines_means_no_view() -> None:
    agent = ResearchAgent(provider=ScriptedProvider(response=bullish()))
    view = agent.run("AAA", "A company", [], NOW)
    assert view.stance is Stance.NEUTRAL


def test_research_agent_works_with_the_null_provider() -> None:
    # SPEC §2.1(4): the system stays functional with the LLM disabled.
    agent = ResearchAgent(provider=NullProvider())
    assert agent.run("AAA", "A company", headlines(), NOW).stance is Stance.NEUTRAL


# ===========================================================================
# Fundamental agent (SPEC §5.2)
# ===========================================================================


def fundamentals() -> Fundamentals:
    return Fundamentals(
        symbol="AAA",
        as_of=NOW,
        period_end=datetime(2023, 12, 31, tzinfo=UTC),
        revenue=D("1000"),
        cost_of_goods_sold=D("600"),
        gross_profit=D("400"),
        operating_income=D("200"),
        interest_expense=D("50"),
        pretax_income=D("150"),
        net_income=D("105"),
        cash_flow_operations=D("180"),
        total_assets=D("2000"),
        total_equity=D("800"),
        total_liabilities=D("1200"),
        current_assets=D("500"),
        current_liabilities=D("250"),
        inventory=D("200"),
        receivables=D("125"),
        cash=D("100"),
        long_term_debt=D("600"),
    )


def test_ratio_table_is_computed_in_python_not_by_the_model() -> None:
    table = ratio_table(fundamentals())
    # Cross-checked against the §6.4 golden values.
    assert table["net_margin"] == D("0.105")
    assert table["return_on_equity"] == D("0.13125")
    assert table["accruals_ratio"] == D("-0.0375")
    assert table["interest_coverage"] == D("4")


def test_ratio_table_omits_what_it_cannot_compute() -> None:
    sparse = Fundamentals(symbol="AAA", as_of=NOW, period_end=None, revenue=D("1000"))
    table = ratio_table(sparse)
    assert "return_on_equity" not in table
    assert "current_ratio" not in table


def test_hallucinated_figure_is_detected() -> None:
    table = ratio_table(fundamentals())
    invented = hallucinated_figures("ROE of 0.9999 is exceptional", table)
    assert "0.9999" in invented


def test_figures_actually_in_the_table_are_not_flagged() -> None:
    table = ratio_table(fundamentals())
    shown = render_table(table)
    assert "0.1050" in shown
    assert hallucinated_figures("net margin of 0.1050 is solid", table) == []


def test_fundamental_agent_logs_a_hallucinated_figure() -> None:
    audit = AuditLog()
    response = FundamentalView(
        ticker="AAA",
        stance=Stance.BULLISH,
        conviction=3,
        rationale="Revenue grew 47.3 percent, which is excellent.",
        figures_cited=["net_margin"],
    )
    agent = FundamentalAgent(provider=ScriptedProvider(response=response), audit=audit)
    agent.run(fundamentals(), NOW)

    events = audit.by_code("HALLUCINATED_FIGURE")
    assert events
    assert events[0].standard is Standard.I_C_MISREPRESENTATION


def test_fundamental_agent_returns_neutral_on_an_empty_table() -> None:
    empty = Fundamentals(symbol="AAA", as_of=NOW, period_end=None)
    agent = FundamentalAgent(provider=ScriptedProvider(response=None))
    assert agent.run(empty, NOW).stance is Stance.NEUTRAL


# ===========================================================================
# Macro agent (SPEC §5.3) — phase by rule, never by the model
# ===========================================================================


def signals(**overrides: object) -> MacroSignals:
    base: dict[str, object] = {
        "as_of": NOW,
        "term_spread": D("1.2"),
        "unemployment": D("3.8"),
        "unemployment_change": D("-0.6"),
        "inflation_yoy": D("0.02"),
        "fed_funds": D("5.25"),
        "fed_funds_change": D("0.5"),
    }
    base.update(overrides)
    return MacroSignals(**base)  # type: ignore[arg-type]


def test_inverted_curve_with_rising_unemployment_is_a_contraction() -> None:
    phase = classify_phase(signals(term_spread=D("-0.5"), unemployment_change=D("0.8")))
    assert phase is CyclePhase.CONTRACTION


def test_inverted_curve_with_tight_policy_is_a_peak() -> None:
    phase = classify_phase(
        signals(term_spread=D("-0.3"), unemployment_change=D("0.0"), fed_funds_change=D("0.75"))
    )
    assert phase is CyclePhase.PEAK


def test_falling_unemployment_with_a_normal_curve_is_expansion() -> None:
    assert classify_phase(signals()) is CyclePhase.EXPANSION


def test_disagreeing_signals_produce_neutral_not_a_guess() -> None:
    # A phase the data does not support is worse than no phase.
    phase = classify_phase(
        MacroSignals(as_of=NOW, term_spread=None, unemployment_change=None)
    )
    assert phase is CyclePhase.NEUTRAL


def test_the_phase_is_classified_before_the_model_is_called() -> None:
    # SPEC §5.3: the model writes narrative and cannot change the phase.
    agent = MacroAgent(provider=NullProvider())
    view = agent.run(signals(term_spread=D("-0.5"), unemployment_change=D("0.8")))
    assert view.phase is CyclePhase.CONTRACTION
    assert view.narrative.stance is Stance.NEUTRAL


def test_macro_signals_are_read_point_in_time() -> None:
    series = PointInTimeSeries(
        [
            Vintage(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), D("1.0")),
            Vintage(datetime(2024, 5, 1, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC), D("-0.5")),
        ]
    )
    # The July release must be invisible in June.
    read = read_signals(NOW, term_spread=series)
    assert read.term_spread == D("1.0")


def test_missing_signals_stay_missing() -> None:
    read = read_signals(NOW)
    assert read.term_spread is None
    assert "not yet published" in read.render()


# ===========================================================================
# Aggregator (SPEC §5.4) — deterministic, no LLM
# ===========================================================================


def test_the_shipped_view_mapping_loads() -> None:
    mapping = load_mapping()
    assert mapping.tilt(Stance.BULLISH, 5) == D("0.0200")
    assert mapping.tilt(Stance.BEARISH, 5) == D("-0.0200")
    assert mapping.minimum_citations == 1


def test_a_neutral_view_never_tilts_the_portfolio() -> None:
    # "No opinion" must not quietly have an opinion.
    mapping = load_mapping()
    for conviction in range(1, 6):
        assert mapping.tilt(Stance.NEUTRAL, conviction) == D("0")


def test_mapping_rejects_a_non_zero_neutral() -> None:
    document = {
        "tilts": {
            "BULLISH": {i: "0.01" for i in range(1, 6)},
            "NEUTRAL": {i: "0.005" for i in range(1, 6)},
            "BEARISH": {i: "-0.01" for i in range(1, 6)},
        },
        "agent_weights": {"research": "1.0"},
    }
    with pytest.raises(MappingError, match="must be zero"):
        mapping_from_document(document)


def test_mapping_rejects_weights_that_do_not_sum_to_one() -> None:
    document = {
        "tilts": {
            "BULLISH": {i: "0.01" for i in range(1, 6)},
            "NEUTRAL": {i: "0" for i in range(1, 6)},
            "BEARISH": {i: "-0.01" for i in range(1, 6)},
        },
        "agent_weights": {"research": "0.5", "macro": "0.2"},
    }
    with pytest.raises(MappingError, match="sum to 1"):
        mapping_from_document(document)


def test_aggregation_is_a_weighted_table_lookup() -> None:
    mapping = load_mapping()
    opinions = [
        AgentOpinion("fundamental", "AAA", Stance.BULLISH, 5),  # 0.50 x 0.0200 = 0.0100
        AgentOpinion("research", "AAA", Stance.BULLISH, 3),  # 0.30 x 0.0100 = 0.0030
        AgentOpinion("macro", "AAA", Stance.BEARISH, 2),  # 0.20 x -0.0050 = -0.0010
    ]
    views = aggregate(opinions, mapping)
    assert views["AAA"].tilt == D("0.0120")


def test_aggregation_is_clamped() -> None:
    mapping = load_mapping()
    opinions = [
        AgentOpinion(agent, "AAA", Stance.BULLISH, 5)
        for agent in ("fundamental", "research", "macro")
    ]
    assert aggregate(opinions, mapping)["AAA"].tilt <= mapping.max_absolute_tilt


def test_aggregation_records_its_provenance() -> None:
    # "Why is this name overweight" must be answerable with a row, not a vibe.
    mapping = load_mapping()
    views = aggregate([AgentOpinion("research", "AAA", Stance.BULLISH, 2)], mapping)
    assert views["AAA"].contributions["research"] == D("0.30") * D("0.0050")
    assert views["AAA"].opinions[0].stance is Stance.BULLISH


def test_aggregation_is_deterministic() -> None:
    mapping = load_mapping()
    opinions = [AgentOpinion("research", "AAA", Stance.BULLISH, 3)]
    assert aggregate(opinions, mapping) == aggregate(opinions, mapping)


def test_tilts_reduce_to_the_optimizer_input() -> None:
    mapping = load_mapping()
    views = aggregate([AgentOpinion("macro", "AAA", Stance.BEARISH, 4)], mapping)
    tilts = tilts_for_optimizer(views)
    assert tilts == {"AAA": D("0.20") * D("-0.0150")}


def test_an_unknown_agent_is_reported() -> None:
    mapping = load_mapping()
    with pytest.raises(MappingError, match="no weight configured"):
        aggregate([AgentOpinion("astrologer", "AAA", Stance.BULLISH, 3)], mapping)


# ===========================================================================
# Narrator (SPEC §5.5) — a formatter, not a decision maker
# ===========================================================================


def test_narrator_altering_a_number_is_flagged() -> None:
    audit = AuditLog()
    response = DecisionNarrative(
        headline="Portfolio rebalanced",
        summary="AAA now sits at 0.9999 of the book.",
        facts=[],
        opinions=[],
    )
    narrator = Narrator(provider=ScriptedProvider(response=response), audit=audit)
    narrator.run({"AAA": D("0.0700")}, {"sharpe": "1.01"}, NOW)

    events = audit.by_code("NARRATOR_ALTERED_FIGURE")
    assert events
    assert events[0].standard is Standard.I_C_MISREPRESENTATION


def test_narrator_echoing_verbatim_is_not_flagged() -> None:
    context = render_context({"AAA": D("0.0700")}, {"sharpe": "1.01"})
    narrative = DecisionNarrative(
        headline="Rebalanced",
        summary="AAA is 0.0700 of the book; Sharpe is 1.01.",
        facts=["AAA weight 0.0700"],
        opinions=["The book looks defensively positioned."],
    )
    assert verify_numbers_echoed(narrative, context) == []


def test_narrator_separates_fact_from_opinion() -> None:
    # CFA Standard V(B), enforced by the schema rather than by instruction.
    narrator = Narrator(provider=NullProvider())
    narrative = narrator.run({"AAA": D("0.07")}, {"sharpe": "1.01"}, NOW)
    assert isinstance(narrative.facts, list)
    assert isinstance(narrative.opinions, list)


# ===========================================================================
# Audit log (SPEC §6.9)
# ===========================================================================


def test_audit_log_counts_and_filters() -> None:
    log = AuditLog()
    log.record(
        AuditEvent(NOW, "research", "MISSING_CITATION", Standard.V_A_DILIGENCE, "no source")
    )
    log.record(
        AuditEvent(NOW, "research", "MISSING_CITATION", Standard.V_A_DILIGENCE, "no source")
    )
    log.record(
        AuditEvent(NOW, "risk", "MAX_SECTOR_WEIGHT", Standard.III_C_SUITABILITY, "capped")
    )

    assert log.counts() == {"MISSING_CITATION": 2, "MAX_SECTOR_WEIGHT": 1}
    assert len(log.by_standard(Standard.V_A_DILIGENCE)) == 2
    assert len(log) == 3


def test_audit_log_persists_as_json_lines(tmp_path: Path) -> None:
    log = AuditLog()
    log.record(AuditEvent(NOW, "risk", "NO_LEVERAGE", Standard.III_A_LOYALTY, "scaled"))
    path = tmp_path / "audit" / "log.jsonl"
    log.write(path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "III(A)" in lines[0]


# ===========================================================================
# Provider selection (SPEC §8: one env var)
# ===========================================================================


def test_the_default_provider_needs_no_api_key() -> None:
    # A fresh checkout with no keys must run the whole pipeline.
    from src.llm import get_provider

    assert isinstance(get_provider(), NullProvider)


def test_provider_selection_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.llm import get_provider

    monkeypatch.setenv("LLM_PROVIDER", "null")
    assert isinstance(get_provider(), NullProvider)


def test_unknown_provider_is_reported() -> None:
    from src.llm import get_provider

    with pytest.raises(LLMError, match="unknown LLM provider"):
        get_provider("oracle")


def test_live_providers_refuse_to_run_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.llm.gemini import GeminiProvider
    from src.llm.groq import GroqProvider

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMError, match="no Gemini API key"):
        GeminiProvider().complete("s", "u", ResearchView)
    with pytest.raises(LLMError, match="no Groq API key"):
        GroqProvider().complete("s", "u", ResearchView)


def test_live_provider_schemas_still_pass_the_numeric_guard() -> None:
    # The §2.1 guard runs in complete(), so it fires before any network call.
    from src.llm.gemini import GeminiProvider

    class Numeric(BaseModel):
        target: float

    with pytest.raises(Exception, match="numeric"):
        GeminiProvider(api_key="x").complete("s", "u", Numeric)
