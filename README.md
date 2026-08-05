# agentic-portfolio-manager

Multi-agent AI portfolio manager that researches markets and trades a paper account under a
deterministic risk engine. Python, FastAPI, React, AWS.

> **Educational paper-trading simulation. Not investment advice.**
> There is no live broker endpoint and there never will be one.

**Status: complete — M0 through M10.** See [SPEC.md](SPEC.md) for the design and
[RESULTS.md](RESULTS.md) for backtest output.

| | |
|---|---|
| Tests | **569** passing |
| Coverage | **96%** on `src/cfa/` and `src/risk/` (SPEC §11 requires 90%) |
| Type checking | `mypy --strict`, clean across 79 files |
| Architecture | 3 import-linter contracts, enforced in CI |

> ### ⚠️ Known limitation: survivorship bias
>
> The backtest universe ([config/universe.yaml](config/universe.yaml)) is a **fixed, current**
> list of instruments, not point-in-time index membership. Names that were delisted, acquired, or
> went to zero never enter the sample, so the strategy is never charged for having held them.
> **Absolute returns are overstated — treat them as an upper bound.**
>
> Point-in-time constituent history (CRSP, Compustat) is commercial data with no free tier. The
> honest options were to buy it or to state the limitation and refuse to present the backtest as
> unbiased; this project does the second. Relative figures against the SPY/AGG benchmark are less
> distorted, since the benchmark carries the same bias in the same direction. Constraint counts,
> turnover, and implementation shortfall are unaffected.

---

## The two invariants

Everything else in the design defers to these.

**1. The LLM proposes, deterministic code disposes** (SPEC §2.1). An LLM may emit a categorical
view (`BULLISH`/`NEUTRAL`/`BEARISH`), a 1–5 conviction ordinal, and a rationale with citations.
Every number — valuation, statistics, optimization, risk metrics, position weights — comes from
Python. This is enforced, not merely intended: `validate_llm_schema` runs inside
`LLMProvider.complete`, so a numeric field in any response schema raises before a request is
made. `NullProvider` returns `NEUTRAL` for everything, and the pipeline is tested against it —
the system stays fully functional with the LLM switched off.

**2. The decision layer knows nothing about execution** (SPEC §2.2). The portfolio manager decides
*what to hold* and emits target weights, not orders. Sizing, venues, slicing, and brokers live
below the boundary defined in [`proto/execution.proto`](proto/execution.proto). A separate C++
execution engine will implement that same service later; when it does, this repo changes one
config value.

## Framing

- **Market efficiency.** Semi-strong form is assumed. The objective is risk-adjusted portfolio
  construction under explicit constraints, not alpha generation.
- **Headline return metric is TWR**, because GIPS requires time-weighted returns: they isolate the
  strategy from the timing of external cash flows. MWR is reported alongside.
- **Deliberately out of scope:** technical analysis (it conflicts with the semi-strong framing),
  capital structure theory, inventory and lease accounting, international economics.
- **LLM privacy.** Free provider tiers generally train on inputs. Everything sent to a provider
  here is public market data, which is fine — but it is worth stating plainly.

Two limitations that affect how the results should be read: **survivorship bias** in the backtest
universe (boxed above, and in the dashboard footer), and the fact that `InstantFillModel` is
optimistic by construction — which is why every result is reported under both fill models.

---

## Quick start

Requires Python 3.12 or 3.13.

```bash
make install      # venv + dependencies
make proto        # generate protobuf stubs (gitignored — regenerate, never commit)
make check        # tests + mypy + import contracts + coverage gate
make results      # regenerate RESULTS.md's numbers
make serve        # read-only dashboard API on :8000
cd web && npm install && npm run dev   # the dashboard
```

**No API keys are required for any of this.** The LLM defaults to `NullProvider`, the data
layer replays from cache or synthetic data, and the executor defaults to the simulator. Keys
(`GEMINI_API_KEY`, `FRED_API_KEY`, `ALPACA_API_KEY_ID`) only widen what the system can reach.

`make check` gates on tests, `mypy --strict`, the import contracts, and ≥90% coverage of
`src/cfa/` and `src/risk/` (SPEC §11 — currently **96%**).

## The CFA core (M1)

`src/cfa/` is the deterministic half of §2.1 — every number the system acts on originates here.
Pure functions, zero I/O, no LLM, no network. Each one names its CFA topic area in its docstring
and is checked against a **hand-computed** golden value in
[tests/test_cfa_golden.py](tests/test_cfa_golden.py); nothing is asserted against the
implementation's own output.

| Module | SPEC | Topic area |
|---|---|---|
| [returns.py](src/cfa/returns.py) | §6.1 | Quantitative Methods — TWR/MWR, dispersion, OLS beta |
| [portfolio.py](src/cfa/portfolio.py) | §6.2 | Portfolio Management — frontier, CAPM, Sharpe/Treynor/M², shrinkage |
| [ratios.py](src/cfa/ratios.py) | §6.4 | Financial Statement Analysis — DuPont, accruals |
| [valuation.py](src/cfa/valuation.py) | §6.5 | Equity Investments — DDM, FCFE, multiples |
| [fixed_income.py](src/cfa/fixed_income.py) | §6.6 | Fixed Income — duration, convexity, yield conventions |
| [derivatives.py](src/cfa/derivatives.py) | §6.7 | Derivatives — parity, forwards, payoffs |
| [alternatives.py](src/cfa/alternatives.py) | §6.8 | Alternative Investments — fees, smoothing |

Four decisions worth knowing about:

- **Beta by regression, not `Cov/Var`.** The shortcut gives a point estimate and nothing else.
  OLS also yields R², the standard errors, and the t-statistic on the intercept — without which
  "the strategy has positive alpha" is unfalsifiable.
- **Put-call parity is checked two ways.** Strict equality `C + PV(X) = P + S₀` holds only for
  European options and is applied only to those. US listed equity options are American, so early
  exercise breaks the identity; those get **bounds**, and only a breach outside them is flagged.
  A strict check would fire constantly on legitimate quotes.
- **`Decimal` at every public boundary.** Matrix inversion, regression, and root-finding have no
  exact-decimal implementation, so they run in float64 and convert back — through
  [_numeric.py](src/cfa/_numeric.py), the single place the two representations meet.
- **Models return `None` rather than a number when their assumptions break.** Gordon Growth at
  `g ≥ r` does not mean infinite value, it means the model does not apply. `None` propagating
  into "no view" is correct; a fabricated number propagating into a portfolio weight is not.

## The data layer (M2)

Everything that touches the network goes through [src/data/](src/data/), and every accessor takes
an `as_of` instant.

| Module | Role |
|---|---|
| [pit.py](src/data/pit.py) | Point-in-time store — the visibility rule, generic over value type |
| [edgar.py](src/data/edgar.py) | SEC fundamentals, indexed by **filing date** |
| [fred.py](src/data/fred.py) | Macro series, **vintage-aware** so revisions don't leak backwards |
| [sources.py](src/data/sources.py) | Daily bars behind `MarketDataSource` |
| [cache.py](src/data/cache.py) | The network boundary: caching, offline replay |

**The whole suite runs with no network and no API keys.** Clients take their fetcher by
injection, so the same code serves a live request, a cached replay, or a test stub. A cache miss
in offline mode raises rather than returning nothing — a missing input must fail loudly, not
become a gap the optimizer interpolates over.

Three details that matter more than they look:

- **Only publication date governs visibility.** A datum carries `period_end` (what it describes)
  and `published` (when it became public). Indexing fundamentals by fiscal period end is the
  classic backtest error: FY2023 figures describe a period ending 31 December but aren't filed
  until February.
- **Revisions don't leak backwards.** Q1 GDP released in April at 2.1% and revised in May to 1.6%
  must read 2.1% for any `as_of` in late April. That's what the world believed at the time.
- **Filings get a publication lag.** EDGAR reports `filed` as a bare date. Treating a filing as
  public at 00:00 UTC would make it visible before the market opened, so filings become visible at
  the *end* of the filing day. If that's wrong, it's wrong in the safe direction.

[tests/test_point_in_time.py](tests/test_point_in_time.py) pins all of this, including the case
SPEC §4.4 names by hand: a Q4 filing published in February is invisible to an `as_of` in January.

## The milestones

| | Ships | Notes |
|---|---|---|
| M0 | Clock, event model, LLM boundary, execution contract | The three retrofit-expensive choices |
| M1 | [`src/cfa/`](src/cfa/) — CFA Level I core | Hand-computed golden tests |
| M2 | [`src/data/`](src/data/) — point-in-time accessors | Lookahead impossible by construction |
| M3 | [`src/risk/`](src/risk/) — IPS engine | 14 constraints, 10,000-case property test |
| M4 | [`src/decision/`](src/decision/) — optimizer, mandate | Shrinkage + caps + CAPM inputs |
| M5 | [`src/execution/`](src/execution/) — the boundary | Both fill models, shortfall |
| M6 | [`src/backtest/`](src/backtest/) — walk-forward | [RESULTS.md](RESULTS.md) |
| M7 | [`src/agents/`](src/agents/) — LLM agents | Views become numbers by table lookup |
| M8 | Naive executor vs paper broker | Content-hash idempotency |
| M9 | [`src/api/`](src/api/) + [`web/`](web/) — dashboard | Vetoed trades first |
| M10 | [`infra/`](infra/) — AWS CDK | Lambda, EventBridge, DynamoDB, CloudFront |

### Where the qualitative layer plugs in

[`src/agents/pipeline.py`](src/agents/pipeline.py) is the only seam between the LLM half of the
system and the quantitative half. Agents produce categorical views, the aggregator turns them
into numeric tilts by table lookup, and the tilts adjust the CAPM baseline the optimizer starts
from. Nothing else crosses.

The default is `NoViews` — **every number in [RESULTS.md](RESULTS.md) was produced with the LLM
contributing nothing**, which is worth stating plainly. With `NullProvider` the agent pipeline and
no pipeline at all produce a byte-identical result digest, and
[a test asserts it](tests/test_pipeline_null_llm.py): if they differed, the LLM would be
influencing the portfolio while switched off.

## What Milestone 0 ships

| Piece | Location | Why it is first |
|---|---|---|
| Clock abstraction | [`src/time/clock.py`](src/time/clock.py) | One code path serves both a backtest and a live cycle |
| Event model | [`src/data/events.py`](src/data/events.py) | Timestamp-native, so intraday works later without a rewrite |
| LLM provider ABC + `NullProvider` | [`src/llm/`](src/llm/) | The system must run with the LLM disabled |
| Schema guard | [`src/llm/schema_guard.py`](src/llm/schema_guard.py) | Makes "no LLM-generated numbers" a check, not a convention |
| Execution contract | [`proto/execution.proto`](proto/execution.proto) | Language-neutral, so the C++ engine needs no renegotiation |

Three choices here are the ones that would be expensive to retrofit: time is a *clock*, not
`datetime.now()`; data arrives as *events at instants*, not rows indexed by date; and money is
`Decimal`, never `float`.

## Structural tests

These run in CI and fail the build. Each has been verified to fail when its invariant is violated.

| Test | Enforces |
|---|---|
| [`test_no_wall_clock.py`](tests/test_no_wall_clock.py) | No `datetime.now()` / `date.today()` outside `src/time/` (SPEC §4.1) |
| [`test_layer_isolation.py`](tests/test_layer_isolation.py) | `src/decision/` never imports `src/execution/`; no broker names above the boundary (SPEC §2.2) |
| [`test_pipeline_null_llm.py`](tests/test_pipeline_null_llm.py) | The pipeline runs with the LLM disabled (SPEC §2.1) |
| [`test_llm_schema_guard.py`](tests/test_llm_schema_guard.py) | No numeric field in any LLM response schema (SPEC §2.1) |

`import-linter` enforces the layering contracts independently of the test suite.

## Repo layout

Deviation from SPEC §9, deliberate: `src` is a real package, so modules are imported as
`src.time.clock`. If `src/` were placed on `sys.path` instead, `src/time/` would shadow the
standard library's `time` module and break numpy, pandas, and anything else importing it. The
directory layout is otherwise exactly as specified.
