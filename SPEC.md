# Agentic AI Portfolio Manager — Build Specification v2

> Save as `SPEC.md` in the repo root. Point Claude Code at it with:
> *"Read SPEC.md. Implement Milestone 0 only. Stop and report before Milestone 1."*
> One milestone per session. Do not let it build the whole system in one pass.

**Changes from v1:** execution is now a replaceable layer behind a language-neutral contract;
the time and data model is event-driven and timestamp-native so intraday works later without a
rewrite; point-in-time data handling is mandatory; several CFA formulas are corrected.

---

## 1. Mission and scope

Build a system that constructs and manages a **paper-traded** portfolio of US-listed equities and
ETFs. LLM agents contribute qualitative judgment. All quantitative work is deterministic Python.
A rules engine enforces an Investment Policy Statement encoded as configuration.

**This system is complete and useful on its own.** A separate C++ execution engine will be built
afterward as a distinct project. This specification defines the boundary that engine will plug
into, and ships a working default implementation behind it. Nothing in this repo may depend on
the C++ engine existing.

### Non-goals
- Real-money trading. No live broker endpoint, ever.
- Investment advice. The UI carries a persistent banner: *"Educational paper-trading simulation.
  Not investment advice."*
- Building execution algorithms here. This repo emits intent; the future engine implements
  execution. Ship a naive executor and stop.
- LLM-generated arithmetic (§2.1).

---

## 2. Architectural invariants

Two rules. Every design decision defers to them.

### 2.1 The LLM proposes, deterministic code disposes

| Layer | Owner | May produce numbers? |
|---|---|---|
| Research synthesis, news reading, ratio interpretation | LLM | **No** |
| Valuation, statistics, optimization, risk metrics | Python | Yes — sole source |
| Constraint checking, mandate generation | Python | Yes — sole source |
| Narrating decisions already made | LLM | **No** |

Enforcement:
1. All LLM output is Pydantic-validated structured data. Never free text parsed by regex.
2. An LLM may emit a categorical view (`BULLISH`/`NEUTRAL`/`BEARISH`), an integer conviction 1–5,
   and a rationale with citations. **Any numeric field in an LLM response schema is a bug.**
3. On invalid output after 2 reparse attempts, fall back to `NEUTRAL` and continue.
4. `NullProvider` returns `NEUTRAL` for everything. A test runs the full pipeline with it. **The
   system must remain fully functional with the LLM disabled.**

### 2.2 The decision layer knows nothing about execution

The portfolio manager decides **what to hold**. It has no knowledge of orders, venues, slicing,
fills, or brokers. It emits a `RebalanceMandate` and receives an `ExecutionReport`. That is the
entire surface.

Enforcement:
- Nothing under `src/decision/` may import from `src/execution/`. Enforce with an import-linter
  rule in CI. A violation fails the build.
- Nothing outside `src/execution/` may reference Alpaca, order types, share counts, or venues.
- Swapping executors must require changing exactly one config value.

---

## 3. The execution boundary

### 3.1 What crosses it

The portfolio manager emits **target weights, not orders.**

Rationale: weights are the actual decision. Share counts are a function of weights, prices, and
portfolio value at execution time — and a real execution algorithm needs to recompute them as it
works the order over minutes. Emitting fixed share counts would force the C++ engine to either
ignore them or reverse-engineer the intent. Weights also make the contract price-independent,
which means the mandate stays valid while it's being worked.

Sizing therefore belongs to the execution layer. The decision layer's job ends at intent plus
the constraints that intent must respect.

### 3.2 Contract — `proto/execution.proto`

Define this in protobuf now, even though v1's executor is in-process Python. Protobuf is
language-neutral, so the C++ engine implements the same service later with no renegotiation.

```protobuf
syntax = "proto3";
package portfolio.execution.v1;
import "google/protobuf/timestamp.proto";

service ExecutionEngine {
  rpc Execute(RebalanceMandate) returns (stream ExecutionUpdate);
  rpc GetCapabilities(CapabilitiesRequest) returns (Capabilities);
}

// All monetary and ratio values are decimal strings. Never float.
message RebalanceMandate {
  string mandate_id = 1;                          // idempotency key
  google.protobuf.Timestamp decision_time = 2;    // for shortfall measurement
  string portfolio_value = 3;
  repeated TargetWeight targets = 4;
  ExecutionConstraints constraints = 5;
  Urgency urgency = 6;
}

message TargetWeight {
  string symbol = 1;
  string target_weight = 2;      // "0.0700"
  string current_weight = 3;     // "0.0400"
}

message ExecutionConstraints {
  string min_trade_notional = 1;         // "100.00"
  string max_turnover = 2;               // "0.20" of portfolio value
  string max_participation_rate = 3;     // "0.10" of ADV — ignored by naive executor
  bool allow_partial = 4;
  google.protobuf.Timestamp deadline = 5;
}

enum Urgency { PATIENT = 0; NORMAL = 1; AGGRESSIVE = 2; }

message ExecutionUpdate {
  oneof update {
    Fill fill = 1;
    Rejection rejection = 2;
    ExecutionReport completion = 3;
  }
}

message Fill {
  string symbol = 1; string quantity = 2; string price = 3;
  google.protobuf.Timestamp timestamp = 4; string venue = 5;
}

message Rejection { string symbol = 1; string reason_code = 2; string detail = 3; }

message ExecutionReport {
  string mandate_id = 1;
  repeated PositionSnapshot final_positions = 2;
  string realized_turnover = 3;
  string implementation_shortfall_bps = 4;   // vs decision_time price
  string total_commission = 5;
  repeated Rejection rejections = 6;
}

message Capabilities {
  bool supports_intraday = 1;
  bool supports_participation_limits = 2;
  bool supports_streaming_updates = 3;
  string engine_name = 4;
  string engine_version = 5;
}
```

`GetCapabilities` lets the decision layer degrade gracefully: if the executor reports
`supports_participation_limits = false`, log that the constraint is advisory rather than
silently assuming it was honored.

### 3.3 Implementations

- **`NaiveExecutor`** (v1, ships in this repo). Computes integer share counts from target weights
  and last price, submits market orders to Alpaca paper, streams fills back. Ignores
  participation limits and urgency. This is the default and it is sufficient.
- **`SimulatedExecutor`** (v1, backtest only). Applies the configured `FillModel` (§4.3).
- **`GrpcExecutor`** (later). Thin client pointing at the C++ engine. **Do not build now.**
  Its existence must require zero changes outside `src/execution/`.

### 3.4 Round trip

```
Optimizer → TargetPortfolio (weights)
          → Risk & IPS Engine → approved/modified weights + constraints
          → RebalanceMandate  ──────► [ EXECUTION BOUNDARY ] ──────► Executor
          ◄── ExecutionReport ───────────────────────────────────────┘
          → reconcile realized vs target weights
          → log drift, turnover, implementation shortfall
          → performance & attribution
```

**Post-trade reconciliation is mandatory.** Realized weights will not equal target weights. Log
the residual drift and carry it into the next cycle's corridor check. Systems that assume
perfect execution produce backtests that lie.

---

## 4. Time, data, and event model

These three choices are what make intraday possible later without a rewrite. Get them right in
Milestone 0 — retrofitting any of them is a multi-week rewrite.

### 4.1 Clock abstraction

```python
class Clock(Protocol):
    def now(self) -> datetime: ...          # always tz-aware UTC

class SimulationClock(Clock):  # advances by event, backtest
class WallClock(Clock):        # real time, live paper trading
```

**No module may call `datetime.now()` or `date.today()` directly.** Everything takes a `Clock`.
Add a CI grep test that fails the build on direct calls outside `src/time/`. This is the single
highest-leverage rule in the spec: it's what lets the identical code path run a 3-year backtest
and a live daily cycle.

### 4.2 Event-driven core

Do **not** write `for day in trading_days:`. Write an event loop.

```python
@dataclass(frozen=True)
class MarketEvent:
    timestamp: datetime            # tz-aware UTC, always
    symbol: str
    kind: Literal["BAR", "QUOTE", "TRADE", "CORPORATE_ACTION"]
    payload: BarPayload | QuotePayload | TradePayload | ActionPayload

class MarketDataSource(Protocol):
    def stream(self, start: datetime, end: datetime) -> Iterator[MarketEvent]: ...
```

v1 ships `DailyBarSource`. A future `StreamingQuoteSource` implements the same interface over a
websocket and the engine loop is unchanged. Timestamps are **instants**, never dates — a date
index is the thing that makes intraday impossible later.

### 4.3 Pluggable fill models (backtest)

```python
class FillModel(Protocol):
    def fill(self, order: Order, event: MarketEvent) -> list[Fill]: ...
```

- `InstantFillModel` — fills at close, in full. v1 default. Optimistic; label it as such.
- `SpreadCrossFillModel` — fills at the far side of the quoted spread plus fixed commission.
  Cheap and materially more honest. Build this in v1 too.
- `QueuePositionFillModel` — later, backed by the C++ simulator. Interface is already here.

Report backtest results under **both** fill models. The gap between them is your execution-cost
sensitivity, and quoting only the optimistic number is the classic amateur tell.

### 4.4 Point-in-time data — non-negotiable

**Lookahead bias.** Fundamentals from EDGAR must be indexed by **filing date**, never fiscal
period end. Every data accessor takes an `as_of: datetime` and returns only what was public at
that instant:

```python
def get_fundamentals(symbol: str, as_of: datetime) -> Fundamentals | None: ...
```

Write a test that asserts a Q4 filing published in February is invisible to an `as_of` of
January. A reviewer will check this within the first minute of looking at your backtest.

**Survivorship bias.** Do not backtest today's index constituents over history — you've selected
for survivors and your returns are meaningless. Source a point-in-time constituent list, or if
you cannot, state the limitation prominently in the README and in the dashboard footer.

**Corporate actions.** Splits and dividends must be handled explicitly. Use adjusted prices for
return calculation and unadjusted for share-count arithmetic; never mix them.

---

## 5. Agents

Each implements `run(context: Context) -> Output`, returns Pydantic models, calls LLMs only
through the provider abstraction (§8).

**5.1 Research Agent (LLM).** Input: ticker, headlines from the last 14 days (respecting
`as_of`), company description. Output: `ResearchView { ticker, stance, conviction: int 1-5,
rationale: str, citations: list[Citation] }`. Requires ≥1 dated citation with URL or the view is
discarded to `NEUTRAL`. Implements CFA Standard V(A), reasonable basis.

**5.2 Fundamental Analyst (deterministic + LLM).** Python computes every ratio in §6.4 from
point-in-time EDGAR data. The LLM receives the finished table and returns a categorical view. If
its rationale contains a figure absent from the input table, log `HALLUCINATED_FIGURE`.

**5.3 Macro/Regime Agent (deterministic + LLM).** FRED signals: `T10Y3M` term spread, `UNRATE`
trend, `CPIAUCSL` YoY, fed funds direction. Business-cycle phase classified by **rule**, not by
the LLM. The LLM writes narrative only.

**5.4 View Aggregator (deterministic).** Maps categorical views to numeric tilts via a fixed
table in `config/view_mapping.yaml`. Auditable config, not model judgment.

**5.5 Narrator (LLM).** Receives final decisions and metrics; formats them for the dashboard.
Must echo numbers verbatim. A formatter, not a decision maker.

---

## 6. CFA Level I concept map

Every item is a named, individually unit-tested function with a golden test against a
hand-computed value and the CFA topic area in its docstring.

> Items marked **[CORRECTED]** were wrong in v1. Do not revert them.

### 6.1 Quantitative Methods — `src/cfa/returns.py`

| Concept | Formula |
|---|---|
| Holding period return | `HPR = (P₁ − P₀ + D₁) / P₀` |
| Time-weighted return | `TWR = Π(1 + HPRᵢ) − 1` |
| Money-weighted return | IRR of the dated cash-flow series |
| Geometric mean return | `(Π(1 + Rᵢ))^(1/n) − 1` |
| Arithmetic mean return | `Σ Rᵢ / n` |
| Sample variance | `Σ(Rᵢ − R̄)² / (n − 1)` |
| Covariance / correlation | `ρᵢⱼ = Cov(i,j) / (σᵢ σⱼ)` |
| Coefficient of variation | `σ / R̄` |
| Downside deviation | σ of returns below the minimum acceptable return |
| Continuously compounded return | `R_cc = ln(1 + HPR)` |
| **[CORRECTED]** Roy's safety-first | `SFRatio = (E(Rp) − R_L) / σp`, maximize |
| **[CORRECTED]** OLS beta estimation | regress excess returns; report slope, R², SE(slope), t-stat on intercept |

**Report TWR as the headline metric and state why:** GIPS requires time-weighted returns because
they isolate the strategy from the timing of external cash flows. Report MWR alongside and
explain the divergence on the dashboard.

**Beta by regression, not by the covariance shortcut.** `Cov/Var` gives the point estimate and
nothing else. OLS gives R² — the directly interpretable systematic share of variance — plus the
standard error and the t-statistic on the intercept, which is the only honest way to say whether
Jensen's alpha is distinguishable from zero.

### 6.2 Portfolio Management — `src/cfa/portfolio.py`

`E(Rp) = Σ wᵢE(Rᵢ)` · `σp² = wᵀΣw` · two-asset variance as the unit-test check · minimum-variance
portfolio `min wᵀΣw s.t. Σw = 1` · efficient frontier · tangency portfolio `max (E(Rp) − R_f)/σp`
· CAL `E(Rp) = R_f + [(E(Rᵢ) − R_f)/σᵢ]σp` · CML · CAPM `E(Rᵢ) = R_f + βᵢ(E(R_m) − R_f)` ·
portfolio beta `βp = Σ wᵢβᵢ` · SML · Jensen's alpha · Sharpe · Treynor · M² `(Rp − R_f)(σₘ/σp) +
R_f` · information ratio · systematic/unsystematic decomposition into `βp²σₘ²` plus residual.

**[CORRECTED] Risk-free rate.** FRED `DGS3MO` is quoted on a **discount basis**. Convert to
bond-equivalent yield before using it in Sharpe, CAPM, or the CAL. Using the discount yield
directly makes every risk-adjusted metric wrong.

**[CORRECTED] Covariance estimation.** The sample covariance matrix is unstable and mean-variance
optimization is notoriously sensitive to it. Apply **Ledoit-Wolf shrinkage**. Be able to explain
the input-sensitivity problem and why weight constraints plus shrinkage are your defense —
"how do you handle MVO instability?" is a near-certain interview question.

**Validation.** Walk-forward, not a single in-sample backtest. Estimate on a rolling window,
trade the following period, roll forward. A single backtest over the full history is curve
fitting, and iterating on it until it looks good is how you fit noise.

### 6.3 IPS — `config/ips.yaml`, enforced by `src/risk/engine.py`

Return objective 8% nominal, benchmark 60/40 SPY/AGG. Risk objective: ability above average,
willingness moderate, **the lower binds** — assert and unit-test this. Max portfolio beta 1.20,
max annualized volatility 18%.

Constraints by the Level I taxonomy — Liquidity (5% minimum cash), Legal/regulatory (long only,
no leverage), Tax (penalize holding under 366 days; 30-day wash-sale window), Time horizon
(10 years, single stage), Unique circumstances (exclusion lists).

### 6.4 Financial Statement Analysis — `src/cfa/ratios.py`

DuPont 3-step and 5-step · liquidity (current, quick) · solvency (D/E, interest coverage
`EBIT/interest`) · profitability (gross/operating/net margin, ROA, ROE) · activity (inventory,
receivables, total asset turnover).

**[CORRECTED] Accruals ratio** `(Net income − CFO) / average total assets` as an earnings-quality
red flag. Used as a deterministic pre-screen alongside solvency thresholds.

### 6.5 Equity Investments — `src/cfa/valuation.py`

Gordon Growth `V₀ = D₁/(r − g)` with `r` from CAPM · sustainable growth `g = (1 − payout) × ROE`
· justified trailing P/E `[payout(1+g)]/(r − g)` · **[CORRECTED]** justified leading P/E
`payout/(r − g)` · **[CORRECTED]** `EV = market cap + total debt − cash` · relative multiples vs
sector median · guard: `g ≥ r` returns `None`.

**[CORRECTED] Non-dividend payers.** Most of a large-cap tech universe pays no dividend, so DDM
returns `None` for the majority of names. Implement an FCFE fallback and document the hierarchy:
DDM where dividends exist, FCFE otherwise, multiples as the cross-check.

Market efficiency: assume semi-strong form; frame the objective as risk-adjusted construction
under constraints, not alpha generation. State this in the README.

### 6.6 Fixed Income — `src/cfa/fixed_income.py`

YTM · current yield · Macaulay and modified duration · `%ΔP ≈ −ModDur·Δy + ½·Convexity·(Δy)²` ·
portfolio duration as weighted average · **[CORRECTED]** money-market yield conversions
(discount basis ↔ bond-equivalent yield, holding-period yield).

### 6.7 Derivatives — `src/cfa/derivatives.py`

**[CORRECTED] Put-call parity `C + PV(X) = P + S₀` holds for European options only.** US listed
equity options are American; early exercise rights break strict parity, so the v1 check would
have fired constantly on legitimate quotes. Either run the equality check only on European-style
index options (SPX), or implement American put-call parity **bounds** and flag only breaches
outside them. Document which you chose.

**[CORRECTED] Forward price with carry:** `F₀ = (S₀ − PV(dividends))(1 + r)^T`. The v1 form
`S₀(1+r)^T` ignores dividends and is wrong for any dividend-paying equity.

Covered call payoff and breakeven · protective put payoff and breakeven · moneyness · intrinsic
vs time value · payoff diagrams for the four basic positions · forwards vs futures (mark-to-market
and credit exposure).

Simulated overlay: when portfolio volatility exceeds the IPS ceiling, propose a protective-put
overlay and display the payoff diagram and cost drag. Simulated only; trade no options.

### 6.8 Alternative Investments

Hold REIT and commodity ETFs in the universe. Implement the category distinctions, the fee
convention (management plus incentive fee, high-water mark), and — directly relevant to §6.2 —
why illiquidity and smoothed pricing bias reported correlations downward and therefore corrupt a
covariance matrix that includes them.

### 6.9 Ethics — `src/audit/`

V(A) reasonable basis → citation requirement. V(B) communication → `fact` vs `opinion` fields in
the audit log. I(C) misrepresentation → `HALLUCINATED_FIGURE` check. III(A) loyalty/prudence →
IPS binding, no runtime override. III(C) suitability → the IPS check *is* a suitability test.
**[CORRECTED] GIPS** → the reason TWR is the headline metric; note composite construction and
required disclosures in the README.

Excluded deliberately, and say so in the README: technical analysis (conflicts with the
semi-strong EMH framing), capital structure theory, inventory and lease accounting,
international economics.

---

## 7. Risk & IPS Engine

Pure function. No I/O, no LLM, no randomness. Operates on **weights**, upstream of the boundary.

| Code | Rule |
|---|---|
| `MAX_POSITION_WEIGHT` | no name > 10% |
| `MAX_SECTOR_WEIGHT` | no GICS sector > 30% |
| `MIN_CASH_BUFFER` | cash ≥ 5% |
| `MAX_PORTFOLIO_BETA` | `βp ≤ 1.20` |
| `MAX_VOLATILITY` | trailing 60d annualized σ ≤ 18% |
| `SAFETY_FIRST_THRESHOLD` | Roy's SFRatio below floor → reject |
| `NO_LEVERAGE` | `Σw ≤ 1.0` |
| `NO_SHORTING` | all `wᵢ ≥ 0` |
| `REBALANCE_CORRIDOR` | trade only on drift > ±5% absolute |
| `MAX_TURNOVER` | ≤ 20% per rebalance |
| `MIN_TRADE_NOTIONAL` | passed to executor as a constraint |
| `UNIVERSE_WHITELIST` | approved liquid tickers only |
| `DRAWDOWN_CIRCUIT_BREAKER` | peak-to-trough > 15% → force minimum-variance, halt risk increases |
| `WASH_SALE_WINDOW` | block repurchase within 30 days of a loss sale |

Returns `APPROVED`, `MODIFIED` (scaled weights), or `REJECTED` + code. **Every rejection is
persisted** and surfaced in the dashboard's vetoed-trades panel — the screen to demo first.

**Property test with `hypothesis`:** for arbitrary portfolios and arbitrary proposed weights, no
approved output ever violates any constraint. 10,000 cases. Write this before the engine.

---

## 8. LLM provider layer

⚠️ **Grok** is xAI's model (paid). **Groq** is an inference provider with a free tier. Different
companies.

**Primary: Google Gemini Flash** (AI Studio) — 1,500 requests/day, no credit card, large context
suits full-filing prompts. **Fallback: Groq** — very fast but a ~100K tokens/day ceiling on the
70B model, so reserve it for short classification calls. Free-tier quotas change often; verify
before building.

```python
class LLMProvider(ABC):
    def complete(self, system: str, user: str, schema: Type[BaseModel]) -> BaseModel: ...
```

Implement `GeminiProvider`, `GroqProvider`, `NullProvider`. Selection by `LLM_PROVIDER` env var.
Include token-bucket rate limiting per published limits, exponential backoff on 429, automatic
failover, and a response cache keyed on hash(prompt + model + schema) so re-runs are free.
Swapping providers touches nothing outside `src/llm/`.

Note in the README that most free tiers train on inputs. Everything sent here is public market
data, which is fine — but say so.

---

## 9. Repo layout

```
portfolio-manager/
├── SPEC.md  README.md
├── proto/execution.proto            # the boundary contract
├── config/  ips.yaml  universe.yaml  view_mapping.yaml
├── src/
│   ├── time/          clock.py                  # ONLY place datetime.now() may appear
│   ├── cfa/           returns.py portfolio.py ratios.py valuation.py
│   │                  fixed_income.py derivatives.py
│   ├── data/          events.py sources.py edgar.py fred.py cache.py pit.py
│   ├── agents/        research.py fundamental.py macro.py aggregator.py narrator.py
│   ├── llm/           base.py gemini.py groq.py null.py cache.py
│   ├── decision/      optimizer.py mandate.py    # MUST NOT import src/execution
│   ├── risk/          engine.py codes.py
│   ├── execution/     base.py naive.py simulated.py fill_models.py
│   │                  grpc_client.py            # stub only, do not implement
│   ├── backtest/      engine.py walkforward.py metrics.py
│   ├── audit/         log.py standards.py
│   └── api/           routes.py schemas.py ws.py
├── web/                                          # React + TypeScript + Vite
├── infra/                                        # AWS CDK
└── tests/
    ├── test_cfa_golden.py           # hand-computed values
    ├── test_risk_properties.py      # hypothesis, 10k cases
    ├── test_pipeline_null_llm.py    # runs with LLM disabled
    ├── test_point_in_time.py        # lookahead bias
    ├── test_no_wall_clock.py        # grep for datetime.now()
    └── test_layer_isolation.py      # decision must not import execution
```

Strict `mypy`. Pydantic v2. **`Decimal` for all money and weights — never float.** All randomness
seeded; identical inputs produce a byte-identical trade log.

---

## 10. Milestones

Each ends with passing tests and a commit. Stop and report after each.

- **M0 — Foundations.** Repo, `Clock`, event types, `LLMProvider` ABC + `NullProvider`,
  `proto/execution.proto`, CI with the four structural tests. No business logic.
- **M1 — CFA core.** All of `src/cfa/` with golden tests. Pure functions, zero I/O.
- **M2 — Data layer.** EDGAR/FRED/Alpaca behind `MarketDataSource`, point-in-time accessors,
  caching, offline replay.
- **M3 — Risk engine.** Constraints, reason codes, hypothesis property test. No LLM yet.
- **M4 — Optimizer + mandate.** MVO with Ledoit-Wolf shrinkage, frontier, tangency portfolio,
  `RebalanceMandate` emission.
- **M5 — Execution boundary.** `ExecutionProvider` ABC, `SimulatedExecutor` with both fill
  models, reconciliation, shortfall measurement.
- **M6 — Backtest.** Walk-forward harness, full metrics, results under both fill models.
  **← This is the checkpoint that matters. A finished M6 beats a half-built M9.**
- **M7 — LLM agents.** Providers, three agents, aggregator, caching.
- **M8 — Live paper trading.** `NaiveExecutor` against Alpaca paper, idempotency, audit log.
- **M9 — API + dashboard.** FastAPI + React/TS. Screens: efficient frontier, holdings,
  **vetoed trades**, TWR/MWR/benchmark, attribution, audit trail, executor capabilities.
- **M10 — AWS deploy.** Lambda for the scheduled cycle, EventBridge cron, DynamoDB, S3 +
  CloudFront.

**Out of scope for this repo:** the C++ execution engine. It implements `proto/execution.proto`
as a separate project with its own spec. When it exists, this repo changes one config value.

---

## 11. Acceptance criteria

Structural — all must pass:
- [ ] Full pipeline runs with `NullProvider` (no LLM)
- [ ] Full pipeline runs with `NaiveExecutor` (no C++ engine)
- [ ] `src/decision/` imports nothing from `src/execution/`
- [ ] No `datetime.now()` outside `src/time/`
- [ ] Point-in-time test passes: future filings invisible at earlier `as_of`
- [ ] Risk property test passes 10,000 cases
- [ ] Two identical backtest runs produce identical output hashes
- [ ] ≥90% coverage on `src/cfa/` and `src/risk/`

Numeric — fill every blank from your own output before touching the résumé:
- Walk-forward window: ____ to ____ · universe: ____ instruments
- Annualized TWR ____% vs benchmark ____% · MWR ____%
- Sharpe ____ · Treynor ____ · Jensen's α ____ (t-stat ____) · IR ____
- Max drawdown ____% · portfolio β ____
- Results under `InstantFillModel` ____% vs `SpreadCrossFillModel` ____%
- Constraints enforced ____ · orders vetoed ____ of ____ proposed
- Cycle latency ____ s · LLM calls per cycle ____
- Tests ____ · coverage ____%

---

## 12. Instructions to Claude Code

- Read this file fully. Confirm which milestone you are on before writing code.
- Never let an LLM produce a number used downstream (§2.1). If a design pressures you toward it,
  stop and flag it rather than working around it.
- Never let the decision layer learn about orders or fills (§2.2). Same rule.
- Write the test before the implementation in `src/cfa/`, `src/risk/`, and `src/data/pit.py`.
- Cite the CFA topic area in every `src/cfa/` docstring.
- `Decimal` for money and weights. Timestamps tz-aware UTC, always.
- Do not implement `grpc_client.py`. A stub raising `NotImplementedError` is correct.
- No new dependencies without asking. Baseline: `numpy scipy pandas scikit-learn pydantic
  fastapi httpx grpcio grpcio-tools pytest hypothesis mypy import-linter pyyaml`.
- End each milestone with: what was built, what is tested, what the next milestone needs. Stop.
