# CLAUDE.md

Instructions for Claude Code working in this repository.

**Read [SPEC.md](SPEC.md) before making non-trivial changes.** It is the design document
and it is authoritative. This file is the operating manual: what must not break, how to run
things, and which mistakes have already been made here so they are not made again.

---

## What this is

A paper-trading portfolio manager. LLM agents contribute *qualitative judgment only*; all
quantitative work is deterministic Python; a rules engine enforces an Investment Policy
Statement encoded as configuration.

**Educational simulation. Not investment advice. There is no live broker endpoint and there
must never be one** (SPEC §1). `AlpacaPaperBroker` refuses to construct against the live
host, and a test asserts it.

---

## Commands

```bash
make install        # venv (.venv, Python 3.12) + dependencies
make check          # THE gate: tests + mypy strict + import contracts + coverage
make test           # tests only
make typecheck      # mypy --strict src tests
make lint-imports   # the three architectural contracts
make coverage       # coverage report
make results        # regenerate RESULTS.md's numbers
make proto          # regenerate protobuf stubs (gitignored — never commit them)
make serve          # read-only dashboard API on :8000
make web            # build the React dashboard
```

**`make check` must pass before any commit.** It takes ~2.5 min, most of it the
10,000-case property test.

Front end: `cd web && npm install && npm run dev` (proxies `/api` to `:8000`).

---

## The two invariants

Everything defers to these. Both are enforced mechanically — if you find yourself wanting to
work around either, stop and raise it instead.

### 1. The LLM proposes, deterministic code disposes (SPEC §2.1)

An LLM may emit `BULLISH`/`NEUTRAL`/`BEARISH`, a 1–5 conviction ordinal, and prose with
citations. **Any numeric field in an LLM response schema is a bug.**

- `validate_llm_schema()` runs inside `LLMProvider.complete()`, so it fires *before* a
  request is made. No provider can opt out.
- The single permitted integer is `src.llm.base.Conviction`, which carries an explicit
  marker object in its `Annotated` metadata.
- A view becomes a number in exactly one place: table lookup in
  `config/view_mapping.yaml`. Never in a model, never in an agent.
- `NullProvider` answers everything `NEUTRAL`. `tests/test_pipeline_null_llm.py` proves the
  full cycle runs with the LLM disabled — and that it produces a **byte-identical result
  digest** to running with no agents at all.

### 2. The decision layer knows nothing about execution (SPEC §2.2)

`src/decision/` emits target weights and receives an `ExecutionReport`. That is the whole
surface. Orders, venues, slicing, fills, brokers, share counts all live in `src/execution/`.

Sizing belongs below the boundary because share counts are a function of weights, prices,
and portfolio value *at execution time*.

---

## Enforced architecture

Three import-linter contracts in `pyproject.toml`, plus AST checks in
`tests/test_layer_isolation.py`:

| Rule | Enforced by |
|---|---|
| `src.decision` must not import `src.execution` | contract + AST test |
| `src.cfa` must not import `llm`/`execution`/`data`/`api` | contract |
| `src.risk` must not import `llm`/`execution`/`api` | contract |
| `src/` must not import `tests/` | AST test |
| Order-placement surface only under `src/execution/` | token scan |
| No `datetime.now()` outside `src/time/` | `tests/test_no_wall_clock.py` |

Note the fifth rule bans the *order-placement surface* (`api.alpaca.markets`,
`submit_order`, `/v2/orders`), **not** the string "alpaca". Alpaca is both a market-data
vendor and a broker; SPEC §10 puts its market data in `src/data/`. The hosts differ, so the
check is precise.

---

## Non-negotiables

**`Decimal` for money, prices, weights. Never `float`.**
Matrix inversion, regression and root-finding have no exact-decimal implementation, so they
run in float64 and convert back — through `src/cfa/_numeric.py`, the *only* place the two
representations meet. `BarPayload` rejects float construction outright.

**Timestamps are tz-aware UTC instants, always.**
Never dates. A date index is what makes intraday impossible to add later. Every module takes
a `Clock`; `src/time/clock.py` holds the one permitted `datetime.now()`.

**Point-in-time or nothing (SPEC §4.4).**
Every data accessor takes `as_of` and returns only what was public at that instant.
Visibility is keyed on *publication* date, never fiscal period end. Revisions do not leak
backwards. See `src/data/pit.py`; `tests/test_point_in_time.py` is the most important test
file in the repo.

**Models return `None` when their assumptions break.**
Gordon Growth at `g >= r` does not mean infinite value, it means the model does not apply.
`None` propagating into "no view" is correct; a fabricated number propagating into a
portfolio weight is not.

**Determinism (SPEC §9).**
Identical inputs produce byte-identical output. No wall clock, no unseeded randomness, no
dict-ordering dependence. `result_digest()` checks it.

---

## Testing conventions

- **Golden tests are hand-computed.** `tests/test_cfa_golden.py` asserts against literals
  derived by hand, with the arithmetic in the comment. Never assert against the
  implementation's own output — that proves only self-consistency.
- **The property test re-derives its constraints.** `tests/test_risk_properties.py` does not
  call the engine's verifier. A property test sharing the implementation's checking code
  proves nothing.
- **Guards must be proven to fail.** When adding a structural guard, break the invariant
  deliberately, watch the test fail, then revert. Several guards here were silently vacuous
  until checked.
- **Everything runs offline.** No network, no API keys, in the whole suite. Clients take
  their fetcher by injection.

---

## Traps that already bit us

Each of these passed a green suite before being caught. Watch for the pattern.

1. **Silent no-trade.** A backtest ran green over a flat equity curve because an off-by-one
   made every cycle fall through to no-trade, then because an infeasible position cap made
   the optimizer raise into a swallowed exception. `CycleRecord.note` now records *why* a
   cycle did nothing. A run that executes nothing is a broken backtest, not a passing one.
2. **Lossy cache round-trip.** `json.dumps(default=str)` wrote a `Decimal` as a quoted
   string, so replay returned different types than the live path. `src/data/cache.py` now
   hand-encodes `Decimal` as a bare JSON number.
3. **Production importing tests.** `src/api/` imported the data generator from
   `tests/synthetic.py`. Everything passed; the Lambda would have died at cold start. Now
   `src/data/synthetic.py`, with a guard.
4. **Bare dates parse as midnight.** `datetime.fromisoformat("2024-01-03")` succeeds on
   3.11+, so a date-only fallback never fired and daily bars landed 17 hours early.
5. **Capped frontier asked for infeasible targets.** With a per-name cap you cannot reach
   the best asset's own return. SLSQP reports that as "iteration limit", not infeasibility.
6. **A hand-computed golden was wrong**, not the code — a truncated Taylor series. When a
   golden test fails, check your arithmetic before changing the implementation.

---

## Layout

```
src/time/       Clock — the only place datetime.now() may appear
src/cfa/        CFA Level I core. Pure functions, zero I/O.
src/data/       Events, point-in-time store, EDGAR/FRED/Alpaca, cache, synthetic source
src/risk/       IPS engine (pure) + codes + ips loader
src/decision/   Optimizer, mandate. MUST NOT import src/execution.
src/execution/  Everything below the boundary. Fill models, sizing, brokers.
src/agents/     LLM agents, aggregator, pipeline
src/llm/        Provider ABC, null/gemini/groq, cache, schema guard
src/backtest/   Event loop, walk-forward, metrics
src/audit/      Audit log tagged with CFA Standards
src/api/        Read-only FastAPI + Lambda handlers + state store
config/         ips.yaml, universe.yaml, view_mapping.yaml
proto/          execution.proto — the boundary contract
infra/          AWS CDK (dependencies deliberately separate)
web/            React + TypeScript dashboard
```

Three modules sit outside SPEC §9's layout, deliberately: `src/risk/ips.py` (so the engine
stays I/O-free), `src/api/store.py` (Lambda state), `src/agents/pipeline.py` and
`src/agents/schemas.py`.

---

## Conventions

- `src` is a real package — import as `src.time.clock`. Putting `src/` on `sys.path` would
  let `src/time/` shadow the stdlib `time` module and break numpy.
- Every `src/cfa/` function names its CFA topic area in its docstring.
- Comments explain *why*, not *what*. Prefer explaining a non-obvious trade-off over
  narrating the code.
- No new dependencies without asking (SPEC §12). AWS-only packages live in
  `infra/requirements.txt`, outside the project's dependencies.
- `grpc_client.py` and `QueuePositionFillModel` are **stubs on purpose**. The C++ engine is
  a separate project. Do not implement them here.

---

## Status

M0–M10 complete. 574 tests, `mypy --strict` clean, 3 contracts kept, ~96% coverage on
`src/cfa` and `src/risk`.

**The numbers in RESULTS.md are from synthetic data** (`src/data/synthetic.py`) because no
API keys are configured. They demonstrate the pipeline is complete, deterministic and
internally consistent. They do **not** demonstrate that the strategy works, and RESULTS.md
says so at the top. Do not quote them as if they were market results.

Outstanding, both needing the repository owner:
1. Real data — see `.env.example`. The vendor-specific parsing in `edgar.py`/`fred.py`/
   `sources.py` has only ever run against stub payloads, so expect breakage there first.
2. CI has never been observed running (private repo).
