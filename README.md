# agentic-portfolio-manager

Multi-agent AI portfolio manager that researches markets and trades a paper account under a
deterministic risk engine. Python, FastAPI, React, AWS.

> **Educational paper-trading simulation. Not investment advice.**
> There is no live broker endpoint and there never will be one.

**Status: Milestone 0 (foundations) complete.** See [SPEC.md](SPEC.md) for the full design and the
M0–M10 plan.

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

Known limitations that affect result interpretation (survivorship bias in the backtest universe,
optimistic default fill model) will be documented here and in the dashboard footer as the
milestones that introduce them land.

---

## Quick start

Requires Python 3.12 or 3.13.

```bash
make install      # venv + dependencies
make proto        # generate protobuf stubs (gitignored — regenerate, never commit)
make check        # tests + mypy + import contracts
make coverage     # coverage report
```

SPEC §11 requires ≥90% coverage on `src/cfa/` and `src/risk/`. Both are empty until M1/M3, so
`make coverage-gate` currently reports 0% and fails by design; it joins `make check` once those
packages have code. CI reports coverage on every run without gating.

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
