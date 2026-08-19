# Design

A paper-trading portfolio manager. Language models contribute qualitative judgment;
every number is computed in Python; a rules engine enforces an Investment Policy
Statement held in configuration.

## Two invariants

Everything else defers to these, and both are enforced by CI rather than by convention.

### The model proposes, deterministic code disposes

A model may emit a categorical view (`BULLISH` / `NEUTRAL` / `BEARISH`), a 1–5 conviction
ordinal, and prose with citations. It may not emit a number.

`validate_llm_schema` runs inside `LLMProvider.complete`, so a numeric field in any
response schema raises before a request is made. The single permitted integer is
`Conviction`, which carries an explicit marker in its `Annotated` metadata; an `IntEnum`
is rejected because it would smuggle a number past a categorical-looking field.

A view becomes a number in exactly one place — a table lookup in
[`config/view_mapping.yaml`](config/view_mapping.yaml). That makes the conversion
auditable: the answer to "why is this name overweight" is a row in a config file and a
logged stance, both identical across runs.

`NullProvider` answers everything `NEUTRAL`, and the full cycle is tested against it. With
the model disabled the pipeline produces a byte-identical result digest to running with no
agents at all.

### The decision layer knows nothing about execution

`src/decision/` emits target weights and receives an `ExecutionReport`. Orders, venues,
slicing, fills, brokers and share counts live in `src/execution/`.

Weights are the actual decision. Share counts are a function of weights, prices and
portfolio value at execution time, and a real execution algorithm recomputes them as it
works an order — so sizing belongs below the boundary. The contract is defined in
[`proto/execution.proto`](proto/execution.proto) so an engine written in another language
implements the same service without renegotiation.

## Enforced constraints

| Rule | Enforced by |
|---|---|
| `src.decision` must not import `src.execution` | import-linter contract + AST test |
| `src.cfa` must not import `llm` / `execution` / `data` / `api` | import-linter contract |
| `src.risk` must not import `llm` / `execution` / `api` | import-linter contract |
| `src/` must not import `tests/` | AST test |
| Order-placement surface only under `src/execution/` | token scan |
| No `datetime.now()` outside `src/time/` | source scan |
| No credential in a tracked file | pattern scan |

## Data correctness

**Point-in-time or nothing.** Every accessor takes an `as_of` instant and returns only what
was public then. Visibility keys on publication date, never fiscal period end: FY2023
figures describe a period ending 31 December but are not filed until February, so a
period-indexed store hands a January query six weeks of hindsight. Revisions do not leak
backwards — `CPIAUCSL` carries roughly four per period.

**`Decimal` for money, prices and weights.** Matrix inversion, regression and root-finding
have no exact-decimal implementation, so they run in float64 and convert back through
`src/cfa/_numeric.py`, the only place the two representations meet.

**Adjusted and unadjusted prices are separate fields.** Adjusted for returns, unadjusted
for share arithmetic, never mixed.

**Models return `None` when their assumptions break.** Gordon Growth at `g ≥ r` does not
mean infinite value; it means the model does not apply. A `None` that becomes "no view" is
correct, a fabricated number becomes a portfolio weight.

## Determinism

Identical inputs produce identical output: no wall clock, no unseeded randomness, no
reliance on dict ordering. Mandate ids are content hashes rather than UUIDs, which also
makes them idempotency keys. `result_digest()` checks the property directly.

Credentials are excluded from cache keys, so a replay under a different key — or none —
hits the same entries.

## Known limitations

**Survivorship bias.** The universe is a fixed, current list, not point-in-time index
membership. Absolute returns are an upper bound. Point-in-time constituent history is
commercial data; the limitation is stated rather than hidden.

**One regime.** The recorded window covers a sustained bull market. Risk-adjusted figures
from it say as much about the period as about the strategy.

**Market efficiency.** Semi-strong form is assumed. The objective is risk-adjusted
construction under explicit constraints, not alpha generation.

Deliberately out of scope: technical analysis, capital structure theory, inventory and
lease accounting, international economics.
