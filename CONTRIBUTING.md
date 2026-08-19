# Contributing

## Setup

```bash
make install      # virtualenv and dependencies
make check        # tests, mypy --strict, import contracts, coverage gate
```

`make check` must pass before a commit. Most of its runtime is a 10,000-case property test
on the risk engine.

## Conventions

- `src` is a real package; import as `src.time.clock`. Putting `src/` on `sys.path` would
  let `src/time/` shadow the standard library's `time` module.
- `Decimal` for money, prices and weights. Floats are rejected at construction.
- Timestamps are tz-aware UTC instants, never dates. Every module takes a `Clock`.
- Comments explain why, not what. Prefer stating a non-obvious trade-off over narrating
  the code.
- Every function in `src/cfa/` names its CFA topic area in its docstring.

## Testing

- Golden tests assert against values computed by hand, with the arithmetic in the comment.
  Never assert against the implementation's own output — that proves only self-consistency.
- The risk property test re-derives its constraints rather than calling the engine's
  verifier, for the same reason.
- When adding a structural guard, break the invariant deliberately and watch the test fail
  before trusting it.
- The suite runs with no network and no credentials. Clients take their fetcher by
  injection.

## Data

`scripts/backfill.py` is the only code that fetches. Everything else replays from
`data/cache` offline, so a backtest never depends on the day it was run.

Credentials belong in `.env`, which is gitignored. `.env.example` is committed and must
never contain a value.
