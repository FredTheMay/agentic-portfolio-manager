# Backtest results (SPEC §11)

> ## ⚠️ These numbers are from **synthetic data**, not the market
>
> No API keys were configured, so the data layer has nothing recorded from EDGAR, FRED, or Alpaca.
> The run below uses [tests/synthetic.py](tests/synthetic.py): a seeded geometric random walk with
> a single common market factor, so every symbol has the beta it advertises.
>
> **What this demonstrates:** the full pipeline runs end to end — event loop → shrunk covariance →
> constrained optimizer → risk engine → mandate → execution → reconciliation → attribution — and it
> is deterministic and internally consistent.
>
> **What it does not demonstrate:** that the strategy works. Synthetic prices have no earnings, no
> regimes, no crashes, and no fat tails. Reproduce with real data before quoting any of it. The
> survivorship caveat in the [README](README.md#️-known-limitation-survivorship-bias) applies on
> top of that.

Reproduce with `make results`. Treynor 0.1879, tracking error 8.33%.

---

## Run parameters

| | |
|---|---|
| Walk-forward window | 2022-01-03 → 2024-01-03 (522 trading periods) |
| Universe | 12 instruments + SPY benchmark |
| Rebalance cadence | every 21 sessions, subject to the 5% corridor |
| Estimation window | 100 sessions, Ledoit-Wolf shrunk |
| Risk-free rate | 4.00% (bond-equivalent) |
| Initial capital | 100,000.00 |

## Results under both fill models

SPEC §4.3 requires reporting both. The gap is the strategy's execution-cost sensitivity; quoting
only the optimistic figure is the classic amateur tell.

| Metric | `InstantFillModel` | `SpreadCrossFillModel` |
|---|---:|---:|
| **Annualized TWR** | **16.11%** | **16.07%** |
| Benchmark TWR | 17.09% | 17.09% |
| MWR | 16.73% | 16.69% |
| Annualized volatility | 11.95% | 11.95% |
| Max drawdown | 11.65% | 11.66% |
| Portfolio beta | 0.64 | 0.64 |
| R² vs benchmark | 0.74 | 0.74 |
| Sharpe | 1.01 | 1.01 |
| Jensen's α | 3.70% | 3.66% |
| **α t-statistic** | **0.77** | **0.77** |
| **α significant at 5%?** | **No** | **No** |
| Information ratio | −0.12 | −0.12 |
| Cycles executed | 19 | 19 |
| Total commission | 0.00 | 35.12 |
| Mean implementation shortfall | 0.00 bps | 1.00 bps |
| Result digest | `faff5b3932b917a3…` | `8e2ba7ef95578b76…` |

**Execution cost drag: 4 bps annualized.**

## Reading this honestly

Three things the table says that a résumé bullet would be tempted to leave out.

**The strategy underperformed its benchmark.** 16.11% against 17.09%, and the information ratio is
**negative** (−0.12). On this data the constrained, risk-managed portfolio did worse than simply
holding the benchmark — which is the expected outcome under the semi-strong efficiency assumption
the project states up front. The system is doing risk-adjusted construction under constraints, not
generating alpha.

**The alpha is not statistically significant.** Jensen's α is +3.70%, which looks good in
isolation. Its t-statistic is **0.77**, far below the ~1.96 needed at 5%. The point estimate cannot
be distinguished from zero. This is exactly why SPEC §6.1 requires beta by regression rather than
by `Cov/Var`: the shortcut gives the α and hides the fact that it means nothing.

**Beta 0.64 explains most of the difference.** With R² of 0.74, roughly three-quarters of the
portfolio's variance is systematic. It ran at about two-thirds of market risk — the IPS cash buffer
and position caps binding — so it should be expected to lag a rising benchmark. It did.

## Structural acceptance criteria (SPEC §11)

- [x] Full pipeline runs with `NullProvider` (no LLM)
- [x] Full pipeline runs without the C++ engine
- [x] `src/decision/` imports nothing from `src/execution/`
- [x] No `datetime.now()` outside `src/time/`
- [x] Point-in-time test passes: future filings invisible at earlier `as_of`
- [x] Risk property test passes 10,000 cases
- [x] Two identical backtest runs produce identical output hashes
- [x] ≥90% coverage on `src/cfa/` and `src/risk/`
