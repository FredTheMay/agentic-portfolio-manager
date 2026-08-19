# Backtest results (SPEC §11)

**Real market data.** Prices from Alpaca (IEX feed, split- and dividend-adjusted), fundamentals
from SEC EDGAR indexed by filing date, risk-free rate from FRED `DGS3MO` converted from its
bank-discount quote to a bond-equivalent yield.

Reproduce:

```bash
cp .env.example .env      # add credentials
make backfill             # records ~3y into data/cache — the only step that fetches
make results              # replays offline; deterministic
```

> ### ⚠️ Read these three caveats before quoting any number below
>
> **1. Survivorship bias.** The universe ([config/universe.yaml](config/universe.yaml)) is a
> fixed, current list, not point-in-time index membership. Names that were delisted or acquired
> never enter the sample. Absolute returns are an **upper bound**. Point-in-time constituents are
> commercial data with no free tier.
>
> **2. One regime.** The window is July 2023 – August 2026, a sustained equity bull market. A
> Sharpe of 1.33 says more about that period than about the strategy. There is no 2008, no 2020
> crash, and no rate shock in this sample.
>
> **3. It underperformed.** 19.58% against a 20.33% benchmark, with a **negative information
> ratio**. That is the expected outcome under the semi-strong efficiency assumption this project
> states up front, and it is reported rather than tuned away.

---

## Run parameters

| | |
|---|---|
| Window | 2023-07-20 → 2026-08-18 (773 trading periods) |
| Universe | 27 instruments + SPY benchmark |
| Rebalance | every 21 sessions, subject to the 5% corridor |
| Estimation | 100 sessions, Ledoit-Wolf shrunk |
| Risk-free | 3.96% (DGS3MO, discount → bond-equivalent) |
| Betas | **estimated by regression** against SPY excess returns |
| Initial capital | 100,000.00 |

## Results under both fill models

SPEC §4.3 requires both. The gap is the strategy's execution-cost sensitivity; quoting only the
optimistic figure is the classic amateur tell.

| Metric | `InstantFillModel` | `SpreadCrossFillModel` |
|---|---:|---:|
| **Annualized TWR** | **19.62%** | **19.58%** |
| Benchmark TWR | 20.33% | 20.33% |
| MWR | 19.48% | 19.44% |
| Annualized volatility | 11.76% | 11.75% |
| Max drawdown | 17.04% | 17.05% |
| Portfolio beta | 0.70 | 0.70 |
| R² vs benchmark | 0.78 | 0.78 |
| Sharpe | 1.33 | 1.33 |
| Treynor | 0.2238 | 0.2232 |
| Jensen's α | 4.21% | 4.17% |
| **α t-statistic** | **1.18** | **1.17** |
| **α significant at 5%?** | **No** | **No** |
| Information ratio | −0.10 | −0.11 |
| Tracking error | 7.05% | 7.05% |
| Cycles executed | 17 | 17 |
| **Cycles vetoed** | **15** | **15** |
| Total commission | 0.00 | 29.86 |
| Mean implementation shortfall | 0.00 bps | 1.00 bps |
| Result digest | `bc36acf263427fa9…` | `7c58361c7812cda2…` |

**Execution cost drag: 4 bps annualized.**

## What the risk engine actually did

32 cycles, of which **15 were vetoed outright** and most of the rest modified before execution.

| Veto | Count |
|---|---:|
| `REBALANCE_CORRIDOR` | 11 |
| `MAX_VOLATILITY` | 4 |

| Repair applied | Count |
|---|---:|
| `MIN_CASH_BUFFER` | 30 |
| `MAX_TURNOVER` | 16 |
| `MAX_VOLATILITY` | 4 |
| `MAX_SECTOR_WEIGHT` | 2 |

The corridor is doing the job it was written for: eleven proposed rebalances moved positions by
less than five percentage points and were refused rather than paying spread and commission to
correct noise.

## Estimated betas — a sanity check

Betas are not declared anywhere; they are regressed from realized excess returns (SPEC §6.2
[CORRECTED], by OLS rather than `Cov/Var`, so R² and standard errors come with them).

| Symbol | β | Sanity |
|---|---:|---|
| AGG | 0.07 | Aggregate bond ETF — near-zero equity beta ✓ |
| BRK.B | 0.44 | Defensive conglomerate ✓ |
| AAPL | 1.09 | Large-cap, roughly market ✓ |
| AMZN | 1.43 | High-beta discretionary ✓ |
| AVGO | 2.06 | Semiconductor, high beta ✓ |

None of these were supplied. That they land where a practitioner would expect is evidence the
regression path is wired correctly.

## Reading this honestly

**The alpha is not statistically significant.** Jensen's α is +4.17%, which looks good alone. Its
t-statistic is **1.17**, well below the ~1.96 needed at 5%. The point estimate cannot be
distinguished from zero. This is exactly why SPEC §6.1 requires beta by regression rather than the
`Cov/Var` shortcut: the shortcut hands you the α and hides the fact that it means nothing.

**Beta 0.70 explains the shortfall.** With R² of 0.78, roughly four-fifths of the portfolio's
variance is systematic. It ran at about two-thirds of market risk — the IPS cash buffer (repaired
30 times) and the volatility ceiling binding — so lagging a rising benchmark is the arithmetic
consequence, not a surprise.

**Risk-adjusted, it is competitive.** Sharpe 1.33 against 11.75% volatility versus a benchmark
that took materially more risk for its 20.33%. Whether that trade is worth making is the
investor's call, which is what an IPS is for.

## Structural acceptance criteria (SPEC §11)

- [x] Full pipeline runs with `NullProvider` (no LLM)
- [x] Full pipeline runs without the C++ engine
- [x] `src/decision/` imports nothing from `src/execution/`
- [x] No `datetime.now()` outside `src/time/`
- [x] Point-in-time test passes: future filings invisible at earlier `as_of`
- [x] Risk property test passes 10,000 cases
- [x] Two identical backtest runs produce identical output hashes
- [x] ≥90% coverage on `src/cfa/` and `src/risk/`
