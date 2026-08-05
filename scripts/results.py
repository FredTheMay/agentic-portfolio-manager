"""Regenerate RESULTS.md's numbers (SPEC §11). Run with `make results`."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from src.backtest.engine import BacktestConfig
from src.backtest.metrics import summarize
from src.backtest.walkforward import run_under_both_fill_models
from src.risk.ips import load_policy
from src.time.clock import UTC
from src.data.synthetic import BETAS, SECTORS, make_source

D = Decimal
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL")


def main() -> None:
    start = datetime(2022, 1, 3, 21, tzinfo=UTC)
    config = BacktestConfig(
        start=start,
        end=start + timedelta(days=730),
        initial_cash=D("100000.00"),
        symbols=SYMBOLS,
        benchmark_symbol="SPY",
        rebalance_every=21,
        estimation_window=100,
        market_return=D("0.09"),
        risk_free_rate=D("0.04"),
        periods_per_year=252,
    )
    result = run_under_both_fill_models(
        config, make_source(days=760), load_policy(), SECTORS, BETAS
    )

    print("SYNTHETIC DATA — see RESULTS.md for what this does and does not show.\n")
    print(f"window {config.start.date()} -> {config.end.date()} | universe {len(SYMBOLS)}\n")
    for outcome in (result.optimistic, result.realistic):
        summary = summarize(outcome.metrics)
        print(f"--- {outcome.fill_model} (digest {outcome.digest[:16]}) ---")
        for key, value in summary.items():
            print(f"  {key:28} {value}")
        print(f"  {'executed_cycles':28} {outcome.executed_cycles}")
        print(f"  {'vetoed_cycles':28} {outcome.vetoed_cycles}")
        print(f"  {'total_commission':28} {outcome.total_commission.quantize(D('0.01'))}")
        print(f"  {'mean_shortfall_bps':28} {outcome.mean_shortfall_bps.quantize(D('0.01'))}")
        print()
    print(f"execution cost drag: {result.execution_cost_drag.quantize(D('0.0001'))}")


if __name__ == "__main__":
    main()
