"""Regenerate RESULTS.md's numbers (SPEC §11). Run with `make results`."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from src.backtest.engine import BacktestConfig
from src.backtest.metrics import summarize
from src.backtest.walkforward import run_under_both_fill_models
from src.data.live import resolve_setup
from src.risk.ips import load_policy
from src.time.clock import UTC

D = Decimal
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL")


def main() -> None:
    setup = resolve_setup()
    config = BacktestConfig(
        start=setup.start,
        end=setup.end,
        initial_cash=D("100000.00"),
        symbols=setup.symbols,
        benchmark_symbol=setup.benchmark,
        rebalance_every=21,
        estimation_window=100,
        market_return=setup.market_return,
        risk_free_rate=setup.risk_free_rate,
        periods_per_year=252,
    )
    result = run_under_both_fill_models(
        config, setup.source, load_policy(), setup.sectors, setup.betas
    )

    banner = (
        f"{setup.data_source.upper()}"
        if setup.is_real
        else f"{setup.data_source.upper()}\nSee RESULTS.md for what this does and does not show."
    )
    print(f"{banner}\n")
    print(
        f"window {config.start.date()} -> {config.end.date()} | "
        f"universe {len(config.symbols)} | Rf {config.risk_free_rate}\n"
    )
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
