"""Regenerate RESULTS.md's numbers (SPEC §11). Run with `make results`."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from src.backtest.engine import BacktestConfig
from src.backtest.metrics import summarize
from src.backtest.walkforward import run_under_both_fill_models
from src.data.live import load_bars, read_manifest, universe_inputs
from src.data.synthetic import BETAS, SECTORS, make_source
from src.data.universe import load_universe
from src.risk.ips import load_policy
from src.time.clock import UTC

D = Decimal
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL")


def real_setup() -> tuple[object, ...] | None:
    """Recorded market data, or None when nothing has been backfilled.

    Never falls back silently: main() prints which source it used, because a
    table of synthetic numbers presented as market results is the single most
    misleading thing this project could produce.
    """
    manifest = read_manifest()
    if manifest is None:
        return None
    try:
        universe = load_universe()
        start, end = manifest.start, manifest.end
        source = load_bars(manifest.symbols, start, end, offline=True)
        if not source.events:
            return None
        as_of = source.events[-1].timestamp
        sectors, betas, risk_free, market = universe_inputs(universe, source, as_of)
        symbols = tuple(s for s in universe.tradable() if s in betas)
        if len(symbols) < 10:
            # The IPS caps any name at 10%, so a fully invested portfolio needs
            # at least ten holdings with usable betas.
            return None
        return source, symbols, sectors, betas, risk_free, market, start, end
    except Exception:  # noqa: BLE001 - no recorded data is a normal state
        return None


def main() -> None:
    real = real_setup()
    if real is not None:
        source, symbols, sectors, betas, risk_free, market, start, end = real  # type: ignore[misc]
        config = BacktestConfig(
            start=start,  # type: ignore[arg-type]
            end=end,  # type: ignore[arg-type]
            initial_cash=D("100000.00"),
            symbols=symbols,  # type: ignore[arg-type]
            benchmark_symbol=load_universe().benchmark_equity,
            rebalance_every=21,
            estimation_window=100,
            market_return=market,  # type: ignore[arg-type]
            risk_free_rate=risk_free,  # type: ignore[arg-type]
            periods_per_year=252,
        )
        banner = "REAL MARKET DATA (recorded via `make backfill`)"
        result = run_under_both_fill_models(
            config, source, load_policy(), sectors, betas  # type: ignore[arg-type]
        )
    else:
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
        banner = (
            "SYNTHETIC DATA — see RESULTS.md for what this does and does not show.\n"
            "Run `make backfill` with credentials in .env for real numbers."
        )
        result = run_under_both_fill_models(
            config, make_source(days=760), load_policy(), SECTORS, BETAS
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
