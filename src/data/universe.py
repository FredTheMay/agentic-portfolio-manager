"""The investable universe, loaded from ``config/universe.yaml``.

The whitelist lives in configuration so changing it is a reviewable diff.

Survivorship bias is a property of this file and is carried on
:class:`Universe` so callers surface it rather than discover it: the list is
fixed and current, not point-in-time index membership, so absolute backtest
returns are an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

DEFAULT_UNIVERSE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "universe.yaml"
)


class UniverseError(ValueError):
    """Raised on a malformed or internally inconsistent universe file."""


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    sector: str
    category: str = "EQUITY"


@dataclass(frozen=True, slots=True)
class Universe:
    """Everything the system is permitted to hold."""

    instruments: tuple[Instrument, ...]
    benchmark_equity: str
    benchmark_bonds: str
    benchmark_weights: Mapping[str, Decimal]
    exclusions: Mapping[str, str] = field(default_factory=dict)
    #: True when the list is fixed-and-current rather than point-in-time.
    survivorship_biased: bool = True
    survivorship_reason: str = ""

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(i.symbol for i in self.instruments)

    @property
    def sectors(self) -> dict[str, str]:
        return {i.symbol: i.sector for i in self.instruments}

    def tradable(self) -> tuple[str, ...]:
        """Symbols to optimize over — the benchmark itself is not a holding."""
        return tuple(s for s in self.symbols if s != self.benchmark_equity)

    def fetch_list(self) -> tuple[str, ...]:
        """Every symbol needing price history, benchmark included."""
        ordered = dict.fromkeys((*self.symbols, self.benchmark_equity))
        return tuple(ordered)


def universe_from_document(document: Mapping[str, Any]) -> Universe:
    """Build a :class:`Universe` from a parsed universe document."""
    benchmark = document.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise UniverseError("universe has no `benchmark` section")

    equity = str(benchmark.get("equity", ""))
    bonds = str(benchmark.get("bonds", ""))
    if not equity or not bonds:
        raise UniverseError("benchmark needs both `equity` and `bonds`")

    weights = {
        equity: Decimal(str(benchmark.get("equity_weight", "0.60"))),
        bonds: Decimal(str(benchmark.get("bond_weight", "0.40"))),
    }
    total = sum(weights.values(), Decimal(0))
    if total != Decimal(1):
        raise UniverseError(f"benchmark weights must sum to 1, got {total}")

    instruments: list[Instrument] = []
    seen: set[str] = set()
    for key, default_category in (("equities", "EQUITY"), ("etfs", "ETF")):
        for entry in document.get(key) or ():
            if not isinstance(entry, Mapping) or "symbol" not in entry:
                raise UniverseError(f"malformed entry in `{key}`: {entry!r}")
            symbol = str(entry["symbol"])
            if symbol in seen:
                raise UniverseError(f"{symbol} is listed twice")
            seen.add(symbol)
            instruments.append(
                Instrument(
                    symbol=symbol,
                    sector=str(entry.get("sector", "UNKNOWN")),
                    category=str(entry.get("category", default_category)),
                )
            )

    if not instruments:
        raise UniverseError("universe contains no instruments")

    exclusions = {
        str(entry["symbol"]): str(entry.get("reason", ""))
        for entry in document.get("exclusions") or ()
        if isinstance(entry, Mapping) and "symbol" in entry
    }
    # An instrument that is both whitelisted and excluded is a contradiction,
    # and silently resolving it either way would hide an editing mistake.
    contradictions = sorted(seen & set(exclusions))
    if contradictions:
        raise UniverseError(f"listed and excluded at once: {contradictions}")

    survivorship = document.get("survivorship") or {}
    return Universe(
        instruments=tuple(instruments),
        benchmark_equity=equity,
        benchmark_bonds=bonds,
        benchmark_weights=weights,
        exclusions=exclusions,
        survivorship_biased=not bool(survivorship.get("point_in_time_constituents", False)),
        survivorship_reason=str(survivorship.get("reason", "")),
    )


def load_universe(path: Path | None = None) -> Universe:
    """Read and validate the universe file."""
    source = DEFAULT_UNIVERSE_PATH if path is None else path
    if not source.is_file():
        raise UniverseError(f"universe not found at {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise UniverseError(f"universe at {source} is not a YAML mapping")
    return universe_from_document(document)


def equity_symbols(universe: Universe) -> Sequence[str]:
    """Only the single-name equities — the ones EDGAR has filings for."""
    return [i.symbol for i in universe.instruments if i.category == "EQUITY"]
