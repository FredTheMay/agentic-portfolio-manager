"""The Investment Policy Statement as typed configuration (SPEC §6.3).

Separated from :mod:`src.risk.engine` so the engine stays a pure function with
no I/O: reading YAML happens here, once, and the engine receives an already
validated :class:`InvestmentPolicy`.

(SPEC §9's layout lists only ``engine.py`` and ``codes.py`` under ``src/risk/``.
This third module exists to keep the "no I/O" requirement in §7 literally true;
folding the loader into the engine would break it.)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import yaml

ONE = Decimal(1)

DEFAULT_IPS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "ips.yaml"


class PolicyError(ValueError):
    """Raised on an IPS that is missing or internally inconsistent."""


class RiskLevel(enum.IntEnum):
    """Ordinal risk tolerance. Ordered so ``min()`` means "the lower binds"."""

    BELOW_AVERAGE = 1
    MODERATE = 2
    ABOVE_AVERAGE = 3


@dataclass(frozen=True, slots=True)
class SafetyFirstPolicy:
    """Roy's safety-first parameters (SPEC §6.1).

    Both are policy choices rather than derived quantities.
    """

    threshold_return: Decimal
    minimum_ratio: Decimal


@dataclass(frozen=True, slots=True)
class InvestmentPolicy:
    """Every limit the risk engine enforces, in one immutable object."""

    target_nominal_annual: Decimal
    benchmark: Mapping[str, Decimal]

    ability: RiskLevel
    willingness: RiskLevel
    max_portfolio_beta: Decimal
    max_annualized_volatility: Decimal
    volatility_lookback_days: int
    safety_first: SafetyFirstPolicy

    min_cash_buffer: Decimal
    long_only: bool
    max_gross_exposure: Decimal

    short_term_holding_days: int
    short_term_gain_penalty: Decimal
    wash_sale_window_days: int

    horizon_years: int
    stages: int

    max_position_weight: Decimal
    max_sector_weight: Decimal

    corridor_absolute: Decimal
    max_turnover: Decimal
    min_trade_notional: Decimal

    max_drawdown: Decimal

    @property
    def effective_exposure_ceiling(self) -> Decimal:
        """The tighter of the leverage ceiling and what the cash buffer permits.

        SPEC §7 states these as two independent rules — ``NO_LEVERAGE`` at
        ``sum(w) <= 1.0`` and ``MIN_CASH_BUFFER`` at ``cash >= 5%`` — and the
        second is strictly tighter. They are not in conflict; the liquidity
        floor simply binds first. Naming the combination once stops the two
        limits being applied inconsistently in different places.
        """
        return min(self.max_gross_exposure, ONE - self.min_cash_buffer)

    @property
    def binding_risk_tolerance(self) -> RiskLevel:
        """The lower of ability and willingness (SPEC §6.3).

        Ability is a fact about horizon and balance sheet; willingness is a
        psychological constraint. Taking the higher of the two would build a
        portfolio the investor abandons at the bottom, which converts a
        temporary drawdown into a permanent loss.
        """
        return min(self.ability, self.willingness)

    def __post_init__(self) -> None:
        if self.min_cash_buffer < 0 or self.min_cash_buffer >= 1:
            raise PolicyError(f"min_cash_buffer must lie in [0, 1), got {self.min_cash_buffer}")
        if self.max_position_weight <= 0:
            raise PolicyError("max_position_weight must be positive")
        if self.max_sector_weight < self.max_position_weight:
            raise PolicyError(
                f"max_sector_weight {self.max_sector_weight} is below max_position_weight "
                f"{self.max_position_weight}: no single position could ever be filled"
            )
        if self.max_gross_exposure <= 0:
            raise PolicyError("max_gross_exposure must be positive")


def _decimal(node: Mapping[str, Any], key: str, where: str) -> Decimal:
    if key not in node:
        raise PolicyError(f"{where}.{key} is missing")
    try:
        return Decimal(str(node[key]))
    except InvalidOperation as exc:
        raise PolicyError(f"{where}.{key} is not numeric: {node[key]!r}") from exc


def _int(node: Mapping[str, Any], key: str, where: str) -> int:
    if key not in node:
        raise PolicyError(f"{where}.{key} is missing")
    try:
        return int(node[key])
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{where}.{key} is not an integer: {node[key]!r}") from exc


def _section(document: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    node: Any = document
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            raise PolicyError(f"IPS section {'.'.join(path)} is missing")
        node = node[part]
    if not isinstance(node, Mapping):
        raise PolicyError(f"IPS section {'.'.join(path)} is not a mapping")
    return node


def _risk_level(node: Mapping[str, Any], key: str) -> RiskLevel:
    raw = node.get(key)
    if not isinstance(raw, str):
        raise PolicyError(f"risk_objective.{key} must be a name, got {raw!r}")
    try:
        return RiskLevel[raw.upper()]
    except KeyError as exc:
        allowed = ", ".join(level.name for level in RiskLevel)
        raise PolicyError(f"risk_objective.{key} must be one of {allowed}, got {raw!r}") from exc


def policy_from_document(document: Mapping[str, Any]) -> InvestmentPolicy:
    """Build an :class:`InvestmentPolicy` from a parsed IPS document."""
    returns = _section(document, "return_objective")
    risk = _section(document, "risk_objective")
    safety = _section(document, "risk_objective", "safety_first")
    liquidity = _section(document, "constraints", "liquidity")
    legal = _section(document, "constraints", "legal_regulatory")
    tax = _section(document, "constraints", "tax")
    horizon = _section(document, "constraints", "time_horizon")
    unique = _section(document, "constraints", "unique_circumstances")
    rebalancing = _section(document, "rebalancing")
    breaker = _section(document, "circuit_breaker")

    benchmark_node = _section(document, "return_objective", "benchmark")
    benchmark = {str(k): Decimal(str(v)) for k, v in benchmark_node.items()}
    total = sum(benchmark.values(), Decimal(0))
    if total != Decimal(1):
        raise PolicyError(f"benchmark weights must sum to 1, got {total}")

    return InvestmentPolicy(
        target_nominal_annual=_decimal(returns, "target_nominal_annual", "return_objective"),
        benchmark=benchmark,
        ability=_risk_level(risk, "ability"),
        willingness=_risk_level(risk, "willingness"),
        max_portfolio_beta=_decimal(risk, "max_portfolio_beta", "risk_objective"),
        max_annualized_volatility=_decimal(risk, "max_annualized_volatility", "risk_objective"),
        volatility_lookback_days=_int(risk, "volatility_lookback_days", "risk_objective"),
        safety_first=SafetyFirstPolicy(
            threshold_return=_decimal(safety, "threshold_return", "safety_first"),
            minimum_ratio=_decimal(safety, "minimum_ratio", "safety_first"),
        ),
        min_cash_buffer=_decimal(liquidity, "min_cash_buffer", "liquidity"),
        long_only=bool(legal.get("long_only", True)),
        max_gross_exposure=_decimal(legal, "max_gross_exposure", "legal_regulatory"),
        short_term_holding_days=_int(tax, "short_term_holding_days", "tax"),
        short_term_gain_penalty=_decimal(tax, "short_term_gain_penalty", "tax"),
        wash_sale_window_days=_int(tax, "wash_sale_window_days", "tax"),
        horizon_years=_int(horizon, "years", "time_horizon"),
        stages=_int(horizon, "stages", "time_horizon"),
        max_position_weight=_decimal(unique, "max_position_weight", "unique_circumstances"),
        max_sector_weight=_decimal(unique, "max_sector_weight", "unique_circumstances"),
        corridor_absolute=_decimal(rebalancing, "corridor_absolute", "rebalancing"),
        max_turnover=_decimal(rebalancing, "max_turnover", "rebalancing"),
        min_trade_notional=_decimal(rebalancing, "min_trade_notional", "rebalancing"),
        max_drawdown=_decimal(breaker, "max_drawdown", "circuit_breaker"),
    )


def load_policy(path: Path | None = None) -> InvestmentPolicy:
    """Read and validate the IPS from disk. The only I/O in ``src/risk``."""
    source = DEFAULT_IPS_PATH if path is None else path
    if not source.is_file():
        raise PolicyError(f"IPS not found at {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise PolicyError(f"IPS at {source} is not a YAML mapping")
    return policy_from_document(document)
