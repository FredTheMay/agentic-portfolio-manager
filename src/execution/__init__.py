"""Execution layer: sizing, orders, venues, brokers, fill models.

The only package permitted to name a broker or an order type. Swapping
executors is one config value.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping

from src.execution.base import (
    Account,
    Capabilities,
    ExecutionError,
    ExecutionProvider,
    ExecutionReport,
    ExecutionUpdate,
    Fill,
    MarketSnapshot,
    Order,
    PositionSnapshot,
    Rejection,
    RejectionCode,
    Side,
    implementation_shortfall_bps,
    size_orders,
)
from src.execution.fill_models import (
    FillModel,
    InstantFillModel,
    QueuePositionFillModel,
    SpreadCrossFillModel,
)
from src.execution.simulated import SimulatedExecutor

#: Environment variable selecting the executor. The one config value.
EXECUTOR_ENV = "EXECUTOR"
DEFAULT_EXECUTOR = "simulated"


def _naive_executor() -> ExecutionProvider:
    # Imported lazily so the paper-broker client is not a dependency of a
    # backtest that will never place an order.
    from src.execution.naive import AlpacaPaperBroker, NaiveExecutor

    return NaiveExecutor(broker=AlpacaPaperBroker())


def _grpc_executor() -> ExecutionProvider:
    # Imported lazily: the stub raises on use, and importing it eagerly would
    # make the C++ engine feel like a dependency of this package.
    from src.execution.grpc_client import GrpcExecutor

    return GrpcExecutor()


EXECUTORS: Mapping[str, Callable[[], ExecutionProvider]] = {
    "simulated": lambda: SimulatedExecutor(fill_model=InstantFillModel()),
    "simulated_spread": lambda: SimulatedExecutor(fill_model=SpreadCrossFillModel()),
    "naive": _naive_executor,
    "grpc": _grpc_executor,
}


def get_executor(name: str | None = None) -> ExecutionProvider:
    """Build the configured executor. Selection is by name, nothing else."""
    key = (name or os.environ.get(EXECUTOR_ENV) or DEFAULT_EXECUTOR).lower()
    factory = EXECUTORS.get(key)
    if factory is None:
        raise ExecutionError(
            f"unknown executor {key!r}; available: {', '.join(sorted(EXECUTORS))}"
        )
    return factory()


__all__ = [
    "Account",
    "Capabilities",
    "EXECUTORS",
    "ExecutionError",
    "ExecutionProvider",
    "ExecutionReport",
    "ExecutionUpdate",
    "Fill",
    "FillModel",
    "InstantFillModel",
    "MarketSnapshot",
    "Order",
    "PositionSnapshot",
    "QueuePositionFillModel",
    "Rejection",
    "RejectionCode",
    "Side",
    "SimulatedExecutor",
    "SpreadCrossFillModel",
    "get_executor",
    "implementation_shortfall_bps",
    "size_orders",
]
