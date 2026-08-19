"""Client stub for an out-of-process execution engine.

Deliberately unimplemented. The value of the stub is that it proves the seam:
adding a real client requires no change outside this package, and swapping to
it is one config value in :func:`src.execution.get_executor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from src.decision.mandate import RebalanceMandate
from src.execution.base import (
    Account,
    Capabilities,
    ExecutionProvider,
    ExecutionUpdate,
    MarketSnapshot,
)

DEFAULT_TARGET = "localhost:50051"


@dataclass(slots=True)
class GrpcExecutor(ExecutionProvider):
    """Placeholder for the gRPC client to the C++ engine."""

    target: str = DEFAULT_TARGET

    def capabilities(self) -> Capabilities:
        raise NotImplementedError(
            "The C++ execution engine is a separate project. Until it exists, "
            "use SimulatedExecutor (backtest) or NaiveExecutor (paper trading)."
        )

    def execute(
        self,
        mandate: RebalanceMandate,
        account: Account,
        market: MarketSnapshot,
    ) -> Iterator[ExecutionUpdate]:
        raise NotImplementedError(
            "The C++ execution engine is a separate project. Until it exists, "
            "use SimulatedExecutor (backtest) or NaiveExecutor (paper trading)."
        )
