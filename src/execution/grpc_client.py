"""Client for the C++ execution engine — **stub only** (SPEC §3.3, §12).

The C++ engine is a separate project with its own spec. It will implement
``proto/execution.proto`` as a gRPC service, and this module will become a thin
client pointing at it.

Do not implement it here. SPEC §12 is explicit, and the reason is the point of
the whole boundary: this repo must be complete and useful without the engine
existing. The value of the stub is that it proves the seam — adding the real
client requires zero changes outside ``src/execution/``, and swapping to it is
one config value in :func:`src.execution.get_executor`.
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
