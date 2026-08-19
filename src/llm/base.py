"""LLM provider abstraction.

All output is Pydantic-validated structured data, never free text parsed by
regex. Every call passes through :func:`~src.llm.schema_guard.validate_llm_schema`,
so a numeric field in a response schema fails at the boundary rather than
contaminating a portfolio weight.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Annotated, ClassVar, TypeVar

from pydantic import BaseModel, Field

from src.llm.schema_guard import CONVICTION_MARKER, validate_llm_schema
from src.time.clock import Clock

M = TypeVar("M", bound=BaseModel)


class Stance(str, enum.Enum):
    """The categorical view an LLM agent is permitted to express.

    A ``str`` enum, not an ``IntEnum``: an integer here would be a number the
    LLM produced, and downstream code could do arithmetic on it.
    """

    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


#: The only integer an LLM may emit: an ordinal 1-5, mapped to numeric tilts by
#: ``config/view_mapping.yaml`` rather than used in arithmetic.
Conviction = Annotated[int, CONVICTION_MARKER, Field(ge=1, le=5)]


class LLMError(Exception):
    """Base class for provider failures."""


class RateLimitError(LLMError):
    """Provider returned 429, or the local token bucket refused the call."""


class InvalidResponseError(LLMError):
    """Model output failed schema validation."""


#: (3): two reparse attempts, then fall back to NEUTRAL and continue.
#: A pipeline that halts because one model returned malformed JSON is worse
#: than one that records "no view" and keeps going.
MAX_REPARSE_ATTEMPTS = 2


@dataclass
class TokenBucket:
    """Client-side rate limiter.

    Takes a :class:`~src.time.clock.Clock` rather than reading the wall clock,
    for the same reason everything else does — and usefully, it
    means a test can exhaust and refill a bucket without sleeping.
    """

    capacity: int
    refill_per_second: Decimal
    clock: Clock
    _tokens: Decimal = field(init=False)
    _last: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("token bucket capacity must be positive")
        if self.refill_per_second <= 0:
            raise ValueError("refill rate must be positive")
        self._tokens = Decimal(self.capacity)

    def _refill(self) -> None:
        now = self.clock.now()
        if self._last is not None:
            elapsed = Decimal(str((now - self._last).total_seconds()))
            if elapsed > 0:
                self._tokens = min(
                    Decimal(self.capacity), self._tokens + elapsed * self.refill_per_second
                )
        self._last = now

    def take(self, tokens: int = 1) -> bool:
        """Consume ``tokens`` if available. Returns False rather than blocking."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= Decimal(tokens)
            return True
        return False

    @property
    def available(self) -> Decimal:
        self._refill()
        return self._tokens


class LLMProvider(ABC):
    """A source of structured, schema-validated qualitative judgment.

    Subclasses implement :meth:`_complete`. :meth:`complete` is deliberately
    not overridable in spirit — it is where the guard runs.
    """

    #: Stable identifier used in cache keys and the audit log.
    name: ClassVar[str] = "abstract"

    def complete(self, system: str, user: str, schema: type[M]) -> M:
        """Return an instance of ``schema`` populated by the model.

        Raises :class:`~src.llm.schema_guard.InvalidLLMSchemaError` if the
        schema contains a numeric field.
        """
        validate_llm_schema(schema)
        return self._complete(system, user, schema)

    @abstractmethod
    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        """Provider-specific implementation. Do not call directly."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
