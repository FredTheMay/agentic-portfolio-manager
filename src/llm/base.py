"""LLM provider abstraction (SPEC §8).

All LLM output is Pydantic-validated structured data — never free text parsed
by regex. Every call passes through :func:`~src.llm.schema_guard.validate_llm_schema`,
so a numeric field in a response schema fails loudly at the boundary rather
than quietly contaminating a portfolio weight.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Annotated, ClassVar, TypeVar

from pydantic import BaseModel, Field

from src.llm.schema_guard import CONVICTION_MARKER, validate_llm_schema

M = TypeVar("M", bound=BaseModel)


class Stance(str, enum.Enum):
    """The categorical view an LLM agent is permitted to express (SPEC §2.1).

    A ``str`` enum, not an ``IntEnum``: an integer here would be a number the
    LLM produced, and downstream code could do arithmetic on it.
    """

    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


#: The only integer an LLM may emit: an ordinal 1-5, mapped to numeric tilts by
#: ``config/view_mapping.yaml`` (SPEC §5.4) rather than used in arithmetic.
Conviction = Annotated[int, CONVICTION_MARKER, Field(ge=1, le=5)]


class LLMError(Exception):
    """Base class for provider failures."""


class LLMProvider(ABC):
    """A source of structured, schema-validated qualitative judgment.

    Subclasses implement :meth:`_complete`. :meth:`complete` is deliberately
    not overridable in spirit — it is where the §2.1 guard runs.
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
