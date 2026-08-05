"""LLM provider layer (SPEC §8).

Providers are interchangeable and swapping one touches nothing outside this
package. The LLM contributes qualitative judgment only — see
:mod:`src.llm.schema_guard` for the enforcement of SPEC §2.1.
"""

from src.llm.base import Conviction, LLMProvider, Stance
from src.llm.null import NullProvider
from src.llm.schema_guard import InvalidLLMSchemaError, validate_llm_schema

__all__ = [
    "Conviction",
    "InvalidLLMSchemaError",
    "LLMProvider",
    "NullProvider",
    "Stance",
    "validate_llm_schema",
]
