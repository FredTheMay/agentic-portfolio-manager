"""LLM provider layer (SPEC §8).

Providers are interchangeable and swapping one touches nothing outside this
package — selection is by the ``LLM_PROVIDER`` environment variable.

⚠️ **Groq** is an inference provider with a free tier. **Grok** is xAI's model
and is paid. Different companies, one letter apart.

The LLM contributes qualitative judgment only; :mod:`src.llm.schema_guard`
enforces that (SPEC §2.1), and :class:`~src.llm.null.NullProvider` means the
whole system still runs with the LLM switched off.
"""

from __future__ import annotations

import os

from src.llm.base import (
    Conviction,
    InvalidResponseError,
    LLMError,
    LLMProvider,
    RateLimitError,
    Stance,
    TokenBucket,
)
from src.llm.cache import (
    CachingProvider,
    FailoverProvider,
    LLMResponseCache,
    ResilientProvider,
)
from src.llm.null import NullProvider
from src.llm.schema_guard import InvalidLLMSchemaError, validate_llm_schema

#: Environment variable selecting the provider (SPEC §8).
PROVIDER_ENV = "LLM_PROVIDER"
DEFAULT_PROVIDER = "null"


def get_provider(name: str | None = None) -> LLMProvider:
    """Build the configured provider.

    Defaults to :class:`~src.llm.null.NullProvider`, so a checkout with no API
    keys runs the full pipeline out of the box and every agent returns NEUTRAL.
    Gemini and Groq are imported lazily; neither is needed to run the system.
    """
    key = (name or os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).lower()
    if key == "null":
        return NullProvider()
    if key == "gemini":
        from src.llm.gemini import GeminiProvider

        return GeminiProvider()
    if key == "groq":
        from src.llm.groq import GroqProvider

        return GroqProvider()
    if key == "failover":
        from src.llm.gemini import GeminiProvider
        from src.llm.groq import GroqProvider

        # Gemini first: far more requests per day, and the larger context suits
        # full-filing prompts. Groq is reserved for short classification calls.
        return FailoverProvider(providers=[GeminiProvider(), GroqProvider(), NullProvider()])
    raise LLMError(f"unknown LLM provider {key!r}; use null, gemini, groq, or failover")


__all__ = [
    "CachingProvider",
    "Conviction",
    "DEFAULT_PROVIDER",
    "FailoverProvider",
    "InvalidLLMSchemaError",
    "InvalidResponseError",
    "LLMError",
    "LLMProvider",
    "LLMResponseCache",
    "NullProvider",
    "PROVIDER_ENV",
    "RateLimitError",
    "ResilientProvider",
    "Stance",
    "TokenBucket",
    "get_provider",
    "validate_llm_schema",
]
