"""Groq provider (SPEC §8).

⚠️ **Groq** is an inference provider with a free tier. **Grok** is xAI's model
and is paid. They are different companies and the spec calls this out because
the names are one letter apart.

Fallback, not primary. Groq is very fast but its free tier on the larger models
is capped nearer 100K tokens/day, so it is reserved for short classification
calls rather than for handing a model an entire filing. Free-tier quotas change
often; verify before relying on one.

Uses the OpenAI-compatible chat-completions shape that Groq exposes, with
``response_format: json_object``. As with Gemini, the result is validated with
Pydantic regardless of what the API promises.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.llm.base import InvalidResponseError, LLMError, LLMProvider, RateLimitError

M = TypeVar("M", bound=BaseModel)

API_KEY_ENV = "GROQ_API_KEY"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT = 60.0


@dataclass
class GroqProvider(LLMProvider):
    """Calls Groq and validates the result against the requested schema."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"groq:{self.model}"

    def _resolved_key(self) -> str:
        key = self.api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise LLMError(
                f"no Groq API key: set {API_KEY_ENV} or pass api_key. "
                "The system still runs with LLM_PROVIDER=null."
            )
        return key

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        # The schema goes in the prompt as well as in response_format: the
        # OpenAI-compatible json_object mode guarantees valid JSON, not JSON
        # of a particular shape.
        instructions = (
            f"{system}\n\nRespond with JSON matching exactly this schema:\n"
            f"{json.dumps(schema.model_json_schema(), sort_keys=True)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        try:
            response = httpx.post(
                BASE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self._resolved_key()}"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Groq rate limit reached")
        if response.status_code >= 400:
            raise LLMError(f"Groq returned {response.status_code}: {response.text[:200]}")

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidResponseError(f"unexpected Groq response shape: {exc}") from exc

        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise InvalidResponseError(f"Groq output failed schema validation: {exc}") from exc
