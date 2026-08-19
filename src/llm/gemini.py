"""Google Gemini provider.

Structured output is requested through ``responseSchema`` and validated with
Pydantic regardless — a model that claims to honor a schema and does not is
exactly what the resilient wrapper exists to absorb.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.data.cache import redact
from src.llm.base import InvalidResponseError, LLMError, LLMProvider, RateLimitError

M = TypeVar("M", bound=BaseModel)

API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_TIMEOUT = 60.0


def _strip_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a JSON schema to the subset Gemini's ``responseSchema`` accepts."""
    allowed = {"type", "properties", "required", "items", "enum", "description"}
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _strip_unsupported(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _strip_unsupported(value)
        else:
            cleaned[key] = value
    return cleaned


@dataclass
class GeminiProvider(LLMProvider):
    """Calls Gemini and validates the result against the requested schema."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"gemini:{self.model}"

    def _resolved_key(self) -> str:
        key = self.api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise LLMError(
                f"no Gemini API key: set {API_KEY_ENV} or pass api_key. "
                "The system still runs with LLM_PROVIDER=null."
            )
        return key

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _strip_unsupported(schema.model_json_schema()),
                # Determinism matters more than variety here: the same prompt
                # should give the same view.
                "temperature": 0,
            },
        }

        try:
            response = httpx.post(
                f"{BASE_URL}/{self.model}:generateContent",
                params={"key": self._resolved_key()},
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(redact(f"Gemini request failed: {exc}")) from exc

        if response.status_code == 429:
            raise RateLimitError("Gemini rate limit reached")
        if response.status_code >= 400:
            raise LLMError(redact(f"Gemini returned {response.status_code}: {response.text[:200]}"))

        try:
            body = response.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidResponseError(f"unexpected Gemini response shape: {exc}") from exc

        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise InvalidResponseError(f"Gemini output failed schema validation: {exc}") from exc
