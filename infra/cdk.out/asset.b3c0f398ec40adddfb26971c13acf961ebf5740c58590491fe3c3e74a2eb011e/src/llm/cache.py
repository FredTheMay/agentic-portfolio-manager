"""Response caching, provider failover, and the NEUTRAL fallback.

Three composable wrappers. Caching keys on provider, prompt and schema shape,
so re-runs are free and a backtest replays identically — free tiers are
measured in requests per day. Failover tries providers in order. Resilient
retries schema-invalid output, then answers NEUTRAL rather than raising,
because a malformed response should not take down a rebalance cycle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.base import (
    MAX_REPARSE_ATTEMPTS,
    InvalidResponseError,
    LLMError,
    LLMProvider,
    TokenBucket,
)
from src.llm.null import NullProvider

M = TypeVar("M", bound=BaseModel)


def cache_key(provider: str, system: str, user: str, schema: type[BaseModel]) -> str:
    """Stable key for one completion request.

    Includes the schema name *and* its field structure: changing a response
    model changes what a valid answer looks like, so a cached response from the
    old shape must not be served against the new one.
    """
    shape = json.dumps(
        {name: str(f.annotation) for name, f in schema.model_fields.items()}, sort_keys=True
    )
    payload = "\x1f".join([provider, schema.__name__, shape, system, user])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LLMResponseCache:
    """On-disk cache of model responses, one JSON file per request."""

    root: Path

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str, schema: type[M]) -> M | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            # A cached response that no longer fits the schema is stale, not
            # fatal. Treat it as a miss and let the provider answer again.
            return None

    def put(self, key: str, value: BaseModel) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(value.model_dump_json(), encoding="utf-8")
        temporary.replace(path)


@dataclass
class CachingProvider(LLMProvider):
    """Serves from cache, delegating to ``inner`` on a miss."""

    inner: LLMProvider
    cache: LLMResponseCache
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"cached:{self.inner.name}"

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        key = cache_key(self.inner.name, system, user, schema)
        cached = self.cache.get(key, schema)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        response = self.inner.complete(system, user, schema)
        self.cache.put(key, response)
        return response


@dataclass
class FailoverProvider(LLMProvider):
    """Tries each provider in order until one answers."""

    providers: Sequence[LLMProvider]

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("FailoverProvider needs at least one provider")

    @property
    def name(self) -> str:  # type: ignore[override]
        return "failover:" + ",".join(p.name for p in self.providers)

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.complete(system, user, schema)
            except LLMError as exc:
                errors.append(f"{provider.name}: {exc}")
        raise LLMError("every provider failed: " + "; ".join(errors))


@dataclass
class ResilientProvider(LLMProvider):
    """Retries invalid output, then answers NEUTRAL rather than failing.

    (3). The fallback is deliberate: an agent that cannot parse a
    response has no view, and "no view" is a legitimate, auditable outcome that
    maps to a zero tilt. Raising instead would let one malformed response take
    down a rebalance cycle.
    """

    inner: LLMProvider
    attempts: int = MAX_REPARSE_ATTEMPTS
    bucket: TokenBucket | None = None
    fallbacks: int = field(default=0, init=False)

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"resilient:{self.inner.name}"

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        if self.bucket is not None and not self.bucket.take():
            # Out of local budget: answer neutrally rather than burn the quota
            # or block the cycle.
            self.fallbacks += 1
            return NullProvider().complete(system, user, schema)

        last: Exception | None = None
        for _ in range(max(1, self.attempts)):
            try:
                return self.inner.complete(system, user, schema)
            except (InvalidResponseError, ValidationError) as exc:
                last = exc
            except LLMError as exc:
                last = exc
                break

        self.fallbacks += 1
        return NullProvider().complete(system, user, schema)
