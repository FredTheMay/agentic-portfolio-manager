"""The network boundary: JSON fetching, caching, and offline replay (SPEC §2, M2).

Everything that reaches the outside world goes through :class:`JsonFetcher`.
That buys three things the rest of the system depends on:

**Determinism.** SPEC §9 requires two identical backtest runs to produce
identical output. A live HTTP call cannot promise that. Once a response is
cached it is replayed byte-for-byte, so a re-run is both free and reproducible.

**Offline operation.** The whole test suite, and any backtest over already-
fetched data, runs with no network and no API keys. An offline fetcher that is
asked for something it does not have raises :class:`OfflineError` rather than
silently returning nothing — a missing input must fail loudly, not become a
gap in the data that the optimizer quietly interpolates over.

**Politeness.** EDGAR and FRED both rate-limit and EDGAR requires a contact in
the User-Agent. One chokepoint is one place to get that right.

JSON numbers are parsed with ``parse_float=Decimal``. Financial values that
arrive as JSON floats must never round-trip through binary floating point
(SPEC §9); doing the conversion here means no caller can forget.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import httpx

#: Environment variable holding the contact string SEC EDGAR requires.
USER_AGENT_ENV = "EDGAR_USER_AGENT"

#: Sent on every outbound request. SEC EDGAR's fair-access policy requires a
#: real contact address and throttles or blocks traffic without one, so this
#: default is a placeholder that must be overridden via ``$EDGAR_USER_AGENT``
#: before any live EDGAR call. See:
#: https://www.sec.gov/os/webmaster-faq#developers
FALLBACK_USER_AGENT = "agentic-portfolio-manager (educational paper trading; set EDGAR_USER_AGENT)"


def default_user_agent() -> str:
    """Contact string for outbound requests, from ``$EDGAR_USER_AGENT``."""
    return os.environ.get(USER_AGENT_ENV) or FALLBACK_USER_AGENT


def user_agent_is_configured() -> bool:
    """Whether a real contact has been supplied.

    Checked before a live EDGAR run rather than discovered as a 403 halfway
    through backfilling a universe.
    """
    configured = os.environ.get(USER_AGENT_ENV, "")
    return "@" in configured


DEFAULT_TIMEOUT_SECONDS = 30.0


class FetchError(RuntimeError):
    """A request failed or returned something that is not usable JSON."""


class OfflineError(FetchError):
    """Offline mode was asked for a response that is not in the cache."""


#: Query parameters excluded from the canonical form of a request.
#:
#: Two reasons, both load-bearing. A credential must not decide a cache key, or
#: a replay under a different (or absent) key misses every entry and silently
#: falls back — which is exactly what happened before this existed. And the
#: canonical form appears verbatim in error messages, so including a key would
#: put it into logs, tracebacks, and CI output.
SENSITIVE_PARAMS = frozenset({"api_key", "apikey", "token", "access_token", "key"})


def _canonical(url: str, params: Mapping[str, str] | None) -> str:
    """Stable text form of a request, with credentials excluded.

    Independent of dict ordering, and independent of which key made the call.
    """
    if not params:
        return url
    visible = {k: v for k, v in params.items() if k.lower() not in SENSITIVE_PARAMS}
    if not visible:
        return url
    ordered = "&".join(f"{k}={visible[k]}" for k in sorted(visible))
    return f"{url}?{ordered}"


def cache_key(url: str, params: Mapping[str, str] | None = None) -> str:
    """Content-addressed key for a request."""
    return hashlib.sha256(_canonical(url, params).encode("utf-8")).hexdigest()


def loads(text: str) -> Any:
    """Parse JSON, decoding every fractional number straight to ``Decimal``.

    A price that arrives as the JSON literal ``0.1`` becomes exactly ``0.1``,
    not the nearest binary double. Doing this at the boundary means no caller
    downstream has to remember (SPEC §9).
    """
    try:
        return json.loads(text, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise FetchError(f"response was not valid JSON: {exc}") from exc


def dumps(value: Any) -> str:
    """Serialize to canonical JSON: sorted keys, no incidental whitespace.

    Hand-rolled rather than delegating to ``json.dumps(default=str)``, because
    that would emit a ``Decimal`` as a quoted *string*. The replayed value would
    then differ in type from the live one and the two paths would silently
    diverge — precisely the determinism guarantee this cache exists to provide.
    Here a ``Decimal`` is written as a bare JSON number, so :func:`loads`
    reconstructs it exactly.

    Canonical form also matters because the cache file *is* the record of what
    the system saw: a diff on it should show a data change, not a reordering.
    """
    return _encode(value)


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FetchError(f"cannot serialize non-finite Decimal {value}")
        # Plain notation: "1E+2" and "100" must not round-trip differently.
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Mapping):
        items = sorted((str(k), v) for k, v in value.items())
        return "{" + ",".join(f"{json.dumps(k)}:{_encode(v)}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    raise FetchError(f"cannot serialize {type(value).__name__} to canonical JSON")


@runtime_checkable
class JsonFetcher(Protocol):
    """Something that can produce a JSON document for a URL."""

    def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class ResponseCache:
    """On-disk store of JSON responses, one file per request."""

    root: Path

    def path_for(self, key: str) -> Path:
        # Two-character shard: directories with tens of thousands of entries
        # are slow to list on every filesystem worth naming.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        return loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: an interrupted run must not leave a half-written
        # file that later parses as valid but truncated data.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(dumps(value), encoding="utf-8")
        temporary.replace(path)

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()


class HttpxFetcher:
    """Live HTTP. The only class in the system that opens a socket.

    ``extra_headers`` carries per-vendor authentication. EDGAR and FRED need
    none — EDGAR identifies callers by User-Agent and FRED takes its key as a
    query parameter — but Alpaca authenticates by header, so without this a
    live market-data call returns 401. See
    :func:`src.data.sources.alpaca_headers`.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._headers = {
            "User-Agent": user_agent or default_user_agent(),
            "Accept": "application/json",
            **(dict(extra_headers) if extra_headers else {}),
        }
        self._timeout = timeout

    def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        try:
            response = httpx.get(
                url,
                params=dict(params) if params else None,
                headers=self._headers,
                timeout=self._timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"request to {url} failed: {exc}") from exc
        return loads(response.text)


class OfflineFetcher:
    """Refuses every request. The default when no network is permitted."""

    def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        raise OfflineError(
            f"offline: {_canonical(url, params)} is not cached. "
            "Run once with a live fetcher to record it."
        )


class StubFetcher:
    """Test double serving a fixed map of canonical URL -> payload."""

    def __init__(self, responses: Mapping[str, Any]) -> None:
        self._responses = dict(responses)
        self.calls: list[str] = []

    def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        canonical = _canonical(url, params)
        self.calls.append(canonical)
        if canonical not in self._responses:
            raise FetchError(f"StubFetcher has no response for {canonical}")
        return self._responses[canonical]


class CachingFetcher:
    """Serves from cache, falling through to ``inner`` on a miss.

    With ``offline=True`` there is no fall-through: an uncached request raises.
    That is the mode backtests and CI run in, so a run can never depend on the
    network being up or on a vendor's data having stayed the same.
    """

    def __init__(
        self,
        cache: ResponseCache,
        inner: JsonFetcher | None = None,
        offline: bool = False,
    ) -> None:
        if inner is None and not offline:
            raise ValueError("a live fetcher is required unless offline=True")
        self._cache = cache
        self._inner = inner
        self._offline = offline
        self.hits = 0
        self.misses = 0

    def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        key = cache_key(url, params)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        if self._offline or self._inner is None:
            raise OfflineError(
                f"offline: {_canonical(url, params)} is not cached. "
                "Run once with a live fetcher to record it."
            )

        fetched = self._inner.get_json(url, params)
        self._cache.put(key, fetched)
        return fetched
