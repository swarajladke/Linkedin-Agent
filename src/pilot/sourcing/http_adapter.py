"""Base HTTP client with rate limit awareness, exponential backoff, and caching."""

import time
from pathlib import Path
from typing import Any

import httpx

from pilot.ingestion.cache import IngestionCache
from pilot.sourcing.base import (
    JobSourceError,
    JobSourceNotFoundError,
    JobSourceRateLimitError,
)


class BaseHttpAdapter:
    """Base class for ATS/job board adapters sharing HTTP behaviors."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.cache = IngestionCache(cache_dir=cache_dir)
        self.transport = transport
        self.timeout = httpx.Timeout(timeout, connect=5.0)

    def _get_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "Pilot-Autonomous-Career-Agent",
        }

    def _request_json(
        self,
        url: str,
        namespace: str,
        cache_key: str,
        max_retries: int = 3,
        force_refresh: bool = False,
    ) -> Any:
        """Fetch JSON payload with local caching, rate-limit detection, and 5xx retries."""
        if not force_refresh:
            cached = self.cache.get(namespace, cache_key)
            if cached is not None:
                return cached

        headers = self._get_headers()
        backoff = 0.2

        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = client.get(url, headers=headers)

                    # Rate Limit handling
                    remaining = response.headers.get("X-RateLimit-Remaining")
                    if remaining is not None and int(remaining) == 0:
                        reset_ts = response.headers.get("X-RateLimit-Reset", "0")
                        raise JobSourceRateLimitError(
                            f"Rate limit exceeded for {url}. Reset timestamp: {reset_ts}."
                        )

                    if response.status_code == 429:
                        raise JobSourceRateLimitError(f"HTTP 429 Rate limit exceeded for {url}")

                    # Not found
                    if response.status_code == 404:
                        raise JobSourceNotFoundError(f"Target resource not found: {url}")

                    # 5xx Server Errors -> retry
                    if response.status_code >= 500:
                        if attempt < max_retries:
                            time.sleep(backoff)
                            backoff *= 2
                            continue
                        response.raise_for_status()

                    response.raise_for_status()
                    payload = response.json()
                    self.cache.set(namespace, cache_key, payload)
                    return payload

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise JobSourceError(f"Network error communicating with {url}: {exc}") from exc

            raise JobSourceError(f"Failed to fetch {url} after {max_retries} retries")
