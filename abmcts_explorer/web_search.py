from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    description: str
    position: int


@dataclass(frozen=True)
class WebSearchResponse:
    success: bool
    results: list[WebSearchResult]
    error: str | None = None


def parse_searxng_engines(raw: Optional[str]) -> list[str]:
    if not raw or not raw.strip():
        return []
    value = raw.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class SearXNGSearchConfig:
    base_url: str
    engines: tuple[str, ...] = ()
    language: str | None = None
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> SearXNGSearchConfig:
        language = os.getenv("SEARXNG_LANGUAGE", "").strip()
        return cls(
            base_url=os.getenv("SEARXNG_URL", "").strip().rstrip("/"),
            engines=tuple(parse_searxng_engines(os.getenv("SEARXNG_ENGINE"))),
            language=language or None,
            timeout_seconds=float(
                os.getenv(
                    "SEARXNG_TIMEOUT_SECONDS",
                    str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
                )
            ),
        )


class SearXNGSearchClient:
    def __init__(self, config: SearXNGSearchConfig | None = None) -> None:
        self.config = config or SearXNGSearchConfig.from_env()

    @property
    def name(self) -> str:
        return "searxng"

    def is_available(self) -> bool:
        return bool(self.config.base_url)

    def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        if not self.config.base_url:
            return WebSearchResponse(
                success=False,
                results=[],
                error="SEARXNG_URL is not set",
            )

        if not self.config.engines:
            return self._search_once(query=query, limit=limit, engine=None)

        last_error: str | None = None
        for engine in self.config.engines:
            response = self._search_once(query=query, limit=limit, engine=engine)
            if not response.success:
                last_error = response.error or "Unknown error"
                logger.warning(
                    "SearXNG engine '%s' failed for '%s': %s; trying next",
                    engine,
                    query,
                    last_error,
                )
                continue
            if response.results:
                logger.info(
                    "SearXNG search '%s': %d results via engine '%s'",
                    query,
                    len(response.results),
                    engine,
                )
                return response
            logger.info(
                "SearXNG engine '%s' returned no results for '%s'; trying next",
                engine,
                query,
            )

        if last_error:
            return WebSearchResponse(success=False, results=[], error=last_error)
        return WebSearchResponse(success=True, results=[])

    def _search_once(
        self,
        *,
        query: str,
        limit: int,
        engine: str | None,
    ) -> WebSearchResponse:
        import httpx

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if engine:
            params["engines"] = engine
        if self.config.language:
            params["language"] = self.config.language

        try:
            response = httpx.get(
                f"{self.config.base_url}/search",
                params=params,
                timeout=self.config.timeout_seconds,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error (engine=%s): %s", engine, exc)
            return WebSearchResponse(
                success=False,
                results=[],
                error=f"SearXNG returned HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error (engine=%s): %s", engine, exc)
            return WebSearchResponse(
                success=False,
                results=[],
                error=f"Could not reach SearXNG at {self.config.base_url}: {exc}",
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.warning("SearXNG response parse error (engine=%s): %s", engine, exc)
            return WebSearchResponse(
                success=False,
                results=[],
                error="Could not parse SearXNG response as JSON",
            )

        raw_results = data.get("results", [])
        sorted_results = sorted(
            raw_results,
            key=lambda result: float(result.get("score", 0)),
            reverse=True,
        )[:limit]

        results = [
            WebSearchResult(
                title=str(result.get("title", "")),
                url=str(result.get("url", "")),
                description=str(result.get("content", "")),
                position=index + 1,
            )
            for index, result in enumerate(sorted_results)
        ]

        return WebSearchResponse(success=True, results=results)
