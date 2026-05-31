"""
Semantic Scholar API client.

Design choices:
  - Single shared httpx.AsyncClient: created once at construction, reused for
    all requests. This keeps HTTP connections alive across calls (connection
    pooling) and is far cheaper than opening a new connection per tool call.
    Call aclose() during server shutdown to drain the connection pool cleanly.

  - In-memory TTL cache (_Cache): avoids hitting the API twice for the same
    query in one session. The cache key is (url, sorted params), which is
    deterministic for identical tool calls regardless of dict ordering.
    No external dependency (Redis, diskcache) needed for a local MCP server.

  - Structured error returns: rather than raising exceptions that FastMCP
    would swallow into opaque errors, failed requests return a dict with an
    "error" key. Tool functions can then surface a useful message to the LLM.

  - Field projection: every endpoint accepts an explicit list of fields so the
    caller controls bandwidth. Defaults live in config.py and can be overridden
    per tool call.
"""

import time
from typing import Any

import httpx

from ..config import Config, DEFAULT_PAPER_FIELDS, DEFAULT_AUTHOR_FIELDS

API_BASE = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE = "https://api.semanticscholar.org/recommendations/v1"


class _Cache:
    """Minimal TTL cache backed by a plain dict. Thread-safe for asyncio single-threaded use."""

    def __init__(self, ttl: int) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry and time.monotonic() - entry[0] < self._ttl:
            return entry[1]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)


class SemanticScholarClient:
    def __init__(self, config: Config) -> None:
        headers: dict[str, str] = {
            "User-Agent": "medical-imaging-mcp/1.0",
            "Accept": "application/json",
        }
        if config.api_key:
            headers["x-api-key"] = config.api_key

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=config.request_timeout_seconds,
        )
        self._cache = _Cache(ttl=config.cache_ttl_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cache_key = f"{url}|{sorted((params or {}).items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            res = await self._client.get(url, params=params)
            res.raise_for_status()
            data: dict[str, Any] = res.json()
            self._cache.set(cache_key, data)
            return data
        except httpx.HTTPStatusError as exc:
            return {
                "error": f"HTTP {exc.response.status_code}",
                "detail": exc.response.text[:300],
            }
        except httpx.TimeoutException:
            return {"error": "Request timed out. Semantic Scholar may be slow — try again."}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Paper endpoints
    # ------------------------------------------------------------------

    async def search_papers(
        self,
        query: str,
        fields: list[str] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Full-text search across all papers indexed by Semantic Scholar."""
        return await self._get(
            f"{API_BASE}/paper/search",
            params={
                "query": query,
                "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
                "limit": min(limit, 100),
                "offset": offset,
            },
        )

    async def get_paper(
        self,
        paper_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch a single paper by ID.

        paper_id accepts several formats:
          - Semantic Scholar ID:  "649def34f8be52c8b66281af98ae884c09aef38b"
          - arXiv ID:             "arXiv:2301.00001"
          - DOI:                  "DOI:10.1038/..."
          - PubMed ID:            "PMID:12345678"
          - Corpus ID:            "CorpusID:12345678"
        """
        return await self._get(
            f"{API_BASE}/paper/{paper_id}",
            params={"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)},
        )

    async def get_paper_citations(
        self,
        paper_id: str,
        fields: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Papers that cite the given paper (forward citations)."""
        citation_fields = [
            f"citingPaper.{f}" for f in (fields or DEFAULT_PAPER_FIELDS)
        ]
        return await self._get(
            f"{API_BASE}/paper/{paper_id}/citations",
            params={
                "fields": ",".join(citation_fields),
                "limit": min(limit, 1000),
            },
        )

    async def get_paper_references(
        self,
        paper_id: str,
        fields: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Papers referenced by the given paper (backward citations)."""
        ref_fields = [f"citedPaper.{f}" for f in (fields or DEFAULT_PAPER_FIELDS)]
        return await self._get(
            f"{API_BASE}/paper/{paper_id}/references",
            params={
                "fields": ",".join(ref_fields),
                "limit": min(limit, 1000),
            },
        )

    # ------------------------------------------------------------------
    # Author endpoints
    # ------------------------------------------------------------------

    async def get_author(
        self,
        author_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch author profile by Semantic Scholar author ID."""
        return await self._get(
            f"{API_BASE}/author/{author_id}",
            params={"fields": ",".join(fields or DEFAULT_AUTHOR_FIELDS)},
        )

    async def get_author_papers(
        self,
        author_id: str,
        fields: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Papers published by a specific author, sorted by citation count (desc)."""
        return await self._get(
            f"{API_BASE}/author/{author_id}/papers",
            params={
                "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
                "limit": min(limit, 1000),
            },
        )

    # ------------------------------------------------------------------
    # Recommendations endpoint
    # ------------------------------------------------------------------

    async def get_recommendations(
        self,
        paper_id: str,
        fields: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Papers algorithmically recommended as similar to the given paper.

        Uses the Recommendations API (separate base URL from the main graph).
        Useful for discovering related work the LLM might not surface via search.
        """
        return await self._get(
            f"{RECOMMENDATIONS_BASE}/papers/forpaper/{paper_id}",
            params={
                "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
                "limit": min(limit, 500),
            },
        )
