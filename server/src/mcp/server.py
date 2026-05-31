"""
FastMCP server — tool definitions for medical imaging and ML research.

Architecture notes:
  - All tools return plain strings. FastMCP wraps these as TextContent for the
    MCP protocol. Returning formatted prose (rather than raw JSON) reduces the
    token overhead of nested dicts and makes it easier for the LLM to reason
    over results.

  - The SemanticScholarClient is created once at module load and shared across
    all tool calls. The lifespan context manager closes the underlying
    httpx.AsyncClient cleanly when the server shuts down.

  - Tool docstrings become the tool descriptions sent to the LLM. Keep them
    precise — vague descriptions cause the model to pick the wrong tool.

  - Domain-specific topic seeds (MEDICAL_IMAGING_TOPICS in config.py) let
    search_medical_imaging build optimized queries without requiring the caller
    to know the right keyword mix for each subfield.
"""

from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..config import load_config, MEDICAL_IMAGING_TOPICS
from ..ingestion.semantic_scholar import SemanticScholarClient

config = load_config()
client = SemanticScholarClient(config)


@asynccontextmanager
async def lifespan(app: Any):
    """Close the shared HTTP client when the server exits."""
    yield
    await client.aclose()


mcp = FastMCP(
    "medical-imaging-research",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Formatting helpers (private — not exposed as tools)
# ------------------------------------------------------------------

def _authors_str(authors: list[dict]) -> str:
    names = [a.get("name", "Unknown") for a in (authors or [])]
    if len(names) > 4:
        return ", ".join(names[:4]) + f" … (+{len(names) - 4} more)"
    return ", ".join(names) or "Unknown"


def _pdf_str(open_access_pdf: dict | None) -> str:
    if open_access_pdf and open_access_pdf.get("url"):
        return open_access_pdf["url"]
    return "Not available"


def _format_paper(paper: dict) -> str:
    if "error" in paper:
        return f"Error: {paper['error']}\n{paper.get('detail', '')}"

    external = paper.get("externalIds") or {}
    arxiv = external.get("ArXiv")
    doi = external.get("DOI")
    id_line = f"  ID:        {paper.get('paperId', 'N/A')}"
    if arxiv:
        id_line += f"  |  arXiv: {arxiv}"
    if doi:
        id_line += f"  |  DOI: {doi}"

    return f"""Title:     {paper.get('title', 'N/A')}
Authors:   {_authors_str(paper.get('authors', []))}
Year:      {paper.get('year', 'N/A')}
Venue:     {paper.get('venue') or 'N/A'}
Citations: {paper.get('citationCount', 'N/A')}
Open PDF:  {_pdf_str(paper.get('openAccessPdf'))}
{id_line}

Abstract:
{paper.get('abstract') or 'No abstract available.'}
""".strip()


def _format_paper_list(data: dict, nested_key: str | None = None) -> str:
    if "error" in data:
        return f"Error: {data['error']}\n{data.get('detail', '')}"

    total = data.get("total", "?")
    items: list[dict] = data.get("data", [])

    if not items:
        return "No results found."

    lines = [f"Found {total} total. Showing {len(items)}:\n"]
    for i, item in enumerate(items, 1):
        paper = item.get(nested_key, item) if nested_key else item
        title = paper.get("title", "N/A")
        year = paper.get("year", "N/A")
        authors = _authors_str(paper.get("authors", []))
        cites = paper.get("citationCount", "?")
        pid = paper.get("paperId", "")
        lines.append(f"{i}. [{year}] {title}")
        lines.append(f"   Authors: {authors}")
        lines.append(f"   Citations: {cites}  |  ID: {pid}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

@mcp.tool()
async def search_papers(query: str, limit: int = 10) -> str:
    """
    Search Semantic Scholar for research papers matching a free-text query.

    Returns title, authors, year, venue, citation count, and paper ID for each
    result. Use the paper ID with get_paper_details, get_paper_citations, or
    get_recommended_papers for deeper exploration.

    Args:
        query: Free-text search query (e.g., "transformer segmentation MRI 2023").
        limit: Number of results to return (1-100, default 10).
    """
    data = await client.search_papers(query=query, limit=limit)
    return _format_paper_list(data)


@mcp.tool()
async def search_medical_imaging(
    topic: str,
    extra_query: str = "",
    limit: int = 10,
) -> str:
    """
    Search for medical imaging and ML papers using a pre-built domain topic seed.

    The topic parameter selects a curated keyword base optimised for that
    subfield. Add extra_query to narrow results further (e.g. a specific
    architecture, dataset, or year range).

    Available topics:
        segmentation, classification, detection, reconstruction, registration,
        generation, mri, ct, xray, ultrasound, pathology, fundus, dermoscopy,
        pet, endoscopy, umap

    Args:
        topic:       One of the topic keys listed above.
        extra_query: Additional keywords appended to the topic seed (optional).
        limit:       Number of results to return (1-100, default 10).
    """
    seed = MEDICAL_IMAGING_TOPICS.get(topic.lower())
    if seed is None:
        available = ", ".join(sorted(MEDICAL_IMAGING_TOPICS.keys()))
        return f"Unknown topic '{topic}'. Available topics: {available}"

    full_query = f"{seed} {extra_query}".strip()
    data = await client.search_papers(query=full_query, limit=limit)
    return f"Query: {full_query}\n\n" + _format_paper_list(data)


@mcp.tool()
async def get_paper_details(paper_id: str) -> str:
    """
    Get full details for a single paper by its Semantic Scholar paper ID.

    Accepted ID formats:
      - Semantic Scholar hash: "649def34f8be52c8b66281af98ae884c09aef38b"
      - arXiv:                 "arXiv:2301.00001"
      - DOI:                   "DOI:10.18653/v1/..."
      - PubMed:                "PMID:12345678"
      - Corpus:                "CorpusID:12345678"

    Returns title, authors, year, venue, abstract, citation count, and a PDF
    link if the paper is open-access.

    Args:
        paper_id: Paper identifier in any of the formats listed above.
    """
    data = await client.get_paper(paper_id=paper_id)
    return _format_paper(data)


@mcp.tool()
async def get_paper_citations(paper_id: str, limit: int = 10) -> str:
    """
    Get papers that cite the given paper (forward citations).

    Useful for finding follow-up work, implementations, or critiques of a
    foundational paper. Results are ordered by Semantic Scholar's relevance
    ranking.

    Args:
        paper_id: Semantic Scholar paper ID (or arXiv/DOI/PMID prefix).
        limit:    Number of citing papers to return (default 10, max 1000).
    """
    data = await client.get_paper_citations(paper_id=paper_id, limit=limit)
    return _format_paper_list(data, nested_key="citingPaper")


@mcp.tool()
async def get_paper_references(paper_id: str, limit: int = 10) -> str:
    """
    Get papers referenced by the given paper (backward citations).

    Useful for tracing the foundational methods and datasets a paper builds on.

    Args:
        paper_id: Semantic Scholar paper ID (or arXiv/DOI/PMID prefix).
        limit:    Number of referenced papers to return (default 10, max 1000).
    """
    data = await client.get_paper_references(paper_id=paper_id, limit=limit)
    return _format_paper_list(data, nested_key="citedPaper")


@mcp.tool()
async def get_author_details(author_id: str) -> str:
    """
    Get profile and publication stats for an author by their Semantic Scholar author ID.

    Returns name, affiliations, h-index, total paper count, and total citation count.
    Author IDs appear in paper search results under the 'authors' field.

    Args:
        author_id: Numeric Semantic Scholar author ID (e.g., "1741101").
    """
    data = await client.get_author(author_id=author_id)
    if "error" in data:
        return f"Error: {data['error']}\n{data.get('detail', '')}"

    affiliations = ", ".join(data.get("affiliations") or []) or "Not listed"
    return f"""Name:         {data.get('name', 'N/A')}
Author ID:    {data.get('authorId', 'N/A')}
Affiliations: {affiliations}
Papers:       {data.get('paperCount', 'N/A')}
Citations:    {data.get('citationCount', 'N/A')}
h-index:      {data.get('hIndex', 'N/A')}"""


@mcp.tool()
async def get_author_papers(author_id: str, limit: int = 10) -> str:
    """
    Get papers published by a specific author, ordered by citation count.

    Args:
        author_id: Numeric Semantic Scholar author ID.
        limit:     Number of papers to return (default 10, max 1000).
    """
    data = await client.get_author_papers(author_id=author_id, limit=limit)
    return _format_paper_list(data)


@mcp.tool()
async def get_recommended_papers(paper_id: str, limit: int = 10) -> str:
    """
    Get papers algorithmically recommended as similar to the given paper.

    Uses Semantic Scholar's recommendations engine, which considers citation
    graph proximity and content similarity. Useful for discovering related work
    that a keyword search might miss.

    Args:
        paper_id: Semantic Scholar paper ID (hash format only for this endpoint).
        limit:    Number of recommended papers to return (default 10, max 500).
    """
    data = await client.get_recommendations(paper_id=paper_id, limit=limit)
    # Recommendations API returns {"recommendedPapers": [...]} not {"data": [...]}
    if "error" in data:
        return f"Error: {data['error']}\n{data.get('detail', '')}"

    papers = data.get("recommendedPapers", [])
    if not papers:
        return "No recommendations found for this paper."

    lines = [f"Showing {len(papers)} recommended papers:\n"]
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "N/A")
        year = paper.get("year", "N/A")
        authors = _authors_str(paper.get("authors", []))
        cites = paper.get("citationCount", "?")
        pid = paper.get("paperId", "")
        lines.append(f"{i}. [{year}] {title}")
        lines.append(f"   Authors: {authors}")
        lines.append(f"   Citations: {cites}  |  ID: {pid}")
    return "\n".join(lines)
