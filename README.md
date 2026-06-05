# Medical Imaging ML Research MCP

An MCP server that gives AI assistants (Claude Desktop, Claude Code, or any MCP-compatible host) live access to the [Semantic Scholar](https://www.semanticscholar.org/) API for medical imaging and ML research. Instead of hallucinating citations, the assistant can search papers, trace citations, explore author profiles, and get algorithmic recommendations — all grounded in real data.

## Tools exposed

| Tool | Description |
|---|---|
| `search_papers` | Free-text search across all of Semantic Scholar |
| `search_medical_imaging` | Domain-focused search using pre-built topic seeds (segmentation, MRI, CT, pathology, etc.) |
| `get_paper_details` | Full metadata + abstract for a single paper by ID |
| `get_paper_citations` | Forward citations (papers that cite this one) |
| `get_paper_references` | Backward citations (papers this one cites) |
| `get_author_details` | Author profile: affiliations, h-index, citation count |
| `get_author_papers` | Papers by a specific author, sorted by citations |
| `get_recommended_papers` | Algorithmically similar papers via Semantic Scholar's recommendation engine |

## Project structure

```text
.
├── main.py                         # Entry point — configures and runs the MCP server
├── pyproject.toml                  # Dependencies and project metadata (managed by uv)
├── uv.lock                         # Locked dependency versions
├── Dockerfile                      # Container build (python:3.13-slim + uv)
├── docker-compose.yml              # Compose service definition
└── server/
    └── src/
        ├── config.py               # All env-var-driven configuration + topic seeds
        ├── mcp/
        │   └── server.py           # FastMCP tool definitions (the 8 tools above)
        └── ingestion/
            └── semantic_scholar.py # Async httpx client wrapping the Semantic Scholar API
```

## Configuration

All values are set via environment variables (or a `.env` file in the project root).

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | _(none)_ | API key for higher rate limits (1 req/s vs ~100 req/5 min unauthenticated). Get one at [semanticscholar.org](https://www.semanticscholar.org/product/api). |
| `CACHE_TTL_SECONDS` | `300` | How long to cache identical API responses within a session. |
| `MAX_RESULTS_DEFAULT` | `10` | Default result count when tools don't specify a limit. |
| `REQUEST_TIMEOUT_SECONDS` | `30.0` | HTTP request timeout. |
| `REQUEST_INTERVAL_SECONDS` | `1.0` | Minimum delay between API calls (respects rate limits). |
| `MCP_TRANSPORT` | `stdio` | Transport mode. Use `streamable-http` to run as a network server. |

Example `.env`:

```env
SEMANTIC_SCHOLAR_API_KEY=your_key_here
CACHE_TTL_SECONDS=300
```

## Running locally

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
# Clone and install dependencies
git clone <repo-url>
cd medical-imaging-ml-research-mcp

# Copy and fill in env vars
cp .env.example .env   # or create .env manually

# Run the server (stdio transport — for direct MCP host integration)
uv run main.py

# Run as an HTTP server (for testing with curl or multiple clients)
MCP_TRANSPORT=streamable-http uvicorn main:app --port 8000
```

## Running with Docker

```bash
# Build and run
docker compose up --build

# Or build/run directly
docker build -t medical-imaging-mcp .
docker run --env-file .env -i medical-imaging-mcp
```

## Registering with Claude Code

**Option 1 — CLI (quickest):**

```bash
claude mcp add medical-imaging-research -- uv run main.py
```

**Option 2 — project-level config** (`.claude/mcp.json`):

```json
{
  "mcpServers": {
    "medical-imaging-research": {
      "command": "uv",
      "args": ["run", "main.py"],
      "cwd": "/absolute/path/to/this/directory",
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "<your key>"
      }
    }
  }
}
```

## Tech stack

- **MCP framework:** [FastMCP](https://github.com/jlowin/fastmcp) (Python MCP SDK)
- **HTTP client:** `httpx` (async)
- **Runtime/package manager:** `uv`
- **Python:** 3.13+
- **Data source:** [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/graph/v1)
