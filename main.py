"""
Entry point for the medical imaging MCP server.

Transport:
  The default transport is stdio, which is what Claude Desktop and Claude Code
  expect — the MCP host spawns this process and communicates over stdin/stdout.

  To run as a network server instead (e.g., for testing with curl or multiple
  clients), set the MCP_TRANSPORT environment variable:

    MCP_TRANSPORT=streamable-http uvicorn main:app --port 8000

  or switch the mcp.run() call to mcp.run(transport="streamable-http").

Running locally:
    uv run main.py

Registering with Claude Code (project-level):
    Add to .claude/mcp.json:
    {
      "mcpServers": {
        "medical-imaging-research": {
          "command": "uv",
          "args": ["run", "main.py"],
          "cwd": "<absolute path to this directory>",
          "env": {
            "SEMANTIC_SCHOLAR_API_KEY": "<your key>"
          }
        }
      }
    }

    Or register from the CLI:
    claude mcp add medical-imaging-research -- uv run main.py
"""

from server.src.mcp.server import mcp
import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
