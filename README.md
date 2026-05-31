# Medical Imaging and ML Research Paper AI Assistant

> An MCP server that grounds AI assistants in medical imaging research

## Tech Stack

- **Backend**: Python + FastAPI
- **MCP server**: Python MCP SDK (mcp package, official Anthropic SDK)
- **DB**: PostgreSQL + pgvector (or SQLite + sqlite-vec for simpler setup)
- **Embeddings**: OpenAI text-embedding-3-small or open-source sentence-transformers
- **Frontend**: Next.js + TypeScript + Tailwind
- **LLM** (for any AI-side features): Claude via Anthropic API
- **Deployment**: Docker
