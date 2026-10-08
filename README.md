# Enterprise AI Assistant

An open, extensible enterprise knowledge assistant. The bundled demo persona is a generic commercial bank (configurable, see Branding). It answers questions over internal
documents (policies, runbooks, incident reports, architecture documents, product specifications, meeting notes),
calls enterprise tools, cites its evidence, and enforces role-based access to both documents and tools.

Stack: FastAPI (async) | LangGraph multi-agent orchestration | Recursive Language Model (RLM) research agent |
hybrid dense + BM25 retrieval on Pinecone | MCP tool server | LangSmith tracing | Streamlit UI with a live agent activity panel.

## Features

- Multi-agent LangGraph graph: input guard, memory, supervisor, retrieval, research (RLM), tools, human approval, response, validator.
- Hybrid retrieval: dense embeddings + BM25 sparse vectors, weighted fusion, reranking, metadata filters, one Pinecone namespace per access tier.
- RLM research agent: the model writes a Python search plan, explores the corpus through a restricted API, reads only targeted sections,
  processes batches in parallel with sub-model calls, recurses on oversized contexts, and aggregates cited findings.
- Tools: knowledge search, sandboxed pandas analysis, MCP server (employee directory, service catalog, incident records, ticket creation).
- Security: JWT auth, RBAC (viewer, analyst, administrator), prompt-injection detection (user input, retrieved documents, tool output),
  tool registry with validation and human approval for write tools, citation verification, secret and card-number redaction, audit log.
- Memory: working memory, rolling summary, long-term episodic memory and preferences, feedback-weighted recall.
- Reliability: retries, circuit breakers, timeouts, graceful degradation when the LLM, vector database or MCP server is unavailable.
- Observability: structured JSON logs, LangSmith traces for every turn, node, tool call, retrieval and LLM call.
- Token-bucket rate limiting per user with configurable thresholds.

## Architecture overview

```
Streamlit UI  --SSE-->  FastAPI  (JWT auth, RBAC, rate limit, validation)
                           |
                      LangGraph graph
   input_guard -> memory_load -> supervisor -> [retrieval | research (RLM) | tools <-> approval]
                                                      |
                                            response -> validator -> memory_write
                           |
        Tool registry (RBAC, args validation, approval, audit) --> Pinecone | MCP server | pandas sandbox
```

Access model: documents are tagged `public`, `internal`, `confidential` or `restricted`; each tier is a Pinecone namespace.

| Role | Tools | Readable document tiers |
|---|---|---|
| viewer | chat, search | public, internal |
| analyst | + analytics, MCP read tools | + confidential |
| administrator | all tools (write tools need approval) | + restricted |

## Quick start (no API keys needed)

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env

uvicorn app.main:app --port 8000     # terminal 1 (first start builds the index automatically)
streamlit run ui/streamlit_app.py    # terminal 2, opens http://localhost:8501
```

With empty keys the app runs in offline mode: a deterministic mock LLM and a local vector store. Every feature works,
but answers are extractive. The activity panel shows which LLM provider and vector backend are active.

### Demo users

| user | password | role |
|---|---|---|
| viewer | viewer123 | viewer |
| analyst | analyst123 | analyst |
| admin | admin123 | administrator |

These are hard-coded for local use only. Replace `authenticate()` and `decode_token()` in `app/security/auth.py` with your
identity provider (for example OIDC) for any shared deployment. The RBAC code does not change.

## Configuration

All settings are environment variables (see `.env.example` for the full list).

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `auto` (default), `openai`, `azure`, `mock` |
| `OPENAI_API_KEY`, `LLM_MODEL_FAST`, `LLM_MODEL_STRONG`, `EMBEDDING_MODEL` | OpenAI models |
| `AZURE_OPENAI_*`, `AZURE_DEPLOYMENT_*` | Azure OpenAI endpoint and deployments |
| `VECTOR_BACKEND` | `auto` (Pinecone if a key is set), `pinecone`, `local` |
| `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_CLOUD`, `PINECONE_REGION` | Pinecone |
| `HYBRID_ALPHA`, `HYBRID_MODE` | dense/sparse weight; `fusion` (two queries, explainable) or `native` (one query) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | tracing |
| `ORG_NAME`, `ASSISTANT_NAME` | persona used in the system prompt and greetings |
| `JWT_SECRET` | signing key for access tokens (set a long random value) |
| `RATE_LIMITS` | per-role token bucket (JSON: capacity, refill_per_sec) |
| `MCP_TRANSPORT`, `MCP_URL` | `stdio` (spawned on demand) or `http` (separate server) |
| `RLM_CONTEXT_CHARS`, `RLM_MAX_DEPTH` | RLM recursion tuning |
| `TODAY_OVERRIDE` | pin the date so relative phrases like "last year" are reproducible |

### Using real services

```ini
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=enterprise-ai-assistant
```

Then rebuild the index and restart the API:

```bash
python -m app.ingestion.ingest       # embeds chunks, writes data/index, upserts to Pinecone namespaces
uvicorn app.main:app --port 8000
```

Changing the LLM provider or embedding model changes vector dimensions, so the index is rebuilt automatically on the next start.
Verify with `curl localhost:8000/ready`, which reports the active LLM provider and vector backend.

## Branding and adapting to your organization

Set `ORG_NAME` and `ASSISTANT_NAME` in `.env`. The tone rules, banned phrases and regulatory disclaimer live in `app/brand.py`;
edit them to match your industry. The bundled documents, departments and services (`app/domain.py`, `scripts/generate_sample_data.py`)
are demo data to replace with your own.

## Your own documents

Put Markdown files with YAML front matter in `data/sample_docs/` and run `python -m app.ingestion.ingest`.

```markdown
---
doc_id: POL-001
title: Example Policy
department: payments
document_type: policy
access_level: internal
created_date: 2026-01-15
---
# Example Policy
## Section heading
Body text...
```

Required fields: `doc_id`, `title`, `department`, `document_type`, `access_level`, `created_date`.
Optional incident fields: `service`, `severity`, `root_cause_category`, `duration_minutes`.
To regenerate the bundled sample corpus: `python scripts/generate_sample_data.py`.

## MCP server

By default the API spawns the MCP server over stdio per call. To run it as a separate process:

```bash
MCP_SERVER_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 python -m app.mcp_server.server
# .env: MCP_TRANSPORT=http  MCP_URL=http://127.0.0.1:8765/mcp
```

Tools: `search_employees`, `list_services`, `get_service`, `list_incidents`, `get_incident`, `create_incident_ticket` (write).

## Docker Compose

```bash
cp .env.example .env
docker compose up --build                 # API :8000, UI :8501, MCP :8765
docker compose run --rm ingest            # rebuild index / push to Pinecone
```

## Tests

```bash
python -m pytest -q
```

The suite runs fully offline (mock LLM, local vector store) and includes failure-injection tests for the LLM, vector database and MCP server.

## API

`POST /auth/login`, `GET /me`, `POST /chat/stream` (server-sent events), `POST /chat/approve` (server-sent events),
`GET /sessions/{id}/memory`, `POST /feedback`, `GET /health`, `GET /ready`, and admin endpoints
`GET /admin/audit`, `GET /admin/feedback`, `POST /admin/reindex`. Interactive docs at `http://localhost:8000/docs`.

## Project layout

```
app/
  main.py, api/             FastAPI app and routes (SSE streaming)
  services/                 container (wiring) and orchestrator (run and resume a turn)
  agents/                   LangGraph graph, nodes, state, heuristics
  agents/rlm/               planner, environment, runner, restricted runtime
  retrieval/                sparse encoder, vector stores, hybrid retriever, catalog, filters
  tools/                    tool registry, built-in tools, pandas sandbox, MCP client
  mcp_server/               dummy enterprise data over MCP
  memory/                   session and long-term memory
  security/                 rbac, auth, rate limiting, injection, guardrails, audit
  llm/                      OpenAI / Azure / mock provider
  ingestion/                chunking and index build
ui/streamlit_app.py         chat and agent activity panel
data/                       sample documents and MCP data
scripts/                    sample data generator
tests/
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says the API is not reachable | start `uvicorn` first and check `API_URL` |
| Pinecone dimension mismatch | delete the index in the Pinecone console or change `PINECONE_INDEX` |
| `NotImplementedError` from asyncio subprocess on Windows | use WSL or Docker (developed and tested on Linux) |
| `mcp` import errors | keep `mcp<2` (version 2 renamed the server API) |
| HTTP 429 | expected rate limiting; tune `RATE_LIMITS` |
