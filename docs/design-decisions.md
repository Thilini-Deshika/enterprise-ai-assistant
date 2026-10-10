# Design decisions

## Assumptions

- The assistant speaks for a commercial bank. Name and organisation are set with `ORG_NAME` and `ASSISTANT_NAME`. Tone is calm and precise, with no guarantees and no regulatory or financial advice.
- Documents have four access tiers (`public`, `internal`, `confidential`, `restricted`), mapped to roles, so access control applies to data and not only to tools.
- Analysts may use read-only MCP tools. Write tools are administrator-only and need human approval.
- Authentication uses hard-coded demo users and JWT. The RBAC layer does not depend on how identity is established, so an OIDC provider such as Keycloak would only replace `authenticate()` and `decode_token()`.
- "Last year" means the previous 12 months. `TODAY_OVERRIDE` pins the date for repeatable runs.
- The sample data is 37 generated documents. One meeting note contains a planted injection, and one document is `restricted`.
- One API replica. Rate limits, session memory and the LangGraph checkpointer live in the process.
- An offline mock LLM exists so tests and demos run without keys.

## Main decisions

| Decision | Reason | Cost |
|---|---|---|
| Fixed LangGraph topology, plan-driven path | Easy to test, trace and bound | Less free-form than a ReAct loop |
| Dense and sparse queried separately, fused in code | Separate scores show why a chunk ranked | Two queries instead of one |
| Own BM25 encoder | No downloads, deterministic, works with a `dotproduct` index | English only, simple tokenizer |
| Namespace per access tier | Access is enforced by where we search | Re-tiering means re-ingesting |
| Heuristic reranker behind a `rerank()` interface | Free and explainable | A cross-encoder would rank better |
| Stream a draft, then replace it with the validated answer | Keeps streaming responsive | The user may briefly see a draft that gets corrected |
| Citations appended by code | Sources come from verified evidence, never from the model | None |
| Server-Sent Events | One-way, simple, works with Streamlit | WebSockets not needed |

## Model choice

All chat calls use `gpt-4.1-mini` on Azure OpenAI. Embeddings use `text-embedding-3-small` (1536 dimensions). The mini model has solid JSON output and instruction following at low cost
and latency, and it is what the development account has quota for. The code has a `fast` and a `strong` tier, so production can use a cheaper model for guards and routing and a larger one for planning and answers. Here both tiers point at one deployment.
A small model can write weak research plans, which is why plans are validated and a default plan exists. OpenAI and any OpenAI-compatible endpoint also work.

## Memory

- Working memory: the last 4 exchanges, kept verbatim for follow-up questions.
- Summary: older turns are compressed into a rolling summary after 8 exchanges to keep prompts small.
- Long-term (SQLite): per-user past answers with the documents they cited, and stated preferences.

Recall ranks by keyword overlap, recency (30-day half-life) and a feedback weight, so poorly rated answers are not reused. Each stored answer keeps the access level of its evidence and is filtered by the
user's current role. Degraded or uncited answers are never stored. Memory is passed to the model inside `<memory>` tags and treated as untrusted text. Sessions are owned by their creator.
The limits: session state is in-process, and recall uses keywords rather than embeddings.

## Failure handling

Every dependency has a timeout, jittered retries and a circuit breaker. When one fails, the system degrades and says so:

| Failure | Result |
|---|---|
| LLM down | rule-based routing and an extractive answer |
| Embeddings down | sparse-only retrieval |
| Pinecone down | local BM25 fallback |
| MCP down | tool marked unavailable, answer discloses it |
| Tool timeout | status `timeout`, other steps continue |
| Invalid request | 422 with field-level errors |
| Bad citation | regenerate, or strip on the last attempt |
| Rate limit | 429 with `Retry-After` |

A crashed node does not crash the run. The response agent is told which capability failed and must mention it. If the validator itself fails, the answer is replaced with a safe message
and never returned unchecked. Offline tests cover each case.

## Issues found and fixed

- A crash in the response node made the response and validator nodes loop until LangGraph's recursion limit. Failed attempts now count, and the validator fails closed.
- `mcp` 2.x renamed `FastMCP`, which broke the import. The dependency is pinned to `mcp<2`.
- Azure v1 endpoints (`.../openai/v1`) returned 404 because the Azure client added its own deployment path. Those endpoints now use the plain OpenAI client.
- Large Pinecone upserts timed out on a slow WSL network. Upserts now go in small batches with retries.
- The mock LLM crashed when the research summary was missing. It now handles that case.

## Limitations and next steps

- The injection detector is pattern-based and can be evaded. The controls that matter most do not depend on it (see [security.md](security.md)).
- The reranker is heuristic, and the analysis sandbox is a subprocess, not a container.
- Docker Compose is provided but not covered by tests.
- Next: Redis for rate limits and sessions, a Postgres checkpointer, an OIDC provider, a cross-encoder reranker, and an evaluation set built from the feedback table.
