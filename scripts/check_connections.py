"""Verify that .env credentials work for every external service.

Usage (from the repo root, venv active):
    python -m scripts.check_connections

Checks: LLM chat (plain + JSON mode), embeddings (and dimension), Pinecone (index, dimension, namespaces),
LangSmith (auth + a test trace). Exit code is 0 only if every configured service passes.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

from app.config import get_settings

OK, BAD, SKIP = "[ OK ]", "[FAIL]", "[SKIP]"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"{status} {name}" + (f"  -  {detail}" if detail else ""), flush=True)


async def check_llm(s) -> int | None:
    from app.llm.service import LLMService
    llm = LLMService(s)
    if llm.provider == "mock":
        record("LLM", SKIP, "provider resolved to mock (no OpenAI/Azure key found in .env)")
        return None
    label = f"{llm.provider}"
    for tier in ("fast", "strong"):
        name = f"LLM chat [{tier}] ({llm._model_name(tier)})"
        try:
            t0 = time.time()
            out = await llm.complete("Reply with the single word: pong", "ping", task="connection_check", tier=tier)
            record(name, OK, f"{label}, {time.time() - t0:.1f}s, reply={out.strip()[:30]!r}")
        except Exception as exc:  # noqa: BLE001
            record(name, BAD, str(exc)[:300])
    try:
        data = await llm.json('Return JSON like {"ok": true}.', "go", task="connection_check_json", tier="fast")
        record("LLM JSON mode", OK, f"parsed={data}")
    except Exception as exc:  # noqa: BLE001
        record("LLM JSON mode", BAD, str(exc)[:300])
    try:
        vecs = await llm.embed(["connection check"])
        dim = len(vecs[0])
        if dim == s.effective_embedding_dim:
            record("Embeddings", OK, f"dimension={dim}")
        else:
            record("Embeddings", BAD, f"returned dim {dim} but EMBEDDING_DIM expects {s.effective_embedding_dim}")
        return dim
    except Exception as exc:  # noqa: BLE001
        record("Embeddings", BAD, str(exc)[:300])
        return None


def check_pinecone(s) -> None:
    if not s.pinecone_api_key:
        record("Pinecone", SKIP, "PINECONE_API_KEY not set (app will use the local vector store)")
        return
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=s.pinecone_api_key)
        names = [i["name"] if isinstance(i, dict) else i.name for i in pc.list_indexes()]
        record("Pinecone auth", OK, f"indexes={names}")
        if s.pinecone_index not in names:
            record("Pinecone index", BAD, f"'{s.pinecone_index}' not found; run: python -m app.ingestion.ingest")
            return
        desc = pc.describe_index(s.pinecone_index)
        dim = int(desc["dimension"] if isinstance(desc, dict) else desc.dimension)
        metric = desc["metric"] if isinstance(desc, dict) else desc.metric
        if dim != s.effective_embedding_dim:
            record("Pinecone index", BAD, f"index dim {dim} != EMBEDDING_DIM {s.effective_embedding_dim}; delete the index and re-ingest")
        elif str(metric) != "dotproduct":
            record("Pinecone index", BAD, f"metric is {metric}; hybrid search needs dotproduct")
        else:
            record("Pinecone index", OK, f"{s.pinecone_index}: dim={dim}, metric={metric}")
        stats = pc.Index(s.pinecone_index).describe_index_stats()
        ns = stats["namespaces"] if isinstance(stats, dict) else stats.namespaces
        counts = {k: (v["vector_count"] if isinstance(v, dict) else v.vector_count) for k, v in ns.items()}
        if counts:
            record("Pinecone data", OK, f"namespaces={counts}")
        else:
            record("Pinecone data", BAD, "index is empty; run: python -m app.ingestion.ingest")
    except Exception as exc:  # noqa: BLE001
        record("Pinecone", BAD, str(exc)[:300])


def check_langsmith() -> None:
    key = os.environ.get("LANGSMITH_API_KEY", "")
    if not key:
        record("LangSmith", SKIP, "LANGSMITH_API_KEY not set")
        return
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        record("LangSmith tracing flag", BAD, "LANGSMITH_TRACING must be true")
    try:
        from langsmith import Client, traceable
        client = Client()
        next(iter(client.list_projects(limit=1)), None)
        record("LangSmith auth", OK, f"endpoint={client.api_url}")

        @traceable(name="connection_check", run_type="chain")
        def ping(x: str) -> str:
            return f"pong:{x}"
        ping("hello")
        client.flush()
        project = os.environ.get("LANGSMITH_PROJECT", "default")
        record("LangSmith trace", OK, f"sent a run named 'connection_check' to project '{project}'; confirm it in the UI")
    except Exception as exc:  # noqa: BLE001
        record("LangSmith", BAD, str(exc)[:300])


async def main() -> int:
    s = get_settings()
    print(f"provider={s.resolved_llm_provider} vector_backend={s.resolved_vector_backend} index={s.pinecone_index}\n")
    await check_llm(s)
    check_pinecone(s)
    check_langsmith()
    failed = [r for r in results if r[1] == BAD]
    print(f"\n{len(results) - len(failed)} passed/skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))