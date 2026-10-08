"""Hybrid retriever: dense + sparse retrieval, weighted fusion, rerank, graceful degradation.

Pipeline (each stage emits an event, so the activity panel can show it live):
  1. RBAC scope   - namespaces = access tiers the user may read (server-side, never from the LLM)
  2. embed query  - dense vector (falls back to sparse-only if the embedding API is down)
  3. retrieve     - fusion mode: dense query + sparse query in parallel (explainable per-signal scores)
                    native mode: single Pinecone query with alpha-scaled dense & sparse vectors
  4. fuse         - fused = alpha * minmax(dense) + (1-alpha) * minmax(sparse)
  5. rerank       - lexical coverage + title match + recency, max 3 chunks per document
  6. degrade      - Pinecone down -> local BM25 over the on-disk index, flagged `degraded`
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langsmith import traceable

from app.config import Settings
from app.errors import DependencyUnavailable, VectorStoreUnavailable
from app.events import emit
from app.llm.service import LLMService
from app.logging_config import get_logger
from app.retrieval.filters import SearchFilters
from app.retrieval.models import Hit
from app.retrieval.sparse import BM25SparseEncoder, tokenize
from app.retrieval.stores import LocalVectorStore, RawMatch, VectorStore
from app.security.rbac import UserContext

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    diagnostics: dict = field(default_factory=dict)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _scale(dense: list[float] | None, sparse: dict | None, alpha: float):
    """Pinecone's documented convex-combination trick for single-query hybrid search."""
    d = [v * alpha for v in dense] if dense is not None else None
    s = {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]} if sparse else None
    return d, s


class Reranker:
    """Cheap, deterministic reranker. Interface is `rerank(query, hits) -> hits` so a cross-encoder
    (Cohere Rerank / Pinecone Inference rerank / bge-reranker) can replace it without touching callers."""

    def rerank(self, query: str, hits: list[Hit], top_k: int, per_doc: int = 3) -> list[Hit]:
        q_terms = set(tokenize(query))
        for h in hits:
            body = set(tokenize(h.text))
            title = set(tokenize(h.title + " " + h.section))
            coverage = len(q_terms & body) / len(q_terms) if q_terms else 0.0
            title_match = len(q_terms & title) / len(q_terms) if q_terms else 0.0
            h.rerank_score = 0.6 * h.fused_score + 0.3 * coverage + 0.1 * title_match
        ordered = sorted(hits, key=lambda h: -h.rerank_score)
        out: list[Hit] = []
        per: dict[str, int] = {}
        for h in ordered:
            if per.get(h.doc_id, 0) < per_doc:
                out.append(h)
                per[h.doc_id] = per.get(h.doc_id, 0) + 1
            if len(out) >= top_k:
                break
        return out


class HybridRetriever:
    def __init__(self, settings: Settings, llm: LLMService, store: VectorStore, fallback: LocalVectorStore | None,
                 encoder: BM25SparseEncoder, reranker: Reranker | None = None):
        self.s, self.llm, self.store, self.fallback, self.encoder = settings, llm, store, fallback, encoder
        self.reranker = reranker or Reranker()

    @staticmethod
    def _to_hit(m: RawMatch) -> Hit:
        md = dict(m.metadata)
        return Hit(chunk_id=m.id, doc_id=md.get("doc_id", m.id.split("#")[0]), title=md.get("title", ""),
                   section=md.get("section", ""), text=md.get("text", ""), metadata=md)

    @traceable(name="hybrid_retrieval", run_type="retriever")
    async def search(self, query: str, user: UserContext, filters: SearchFilters, top_k: int | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        top_k = top_k or self.s.retrieval_top_k
        levels = user.allowed_access_levels
        namespaces, flt = list(levels), filters.to_pinecone(levels)
        candidates = max(top_k * 3, 20) if top_k < 30 else top_k * 2
        alpha = self.s.hybrid_alpha
        diag: dict = {"backend": self.store.name, "mode": self.s.hybrid_mode, "alpha": alpha, "namespaces": namespaces,
                      "filter": filters.to_dict(), "degraded": False, "notes": []}
        emit("retrieval", status="started", query=query, **{k: diag[k] for k in ("backend", "mode", "namespaces", "filter")})

        sparse = self.encoder.encode_query(query)
        dense: list[float] | None = None
        try:
            dense = (await self.llm.embed([query]))[0]
        except DependencyUnavailable as exc:
            diag["degraded"] = True
            diag["notes"].append(f"embedding unavailable -> sparse-only ({exc.message[:80]})")
            emit("retrieval", status="degraded", reason="embedding service unavailable; using keyword (BM25) search only")

        dense_m: list[RawMatch] = []
        sparse_m: list[RawMatch] = []
        native_m: list[RawMatch] = []
        store = self.store
        try:
            dense_m, sparse_m, native_m = await self._query(store, dense, sparse, namespaces, flt, candidates, alpha)
        except VectorStoreUnavailable as exc:
            if self.fallback is None:
                raise
            diag.update(degraded=True, backend=f"{self.fallback.name} (fallback)")
            diag["notes"].append(f"{store.name} unavailable -> local BM25 fallback ({exc.message[:80]})")
            emit("retrieval", status="degraded", reason="vector database unavailable; serving keyword results from the local index")
            dense_m, sparse_m, native_m = await self._query(self.fallback, None, sparse, namespaces, flt, candidates, 0.0)

        hits = self._fuse(dense_m, sparse_m, native_m, alpha)
        diag["candidates"] = {"dense": len(dense_m), "sparse": len(sparse_m), "native": len(native_m), "fused": len(hits)}
        hits = self.reranker.rerank(query, hits, top_k)
        diag["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        emit("retrieval", status="completed", diagnostics=diag, hits=[h.to_dict(with_text=False) for h in hits])
        log.info("retrieval_done", query=query[:80], hits=len(hits), **{k: diag[k] for k in ("backend", "degraded", "latency_ms")})
        return RetrievalResult(hits=hits, diagnostics=diag)

    async def _query(self, store: VectorStore, dense, sparse, namespaces, flt, k, alpha):
        import asyncio
        if self.s.hybrid_mode == "native" and dense is not None and store.name != "local":
            d, s = _scale(dense, sparse, alpha)
            return [], [], await store.query(dense=d, sparse=s, namespaces=namespaces, flt=flt, top_k=k)
        jobs = []
        if dense is not None:
            jobs.append(store.query(dense=dense, sparse=None, namespaces=namespaces, flt=flt, top_k=k))
        jobs.append(store.query(dense=None, sparse=sparse, namespaces=namespaces, flt=flt, top_k=k))
        res = await asyncio.gather(*jobs)
        return (res[0], res[1], []) if dense is not None else ([], res[0], [])

    @staticmethod
    def _fuse(dense_m, sparse_m, native_m, alpha: float) -> list[Hit]:
        hits: dict[str, Hit] = {}
        for lst, key, label in ((dense_m, "dense_score", "dense"), (sparse_m, "sparse_score", "sparse"), (native_m, "fused_score", "native")):
            for m in lst:
                h = hits.setdefault(m.id, HybridRetriever._to_hit(m))
                setattr(h, key, m.score)
                h.sources.append(label)
        if native_m:
            return list(hits.values())
        nd = _minmax({h.chunk_id: h.dense_score for h in hits.values() if "dense" in h.sources})
        ns = _minmax({h.chunk_id: h.sparse_score for h in hits.values() if "sparse" in h.sources})
        only_sparse = not dense_m
        for h in hits.values():
            h.fused_score = ns.get(h.chunk_id, 0.0) if only_sparse else alpha * nd.get(h.chunk_id, 0.0) + (1 - alpha) * ns.get(h.chunk_id, 0.0)
        return list(hits.values())
