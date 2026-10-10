"""Vector store abstraction with two implementations:

  PineconeStore     - production path. One *hybrid* serverless index (metric=dotproduct) holding dense +
                      sparse values per record. Namespace = access tier (RBAC isolation).
  LocalVectorStore  - same maths in numpy; used when no Pinecone key is configured AND as the
                      degraded-mode fallback when Pinecone is unreachable.
Both accept already alpha-scaled vectors, so "dense only", "sparse only" and "native hybrid" are the
same call with different arguments.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.errors import VectorStoreUnavailable
from app.logging_config import get_logger
from app.resilience import CircuitBreaker, retry_async
from app.retrieval.filters import match_filter

log = get_logger(__name__)


@dataclass
class IndexedRecord:
    id: str
    namespace: str
    dense: list[float]
    sparse: dict
    metadata: dict


@dataclass
class RawMatch:
    id: str
    score: float
    metadata: dict


class VectorStore(ABC):
    name = "abstract"

    @abstractmethod
    async def upsert(self, records: list[IndexedRecord]) -> None: ...

    @abstractmethod
    async def query(self, *, dense: list[float] | None, sparse: dict | None, namespaces: list[str],
                    flt: dict, top_k: int) -> list[RawMatch]: ...

    @abstractmethod
    async def healthy(self) -> bool: ...


class LocalVectorStore(VectorStore):
    name = "local"

    def __init__(self, records: list[IndexedRecord], dim: int):
        self.dim = dim
        self.ids = [r.id for r in records]
        self.ns = [r.namespace for r in records]
        self.meta = [r.metadata for r in records]
        self.sparse = [dict(zip(r.sparse["indices"], r.sparse["values"])) for r in records]
        self.dense = np.array([r.dense for r in records], dtype=np.float32) if records else np.zeros((0, dim), np.float32)

    async def upsert(self, records: list[IndexedRecord]) -> None:  # local store is rebuilt from artifacts
        raise NotImplementedError("LocalVectorStore is read-only; rebuild via ingestion")

    async def query(self, *, dense, sparse, namespaces, flt, top_k) -> list[RawMatch]:
        if not self.ids:
            return []
        mask = np.array([self.ns[i] in namespaces and match_filter(self.meta[i], flt) for i in range(len(self.ids))])
        if not mask.any():
            return []
        scores = np.zeros(len(self.ids), dtype=np.float32)
        if dense is not None and self.dense.shape[1] == len(dense):
            scores += self.dense @ np.array(dense, dtype=np.float32)
        if sparse and sparse["indices"]:
            q = dict(zip(sparse["indices"], sparse["values"]))
            for i, doc in enumerate(self.sparse):
                if mask[i]:
                    scores[i] += sum(w * doc.get(t, 0.0) for t, w in q.items())
        order = np.argsort(-scores)
        out = []
        for i in order:
            if mask[i] and scores[i] > 0:
                out.append(RawMatch(self.ids[i], float(scores[i]), self.meta[i]))
                if len(out) >= top_k:
                    break
        return out

    async def healthy(self) -> bool:
        return True


def _field(obj, name, default=None):
    """Read a field from a Pinecone SDK response regardless of SDK version (object attrs or dict-like)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, None)
    if val is None:
        try:
            val = obj[name]
        except Exception:  # noqa: BLE001
            val = default
    return default if val is None else val


class PineconeStore(VectorStore):
    name = "pinecone"

    def __init__(self, settings: Settings, dim: int):
        from pinecone import Pinecone
        self.s, self.dim = settings, dim
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = None
        self.breaker = CircuitBreaker("pinecone", failure_threshold=3, reset_after_s=30)

    # -- admin ---------------------------------------------------------------------------
    def ensure_index(self) -> None:
        """Create the hybrid-capable serverless index if missing (idempotent, blocking)."""
        from pinecone import ServerlessSpec
        if not self.pc.has_index(self.s.pinecone_index):
            log.info("pinecone_create_index", name=self.s.pinecone_index, dim=self.dim)
            self.pc.create_index(name=self.s.pinecone_index, dimension=self.dim, metric="dotproduct",
                                 spec=ServerlessSpec(cloud=self.s.pinecone_cloud, region=self.s.pinecone_region))
        else:
            existing = self.pc.describe_index(self.s.pinecone_index)
            if int(_field(existing, "dimension", self.dim)) != self.dim:
                raise VectorStoreUnavailable(f"Pinecone index dimension {_field(existing, 'dimension')} != embedding dim {self.dim}; "
                                             "delete the index or change PINECONE_INDEX")

    def _idx(self):
        if self._index is None:
            host = _field(self.pc.describe_index(self.s.pinecone_index), "host")
            self._index = self.pc.Index(host=host)
        return self._index

    # -- data plane ----------------------------------------------------------------------
    async def upsert(self, records: list[IndexedRecord]) -> None:
        by_ns: dict[str, list[dict]] = {}
        for r in records:
            by_ns.setdefault(r.namespace, []).append({
                "id": r.id, "values": r.dense, "sparse_values": r.sparse,
                "metadata": {k: v for k, v in r.metadata.items() if v is not None}})

        batch = max(1, self.s.pinecone_upsert_batch)

        def _do():
            import time
            idx = self._idx()
            for ns, vecs in by_ns.items():
                for i in range(0, len(vecs), batch):
                    chunk = vecs[i:i + batch]
                    for attempt in range(1, 5):  # slow or flaky uplinks (VPN, WSL) time out on large bodies
                        try:
                            idx.upsert(vectors=chunk, namespace=ns, timeout=self.s.pinecone_upsert_timeout_s)
                            break
                        except Exception as exc:  # noqa: BLE001
                            if attempt == 4:
                                raise
                            log.warning("pinecone_upsert_retry", namespace=ns, attempt=attempt, error=str(exc)[:120])
                            time.sleep(1.5 * attempt)
                log.info("pinecone_upserted", namespace=ns, vectors=len(vecs))
        await asyncio.to_thread(_do)

    async def query(self, *, dense, sparse, namespaces, flt, top_k) -> list[RawMatch]:
        vector = dense if dense is not None else [0.0] * self.dim  # sparse-only query on a dotproduct index
        sparse_arg = sparse if sparse and sparse["indices"] else None

        async def _one(ns: str) -> list[RawMatch]:
            def _q():
                return self._idx().query(vector=vector, sparse_vector=sparse_arg, top_k=top_k, filter=flt,
                                         namespace=ns, include_metadata=True)
            res = await asyncio.wait_for(asyncio.to_thread(_q), timeout=self.s.retrieval_timeout_s)
            return [RawMatch(_field(m, "id"), float(_field(m, "score", 0.0)), dict(_field(m, "metadata", {}) or {}))
                    for m in (_field(res, "matches", []) or [])]

        async def _all():
            parts = await asyncio.gather(*[_one(ns) for ns in namespaces])
            return sorted((m for p in parts for m in p), key=lambda m: -m.score)[:top_k]

        try:
            return await self.breaker.call(lambda: retry_async(_all, attempts=2, base_delay=0.3, op="pinecone.query"))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"Pinecone query failed: {exc}") from exc

    async def healthy(self) -> bool:
        try:
            await asyncio.wait_for(asyncio.to_thread(lambda: self._idx().describe_index_stats()), timeout=5)
            return True
        except Exception:  # noqa: BLE001
            return False
