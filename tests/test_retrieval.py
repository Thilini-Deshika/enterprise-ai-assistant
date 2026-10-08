import pytest

from app.config import get_settings
from app.errors import VectorStoreUnavailable
from app.retrieval.filters import SearchFilters
from app.retrieval.loader import load_retrieval_stack
from app.retrieval.sparse import BM25SparseEncoder
from app.retrieval.stores import VectorStore


def test_bm25_prefers_rare_matching_terms():
    enc = BM25SparseEncoder().fit(["payment gateway pool", "kafka consumer lag", "tls certificate renewal"])
    q = enc.encode_query("kafka lag")
    d_match, d_other = enc.encode_document("kafka consumer lag spike"), enc.encode_document("payment gateway pool")
    score = lambda d: sum(w * dict(zip(d["indices"], d["values"])).get(i, 0) for i, w in zip(q["indices"], q["values"]))
    assert score(d_match) > score(d_other) == 0


@pytest.fixture(scope="module")
def stack(container):
    return container.retriever, container.catalog


async def test_namespace_rbac_viewer_never_sees_restricted(stack, users):
    retriever, _ = stack
    res = await retriever.search("security audit findings stale privileged accounts", users["viewer"], SearchFilters(), 10)
    assert all(h.metadata["access_level"] in ("public", "internal") for h in res.hits)
    assert "SEC-AUD-2026" not in {h.doc_id for h in res.hits}
    admin = await retriever.search("security audit findings stale privileged accounts", users["admin"], SearchFilters(), 10)
    assert "SEC-AUD-2026" in {h.doc_id for h in admin.hits}


async def test_metadata_filters_and_scores_exposed(stack, users):
    retriever, _ = stack
    f = SearchFilters(document_type=["incident"], date_from="2026-06-01", date_to="2026-12-31")
    res = await retriever.search("payment gateway outage", users["analyst"], f, 10)
    assert res.hits and all(h.metadata["document_type"] == "incident" and h.metadata["created_date"] >= "2026-06-01" for h in res.hits)
    assert any(h.dense_score > 0 for h in res.hits) and any(h.sparse_score > 0 for h in res.hits)


async def test_vector_db_failure_degrades_to_local_bm25(container, users):
    class Broken(VectorStore):
        name = "pinecone"
        async def upsert(self, records): ...
        async def query(self, **kw): raise VectorStoreUnavailable("simulated outage")
        async def healthy(self): return False
    retriever = container.retriever
    original = retriever.store
    retriever.store = Broken()
    try:
        res = await retriever.search("password policy", users["viewer"], SearchFilters(), 5)
    finally:
        retriever.store = original
    assert res.diagnostics["degraded"] and res.hits and "fallback" in res.diagnostics["backend"]


def test_catalog_is_rbac_filtered(stack, users):
    _, catalog = stack
    assert catalog.get_chunks(users["viewer"], "SEC-AUD-2026") == []
    assert catalog.get_chunks(users["admin"], "SEC-AUD-2026")
    n_view = len(catalog.list_docs(users["viewer"], SearchFilters(document_type=["incident"])))
    n_analyst = len(catalog.list_docs(users["analyst"], SearchFilters(document_type=["incident"])))
    assert n_analyst > n_view
