"""Load on-disk index artifacts and assemble the runtime retrieval stack."""
from __future__ import annotations

import json

import numpy as np

from app.config import Settings
from app.llm.service import LLMService
from app.logging_config import get_logger
from app.retrieval.catalog import DocumentCatalog
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import ChunkRecord
from app.retrieval.sparse import BM25SparseEncoder
from app.retrieval.stores import IndexedRecord, LocalVectorStore, PineconeStore, VectorStore

log = get_logger(__name__)


class IndexMissing(RuntimeError):
    pass


def load_retrieval_stack(settings: Settings, llm: LLMService) -> tuple[HybridRetriever, DocumentCatalog, dict]:
    d = settings.index_dir
    if not (d / "chunks.jsonl").exists():
        raise IndexMissing("Index not built. Run:  python -m app.ingestion.ingest")
    meta = json.loads((d / "index_meta.json").read_text())
    catalog = DocumentCatalog.load(d)
    encoder = BM25SparseEncoder.load(d / "bm25.json")
    dense = np.load(d / "vectors.npy")
    if meta["dim"] != llm.dim or meta.get("provider") != llm.provider:
        # e.g. index built offline with mock embeddings, now running with OpenAI: vectors are not comparable
        raise IndexMissing(f"Index was built with provider={meta.get('provider')} dim={meta['dim']} but runtime is "
                           f"provider={llm.provider} dim={llm.dim}; rebuilding")
    records = [IndexedRecord(c.chunk_id, c.namespace, dense[i].tolist(), encoder.encode_document(f"{c.title} | {c.section}\n{c.text}"), c.metadata)
               for i, c in enumerate(catalog.chunks)]
    local = LocalVectorStore(records, dim=meta["dim"])
    primary: VectorStore = PineconeStore(settings, dim=meta["dim"]) if settings.resolved_vector_backend == "pinecone" else local
    retriever = HybridRetriever(settings, llm, primary, fallback=local if primary is not local else local, encoder=encoder)
    return retriever, catalog, meta
