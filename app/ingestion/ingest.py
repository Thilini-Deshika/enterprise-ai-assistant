"""Ingestion pipeline:  docs -> chunks -> (dense embeddings + BM25 sparse) -> local artifacts + Pinecone.

    python -m app.ingestion.ingest                # build local index (+ push to Pinecone if configured)
    python -m app.ingestion.ingest --local-only   # never touch Pinecone
Artifacts in data/index/: chunks.jsonl (catalog), vectors.npy, bm25.json, index_meta.json
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time

import numpy as np

from app.config import Settings, get_settings
from app.ingestion.chunker import chunk_document, parse_document
from app.llm.service import LLMService
from app.logging_config import configure_logging, get_logger
from app.retrieval.sparse import BM25SparseEncoder
from app.retrieval.stores import IndexedRecord, PineconeStore

log = get_logger(__name__)


async def build_index(settings: Settings, llm: LLMService, *, push_remote: bool = True) -> dict:
    t0 = time.perf_counter()
    paths = sorted(settings.docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No documents in {settings.docs_dir}. Run: python scripts/generate_sample_data.py")
    chunks = []
    for p in paths:
        meta, body = parse_document(p)
        chunks.extend(chunk_document(meta, body))
    log.info("chunked", documents=len(paths), chunks=len(chunks))

    embed_inputs = [f"{c.title} | {c.section}\n{c.text}" for c in chunks]
    encoder = BM25SparseEncoder().fit(embed_inputs)
    sem = asyncio.Semaphore(3)

    async def _batch(texts):
        async with sem:
            return await llm.embed(texts)

    batches = [embed_inputs[i:i + 64] for i in range(0, len(embed_inputs), 64)]
    dense = [v for part in await asyncio.gather(*[_batch(b) for b in batches]) for v in part]

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(dataclasses.asdict(c)) for c in chunks), encoding="utf-8")
    np.save(out / "vectors.npy", np.array(dense, dtype=np.float32))
    encoder.save(out / "bm25.json")
    meta = {"provider": llm.provider, "embedding_model": settings.embedding_model if llm.provider != "mock" else "hash-embed",
            "dim": len(dense[0]), "chunks": len(chunks), "documents": len(paths), "built_at": time.time()}
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2))

    remote = None
    if push_remote and settings.resolved_vector_backend == "pinecone":
        store = PineconeStore(settings, dim=len(dense[0]))
        await asyncio.to_thread(store.ensure_index)
        records = [IndexedRecord(c.chunk_id, c.namespace, dense[i], encoder.encode_document(embed_inputs[i]), c.metadata)
                   for i, c in enumerate(chunks)]
        await store.upsert(records)
        remote = {"index": settings.pinecone_index, "namespaces": sorted({c.namespace for c in chunks}), "vectors": len(records)}
        log.info("pinecone_upserted", **remote)
    stats = {**meta, "remote": remote, "seconds": round(time.perf_counter() - t0, 1)}
    log.info("ingestion_complete", **{k: v for k, v in stats.items() if k != "remote"})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge index")
    parser.add_argument("--local-only", action="store_true", help="skip Pinecone upsert")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    stats = asyncio.run(build_index(settings, LLMService(settings), push_remote=not args.local_only))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
