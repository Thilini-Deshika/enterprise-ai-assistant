"""Document catalog: the source-of-truth view of what was ingested (chunks.jsonl).

Pinecone answers "what is *semantically* close"; the catalog answers "what exists" - list by metadata,
read a specific section, grep. The RLM research agent uses it to *explore* a collection instead of
stuffing it into a prompt. Every accessor is RBAC-filtered by the caller's allowed access levels.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.retrieval.filters import SearchFilters, match_filter
from app.retrieval.models import ChunkRecord
from app.security.rbac import UserContext

_DOC_FIELDS = ("doc_id", "title", "department", "document_type", "access_level", "created_date", "service",
               "severity", "root_cause_category", "duration_minutes")


class DocumentCatalog:
    def __init__(self, chunks: list[ChunkRecord]):
        self.chunks = chunks
        self._by_doc: dict[str, list[ChunkRecord]] = {}
        for c in chunks:
            self._by_doc.setdefault(c.doc_id, []).append(c)

    @classmethod
    def load(cls, index_dir: Path) -> "DocumentCatalog":
        path = index_dir / "chunks.jsonl"
        if not path.exists():
            return cls([])
        chunks = [ChunkRecord(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        return cls(chunks)

    def _visible(self, user: UserContext, doc_id: str) -> bool:
        chunks = self._by_doc.get(doc_id)
        return bool(chunks) and chunks[0].metadata.get("access_level") in user.allowed_access_levels

    def doc_meta(self, doc_id: str) -> dict:
        m = self._by_doc[doc_id][0].metadata
        return {k: m.get(k) for k in _DOC_FIELDS if m.get(k) is not None}

    def list_docs(self, user: UserContext, filters: SearchFilters, limit: int = 200) -> list[dict]:
        flt = filters.to_pinecone(user.allowed_access_levels)
        docs = [self.doc_meta(d) for d, cs in self._by_doc.items() if match_filter(cs[0].metadata, flt)]
        docs.sort(key=lambda d: (d.get("created_date", ""), d["doc_id"]))
        return docs[:limit]

    def get_chunks(self, user: UserContext, doc_id: str, sections: list[str] | None = None) -> list[ChunkRecord]:
        if not self._visible(user, doc_id):
            return []
        chunks = self._by_doc[doc_id]
        if sections:
            wanted = [s.lower() for s in sections]
            picked = [c for c in chunks if any(w in c.section.lower() for w in wanted)]
            return picked or chunks[:2]
        return chunks

    def grep(self, user: UserContext, term: str, doc_ids: list[str] | None = None, limit: int = 20) -> list[dict]:
        term_l, out = term.lower(), []
        for c in self.chunks:
            if c.metadata.get("access_level") not in user.allowed_access_levels:
                continue
            if doc_ids and c.doc_id not in doc_ids:
                continue
            if term_l in c.text.lower():
                out.append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "section": c.section,
                            "line": next(l for l in c.text.splitlines() if term_l in l.lower())[:200]})
                if len(out) >= limit:
                    break
        return out

    def stats(self, user: UserContext, filters: SearchFilters) -> dict:
        docs = self.list_docs(user, filters, limit=10_000)
        by_type, by_dept, by_year = {}, {}, {}
        for d in docs:
            by_type[d["document_type"]] = by_type.get(d["document_type"], 0) + 1
            by_dept[d["department"]] = by_dept.get(d["department"], 0) + 1
            y = d.get("created_date", "")[:4]
            by_year[y] = by_year.get(y, 0) + 1
        return {"documents": len(docs), "by_type": by_type, "by_department": by_dept, "by_year": by_year}
