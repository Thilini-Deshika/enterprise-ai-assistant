from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    """One indexed passage. `metadata` is exactly what is stored next to the vector in Pinecone."""
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict

    @property
    def namespace(self) -> str:
        return self.metadata["access_level"]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    sources: list[str] = field(default_factory=list)  # which retrievers found it: dense / sparse / native

    def to_dict(self, *, with_text: bool = True) -> dict:
        d = {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "title": self.title, "section": self.section,
             "metadata": {k: v for k, v in self.metadata.items() if k != "text"},
             "dense": round(self.dense_score, 4), "sparse": round(self.sparse_score, 4),
             "fused": round(self.fused_score, 4), "rerank": round(self.rerank_score, 4), "found_by": self.sources}
        if with_text:
            d["text"] = self.text
        return d
