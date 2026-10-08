"""BM25 sparse encoder producing Pinecone-compatible sparse vectors (indices + values).

Why our own instead of pinecone-text: no NLTK/model downloads, deterministic, fully explainable.
  document side: value = BM25 term-frequency saturation  tf*(k1+1) / (tf + k1*(1-b+b*len/avglen))
  query side:    value = IDF(term) normalised to sum to 1
  dot(query, doc) = IDF-weighted BM25 score, so Pinecone's dotproduct == BM25 relevance.
Terms are hashed to uint32 indices (md5), so no vocabulary has to be shipped to the index.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset("a an the and or of to in on for with by is are was were be been this that these those it its as at from which who into than then also not no but if so can could should would will may might do does did how what when where why i you we they he she our your their".split())


def stem(tok: str) -> str:
    for suf in ("ingly", "ing", "edly", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(stem(tok))
        if "-" in tok:  # "payment-gateway" is searchable as both the compound and its parts
            out.extend(stem(p) for p in tok.split("-") if p and p not in _STOP)
    return out


def term_index(term: str) -> int:
    return int.from_bytes(hashlib.md5(term.encode()).digest()[:4], "big")


class BM25SparseEncoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = 0
        self.avg_len = 1.0
        self.df: dict[int, int] = {}

    def fit(self, texts: list[str]) -> "BM25SparseEncoder":
        df: Counter = Counter()
        total = 0
        for t in texts:
            toks = tokenize(t)
            total += len(toks)
            df.update({term_index(x) for x in toks})
        self.n_docs, self.avg_len, self.df = len(texts), (total / max(1, len(texts))) or 1.0, dict(df)
        return self

    def encode_document(self, text: str) -> dict:
        toks = tokenize(text)
        tf = Counter(term_index(t) for t in toks)
        dl = len(toks) or 1
        idx, val = [], []
        for i, f in tf.items():
            idx.append(i)
            val.append(round(f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)), 5))
        return {"indices": idx, "values": val}

    def encode_query(self, text: str) -> dict:
        terms = {term_index(t) for t in tokenize(text)}
        idf = {i: math.log(1 + (self.n_docs - self.df.get(i, 0) + 0.5) / (self.df.get(i, 0) + 0.5)) for i in terms}
        total = sum(idf.values()) or 1.0
        return {"indices": list(idf), "values": [round(v / total, 5) for v in idf.values()]}

    # persistence
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"k1": self.k1, "b": self.b, "n_docs": self.n_docs, "avg_len": self.avg_len,
                                    "df": {str(k): v for k, v in self.df.items()}}))

    @classmethod
    def load(cls, path: Path) -> "BM25SparseEncoder":
        d = json.loads(path.read_text())
        enc = cls(d["k1"], d["b"])
        enc.n_docs, enc.avg_len, enc.df = d["n_docs"], d["avg_len"], {int(k): v for k, v in d["df"].items()}
        return enc
