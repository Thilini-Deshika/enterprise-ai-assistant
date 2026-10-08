"""Markdown + YAML front-matter parsing and section-aware chunking.

Chunks follow the document's own structure (one chunk per heading section, long sections split on
paragraph boundaries). That keeps passages semantically coherent AND makes "read only the Root Cause
section of 20 incident reports" possible for the RLM agent - sections are addressable units.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import yaml

from app.retrieval.filters import to_ordinal
from app.retrieval.models import ChunkRecord

_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)
_META_FIELDS = ("department", "document_type", "access_level", "created_date", "service", "severity",
                "root_cause_category", "duration_minutes")


def parse_document(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise ValueError(f"{path.name}: missing YAML front-matter")
    meta = yaml.safe_load(m.group(1)) or {}
    for k, v in list(meta.items()):
        if isinstance(v, (date, datetime)):
            meta[k] = v.isoformat()[:10]
    for required in ("doc_id", "title", "department", "document_type", "access_level", "created_date"):
        if required not in meta:
            raise ValueError(f"{path.name}: front-matter missing '{required}'")
    return meta, m.group(2)


def _sections(body: str) -> list[tuple[str, str]]:
    marks = list(_HEADING.finditer(body))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        text = body[m.end():end].strip()
        if text:
            out.append((m.group(2), text))
    return out or [("Overview", body.strip())]


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|(?<=[.!?])\s+", text):
        if cur and len(cur) + len(para) + 1 > max_chars:
            parts.append(cur.strip())
            cur = cur[-100:] + " "  # small overlap preserves context across the boundary
        cur += para + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_document(meta: dict, body: str, max_chars: int = 900) -> list[ChunkRecord]:
    base = {k: meta[k] for k in _META_FIELDS if k in meta}
    base["created_ts"] = to_ordinal(meta["created_date"])
    chunks, n = [], 0
    for heading, text in _sections(body):
        for piece in _split_long(text, max_chars):
            n += 1
            cid = f"{meta['doc_id']}#{n}"
            md = {**base, "doc_id": meta["doc_id"], "title": meta["title"], "section": heading, "chunk_id": cid, "text": piece}
            chunks.append(ChunkRecord(cid, meta["doc_id"], meta["title"], heading, piece, md))
    return chunks
