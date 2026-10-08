"""Metadata filters. Built server-side; LLM-suggested filters are sanitised against known vocabularies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.domain import DEPARTMENTS, DOC_TYPES, SERVICES


def to_ordinal(iso: str | None) -> int | None:
    try:
        return date.fromisoformat(iso).toordinal() if iso else None
    except ValueError:
        return None


@dataclass
class SearchFilters:
    department: list[str] | None = None
    document_type: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    service: str | None = None

    @classmethod
    def sanitize(cls, raw: dict | None) -> "SearchFilters":
        raw = raw or {}

        def _vocab(val, allowed):
            if not val:
                return None
            vals = [val] if isinstance(val, str) else list(val)
            vals = [str(v).strip().lower() for v in vals]
            vals = [v for v in vals if v in allowed]
            return vals or None

        def _date(v):
            return v if isinstance(v, str) and to_ordinal(v) else None

        svc = raw.get("service")
        return cls(_vocab(raw.get("department"), DEPARTMENTS), _vocab(raw.get("document_type"), DOC_TYPES),
                   _date(raw.get("date_from")), _date(raw.get("date_to")),
                   svc if isinstance(svc, str) and svc in SERVICES else None)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    def relaxed(self, level: int) -> "SearchFilters":
        """level 1: drop department/service; level 2: drop everything except the date window."""
        if level <= 0:
            return self
        if level == 1:
            return SearchFilters(None, self.document_type, self.date_from, self.date_to, None)
        return SearchFilters(None, None, self.date_from, self.date_to, None)

    def to_pinecone(self, allowed_levels: tuple[str, ...]) -> dict:
        """Pinecone metadata filter. access_level is ALWAYS present (defence in depth next to namespaces)."""
        flt: dict = {"access_level": {"$in": list(allowed_levels)}}
        if self.department:
            flt["department"] = {"$in": self.department}
        if self.document_type:
            flt["document_type"] = {"$in": self.document_type}
        if self.service:
            flt["service"] = {"$eq": self.service}
        lo, hi = to_ordinal(self.date_from), to_ordinal(self.date_to)
        if lo is not None or hi is not None:
            rng = {}
            if lo is not None:
                rng["$gte"] = lo
            if hi is not None:
                rng["$lte"] = hi
            flt["created_ts"] = rng
        return flt


def match_filter(metadata: dict, flt: dict) -> bool:
    """Evaluate the Pinecone filter subset we use ($eq/$in/$gte/$lte) locally (local store + catalog)."""
    for field, cond in flt.items():
        val = metadata.get(field)
        for op, arg in cond.items():
            if op == "$eq" and val != arg:
                return False
            if op == "$in" and val not in arg:
                return False
            if op == "$gte" and (val is None or val < arg):
                return False
            if op == "$lte" and (val is None or val > arg):
                return False
    return True
