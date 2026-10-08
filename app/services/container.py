"""Composition root: builds and wires every component once at startup."""
from __future__ import annotations

from app.agents.deps import AgentDeps
from app.config import Settings
from app.db import Database
from app.ingestion.ingest import build_index
from app.llm.service import LLMService
from app.logging_config import get_logger
from app.memory.manager import MemoryManager
from app.retrieval.loader import IndexMissing, load_retrieval_stack
from app.security.audit import AuditLog
from app.security.guardrails import BrandGuard, OutputGuard
from app.security.rate_limit import RateLimiter
from app.services.orchestrator import Orchestrator
from app.tools.builtin import build_registry
from app.tools.mcp_client import MCPClient

log = get_logger("container")


class Container:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.db_path)
        self.audit = AuditLog(self.db)
        self.llm = LLMService(settings)
        self.memory = MemoryManager(settings, self.db, self.llm)
        self.mcp = MCPClient(settings)
        self.rate_limiter = RateLimiter(settings.rate_limits, name="chat")
        self.retriever = self.catalog = self.index_meta = None
        self.registry = build_registry(settings, retriever_getter=lambda: self.retriever, mcp=self.mcp, audit=self.audit, reindex=self.reindex)
        self.deps = AgentDeps(settings, self.llm, self.registry, self.memory, self.audit, lambda: self.catalog,
                              OutputGuard(settings.canary_token), BrandGuard())
        self.orchestrator = Orchestrator(self.deps, settings.resolved_vector_backend)

    @classmethod
    async def create(cls, settings: Settings) -> "Container":
        c = cls(settings)
        try:
            c.retriever, c.catalog, c.index_meta = load_retrieval_stack(settings, c.llm)
        except IndexMissing as exc:
            log.warning("index_missing_or_stale_building", reason=str(exc))
            await build_index(settings, c.llm)
            c.retriever, c.catalog, c.index_meta = load_retrieval_stack(settings, c.llm)
        log.info("container_ready", llm=c.llm.provider, vector=settings.resolved_vector_backend, chunks=c.index_meta["chunks"])
        return c

    async def reindex(self) -> dict:
        stats = await build_index(self.settings, self.llm)
        self.retriever, self.catalog, self.index_meta = load_retrieval_stack(self.settings, self.llm)
        return {k: stats[k] for k in ("documents", "chunks", "seconds", "remote")}
