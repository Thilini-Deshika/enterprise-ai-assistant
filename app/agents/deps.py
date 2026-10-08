from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.config import Settings
from app.llm.service import LLMService
from app.memory.manager import MemoryManager
from app.retrieval.catalog import DocumentCatalog
from app.security.audit import AuditLog
from app.security.guardrails import BrandGuard, OutputGuard
from app.tools.base import ToolRegistry


@dataclass
class AgentDeps:
    settings: Settings
    llm: LLMService
    registry: ToolRegistry
    memory: MemoryManager
    audit: AuditLog
    catalog_getter: Callable[[], DocumentCatalog]
    output_guard: OutputGuard
    brand_guard: BrandGuard
