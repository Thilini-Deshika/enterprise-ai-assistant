"""Concrete tools + registry factory. Each tool = (pydantic args schema, permission, handler)."""
from __future__ import annotations

from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import ToolError
from app.events import emit
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import HybridRetriever
from app.security import injection
from app.security.audit import AuditLog
from app.security.rate_limit import RateLimiter
from app.security.rbac import Permission
from app.tools.base import ToolContext, ToolRegistry, ToolSpec
from app.tools.mcp_client import MCPClient
from app.tools.sandbox import run_analysis


# ---------------------------------------------------------------------------------- argument schemas
class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    department: list[str] | None = None
    document_type: list[str] | None = None
    date_from: str | None = Field(default=None, description="ISO date")
    date_to: str | None = Field(default=None, description="ISO date")
    service: str | None = None
    top_k: int = Field(default=6, ge=1, le=100)


class PythonAnalysisArgs(BaseModel):
    dataset: Literal["incidents", "evidence"]
    code: str = Field(min_length=3, max_length=3000, description="pandas code; DataFrame `df`; assign final value to `result`")


class SearchEmployeesArgs(BaseModel):
    query: str = Field(min_length=2, max_length=80)
    limit: int = Field(default=5, ge=1, le=20)


class ServiceArgs(BaseModel):
    service: str = Field(pattern=r"^[a-z][a-z0-9\-]{2,40}$")


class NoArgs(BaseModel):
    pass


class ListIncidentsArgs(BaseModel):
    service: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9\-]{2,40}$")
    severity: Literal["sev1", "sev2", "sev3", "sev4"] | None = None
    since: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    limit: int = Field(default=100, ge=1, le=200)


class GetIncidentArgs(BaseModel):
    incident_id: str = Field(pattern=r"^INC-\d{4}-\d{3}$")


class CreateTicketArgs(BaseModel):
    title: str = Field(min_length=5, max_length=200)
    severity: Literal["sev1", "sev2", "sev3", "sev4"] = "sev3"
    service: str = Field(pattern=r"^[a-z][a-z0-9\-]{2,40}$")
    description: str = Field(min_length=5, max_length=1000)


def build_registry(settings: Settings, *, retriever_getter: Callable[[], HybridRetriever], mcp: MCPClient, audit: AuditLog,
                   reindex: Callable[[], Awaitable[dict]]) -> ToolRegistry:
    registry = ToolRegistry(settings, audit, RateLimiter({}, settings.tool_rate_limit, name="tools"))

    async def knowledge_search(args: KnowledgeSearchArgs, ctx: ToolContext) -> dict:
        filters = SearchFilters.sanitize(args.model_dump(exclude={"query", "top_k"}))
        result = await retriever_getter().search(args.query, ctx.user, filters, args.top_k)
        hits = []
        for h in result.hits:  # indirect-injection defence: strip instruction-like sentences from retrieved text
            clean, removed = injection.sanitize_untrusted(h.text)
            if removed:
                emit("guardrail", kind="injection_quarantined", chunk_id=h.chunk_id, doc_id=h.doc_id, removed=removed[:2])
                await audit.record(user_id=ctx.user.user_id, role=ctx.user.role.value, action="retrieval", outcome="injection_quarantined", resource=h.chunk_id)
                h.text = clean
            hits.append(h.to_dict())
        return {"hits": hits, "diagnostics": result.diagnostics}

    async def python_analysis(args: PythonAnalysisArgs, ctx: ToolContext) -> dict:
        emit("tool_call", id="code", tool="python_analysis", phase="code", code=args.code)
        return await run_analysis(args.code, ctx.datasets.get(args.dataset, []))

    def mcp_tool(mcp_name: str):
        async def handler(args: BaseModel, ctx: ToolContext) -> dict:
            payload = {k: v for k, v in args.model_dump().items() if v is not None}
            data = await mcp.call(mcp_name, payload)
            if isinstance(data, dict) and data.get("error"):
                raise ToolError(data["error"])
            return data
        return handler

    async def admin_reindex(args: NoArgs, ctx: ToolContext) -> dict:
        return await reindex()

    T = ToolSpec
    registry.register(T("knowledge_search", "Hybrid (dense+BM25) search over indexed documents the user may read.",
                        KnowledgeSearchArgs, Permission.SEARCH, knowledge_search, settings.retrieval_timeout_s + 10, untrusted_output=False))
    registry.register(T("python_analysis", "Run pandas code on a structured dataset ('incidents' from mcp_list_incidents, or 'evidence'). Put the answer in `result`.",
                        PythonAnalysisArgs, Permission.ANALYTICS, python_analysis, 15))
    registry.register(T("mcp_search_employees", "Employee directory search by name/title/department.", SearchEmployeesArgs,
                        Permission.MCP_READ, mcp_tool("search_employees"), settings.tool_timeout_s, untrusted_output=True))
    registry.register(T("mcp_list_services", "List the service catalog (tier, owning team).", NoArgs, Permission.MCP_READ,
                        mcp_tool("list_services"), settings.tool_timeout_s, untrusted_output=True))
    registry.register(T("mcp_get_service", "Service record: tier, owner, on-call engineer, runbook, dependencies.", ServiceArgs,
                        Permission.MCP_READ, mcp_tool("get_service"), settings.tool_timeout_s, untrusted_output=True))
    registry.register(T("mcp_list_incidents", "Structured incident records filtered by service/severity/since (ISO date).", ListIncidentsArgs,
                        Permission.MCP_READ, mcp_tool("list_incidents"), settings.tool_timeout_s, untrusted_output=True))
    registry.register(T("mcp_get_incident", "One incident record by id.", GetIncidentArgs, Permission.MCP_READ,
                        mcp_tool("get_incident"), settings.tool_timeout_s, untrusted_output=True))
    registry.register(T("mcp_create_incident_ticket", "WRITE: create an incident ticket.", CreateTicketArgs, Permission.MCP_WRITE,
                        mcp_tool("create_incident_ticket"), settings.tool_timeout_s, requires_approval=True, untrusted_output=True))
    registry.register(T("admin_reindex", "ADMIN: rebuild the knowledge index from source documents.", NoArgs, Permission.ADMIN_TOOLS,
                        admin_reindex, 120, requires_approval=True))
    return registry
