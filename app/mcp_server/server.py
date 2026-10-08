"""Enterprise-data MCP server (dummy data): employee directory, service catalog, incident records.

    python -m app.mcp_server.server                       # stdio (spawned by the API on demand)
    MCP_SERVER_TRANSPORT=streamable-http python -m app.mcp_server.server   # http://0.0.0.0:8765/mcp
The write tool (create_incident_ticket) is a stub; in the assistant it is admin-only AND needs human approval.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(os.getenv("MCP_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "mcp_data"))
mcp = FastMCP("enterprise-data", host=os.getenv("MCP_HOST", "0.0.0.0"), port=int(os.getenv("MCP_PORT", "8765")))
_ticket_no = itertools.count(1001)


def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / f"{name}.json").read_text())


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@mcp.tool()
def search_employees(query: str, limit: int = 5) -> str:
    """Search the employee directory by name, title or department."""
    q = query.lower()
    hits = [e for e in _load("employees") if q in " ".join((e["name"], e["title"], e["department"])).lower()]
    return _j({"items": hits[: max(1, min(limit, 20))], "count": len(hits)})


@mcp.tool()
def list_services() -> str:
    """List all services in the service catalog with tier and owning team."""
    return _j({"items": [{k: s[k] for k in ("service", "tier", "owner_team")} for s in _load("services")]})


@mcp.tool()
def get_service(service: str) -> str:
    """Get a service record: tier, owner team, current on-call engineer, runbook and dependencies."""
    employees = {e["employee_id"]: e for e in _load("employees")}
    for s in _load("services"):
        if s["service"] == service:
            oncall = employees.get(s["oncall_employee_id"], {})
            return _j({"service": s["service"], "tier": s["tier"], "owner_team": s["owner_team"],
                       "oncall": {"name": oncall.get("name"), "email": oncall.get("email")},
                       "runbook_doc_id": s["runbook_doc_id"] or None, "depends_on": s["depends_on"]})
    return _j({"error": f"service '{service}' not found"})


@mcp.tool()
def list_incidents(service: str | None = None, severity: str | None = None, since: str | None = None, limit: int = 100) -> str:
    """List incident records, optionally filtered by service, severity (sev1..sev4) and ISO start date."""
    items = [i for i in _load("incidents")
             if (not service or i["service"] == service) and (not severity or i["severity"] == severity)
             and (not since or i["started_at"] >= since)]
    return _j({"items": items[: max(1, min(limit, 200))], "count": len(items)})


@mcp.tool()
def get_incident(incident_id: str) -> str:
    """Get one incident record by id (e.g. INC-2026-031)."""
    for i in _load("incidents"):
        if i["incident_id"] == incident_id:
            return _j(i)
    return _j({"error": f"incident '{incident_id}' not found"})


@mcp.tool()
def create_incident_ticket(title: str, severity: str, service: str, description: str) -> str:
    """WRITE: create an incident ticket (dummy - returns a ticket id, stores nothing)."""
    return _j({"ticket_id": f"TCK-{next(_ticket_no)}", "title": title[:200], "severity": severity, "service": service, "status": "open"})


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_SERVER_TRANSPORT", "stdio"))  # "stdio" | "streamable-http"
