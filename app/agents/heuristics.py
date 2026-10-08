"""Deterministic, rule-based fallbacks.

Used (a) by the offline mock LLM so the whole system runs without API keys, and (b) as the graceful
degradation path when the real LLM is down: routing and tool selection keep working, only the
natural-language quality drops. Everything here is explainable and unit-tested.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from app.domain import DEPARTMENTS, DOC_TYPES, SERVICES

INTENTS = ("chitchat", "knowledge_qa", "deep_research", "analytics", "directory_lookup", "admin_action")

_CHITCHAT = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|who are you|what can you do|help)\b[^?]{0,40}\??\s*$", re.I)
_ADMIN = re.compile(r"\b(create|open|raise|file|log)\b.{0,25}\b(ticket|incident record)\b|\bre-?index\b|\brebuild\b.{0,15}\bindex\b", re.I)
_DIRECTORY = re.compile(r"\bwho (is|are|owns|owned|manages|leads|runs)\b|\bon-?call\b|\bowner of\b|\bcontact\b|\bemail (of|for)\b|\bwhich team\b|\bservice (tier|owner|catalog)\b|\bemployee\b", re.I)
_ANALYTICS = re.compile(r"\bhow many\b|\bcount\b|\baverage\b|\bmean\b|\bmedian\b|\bdistribution\b|\bbreakdown\b|\bstatistics\b|\bmttr\b|\brank\b|\btop \d+\b|\bby (service|severity|root cause)\b|\bpercent(age)?\b", re.I)
_DEEP = re.compile(r"\bsummari[sz]e all\b|\ball\b.{0,40}\b(reports?|incidents?|outages?|postmortems?)\b|\brecurring\b|\broot causes?\b|\btrends?\b|\bacross (all|the)\b|\b(last|past) (year|\d+ months?)\b|\bcommon (causes|themes)\b|\bpatterns?\b", re.I)

_DEPT_KEYWORDS = {
    "payments": r"\bpayments?\b", "security": r"\b(security|cyber|fraud)\b", "customer-support": r"\bcustomer support\b",
    "risk-compliance": r"\b(compliance|regulat\w+|risk)\b", "retail-banking": r"\bretail\b",
    "platform-engineering": r"\bplatform\b", "data-engineering": r"\bdata engineering\b",
}
_TYPE_KEYWORDS = {
    "incident": r"\b(incidents?|outages?|post-?mortems?)\b", "runbook": r"\b(runbooks?|playbooks?)\b",
    "policy": r"\b(polic(y|ies)|standards?)\b", "architecture": r"\b(architecture|design doc)\b",
    "product_spec": r"\b(specs?|specifications?)\b", "meeting_notes": r"\b(meetings?|minutes)\b",
}


def classify_intent(question: str) -> str:
    q = question.strip()
    if _CHITCHAT.match(q):
        return "chitchat"
    if _ADMIN.search(q):
        return "admin_action"
    if _DIRECTORY.search(q):
        return "directory_lookup"
    if _DEEP.search(q) and not re.search(r"\bhow many\b|\bcount\b", q, re.I):
        return "deep_research"
    if _ANALYTICS.search(q):
        return "analytics"
    return "knowledge_qa"


def _date_range(q: str, today: date) -> tuple[str | None, str | None]:
    ql = q.lower()
    if re.search(r"\b(last|past|previous) (year|12 months)\b", ql):
        return (today - timedelta(days=365)).isoformat(), today.isoformat()
    if m := re.search(r"\b(last|past) (\d+) (day|week|month)s?\b", ql):
        n, unit = int(m.group(2)), m.group(3)
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        return (today - timedelta(days=days)).isoformat(), today.isoformat()
    if re.search(r"\blast quarter\b", ql):
        return (today - timedelta(days=90)).isoformat(), today.isoformat()
    if re.search(r"\blast month\b", ql):
        return (today - timedelta(days=30)).isoformat(), today.isoformat()
    if re.search(r"\bthis year\b", ql):
        return date(today.year, 1, 1).isoformat(), today.isoformat()
    if m := re.search(r"(?<![\w-])(20\d\d)(?![\w-])", q):
        y = int(m.group(1))
        return date(y, 1, 1).isoformat(), date(y, 12, 31).isoformat()
    return None, None


def extract_filters(question: str, today: date) -> dict:
    """Metadata filters implied by the question text (department, doc type, date window)."""
    ql = question.lower()
    dept = [d for d, pat in _DEPT_KEYWORDS.items() if re.search(pat, ql)]
    types = [t for t, pat in _TYPE_KEYWORDS.items() if re.search(pat, ql)]
    date_from, date_to = _date_range(question, today)
    service = next((s for s in SERVICES if s in ql), None)
    return {"department": dept or None, "document_type": types or None, "date_from": date_from,
            "date_to": date_to, "service": service}


def default_plan(intent: str) -> list[dict]:
    return {
        "chitchat": [],
        "knowledge_qa": [{"agent": "retrieval", "goal": "find supporting passages"}],
        "deep_research": [{"agent": "research", "goal": "recursive multi-document investigation"}],
        "analytics": [{"agent": "tools", "goal": "fetch structured incident data and analyse it"}],
        "directory_lookup": [{"agent": "tools", "goal": "look up directory / service catalog data"}],
        "admin_action": [{"agent": "tools", "goal": "perform the requested administrative action"}],
    }[intent]


_ANALYSIS_TEMPLATES = {
    "service": 'result = df.groupby("service").agg(incidents=("incident_id", "count"), avg_minutes=("duration_minutes", "mean")).sort_values("incidents", ascending=False).round(1).reset_index().to_dict("records")',
    "severity": 'result = df.groupby("severity").agg(incidents=("incident_id", "count"), avg_minutes=("duration_minutes", "mean")).round(1).reset_index().to_dict("records")',
    "root_cause": 'result = df.groupby("root_cause_category").agg(incidents=("incident_id", "count"), avg_minutes=("duration_minutes", "mean")).sort_values("incidents", ascending=False).round(1).reset_index().to_dict("records")',
    "mttr": 'result = {"incidents": int(len(df)), "mean_minutes": round(float(df["duration_minutes"].mean()), 1), "median_minutes": round(float(df["duration_minutes"].median()), 1), "max_minutes": int(df["duration_minutes"].max())}',
}


def analysis_code(question: str) -> str:
    q = question.lower()
    if re.search(r"\bmttr\b|average duration|mean time|how long", q):
        return _ANALYSIS_TEMPLATES["mttr"]
    if re.search(r"by severity|severity", q):
        return _ANALYSIS_TEMPLATES["severity"]
    if re.search(r"by service|per service|which service", q):
        return _ANALYSIS_TEMPLATES["service"]
    return _ANALYSIS_TEMPLATES["root_cause"]


def pick_tool_calls(question: str, intent: str, available: set[str], prior: list[dict], filters: dict) -> dict:
    """Rule-based tool selection. `prior` = tool results so far (to decide when we're done)."""
    done_tools = {r["tool"] for r in prior if r.get("status") == "ok"}
    q = question.lower()
    calls: list[dict] = []
    if intent == "admin_action":
        if re.search(r"re-?index|rebuild", q):
            calls = [{"tool": "admin_reindex", "args": {}}]
        else:
            svc = filters.get("service") or "payment-gateway"
            calls = [{"tool": "mcp_create_incident_ticket", "args": {
                "title": question.strip()[:120], "severity": "sev3", "service": svc, "description": question.strip()[:400]}}]
        return {"calls": [] if {c["tool"] for c in calls} <= done_tools or prior else calls, "done": bool(prior)}
    if intent == "directory_lookup":
        if done_tools:
            return {"calls": [], "done": True}
        svc = filters.get("service")
        if svc or re.search(r"on-?call|owner|owns|tier|service", q):
            return {"calls": [{"tool": "mcp_get_service", "args": {"service": svc or "payment-gateway"}}], "done": False}
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", question)
        names = [n for n in names if n.lower() not in {"who", "what", "which", "the", "is"}]
        return {"calls": [{"tool": "mcp_search_employees", "args": {"query": names[-1] if names else question[-40:]}}], "done": False}
    if intent == "analytics":
        if "mcp_list_incidents" not in done_tools:
            args = {k: v for k, v in {"service": filters.get("service"), "since": filters.get("date_from")}.items() if v}
            return {"calls": [{"tool": "mcp_list_incidents", "args": args}], "done": False}
        if "python_analysis" not in done_tools:
            return {"calls": [{"tool": "python_analysis", "args": {"dataset": "incidents", "code": analysis_code(question)}}], "done": False}
        return {"calls": [], "done": True}
    return {"calls": [], "done": True}
