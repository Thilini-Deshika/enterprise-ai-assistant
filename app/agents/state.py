"""LangGraph state. Design rules:

* Each agent writes only to its OWN keys (supervisor: plan/filters, retrieval+research: evidence/research,
  tools: tool_results, response: draft, validator: validation/answer). No agent mutates another's output.
* List/dict keys that several agents may extend use reducers (merge, not overwrite).
* Failures never raise through the graph: the wrapper records `agent_status[node]="failed"` and the
  router moves on, so downstream agents see *what is missing* and adapt (bounded blast radius).
* Everything is JSON-serialisable so the checkpointer can persist it (needed for human-in-the-loop resume).
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def merge_evidence(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    merged: dict[str, dict] = {e["chunk_id"]: e for e in (old or [])}
    for e in new or []:
        cur = merged.get(e["chunk_id"])
        if cur is None or e.get("rerank", 0) > cur.get("rerank", 0):
            merged[e["chunk_id"]] = e
    return list(merged.values())


def merge_dicts(old: dict | None, new: dict | None) -> dict:
    return {**(old or {}), **(new or {})}


class AgentState(TypedDict, total=False):
    # request
    request_id: str
    message_id: str
    session_id: str
    user: dict
    question: str
    # input guard
    blocked: bool
    refusal_reason: str
    guard: dict
    # memory
    memory: dict
    # supervisor
    intent: str
    plan: list[dict]
    step_idx: int
    filters: dict
    standalone_query: str
    routing_reasoning: str
    # work products
    evidence: Annotated[list[dict], merge_evidence]
    research: dict
    tool_results: Annotated[list[dict], operator.add]
    pending_tool_calls: list[dict]
    approvals: dict
    # response + validation
    draft: str
    attempts: int
    answer: str
    citations: list[dict]
    validation: dict
    degraded: bool
    # control / observability
    agent_status: Annotated[dict, merge_dicts]
    notes: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
