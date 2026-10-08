"""Shared helpers: node instrumentation (events + failure containment) and small utilities."""
from __future__ import annotations

import functools
import time
from typing import Awaitable, Callable

from langgraph.errors import GraphBubbleUp

from app.events import emit
from app.logging_config import get_logger
from app.security.rbac import UserContext
from app.tools.base import ToolContext

log = get_logger("agents")

DESCRIPTIONS = {
    "input_guard": "Validating the request and scanning for prompt injection",
    "memory_load": "Loading conversation and long-term memory",
    "supervisor": "Understanding intent, decomposing the task, routing to agents",
    "retrieval": "Hybrid retrieval (dense + BM25) with RBAC scoping",
    "research": "Recursive Language Model investigation (plan -> explore -> map -> reduce)",
    "tools": "Selecting and executing tools through the authorised registry",
    "approval": "Waiting for human approval of a sensitive action",
    "response": "Generating the grounded answer",
    "validator": "Validating citations, access, safety and brand voice",
    "memory_write": "Updating conversation and long-term memory",
    "refuse": "Declining the request safely",
}


def user_of(state: dict) -> UserContext:
    return UserContext.from_dict(state["user"])


def tool_context(state: dict) -> ToolContext:
    return ToolContext(user=user_of(state), request_id=state["request_id"], start_no=len(state.get("tool_results", [])) + 1)


def trim_tool_result(res: dict) -> dict:
    """Keep state small: bulky payloads (search hits) live in `evidence`, not in tool_results."""
    out = dict(res)
    if res["tool"] == "knowledge_search" and res.get("data"):
        out["data"] = {"hits": len(res["data"]["hits"]), "diagnostics": res["data"].get("diagnostics")}
    return out


def instrument(name: str, *, advances: bool = False, catch: bool = True, on_fail: Callable[[dict], dict] | None = None):
    """Wrap a node: emit node_start/node_end, time it, and contain failures.

    advances=True marks plan-step agents; on failure they still advance `step_idx` so the graph keeps
    moving and the response agent can explain what was unavailable.
    on_fail(state) returns extra state needed to keep control flow terminating (e.g. response must still
    count the attempt, validator must still produce a safe final answer) - otherwise a failing node
    could make the graph loop.
    """
    def deco(fn: Callable[[dict], Awaitable[dict]]):
        @functools.wraps(fn)
        async def wrapper(state: dict) -> dict:
            t0 = time.perf_counter()
            emit("node_start", node=name, description=DESCRIPTIONS.get(name, name))
            emit("agent_state", agent=name, status="running", detail=DESCRIPTIONS.get(name, name))
            try:
                update = await fn(state)
            except GraphBubbleUp:
                raise  # interrupt() for human-in-the-loop must propagate
            except Exception as exc:  # noqa: BLE001
                if not catch:
                    raise
                log.exception("node_failed", node=name)
                emit("node_end", node=name, status="failed", error=f"{type(exc).__name__}: {str(exc)[:160]}",
                     duration_ms=round((time.perf_counter() - t0) * 1000))
                update = {"agent_status": {name: "failed"}, "errors": [f"{name}: {type(exc).__name__}: {str(exc)[:160]}"]}
                if advances:
                    update["step_idx"] = state.get("step_idx", 0) + 1
                if on_fail:
                    update.update(on_fail(state))
                return update
            status = update.get("agent_status", {}).get(name, "ok")
            update["agent_status"] = {**update.get("agent_status", {}), name: status}
            emit("node_end", node=name, status=status, duration_ms=round((time.perf_counter() - t0) * 1000))
            return update
        return wrapper
    return deco
