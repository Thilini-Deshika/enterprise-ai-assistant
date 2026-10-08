"""LangGraph assembly.

        START -> input_guard --blocked--> refuse -> END
                     |
                 memory_load -> supervisor -> (plan-driven dispatch)
                                   |
         +-------------+-----------+-----------+
     retrieval      research(RLM)           tools <-> approval (HITL interrupt)
         +-------------+-----------+-----------+
                                   |
                               response -> validator --fail(retry)--> response
                                               |pass
                                          memory_write -> END
`dispatch` reads the supervisor's plan, so the graph topology is fixed (auditable) while the *path*
through it is decided per request (dynamic orchestration).
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.deps import AgentDeps
from app.agents.nodes_final import make_memory_write, make_response, make_validator
from app.agents.nodes_front import make_input_guard, make_memory_load, make_refuse, make_supervisor
from app.agents.nodes_work import make_approval, make_research, make_retrieval, make_tools
from app.agents.state import AgentState

ROUTES = {"retrieval": "retrieval", "research": "research", "tools": "tools", "response": "response"}


def dispatch(state: dict) -> str:
    plan, idx = state.get("plan", []), state.get("step_idx", 0)
    return plan[idx]["agent"] if idx < len(plan) else "response"


def after_tools(state: dict) -> str:
    return "approval" if state.get("pending_tool_calls") else dispatch(state)


def after_validation(state: dict) -> str:
    return "memory_write" if state.get("validation", {}).get("passed") else "response"


def build_graph(deps: AgentDeps):
    g = StateGraph(AgentState)
    g.add_node("input_guard", make_input_guard(deps))
    g.add_node("refuse", make_refuse(deps))
    g.add_node("memory_load", make_memory_load(deps))
    g.add_node("supervisor", make_supervisor(deps))
    g.add_node("retrieval", make_retrieval(deps))
    g.add_node("research", make_research(deps))
    g.add_node("tools", make_tools(deps))
    g.add_node("approval", make_approval(deps))
    g.add_node("response", make_response(deps))
    g.add_node("validator", make_validator(deps))
    g.add_node("memory_write", make_memory_write(deps))

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: "refuse" if s.get("blocked") else "memory_load",
                            {"refuse": "refuse", "memory_load": "memory_load"})
    g.add_edge("memory_load", "supervisor")
    g.add_conditional_edges("supervisor", dispatch, ROUTES)
    g.add_conditional_edges("retrieval", dispatch, ROUTES)
    g.add_conditional_edges("research", dispatch, ROUTES)
    g.add_conditional_edges("tools", after_tools, {**ROUTES, "approval": "approval"})
    g.add_edge("approval", "tools")
    g.add_edge("response", "validator")
    g.add_conditional_edges("validator", after_validation, {"memory_write": "memory_write", "response": "response"})
    g.add_edge("memory_write", END)
    g.add_edge("refuse", END)
    # MemorySaver is only needed so a paused (human-approval) run can be resumed within the same process;
    # thread_id is unique per turn so it never leaks state between turns. Use a Postgres/Redis saver in prod.
    return g.compile(checkpointer=MemorySaver())
