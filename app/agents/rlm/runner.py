"""Recursive research loop (RLM).

    plan (LLM writes Python)  ->  validate (AST allow-list)  ->  execute against `env`
         ^                                                           |
         +---------- observation (if not done) <---------------------+
Nested agents (`env.delegate`) run the same loop one level deeper with a smaller step budget.
Failure policy: rejected/failed LLM plan -> deterministic default plan; budget exhausted -> partial result.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from langsmith import traceable

from app.agents.deps import AgentDeps
from app.agents.rlm.env import Budget, ResearchEnv
from app.agents.rlm.planner import default_plan_code, generate_plan
from app.agents.rlm.runtime import execute_plan
from app.errors import BudgetExceeded, GuardrailViolation
from app.events import emit
from app.logging_config import get_logger
from app.retrieval.filters import SearchFilters
from app.security.rbac import UserContext
from app.tools.base import ToolContext

log = get_logger(__name__)


@dataclass
class ResearchResult:
    summary: str = ""
    doc_ids: list[str] = field(default_factory=list)
    status: str = "complete"  # complete | partial | failed
    steps: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ResearchRunner:
    def __init__(self, deps: AgentDeps, user: UserContext, ctx: ToolContext, budget: Budget, *, depth: int = 0,
                 evidence: dict[str, dict] | None = None, sem: asyncio.Semaphore | None = None):
        self.deps, self.user, self.ctx, self.budget, self.depth = deps, user, ctx, budget, depth
        self.evidence = evidence if evidence is not None else {}
        self.sem = sem or asyncio.Semaphore(deps.settings.rlm_concurrency)

    async def _delegate(self, question: str, filters: dict, depth: int) -> ResearchResult:
        child = ResearchRunner(self.deps, self.user, self.ctx, self.budget, depth=depth, evidence=self.evidence, sem=self.sem)
        return await child.run(question, filters)

    async def _plan(self, question: str, filters: dict, observations: list[str]) -> tuple[str, str]:
        s = self.deps.settings
        default = default_plan_code(question, filters, s.rlm_batch_size)
        if self.depth > 0 or self.deps.llm.provider == "mock":
            return default, "default" if self.deps.llm.provider != "mock" else "offline-template"
        stats = self.deps.catalog_getter().stats(self.user, SearchFilters.sanitize(filters))
        feedback = ""
        for attempt in range(2):
            try:
                code = await generate_plan(self.deps.llm, question, filters, stats, observations, feedback)
                from app.security.safe_python import validate_python
                validate_python(code, mode="rlm")
                return code, "llm"
            except GuardrailViolation as exc:
                feedback = exc.message
                emit("guardrail", kind="rlm_plan_rejected", reason=exc.message, attempt=attempt + 1)
            except Exception as exc:  # noqa: BLE001
                emit("rlm", action="degraded", depth=self.depth, reason=f"planner LLM unavailable ({type(exc).__name__}); using default plan")
                break
        return default, "default-fallback"

    @traceable(name="rlm_research", run_type="chain", process_inputs=lambda i: {"question": i.get("question"), "filters": i.get("filters")})
    async def run(self, question: str, filters: dict) -> ResearchResult:
        s = self.deps.settings
        result = ResearchResult()
        observations: list[str] = []
        max_steps = s.rlm_max_steps if self.depth == 0 else 1
        env = ResearchEnv(deps=self.deps, user=self.user, ctx=self.ctx, budget=self.budget, depth=self.depth,
                          evidence=self.evidence, sem=self.sem, delegate_fn=self._delegate)
        emit("rlm", action="start", depth=self.depth, question=question[:160], max_steps=max_steps)
        for step in range(1, max_steps + 1):
            code, source = await self._plan(question, filters, observations)
            emit("rlm", action="plan", depth=self.depth, step=step, source=source, code=code)
            t0 = time.perf_counter()
            try:
                out = await execute_plan(code, env, s.rlm_plan_timeout_s)
            except BudgetExceeded as exc:
                emit("rlm", action="budget_exceeded", depth=self.depth, reason=exc.message)
                result.status = "partial"
                result.notes.append(f"Research stopped early: {exc.message}.")
                break
            except Exception as exc:  # noqa: BLE001 - bad generated code, runtime error, timeout
                emit("rlm", action="plan_failed", depth=self.depth, step=step, source=source, error=f"{type(exc).__name__}: {str(exc)[:200]}")
                if source in ("llm",):
                    code = default_plan_code(question, filters, s.rlm_batch_size)
                    emit("rlm", action="plan", depth=self.depth, step=step, source="default-fallback", code=code)
                    try:
                        out = await execute_plan(code, env, s.rlm_plan_timeout_s)
                    except Exception as exc2:  # noqa: BLE001
                        result.status, out = "failed", None
                        result.notes.append(f"Research failed: {type(exc2).__name__}")
                        break
                else:
                    result.status, out = "failed", None
                    result.notes.append(f"Research failed: {type(exc).__name__}")
                    break
            out = out if isinstance(out, dict) else {"summary": str(out), "done": True}
            result.summary = str(out.get("summary", ""))
            result.doc_ids = [str(d) for d in out.get("doc_ids", [])]
            result.steps.append({"step": step, "source": source, "seconds": round(time.perf_counter() - t0, 2), "notes": env.notes[-3:]})
            observations.append(f"step {step}: {result.summary[:500]}")
            emit("rlm", action="step_result", depth=self.depth, step=step, summary=result.summary[:400], done=bool(out.get("done", True)))
            if out.get("done", True):
                break
        result.notes.extend(env.notes[-2:])
        b = self.budget
        result.stats = {"searches": b.searches, "sub_llm_calls": b.llm_calls, "batches": b.batches, "max_depth_reached": b.depth_reached,
                        "sub_agents": b.sub_agents, "documents_analysed": len(result.doc_ids), "chunks_read": len(self.evidence)}
        if self.depth == 0:
            emit("rlm", action="done", depth=0, status=result.status, stats=result.stats)
        return result


def new_budget(deps: AgentDeps) -> Budget:
    s = deps.settings
    return Budget(max_searches=s.rlm_max_searches, max_llm_calls=s.rlm_max_llm_calls, max_depth=s.rlm_max_depth,
                  deadline=time.monotonic() + s.rlm_plan_timeout_s * 1.5)
