"""Runs one chat turn through the graph, streaming activity events, and resumes human approvals."""
from __future__ import annotations

import asyncio
import uuid

from langgraph.types import Command

from app.agents.deps import AgentDeps
from app.agents.graph import build_graph
from app.errors import AppError
from app.events import EventEmitter, bind_emitter, emit
from app.logging_config import bind_request_context, clear_request_context, get_logger
from app.security.rbac import UserContext

log = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, deps: AgentDeps, vector_backend_name):
        self.deps = deps
        self.graph = build_graph(deps)
        self.vector_backend_name = vector_backend_name
        self.pending: dict[str, dict] = {}   # session_id -> paused run (awaiting human approval)
        self.runs: dict[str, str] = {}       # message_id -> LangSmith run id (for feedback)

    def _config(self, user: UserContext, session_id: str, thread_id: str, run_id: uuid.UUID, request_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "run_id": run_id, "run_name": "chat_turn", "recursion_limit": 40,
                "tags": ["enterprise-ai-assistant", f"role:{user.role.value}"],
                "metadata": {"user_id": user.user_id, "role": user.role.value, "session_id": session_id, "request_id": request_id}}

    async def run_turn(self, user: UserContext, session_id: str, message: str, emitter: EventEmitter) -> None:
        bind_emitter(emitter)
        request_id, message_id = uuid.uuid4().hex[:12], uuid.uuid4().hex[:16]
        bind_request_context(request_id=request_id, user_id=user.user_id, session_id=session_id)
        run_id = uuid.uuid4()
        thread_id = f"{session_id}:{request_id}"
        self.runs[message_id] = str(run_id)
        s = self.deps.settings
        emit("status", phase="started", request_id=request_id, message_id=message_id, trace_run_id=str(run_id),
             langsmith_project=__import__("os").environ.get("LANGSMITH_PROJECT", "default"),
             llm_provider=self.deps.llm.provider, vector_backend=self.vector_backend_name, role=user.role.value)
        state0 = {"request_id": request_id, "message_id": message_id, "session_id": session_id, "user": user.to_dict(), "question": message,
                  "evidence": [], "tool_results": [], "notes": [], "errors": [], "agent_status": {}, "attempts": 0, "plan": [], "step_idx": 0}
        config = self._config(user, session_id, thread_id, run_id, request_id)
        try:
            await self._drive(self.graph.astream(state0, config, stream_mode="updates"), s.turn_timeout_s)
            await self._after(user, session_id, config, thread_id)
        except Exception as exc:  # noqa: BLE001
            self._report(exc)
        finally:
            clear_request_context()

    async def resume(self, user: UserContext, session_id: str, approve: bool, comment: str, emitter: EventEmitter) -> None:
        bind_emitter(emitter)
        paused = self.pending.pop(session_id, None)
        if not paused or paused["user_id"] != user.user_id:
            emit("error", code="no_pending_approval", message="There is no pending action to approve for this session.")
            return
        request_id = paused["request_id"]
        bind_request_context(request_id=request_id, user_id=user.user_id, session_id=session_id)
        run_id = uuid.uuid4()
        self.runs[paused["message_id"]] = str(run_id)
        emit("status", phase="resumed", request_id=request_id, message_id=paused["message_id"], trace_run_id=str(run_id), approve=approve)
        config = self._config(user, session_id, paused["thread_id"], run_id, request_id)
        try:
            await self._drive(self.graph.astream(Command(resume={"approve": approve, "comment": comment}), config, stream_mode="updates"),
                              self.deps.settings.turn_timeout_s)
            await self._after(user, session_id, config, paused["thread_id"], paused)
        except Exception as exc:  # noqa: BLE001
            self._report(exc)
        finally:
            clear_request_context()

    async def _drive(self, stream, timeout_s: float) -> None:
        async with asyncio.timeout(timeout_s):
            async for _ in stream:  # nodes emit their own activity events; we only drive the graph
                pass

    async def _after(self, user: UserContext, session_id: str, config: dict, thread_id: str, prior: dict | None = None) -> None:
        snap = await self.graph.aget_state(config)
        interrupts = [i for t in snap.tasks for i in getattr(t, "interrupts", [])]
        if snap.next and interrupts:
            values = snap.values
            self.pending[session_id] = {"thread_id": thread_id, "user_id": user.user_id, "request_id": values["request_id"],
                                        "message_id": values["message_id"]}
            emit("approval_required", calls=interrupts[0].value["calls"], message_id=values["message_id"],
                 message="This action changes state and needs a human decision before it runs.")
        else:
            emit("status", phase="completed")

    @staticmethod
    def _report(exc: BaseException) -> None:
        if isinstance(exc, TimeoutError):
            emit("error", code="turn_timeout", message="The request took too long and was stopped. Please try a narrower question.")
        elif isinstance(exc, AppError):
            emit("error", code=exc.code, message=exc.message)
        else:
            log.exception("turn_failed")
            emit("error", code="internal_error", message="Something went wrong while processing your request. Please try again.")
