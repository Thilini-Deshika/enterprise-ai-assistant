"""HTTP API. Streaming endpoints return Server-Sent Events (one JSON activity event per `data:` line)."""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.errors import AuthorizationError
from app.events import EventEmitter, emit
from app.security.auth import authenticate, create_token, current_user
from app.security.rbac import ROLE_PERMISSIONS, Permission, UserContext
from app.security.validation import ApprovalRequest, ChatRequest, FeedbackRequest, LoginRequest

router = APIRouter()


def _c(request: Request):
    return request.app.state.container


def sse(run: Callable[[EventEmitter], Awaitable[None]]) -> StreamingResponse:
    emitter = EventEmitter()

    async def runner():
        try:
            await run(emitter)
        except Exception as exc:  # noqa: BLE001 - last-resort guard so the stream always terminates cleanly
            emitter.emit("error", code="internal_error", message=f"{type(exc).__name__}")
        finally:
            emitter.close()

    task = asyncio.create_task(runner())

    async def gen():
        try:
            async for ev in emitter:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        finally:
            if not task.done():  # client disconnected -> stop doing work
                task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ auth / identity
@router.post("/auth/login")
async def login(body: LoginRequest):
    user = authenticate(body.username, body.password)
    return {"access_token": create_token(user), "token_type": "bearer", "user": user.to_dict()}


@router.get("/me")
async def me(request: Request, user: UserContext = Depends(current_user)):
    c = _c(request)
    return {"user": user.to_dict(), "permissions": sorted(p.value for p in ROLE_PERMISSIONS[user.role]),
            "access_levels": list(user.allowed_access_levels), "tools": [t.name for t in c.registry.visible_to(user)],
            "rate_limit_tokens_remaining": c.rate_limiter.remaining(user.user_id, user.role.value)}


# ------------------------------------------------------------------ chat
@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: UserContext = Depends(current_user)):
    c = _c(request)
    if not user.has(Permission.CHAT):
        raise AuthorizationError("Your role may not use chat.")
    await c.rate_limiter.check(user.user_id, user.role.value)  # raises 429 before any work starts
    c.memory.get_session(body.session_id, user)               # 403 if the session belongs to someone else
    return sse(lambda em: c.orchestrator.run_turn(user, body.session_id, body.message, em))


@router.post("/chat/approve")
async def chat_approve(body: ApprovalRequest, request: Request, user: UserContext = Depends(current_user)):
    c = _c(request)
    c.memory.get_session(body.session_id, user)
    return sse(lambda em: c.orchestrator.resume(user, body.session_id, body.approve, body.comment, em))


@router.get("/sessions/{session_id}/memory")
async def session_memory(session_id: str, request: Request, user: UserContext = Depends(current_user)):
    c = _c(request)
    return await c.memory.snapshot(user, c.memory.get_session(session_id, user))


@router.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request, user: UserContext = Depends(current_user)):
    c = _c(request)
    c.memory.get_session(body.session_id, user)
    import time
    run_id = c.orchestrator.runs.get(body.message_id)
    await c.db.execute("INSERT INTO feedback(ts,user_id,session_id,message_id,run_id,rating,comment) VALUES (?,?,?,?,?,?,?)",
                       (time.time(), user.user_id, body.session_id, body.message_id, run_id, body.rating, body.comment))
    await c.memory.apply_feedback(user, body.message_id, body.rating)
    pushed = False
    if run_id:  # attach the rating to the LangSmith trace so quality can be filtered/evaluated there
        try:
            import os
            if os.environ.get("LANGSMITH_API_KEY") and os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
                from langsmith import Client
                await asyncio.to_thread(Client().create_feedback, run_id, key="user_rating", score=body.rating, comment=body.comment or None)
                pushed = True
        except Exception:  # noqa: BLE001 - feedback must never fail the request
            pass
    return {"ok": True, "langsmith_feedback": pushed}


# ------------------------------------------------------------------ ops / admin
@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    c = _c(request)
    vector_ok = await c.retriever.store.healthy()
    checks = {"index_loaded": c.retriever is not None, "vector_store": vector_ok, "llm_provider": c.llm.provider,
              "llm_circuit": c.llm.breaker.state, "mcp_circuit": c.mcp.breaker.state, "vector_backend": c.retriever.store.name,
              "langsmith_tracing": __import__("os").environ.get("LANGSMITH_TRACING", "false")}
    return JSONResponse({"status": "ready" if vector_ok else "degraded", "checks": checks}, status_code=200 if vector_ok else 503)


def _require_admin(user: UserContext) -> None:
    if not user.has(Permission.ADMIN_TOOLS):
        raise AuthorizationError("Administrator role required.")


@router.get("/admin/audit")
async def audit(request: Request, limit: int = 100, user: UserContext = Depends(current_user)):
    _require_admin(user)
    return {"items": await _c(request).audit.recent(min(limit, 500))}


@router.get("/admin/feedback")
async def admin_feedback(request: Request, user: UserContext = Depends(current_user)):
    _require_admin(user)
    return {"items": await _c(request).db.fetchall("SELECT * FROM feedback ORDER BY id DESC LIMIT 200")}


@router.post("/admin/reindex")
async def admin_reindex(request: Request, user: UserContext = Depends(current_user)):
    _require_admin(user)
    c = _c(request)
    await c.audit.record(user_id=user.user_id, role=user.role.value, action="reindex", outcome="started", resource="api")
    return await c.reindex()
