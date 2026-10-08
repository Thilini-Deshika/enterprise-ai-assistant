"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.errors import AppError
from app.logging_config import bind_request_context, clear_request_context, configure_logging, get_logger
from app.services.container import Container

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.container = await Container.create(settings)
    yield


app = FastAPI(title="Enterprise AI Assistant API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    bind_request_context(http_request_id=rid, path=request.url.path)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        log.info("http_request", method=request.method, status=getattr(locals().get("response"), "status_code", 500),
                 ms=round((time.perf_counter() - t0) * 1000))
        clear_request_context()
    response.headers["x-request-id"] = rid
    return response


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    headers = {"Retry-After": str(exc.details["retry_after"])} if "retry_after" in exc.details else {}
    return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.http_status, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    problems = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse({"error": {"code": "validation_failed", "message": "Invalid request", "problems": problems}}, status_code=422)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled_error")
    return JSONResponse({"error": {"code": "internal_error", "message": "Internal server error"}}, status_code=500)


app.include_router(router)
