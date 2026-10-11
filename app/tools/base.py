"""Tool registry: the ONE chokepoint every tool call passes through (agents cannot bypass it).

Order of checks for each call:
  1 known tool?  2 RBAC permission  3 per-user tool rate limit  4 pydantic argument validation
  5 injection scan of string arguments  6 human-approval gate (write tools)  7 execution with timeout
  8 output sanitisation (tool output is untrusted data)  9 audit + activity events
Because authorisation lives here - not in a prompt - a jailbroken or hallucinating LLM can at worst
*ask* for a forbidden tool; it is denied, audited and surfaced in the UI.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langsmith import traceable
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.errors import ApprovalRequired, AppError, DependencyUnavailable, RateLimitExceeded
from app.events import emit
from app.logging_config import get_logger
from app.security import injection
from app.security.audit import AuditLog
from app.security.rate_limit import RateLimiter
from app.security.rbac import Permission, UserContext

log = get_logger(__name__)


@dataclass
class ToolContext:
    user: UserContext
    request_id: str
    datasets: dict[str, list[dict]] = field(default_factory=dict)
    start_no: int = 1
    _counter: Any = None

    def next_no(self) -> int:
        if self._counter is None:
            self._counter = itertools.count(self.start_no)
        return next(self._counter)


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    permission: Permission
    handler: Callable[[Any, ToolContext], Awaitable[Any]]
    timeout_s: float = 20.0
    requires_approval: bool = False
    untrusted_output: bool = False

    def schema_hint(self) -> str:
        props = self.args_model.model_json_schema().get("properties", {})
        return ", ".join(f"{k}: {v.get('type', v.get('anyOf', [{}])[0].get('type', 'any'))}" for k, v in props.items())


def _strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _sanitize_output(obj: Any, found: list[str]) -> Any:
    if isinstance(obj, str):
        clean, removed = injection.sanitize_untrusted(obj)
        found.extend(removed)
        return clean
    if isinstance(obj, dict):
        return {k: _sanitize_output(v, found) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_output(v, found) for v in obj]
    return obj


_SEVERITY_ALIASES = {"p1": "sev1", "p2": "sev2", "p3": "sev3", "p4": "sev4", "critical": "sev1", "high": "sev2", "major": "sev2",
                     "medium": "sev3", "moderate": "sev3", "low": "sev4", "minor": "sev4"}


def _coerce_enum(value: str, allowed: list) -> str:
    """Map common spellings (P2, Sev 2, High) onto an enum the schema accepts; leave anything else for validation to reject."""
    v = value.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    for a in allowed:
        if v == str(a).lower().replace("_", "").replace("-", ""):
            return a
    if v.isdigit() and f"sev{v}" in allowed:
        return f"sev{v}"
    alias = _SEVERITY_ALIASES.get(v)
    return alias if alias in allowed else value


def normalise_args(model: type[BaseModel], raw: dict | None) -> dict:
    """Repair common LLM slips before strict validation: empty/null optionals are dropped,
    unknown keys are ignored and numbers are clamped to the schema's bounds. Validation still decides."""
    props = model.model_json_schema().get("properties", {})
    out: dict = {}
    for key, value in (raw or {}).items():
        spec = props.get(key)
        if spec is None:
            continue
        if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none", "any", "all")):
            continue
        enum = spec.get("enum") or next((o["enum"] for o in spec.get("anyOf", []) if "enum" in o), None)
        if enum and isinstance(value, (str, int)) and not isinstance(value, bool):
            value = _coerce_enum(str(value), enum)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "maximum" in spec:
                value = min(value, spec["maximum"])
            if "minimum" in spec:
                value = max(value, spec["minimum"])
        out[key] = value
    return out


class ToolRegistry:
    def __init__(self, settings: Settings, audit: AuditLog, limiter: RateLimiter):
        self.s, self.audit, self.limiter = settings, audit, limiter
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def all_names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def visible_to(self, user: UserContext) -> list[ToolSpec]:
        return [t for t in self._tools.values() if user.has(t.permission)]

    def catalogue(self, user: UserContext) -> str:
        return "\n".join(f"- {t.name}({t.schema_hint()}): {t.description}" +
                         (" [REQUIRES HUMAN APPROVAL]" if t.requires_approval else "") for t in self.visible_to(user))

    @traceable(run_type="tool", name="tool_execute",
               process_inputs=lambda i: {"tool": i.get("name"), "args": i.get("raw_args"), "approved": i.get("approved")})
    async def execute(self, name: str, raw_args: dict | None, ctx: ToolContext, *, approved: bool = False) -> dict:
        t0 = time.perf_counter()
        call_id = f"tool:{name}#{ctx.next_no()}"
        base = {"id": call_id, "tool": name, "args": raw_args or {}}
        user = ctx.user

        def finish(status: str, *, data: Any = None, error: str = "") -> dict:
            res = {**base, "status": status, "ok": status == "ok", "data": data, "error": error,
                   "duration_ms": round((time.perf_counter() - t0) * 1000)}
            preview = json.dumps(data, default=str)[:500] if data is not None else error
            emit("tool_result", id=call_id, tool=name, status=status, duration_ms=res["duration_ms"], preview=preview)
            return res

        spec = self._tools.get(name)
        if spec is None:
            return finish("error", error=f"Unknown tool '{name}'")
        emit("tool_call", id=call_id, tool=name, args=raw_args or {}, permission=spec.permission.value,
             requires_approval=spec.requires_approval, approved=approved)

        if not user.has(spec.permission):  # (2) RBAC
            await self.audit.record(user_id=user.user_id, role=user.role.value, action="tool_call", outcome="denied", resource=name)
            emit("guardrail", kind="authorization_denied", tool=name, role=user.role.value, required=spec.permission.value)
            return finish("denied", error=f"Your role ({user.role.value}) is not permitted to use '{name}'.")
        try:  # (3) rate limit
            await self.limiter.check(user.user_id, user.role.value)
        except RateLimitExceeded as exc:
            return finish("rate_limited", error=exc.message)
        try:  # (4) validation
            args = spec.args_model.model_validate(normalise_args(spec.args_model, raw_args))
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())[:300]
            return finish("invalid_args", error=f"Invalid arguments: {msg}")
        for s in _strings(args.model_dump()):  # (5) injection in args
            rep = injection.scan(s)
            if rep.score >= self.s.injection_block_threshold:
                await self.audit.record(user_id=user.user_id, role=user.role.value, action="tool_call", outcome="blocked_injection",
                                        resource=name, categories=rep.categories)
                emit("guardrail", kind="tool_argument_injection", tool=name, categories=rep.categories)
                return finish("blocked", error="Tool arguments were blocked by the injection guardrail.")
        if spec.requires_approval and not approved:  # (6) HITL gate
            raise ApprovalRequired(f"'{name}' requires human approval", call_id=call_id, tool=name, args=args.model_dump())
        try:  # (7) run
            data = await asyncio.wait_for(spec.handler(args, ctx), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            await self.audit.record(user_id=user.user_id, role=user.role.value, action="tool_call", outcome="timeout", resource=name)
            return finish("timeout", error=f"'{name}' timed out after {spec.timeout_s:.0f}s.")
        except DependencyUnavailable as exc:
            return finish("unavailable", error=exc.message)
        except AppError as exc:
            return finish("error", error=exc.message)
        except Exception as exc:  # noqa: BLE001 - tools must never crash the graph
            log.exception("tool_crashed", tool=name)
            return finish("error", error=f"Internal error while running '{name}': {type(exc).__name__}")
        if spec.untrusted_output:  # (8) treat tool output as data
            found: list[str] = []
            data = _sanitize_output(data, found)
            if found:
                emit("guardrail", kind="injection_in_tool_output", tool=name, removed=found[:3])
                await self.audit.record(user_id=user.user_id, role=user.role.value, action="tool_output", outcome="sanitized", resource=name)
        await self.audit.record(user_id=user.user_id, role=user.role.value, action="tool_call", outcome="ok", resource=name,
                                approved=approved)  # (9)
        return finish("ok", data=data)
