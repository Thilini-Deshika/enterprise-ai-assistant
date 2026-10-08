"""Restricted execution of LLM-generated research plans (`async def run(env)`)."""
from __future__ import annotations

import asyncio
import builtins
import collections

from app.security.safe_python import validate_python

_SAFE = ("len", "range", "sorted", "reversed", "set", "list", "dict", "tuple", "sum", "min", "max", "enumerate", "zip", "any", "all",
         "str", "int", "float", "bool", "abs", "round", "isinstance", "map", "filter", "Exception", "ValueError", "KeyError", "TypeError")


async def execute_plan(code: str, env, timeout_s: float):
    """Validate (AST allow-list) then run the plan with restricted builtins and a wall-clock timeout.

    The plan can only touch the world through `env`, whose every method is budgeted, RBAC-scoped,
    emits an activity event and is traced. CPU-bound loops are bounded by constant-size limits in
    the validator (no `while`, no huge ints)."""
    validate_python(code, mode="rlm")
    namespace = {"__builtins__": {n: getattr(builtins, n) for n in _SAFE}, "Counter": collections.Counter}
    exec(compile(code, "<rlm_plan>", "exec"), namespace)  # noqa: S102 - validated, restricted namespace
    return await asyncio.wait_for(namespace["run"](env), timeout=timeout_s)
