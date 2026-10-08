"""Run LLM-written pandas analysis code in an isolated subprocess.

Layers: AST allow-list (parent) -> separate interpreter with `-I`, empty environment, restricted
builtins, CPU/FSIZE rlimits on POSIX -> hard wall-clock timeout -> output size cap.
Not a security boundary on its own (see docs); production: gVisor/Firecracker/e2b container.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from app.errors import ToolError, ToolTimeout
from app.security.safe_python import validate_python

_RUNNER = r'''
import sys, json, builtins, io, contextlib, math, statistics, collections
payload = json.loads(sys.stdin.read())
import pandas as pd
data = payload["data"]
df = pd.DataFrame(data)
names = ("len","range","sorted","reversed","set","list","dict","tuple","sum","min","max","enumerate","zip","any",
         "all","str","int","float","bool","abs","round","isinstance","map","filter","print","Exception","ValueError","KeyError","TypeError")
env = {"__builtins__": {n: getattr(builtins, n) for n in names},
       "df": df, "data": data, "pd": pd, "Counter": collections.Counter, "statistics": statistics, "math": math}
buf = io.StringIO()
def _default(o):
    return o.item() if hasattr(o, "item") else str(o)
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(payload["code"], "<analysis>", "exec"), env)
    result = env.get("result")
    if isinstance(result, pd.DataFrame):
        result = result.to_dict("records")
    elif isinstance(result, pd.Series):
        result = result.to_dict()
    print(json.dumps({"ok": True, "result": result, "stdout": buf.getvalue()[:2000]}, default=_default))
except Exception as e:
    print(json.dumps({"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}))
'''


def _limits():  # POSIX only
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


async def run_analysis(code: str, rows: list[dict], timeout_s: float = 12.0) -> dict:
    validate_python(code, mode="analysis")
    if not rows:
        raise ToolError("The dataset is empty - fetch data first (e.g. with mcp_list_incidents).")
    env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    kwargs = {"preexec_fn": _limits} if os.name == "posix" else {}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-c", _RUNNER, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, **kwargs)
    try:
        out, err = await asyncio.wait_for(proc.communicate(json.dumps({"code": code, "data": rows}).encode()), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        raise ToolTimeout(f"Analysis exceeded {timeout_s:.0f}s and was terminated.")
    if proc.returncode != 0 or not out:
        raise ToolError(f"Analysis process failed: {err.decode()[-200:] or 'no output'}")
    payload = json.loads(out.decode().strip().splitlines()[-1])
    if not payload.get("ok"):
        raise ToolError(payload.get("error", "analysis failed"))
    return {"result": payload["result"], "stdout": payload.get("stdout", ""), "rows_analysed": len(rows)}
