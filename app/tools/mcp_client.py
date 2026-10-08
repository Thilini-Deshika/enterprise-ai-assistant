"""MCP client. Talks to the enterprise-data MCP server over stdio (spawned per call) or streamable HTTP.

Failure handling: timeout + retry + circuit breaker; any failure becomes MCPUnavailable which the
tool registry converts to status "unavailable" so the agents keep going with partial information.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from app.config import ROOT, Settings
from app.errors import MCPUnavailable, ToolError
from app.logging_config import get_logger
from app.resilience import CircuitBreaker, retry_async

log = get_logger(__name__)


class MCPClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.breaker = CircuitBreaker("mcp", failure_threshold=3, reset_after_s=20)

    async def _session_call(self, read, write, tool: str, arguments: dict):
        from mcp import ClientSession
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, arguments)

    async def _call_once(self, tool: str, arguments: dict):
        if self.s.mcp_transport == "http":
            from mcp.client.streamable_http import streamablehttp_client
            async with streamablehttp_client(self.s.mcp_url) as (read, write, _):
                return await self._session_call(read, write, tool, arguments)
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "app.mcp_server.server"], cwd=str(ROOT),
            env={"PYTHONPATH": str(ROOT), "MCP_DATA_DIR": str(self.s.mcp_data_dir), "PATH": os.environ.get("PATH", ""),
                 "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
        async with stdio_client(params) as (read, write):
            return await self._session_call(read, write, tool, arguments)

    async def call(self, tool: str, arguments: dict) -> dict:
        async def _guarded():
            return await asyncio.wait_for(self._call_once(tool, arguments), timeout=self.s.tool_timeout_s - 2)
        try:
            result = await self.breaker.call(lambda: retry_async(_guarded, attempts=2, base_delay=0.3, op=f"mcp:{tool}"))
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp_call_failed", tool=tool, error=str(exc)[:200])
            raise MCPUnavailable(f"MCP server unavailable ({type(exc).__name__})") from exc
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            raise ToolError(f"MCP tool error: {text[:200]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
