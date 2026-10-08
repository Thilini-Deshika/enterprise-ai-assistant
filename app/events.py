"""Real-time activity events.

Every agent/node/tool calls `emit(...)`; the API layer drains the emitter and forwards each event
to the browser as Server-Sent Events. The Streamlit "Agent Activity" panel is a pure renderer of
this stream, so what you see is exactly what the backend did.

Event types: agent_state, node_start, node_end, plan, retrieval, tool_call, tool_result, guardrail,
memory, rlm, validation, token, token_reset, approval_required, final, error, status
"""
from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from typing import AsyncIterator


class EventEmitter:
    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._seq = 0
        self._closed = False

    def emit(self, type_: str, **data) -> None:
        if self._closed:
            return
        self._seq += 1
        self._queue.put_nowait({"seq": self._seq, "ts": time.time(), "type": type_, **data})

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[dict]:
        return self._iterate()

    async def _iterate(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


_current: ContextVar[EventEmitter | None] = ContextVar("current_emitter", default=None)


def bind_emitter(emitter: EventEmitter | None):
    return _current.set(emitter)


def emit(type_: str, **data) -> None:
    emitter = _current.get()
    if emitter is not None:
        emitter.emit(type_, **data)
