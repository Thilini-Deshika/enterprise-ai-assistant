"""Small resilience toolkit: retry with jittered backoff + circuit breaker."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable, TypeVar

from app.errors import DependencyUnavailable
from app.logging_config import get_logger

log = get_logger(__name__)
T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 4.0,
    no_retry_on: tuple[type[BaseException], ...] = (),
    op: str = "operation",
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except no_retry_on:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 - we deliberately retry any transient failure
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)
            log.warning("retrying", op=op, attempt=attempt, error=str(exc)[:200], delay_s=round(delay, 2))
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (after reset_after_s) -> half-open -> closed/open.

    Stops a failing dependency (LLM, MCP, Pinecone) from being hammered and from stalling every
    request for the full timeout: callers get an instant DependencyUnavailable and can degrade.
    """

    def __init__(self, name: str, failure_threshold: int = 3, reset_after_s: float = 30.0):
        self.name, self.failure_threshold, self.reset_after_s = name, failure_threshold, reset_after_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half-open" if time.monotonic() - self._opened_at >= self.reset_after_s else "open"

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        if self.state == "open":
            raise DependencyUnavailable(f"{self.name} circuit is open; skipping call", dependency=self.name)
        try:
            result = await func()
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                if self._opened_at is None:
                    log.error("circuit_opened", dependency=self.name)
                self._opened_at = time.monotonic()
            raise
        self._failures, self._opened_at = 0, None
        return result
