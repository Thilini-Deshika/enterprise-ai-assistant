"""Token-bucket rate limiting (per user, per role, configurable).

Each bucket holds up to `capacity` tokens and refills continuously at `refill_per_sec`. A request
costs 1 token; bursts up to `capacity` are allowed, the sustained rate is `refill_per_sec`.
Rejections raise RateLimitExceeded carrying retry_after so the API can answer 429 + Retry-After.
In-process state is fine for one API replica; for N replicas move the bucket state to Redis (Lua script).
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Callable

from app.config import RateLimitConfig
from app.errors import RateLimitExceeded


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float, clock: Callable[[], float] = time.monotonic):
        self.capacity, self.refill_per_sec, self._clock = capacity, refill_per_sec, clock
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_per_sec)
        self._last = now

    async def try_consume(self, cost: float = 1.0) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        async with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return True, 0.0
            deficit = cost - self._tokens
            return False, deficit / self.refill_per_sec if self.refill_per_sec > 0 else math.inf

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens


class RateLimiter:
    def __init__(self, per_role: dict[str, RateLimitConfig], default: RateLimitConfig | None = None, name: str = "chat"):
        self.per_role, self.name = per_role, name
        self.default = default or RateLimitConfig(capacity=5, refill_per_sec=0.1)
        self._buckets: dict[str, TokenBucket] = {}

    def _bucket(self, user_id: str, role: str) -> TokenBucket:
        if user_id not in self._buckets:
            cfg = self.per_role.get(role, self.default)
            self._buckets[user_id] = TokenBucket(cfg.capacity, cfg.refill_per_sec)
        return self._buckets[user_id]

    async def check(self, user_id: str, role: str, cost: float = 1.0) -> None:
        ok, retry_after = await self._bucket(user_id, role).try_consume(cost)
        if not ok:
            raise RateLimitExceeded(
                f"Rate limit reached for {self.name}. Please retry in {math.ceil(retry_after)}s.",
                retry_after=math.ceil(retry_after),
            )

    def remaining(self, user_id: str, role: str) -> float:
        return round(self._bucket(user_id, role).tokens, 2)
