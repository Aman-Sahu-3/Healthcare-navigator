"""Small in-memory sliding-window rate limiter (per client IP).

Good for a single API instance. With several instances, move the counters to Redis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, Request

from app.api.errors import APIError
from app.core.config import Settings, get_settings

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_rate(rate: str) -> tuple[int, int]:
    """'20/minute' -> (20, 60)."""
    count, period = rate.split("/")
    return int(count), _PERIODS[period]


class RateLimiter:
    def __init__(self, limit: int, window_s: int, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns (allowed, seconds until a slot frees up)."""
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(q[0] + self.window_s - now) + 1)
            q.append(now)
            return True, 0


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter: RateLimiter = request.app.state.rate_limiter
    allowed, retry_after = limiter.hit(client_ip(request, settings.trust_proxy_headers))
    if not allowed:
        raise APIError(
            429,
            "rate_limited",
            f"Too many requests. Limit is {settings.rate_limit}.",
            headers={"Retry-After": str(retry_after)},
        )
