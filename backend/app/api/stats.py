"""Aggregate usage statistics for GET /stats.

Counts and latencies only — never question text — so it is safe to expose publicly.
In-memory: resets when the process restarts (fine for a single small instance).
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


class Stats:
    def __init__(self, window: int = 500, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self._lock = threading.Lock()
        self.queries = 0
        self.cache_hits = 0
        self.insufficient = 0
        self.confidence: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self._latency: deque[int] = deque(maxlen=window)  # computed answers only

    def record_answer(self, level: str, insufficient: bool, latency_ms: int, cached: bool):
        with self._lock:
            self.queries += 1
            self.confidence[level] += 1
            if insufficient:
                self.insufficient += 1
            if cached:
                self.cache_hits += 1
            else:
                self._latency.append(latency_ms)

    def record_error(self, code: str):
        with self._lock:
            self.errors[code] += 1

    def snapshot(self, cache_size: int) -> dict:
        with self._lock:
            lat = list(self._latency)
            answered = self.queries
            return {
                "uptime_s": int(self.clock() - self.started),
                "queries": answered,
                "cache": {
                    "hits": self.cache_hits,
                    "hit_rate": round(self.cache_hits / answered, 3) if answered else 0.0,
                    "entries": cache_size,
                },
                "confidence": {
                    lvl: self.confidence.get(lvl, 0) for lvl in ("HIGH", "MEDIUM", "LOW")
                },
                "insufficient_evidence": self.insufficient,
                "errors": dict(self.errors),
                "latency_ms": {
                    "p50": _percentile(lat, 50),
                    "p95": _percentile(lat, 95),
                    "samples": len(lat),
                },
            }
