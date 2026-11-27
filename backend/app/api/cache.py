"""In-memory answer cache (LRU + TTL).

The same clinical question is often asked repeatedly (demo prompts, teammates). Serving
it from memory skips retrieval and the LLM call: instant answers, no provider quota used.
Entries are tiny (~5 KB), so 256 of them cost ~1 MB of RAM. The cache is cleared whenever
the index changes, so it never serves an answer built from outdated documents.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict

from app.schemas.rag import RAGAnswer

_SPACE = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s?.!]+$")


def cache_key(question: str, top_k: int, model: str) -> str:
    """Case/whitespace/trailing-punctuation insensitive key."""
    q = _TRAILING.sub("", _SPACE.sub(" ", question.strip().lower()))
    return f"{model}|{top_k}|{q}"


class AnswerCache:
    def __init__(self, max_entries: int = 256, ttl_s: float = 3600.0, clock=time.monotonic):
        self.max_entries, self.ttl_s, self.clock = max_entries, ttl_s, clock
        self._data: OrderedDict[str, tuple[float, RAGAnswer]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_s > 0

    def get(self, key: str) -> RAGAnswer | None:
        if not self.enabled:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, answer = item
            if self.clock() - stored_at > self.ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)  # most recently used
            return answer

    def put(self, key: str, answer: RAGAnswer) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (self.clock(), answer)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # evict least recently used

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
