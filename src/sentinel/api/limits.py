"""Lightweight, dependency-free guards so the public API survives load and abuse.

Single-process and in-memory: a per-client sliding-window rate limiter and (in
app.py) a global concurrency cap on the expensive inference route. For a
multi-worker deployment, move rate limiting to the reverse proxy (nginx
`limit_req`); these bound one process and degrade gracefully (429/503) rather
than letting the box run out of memory.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

# Sweep stale per-client buckets once the table grows past this many keys, so a
# long-running public server doesn't leak memory across many unique IPs.
_SWEEP_THRESHOLD = 10_000


class RateLimiter:
    """Sliding-window per-key rate limit. ``allow(key)`` is O(1) amortized."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        cutoff = moment - self._window
        with self._lock:
            if len(self._hits) > _SWEEP_THRESHOLD:
                self._sweep(cutoff)
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self._max:
                return False
            bucket.append(moment)
            return True

    def _sweep(self, cutoff: float) -> None:
        """Drop buckets with no hits inside the current window. Caller holds the lock."""
        stale = [key for key, bucket in self._hits.items() if not bucket or bucket[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
