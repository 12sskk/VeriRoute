"""Token-Bucket Rate Limiter per Tenant.

Implements O(1) lock-free/in-memory token bucket rate limiting with configurable
burst capacity and continuous token refill rate per tenant ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class BucketState:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Per-tenant Token Bucket Rate Limiter."""

    def __init__(self, capacity: float = 60.0, refill_per_sec: float = 2.0) -> None:
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._buckets: dict[str, BucketState] = {}

    def check_and_consume(self, tenant_id: str = "default", cost: float = 1.0) -> tuple[bool, float, float]:
        """Attempt to consume `cost` tokens for `tenant_id`.

        Returns:
            (allowed, remaining_tokens, retry_after_sec)
        """
        now = time.monotonic()
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            bucket = BucketState(tokens=self.capacity, last_refill=now)
            self._buckets[tenant_id] = bucket
        else:
            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_sec)
            bucket.last_refill = now

        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, round(bucket.tokens, 2), 0.0

        deficit = cost - bucket.tokens
        retry_after = deficit / max(1e-6, self.refill_per_sec)
        return False, round(bucket.tokens, 2), round(retry_after, 2)
