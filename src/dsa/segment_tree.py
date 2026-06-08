"""Segment Tree for O(log n) range queries on freshness-weighted interactions.

Used to compute aggregate metrics over sliding time windows (e.g., "total
interactions in the last 7 days") without scanning all events.

Operations:
- Range sum/min/max query: O(log n)
- Point update (add interaction): O(log n)
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T", int, float)


class SegmentTree(Generic[T]):
    """Generic segment tree for range queries over time-bucketed data.

    Stores interaction counts/scores in time buckets (e.g., hourly buckets)
    and supports fast sliding-window aggregation.
    """

    def __init__(
        self,
        num_buckets: int,
        default_value: T = 0,
        agg_fn: Callable[[T, T], T] = lambda a, b: a + b,
    ) -> None:
        self.num_buckets = num_buckets
        self.default_value = default_value
        self.agg_fn = agg_fn
        self.size = 1
        while self.size < num_buckets:
            self.size <<= 1
        self.tree: List[T] = [default_value] * (2 * self.size)

    def update(self, index: int, value: T) -> None:
        """Update bucket at index by aggregating value (typically adding)."""
        if not 0 <= index < self.num_buckets:
            raise IndexError(f"Index {index} out of range [0, {self.num_buckets})")
        i = index + self.size
        self.tree[i] = self.agg_fn(self.tree[i], value)
        i >>= 1
        while i:
            self.tree[i] = self.agg_fn(self.tree[2 * i], self.tree[2 * i + 1])
            i >>= 1

    def set(self, index: int, value: T) -> None:
        """Set bucket at index to an absolute value."""
        if not 0 <= index < self.num_buckets:
            raise IndexError(f"Index {index} out of range [0, {self.num_buckets})")
        i = index + self.size
        self.tree[i] = value
        i >>= 1
        while i:
            self.tree[i] = self.agg_fn(self.tree[2 * i], self.tree[2 * i + 1])
            i >>= 1

    def query(self, left: int, right: int) -> T:
        """Range query over [left, right) (half-open interval). O(log n)."""
        if left < 0:
            left = 0
        if right > self.num_buckets:
            right = self.num_buckets
        if left >= right:
            return self.default_value

        left += self.size
        right += self.size
        res_left = self.default_value
        res_right = self.default_value

        while left < right:
            if left & 1:
                res_left = self.agg_fn(res_left, self.tree[left])
                left += 1
            if right & 1:
                right -= 1
                res_right = self.agg_fn(self.tree[right], res_right)
            left >>= 1
            right >>= 1

        return self.agg_fn(res_left, res_right)

    def query_all(self) -> T:
        """Get the aggregate over all buckets."""
        return self.tree[1]


@dataclass
class FreshnessWeightedWindow:
    """Sliding window over time buckets with freshness decay.

    Each interaction is bucketed into a time slot and decayed by recency.
    The segment tree allows O(log n) sliding-window queries.
    """

    bucket_span_seconds: int = 3600  # 1 hour buckets
    num_buckets: int = 24 * 365  # 1 year of hourly buckets
    half_life_seconds: int = 14 * 86400  # 14 days

    def __post_init__(self) -> None:
        """Initialize the segment tree."""
        self._tree = SegmentTree[int](self.num_buckets)

    def _bucket_index(self, timestamp: float) -> int:
        """Convert a Unix timestamp to a bucket index."""
        return int(timestamp) // self.bucket_span_seconds % self.num_buckets

    def _freshness_weight(self, age_seconds: float) -> float:
        """Exponential decay: w = 2^(-age / half_life)."""
        return 2.0 ** (-age_seconds / self.half_life_seconds)

    def record_interaction(self, timestamp: float, weight: float = 1.0) -> None:
        """Record an interaction with freshness weighting."""
        idx = self._bucket_index(timestamp)
        self._tree.update(idx, int(weight * 1000))  # Store as integer millis

    def query_window(self, start_time: float, end_time: float) -> int:
        """Get total interaction weight over [start_time, end_time)."""
        start_bucket = self._bucket_index(start_time)
        end_bucket = self._bucket_index(end_time)

        if end_bucket > start_bucket:
            return self._tree.query(start_bucket, end_bucket)
        else:
            # Wraparound case
            return self._tree.query(start_bucket, self.num_buckets) + self._tree.query(0, end_bucket)

    def query_last_n_days(self, days: int, now: Optional[float] = None) -> int:
        """Get total interaction weight in the last N days."""
        now = now or 1735689600.0  # Default: ~Jan 2025
        start = now - days * 86400
        return self.query_window(start, now)


@dataclass
class MultiCriteriaMinHeap:
    """Min-heap that maintains top-K items by multi-criteria scoring.

    Combines: relevance_score, recency, inventory_level, margin_weight
    into a single scalar via weighted sum.
    """

    relevance_weight: float = 0.55
    recency_weight: float = 0.20
    inventory_weight: float = 0.10
    margin_weight: float = 0.15

    def __post_init__(self) -> None:
        self._heap: List[Tuple[float, str, float, float, float, float]] = []
        # (composite_score, item_id, relevance, recency, inventory, margin)

    def _composite_score(
        self,
        relevance: float,
        recency: float,
        inventory: float,
        margin: float,
    ) -> float:
        """Compute the weighted composite score."""
        return (
            self.relevance_weight * relevance
            + self.recency_weight * recency
            + self.inventory_weight * inventory
            + self.margin_weight * margin
        )

    def push(
        self,
        item_id: str,
        relevance: float = 0.0,
        recency: float = 0.0,
        inventory: float = 1.0,
        margin: float = 0.0,
    ) -> None:
        """Push an item onto the heap."""
        score = self._composite_score(relevance, recency, inventory, margin)
        # Negate for min-heap behavior (we want max score on top)
        heapq.heappush(self._heap, (-score, item_id, relevance, recency, inventory, margin))

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Get the top-K items by composite score.

        Returns: [(item_id, composite_score), ...] sorted descending.
        """
        results = heapq.nlargest(k, self._heap)
        return [(item_id, -neg_score) for neg_score, item_id, _, _, _, _ in results]
