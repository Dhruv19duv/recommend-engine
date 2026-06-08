"""Two-level LRU/LFU hybrid cache with Kafka-triggered invalidation.

L1: Top 1% hottest users (small, fast, purely LRU)
L2: Warm users (larger, hybrid LRU-LFU with frequency bias)
Invalidation: Kafka consumer drives cache eviction on purchase events.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry with metadata for eviction decisions."""

    key: str
    value: Any
    frequency: int = 1  # Access count
    last_access: float = 0.0  # Unix timestamp
    created_at: float = 0.0
    ttl: float = 300.0  # Time-to-live in seconds
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def lfu_score(self) -> float:
        """LFU score: higher frequency = more valuable."""
        age = time.time() - self.created_at + 1
        return self.frequency / age


class LRUCache:
    """Pure LRU cache for L1 (top 1% hottest users)."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get a value, moving it to the front (most recently used)."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired:
                if entry is not None:
                    del self._cache[key]
                return None
            self._cache.move_to_end(key)
            entry.frequency += 1
            entry.last_access = time.time()
            return entry.value

    def put(self, key: str, value: Any, ttl: float = 300.0) -> None:
        """Insert or update a cache entry."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.value = value
                entry.last_access = time.time()
                entry.frequency += 1
                entry.ttl = ttl
            else:
                while len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)  # Remove LRU
                self._cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    last_access=time.time(),
                    created_at=time.time(),
                    ttl=ttl,
                )

    def invalidate(self, key: str) -> None:
        """Remove a key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def invalidate_many(self, keys: List[str]) -> None:
        """Remove multiple keys from cache."""
        with self._lock:
            for key in keys:
                self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class LFUHybridCache:
    """Hybrid LRU-LFU cache for L2 (warm users).

    Eviction policy: evict entries with lowest LFU score within LRU tail.
    Balances recency (LRU) and popularity (LFU).
    """

    def __init__(
        self,
        capacity: int = 10_000_000,
        lru_fraction: float = 0.3,  # 30% weight on recency
        lfu_fraction: float = 0.7,  # 70% weight on frequency
    ) -> None:
        self.capacity = capacity
        self.lru_fraction = lru_fraction
        self.lfu_fraction = lfu_fraction
        self._cache: Dict[str, CacheEntry] = {}
        self._frequency_buckets: Dict[int, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._eviction_counter: int = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value, updating frequency."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired:
                if entry is not None:
                    self._remove_entry(key)
                return None

            # Update frequency
            old_freq = entry.frequency
            entry.frequency += 1
            entry.last_access = time.time()
            self._frequency_buckets[old_freq].discard(key)
            self._frequency_buckets[entry.frequency].add(key)
            return entry.value

    def put(self, key: str, value: Any, ttl: float = 60.0) -> None:
        """Insert or update a cache entry."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.last_access = time.time()
                entry.frequency += 1
                entry.ttl = ttl
            else:
                while len(self._cache) >= self.capacity:
                    self._evict_one()
                self._cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    last_access=time.time(),
                    created_at=time.time(),
                    ttl=ttl,
                )
                self._frequency_buckets[1].add(key)

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from all internal structures."""
        entry = self._cache.pop(key, None)
        if entry:
            self._frequency_buckets[entry.frequency].discard(key)

    def _evict_one(self) -> None:
        """Evict the lowest-scoring entry."""
        if not self._cache:
            return

        now = time.time()
        best_key: Optional[str] = None
        best_score = float("inf")

        # Sample from frequency buckets for efficiency
        # Focus on lowest-frequency buckets first
        for freq in sorted(self._frequency_buckets.keys()):
            candidates = self._frequency_buckets[freq]
            if not candidates:
                continue

            for key in list(candidates)[:10]:  # Sample up to 10 per bucket
                entry = self._cache.get(key)
                if entry is None:
                    continue

                # Composite score: lower = better to evict
                recency = 1.0 / (now - entry.last_access + 1)
                freq_score = entry.frequency / (now - entry.created_at + 1)
                score = -(self.lru_fraction * recency + self.lfu_fraction * freq_score)

                if score < best_score:
                    best_score = score
                    best_key = key

            if best_key is not None:
                break

        if best_key is not None:
            self._remove_entry(best_key)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._remove_entry(key)

    def invalidate_many(self, keys: List[str]) -> None:
        with self._lock:
            for key in keys:
                self._remove_entry(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._frequency_buckets.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class TwoLevelCache:
    """Two-level L1/LFU hybrid cache with Kafka-triggered invalidation.

    L1 (LRU): Top 1% hottest users — fastest access
    L2 (Hybrid): Warm users — larger capacity with frequency bias

    Invalidated by Kafka consumer on purchase/inventory events.
    """

    def __init__(
        self,
        l1_capacity: int = 100_000,
        l2_capacity: int = 10_000_000,
        l1_ttl: float = 300.0,
        l2_ttl: float = 60.0,
    ) -> None:
        self.l1 = LRUCache(capacity=l1_capacity)
        self.l2 = LFUHybridCache(capacity=l2_capacity)
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl

        # Hottest users (promoted to L1)
        self._hot_users: Set[str] = set()
        self._hot_user_max: int = int(l1_capacity * 0.8)  # 80% of L1 for users

        self._lock = threading.RLock()
        self._invalidation_listeners: List[Callable[[str], None]] = []

    def get(self, key: str, user_id: Optional[str] = None) -> Optional[Any]:
        """Get from L1 first, then L2. Promotes hot keys."""
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Promote to L1
            self.l1.put(key, value, ttl=self.l1_ttl)
            if user_id and user_id not in self._hot_users:
                self._promote_user(user_id)
            return value

        return None

    def put(self, key: str, value: Any, user_id: Optional[str] = None) -> None:
        """Insert into L1 for hot users, L2 otherwise."""
        if user_id and user_id in self._hot_users:
            self.l1.put(key, value, ttl=self.l1_ttl)
        else:
            self.l2.put(key, value, ttl=self.l2_ttl)

    def _promote_user(self, user_id: str) -> None:
        """Promote a user to the hot list (L1)."""
        with self._lock:
            if len(self._hot_users) < self._hot_user_max:
                self._hot_users.add(user_id)

    def invalidate(self, key: str) -> None:
        """Invalidate a key from both caches and notify listeners."""
        self.l1.invalidate(key)
        self.l2.invalidate(key)
        for listener in self._invalidation_listeners:
            try:
                listener(key)
            except Exception:
                logger.exception("Invalidation listener failed")

    def invalidate_many(self, keys: List[str]) -> None:
        """Invalidate multiple keys."""
        self.l1.invalidate_many(keys)
        self.l2.invalidate_many(keys)

    def handle_kafka_invalidation(self, purchase_event: Dict[str, Any]) -> None:
        """Handle a Kafka purchase event — invalidate affected caches.

        Called by the Kafka consumer for every purchase event.
        Ensures the user's recommendation cache is invalidated.
        """
        user_id = purchase_event.get("user_id")
        product_ids = purchase_event.get("product_ids", [])

        if user_id:
            cache_keys = [f"recs:{user_id}", user_id]
            for pid in product_ids:
                cache_keys.append(f"recs:{user_id}:{pid}")
            self.invalidate_many(cache_keys)
            logger.debug(f"Invalidated cache for user {user_id} after purchase")

    def add_invalidation_listener(self, listener: Callable[[str], None]) -> None:
        self._invalidation_listeners.append(listener)

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()
        self._hot_users.clear()

    @property
    def total_size(self) -> int:
        return self.l1.size + self.l2.size
