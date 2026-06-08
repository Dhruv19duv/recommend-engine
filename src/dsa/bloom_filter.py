"""Bloom Filter for O(1) "has the user already seen this item" checks.

Uses mmh3 for fast hashing. Handles 1B+ items with configurable false positive rate.
"""

from __future__ import annotations

import math
from array import array
from typing import List, Optional, Set

import mmh3


class BloomFilter:
    """Space-efficient probabilistic membership test.

    Used to prevent duplicate recommendations within a session.
    False positives are possible (configurable <1%), false negatives are not.
    """

    def __init__(
        self,
        capacity: int = 1_000_000_000,
        false_positive_rate: float = 0.01,
    ) -> None:
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate

        # Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        self.num_bits = int(
            -capacity * math.log(false_positive_rate) / (math.log(2) ** 2)
        )
        # Optimal number of hash functions: k = (m / n) * ln(2)
        self.num_hashes = max(1, int((self.num_bits / capacity) * math.log(2)))

        # Use array of unsigned 8-bit integers (byte array)
        self.num_bytes = (self.num_bits + 7) // 8
        self._bit_array = array("B", [0]) * self.num_bytes

        # Counter for actual inserted items
        self._inserted_count: int = 0

    def _hashes(self, item: str) -> List[int]:
        """Compute all hash positions for an item."""
        seed1 = mmh3.hash(item, signed=False)
        seed2 = mmh3.hash(item, seed=42, signed=False)
        return [
            (seed1 + i * seed2) % self.num_bits
            for i in range(self.num_hashes)
        ]

    def add(self, item: str) -> None:
        """Insert an item into the bloom filter."""
        for pos in self._hashes(item):
            byte_idx = pos >> 3
            bit_idx = pos & 7
            self._bit_array[byte_idx] |= 1 << bit_idx
        self._inserted_count += 1

    def __contains__(self, item: str) -> bool:
        """Check if an item may have been inserted."""
        for pos in self._hashes(item):
            byte_idx = pos >> 3
            bit_idx = pos & 7
            if not (self._bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def check_many(self, items: List[str]) -> List[bool]:
        """Batch membership check. Returns list of booleans."""
        return [item in self for item in items]

    @property
    def estimated_fill_ratio(self) -> float:
        """Estimated proportion of bits set to 1."""
        bits_set = sum(bin(byte).count("1") for byte in self._bit_array)
        return bits_set / (self.num_bytes * 8)

    @property
    def current_false_positive_rate(self) -> float:
        """Estimate the current false positive rate based on fill ratio."""
        p = self.estimated_fill_ratio
        return p ** self.num_hashes

    def clear(self) -> None:
        """Reset the filter."""
        for i in range(self.num_bytes):
            self._bit_array[i] = 0
        self._inserted_count = 0

    def __len__(self) -> int:
        return self._inserted_count


class SessionBloomFilter(BloomFilter):
    """Per-session bloom filter with automatic TTL-based expiry.

    Uses a ring buffer of bloom filters, each covering a time window.
    """

    def __init__(
        self,
        num_slots: int = 24,
        slot_duration_seconds: int = 3600,
        capacity: int = 100_000_000,
        false_positive_rate: float = 0.01,
    ) -> None:
        self.num_slots = num_slots
        # Don't call super().__init__ — SessionBloomFilter uses slot filters, not the large parent bit array
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate
        self.num_bits = 0
        self.num_bytes = 0
        self._bit_array = array("B")
        self._inserted_count = 0
        self.num_hashes = 4  # Fixed small number of hashes per slot
        self.slot_duration = slot_duration_seconds
        self._slot_filters: List[BloomFilter] = [
            BloomFilter(capacity=capacity // num_slots, false_positive_rate=false_positive_rate)
            for _ in range(num_slots)
        ]
        self._current_slot: int = 0

    def _get_slot(self, timestamp: float) -> int:
        """Get slot index for a given timestamp."""
        return int(timestamp) // self.slot_duration % self.num_slots

    def add_with_timestamp(self, item: str, timestamp: float) -> None:
        """Add item with a specific timestamp to the appropriate slot."""
        slot = self._get_slot(timestamp)
        if slot != self._current_slot:
            # Advance to current slot, clearing filters for slots we skip
            while self._current_slot != slot:
                self._current_slot = (self._current_slot + 1) % self.num_slots
                self._slot_filters[self._current_slot].clear()
        self._slot_filters[slot].add(item)

    def check_in_window(self, item: str, timestamp: float, window_slots: int = 4) -> bool:
        """Check if item appears in the last N time slots."""
        current_slot = self._get_slot(timestamp)
        for offset in range(window_slots):
            slot = (current_slot - offset) % self.num_slots
            if item in self._slot_filters[slot]:
                return True
        return False
