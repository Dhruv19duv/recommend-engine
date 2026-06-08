"""Consistent hashing for sharding 400M user profiles across nodes.

Uses virtual nodes for even load distribution and minimal reshuffling
when nodes are added or removed.
"""

from __future__ import annotations

import hashlib
import struct
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

import mmh3


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes for even load distribution.

    Distributes user profiles across shards with O(log n) lookup.
    Adding/removing a node only reshuffles 1/n of the keys.
    """

    def __init__(
        self,
        num_virtual_nodes: int = 100,
        num_shards: int = 128,
    ) -> None:
        self.num_virtual_nodes = num_virtual_nodes
        self.num_shards = num_shards
        self._ring: List[Tuple[int, int]] = []  # (hash, shard_id)
        self._shard_to_vnodes: Dict[int, List[int]] = {}
        self._sorted = False

    def _hash(self, key: str, seed: int = 0) -> int:
        """Compute a 64-bit hash using mmh3."""
        return mmh3.hash64(key, seed=seed, signed=False)[0]

    def build_ring(self, shard_ids: Optional[List[int]] = None) -> None:
        """Build the hash ring from shard IDs. If None, uses 0..num_shards-1."""
        if shard_ids is None:
            shard_ids = list(range(self.num_shards))

        self._ring = []
        self._shard_to_vnodes = {}

        for shard_id in shard_ids:
            vnodes: List[int] = []
            for vnode in range(self.num_virtual_nodes):
                key = f"{shard_id}:{vnode}"
                h = self._hash(key)
                self._ring.append((h, shard_id))
                vnodes.append(h)
            self._shard_to_vnodes[shard_id] = vnodes

        self._ring.sort(key=lambda x: x[0])
        self._sorted = True

    def get_shard(self, user_id: str) -> int:
        """Get the shard responsible for a user ID.

        Returns the nearest shard clockwise from the user's hash position.
        """
        if not self._sorted:
            self.build_ring()
        user_hash = self._hash(user_id)
        idx = bisect_right([h for h, _ in self._ring], user_hash)
        if idx >= len(self._ring):
            idx = 0
        return self._ring[idx][1]

    def get_shards_for_user(self, user_id: str, replication_factor: int = 3) -> List[int]:
        """Get multiple shards for a user (for replication)."""
        if not self._sorted:
            self.build_ring()
        user_hash = self._hash(user_id)
        idx = bisect_right([h for h, _ in self._ring], user_hash)
        shards: List[int] = []
        for i in range(replication_factor):
            pos = (idx + i) % len(self._ring)
            shard = self._ring[pos][1]
            if shard not in shards:
                shards.append(shard)
        return shards

    def add_shard(self, shard_id: int) -> None:
        """Add a new shard to the ring."""
        self._shard_to_vnodes[shard_id] = []
        for vnode in range(self.num_virtual_nodes):
            key = f"{shard_id}:{vnode}"
            h = self._hash(key)
            self._ring.append((h, shard_id))
            self._shard_to_vnodes[shard_id].append(h)
        self._ring.sort(key=lambda x: x[0])
        self.num_shards += 1

    def remove_shard(self, shard_id: int) -> None:
        """Remove a shard from the ring."""
        if shard_id in self._shard_to_vnodes:
            hashes = set(self._shard_to_vnodes[shard_id])
            self._ring = [(h, s) for h, s in self._ring if h not in hashes]
            del self._shard_to_vnodes[shard_id]
            self.num_shards -= 1

    def get_load_distribution(self) -> Dict[int, float]:
        """Get the fraction of the ring owned by each shard."""
        if not self._sorted or not self._ring:
            return {}
        total_arc = 2**64
        distribution: Dict[int, float] = {}
        for i in range(len(self._ring)):
            shard = self._ring[i][1]
            next_hash = self._ring[(i + 1) % len(self._ring)][0]
            curr_hash = self._ring[i][0]
            arc_length = (next_hash - curr_hash) % total_arc
            distribution[shard] = distribution.get(shard, 0.0) + arc_length / total_arc
        return distribution


class ShardedUserProfileStore:
    """Distributed user profile store using consistent hashing.

    Provides read/write access to user profiles sharded across nodes.
    Replication ensures fault tolerance.
    """

    def __init__(
        self,
        num_shards: int = 128,
        replication_factor: int = 3,
        num_virtual_nodes: int = 100,
    ) -> None:
        self.hash_ring = ConsistentHashRing(
            num_virtual_nodes=num_virtual_nodes,
            num_shards=num_shards,
        )
        self.hash_ring.build_ring()
        self.replication_factor = replication_factor
        self._stores: Dict[int, Dict[str, Any]] = {}

    def _get_or_create_shard(self, shard_id: int) -> Dict[str, Any]:
        """Get the in-memory store for a shard."""
        if shard_id not in self._stores:
            self._stores[shard_id] = {}
        return self._stores[shard_id]

    def set_user_profile(self, user_id: str, profile: Any) -> None:
        """Store a user profile across its shards."""
        shards = self.hash_ring.get_shards_for_user(user_id, self.replication_factor)
        for shard_id in shards:
            store = self._get_or_create_shard(shard_id)
            store[user_id] = profile

    def get_user_profile(self, user_id: str) -> Optional[Any]:
        """Get a user profile from its primary shard."""
        shard_id = self.hash_ring.get_shard(user_id)
        store = self._get_or_create_shard(shard_id)
        return store.get(user_id)

    def delete_user_profile(self, user_id: str) -> None:
        """Delete a user profile from all its shards."""
        shards = self.hash_ring.get_shards_for_user(user_id, self.replication_factor)
        for shard_id in shards:
            store = self._get_or_create_shard(shard_id)
            store.pop(user_id, None)

    @property
    def total_profiles(self) -> int:
        """Total number of user profiles across all shards."""
        return sum(len(store) for store in self._stores.values())
