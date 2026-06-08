"""Tests for the Data Structures & Algorithms layer."""

import time

import numpy as np
import pytest

from src.dsa.bloom_filter import BloomFilter
from src.dsa.consistent_hash import ConsistentHashRing, ShardedUserProfileStore
from src.dsa.heterogeneous_graph import EdgeType, HeterogeneousGraph, NodeType
from src.dsa.segment_tree import FreshnessWeightedWindow, MultiCriteriaMinHeap, SegmentTree
from src.dsa.sparse_matrix import BlockSparseSVD, Interaction


class TestSegmentTree:
    def test_range_sum(self):
        tree = SegmentTree[int](num_buckets=10)
        tree.update(2, 5)
        tree.update(5, 3)
        tree.update(7, 8)
        assert tree.query(0, 10) == 16
        assert tree.query(2, 3) == 5
        assert tree.query(0, 3) == 5
        assert tree.query(5, 8) == 11

    def test_empty_query(self):
        tree = SegmentTree[int](num_buckets=10)
        assert tree.query(0, 10) == 0

    def test_set_value(self):
        tree = SegmentTree[int](num_buckets=5)
        tree.set(0, 10)
        tree.set(1, 20)
        assert tree.query(0, 2) == 30


class TestMultiCriteriaMinHeap:
    def test_top_k(self):
        heap = MultiCriteriaMinHeap()
        heap.push("item_1", relevance=0.9, recency=0.8, inventory=1.0, margin=0.7)
        heap.push("item_2", relevance=0.5, recency=0.3, inventory=0.5, margin=0.2)
        heap.push("item_3", relevance=0.8, recency=0.7, inventory=0.9, margin=0.6)

        top = heap.top_k(2)
        assert len(top) == 2
        assert top[0][0] in {"item_1", "item_3"}


class TestHeterogeneousGraph:
    def test_add_get_node(self):
        graph = HeterogeneousGraph()
        graph.add_node("user_1", NodeType.USER, timestamp=1000.0)
        node = graph.get_node("user_1")
        assert node is not None
        assert node.node_type == NodeType.USER

    def test_add_edge(self):
        graph = HeterogeneousGraph()
        graph.add_node("user_1", NodeType.USER)
        graph.add_node("prod_1", NodeType.PRODUCT)
        graph.add_edge("user_1", "prod_1", EdgeType.VIEWED, weight=1.0, timestamp=1000.0)

        edges = graph.get_edges(source_id="user_1", target_id="prod_1")
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.VIEWED

    def test_node_types(self):
        graph = HeterogeneousGraph()
        graph.add_node("user_1", NodeType.USER)
        graph.add_node("user_2", NodeType.USER)
        graph.add_node("prod_1", NodeType.PRODUCT)
        users = graph.get_nodes_by_type(NodeType.USER)
        assert len(users) == 2


class TestBloomFilter:
    def test_membership(self):
        bf = BloomFilter(capacity=1000, false_positive_rate=0.01)
        bf.add("item_1")
        bf.add("item_2")
        assert "item_1" in bf
        assert "item_2" in bf

    def test_false_positive_rate(self):
        bf = BloomFilter(capacity=1000, false_positive_rate=0.1)
        for i in range(500):
            bf.add(f"item_{i}")
        # Very unlikely to have false negatives
        for i in range(500):
            assert f"item_{i}" in bf


class TestConsistentHash:
    def test_shard_distribution(self):
        ring = ConsistentHashRing(num_virtual_nodes=10, num_shards=10)
        ring.build_ring()
        shards = [ring.get_shard(f"user_{i}") for i in range(1000)]
        assert len(set(shards)) > 1  # Distributes across shards

    def test_shard_stability(self):
        ring = ConsistentHashRing(num_virtual_nodes=10, num_shards=10)
        ring.build_ring()
        assert ring.get_shard("user_42") == ring.get_shard("user_42")


class TestShardedUserProfileStore:
    def test_read_write(self):
        store = ShardedUserProfileStore(num_shards=4, replication_factor=2)
        store.set_user_profile("user_1", {"name": "Alice"})
        profile = store.get_user_profile("user_1")
        assert profile == {"name": "Alice"}

    def test_delete(self):
        store = ShardedUserProfileStore(num_shards=4)
        store.set_user_profile("user_1", {"name": "Alice"})
        store.delete_user_profile("user_1")
        assert store.get_user_profile("user_1") is None


class TestBlockSparseSVD:
    def test_fit_and_predict(self):
        svd = BlockSparseSVD(n_factors=10)
        svd.add_interaction(Interaction("user_1", "item_1", rating=5.0, timestamp=1000))
        svd.add_interaction(Interaction("user_1", "item_2", rating=3.0, timestamp=1001))
        svd.add_interaction(Interaction("user_2", "item_1", rating=4.0, timestamp=1002))

        model = svd.fit(n_components=5)
        assert model is not None
        score = svd.predict("user_1", "item_1")
        assert isinstance(score, float)

    def test_recommend(self):
        svd = BlockSparseSVD(n_factors=10)
        for u in range(5):
            for i in range(10):
                svd.add_interaction(Interaction(
                    user_id=f"user_{u}",
                    item_id=f"item_{i}",
                    rating=float((u + i) % 5 + 1),
                    timestamp=float(1000 + i),
                ))
        svd.fit(n_components=5)
        recs = svd.recommend_for_user("user_1", [f"item_{i}" for i in range(10)], top_k=3)
        assert len(recs) == 3
