"""Heterogeneous Graph for multi-entity recommendation relationships.

Nodes: users, products, categories, sellers
Edges: viewed, bought, rated, bundled, searched, added_to_cart, saved_for_later

This graph powers the GraphSAGE message-passing and the life-stage diffusion
forking used by the five innovations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np


class NodeType(str, Enum):
    """Entity types in the heterogeneous graph."""

    USER = "user"
    PRODUCT = "product"
    CATEGORY = "category"
    SELLER = "seller"
    ATTRIBUTE = "attribute"  # e.g., brand, color, size


class EdgeType(str, Enum):
    """Relationship types between nodes."""

    VIEWED = "viewed"
    BOUGHT = "bought"
    RATED = "rated"
    BUNDLED = "bundled"
    ADDED_TO_CART = "added_to_cart"
    SAVED_FOR_LATER = "saved_for_later"
    SEARCHED = "searched"
    RETURNED = "returned"
    BELONGS_TO = "belongs_to"  # product -> category
    SELLS = "sells"  # seller -> product
    RECOMMENDED = "recommended"
    RECOMMENDED_AND_ACCEPTED = "recommended_and_accepted"


@dataclass
class Node:
    """A node in the heterogeneous graph."""

    node_id: str
    node_type: NodeType
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class Edge:
    """A typed, weighted edge between two nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    edge_id: str = field(default_factory=lambda: str(uuid4()))


class HeterogeneousGraph:
    """In-memory heterogeneous graph for recommendation relationships.

    Supports:
    - Typed nodes and edges with metadata
    - Subgraph extraction by node/edge type
    - N-hop neighbor sampling for GraphSAGE
    - Life-stage divergence detection for embedding forking
    - Freshness-weighted edge pruning
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._adjacency: Dict[str, Dict[str, List[Edge]]] = {}  # node_id -> {neighbor_id -> [edges]}
        self._node_type_index: Dict[NodeType, Set[str]] = {nt: set() for nt in NodeType}
        self._edge_type_index: Dict[EdgeType, List[Edge]] = {et: [] for et in EdgeType}

    # ── Node Operations ──────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: float = 0.0,
    ) -> Node:
        """Add or update a node in the graph."""
        node = Node(
            node_id=node_id,
            node_type=node_type,
            embedding=embedding,
            metadata=metadata or {},
            created_at=timestamp if node_id not in self._nodes else self._nodes[node_id].created_at,
            updated_at=timestamp,
        )
        self._nodes[node_id] = node
        self._node_type_index[node_type].add(node_id)
        if node_id not in self._adjacency:
            self._adjacency[node_id] = {}
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return False
        node = self._nodes[node_id]
        self._node_type_index[node.node_type].discard(node_id)

        # Remove all edges involving this node
        self._edges = [
            e for e in self._edges
            if e.source_id != node_id and e.target_id != node_id
        ]
        # Rebuild edge type index
        self._edge_type_index = {et: [] for et in EdgeType}
        for e in self._edges:
            self._edge_type_index[e.edge_type].append(e)

        # Remove adjacency entries
        self._adjacency.pop(node_id, None)
        for neighbor in self._adjacency.values():
            neighbor.pop(node_id, None)

        del self._nodes[node_id]
        return True

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a given type."""
        return [self._nodes[nid] for nid in self._node_type_index[node_type] if nid in self._nodes]

    # ── Edge Operations ──────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        timestamp: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """Add a typed, weighted edge between two nodes."""
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self._edges.append(edge)
        self._edge_type_index[edge_type].append(edge)

        # Build bidirectional adjacency
        if target_id not in self._adjacency.setdefault(source_id, {}):
            self._adjacency[source_id][target_id] = []
        self._adjacency[source_id][target_id].append(edge)

        if source_id not in self._adjacency.setdefault(target_id, {}):
            self._adjacency[target_id][source_id] = []
        self._adjacency[target_id][source_id].append(edge)

        return edge

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        """Get edges with optional filtering."""
        result = self._edges
        if source_id:
            result = [e for e in result if e.source_id == source_id]
        if target_id:
            result = [e for e in result if e.target_id == target_id]
        if edge_type:
            result = [e for e in result if e.edge_type == edge_type]
        return result

    # ── Neighbor Operations ──────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Set[EdgeType]] = None,
        max_hops: int = 1,
        max_per_hop: int = 50,
    ) -> Dict[int, List[Tuple[str, EdgeType, float]]]:
        """Get n-hop neighbors with optional edge type filtering.

        Returns: {hop: [(neighbor_id, edge_type, weight), ...]}
        """
        result: Dict[int, List[Tuple[str, EdgeType, float]]] = {}
        visited: Set[str] = {node_id}
        current_layer: Set[str] = {node_id}

        for hop in range(1, max_hops + 1):
            next_layer: Set[str] = set()
            hop_neighbors: List[Tuple[str, EdgeType, float]] = []

            for nid in current_layer:
                if nid not in self._adjacency:
                    continue
                for neighbor_id, edges in self._adjacency[nid].items():
                    if neighbor_id in visited:
                        continue
                    for edge in edges:
                        if edge_types is None or edge.edge_type in edge_types:
                            hop_neighbors.append((neighbor_id, edge.edge_type, edge.weight))
                    next_layer.add(neighbor_id)
                    visited.add(neighbor_id)

            # Limit per hop
            if len(hop_neighbors) > max_per_hop:
                hop_neighbors.sort(key=lambda x: x[2], reverse=True)
                hop_neighbors = hop_neighbors[:max_per_hop]

            if hop_neighbors:
                result[hop] = hop_neighbors
            current_layer = next_layer

        return result

    def sample_subgraph(
        self,
        seed_nodes: List[str],
        num_hops: int = 2,
        samples_per_hop: int = 20,
    ) -> HeterogeneousGraph:
        """Sample a subgraph around seed nodes for mini-batch training."""
        subgraph = HeterogeneousGraph()
        frontier: Set[str] = set(seed_nodes)

        for hop in range(num_hops):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                if node_id not in self._adjacency:
                    continue
                neighbors = list(self._adjacency[node_id].items())
                if len(neighbors) > samples_per_hop:
                    np.random.shuffle(neighbors)
                    neighbors = neighbors[:samples_per_hop]

                for neighbor_id, edges in neighbors:
                    if neighbor_id not in subgraph._nodes:
                        if neighbor_id in self._nodes:
                            n = self._nodes[neighbor_id]
                            subgraph.add_node(
                                node_id=n.node_id,
                                node_type=n.node_type,
                                embedding=n.embedding,
                                metadata=n.metadata.copy(),
                            )
                    for edge in edges:
                        subgraph.add_edge(
                            source_id=edge.source_id,
                            target_id=edge.target_id,
                            edge_type=edge.edge_type,
                            weight=edge.weight,
                            timestamp=edge.timestamp,
                            metadata=edge.metadata.copy() if edge.metadata else None,
                        )
                    next_frontier.add(neighbor_id)

            # Add seed nodes to subgraph
            for node_id in frontier:
                if node_id in self._nodes and node_id not in subgraph._nodes:
                    n = self._nodes[node_id]
                    subgraph.add_node(node_id=n.node_id, node_type=n.node_type)

            frontier = next_frontier

        return subgraph

    # ── Utility ──────────────────────────────────────────────────────

    def compute_degree(self, node_id: str) -> int:
        """Get the degree of a node."""
        return len(self._adjacency.get(node_id, {}))

    def prune_old_edges(self, before_timestamp: float) -> int:
        """Remove edges older than a timestamp. Returns count removed."""
        before = len(self._edges)
        self._edges = [e for e in self._edges if e.timestamp >= before_timestamp]

        # Rebuild adjacency and edge index
        self._adjacency = {}
        self._edge_type_index = {et: [] for et in EdgeType}
        for e in self._edges:
            self._edge_type_index[e.edge_type].append(e)
            self._adjacency.setdefault(e.source_id, {}).setdefault(e.target_id, []).append(e)
            self._adjacency.setdefault(e.target_id, {}).setdefault(e.source_id, []).append(e)

        return before - len(self._edges)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def to_dgl(self):
        """Convert to DGL heterogeneous graph for GNN training.

        This is a stub - real conversion requires DGL's dgl.heterograph() API.
        """
        # In production, construct a dict of (node_type, edge_type, node_type) -> (src, dst)
        raise NotImplementedError(
            "Use dgl.heterograph() with edge_tuples for production training. "
            "See: https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html"
        )
