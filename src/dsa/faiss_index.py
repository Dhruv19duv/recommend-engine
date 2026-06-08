"""FAISS ANN index for sub-millisecond retrieval over 256-dimensional embeddings.

Manages 350M product embeddings with HNSW+IVF indexing for high-throughput,
low-latency similarity search.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Wrapper around FAISS for ANN similarity search.

    Uses IVF with HNSW coarse quantizer for fast, accurate retrieval.
    Indexes 256-dimensional product embeddings at 350M scale.
    """

    def __init__(
        self,
        dimension: int = 256,
        index_type: str = "IVF65536_HNSW32,PQ64",
        nprobe: int = 128,
        nlist: int = 65536,
        metric: str = "inner_product",
    ) -> None:
        self.dimension = dimension
        self.index_type = index_type
        self.nprobe = nprobe
        self.nlist = nlist
        self.metric = metric

        self._index = None
        self._id_map: Dict[int, str] = {}  # FAISS local_id -> item_id
        self._item_to_faiss_id: Dict[str, int] = {}
        self._next_id: int = 0
        self._trained: bool = False
        self._built: bool = False

    def _build_index(self) -> None:
        """Build the FAISS index based on configuration."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required. Install: pip install faiss-cpu")

        if self.metric == "inner_product":
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            metric = faiss.METRIC_L2
        else:
            metric = faiss.METRIC_INNER_PRODUCT

        # Parse index type components
        # e.g., "IVF65536_HNSW32,PQ64" -> IVF with HNSW coarse quantizer + PQ encoding
        if "HNSW" in self.index_type:
            # Build HNSW coarse quantizer
            coarse_quantizer = faiss.IndexHNSWFlat(self.dimension, 32)
            coarse_quantizer.hnsw.efConstruction = 200
            self._index = faiss.IndexIVFPQ(
                coarse_quantizer,
                self.dimension,
                self.nlist,
                64,  # PQ m (sub-vectors)
                8,   # PQ bits per sub-vector
                metric,
            )
        elif "PQ" in self.index_type:
            self._index = faiss.IndexIVFPQ(
                faiss.IndexFlatIP(self.dimension),
                self.dimension,
                self.nlist,
                64, 8, metric,
            )
        else:
            self._index = faiss.IndexFlatIP(self.dimension)

        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.nprobe

        self._built = True

    def train(self, embeddings: np.ndarray) -> None:
        """Train the index on a sample of embeddings."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required")

        if not self._built:
            self._build_index()

        if self._index is not None and self._index.is_trained:
            logger.info("Index already trained, skipping")
            self._trained = True
            return

        logger.info(f"Training FAISS index on {len(embeddings)} vectors...")
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        if self._index is not None:
            self._index.train(embeddings)
            self._trained = True
            logger.info("FAISS index training complete")

    def add(self, item_ids: List[str], embeddings: np.ndarray) -> None:
        """Add items to the index.

        Args:
            item_ids: List of product IDs
            embeddings: (n_items, dimension) float32 array
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required")

        if not self._trained:
            raise RuntimeError("Index not trained. Call train() first.")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        n = len(item_ids)
        faiss_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)

        for i, item_id in enumerate(item_ids):
            fid = self._next_id + i
            self._id_map[fid] = item_id
            self._item_to_faiss_id[item_id] = fid

        if self._index is not None:
            self._index.add_with_ids(embeddings, faiss_ids)
        self._next_id += n
        logger.info(f"Added {n} items to FAISS index (total: {self._next_id})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        filter_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[float]]:
        """Search the index for nearest neighbors.

        Args:
            query_embedding: (dimension,) or (1, dimension) array
            top_k: Number of results to return
            filter_ids: Optional whitelist of item IDs to restrict results

        Returns:
            (item_ids, scores) where each is a list of length top_k
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required")

        if self._index is None or self._index.ntotal == 0:
            return [], []

        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)

        if filter_ids:
            # Filter-based search: encode whitelist as a bitset
            # For simplicity, search all then post-filter
            k_search = min(top_k * 10, self._index.ntotal)
            distances, indices = self._index.search(query, int(k_search))
            return self._filter_results(distances[0], indices[0], filter_ids, top_k)

        distances, indices = self._index.search(query, top_k)
        return self._format_results(distances[0], indices[0])

    def _format_results(
        self, distances: np.ndarray, indices: np.ndarray
    ) -> Tuple[List[str], List[float]]:
        """Convert FAISS search results to item IDs and scores."""
        item_ids: List[str] = []
        scores: List[float] = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            item_id = self._id_map.get(int(idx))
            if item_id is not None:
                item_ids.append(item_id)
                scores.append(float(distances[i]))
        return item_ids, scores

    def _filter_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        filter_ids: List[str],
        top_k: int,
    ) -> Tuple[List[str], List[float]]:
        """Post-filter results to whitelist."""
        filter_set = set(filter_ids)
        item_ids: List[str] = []
        scores: List[float] = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            item_id = self._id_map.get(int(idx))
            if item_id is not None and item_id in filter_set:
                item_ids.append(item_id)
                scores.append(float(distances[i]))
                if len(item_ids) >= top_k:
                    break
        return item_ids, scores

    def remove(self, item_ids: List[str]) -> None:
        """Remove items from the index by ID."""
        ids_to_remove = [
            self._item_to_faiss_id[item_id]
            for item_id in item_ids
            if item_id in self._item_to_faiss_id
        ]
        if self._index is not None and ids_to_remove:
            self._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            for item_id in item_ids:
                fid = self._item_to_faiss_id.pop(item_id, None)
                self._id_map.pop(fid, None)

    def save(self, path: str) -> None:
        """Save the index and metadata to disk."""
        import faiss

        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump({
                "id_map": self._id_map,
                "item_to_faiss_id": self._item_to_faiss_id,
                "next_id": self._next_id,
                "dimension": self.dimension,
                "trained": self._trained,
            }, f)
        logger.info(f"Index saved to {path}")

    def load(self, path: str) -> None:
        """Load the index and metadata from disk."""
        import faiss

        self._index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self._id_map = meta["id_map"]
            self._item_to_faiss_id = meta["item_to_faiss_id"]
            self._next_id = meta["next_id"]
            self._trained = meta["trained"]
        self._built = True
        logger.info(f"Index loaded from {path} ({self._index.ntotal} vectors)")

    @property
    def total_items(self) -> int:
        """Number of items currently indexed."""
        return self._index.ntotal if self._index is not None else 0

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Reconstruct the embedding for a given item ID."""
        import faiss

        if self._index is None:
            return None
        fid = self._item_to_faiss_id.get(item_id)
        if fid is None:
            return None
        try:
            return faiss.reconstruct(self._index, fid)
        except Exception:
            return None
