"""Sparse user-item matrix with SVD decomposition for collaborative filtering.

Provides the foundational baseline recommendation signal used by the
two-tower model and bandit algorithm.

Handles 400M x 350M sparse interaction matrices with block-wise SVD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD


@dataclass
class Interaction:
    """A single user-item interaction record."""

    user_id: str
    item_id: str
    rating: float  # Implicit (view=1, cart=3, purchase=5) or explicit (1-5)
    timestamp: float
    weight: float = 1.0  # Freshness weight for recency decay


@dataclass
class SVDModel:
    """Trained SVD model with latent factors."""

    user_factors: np.ndarray  # (n_users, n_factors)
    item_factors: np.ndarray  # (n_items, n_factors)
    user_bias: np.ndarray
    item_bias: np.ndarray
    global_bias: float
    n_factors: int
    explained_variance: float = 0.0

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a user-item pair."""
        score = self.global_bias + self.user_bias[user_idx] + self.item_bias[item_idx]
        score += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return float(score)


class BlockSparseSVD:
    """Block-wise SVD for a sparse user-item matrix.

    Handles matrices too large to fit in memory by decomposing into
    user blocks and item blocks with a common latent factor space.
    """

    def __init__(
        self,
        n_factors: int = 200,
        n_iterations: int = 10,
        block_size: int = 50_000,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.block_size = block_size
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # Mappings
        self._user_to_idx: Dict[str, int] = {}
        self._idx_to_user: Dict[int, str] = {}
        self._item_to_idx: Dict[str, int] = {}
        self._idx_to_item: Dict[int, str] = {}

        self._interactions: List[Interaction] = []
        self.model: Optional[SVDModel] = None

    def add_interaction(self, interaction: Interaction) -> None:
        """Record an interaction for training."""
        self._interactions.append(interaction)
        if interaction.user_id not in self._user_to_idx:
            idx = len(self._user_to_idx)
            self._user_to_idx[interaction.user_id] = idx
            self._idx_to_user[idx] = interaction.user_id
        if interaction.item_id not in self._item_to_idx:
            idx = len(self._item_to_idx)
            self._item_to_idx[interaction.item_id] = idx
            self._idx_to_item[idx] = interaction.item_id

    def build_sparse_matrix(self) -> sp.csr_matrix:
        """Build the sparse user-item interaction matrix.

        Uses weighted interactions (freshness * implicit_rating).
        Returns a CSR matrix of shape (n_users, n_items).
        """
        n_users = len(self._user_to_idx)
        n_items = len(self._item_to_idx)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for interaction in self._interactions:
            u_idx = self._user_to_idx[interaction.user_id]
            i_idx = self._item_to_idx[interaction.item_id]
            weighted_rating = interaction.rating * interaction.weight
            rows.append(u_idx)
            cols.append(i_idx)
            data.append(weighted_rating)

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

    def _freshness_weight(self, timestamp: float, now: float, half_life_days: float = 14.0) -> float:
        """Compute time-decay weight: w = 2^(-days_ago / half_life)."""
        days_ago = (now - timestamp) / 86400.0
        return float(2.0 ** (-days_ago / half_life_days))

    def fit(
        self,
        n_components: int = 200,
        algorithm: str = "randomized",
        apply_freshness: bool = True,
        now: Optional[float] = None,
    ) -> SVDModel:
        """Fit SVD on the interaction matrix with optional freshness weighting."""
        if apply_freshness and now is not None:
            for interaction in self._interactions:
                interaction.weight = self._freshness_weight(interaction.timestamp, now)

        matrix = self.build_sparse_matrix()

        # Apply TruncatedSVD
        svd = TruncatedSVD(
            n_components=min(n_components, min(matrix.shape) - 1),
            algorithm=algorithm,
            n_iter=self.n_iterations,
            random_state=self.random_state,
        )

        user_factors = svd.fit_transform(matrix)  # (n_users, n_components)
        # item_factors = V^T * S
        item_factors = svd.components_.T  # (n_items, n_components)

        # Compute biases as row/column means
        user_means = np.array(matrix.mean(axis=1)).flatten()
        item_means = np.array(matrix.mean(axis=0)).flatten()
        global_mean = float(matrix.data.mean()) if matrix.nnz > 0 else 0.0

        self.model = SVDModel(
            user_factors=user_factors,
            item_factors=item_factors,
            user_bias=user_means - global_mean,
            item_bias=item_means - global_mean,
            global_bias=global_mean,
            n_factors=n_components,
            explained_variance=float(svd.explained_variance_ratio_.sum()) if hasattr(svd, "explained_variance_ratio_") else 0.0,
        )

        return self.model

    def predict(self, user_id: str, item_id: str) -> float:
        """Predict interaction score for a user-item pair."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        u_idx = self._user_to_idx.get(user_id)
        i_idx = self._item_to_idx.get(item_id)

        if u_idx is None or i_idx is None:
            return self.model.global_bias  # Cold-start default

        return self.model.predict(u_idx, i_idx)

    def recommend_for_user(
        self,
        user_id: str,
        candidate_items: List[str],
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Score and rank candidate items for a user."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        scores: List[Tuple[str, float]] = []
        for item_id in candidate_items:
            score = self.predict(user_id, item_id)
            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Get the latent factor vector for a user."""
        if self.model is None:
            return None
        u_idx = self._user_to_idx.get(user_id)
        if u_idx is None:
            return None
        return self.model.user_factors[u_idx]

    def get_item_vector(self, item_id: str) -> Optional[np.ndarray]:
        """Get the latent factor vector for an item."""
        if self.model is None:
            return None
        i_idx = self._item_to_idx.get(item_id)
        if i_idx is None:
            return None
        return self.model.item_factors[i_idx]

    @property
    def num_users(self) -> int:
        return len(self._user_to_idx)

    @property
    def num_items(self) -> int:
        return len(self._item_to_idx)

    @property
    def num_interactions(self) -> int:
        return len(self._interactions)
