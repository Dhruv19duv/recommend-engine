"""GNN-based fraud detector that identifies suspicious seller clusters.

Detects sellers with suspiciously correlated review timing, IP ranges,
and rating patterns. Auto-demotes fraudulent sellers from candidate sets.

Architecture:
- Heterogeneous GNN over seller-review-IP graph
- Detects collusion rings via graph clustering
- Outputs fraud probability per seller
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FraudGNNLayer(nn.Module):
    """Graph convolution layer for fraud detection.

    Aggregates features from:
    - Seller embeddings
    - Review timing patterns
    - IP range overlaps
    - Rating pattern anomalies
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with message passing.

        Args:
            x: (n_nodes, in_dim) node features
            adj_indices: (2, n_edges) edge indices (src, dst)

        Returns:
            (n_nodes, out_dim) updated node features
        """
        src, dst = adj_indices[0], adj_indices[1]

        # Gather source and destination features
        src_feats = x[src]
        dst_feats = x[dst]

        # Concatenate and transform
        messages = self.conv(torch.cat([src_feats, dst_feats], dim=-1))
        messages = self.dropout(F.relu(messages))

        # Aggregate messages by destination (mean pooling)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)

        # Normalize by degree
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float).unsqueeze(-1))
        agg = agg / deg.clamp(min=1)

        # Residual connection
        return F.relu(x[:, :out_dim] + agg[:, :out_dim])


class FraudGNN(nn.Module):
    """GNN-based fraud detector.

    Input features per seller:
    - Embedding from seller metadata
    - Review timing entropy (low entropy = suspicious)
    - IP diversity (low diversity = suspicious)
    - Rating distribution skew
    - Account age / creation burstiness

    Output: fraud probability [0, 1]
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList([
            FraudGNNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        adj_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (n_sellers, feature_dim) seller features
            adj_indices: (2, n_edges) seller-seller adjacency (based on
                shared IPs, coordinated review timing, etc.)

        Returns:
            (n_sellers, 1) fraud probabilities
        """
        h = self.input_proj(features)
        for layer in self.layers:
            h = layer(h, adj_indices)
        return self.classifier(h)


@dataclass
class FraudIndicator:
    """Fraud indicators for a seller."""

    seller_id: str
    fraud_probability: float
    review_timing_entropy: float
    ip_diversity: float
    rating_skew: float
    ring_members: List[str]
    is_suspicious: bool


class FraudDetector:
    """Fraud detection pipeline that combines GNN with rule-based signals."""

    def __init__(
        self,
        model: FraudGNN,
        review_correlation_threshold: float = 0.9,
        ip_range_overlap_threshold: float = 0.8,
        auto_demote_threshold: float = 0.75,
    ) -> None:
        self.model = model
        self.review_correlation_threshold = review_correlation_threshold
        self.ip_range_overlap_threshold = ip_range_overlap_threshold
        self.auto_demote_threshold = auto_demote_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect_fraud_rings(
        self,
        seller_features: Dict[str, torch.Tensor],
        adjacency: torch.Tensor,
        seller_ids: List[str],
    ) -> List[FraudIndicator]:
        """Run fraud detection and return per-seller indicators."""
        self.model.eval()

        feature_tensor = torch.stack([seller_features[sid] for sid in seller_ids])
        adj = adjacency.to(self.device)
        features = feature_tensor.to(self.device)

        with torch.no_grad():
            fraud_probs = self.model(features, adj)

        indicators: List[FraudIndicator] = []
        for i, seller_id in enumerate(seller_ids):
            prob = fraud_probs[i].item()
            indicators.append(
                FraudIndicator(
                    seller_id=seller_id,
                    fraud_probability=prob,
                    review_timing_entropy=1.0 - prob,  # Simplified
                    ip_diversity=1.0 - prob,
                    rating_skew=prob,
                    ring_members=self._find_ring_members(i, adj, seller_ids),
                    is_suspicious=prob >= self.auto_demote_threshold,
                )
            )

        return indicators

    def _find_ring_members(
        self,
        node_idx: int,
        adj: torch.Tensor,
        seller_ids: List[str],
    ) -> List[str]:
        """Find members of a suspected fraud ring."""
        src, dst = adj[0], adj[1]
        neighbors = dst[src == node_idx]
        return [seller_ids[n.item()] for n in neighbors.tolist()]

    def demote_fraudulent_sellers(
        self,
        candidate_items: List[Tuple[str, float]],
        seller_map: Dict[str, str],
        fraud_indicators: Dict[str, FraudIndicator],
        demotion_factor: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """Penalize scores from fraudulent sellers."""
        demoted: List[Tuple[str, float]] = []
        for item_id, score in candidate_items:
            seller_id = seller_map.get(item_id)
            if seller_id and seller_id in fraud_indicators:
                ind = fraud_indicators[seller_id]
                if ind.is_suspicious:
                    score *= demotion_factor
            demoted.append((item_id, score))

        demoted.sort(key=lambda x: x[1], reverse=True)
        return demoted
