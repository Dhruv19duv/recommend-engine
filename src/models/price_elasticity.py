"""Per-user price elasticity model for personalized ranking.

High-sensitivity users get value-ranked results (best price-to-quality ratio).
Low-sensitivity users get quality-ranked results (highest-rated, regardless of price).

Model: Bayesian logistic regression on price vs. purchase probability.
Learns a price elasticity coefficient per user segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PriceElasticityProfile:
    """Price sensitivity profile per user."""

    user_id: str
    elasticity_coefficient: float  # Negative: more sensitive to price
    price_sensitivity: str  # "high", "medium", "low"
    preferred_price_range: Tuple[float, float]
    willingness_to_pay: float  # Max price they'll typically pay
    discount_sensitivity: float  # How much discounts affect their purchase
    value_quality_weight: float  # 0 = pure value, 1 = pure quality


class PriceElasticityModel(nn.Module):
    """Neural network that predicts price sensitivity from user features.

    Architecture:
    1. User feature encoder (purchase history, average order value,
       discount redemption rate, category distribution)
    2. Predicts elasticity coefficient and value-quality weight
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Head 1: Price elasticity coefficient
        self.elasticity_head = nn.Linear(hidden_dim, 1)

        # Head 2: Value vs quality preference weight (0-1)
        self.value_quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Head 3: Willingness to pay (log-normal)
        self.wtp_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        user_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            user_features: (batch_size, feature_dim)

        Returns:
            (elasticity, value_quality_weight, willingness_to_pay)
        """
        h = self.feature_encoder(user_features)
        elasticity = self.elasticity_head(h)
        vq_weight = self.value_quality_head(h)
        wtp = self.wtp_head(h)
        return elasticity, vq_weight, wtp


class ElasticityAwareRanker:
    """Re-ranks recommendation scores based on price elasticity.

    For high-sensitivity users: boosts items with better price-to-rating ratio.
    For low-sensitivity users: boosts items with higher ratings.
    """

    def __init__(
        self,
        model: PriceElasticityModel,
        elasticity_threshold_low: float = -0.5,
        elasticity_threshold_high: float = -2.0,
    ) -> None:
        self.model = model
        self.model.eval()
        self.elasticity_threshold_low = elasticity_threshold_low
        self.elasticity_threshold_high = elasticity_threshold_high
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def classify_sensitivity(self, elasticity: float) -> str:
        """Classify price sensitivity level."""
        if elasticity >= self.elasticity_threshold_low:
            return "low"
        elif elasticity >= self.elasticity_threshold_high:
            return "medium"
        else:
            return "high"

    def compute_adjusted_score(
        self,
        base_score: float,
        price: float,
        rating: float,
        value_quality_weight: float,
    ) -> float:
        """Compute price-adjusted score.

        score = (1 - vq_weight) * value_score + vq_weight * quality_score

        Where:
        - value_score = rating / max(price, 0.01)  (value = quality per dollar)
        - quality_score = rating  (pure quality, ignoring price)
        """
        if self.classify_sensitivity(-1.0) == "high":
            # Value-focused: quality per dollar
            value_score = rating / max(price, 0.01)
            quality_score = rating
        else:
            value_score = rating
            quality_score = rating * (1.0 + 0.1 * (5.0 - price))  # Slight price bonus

        adjusted = (1.0 - value_quality_weight) * value_score + value_quality_weight * quality_score
        return base_score * (0.5 + 0.5 * adjusted / max(rating, 1.0))

    def rerank(
        self,
        user_id: str,
        candidates: List[Tuple[str, float]],
        user_features: Dict[str, np.ndarray],
        item_prices: Dict[str, float],
        item_ratings: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """Rerank candidates based on price elasticity."""
        user_feat = user_features.get(user_id)
        if user_feat is None:
            return candidates  # No adjustment possible

        feature_tensor = torch.from_numpy(user_feat).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            elasticity, vq_weight, wtp = self.model(feature_tensor)

        elasticity_val = elasticity.item()
        vq_weight_val = vq_weight.item()
        sensitivity = self.classify_sensitivity(elasticity_val)

        logger.debug(
            f"User {user_id}: elasticity={elasticity_val:.3f}, "
            f"sensitivity={sensitivity}, vq_weight={vq_weight_val:.3f}"
        )

        reranked: List[Tuple[str, float]] = []
        for item_id, base_score in candidates:
            price = item_prices.get(item_id, 10.0)
            rating = item_ratings.get(item_id, 3.0)
            adjusted = self.compute_adjusted_score(base_score, price, rating, vq_weight_val)
            reranked.append((item_id, adjusted))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
