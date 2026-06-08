"""Utility modules for the recommendation engine.

Shared metrics, top-K scoring, and evaluation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg


@dataclass
class RecommendationResult:
    """Single recommendation result with scoring metadata."""

    item_id: str
    score: float
    relevance_score: float = 0.0
    recency_score: float = 0.0
    inventory_score: float = 1.0
    margin_score: float = 0.0
    discovery_score: float = 0.0
    regret_penalty: float = 0.0
    explanation: Optional[str] = None


@dataclass
class RecommendationResponse:
    """Full recommendation response for a user."""

    user_id: str
    results: List[RecommendationResult]
    request_id: str
    latency_ms: float = 0.0
    surface: str = "homepage"
    context: Optional[Dict[str, float]] = None


def compute_ndcg(
    relevance_scores: List[float],
    k: int = 50,
) -> float:
    """Compute Normalized Discounted Cumulative Gain @ k.

    Args:
        relevance_scores: Ground-truth relevance scores (sorted by predicted rank)
        k: Cutoff rank

    Returns:
        NDCG@k score (0 to 1, higher is better)
    """
    if not relevance_scores:
        return 0.0

    ideal = sorted(relevance_scores, reverse=True)[:k]
    actual = relevance_scores[:k]

    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(actual))
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))

    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_ctr(
    impressions: int,
    clicks: int,
) -> float:
    """Compute Click-Through Rate."""
    return clicks / max(impressions, 1)


def compute_ctr_lift(
    treatment_ctr: float,
    control_ctr: float,
) -> float:
    """Compute CTR lift vs control."""
    if control_ctr == 0:
        return 0.0
    return (treatment_ctr - control_ctr) / control_ctr


def compute_return_rate(
    purchased: int,
    returned: int,
) -> float:
    """Compute return rate."""
    return returned / max(purchased, 1)


def compute_discovery_score(
    recommended_categories: List[str],
    user_history_categories: List[str],
) -> float:
    """Compute discovery score: fraction of recommended categories
    the user has NOT interacted with before.

    Higher is better for anti-echo-chamber injection.
    """
    if not recommended_categories:
        return 0.0
    history_set = set(user_history_categories)
    novel = sum(1 for c in recommended_categories if c not in history_set)
    return novel / len(recommended_categories)


def compute_precision_recall_at_k(
    predicted: List[str],
    actual: List[str],
    k: int = 50,
) -> Tuple[float, float]:
    """Compute precision@k and recall@k."""
    predicted_set = set(predicted[:k])
    actual_set = set(actual)
    hits = len(predicted_set & actual_set)
    precision = hits / min(k, len(predicted))
    recall = hits / max(len(actual_set), 1)
    return precision, recall


def compute_fraud_ring_recall(
    detected_rings: List[Set[str]],
    actual_rings: List[Set[str]],
) -> float:
    """Compute recall of fraud ring detection.

    Measures what fraction of actual fraudulent sellers were flagged.
    """
    detected_flat = set().union(*detected_rings) if detected_rings else set()
    actual_flat = set().union(*actual_rings) if actual_rings else set()

    if not actual_flat:
        return 1.0

    hits = len(detected_flat & actual_flat)
    return hits / len(actual_flat)
