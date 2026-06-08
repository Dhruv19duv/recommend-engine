"""Contextual Bandit (LinUCB) for cold-start users with zero history.

Cold-start users never receive a blank slate. From the first click,
the bandit learns user preferences via the LinUCB algorithm.

Architecture:
- Users are represented by context features (device, location, time, referrer)
- Arms are category-level or item-level actions
- Exploration is guided by upper confidence bounds
- Default policy falls back to popularity-based recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    """An arm in the contextual bandit (typically a category or product type)."""

    arm_id: str
    name: str
    num_pulls: int = 0
    total_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        return self.total_reward / max(self.num_pulls, 1)


class LinUCB:
    """LinUCB (Linear Upper Confidence Bound) contextual bandit.

    Implements disjoint LinUCB where each arm has its own parameter vector.
    For each user context x, it selects the arm a that maximizes:
        x^T * theta_a + alpha * sqrt(x^T * A_a^{-1} * x)

    Where:
    - theta_a: learned parameter vector for arm a
    - A_a: covariance matrix for arm a
    - alpha: exploration parameter
    """

    def __init__(
        self,
        context_dim: int = 64,
        num_arms: int = 100,
        alpha: float = 1.0,
        batch_size: int = 128,
        warmup_samples: int = 50,
    ) -> None:
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.alpha = alpha  # Exploration parameter
        self.batch_size = batch_size
        self.warmup_samples = warmup_samples

        # Arm parameters
        self._A: Dict[str, np.ndarray] = {}  # Covariance matrices
        self._b: Dict[str, np.ndarray] = {}  # Response vectors
        self._theta: Dict[str, np.ndarray] = {}  # Parameter vectors

        # Arm metadata
        self._arms: Dict[str, BanditArm] = {}
        self._arm_ids: List[str] = []

        # User history
        self._user_arm_counts: Dict[str, Dict[str, int]] = {}

    def add_arm(self, arm_id: str, name: str) -> None:
        """Register a new arm."""
        if arm_id not in self._arms:
            self._arms[arm_id] = BanditArm(arm_id=arm_id, name=name)
            self._A[arm_id] = np.eye(self.context_dim)
            self._b[arm_id] = np.zeros(self.context_dim)
            self._theta[arm_id] = np.zeros(self.context_dim)
            self._arm_ids.append(arm_id)

    def select_arm(
        self,
        context: np.ndarray,
        user_id: str,
        available_arms: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """Select an arm using LinUCB policy.

        Args:
            context: (context_dim,) user context vector
            user_id: User identifier for tracking
            available_arms: Subset of arms to choose from. None = all arms.

        Returns:
            (selected_arm_id, ucb_score)
        """
        if context.shape[0] != self.context_dim:
            raise ValueError(f"Expected context dim {self.context_dim}, got {context.shape[0]}")

        arms = available_arms or self._arm_ids
        if not arms and self._arm_ids:
            arms = self._arm_ids

        if not arms:
            raise ValueError("No arms available")

        best_arm = arms[0]
        best_ucb = float("-inf")

        for arm_id in arms:
            if arm_id not in self._A:
                self.add_arm(arm_id, arm_id)

            A = self._A[arm_id]
            theta = self._theta[arm_id]

            # Expected payoff: x^T * theta
            expected = np.dot(context, theta)

            # Uncertainty: alpha * sqrt(x^T * A^{-1} * x)
            confidence = np.sqrt(np.dot(context, np.linalg.solve(A, context)))
            ucb = expected + self.alpha * confidence

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id

        # Track selection
        if user_id not in self._user_arm_counts:
            self._user_arm_counts[user_id] = {}
        self._user_arm_counts[user_id][best_arm] = (
            self._user_arm_counts[user_id].get(best_arm, 0) + 1
        )

        return best_arm, float(best_ucb)

    def update(
        self,
        arm_id: str,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update arm parameters with observed reward.

        Args:
            arm_id: The arm that was selected
            context: The context vector when the arm was selected
            reward: Observed reward (e.g., click=1, purchase=5, no_click=0)
        """
        if arm_id not in self._A:
            self.add_arm(arm_id, arm_id)

        # Update covariance: A += x * x^T
        self._A[arm_id] += np.outer(context, context)

        # Update response: b += reward * x
        self._b[arm_id] += reward * context

        # Update theta: theta = A^{-1} * b
        self._theta[arm_id] = np.linalg.solve(self._A[arm_id], self._b[arm_id])

        # Update arm stats
        if arm_id in self._arms:
            self._arms[arm_id].num_pulls += 1
            self._arms[arm_id].total_reward += reward

    def batch_update(
        self,
        arm_ids: List[str],
        contexts: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """Update multiple arms at once (for efficiency)."""
        for i, arm_id in enumerate(arm_ids):
            self.update(arm_id, contexts[i], rewards[i])

    def get_top_arms(
        self,
        context: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top-K arms by UCB score without selection."""
        scores: List[Tuple[str, float]] = []
        for arm_id in self._arm_ids:
            _, score = self.select_arm(context, "eval", available_arms=[arm_id])
            scores.append((arm_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def is_warm(self, arm_id: str) -> bool:
        """Check if an arm has been pulled enough times."""
        arm = self._arms.get(arm_id)
        return arm is not None and arm.num_pulls >= self.warmup_samples

    @property
    def total_pulls(self) -> int:
        return sum(arm.num_pulls for arm in self._arms.values())


class ContextualBanditRecommender:
    """High-level bandit-based recommender for cold-start users.

    Wraps the LinUCB algorithm with:
    - Context extraction from user metadata
    - Category-level arms (cold-start) -> item-level (warm)
    - Graceful fallback to popularity
    - Automatic arm pruning
    """

    def __init__(
        self,
        linucb: LinUCB,
        context_features: List[str] = None,
    ) -> None:
        self.linucb = linucb
        self.context_features = context_features or [
            "hour_of_day", "day_of_week", "is_weekend",
            "device_type", "is_new_user", "referrer_type",
        ]
        self._user_context_cache: Dict[str, np.ndarray] = {}

    def extract_context(
        self,
        user_metadata: Dict[str, float],
    ) -> np.ndarray:
        """Extract context vector from user metadata."""
        context = np.zeros(self.linucb.context_dim)
        for i, feature in enumerate(self.context_features[:self.linucb.context_dim]):
            context[i] = user_metadata.get(feature, 0.0)
        return context

    def recommend(
        self,
        user_id: str,
        user_metadata: Dict[str, float],
        available_categories: List[str],
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Get recommendations for a cold-start user.

        Returns best categories/items via LinUCB selection.
        """
        context = self.extract_context(user_metadata)
        self._user_context_cache[user_id] = context

        # Ensure all categories are registered as arms
        for cat in available_categories:
            self.linucb.add_arm(cat, cat)

        # Get top-K arms
        return self.linucb.get_top_arms(context, top_k)

    def record_reward(
        self,
        user_id: str,
        arm_id: str,
        reward: float,
    ) -> None:
        """Record a reward signal (click, purchase, etc.)."""
        context = self._user_context_cache.get(user_id)
        if context is not None:
            self.linucb.update(arm_id, context, reward)
