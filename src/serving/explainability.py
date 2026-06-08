"""SHAP explainability layer for every recommendation.

Surfaces the top three driving features as a human-readable "recommended because" string.
Provides transparency into why each item was recommended to a specific user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """A human-readable explanation for a recommendation."""

    item_id: str
    user_id: str
    reason: str  # Human-readable string, e.g., "Because you viewed similar kitchen products"
    top_features: List[Tuple[str, float]]  # [(feature_name, shap_value), ...]
    shap_values: np.ndarray  # Raw SHAP values
    base_value: float  # Expected model output baseline
    confidence: float  # How confident we are in this explanation


class SHAPExplainer:
    """SHAP-based explainability for recommendation decisions.

    Explains why each item was recommended to a particular user by
    attributing the recommendation score to input features.

    Feature names used in explanations:
    - "Your interest in {category}"
    - "Your recent view of similar items"
    - "Popular in your area"
    - "Similar to items you've purchased"
    - "Your price range preference"
    - "Trending this week"
    - "Other {category} buyers also liked"
    - "Based on your {time_of_day} shopping patterns"
    - "Discovery pick (outside your usual categories)"
    """

    def __init__(self, n_features: int = 64) -> None:
        self.n_features = n_features
        self._feature_names: List[str] = []
        self._background_data: Optional[np.ndarray] = None

    def set_feature_names(self, names: List[str]) -> None:
        """Set human-readable feature names."""
        assert len(names) == self.n_features
        self._feature_names = names

    def set_background(self, data: np.ndarray) -> None:
        """Set background dataset for SHAP expectation computation."""
        self._background_data = data

    def explain(
        self,
        model_output: float,
        input_features: np.ndarray,
        shap_values: np.ndarray,
        user_id: str,
        item_id: str,
    ) -> Explanation:
        """Generate an explanation for a single recommendation.

        Args:
            model_output: The recommendation score
            input_features: (n_features,) input feature vector
            shap_values: (n_features,) SHAP values for this prediction
            user_id: The user being recommended to
            item_id: The item being recommended

        Returns:
            Explanation with human-readable reason string
        """
        # Get top features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-3:][::-1]  # Top 3

        top_features = [
            (
                self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}",
                float(shap_values[i]),
            )
            for i in top_indices
        ]

        # Generate human-readable reason
        reason = self._generate_reason(top_features, input_features, top_indices)

        return Explanation(
            item_id=item_id,
            user_id=user_id,
            reason=reason,
            top_features=top_features,
            shap_values=shap_values,
            base_value=float(shap_values.mean() if len(shap_values) > 0 else 0.0),
            confidence=min(1.0, float(np.max(abs_shap))),
        )

    def _generate_reason(
        self,
        top_features: List[Tuple[str, float]],
        input_features: np.ndarray,
        top_indices: np.ndarray,
    ) -> str:
        """Generate a human-readable explanation string from top features."""
        reasons: List[str] = []

        for name, value in top_features:
            if value > 0:
                # Positive contribution
                if "view" in name.lower() or "seen" in name.lower():
                    reasons.append(f"Based on items you recently viewed")
                elif "purchase" in name.lower() or "bought" in name.lower():
                    reasons.append(f"Matches items you've previously purchased")
                elif "category" in name.lower() or "interest" in name.lower():
                    reasons.append(f"Similar to categories you engage with")
                elif "popular" in name.lower() or "trending" in name.lower():
                    reasons.append(f"Popular with shoppers like you")
                elif "price" in name.lower():
                    reasons.append(f"Fits your usual price range")
                elif "discovery" in name.lower() or "outside" in name.lower():
                    reasons.append(f"You might discover something new")
                elif "time" in name.lower() or "shopping" in name.lower():
                    reasons.append(f"Relevant for your current shopping time")
                else:
                    reasons.append(f"Recommended based on your preferences")
            else:
                # Negative contribution (we still recommend despite this)
                pass

        if not reasons:
            return f"Recommended for you"

        if len(reasons) == 1:
            return reasons[0]
        elif len(reasons) == 2:
            return f"{reasons[0]} and {reasons[1]}"
        else:
            return f"{reasons[0]}, {reasons[1]}, and {reasons[2]}"
