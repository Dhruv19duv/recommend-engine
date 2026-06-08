"""Innovation 2: Anti-Echo-Chamber Injection.

Problem: Recommendation bubbles cause long-term retention damage by showing
users only what they've already shown interest in.

Solution: Every 7th recommendation is deliberately outside the user's usual
category, tracked by a "discovery score" metric. This prevents filter bubbles
while maintaining relevance.

The injection uses a separate discovery model that scores items by their
"surprising relevance" — items the user would not normally click, but when
they do, they engage meaningfully.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryItem:
    """An item considered for anti-echo-chamber injection."""

    item_id: str
    category_id: str
    relevance_score: float  # Base relevance to user
    distance_from_history: float  # Category distance from user history (0-1)
    discovery_potential: float  # Expected engagement from discovery
    serendipity_score: float  # Combined: relevance * distance * novelty


class AntiEchoChamberInjector:
    """Injects serendipitous recommendations to break filter bubbles.

    Tracks a per-user 'discovery score' and ensures at least a target fraction
    of recommendations are outside the user's usual consumption patterns.

    The 7th-recommendation rule:
    - Positions 1-6: Standard high-relevance recommendations
    - Position 7: Deliberately outside user's usual category
    - The pattern resets every 7 items
    """

    def __init__(
        self,
        injection_rate: float = 1.0 / 7.0,  # Every 7th recommendation
        discovery_score_target: float = 0.15,
        category_distance_threshold: float = 0.6,
    ) -> None:
        self.injection_rate = injection_rate
        self.discovery_score_target = discovery_score_target
        self.category_distance_threshold = category_distance_threshold

        # Per-user state
        self._user_history: Dict[str, Set[str]] = {}  # user_id -> set of category_ids
        self._user_discovery_scores: Dict[str, float] = {}
        self._injection_positions: Dict[str, int] = {}  # user_id -> current position in cycle

    def register_interaction(
        self,
        user_id: str,
        item_id: str,
        category_id: str,
        clicked: bool = False,
    ) -> None:
        """Register a user interaction to update history."""
        if user_id not in self._user_history:
            self._user_history[user_id] = set()
        self._user_history[user_id].add(category_id)

    def compute_discovery_score(
        self,
        user_id: str,
        recommended_categories: List[str],
    ) -> float:
        """Compute what fraction of recommended categories are new to the user."""
        history = self._user_history.get(user_id, set())
        if not recommended_categories:
            return 0.0
        novel = sum(1 for c in recommended_categories if c not in history)
        return novel / len(recommended_categories)

    def get_discovery_candidates(
        self,
        user_id: str,
        all_candidates: List[DiscoveryItem],
        top_k: int = 50,
    ) -> List[DiscoveryItem]:
        """Select discovery candidates that are outside the user's usual categories.

        Scores items by serendipity = relevance * distance_from_history,
        ensuring they meet the minimum distance threshold.
        """
        history = self._user_history.get(user_id, set())

        discovery_candidates = [
            item for item in all_candidates
            if item.distance_from_history >= self.category_distance_threshold
        ]

        # Sort by serendipity score
        for item in discovery_candidates:
            item.serendipity_score = (
                item.relevance_score * item.distance_from_history
            )

        discovery_candidates.sort(key=lambda x: x.serendipity_score, reverse=True)
        return discovery_candidates[:top_k]

    def inject_discovery(
        self,
        user_id: str,
        ranked_items: List[Tuple[str, float]],
        discovery_items: List[DiscoveryItem],
        category_map: Dict[str, str],  # item_id -> category_id
    ) -> List[Tuple[str, float, bool]]:
        """Inject discovery recommendations into ranked list.

        Returns: [(item_id, score, is_discovery), ...]
        """
        if not ranked_items:
            return []

        # Get or initialize position in injection cycle
        pos = self._injection_positions.get(user_id, 0)
        discovery_pool = self.get_discovery_candidates(user_id, discovery_items)

        if not discovery_pool:
            # Fall back to marking existing results
            return [(item_id, score, False) for item_id, score in ranked_items]

        result: List[Tuple[str, float, bool]] = []
        discovery_idx = 0

        for i, (item_id, score) in enumerate(ranked_items):
            # Check if this position should be an injection
            should_inject = (
                (pos + 1) % 7 == 0
                and discovery_idx < len(discovery_pool)
            )

            if should_inject:
                # Inject discovery item
                disc_item = discovery_pool[discovery_idx]
                result.append((disc_item.item_id, score * 0.9, True))
                discovery_idx += 1
            else:
                result.append((item_id, score, False))

            pos = (pos + 1) % 7

        self._injection_positions[user_id] = pos

        # Update discovery score
        categories = [
            category_map.get(item_id, "unknown")
            for item_id, _, _ in result
        ]
        self._user_discovery_scores[user_id] = self.compute_discovery_score(
            user_id, categories
        )

        return result

    def needs_more_discovery(self, user_id: str) -> bool:
        """Check if a user's discovery score is below target."""
        current = self._user_discovery_scores.get(user_id, 0.0)
        return current < self.discovery_score_target

    def get_discovery_score(self, user_id: str) -> float:
        """Get the current discovery score for a user."""
        return self._user_discovery_scores.get(user_id, 0.0)
