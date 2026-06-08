"""Innovation 1: Preference Volatility Engine.

The core insight: a user shopping at 2am after payday is a completely different
buyer than the same user on a Tuesday lunch break. Context IS the user.

This module trains the LSTM to recognize distinct emotional/situational contexts:
- After salary credit (increased spending, higher price tolerance)
- After a breakup (comfort purchases, category shift)
- Late-night scroll (impulsive, lower price sensitivity)
- Weekend browsing (leisurely, higher discovery affinity)
- Workday lunch break (efficient, task-oriented)

Each context gets its own embedding and ranking policy, learned end-to-end
from interaction sequences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmotionalContext(Enum):
    """Recognized emotional/situational shopping contexts."""

    POST_PAYDAY = "post_payday"
    LATE_NIGHT = "late_night"
    WEEKEND_LEISURE = "weekend_leisure"
    WORKDAY_LUNCH = "workday_lunch"
    EMOTIONAL_COMFORT = "emotional_comfort"
    HOLIDAY_SHOPPING = "holiday_shopping"
    DEFAULT = "default"


@dataclass
class VolatilityProfile:
    """Preference volatility profile for a user-time context."""

    user_id: str
    time_window: str  # e.g., "weekday_morning", "weekend_late_night", "payday_evening"
    emotional_context: EmotionalContext
    mood_embedding: np.ndarray  # (mood_embedding_dim,)
    category_affinity_shift: Dict[str, float]  # category_id -> affinity delta
    price_tolerance_delta: float  # +/- price tolerance shift
    impulse_score: float  # 0-1, higher = more impulsive
    discovery_affinity: float  # 0-1, higher = more open to discovery
    confidence: float  # 0-1, how confident we are in this profile


class PreferenceVolatilityEngine:
    """The Preference Volatility Engine.

    Ties together the TimeAwareLSTM model with online inference to produce
    context-dependent recommendation policies.

    Key capability: detect that a user's 2am self is distinct from their
    12pm self, and serve dramatically different recommendations accordingly.
    """

    def __init__(
        self,
        lstm_model: Optional[Any] = None,  # TimeAwareLSTM
        mood_embedding_dim: int = 32,
        num_time_buckets: int = 24,
        num_day_buckets: int = 7,
    ) -> None:
        self.lstm_model = lstm_model
        self.mood_embedding_dim = mood_embedding_dim
        self._profiles: Dict[str, VolatilityProfile] = {}

        # Time-based context categorization
        self._time_context_map = self._build_time_context_map(num_time_buckets, num_day_buckets)

    def _build_time_context_map(
        self, num_time_buckets: int, num_day_buckets: int
    ) -> Dict[Tuple[int, int], EmotionalContext]:
        """Map (hour_bucket, day_bucket) to emotional context."""
        mapping: Dict[Tuple[int, int], EmotionalContext] = {}

        for hour in range(num_time_buckets):
            for day in range(num_day_buckets):
                is_weekend = day >= 5
                is_late_night = hour < 6 or hour >= 23
                is_workday = not is_weekend and 7 <= hour <= 17
                is_lunch = not is_weekend and 11 <= hour <= 13

                if is_late_night:
                    ctx = EmotionalContext.LATE_NIGHT
                elif is_weekend and hour >= 10:
                    ctx = EmotionalContext.WEEKEND_LEISURE
                elif is_lunch:
                    ctx = EmotionalContext.WORKDAY_LUNCH
                elif is_workday:
                    ctx = EmotionalContext.WORKDAY_LUNCH
                else:
                    ctx = EmotionalContext.DEFAULT

                mapping[(hour, day)] = ctx

        return mapping

    def get_emotional_context(
        self,
        hour_of_day: int,
        day_of_week: int,
        is_payday: bool = False,
        recent_emotional_signals: Optional[Dict[str, float]] = None,
    ) -> EmotionalContext:
        """Determine the user's emotional context from time and signals."""
        base_context = self._time_context_map.get(
            (hour_of_day, day_of_week), EmotionalContext.DEFAULT
        )

        if is_payday:
            return EmotionalContext.POST_PAYDAY

        if recent_emotional_signals:
            # Check for emotional comfort signals (e.g., search terms, categories)
            comfort_categories = {"comfort", "self_care", "wellness", "entertainment"}
            overlap = set(recent_emotional_signals.keys()) & comfort_categories
            if overlap:
                comfort_score = sum(recent_emotional_signals.get(c, 0.0) for c in overlap)
                if comfort_score > 0.5:
                    return EmotionalContext.EMOTIONAL_COMFORT

        return base_context

    def compute_mood_embedding(
        self,
        user_id: str,
        interaction_sequence: np.ndarray,
        time_bucket: int,
        day_bucket: int,
        is_weekend: bool,
        is_payday: bool,
    ) -> np.ndarray:
        """Compute the mood-state embedding for a user-time context.

        Args:
            user_id: The user's ID
            interaction_sequence: (seq_len, feature_dim) sequence of recent interactions
            time_bucket: Hour of day (0-23)
            day_bucket: Day of week (0-6)
            is_weekend: Weekend flag
            is_payday: Payday flag

        Returns:
            (mood_embedding_dim,) mood-state embedding
        """
        if self.lstm_model is None:
            return np.zeros(self.mood_embedding_dim)

        self.lstm_model.eval()
        with torch.no_grad():
            # Convert to tensors
            seq_tensor = torch.from_numpy(interaction_sequence).float().unsqueeze(0)
            time_tensor = torch.tensor([[time_bucket]])
            day_tensor = torch.tensor([[day_bucket]])
            weekend_tensor = torch.tensor([[1 if is_weekend else 0]])
            payday_tensor = torch.tensor([[1 if is_payday else 0]])

            mood_state, context_logits = self.lstm_model.get_mood_state(
                interaction_sequences=seq_tensor,
                time_buckets=time_tensor,
                day_buckets=day_tensor,
                is_weekend=weekend_tensor,
                is_payday=payday_tensor,
            )

        return mood_state.squeeze(0).numpy()

    def get_volatility_profile(
        self,
        user_id: str,
        hour_of_day: int,
        day_of_week: int,
        is_payday: bool = False,
        interaction_sequence: Optional[np.ndarray] = None,
    ) -> VolatilityProfile:
        """Get or compute the volatility profile for a user-time context."""
        profile_key = f"{user_id}:{hour_of_day}:{day_of_week}:{is_payday}"

        if profile_key in self._profiles:
            return self._profiles[profile_key]

        context = self.get_emotional_context(hour_of_day, day_of_week, is_payday)

        mood_emb = (
            self.compute_mood_embedding(
                user_id, interaction_sequence,
                hour_of_day, day_of_week,
                day_of_week >= 5, is_payday,
            )
            if interaction_sequence is not None
            else np.zeros(self.mood_embedding_dim)
        )

        # Derive behavioral shifts from context
        category_shifts = self._compute_category_affinity_shifts(context)
        price_tolerance = self._compute_price_tolerance_shift(context)
        impulse = self._compute_impulse_score(context)
        discovery = self._compute_discovery_affinity(context)

        profile = VolatilityProfile(
            user_id=user_id,
            time_window=f"{'weekday' if day_of_week < 5 else 'weekend'}_{'morning' if hour_of_day < 12 else 'afternoon' if hour_of_day < 18 else 'evening'}",
            emotional_context=context,
            mood_embedding=mood_emb,
            category_affinity_shift=category_shifts,
            price_tolerance_delta=price_tolerance,
            impulse_score=impulse,
            discovery_affinity=discovery,
            confidence=0.7 if interaction_sequence is not None else 0.3,
        )

        self._profiles[profile_key] = profile
        return profile

    def _compute_category_affinity_shifts(
        self, context: EmotionalContext
    ) -> Dict[str, float]:
        """Compute category affinity shifts for a given context.

        Returns: {category_id: affinity_delta} where positive means
        the user is more likely to engage with this category in this context.
        """
        shifts: Dict[str, float] = {}
        if context == EmotionalContext.POST_PAYDAY:
            shifts.update({
                "electronics": 0.3,
                "luxury": 0.4,
                "home": 0.2,
                "apparel": 0.15,
            })
        elif context == EmotionalContext.LATE_NIGHT:
            shifts.update({
                "entertainment": 0.4,
                "food": 0.3,
                "books": 0.2,
                "wellness": 0.1,
            })
        elif context == EmotionalContext.WEEKEND_LEISURE:
            shifts.update({
                "outdoor": 0.3,
                "sports": 0.2,
                "hobbies": 0.3,
                "travel": 0.2,
            })
        elif context == EmotionalContext.EMOTIONAL_COMFORT:
            shifts.update({
                "comfort_food": 0.4,
                "entertainment": 0.3,
                "self_care": 0.3,
            })
        elif context == EmotionalContext.WORKDAY_LUNCH:
            shifts.update({
                "food": 0.3,
                "office": 0.2,
                "productivity": 0.2,
            })
        return shifts

    def _compute_price_tolerance_shift(self, context: EmotionalContext) -> float:
        """Compute price tolerance shift. Positive = more willing to spend."""
        shifts = {
            EmotionalContext.POST_PAYDAY: 0.4,
            EmotionalContext.LATE_NIGHT: 0.2,
            EmotionalContext.EMOTIONAL_COMFORT: 0.3,
            EmotionalContext.WEEKEND_LEISURE: 0.15,
            EmotionalContext.WORKDAY_LUNCH: -0.1,
            EmotionalContext.DEFAULT: 0.0,
        }
        return shifts.get(context, 0.0)

    def _compute_impulse_score(self, context: EmotionalContext) -> float:
        """Compute impulse buying score (0-1)."""
        scores = {
            EmotionalContext.LATE_NIGHT: 0.7,
            EmotionalContext.POST_PAYDAY: 0.6,
            EmotionalContext.EMOTIONAL_COMFORT: 0.5,
            EmotionalContext.WEEKEND_LEISURE: 0.3,
            EmotionalContext.WORKDAY_LUNCH: 0.2,
            EmotionalContext.DEFAULT: 0.1,
        }
        return scores.get(context, 0.0)

    def _compute_discovery_affinity(self, context: EmotionalContext) -> float:
        """Compute openness to discovery (0-1)."""
        scores = {
            EmotionalContext.WEEKEND_LEISURE: 0.7,
            EmotionalContext.LATE_NIGHT: 0.5,
            EmotionalContext.POST_PAYDAY: 0.4,
            EmotionalContext.EMOTIONAL_COMFORT: 0.3,
            EmotionalContext.WORKDAY_LUNCH: 0.1,
            EmotionalContext.DEFAULT: 0.2,
        }
        return scores.get(context, 0.0)
