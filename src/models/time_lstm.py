"""Time-Aware LSTM for modeling intra-day preference volatility.

Learns context-dependent preference patterns per time window.
Assigns each user a "mood-state" embedding per time-of-day / day-of-week window,
capturing the reality that user preferences shift throughout the day.

Key insight: 2am after payday is a different user than Tuesday lunch break.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TimeAwareLSTM(nn.Module):
    """LSTM that learns intra-day preference shift patterns.

    Architecture:
    1. Embed interaction sequences with time features (hour, day_of_week, is_weekend, is_payday)
    2. LSTM over the sequence produces hidden states
    3. Time-weighted attention pools relevant hidden states
    4. Output: mood-state embedding for the current time window

    The mood-state embedding is used to condition the user tower output,
    effectively changing the recommendation policy per time context.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 32,  # mood-state embedding dimension
        dropout: float = 0.2,
        num_time_buckets: int = 24,  # 24 hourly buckets
        num_day_buckets: int = 7,  # 7 days of week
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_time_buckets = num_time_buckets
        self.num_day_buckets = num_day_buckets

        # Time feature embeddings
        self.time_embedding = nn.Embedding(num_time_buckets, 16)
        self.day_embedding = nn.Embedding(num_day_buckets, 8)
        self.is_weekend_embedding = nn.Embedding(2, 4)
        self.is_payday_embedding = nn.Embedding(2, 4)

        # Emotional context features
        self.emotion_projection = nn.Linear(16 + 8 + 4 + 4, input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Time-weighted attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection to mood-state embedding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def embed_time_context(
        self,
        time_buckets: torch.Tensor,
        day_buckets: torch.Tensor,
        is_weekend: torch.Tensor,
        is_payday: torch.Tensor,
    ) -> torch.Tensor:
        """Embed time context features into a continuous vector."""
        t_emb = self.time_embedding(time_buckets)
        d_emb = self.day_embedding(day_buckets)
        w_emb = self.is_weekend_embedding(is_weekend)
        p_emb = self.is_payday_embedding(is_payday)
        return self.emotion_projection(torch.cat([t_emb, d_emb, w_emb, p_emb], dim=-1))

    def forward(
        self,
        interaction_sequences: torch.Tensor,
        time_buckets: torch.Tensor,
        day_buckets: torch.Tensor,
        is_weekend: torch.Tensor,
        is_payday: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            interaction_sequences: (batch_size, seq_len, input_dim)
            time_buckets: (batch_size, seq_len) hour-of-day indices
            day_buckets: (batch_size, seq_len) day-of-week indices
            is_weekend: (batch_size, seq_len) binary indicator
            is_payday: (batch_size, seq_len) binary indicator
            mask: (batch_size, seq_len) padding mask

        Returns:
            (mood_state_embedding, all_hidden_states)
            mood_state_embedding: (batch_size, output_dim)
            all_hidden_states: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len = interaction_sequences.shape[:2]

        # Embed time context
        time_features = self.embed_time_context(
            time_buckets.view(-1), day_buckets.view(-1),
            is_weekend.view(-1), is_payday.view(-1),
        ).view(batch_size, seq_len, -1)

        # Combine interaction features with time features
        combined = interaction_sequences + time_features
        combined = self.dropout(combined)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(combined)
        # lstm_out: (batch_size, seq_len, hidden_dim)

        # Time-weighted self-attention
        if mask is not None:
            attn_mask = mask.float().masked_fill(mask == 0, float("-inf"))
        else:
            attn_mask = None

        attended, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=(mask.logical_not() if mask is not None else None),
        )

        # Pool: use attention-weighted mean
        if mask is not None:
            attended = attended * mask.unsqueeze(-1).float()
            pooled = attended.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            pooled = attended.mean(dim=1)

        # Project to mood-state embedding
        mood_state = self.output_proj(pooled)

        return mood_state, lstm_out


class PreferenceVolatilityModule(nn.Module):
    """End-to-end module connecting preference volatility to recommendation.

    Wraps the TimeAwareLSTM and injects the mood-state embedding into the
    ranking policy via a context-dependent bias vector.
    """

    def __init__(
        self,
        lstm: TimeAwareLSTM,
        item_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.lstm = lstm
        self.mood_to_bias = nn.Linear(lstm.output_dim, item_embedding_dim)
        self.mood_classifier = nn.Linear(lstm.output_dim, 5)  # 5 emotional contexts

    def get_mood_state(
        self,
        interaction_sequences: torch.Tensor,
        time_buckets: torch.Tensor,
        day_buckets: torch.Tensor,
        is_weekend: torch.Tensor,
        is_payday: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mood-state embedding and emotional context logits.

        Returns:
            (mood_state_embedding, emotional_context_logits)
        """
        mood_state, _ = self.lstm(
            interaction_sequences, time_buckets, day_buckets,
            is_weekend, is_payday, mask,
        )
        context_logits = self.mood_classifier(mood_state)
        return mood_state, context_logits

    def compute_score_bias(
        self,
        mood_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a bias vector to add to item scores based on mood state."""
        return self.mood_to_bias(mood_state)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        mood_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adjusted recommendation scores.

        score = dot(user_emb, item_emb) + dot(mood_bias, item_emb)
        """
        mood_bias = self.compute_score_bias(mood_state)
        base_scores = torch.mm(user_embeddings, item_embeddings.t())
        mood_adjustment = torch.mm(mood_bias, item_embeddings.t())
        return base_scores + mood_adjustment
