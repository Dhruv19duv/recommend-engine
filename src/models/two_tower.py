"""Two-Tower Neural Network for user-item recommendation.

Separate encoders for user context and item features, trained on six months
of interaction logs with in-batch negative sampling.

Architecture:
- User Tower: [features -> dense(512) -> ReLU -> Dropout -> dense(256) -> ReLU -> dense(128)]
- Item Tower: [features -> dense(512) -> ReLU -> Dropout -> dense(256) -> ReLU -> dense(128)]
- Dot product similarity with temperature scaling
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


class UserTower(nn.Module):
    """Encodes user context into a dense embedding vector.

    Input features:
    - User ID embedding
    - Context features (time of day, day of week, device, location)
    - Historical behavior embeddings (recent categories viewed, price range)
    - Session features (pages viewed so far, session duration)
    """

    def __init__(
        self,
        num_users: int = 400_000_000,
        user_embedding_dim: int = 64,
        context_dim: int = 64,
        hidden_dims: List[int] = None,
        output_dim: int = 128,
        dropout: float = 0.2,
        num_hash_buckets: int = 10_000,  # Hashing trick for user IDs
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.num_users = num_users
        self.user_embedding_dim = user_embedding_dim
        self.output_dim = output_dim

        # Use embedding bag (hashing trick) for large-vocab user IDs
        # Real users are hash-mapped to a smaller embedding table
        self.user_embedding = nn.EmbeddingBag(
            num_embeddings=num_hash_buckets,
            embedding_dim=user_embedding_dim,
            mode="mean",
            sparse=True,
        )

        # Context feature encoder
        input_dim = user_embedding_dim + context_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        context_features: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User ID indices (or hashed IDs)
            context_features: (batch_size, context_dim) tensor
            offsets: Offsets for EmbeddingBag (if variable-length sequences)

        Returns:
            (batch_size, output_dim) user embeddings
        """
        user_emb = self.user_embedding(user_ids, offsets)
        combined = torch.cat([user_emb, context_features], dim=-1)
        return self.encoder(combined)


class ItemTower(nn.Module):
    """Encodes item features into a dense embedding vector.

    Input features:
    - Item ID embedding (hash-mapped)
    - Category embedding
    - Seller embedding
    - Price bucket embedding
    - Image/description features (projected via pretrained model)
    """

    def __init__(
        self,
        num_items: int = 350_000_000,
        num_categories: int = 50_000,
        num_sellers: int = 5_000_000,
        item_embedding_dim: int = 64,
        category_dim: int = 32,
        seller_dim: int = 32,
        price_dim: int = 8,
        image_feature_dim: int = 128,
        hidden_dims: List[int] = None,
        output_dim: int = 128,
        dropout: float = 0.2,
        num_hash_buckets: int = 10_000_000,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Hash-mapped item embeddings
        self.item_embedding = nn.EmbeddingBag(
            num_embeddings=num_hash_buckets,
            embedding_dim=item_embedding_dim,
            mode="mean",
            sparse=True,
        )

        # Categorical embeddings
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        self.seller_embedding = nn.EmbeddingBag(num_sellers, seller_dim, mode="mean")
        self.price_embedding = nn.Linear(1, price_dim)  # Continuous price -> embedding

        input_dim = item_embedding_dim + category_dim + seller_dim + price_dim + image_feature_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        seller_ids: torch.Tensor,
        prices: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            item_ids: (batch_size,) item ID indices
            category_ids: (batch_size,) category indices
            seller_ids: (batch_size,) seller indices
            prices: (batch_size, 1) normalized prices
            image_features: (batch_size, image_feature_dim) pretrained features

        Returns:
            (batch_size, output_dim) item embeddings
        """
        item_emb = self.item_embedding(item_ids)
        cat_emb = self.category_embedding(category_ids)
        seller_emb = self.seller_embedding(seller_ids)
        price_emb = self.price_embedding(prices)

        combined = torch.cat([item_emb, cat_emb, seller_emb, price_emb, image_features], dim=-1)
        return self.encoder(combined)


class TwoTowerModel(nn.Module):
    """Two-tower neural recommendation model.

    Trained on six months of interaction logs with in-batch negative sampling.
    Optimizes for recall@K with sampled softmax loss.

    Usage:
        model = TwoTowerModel()
        user_emb = model.user_tower(user_ids, context)
        item_emb = model.item_tower(item_ids, categories, sellers, prices, images)
        scores = model.score(user_emb, item_emb)  # Dot product
    """

    def __init__(
        self,
        user_hidden_dims: List[int] = None,
        item_hidden_dims: List[int] = None,
        output_dim: int = 128,
        temperature: float = 0.05,
        dropout: float = 0.2,
        user_hash_buckets: int = 10_000,
        item_hash_buckets: int = 10_000_000,
    ) -> None:
        super().__init__()
        if user_hidden_dims is None:
            user_hidden_dims = [512, 256, 128]
        if item_hidden_dims is None:
            item_hidden_dims = [512, 256, 128]

        self.output_dim = output_dim
        self.temperature = temperature

        self.user_tower = UserTower(
            hidden_dims=user_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            num_hash_buckets=user_hash_buckets,
        )

        self.item_tower = ItemTower(
            hidden_dims=item_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            num_hash_buckets=item_hash_buckets,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        context_features: torch.Tensor,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        seller_ids: torch.Tensor,
        prices: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both towers.

        Returns:
            (user_embeddings, item_embeddings)
        """
        user_emb = self.user_tower(user_ids, context_features)
        item_emb = self.item_tower(
            item_ids=item_ids,
            category_ids=category_ids,
            seller_ids=seller_ids,
            prices=prices,
            image_features=image_features,
        )
        return user_emb, item_emb

    def score(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dot product similarity with temperature scaling."""
        user_emb = F.normalize(user_embeddings, p=2, dim=-1)
        item_emb = F.normalize(item_embeddings, p=2, dim=-1)
        return torch.mm(user_emb, item_emb.t()) / self.temperature

    def compute_inbatch_loss(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Sampled softmax loss with in-batch negatives.

        In-batch negative sampling: each item in the batch serves as a
        negative sample for all other users in the batch. The positive
        pairs are on the diagonal.

        Args:
            user_embeddings: (batch_size, output_dim)
            item_embeddings: (batch_size, output_dim) — positive items

        Returns:
            Loss scalar
        """
        logits = self.score(user_embeddings, item_embeddings)
        batch_size = user_embeddings.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


@dataclass
class TwoTowerTrainerConfig:
    """Training configuration for the two-tower model."""

    batch_size: int = 4096
    learning_rate: float = 1e-3
    num_epochs: int = 10
    num_negative_samples: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "/data/checkpoints/two_tower"
    log_interval: int = 100
    eval_interval: int = 1000


class TwoTowerTrainer:
    """Trainer for the two-tower model with distributed data parallel support."""

    def __init__(
        self,
        model: TwoTowerModel,
        config: TwoTowerTrainerConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.warmup_steps
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """Single training step with in-batch negative sampling."""
        self.model.train()
        self.optimizer.zero_grad()

        user_ids = batch["user_ids"].to(self.device)
        context = batch["context_features"].to(self.device)
        item_ids = batch["item_ids"].to(self.device)
        categories = batch["category_ids"].to(self.device)
        sellers = batch["seller_ids"].to(self.device)
        prices = batch["prices"].to(self.device)
        images = batch["image_features"].to(self.device)

        user_emb, item_emb = self.model(
            user_ids=user_ids,
            context_features=context,
            item_ids=item_ids,
            category_ids=categories,
            seller_ids=sellers,
            prices=prices,
            image_features=images,
        )

        # Labels: diagonal is positive, rest are in-batch negatives
        batch_size = user_ids.size(0)
        labels = torch.arange(batch_size, device=self.device)

        loss = self.model.compute_inbatch_loss(user_emb, item_emb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
