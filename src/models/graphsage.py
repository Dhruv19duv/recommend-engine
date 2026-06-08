"""GraphSAGE model for propagating signals through the heterogeneous product graph.

Uses DGL for efficient message passing on large graphs.
Propagates "users who bought X also browsed Y" signals into node embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SAGELayer(nn.Module):
    """A single GraphSAGE layer with mean aggregation.

    Supports heterogeneous edge types via separate weight matrices
    per edge type.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.aggregator = aggregator

        # Self and neighbor projections
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        if aggregator == "mean":
            self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        elif aggregator == "lstm":
            self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)
            self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self,
        self_feats: torch.Tensor,
        neighbor_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            self_feats: (n_nodes, in_dim) — current node features
            neighbor_feats: (n_nodes, in_dim) — aggregated neighbor features

        Returns:
            (n_nodes, out_dim) — updated node features
        """
        self_feats = self.dropout(self_feats)
        neighbor_feats = self.dropout(neighbor_feats)

        # Aggregate
        if self.aggregator == "mean":
            neigh_agg = neighbor_feats
        elif self.aggregator == "lstm":
            # Apply LSTM over neighbor sequence
            neigh_agg, _ = self.lstm(neighbor_feats.unsqueeze(0))
            neigh_agg = neigh_agg.squeeze(0)
        else:
            neigh_agg = neighbor_feats

        # Combine self + neighbor
        h = self.W_self(self_feats) + self.W_neigh(neigh_agg) + self.bias
        return F.relu(h)


class HeterogeneousGraphSAGE(nn.Module):
    """GraphSAGE on a heterogeneous graph with typed edges.

    Each edge type gets its own SAGE layer, and outputs are
    combined via mean pooling across edge types.
    """

    def __init__(
        self,
        num_edge_types: int,
        feature_dim: int = 256,
        hidden_dims: List[int] = None,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.num_edge_types = num_edge_types
        self.num_layers = num_layers

        # Per-edge-type, per-layer SAGE modules
        # In production, use nn.ModuleDict with edge-type keys
        self.layers = nn.ModuleList()
        dims = [feature_dim] + hidden_dims + [output_dim]
        for layer_idx in range(num_layers):
            in_dim = dims[layer_idx]
            out_dim = dims[layer_idx + 1]
            edge_layers = nn.ModuleList([
                SAGELayer(in_dim, out_dim, dropout, aggregator)
                for _ in range(num_edge_types)
            ])
            self.layers.append(edge_layers)

        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        neighbor_features_by_type: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through all SAGE layers.

        Args:
            node_features: (n_nodes, feature_dim) initial node features
            neighbor_features_by_type: List of (n_nodes, feature_dim) tensors,
                one per edge type, representing aggregated neighbor features

        Returns:
            (n_nodes, output_dim) updated node embeddings
        """
        h = self.feature_proj(node_features)

        for layer_idx, edge_layers in enumerate(self.layers):
            layer_outputs = []
            for edge_type_idx, sage_layer in enumerate(edge_layers):
                if edge_type_idx < len(neighbor_features_by_type):
                    neigh_feats = neighbor_features_by_type[edge_type_idx]
                    layer_out = sage_layer(h, neigh_feats)
                    layer_outputs.append(layer_out)

            # Combine across edge types via mean pooling
            if layer_outputs:
                h = torch.stack(layer_outputs).mean(dim=0)
            else:
                h = sage_layer(h, h)  # Fallback: self-loop

        return self.output_proj(h)


class GraphSAGETrainer:
    """Trainer for the GraphSAGE model with neighbor sampling."""

    def __init__(
        self,
        model: HeterogeneousGraphSAGE,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(
        self,
        node_features: torch.Tensor,
        neighbor_features_by_type: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> float:
        """Link prediction training step.

        Positive examples are connected node pairs.
        Negative examples are randomly sampled non-connected pairs.
        """
        self.model.train()
        self.optimizer.zero_grad()

        node_features = node_features.to(self.device)
        neighbor_features = [nf.to(self.device) for nf in neighbor_features_by_type]
        labels = labels.to(self.device)

        embeddings = self.model(node_features, neighbor_features)

        # Link prediction loss: dot product similarity for positive pairs
        pos_emb = embeddings[labels == 1]
        neg_emb = embeddings[labels == 0]

        pos_scores = torch.mm(pos_emb, pos_emb.t()).diag()
        neg_scores = torch.mm(neg_emb, neg_emb.t()).diag()

        # Margin ranking loss
        loss = F.margin_ranking_loss(
            pos_scores,
            neg_scores,
            torch.ones_like(pos_scores),
            margin=0.5,
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()
