"""Demand forecasting LSTM per SKU to avoid recommending out-of-stock items.

Forecasts demand over a 30-day horizon using historical sales data.
Flags items expected to go out of stock within 3 days for removal from
recommendation candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DemandForecastLSTM(nn.Module):
    """LSTM for per-SKU demand forecasting.

    Input features per day:
    - Units sold
    - Inventory level
    - Price
    - Day of week (one-hot)
    - Is holiday / promotion (binary)
    - Competitor price delta

    Output: forecasted daily demand for next N days.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 30,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

    def forward(
        self,
        historical_series: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            historical_series: (batch_size, seq_len, input_dim)

        Returns:
            (batch_size, forecast_horizon) forecasted daily demand
        """
        lstm_out, _ = self.lstm(historical_series)
        # Use last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.forecast_head(last_hidden)


class InventoryAwareFilter:
    """Filters recommendation candidates based on inventory forecasts.

    Removes items predicted to be out of stock within the threshold window,
    ensuring we never recommend products that can't be fulfilled.
    """

    def __init__(
        self,
        model: DemandForecastLSTM,
        out_of_stock_threshold_days: int = 3,
        safety_stock_days: int = 7,
    ) -> None:
        self.model = model
        self.model.eval()
        self.out_of_stock_threshold = out_of_stock_threshold_days
        self.safety_stock_days = safety_stock_days
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def filter_out_of_stock(
        self,
        candidate_items: List[Tuple[str, float]],
        item_inventory: Dict[str, float],
        item_forecast_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        """Remove items predicted to go out of stock.

        Args:
            candidate_items: [(item_id, score), ...]
            item_inventory: {item_id: current_inventory_level}
            item_forecast_inputs: {item_id: historical_series_tensor}

        Returns:
            (filtered_candidates, removed_items)
        """
        filtered: List[Tuple[str, float]] = []
        removed: List[str] = []

        for item_id, score in candidate_items:
            inventory = item_inventory.get(item_id, float("inf"))

            if item_id in item_forecast_inputs:
                forecast = self._forecast_demand(item_id, item_forecast_inputs[item_id])
                cumulative_demand = forecast[:self.safety_stock_days].sum().item()

                if cumulative_demand >= inventory:
                    removed.append(item_id)
                    continue

            filtered.append((item_id, score))

        return filtered, removed

    def _forecast_demand(
        self,
        item_id: str,
        historical_series: torch.Tensor,
    ) -> torch.Tensor:
        """Run demand forecast for a single item."""
        self.model.eval()
        with torch.no_grad():
            series = historical_series.unsqueeze(0).to(self.device)
            forecast = self.model(series)
        return forecast.squeeze(0)

    def compute_stockout_probability(
        self,
        current_inventory: float,
        forecasted_demand: torch.Tensor,
        days: int = 7,
    ) -> float:
        """Probability of stockout within N days."""
        cumulative = forecasted_demand[:days].sum().item()
        if cumulative <= 0:
            return 0.0
        return min(1.0, cumulative / max(current_inventory, 1.0))
