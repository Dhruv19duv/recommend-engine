"""Tests for ML models."""

import numpy as np
import pytest
import torch

from src.models.demand_forecast import DemandForecastLSTM, InventoryAwareFilter
from src.models.fraud_gnn import FraudGNN, FraudDetector
from src.models.graphsage import HeterogeneousGraphSAGE, SAGELayer
from src.models.price_elasticity import ElasticityAwareRanker, PriceElasticityModel
from src.models.time_lstm import PreferenceVolatilityModule, TimeAwareLSTM
from src.models.two_tower import TwoTowerModel, UserTower, ItemTower


class TestUserTower:
    def test_forward_shape(self):
        tower = UserTower(hidden_dims=[32, 16], output_dim=32, num_hash_buckets=100)
        batch_size = 8
        user_ids = torch.randint(0, 100, (batch_size,))
        context = torch.randn(batch_size, 64)
        output = tower(user_ids, context)
        assert output.shape == (batch_size, 32)


class TestItemTower:
    def test_forward_shape(self):
        tower = ItemTower(
            hidden_dims=[32, 16], output_dim=32,
            num_categories=10, num_sellers=10, num_hash_buckets=100,
        )
        batch_size = 8
        item_ids = torch.randint(0, 100, (batch_size,))
        category_ids = torch.randint(0, 10, (batch_size,))
        seller_ids = torch.randint(0, 10, (batch_size,))
        prices = torch.randn(batch_size, 1)
        images = torch.randn(batch_size, 128)
        output = tower(item_ids, category_ids, seller_ids, prices, images)
        assert output.shape == (batch_size, 32)


class TestTwoTower:
    def test_forward_shape(self):
        model = TwoTowerModel(output_dim=32, user_hash_buckets=100, item_hash_buckets=100)
        batch_size = 4
        user_ids = torch.randint(0, 100, (batch_size,))
        context = torch.randn(batch_size, 64)
        item_ids = torch.randint(0, 100, (batch_size,))
        category_ids = torch.randint(0, 10, (batch_size,))
        seller_ids = torch.randint(0, 10, (batch_size,))
        prices = torch.randn(batch_size, 1)
        images = torch.randn(batch_size, 128)

        user_emb, item_emb = model(user_ids, context, item_ids, category_ids, seller_ids, prices, images)
        assert user_emb.shape == (batch_size, 32)
        assert item_emb.shape == (batch_size, 32)

    def test_score(self):
        model = TwoTowerModel(output_dim=32)
        user_emb = torch.randn(4, 32)
        item_emb = torch.randn(6, 32)
        scores = model.score(user_emb, item_emb)
        assert scores.shape == (4, 6)


class TestHeterogeneousGraphSAGE:
    def test_forward_shape(self):
        model = HeterogeneousGraphSAGE(
            num_edge_types=3,
            feature_dim=16,
            hidden_dims=[32],
            output_dim=16,
            num_layers=2,
        )
        n_nodes = 10
        features = torch.randn(n_nodes, 16)
        neighbor_feats = [torch.randn(n_nodes, 16) for _ in range(3)]
        output = model(features, neighbor_feats)
        assert output.shape == (n_nodes, 16)


class TestTimeAwareLSTM:
    def test_get_mood_state(self):
        lstm = TimeAwareLSTM(
            input_dim=16, hidden_dim=32, output_dim=8,
        )
        module = PreferenceVolatilityModule(lstm, item_embedding_dim=32)

        batch_size, seq_len = 4, 10
        interaction_seq = torch.randn(batch_size, seq_len, 16)
        time_buckets = torch.randint(0, 24, (batch_size, seq_len))
        day_buckets = torch.randint(0, 7, (batch_size, seq_len))
        is_weekend = torch.randint(0, 2, (batch_size, seq_len))
        is_payday = torch.randint(0, 2, (batch_size, seq_len))

        mood_state, context_logits = module.get_mood_state(
            interaction_seq, time_buckets, day_buckets, is_weekend, is_payday
        )
        assert mood_state.shape == (batch_size, 8)
        assert context_logits.shape == (batch_size, 5)


class TestFraudGNN:
    def test_forward_shape(self):
        model = FraudGNN(feature_dim=16, hidden_dim=8, num_layers=2)
        n_sellers = 10
        features = torch.randn(n_sellers, 16)
        # Create random edges
        edges = torch.randint(0, n_sellers, (2, 20))
        fraud_probs = model(features, edges)
        assert fraud_probs.shape == (n_sellers, 1)
        assert (fraud_probs >= 0).all() and (fraud_probs <= 1).all()


class TestDemandForecast:
    def test_forecast_shape(self):
        model = DemandForecastLSTM(
            input_dim=16, hidden_dim=32, forecast_horizon=30,
        )
        batch_size, seq_len = 8, 60
        series = torch.randn(batch_size, seq_len, 16)
        forecast = model(series)
        assert forecast.shape == (batch_size, 30)


class TestPriceElasticity:
    def test_forward_shape(self):
        model = PriceElasticityModel(feature_dim=16, hidden_dim=8)
        features = torch.randn(4, 16)
        elasticity, vq_weight, wtp = model(features)
        assert elasticity.shape == (4, 1)
        assert vq_weight.shape == (4, 1)
        assert wtp.shape == (4, 1)
        assert (vq_weight >= 0).all() and (vq_weight <= 1).all()
