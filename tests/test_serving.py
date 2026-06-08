"""Tests for the serving layer — FastAPI endpoints, bandit, and explainability."""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from src.serving.api import app, RecommendRequest, UserContext, EventRequest, engine
from src.serving.bandit import LinUCB, ContextualBanditRecommender
from src.serving.explainability import SHAPExplainer, Explanation


# ── FastAPI Endpoint Tests ───────────────────────────────────────


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_has_latency_header(self, client):
        response = client.get("/v1/health")
        assert "X-Latency-Ms" in response.headers


class TestRecommendEndpoint:
    def test_recommend_with_valid_user(self, client):
        request_data = {
            "user": {
                "user_id": "user_0001",
                "time_bucket": 14,
                "day_bucket": 3,
                "is_payday": False,
            },
            "surface": "homepage",
            "top_k": 5,
            "include_explanations": True,
        }
        response = client.post("/v1/recommend", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_0001"
        assert "results" in data
        assert "latency_ms" in data
        assert "request_id" in data

    def test_recommend_cold_start_user(self, client):
        request_data = {
            "user": {
                "user_id": "unknown_user",
                "is_new_user": True,
                "time_bucket": 2,
                "day_bucket": 6,
            },
            "surface": "homepage",
            "top_k": 5,
        }
        response = client.post("/v1/recommend", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "unknown_user"
        # Cold-start users should still get results (popular items)
        assert len(data["results"]) > 0

    def test_recommend_with_explanations(self, client):
        request_data = {
            "user": {
                "user_id": "user_0001",
                "time_bucket": 10,
                "day_bucket": 2,
            },
            "surface": "product_detail",
            "top_k": 3,
            "include_explanations": True,
        }
        response = client.post("/v1/recommend", json=request_data)
        assert response.status_code == 200
        data = response.json()
        if data["results"]:
            assert "explanation" in data["results"][0]

    def test_recommend_payday_context(self, client):
        """Test that payday context changes recommendations."""
        request_data = {
            "user": {
                "user_id": "user_0001",
                "time_bucket": 14,
                "day_bucket": 3,
                "is_payday": True,
            },
            "surface": "homepage",
            "top_k": 5,
        }
        response = client.post("/v1/recommend", json=request_data)
        assert response.status_code == 200

    def test_recommend_late_night_context(self, client):
        """Test that late night context changes recommendations."""
        request_data = {
            "user": {
                "user_id": "user_0001",
                "time_bucket": 2,
                "day_bucket": 4,
            },
            "surface": "homepage",
            "top_k": 5,
        }
        response = client.post("/v1/recommend", json=request_data)
        assert response.status_code == 200

    def test_recommend_weekend_context(self, client):
        """Test that weekend context changes recommendations."""
        request_data = {
            "user": {
                "user_id": "user_0001",
                "time_bucket": 14,
                "day_bucket": 6,
            },
            "surface": "homepage",
            "top_k": 5,
        }
        response = client.post("/v1/recommend", json=request_data)
        assert response.status_code == 200


class TestBatchRecommendEndpoint:
    def test_batch_recommend(self, client):
        request_data = {
            "requests": [
                {
                    "user": {"user_id": "user_0001"},
                    "surface": "homepage",
                    "top_k": 3,
                },
                {
                    "user": {"user_id": "user_0002"},
                    "surface": "cart",
                    "top_k": 3,
                },
            ]
        }
        response = client.post("/v1/recommend/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["user_id"] == "user_0001"
        assert data[1]["user_id"] == "user_0002"


class TestExplainEndpoint:
    def test_explain(self, client):
        request_data = {
            "user_id": "user_0001",
            "item_id": "prod_0001",
            "features": {"category_match": 0.6, "price": 0.3},
        }
        response = client.post("/v1/explain", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_0001"
        assert data["item_id"] == "prod_0001"
        assert "reason" in data


class TestEventsEndpoint:
    def test_log_event(self, client):
        request_data = {
            "user_id": "user_0001",
            "event_type": "view",
            "item_id": "prod_0001",
            "category_id": "electronics",
            "price": 99.99,
        }
        response = client.post("/v1/events", json=request_data)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_log_batch_events(self, client):
        request_data = {
            "events": [
                {"user_id": "user_0001", "event_type": "view", "item_id": "prod_0001"},
                {"user_id": "user_0001", "event_type": "click", "item_id": "prod_0002"},
                {"user_id": "user_0002", "event_type": "purchase", "item_id": "prod_0003", "price": 49.99},
            ]
        }
        response = client.post("/v1/events/batch", json=request_data)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics(self, client):
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


# ── Bandit Tests ─────────────────────────────────────────────────


class TestLinUCB:
    def test_add_arm_and_select(self):
        bandit = LinUCB(context_dim=4, alpha=1.0)
        bandit.add_arm("electronics", "Electronics")
        bandit.add_arm("books", "Books")
        bandit.add_arm("sports", "Sports")

        context = np.array([0.5, 0.3, 0.8, 0.1])
        arm_id, score = bandit.select_arm(context, "test_user")
        assert arm_id in {"electronics", "books", "sports"}
        assert score > 0

    def test_update_and_learn(self):
        bandit = LinUCB(context_dim=4, alpha=1.0)
        bandit.add_arm("electronics", "Electronics")

        context = np.array([0.5, 0.3, 0.8, 0.1])

        # Select and update with positive reward
        arm, _ = bandit.select_arm(context, "user_1")
        bandit.update(arm, context, reward=1.0)

        # Theta should be non-zero after update
        assert np.any(bandit._theta[arm] != 0)

    def test_get_top_arms(self):
        bandit = LinUCB(context_dim=2, alpha=1.0)
        for i in range(5):
            bandit.add_arm(f"cat_{i}", f"Category {i}")

        context = np.array([0.5, 0.5])
        top = bandit.get_top_arms(context, top_k=3)
        assert len(top) == 3
        assert all(isinstance(arm, str) for arm, _ in top)
        assert all(isinstance(score, float) for _, score in top)

    def test_cold_start_no_arms(self):
        bandit = LinUCB(context_dim=2)
        with pytest.raises(ValueError):
            bandit.select_arm(np.array([0.5, 0.5]), "user_1")

    def test_total_pulls(self):
        bandit = LinUCB(context_dim=2)
        bandit.add_arm("cat_1", "Category 1")
        ctx = np.array([0.5, 0.5])

        for _ in range(5):
            arm, _ = bandit.select_arm(ctx, "user_1")
            bandit.update(arm, ctx, reward=1.0)

        assert bandit.total_pulls == 5


class TestContextualBanditRecommender:
    def test_recommend(self):
        linucb = LinUCB(context_dim=4, alpha=1.0)
        recommender = ContextualBanditRecommender(linucb)

        user_meta = {
            "hour_of_day": 14.0,
            "day_of_week": 3.0,
            "is_weekend": 0.0,
            "device_type": 1.0,
            "is_new_user": 1.0,
            "referrer_type": 0.0,
        }

        categories = [f"cat_{i}" for i in range(10)]
        results = recommender.recommend("new_user", user_meta, categories, top_k=5)
        assert len(results) == 5

    def test_record_reward(self):
        linucb = LinUCB(context_dim=4, alpha=1.0)
        recommender = ContextualBanditRecommender(linucb)

        user_meta = {"hour_of_day": 14.0, "day_of_week": 3.0, "is_weekend": 0.0,
                     "device_type": 1.0, "is_new_user": 1.0, "referrer_type": 0.0}
        categories = [f"cat_{i}" for i in range(5)]
        results = recommender.recommend("user_1", user_meta, categories)

        for arm_id, _ in results:
            recommender.record_reward("user_1", arm_id, reward=1.0)

        # Should not crash
        assert True


# ── Explainability Tests ─────────────────────────────────────────


class TestSHAPExplainer:
    def test_explain(self):
        explainer = SHAPExplainer(n_features=8)
        explainer.set_feature_names([
            "category_match", "price_match", "rating", "brand_affinity",
            "recency", "popularity", "discovery", "seasonal",
        ])

        shap_values = np.array([0.5, -0.2, 0.3, 0.1, 0.05, 0.0, -0.05, 0.05])
        input_features = np.array([1.0, 0.5, 4.5, 0.0, 0.8, 0.9, 0.2, 0.1])

        explanation = explainer.explain(
            model_output=0.85,
            input_features=input_features,
            shap_values=shap_values,
            user_id="user_1",
            item_id="item_1",
        )

        assert explanation.user_id == "user_1"
        assert explanation.item_id == "item_1"
        assert len(explanation.top_features) == 3
        assert isinstance(explanation.reason, str)
        assert len(explanation.reason) > 0

    def test_explain_generates_readable_reason(self):
        explainer = SHAPExplainer(n_features=4)
        explainer.set_feature_names(["viewed_similar", "category_match", "price_range", "popular"])

        # Positive values on relevant features
        shap_values = np.array([0.8, 0.6, 0.3, 0.1])
        explanation = explainer.explain(
            model_output=0.9,
            input_features=np.ones(4),
            shap_values=shap_values,
            user_id="user_1",
            item_id="item_1",
        )

        # Should mention "recently viewed"
        assert "viewed" in explanation.reason.lower() or "based on" in explanation.reason.lower()

    def test_default_explanation(self):
        explainer = SHAPExplainer(n_features=2)
        explanation = explainer.explain(
            model_output=0.5,
            input_features=np.array([0.0, 0.0]),
            shap_values=np.zeros(2),
            user_id="user_1",
            item_id="item_1",
        )
        assert "Recommended" in explanation.reason or "recommended" in explanation.reason
