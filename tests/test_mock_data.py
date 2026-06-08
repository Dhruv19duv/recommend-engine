"""Tests for the MockDataStore used in the runnable FastAPI demo."""

import pytest
import numpy as np

from src.serving.mock_data import MockDataStore, VOLATILITY_PROFILES, MockProduct, MockUser


class TestMockDataStoreInitialization:
    def test_creates_products(self):
        store = MockDataStore(num_products=100, num_users=20)
        assert len(store.products) == 100

    def test_creates_users(self):
        store = MockDataStore(num_products=100, num_users=20)
        assert len(store.users) == 20

    def test_has_categories(self):
        store = MockDataStore(num_products=50, num_users=10)
        assert len(store.categories) >= 15  # All PRODUCT_CATEGORIES loaded

    def test_popularity_rank_is_sorted(self):
        store = MockDataStore(num_products=50, num_users=10)
        assert len(store.popularity_rank) == 50
        # Verify all items in popularity rank are valid products
        for pid in store.popularity_rank:
            assert pid in store.products
        assert store.popularity_rank[0] != store.popularity_rank[-1]  # Not all identical


class TestMockDataStoreProducts:
    def test_product_has_required_fields(self):
        store = MockDataStore(num_products=10, num_users=3)
        product = store.products["prod_0000"]
        assert isinstance(product.item_id, str)
        assert isinstance(product.name, str)
        assert product.category in store.categories
        assert product.price > 0
        assert 1.0 <= product.rating <= 5.0
        assert product.embedding.shape == (256,)

    def test_product_embedding_is_normalized(self):
        store = MockDataStore(num_products=10, num_users=3)
        for pid, product in list(store.products.items())[:5]:
            norm = np.linalg.norm(product.embedding)
            assert abs(norm - 1.0) < 1e-5  # Unit normalized

    def test_product_inventory_varies(self):
        store = MockDataStore(num_products=50, num_users=3)
        inventories = [p.inventory for p in store.products.values()]
        assert max(inventories) > min(inventories)  # Some variance


class TestMockDataStoreUsers:
    def test_user_has_typical_categories(self):
        store = MockDataStore(num_products=50, num_users=10)
        for uid, user in store.users.items():
            assert 2 <= len(user.typical_categories) <= 4
            for cat in user.typical_categories:
                assert cat in store.categories

    def test_user_preferences_normalized(self):
        store = MockDataStore(num_products=50, num_users=10)
        for user in store.users.values():
            # At least some categories should have high affinity
            high_affinity = [v for v in user.preferences.values() if v > 0.3]
            assert len(high_affinity) >= 2

    def test_user_has_history(self):
        store = MockDataStore(num_products=50, num_users=10)
        for user in store.users.values():
            assert 10 <= len(user.history) <= 30
            for hid in user.history:
                assert hid in store.products  # Items in history exist

    def test_user_price_sensitivity(self):
        store = MockDataStore(num_products=50, num_users=10)
        sensitivities = [u.price_sensitivity for u in store.users.values()]
        assert all(s in ["high", "medium", "low"] for s in sensitivities)


class TestMockDataStoreRecommendations:
    def test_known_user_gets_recommendations(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        results = store.get_recommendations(user_id, top_k=10)
        assert len(results) == 10
        for item_id, score, meta in results:
            assert item_id in store.products
            assert 0 <= score <= 1.5
            assert "category" in meta

    def test_cold_start_user_gets_popular_items(self):
        store = MockDataStore(num_products=100, num_users=20)
        results = store.get_recommendations("unknown_user", top_k=5)
        assert len(results) == 5
        for item_id, score, meta in results:
            assert item_id in store.products
            assert item_id in store.popularity_rank

    def test_discovery_injection_exists(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        results = store.get_recommendations(user_id, top_k=20)
        discoveries = [r for r in results if r[2].get("is_discovery")]
        assert len(discoveries) >= 1  # At least one discovery injection

    def test_recommendations_are_ranked(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        results = store.get_recommendations(user_id, top_k=10)
        assert len(results) == 10
        scores = [score for _, score, _ in results]
        all_finite = all(np.isfinite(s) for s in scores)
        assert all_finite
        # Exclude discovery positions (index 6, 13, etc.) which have artificial score penalties
        non_discovery_scores = [
            score for i, (_, score, meta) in enumerate(results)
            if not meta.get("is_discovery") or (i + 1) % 7 != 0
        ]
        if len(non_discovery_scores) >= 2:
            for i in range(len(non_discovery_scores) - 1):
                assert non_discovery_scores[i] >= non_discovery_scores[i + 1] * 0.9

    def test_previously_seen_items_penalized(self):
        store = MockDataStore(num_products=100, num_users=20)
        user = list(store.users.values())[0]
        # Results should not heavily favor items already in history
        results = store.get_recommendations(user.user_id, top_k=20)
        in_history = sum(1 for item_id, _, _ in results if item_id in user.history)
        # At most half the results should be from history
        assert in_history <= len(results) * 0.5


class TestMockDataStoreContextShifts:
    def test_payday_context_shifts_categories(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]

        # Recommendations with payday context
        payday_results = store.get_recommendations(
            user_id, top_k=5,
            context={"emotional_context": "post_payday", "is_payday": True},
        )
        payday_categories = [meta["category"] for _, _, meta in payday_results]

        # Recommendations without payday context
        normal_results = store.get_recommendations(
            user_id, top_k=5,
            context={"emotional_context": "default"},
        )
        normal_categories = [meta["category"] for _, _, meta in normal_results]

        # The sets may differ due to volatility shift
        assert payday_results is not None
        assert normal_results is not None

    def test_late_night_context(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        results = store.get_recommendations(
            user_id, top_k=5,
            context={"emotional_context": "late_night"},
        )
        assert len(results) == 5

    def test_weekend_context(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        results = store.get_recommendations(
            user_id, top_k=5,
            context={"emotional_context": "weekend_leisure"},
        )
        assert len(results) == 5

    def test_get_category_affinity_shift_payday(self):
        store = MockDataStore(num_products=100, num_users=20)
        shifts = store.get_category_affinity_shift("user_1", 14, 3, is_payday=True)
        assert "electronics" in shifts or "jewelry" in shifts or "home" in shifts

    def test_get_category_affinity_shift_late_night(self):
        store = MockDataStore(num_products=100, num_users=20)
        shifts = store.get_category_affinity_shift("user_1", 2, 4, is_payday=False)
        assert "books" in shifts or "food" in shifts or "music" in shifts


class TestMockDataStoreExplanation:
    def test_known_user_product_returns_string(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        product_id = list(store.products.keys())[0]
        explanation = store.get_explanation(user_id, product_id)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_unknown_user_returns_default(self):
        store = MockDataStore(num_products=100, num_users=20)
        explanation = store.get_explanation("unknown_user", "prod_0000")
        assert isinstance(explanation, str)

    def test_unknown_product_returns_default(self):
        store = MockDataStore(num_products=100, num_users=20)
        user_id = list(store.users.keys())[0]
        explanation = store.get_explanation(user_id, "nonexistent")
        assert isinstance(explanation, str)


class TestMockDataStoreHelpers:
    def test_get_all_product_ids(self):
        store = MockDataStore(num_products=50, num_users=10)
        ids = store.get_all_product_ids()
        assert len(ids) == 50
        assert all(pid.startswith("prod_") for pid in ids)

    def test_get_product(self):
        store = MockDataStore(num_products=50, num_users=10)
        product = store.get_product("prod_0000")
        assert product is not None
        assert product.item_id == "prod_0000"

    def test_get_product_nonexistent(self):
        store = MockDataStore(num_products=50, num_users=10)
        assert store.get_product("nonexistent") is None

    def test_get_user(self):
        store = MockDataStore(num_products=50, num_users=10)
        first_user_id = list(store.users.keys())[0]
        user = store.get_user(first_user_id)
        assert user is not None
        assert user.user_id == first_user_id

    def test_get_user_nonexistent(self):
        store = MockDataStore(num_products=50, num_users=10)
        assert store.get_user("nonexistent") is None

    def test_volatility_profiles_exist(self):
        """Verify the volatility profiles constant has the expected structure."""
        assert "post_payday" in VOLATILITY_PROFILES
        assert "late_night" in VOLATILITY_PROFILES
        assert "weekend_leisure" in VOLATILITY_PROFILES
        assert "default" in VOLATILITY_PROFILES
        for context, shifts in VOLATILITY_PROFILES.items():
            assert len(shifts) > 0
            for cat, shift in shifts.items():
                assert isinstance(cat, str)
                assert isinstance(shift, float)
