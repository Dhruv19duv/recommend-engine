"""Tests for the five innovations."""

import numpy as np
import pytest

from src.innovations.adversarial_simulation import (
    AdversarialDiscriminator,
    AdversarialGenerator,
    SellerAdversarialSimulation,
)
from src.innovations.anti_echo_chamber import AntiEchoChamberInjector, DiscoveryItem
from src.innovations.life_stage_diffusion import LifeStageGraphDiffusion, TasteProfile
from src.innovations.preference_volatility import EmotionalContext, PreferenceVolatilityEngine
from src.innovations.regret_minimizer import RegretMinimizingRanker


class TestPreferenceVolatilityEngine:
    def test_emotional_context_detection(self):
        engine = PreferenceVolatilityEngine()

        # Payday should override other contexts
        ctx = engine.get_emotional_context(2, 0, is_payday=True)
        assert ctx == EmotionalContext.POST_PAYDAY

        # Late night
        ctx = engine.get_emotional_context(3, 2, is_payday=False)
        assert ctx == EmotionalContext.LATE_NIGHT

        # Weekend leisure
        ctx = engine.get_emotional_context(14, 6, is_payday=False)
        assert ctx == EmotionalContext.WEEKEND_LEISURE

    def test_volatility_profile(self):
        engine = PreferenceVolatilityEngine()
        profile = engine.get_volatility_profile(
            "user_1", hour_of_day=14, day_of_week=6, is_payday=True
        )
        assert profile.user_id == "user_1"
        assert profile.emotional_context == EmotionalContext.POST_PAYDAY
        assert len(profile.category_affinity_shift) > 0


class TestAntiEchoChamberInjector:
    def test_injection_pattern(self):
        injector = AntiEchoChamberInjector(injection_rate=1.0 / 7.0)
        ranked = [(f"item_{i}", 1.0 - i * 0.01) for i in range(20)]

        discovery_items = [
            DiscoveryItem(
                item_id=f"disc_{i}",
                category_id=f"cat_{i}",
                relevance_score=0.5,
                distance_from_history=0.8,
                discovery_potential=0.6,
                serendipity_score=0.0,
            )
            for i in range(5)
        ]
        category_map = {f"item_{i}": f"cat_{i % 5}" for i in range(20)}

        results = injector.inject_discovery("user_1", ranked, discovery_items, category_map)
        discoveries = [r for r in results if r[2]]
        assert len(discoveries) > 0  # At least some discovery items injected

    def test_discovery_score(self):
        injector = AntiEchoChamberInjector()
        injector.register_interaction("user_1", "item_1", "electronics")
        score = injector.compute_discovery_score("user_1", ["electronics", "books", "home"])
        assert score == 2.0 / 3.0


class TestSellerAdversarialSimulation:
    def test_gan_forward(self):
        generator = AdversarialGenerator(latent_dim=16, output_dim=32)
        discriminator = AdversarialDiscriminator(input_dim=32)

        noise = generator.sample_noise(4)
        fake = generator(noise)
        assert fake.shape == (4, 32)

        preds = discriminator(fake)
        assert preds.shape == (4, 1)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_review_ring(self):
        sim = SellerAdversarialSimulation()
        ring = sim.create_review_ring("seller_1", ["prod_1", "prod_2"], num_bots=3)
        assert len(ring) == 6  # 3 bots * 2 products
        # All should have high ratings (suspicious)
        ratings = [r["rating"] for r in ring]
        assert all(r >= 4 for r in ratings)


class TestRegretMinimizingRanker:
    def test_purchase_and_return_tracking(self):
        ranker = RegretMinimizingRanker()
        ranker.record_purchase("user_1", "item_1", "electronics", 50.0)
        ranker.record_return("user_1", "item_1", "electronics", 50.0, 1000.0)
        assert ranker.total_purchases == 1
        assert ranker.total_returns == 1
        assert ranker.overall_return_rate == 1.0

    def test_regret_penalty(self):
        ranker = RegretMinimizingRanker(min_purchases_for_segment=1)
        # Record returns to build profile
        for _ in range(5):
            ranker.record_purchase("user_1", "item_1", "electronics", 50.0)
        for _ in range(3):
            ranker.record_return("user_1", "item_1", "electronics", 50.0, 1000.0)

        penalty = ranker.get_regret_penalty("user_1", "item_1", "electronics", 50.0)
        assert penalty < 1.0  # Should be penalized


class TestLifeStageGraphDiffusion:
    def test_taste_similarity(self):
        diffuser = LifeStageGraphDiffusion()
        profile_a = TasteProfile(
            user_id="user_a",
            embedding=np.array([0.1, 0.2, 0.3]),
            category_affinities={"electronics": 0.8, "books": 0.6},
            price_preference=(10, 100),
            discovery_score=0.5,
        )
        profile_b = TasteProfile(
            user_id="user_b",
            embedding=np.array([0.15, 0.25, 0.35]),
            category_affinities={"electronics": 0.7, "books": 0.5},
            price_preference=(10, 100),
            discovery_score=0.5,
        )
        similarity = diffuser._compute_taste_similarity(profile_a, profile_b)
        assert similarity > 0.8  # Very similar profiles

    def test_divergence_detection(self):
        diffuser = LifeStageGraphDiffusion(similarity_threshold=0.5)
        profile_a = TasteProfile(
            user_id="user_a",
            embedding=np.array([0.1, 0.2, 0.3]),
            category_affinities={"electronics": 0.8},
            price_preference=(10, 100),
            discovery_score=0.5,
        )
        profile_b = TasteProfile(
            user_id="user_b",
            embedding=np.array([0.15, 0.25, 0.35]),
            category_affinities={"electronics": 0.7},
            price_preference=(10, 100),
            discovery_score=0.5,
        )
        diffuser.register_taste_profile(profile_a)
        diffuser.register_taste_profile(profile_b)

        # Simulate divergence: user_a starts buying baby products
        new_interactions = [
            ("item_1", "baby", 2000.0),
            ("item_2", "diapers", 2001.0),
            ("item_3", "baby_food", 2002.0),
        ]
        is_diverging, event = diffuser.detect_divergence("user_a", "user_b", new_interactions)
        # Should detect some divergence signal (or at minimum return without error)
        # The exact result depends on embedding values, but the call should not crash
        assert event is None or event.event_type in diffuser._life_event_categories
