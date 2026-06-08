"""Mock data generator for the runnable FastAPI demo.

Generates realistic fake products, categories, and user profiles
so the API returns meaningful recommendations without needing
trained models or external infrastructure.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Product Catalog ──────────────────────────────────────────────


PRODUCT_CATEGORIES = [
    "electronics", "apparel", "home", "books", "sports",
    "beauty", "food", "toys", "automotive", "health",
    "office", "garden", "music", "pet_supplies", "jewelry",
]

SUB_CATEGORIES: Dict[str, List[str]] = {
    "electronics": ["phones", "laptops", "headphones", "cameras", "smart_home"],
    "apparel": ["men", "women", "kids", "shoes", "accessories"],
    "home": ["furniture", "kitchen", "decor", "bedding", "lighting"],
    "books": ["fiction", "non_fiction", "science", "history", "self_help"],
    "sports": ["fitness", "outdoor", "team_sports", "swimming", "cycling"],
    "beauty": ["skincare", "makeup", "hair", "fragrance", "nail"],
    "food": ["snacks", "beverages", "grocery", "gourmet", "organic"],
    "toys": ["educational", "action_figures", "board_games", "outdoor", "puzzles"],
    "automotive": ["parts", "accessories", "tools", "care", "electronics"],
    "health": ["supplements", "personal_care", "medical", "wellness", "vitamins"],
    "office": ["supplies", "furniture", "technology", "paper", "organization"],
    "garden": ["plants", "tools", "decor", "outdoor_living", "seeds"],
    "music": ["instruments", "accessories", "audio", "sheet_music", "lessons"],
    "pet_supplies": ["dogs", "cats", "fish", "birds", "small_animals"],
    "jewelry": ["necklaces", "rings", "bracelets", "earrings", "watches"],
}

BRANDS = [
    "TechPro", "StyleCraft", "HomeEssentials", "ReadWell", "SportMax",
    "GlowUp", "FreshBites", "PlayWorld", "AutoGear", "WellLife",
    "WorkSmart", "GardenJoy", "MelodyPlus", "PetPals", "SparkleGems",
    "NovaTech", "UrbanWear", "CozyHome", "PageTurner", "FitLife",
]

# Emotional context -> category affinities for the volatility engine
VOLATILITY_PROFILES: Dict[str, Dict[str, float]] = {
    "post_payday": {"electronics": 0.3, "jewelry": 0.4, "home": 0.2, "apparel": 0.15},
    "late_night": {"books": 0.4, "food": 0.3, "music": 0.2, "health": 0.1},
    "weekend_leisure": {"sports": 0.3, "garden": 0.3, "toys": 0.2, "automotive": 0.2},
    "emotional_comfort": {"beauty": 0.4, "food": 0.3, "books": 0.3},
    "workday_lunch": {"food": 0.3, "office": 0.2, "books": 0.2},
    "default": {"apparel": 0.2, "home": 0.2, "electronics": 0.2, "books": 0.2},
}


@dataclass
class MockProduct:
    """A mock product in the catalog."""

    item_id: str
    name: str
    category: str
    subcategory: str
    brand: str
    price: float
    rating: float
    review_count: int
    inventory: int
    seller_id: str
    image_url: str
    tags: List[str] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256))


@dataclass
class MockUser:
    """A mock user profile."""

    user_id: str
    name: str
    preferences: Dict[str, float]  # category -> affinity
    price_sensitivity: str  # high, medium, low
    typical_categories: List[str]
    history: List[str]  # item_ids interacted with
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256))


class MockDataStore:
    """In-memory mock data store for the demo."""

    def __init__(self, num_products: int = 200, num_users: int = 50) -> None:
        self.num_products = num_products
        self.num_users = num_users
        self.products: Dict[str, MockProduct] = {}
        self.users: Dict[str, MockUser] = {}
        self.categories: List[str] = PRODUCT_CATEGORIES
        self.subcategories: Dict[str, List[str]] = SUB_CATEGORIES
        self.popularity_rank: List[str] = []  # item_ids sorted by popularity

        self._generate_products()
        self._generate_users()
        self._compute_popularity()

    def _hash_str(self, s: str, mod: int = 10000) -> int:
        """Simple string hash for deterministic results."""
        return abs(hash(s)) % mod

    def _generate_products(self) -> None:
        """Generate mock products with realistic distributions."""
        rng = random.Random(42)

        for i in range(self.num_products):
            item_id = f"prod_{i:04d}"
            category = rng.choice(self.categories)
            subcategory = rng.choice(self.subcategories[category])
            brand = rng.choice(BRANDS)
            price = round(rng.lognormvariate(3.5, 0.8), 2)  # $5-$200 range
            rating = round(min(5.0, max(1.0, rng.gauss(4.0, 0.6))), 1)
            review_count = rng.randint(0, 5000)
            inventory = rng.randint(0, 1000)
            seller_id = f"seller_{self._hash_str(brand, 100):04d}"

            # Embedding: simple deterministic vector based on category + subcategory
            cat_idx = self.categories.index(category)
            subcat_idx = SUB_CATEGORIES[category].index(subcategory)
            emb = np.zeros(256)
            emb[cat_idx * 16 : cat_idx * 16 + 16] = np.sin(np.arange(16) + subcat_idx)
            emb = emb / np.linalg.norm(emb)

            product = MockProduct(
                item_id=item_id,
                name=f"{brand} {subcategory.title()} #{i}",
                category=category,
                subcategory=subcategory,
                brand=brand,
                price=price,
                rating=rating,
                review_count=review_count,
                inventory=inventory,
                seller_id=seller_id,
                image_url=f"https://picsum.photos/seed/{item_id}/200/200",
                tags=[category, subcategory, brand.lower()],
                embedding=emb,
            )
            self.products[item_id] = product

    def _generate_users(self) -> None:
        """Generate mock users with diverse preference profiles."""
        rng = random.Random(123)

        for i in range(self.num_users):
            user_id = f"user_{i:04d}"

            # Each user has 2-4 preferred categories
            num_cats = rng.randint(2, 4)
            preferred_cats = rng.sample(self.categories, num_cats)

            # Affinity scores for all categories
            preferences: Dict[str, float] = {}
            for cat in self.categories:
                if cat in preferred_cats:
                    preferences[cat] = rng.uniform(0.3, 1.0)
                else:
                    preferences[cat] = rng.uniform(0.0, 0.3)

            # Price sensitivity
            sensitivity = rng.choices(
                ["high", "medium", "low"],
                weights=[0.3, 0.4, 0.3],
            )[0]

            # Build interaction history (10-30 items from preferred categories)
            history_size = rng.randint(10, 30)
            preferred_items = [
                pid for pid, p in self.products.items()
                if p.category in preferred_cats
            ]
            history = rng.sample(
                preferred_items,
                min(history_size, len(preferred_items)),
            )

            # User embedding: aggregate of their product preferences
            if history:
                emb = np.mean([self.products[hid].embedding for hid in history], axis=0)
            else:
                emb = np.zeros(256)

            user = MockUser(
                user_id=user_id,
                name=f"User {i}",
                preferences=preferences,
                price_sensitivity=sensitivity,
                typical_categories=preferred_cats,
                history=history,
                embedding=emb,
            )
            self.users[user_id] = user

    def _compute_popularity(self) -> None:
        """Rank products by a composite popularity score."""
        scores = []
        for pid, product in self.products.items():
            score = product.rating * 0.4 + np.log1p(product.review_count) * 0.3
            inventory_factor = min(1.0, product.inventory / 100.0)
            score *= (0.5 + 0.5 * inventory_factor)
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        self.popularity_rank = [pid for pid, _ in scores]

    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        surface: str = "homepage",
        context: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """Get mock recommendations for a user.

        Returns: [(item_id, score, metadata), ...]
        """
        user = self.users.get(user_id)
        if user is None:
            # Cold start: return popular items
            return self._get_popular_recommendations(top_k, context)

        # Score products based on user preferences
        scored: List[Tuple[str, float, Dict]] = []
        for pid, product in self.products.items():
            if product.inventory <= 0:
                continue

            # Base: category affinity
            cat_affinity = user.preferences.get(product.category, 0.0)

            # Apply volatility shift if context is provided
            if context and "emotional_context" in context:
                ec = context["emotional_context"]
                volatility_shift = VOLATILITY_PROFILES.get(ec, {}).get(product.category, 0.0)
                cat_affinity += volatility_shift

            # Price sensitivity adjustment
            if user.price_sensitivity == "high":
                price_score = max(0, 1.0 - (product.price / 200.0))
            elif user.price_sensitivity == "low":
                price_score = 0.5 + 0.5 * (product.rating / 5.0)
            else:
                price_score = 0.5

            # Rating boost
            rating_boost = product.rating / 5.0

            # Composite score
            score = (
                cat_affinity * 0.4
                + price_score * 0.2
                + rating_boost * 0.2
                + (1.0 if pid in user.history else 0.0) * 0.2
            )

            # Dedup already-seen items (slight penalty)
            if pid in user.history:
                score *= 0.7

            metadata = {
                "category": product.category,
                "brand": product.brand,
                "price": product.price,
                "rating": product.rating,
            }
            scored.append((pid, score, metadata))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Inject anti-echo-chamber (every 7th item from a different category)
        seen_categories: set = set()
        result: List[Tuple[str, float, Dict]] = []
        discovery_pool: List[Tuple[str, float, Dict]] = [
            (pid, s, m) for pid, s, m in scored
            if m["category"] not in user.typical_categories
        ]

        for i, (pid, score, meta) in enumerate(scored):
            if (i + 1) % 7 == 0 and discovery_pool:
                # Inject discovery item
                disc = discovery_pool.pop(0) if discovery_pool else (pid, score, meta)
                result.append((disc[0], disc[1] * 0.9, {**disc[2], "is_discovery": True}))
            else:
                result.append((pid, score, meta))

            if len(result) >= top_k:
                break

        if len(result) < top_k:
            # Fill with remaining scored items
            for pid, score, meta in scored:
                if len(result) >= top_k:
                    break
                if not any(r[0] == pid for r in result):
                    result.append((pid, score, meta))

        return result[:top_k]

    def _get_popular_recommendations(
        self,
        top_k: int = 10,
        context: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """Get popular recommendations for cold-start users."""
        result = []
        ec = (context or {}).get("emotional_context", "default")
        volatility_cats = set(VOLATILITY_PROFILES.get(ec, {}).keys())

        for pid in self.popularity_rank:
            product = self.products[pid]
            score = 0.5

            # Boost items in volatility-relevant categories
            if product.category in volatility_cats:
                score += 0.3

            result.append((
                pid, score,
                {"category": product.category, "brand": product.brand,
                 "price": product.price, "rating": product.rating},
            ))
            if len(result) >= top_k:
                break

        return result

    def get_explanation(
        self,
        user_id: str,
        item_id: str,
    ) -> str:
        """Generate a human-readable recommendation explanation."""
        user = self.users.get(user_id)
        product = self.products.get(item_id)

        if not product or not user:
            return "Recommended for you"

        reasons = []
        cat = product.category

        # Category match
        if user.preferences.get(cat, 0) > 0.5:
            reasons.append(f"Similar to {cat} you browse")

        # Price range match
        if user.price_sensitivity == "high" and product.price < 50:
            reasons.append(f"Great value at ${product.price:.0f}")
        elif user.price_sensitivity == "low" and product.rating >= 4.5:
            reasons.append(f"Top-rated in its category")

        # Rating
        if product.rating >= 4.5:
            reasons.append("Highly rated by other shoppers")

        # Brand affinity
        if any(product.brand.lower() in h.lower() for h in user.history[:5]):
            reasons.append(f"You've engaged with {product.brand} before")

        # Discovery
        if cat not in user.typical_categories:
            reasons.append("A fresh discovery for you")
            reasons.append("You might discover something new")

        return " and ".join(reasons[:3]) if reasons else "Recommended based on your profile"

    def get_category_affinity_shift(
        self,
        user_id: str,
        hour_of_day: int,
        day_of_week: int,
        is_payday: bool,
    ) -> Dict[str, float]:
        """Get category affinity shifts based on time context (volatility)."""
        if is_payday:
            ec = "post_payday"
        elif hour_of_day < 6 or hour_of_day >= 23:
            ec = "late_night"
        elif day_of_week >= 5 and 10 <= hour_of_day <= 20:
            ec = "weekend_leisure"
        elif 11 <= hour_of_day <= 13 and day_of_week < 5:
            ec = "workday_lunch"
        else:
            ec = "default"

        return VOLATILITY_PROFILES.get(ec, {})

    def get_all_product_ids(self) -> List[str]:
        return list(self.products.keys())

    def get_product(self, item_id: str) -> Optional[MockProduct]:
        return self.products.get(item_id)

    def get_user(self, user_id: str) -> Optional[MockUser]:
        return self.users.get(user_id)
