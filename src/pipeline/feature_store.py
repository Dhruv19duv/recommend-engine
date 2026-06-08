"""Feature store integration with Redis and Feast.

Redis: Real-time feature cache for sub-millisecond feature retrieval.
Feast: Online feature store for training/serving feature parity.

Stores:
- User features (profile, embeddings, volatility profile)
- Item features (embeddings, inventory, price, category)
- Context features (time, device, session)
- Pre-computed recommendation scores
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class RedisFeatureStore:
    """Redis-backed feature store for real-time feature retrieval.

    Provides sub-millisecond read/write for user and item features.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        default_ttl: int = 300,  # 5 minutes
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.default_ttl = default_ttl
        self._client = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis

            pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False,
            )
            self._client = redis.Redis(connection_pool=pool)
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis by key."""
        if self._client is None:
            return None
        try:
            data = await self._client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get failed for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in Redis with optional TTL."""
        if self._client is None:
            return
        try:
            data = pickle.dumps(value)
            await self._client.set(key, data, ex=ttl or self.default_ttl)
        except Exception as e:
            logger.error(f"Redis set failed for {key}: {e}")

    async def set_user_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Store a user embedding vector."""
        await self.set(f"user:emb:{user_id}", embedding)

    async def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get a user embedding vector."""
        return await self.get(f"user:emb:{user_id}")

    async def set_item_embedding(
        self,
        item_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Store an item embedding vector."""
        await self.set(f"item:emb:{item_id}", embedding)

    async def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get an item embedding vector."""
        return await self.get(f"item:emb:{item_id}")

    async def set_user_profile(
        self,
        user_id: str,
        profile: Dict[str, Any],
    ) -> None:
        """Store a user profile."""
        await self.set(f"user:profile:{user_id}", profile, ttl=86400)

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user profile."""
        return await self.get(f"user:profile:{user_id}")

    async def set_item_features(
        self,
        item_id: str,
        features: Dict[str, Any],
    ) -> None:
        """Store item features."""
        await self.set(f"item:features:{item_id}", features, ttl=3600)

    async def get_item_features(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item features."""
        return await self.get(f"item:features:{item_id}")

    async def set_recommendation_cache(
        self,
        user_id: str,
        surface: str,
        results: List[Dict[str, Any]],
    ) -> None:
        """Cache recommendation results for a user."""
        await self.set(f"recs:{user_id}:{surface}", results, ttl=60)

    async def get_recommendation_cache(
        self,
        user_id: str,
        surface: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendation results."""
        return await self.get(f"recs:{user_id}:{surface}")

    async def invalidate(self, pattern: str) -> None:
        """Invalidate all keys matching a pattern."""
        if self._client is None:
            return
        try:
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if keys:
                    await self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis invalidation failed: {e}")

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()


class FeastFeatureStore:
    """Feast online feature store wrapper.

    Provides feature parity between training and serving.
    Features defined in feature_store.yaml and registered as FeatureViews.
    """

    def __init__(
        self,
        serving_url: str = "localhost:6566",
        repo_path: str = "./feast",
    ) -> None:
        self.serving_url = serving_url
        self.repo_path = repo_path
        self._client = None

    def connect(self) -> None:
        """Connect to the Feast feature server."""
        try:
            from feast import FeatureStore
            self._client = FeatureStore(repo_path=self.repo_path)
            logger.info(f"Connected to Feast feature store at {self.repo_path}")
        except Exception as e:
            logger.warning(f"Feast connection failed: {e}")

    def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        """Retrieve online features for entities.

        Args:
            feature_refs: List of feature references (e.g., ["user_features:embedding"])
            entity_rows: List of dicts with entity keys (e.g., [{"user_id": "abc"}])

        Returns:
            Dict of feature_name -> list of feature values
        """
        if self._client is None:
            return {}
        try:
            features = self._client.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            ).to_dict()
            return features
        except Exception as e:
            logger.error(f"Feast get_online_features failed: {e}")
            return {}

    def get_user_features(
        self,
        user_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get online features for users."""
        if feature_names is None:
            feature_names = [
                "user_features:embedding",
                "user_features:volatility_profile",
                "user_features:price_elasticity",
            ]
        entity_rows = [{"user_id": uid} for uid in user_ids]
        return self.get_online_features(feature_names, entity_rows)

    def get_item_features(
        self,
        item_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get online features for items."""
        if feature_names is None:
            feature_names = [
                "item_features:embedding",
                "item_features:category",
                "item_features:price",
                "item_features:inventory",
            ]
        entity_rows = [{"item_id": iid} for iid in item_ids]
        return self.get_online_features(feature_names, entity_rows)
