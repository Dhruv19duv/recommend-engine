"""Kafka pipeline for real-time event streaming and cache invalidation.

Handles 1M recommendation requests per second with Kafka-based event processing.
Drives cache invalidation on every purchase event.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KafkaMessage:
    """A Kafka message wrapper."""

    topic: str
    key: str
    value: Dict[str, Any]
    partition: Optional[int] = None
    timestamp: Optional[float] = None


class KafkaEventProducer:
    """Async Kafka producer for recommendation events.

    Produces to topics:
    - user-events: All user interactions (views, clicks, purchases)
    - purchases: Purchase events (triggers cache invalidation)
    - inventory-updates: Inventory level changes
    - embedding-updates: Embedding refresh notifications
    - cache-invalidation: Targeted cache invalidation commands
    - fraud-alerts: Detected fraud patterns
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "rec-engine-producer",
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._producer = None

    async def start(self) -> None:
        """Initialize the Kafka producer."""
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                max_batch_size=16384,
                linger_ms=5,
                compression_type="snappy",
            )
            await self._producer.start()
            logger.info("Kafka producer started")
        except Exception:
            logger.warning("Kafka not available, running in offline mode")

    async def produce(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        """Produce a message to Kafka."""
        if self._producer is None:
            return
        try:
            msg_bytes = json.dumps(value).encode("utf-8")
            key_bytes = key.encode("utf-8")
            await self._producer.send(topic, key=key_bytes, value=msg_bytes)
        except Exception as e:
            logger.error(f"Failed to produce to {topic}: {e}")

    async def produce_user_event(
        self,
        user_id: str,
        event_type: str,
        item_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Produce a user interaction event."""
        await self.produce(
            topic="user-events",
            key=user_id,
            value={
                "user_id": user_id,
                "event_type": event_type,
                "item_id": item_id,
                "timestamp": asyncio.get_event_loop().time(),
                **(metadata or {}),
            },
        )

    async def produce_purchase_event(
        self,
        user_id: str,
        item_ids: List[str],
        total_amount: float,
    ) -> None:
        """Produce a purchase event (triggers cache invalidation)."""
        await self.produce(
            topic="purchases",
            key=user_id,
            value={
                "user_id": user_id,
                "item_ids": item_ids,
                "total_amount": total_amount,
                "timestamp": asyncio.get_event_loop().time(),
            },
        )

    async def stop(self) -> None:
        """Stop the producer."""
        if self._producer:
            await self._producer.stop()


class KafkaEventConsumer:
    """Async Kafka consumer for processing events.

    Processes:
    - Purchase events: Invalidate caches, update regret tracking
    - Inventory events: Update demand forecasts, remove OOS items
    - Embedding updates: Refresh ANN index
    - Fraud alerts: Update fraud detector state
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "rec-engine",
        topics: Optional[List[str]] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or [
            "user-events", "purchases", "inventory-updates",
            "embedding-updates", "cache-invalidation", "fraud-alerts",
        ]
        self._consumer = None
        self._handlers: Dict[str, Callable[[KafkaMessage], Awaitable[None]]] = {}

    def register_handler(
        self,
        topic: str,
        handler: Callable[[KafkaMessage], Awaitable[None]],
    ) -> None:
        """Register a handler for a specific topic."""
        self._handlers[topic] = handler

    async def start(self) -> None:
        """Start consuming messages."""
        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                max_poll_records=500,
            )
            await self._consumer.start()
            logger.info(f"Kafka consumer started (topics: {self.topics})")

            asyncio.create_task(self._consume_loop())
        except Exception as e:
            logger.warning(f"Kafka consumer failed to start: {e}")

    async def _consume_loop(self) -> None:
        """Main consume loop."""
        if self._consumer is None:
            return

        try:
            async for msg in self._consumer:
                topic = msg.topic
                if topic in self._handlers:
                    kafka_msg = KafkaMessage(
                        topic=topic,
                        key=msg.key.decode("utf-8") if msg.key else "",
                        value=json.loads(msg.value.decode("utf-8")),
                        partition=msg.partition,
                        timestamp=msg.timestamp / 1000.0 if msg.timestamp else None,
                    )
                    try:
                        await self._handlers[topic](kafka_msg)
                    except Exception as e:
                        logger.error(f"Handler failed for {topic}: {e}")
        except Exception as e:
            logger.error(f"Consume loop error: {e}")

    async def stop(self) -> None:
        """Stop the consumer."""
        if self._consumer:
            await self._consumer.stop()
