"""Tests for the pipeline layer — Kafka, feature store, and Spark training."""

import pytest

from src.pipeline.kafka_pipeline import KafkaEventProducer, KafkaEventConsumer, KafkaMessage
from src.pipeline.feature_store import RedisFeatureStore, FeastFeatureStore
from src.pipeline.spark_training import SparkTrainingPipeline, TrainingConfig


# ── Kafka Pipeline Tests ─────────────────────────────────────────


class TestKafkaEventProducer:
    @pytest.mark.asyncio
    async def test_producer_initialization(self):
        """Producer should initialize without error (offline mode)."""
        producer = KafkaEventProducer(bootstrap_servers="localhost:9092")
        # Should not crash — runs in offline mode if Kafka unavailable
        await producer.start()
        assert producer._producer is None  # Offline mode
        await producer.stop()

    @pytest.mark.asyncio
    async def test_produce_offline(self):
        """Producing in offline mode should not crash."""
        producer = KafkaEventProducer(bootstrap_servers="localhost:9092")
        await producer.start()

        # These should not raise exceptions
        await producer.produce("test-topic", "key-1", {"data": "test"})
        await producer.produce_user_event("user_1", "view", "item_1")
        await producer.produce_purchase_event("user_1", ["item_1", "item_2"], 99.99)

        await producer.stop()

    @pytest.mark.asyncio
    async def test_produce_user_event(self):
        producer = KafkaEventProducer(bootstrap_servers="localhost:9092")
        await producer.start()
        await producer.produce_user_event(
            user_id="user_1",
            event_type="click",
            item_id="prod_001",
            metadata={"surface": "homepage"},
        )
        await producer.stop()


class TestKafkaEventConsumer:
    @pytest.mark.asyncio
    async def test_consumer_initialization_offline(self):
        consumer = KafkaEventConsumer(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
        )
        # Should not crash — runs in offline mode
        await consumer.start()
        assert consumer._consumer is None  # Offline mode
        await consumer.stop()

    @pytest.mark.asyncio
    async def test_register_handler(self):
        consumer = KafkaEventConsumer(bootstrap_servers="localhost:9092")

        async def handler(msg: KafkaMessage):
            pass  # No-op handler

        consumer.register_handler("user-events", handler)
        assert "user-events" in consumer._handlers


# ── Feature Store Tests ──────────────────────────────────────────


class TestRedisFeatureStore:
    @pytest.mark.asyncio
    async def test_connect_offline(self):
        """Should handle connection failure gracefully."""
        store = RedisFeatureStore(host="nonexistent", port=6379)
        await store.connect()
        assert store._client is None  # Offline mode

    @pytest.mark.asyncio
    async def test_get_offline(self):
        """Getting values in offline mode should return None."""
        store = RedisFeatureStore(host="nonexistent", port=6379)
        await store.connect()

        result = await store.get("test-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_offline(self):
        """Setting values in offline mode should not crash."""
        store = RedisFeatureStore(host="nonexistent", port=6379)
        await store.connect()

        # Should not crash
        await store.set("test-key", {"data": "test"})
        await store.set_user_embedding("user_1", [0.1, 0.2, 0.3])
        await store.set_item_embedding("item_1", [0.4, 0.5, 0.6])
        await store.set_user_profile("user_1", {"preferences": ["electronics"]})
        await store.set_item_features("item_1", {"category": "electronics", "price": 99.99})
        await store.set_recommendation_cache("user_1", "homepage", [{"item_id": "prod_001"}])

    @pytest.mark.asyncio
    async def test_invalidate_offline(self):
        store = RedisFeatureStore(host="nonexistent", port=6379)
        await store.connect()
        await store.invalidate("user:*")
        await store.close()

    @pytest.mark.asyncio
    async def test_round_trip_offline(self):
        """Offline mode getters should return None for unavailable keys."""
        store = RedisFeatureStore(host="nonexistent", port=6379)
        await store.connect()

        assert await store.get_user_embedding("user_1") is None
        assert await store.get_item_embedding("item_1") is None
        assert await store.get_user_profile("user_1") is None
        assert await store.get_item_features("item_1") is None
        assert await store.get_recommendation_cache("user_1", "homepage") is None


class TestFeastFeatureStore:
    def test_connect_offline(self):
        """Should handle missing Feast repo gracefully."""
        store = FeastFeatureStore(serving_url="localhost:6566", repo_path="./nonexistent")
        store.connect()
        assert store._client is None  # Offline mode

    def test_get_online_features_offline(self):
        store = FeastFeatureStore(repo_path="./nonexistent")
        store.connect()

        result = store.get_online_features(
            feature_refs=["user_features:embedding"],
            entity_rows=[{"user_id": "user_1"}],
        )
        assert result == {}

    def test_get_user_features_offline(self):
        store = FeastFeatureStore(repo_path="./nonexistent")
        store.connect()
        result = store.get_user_features(["user_1", "user_2"])
        assert result == {}

    def test_get_item_features_offline(self):
        store = FeastFeatureStore(repo_path="./nonexistent")
        store.connect()
        result = store.get_item_features(["item_1", "item_2"])
        assert result == {}


# ── Spark Training Pipeline Tests ────────────────────────────────


class TestSparkTrainingPipeline:
    def test_initialization(self):
        config = TrainingConfig(
            input_path="/test/input",
            output_path="/test/output",
            num_partitions=10,
        )
        pipeline = SparkTrainingPipeline(config)
        assert pipeline.config.input_path == "/test/input"
        assert pipeline._spark is None

    def test_start_session_offline(self):
        """Starting session without PySpark should log warning, not crash."""
        config = TrainingConfig()
        pipeline = SparkTrainingPipeline(config)
        pipeline.start_session()
        assert pipeline._spark is None  # PySpark likely not available

    def test_load_interactions_offline(self):
        config = TrainingConfig()
        pipeline = SparkTrainingPipeline(config)
        assert pipeline.load_interactions() is None

    def test_generate_negative_samples_offline(self):
        config = TrainingConfig()
        pipeline = SparkTrainingPipeline(config)
        assert pipeline.generate_negative_samples(None) is None

    def test_prepare_training_data_offline(self):
        config = TrainingConfig()
        pipeline = SparkTrainingPipeline(config)
        assert pipeline.prepare_training_data(None, None) is None

    def test_save_training_data_offline(self):
        config = TrainingConfig()
        pipeline = SparkTrainingPipeline(config)
        pipeline.save_training_data(None)  # Should not crash

    def test_config_defaults(self):
        config = TrainingConfig()
        assert config.batch_size == 4096
        assert config.num_epochs == 10
        assert config.negative_samples == 100
        assert config.train_fraction == 0.8
