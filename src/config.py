"""Centralized configuration for the Recommend Engine system."""

from __future__ import annotations

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ServingMode(str, Enum):
    """Serving mode for recommendation pipeline."""

    REAL_TIME = "real_time"
    SHADOW = "shadow"
    BATCH = "batch"


class RecConfig(BaseSettings):
    """Master configuration for the recommendation engine.

    All config is loaded from environment variables with sensible defaults.
    In production, these are set via Kubernetes ConfigMap/Secrets.
    """

    # ── System ──────────────────────────────────────────────────────
    environment: Environment = Environment.DEVELOPMENT
    service_name: str = "recommend-engine"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # ── Scale ────────────────────────────────────────────────────────
    num_users: int = int(os.getenv("NUM_USERS", "400_000_000"))
    num_products: int = int(os.getenv("NUM_PRODUCTS", "350_000_000"))
    num_categories: int = 50_000
    num_sellers: int = 5_000_000

    # ── Latency Budget (milliseconds) ────────────────────────────────
    latency_budget_ms: float = 10.0
    retrieval_budget_ms: float = 2.0
    ranker_budget_ms: float = 3.0
    reranker_budget_ms: float = 2.0
    explainability_budget_ms: float = 1.0
    overhead_budget_ms: float = 2.0

    # ── Serving ──────────────────────────────────────────────────────
    serving_mode: ServingMode = ServingMode.REAL_TIME
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 8
    max_batch_size: int = 256
    request_timeout_s: float = 0.5

    # ── Top-K Retrieval ──────────────────────────────────────────────
    default_top_k: int = 50
    max_top_k: int = 200
    min_relevance_score: float = 0.1
    recency_weight: float = 0.2
    inventory_weight: float = 0.1
    margin_weight: float = 0.05

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_dim: int = 256
    user_embedding_dim: int = 256
    item_embedding_dim: int = 256
    context_embedding_dim: int = 64
    mood_embedding_dim: int = 32

    # ── Two-Tower Model ──────────────────────────────────────────────
    two_tower_hidden_dims: List[int] = [512, 256, 128]
    two_tower_dropout: float = 0.2
    two_tower_lr: float = 1e-3
    two_tower_batch_size: int = 4096
    two_tower_negative_samples: int = 100

    # ── GraphSAGE ────────────────────────────────────────────────────
    graphsage_hidden_dims: List[int] = [128, 64]
    graphsage_num_layers: int = 2
    graphsage_dropout: float = 0.1
    graphsage_aggregator: str = "mean"  # mean, lstm, pooling

    # ── Time-Aware LSTM ──────────────────────────────────────────────
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_seq_length: int = 50
    lstm_time_buckets: int = 24  # 24 hour buckets

    # ── FAISS ANN ────────────────────────────────────────────────────
    faiss_index_type: str = "IVF65536_HNSW32,PQ64"  # Multi-Index
    faiss_nprobe: int = 128
    faiss_nlist: int = 65536
    faiss_train_sample: int = 5_000_000
    faiss_index_path: str = "/data/index/faiss"
    faiss_rebuild_interval_hours: int = 24

    # ── Caching ──────────────────────────────────────────────────────
    cache_l1_capacity: int = 100_000  # Top 1% hottest users
    cache_l2_capacity: int = 10_000_000  # Warm users
    cache_bucket_count: int = 1000
    cache_ttl_s: int = 300  # 5 minutes
    cache_warm_ttl_s: int = 60

    # ── Feature Store ────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_feature_db: int = 0
    redis_session_db: int = 1
    redis_cache_db: int = 2
    redis_password: Optional[str] = None
    redis_max_connections: int = 50

    feast_serving_url: str = "localhost:6566"
    feast_offline_store: str = "file"
    feast_online_store: str = "redis"

    # ── Kafka ────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_num_partitions: int = 12
    kafka_replication_factor: int = 1
    kafka_consumer_group: str = "rec-engine"

    # Kafka Topics
    kafka_topic_events: str = "user-events"
    kafka_topic_purchases: str = "purchases"
    kafka_topic_inventory: str = "inventory-updates"
    kafka_topic_embeddings: str = "embedding-updates"
    kafka_topic_cache_invalidation: str = "cache-invalidation"
    kafka_topic_fraud_alerts: str = "fraud-alerts"

    # ── Spark ────────────────────────────────────────────────────────
    spark_master: str = "local[*]"
    spark_batch_interval_hours: int = 6
    spark_model_checkpoint_interval: int = 1000

    # ── MLflow ───────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommend-engine"
    mlflow_registered_model: str = "recommend-engine-prod"

    # ── Bandit (Cold Start) ──────────────────────────────────────────
    bandit_alpha: float = 1.0  # LinUCB exploration parameter
    bandit_batch_size: int = 128
    bandit_warmup_samples: int = 50

    # ── Anti-Echo-Chamber ────────────────────────────────────────────
    echo_chamber_injection_rate: float = 1.0 / 7.0  # Every 7th rec
    discovery_score_target: float = 0.15
    discovery_category_distance_threshold: float = 0.6

    # ── Adversarial Simulation ───────────────────────────────────────
    adv_sim_fake_click_rate: float = 0.01
    adv_sim_review_ring_size: int = 5
    adv_sim_gan_iters: int = 1000

    # ── Regret Minimizer ─────────────────────────────────────────────
    regret_return_threshold: float = 0.02  # < 2% return rate
    regret_window_days: int = 90
    regret_penalty_lambda: float = 0.3

    # ── Life-Stage Diffusion ─────────────────────────────────────────
    life_stage_similarity_threshold: float = 0.80
    life_stage_divergence_threshold: float = 0.25
    life_stage_fork_embedding_dim: int = 64

    # ── Fraud Detection ──────────────────────────────────────────────
    fraud_review_correlation_threshold: float = 0.9
    fraud_ip_range_overlap_threshold: float = 0.8
    fraud_auto_demote_threshold: float = 0.75

    # ── Demand Forecasting ───────────────────────────────────────────
    forecast_horizon_days: int = 30
    forecast_out_of_stock_threshold_days: int = 3

    # ── Prometheus ───────────────────────────────────────────────────
    prometheus_port: int = 8001
    prometheus_histogram_buckets: List[float] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]

    # ── Shadow Traffic ───────────────────────────────────────────────
    shadow_traffic_ratio: float = 0.05
    shadow_inference_endpoint: str = "http://localhost:8000/v1/recommend"

    # ── Consistent Hashing ───────────────────────────────────────────
    hash_virtual_nodes: int = 100
    hash_num_shards: int = 128

    # ── Bloom Filter ─────────────────────────────────────────────────
    bloom_filter_capacity: int = 1_000_000_000  # 1B items
    bloom_filter_false_positive_rate: float = 0.01

    class Config:
        env_prefix = "REC_"
        case_sensitive = False


# Global singleton config
config = RecConfig()
