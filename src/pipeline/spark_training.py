"""Spark batch training pipeline for data processing and model training.

Handles:
- 6-month interaction log processing
- Negative sampling for two-tower training
- Feature engineering at scale
- Model checkpointing for MLflow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a Spark training job."""

    input_path: str = "/data/interactions"
    output_path: str = "/data/training"
    checkpoint_path: str = "/data/checkpoints"
    num_partitions: int = 200
    batch_size: int = 4096
    num_epochs: int = 10
    negative_samples: int = 100
    train_fraction: float = 0.8
    seed: int = 42


class SparkTrainingPipeline:
    """Spark-based batch training pipeline.

    Processes interaction logs, generates training examples,
    and prepares data for PyTorch model training.
    """

    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        self._spark = None

    def start_session(self) -> None:
        """Start a Spark session."""
        try:
            from pyspark.sql import SparkSession

            self._spark = (
                SparkSession.builder
                .appName("RecommendEngine-Training")
                .master("local[*]")
                .config("spark.sql.shuffle.partitions", self.config.num_partitions)
                .config("spark.memory.offHeap.enabled", "true")
                .config("spark.memory.offHeap.size", "8g")
                .getOrCreate()
            )
            logger.info("Spark session started")
        except ImportError:
            logger.warning("PySpark not available")

    def load_interactions(
        self,
        path: Optional[str] = None,
        format: str = "parquet",
    ) -> Any:
        """Load interaction logs from storage."""
        if self._spark is None:
            return None
        path = path or self.config.input_path
        return self._spark.read.format(format).load(path)

    def generate_negative_samples(
        self,
        df: Any,
        num_negatives: int = 100,
    ) -> Any:
        """Generate negative samples for two-tower training.

        For each user-item positive interaction, samples N negative items
        that the user has not interacted with.
        """
        if self._spark is None or df is None:
            return None

        # Get all items and users
        items = df.select("item_id").distinct()
        users = df.select("user_id").distinct()

        # Cross join for candidate negatives (sampled)
        candidates = users.crossJoin(items)

        # Remove positive interactions
        positives = df.select("user_id", "item_id", "rating")
        negatives = candidates.join(
            positives,
            on=["user_id", "item_id"],
            how="left_anti",
        )

        # Sample N negatives per user
        from pyspark.sql import functions as F

        negatives = (
            negatives.withColumn("rating", F.lit(0.0))
            .withColumn("rand", F.rand(self.config.seed))
        )

        # Stratified sampling: N negatives per user
        window = Window.partitionBy("user_id").orderBy("rand")
        negatives = (
            negatives.withColumn("rank", F.row_number().over(window))
            .filter(F.col("rank") <= num_negatives)
            .drop("rand", "rank")
        )

        return negatives

    def prepare_training_data(
        self,
        df: Any,
        negative_df: Any,
    ) -> Any:
        """Combine positives and negatives, shuffle, and prepare for training."""
        if self._spark is None or df is None:
            return None

        # Combine positive and negative samples
        training_df = df.select(
            "user_id", "item_id", "rating",
            "category_id", "timestamp"
        ).union(negative_df)

        # Shuffle and split
        training_df = training_df.orderBy("rand")
        return training_df

    def save_training_data(
        self,
        df: Any,
        path: Optional[str] = None,
    ) -> None:
        """Save processed training data."""
        if df is None:
            return
        path = path or self.config.output_path
        df.write.mode("overwrite").parquet(path)
        logger.info(f"Training data saved to {path}")

    def stop_session(self) -> None:
        """Stop the Spark session."""
        if self._spark:
            self._spark.stop()
            logger.info("Spark session stopped")


from pyspark.sql import Window  # type: ignore[import-untyped]
