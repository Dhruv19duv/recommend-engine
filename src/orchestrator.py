"""Main recommendation orchestrator — ties all components together.

Coordinates the full recommendation pipeline from request to response,
integrating the DSA layer, AI models, five innovations, and serving layer.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the recommendation orchestrator."""

    top_k_default: int = 50
    max_candidates: int = 500
    enable_cache: bool = True
    enable_bandit: bool = True
    enable_two_tower: bool = True
    enable_graphsage: bool = True
    enable_volatility: bool = True
    enable_elasticity: bool = True
    enable_regret_minimizer: bool = True
    enable_anti_echo: bool = True
    enable_inventory_filter: bool = True
    enable_fraud_detection: bool = True
    enable_explainability: bool = True
    enable_life_stage: bool = True
    shadow_mode: bool = False


class RecommendOrchestrator:
    """The main orchestrator that coordinates the full recommendation pipeline.

    Pipeline steps:
    1. Parse request and extract user context
    2. Check cache (L1 → L2)
    3. If cold-start user → bandit exploration
    4. Retrieve candidates (FAISS ANN + collaborative filtering)
    5. Score candidates (two-tower + preference volatility)
    6. Re-rank candidates (price elasticity + regret minimizer)
    7. Inject anti-echo-chamber discoveries
    8. Filter out-of-stock items (demand forecast)
    9. Apply fraud detection penalties
    10. Generate SHAP explanations
    11. Deduplicate via bloom filter
    12. Cache results and return
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        cache=None,
        bandit=None,
        faiss_index=None,
        two_tower=None,
        graphsage=None,
        volatility_engine=None,
        elasticity_ranker=None,
        regret_ranker=None,
        echo_injector=None,
        inventory_filter=None,
        fraud_detector=None,
        explainer=None,
        life_stage_diffuser=None,
        bloom_filter=None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.bandit = bandit
        self.faiss_index = faiss_index
        self.two_tower = two_tower
        self.graphsage = graphsage
        self.volatility_engine = volatility_engine
        self.elasticity_ranker = elasticity_ranker
        self.regret_ranker = regret_ranker
        self.echo_injector = echo_injector
        self.inventory_filter = inventory_filter
        self.fraud_detector = fraud_detector
        self.explainer = explainer
        self.life_stage_diffuser = life_stage_diffuser
        self.bloom_filter = bloom_filter

        self._latencies: Dict[str, List[float]] = {}

    async def recommend(
        self,
        user_id: str,
        surface: str = "homepage",
        top_k: int = 50,
        context: Optional[Dict[str, Any]] = None,
        candidate_ids: Optional[List[str]] = None,
        include_explanations: bool = False,
        shadow_mode: bool = False,
    ) -> Dict[str, Any]:
        """Run the full recommendation pipeline.

        Args:
            user_id: The user to recommend for
            surface: Recommendation surface (homepage, pdp, cart, post_purchase)
            top_k: Number of recommendations to return
            context: User context (time, device, session)
            candidate_ids: Pre-filtered candidate item IDs
            include_explanations: Whether to generate explanations
            shadow_mode: Run in shadow (no cache updates) mode

        Returns:
            {
                "user_id": str,
                "request_id": str,
                "results": [{"item_id": str, "score": float, "explanation": Optional[str]}, ...],
                "latency_ms": float,
                "surface": str,
                "pipeline_breakdown": {step_name: latency_ms},
            }
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        pipeline_times: Dict[str, float] = {}

        # Step 1: Context extraction
        t0 = time.perf_counter()
        context = context or {}
        user_context = {
            "hour_of_day": context.get("hour_of_day", 12),
            "day_of_week": context.get("day_of_week", 0),
            "is_weekend": context.get("is_weekend", False),
            "is_payday": context.get("is_payday", False),
            "device_type": context.get("device_type", "web"),
            "is_new_user": context.get("is_new_user", False),
        }
        pipeline_times["context_extraction"] = (time.perf_counter() - t0) * 1000

        # Step 2: Cache check
        t0 = time.perf_counter()
        if self.config.enable_cache and self.cache and not shadow_mode:
            cached = self.cache.get(f"recs:{user_id}:{surface}")
            if cached is not None:
                logger.debug(f"Cache HIT for user {user_id} on {surface}")
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "user_id": user_id,
                    "request_id": request_id,
                    "results": cached,
                    "latency_ms": elapsed_ms,
                    "surface": surface,
                    "pipeline_breakdown": {"cache_hit": pipeline_times.get("context_extraction", 0) + (time.perf_counter() - t0) * 1000},
                    "cached": True,
                }
        pipeline_times["cache_check"] = (time.perf_counter() - t0) * 1000

        # Step 3: Candidate retrieval
        t0 = time.perf_counter()
        candidates: List[Tuple[str, float]] = candidate_ids or []
        if not candidates:
            # In production: FAISS ANN + collaborative filtering + GraphSAGE
            candidates = self._retrieve_candidates(user_id, user_context, top_k * 10)
        pipeline_times["retrieval"] = (time.perf_counter() - t0) * 1000

        # Step 4: Cold-start handling
        t0 = time.perf_counter()
        if self.config.enable_bandit and self.bandit and user_context.get("is_new_user"):
            bandit_results = self.bandit.recommend(
                user_id=user_id,
                user_metadata={k: float(v) for k, v in user_context.items() if isinstance(v, (int, float))},
                available_categories=[c[0] for c in candidates[:100]],
                top_k=top_k,
            )
            if bandit_results:
                candidates = bandit_results
        pipeline_times["bandit"] = (time.perf_counter() - t0) * 1000

        # Step 5: Scoring (two-tower + preference volatility)
        t0 = time.perf_counter()
        scored = self._score_candidates(user_id, candidates, user_context)
        pipeline_times["scoring"] = (time.perf_counter() - t0) * 1000

        # Step 6: Price elasticity re-ranking
        t0 = time.perf_counter()
        if self.config.enable_elasticity and self.elasticity_ranker:
            user_features = {}  # In production: get from feature store
            item_prices = {}  # In production: get from feature store
            item_ratings = {}  # In production: get from feature store
            scored = self.elasticity_ranker.rerank(
                user_id, scored, user_features, item_prices, item_ratings,
            )
        pipeline_times["elasticity"] = (time.perf_counter() - t0) * 1000

        # Step 7: Regret minimization
        t0 = time.perf_counter()
        if self.config.enable_regret_minimizer and self.regret_ranker:
            candidates_with_meta = [
                (item_id, score, "default", 0.0)
                for item_id, score in scored
            ]
            scored = self.regret_ranker.rerank_by_regret(user_id, candidates_with_meta)
        pipeline_times["regret"] = (time.perf_counter() - t0) * 1000

        # Step 8: Anti-echo-chamber injection
        t0 = time.perf_counter()
        if self.config.enable_anti_echo and self.echo_injector:
            discovery_items = []  # In production: get from discovery candidate pool
            category_map = {}  # In production: get item_id -> category_id
            injected = self.echo_injector.inject_discovery(
                user_id, scored, discovery_items, category_map,
            )
            scored = [(item_id, score) for item_id, score, _ in injected]
        pipeline_times["anti_echo"] = (time.perf_counter() - t0) * 1000

        # Step 9: Inventory filter
        t0 = time.perf_counter()
        if self.config.enable_inventory_filter and self.inventory_filter:
            item_inventory = {}  # In production: get from inventory service
            item_forecast = {}  # In production: get from forecast model
            scored, _ = self.inventory_filter.filter_out_of_stock(
                scored, item_inventory, item_forecast,
            )
        pipeline_times["inventory"] = (time.perf_counter() - t0) * 1000

        # Step 10: Fraud demotion
        t0 = time.perf_counter()
        if self.config.enable_fraud_detection and self.fraud_detector:
            seller_map = {}  # In production: get item_id -> seller_id
            fraud_indicators = {}  # In production: get from fraud detector
            scored = self.fraud_detector.demote_fraudulent_sellers(
                scored, seller_map, fraud_indicators,
            )
        pipeline_times["fraud"] = (time.perf_counter() - t0) * 1000

        # Step 11: Bloom filter dedup
        t0 = time.perf_counter()
        if self.bloom_filter:
            deduped = []
            for item_id, score in scored:
                if item_id not in self.bloom_filter:
                    deduped.append((item_id, score))
                    self.bloom_filter.add(item_id)
            scored = deduped
        pipeline_times["dedup"] = (time.perf_counter() - t0) * 1000

        # Step 12: Generate explanations
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for item_id, score in scored[:top_k]:
            result = {"item_id": item_id, "score": score}
            if include_explanations and self.explainer:
                # In production: generate SHAP explanation
                result["explanation"] = f"Recommended based on your interests"
            results.append(result)
        pipeline_times["explainability"] = (time.perf_counter() - t0) * 1000

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Cache results
        if self.config.enable_cache and self.cache and not shadow_mode:
            self.cache.put(f"recs:{user_id}:{surface}", results)

        self._latencies.setdefault("total", []).append(elapsed_ms)

        return {
            "user_id": user_id,
            "request_id": request_id,
            "results": results,
            "latency_ms": elapsed_ms,
            "surface": surface,
            "pipeline_breakdown": pipeline_times,
        }

    def _retrieve_candidates(
        self,
        user_id: str,
        user_context: Dict[str, Any],
        num_candidates: int,
    ) -> List[Tuple[str, float]]:
        """Retrieve candidate items from FAISS + collaborative filtering.

        In production, this would:
        1. Get user embedding from two-tower model
        2. Search FAISS index for nearest items
        3. Boost with collaborative filtering signals
        4. Enrich with GraphSAGE graph propagation
        """
        # Placeholder: return empty candidates
        return []

    def _score_candidates(
        self,
        user_id: str,
        candidates: List[Tuple[str, float]],
        user_context: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Score candidates using two-tower model with preference volatility.

        In production, this would:
        1. Encode user context via two-tower user tower
        2. Encode candidate items via two-tower item tower
        3. Compute dot-product similarity
        4. Apply mood-state bias from volatility engine
        """
        return candidates  # Placeholder

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies.get("total"):
            return 0.0
        return sum(self._latencies["total"]) / len(self._latencies["total"])

    @property
    def p99_latency_ms(self) -> float:
        if not self._latencies.get("total"):
            return 0.0
        sorted_latencies = sorted(self._latencies["total"])
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]
