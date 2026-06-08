"""FastAPI serving layer for the recommendation engine.

Endpoints:
- POST /v1/recommend — Get personalized recommendations
- POST /v1/recommend/batch — Batch recommendations
- POST /v1/explain — Get explanation for a recommendation
- POST /v1/events — Log user interaction events
- GET  /v1/health — Health check
- GET  /v1/metrics — Prometheus metrics

Target: p99 latency under 10ms at 1M RPS.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .. import __version__
from .mock_data import MockDataStore

logger = logging.getLogger(__name__)


# ── Request/Response Models ───────────────────────────────────────


class UserContext(BaseModel):
    """User context for recommendation requests."""

    user_id: str
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    is_new_user: bool = False
    time_bucket: Optional[int] = None  # Hour of day (0-23)
    day_bucket: Optional[int] = None  # Day of week (0-6)
    is_payday: bool = False
    metadata: Dict[str, float] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    """Request body for the recommend endpoint."""

    user: UserContext
    surface: str = "homepage"  # homepage, product_detail, cart, post_purchase
    top_k: int = 50
    max_top_k: int = Field(default=200, le=200)
    candidate_item_ids: Optional[List[str]] = None
    include_explanations: bool = False
    shadow_traffic: bool = False


class BatchRecommendRequest(BaseModel):
    """Request body for batch recommendations."""

    requests: List[RecommendRequest]


class RecommendResult(BaseModel):
    """A single recommendation result."""

    item_id: str
    score: float
    is_discovery: bool = False
    explanation: Optional[str] = None


class RecommendResponse(BaseModel):
    """Response from the recommend endpoint."""

    user_id: str
    request_id: str
    results: List[RecommendResult]
    latency_ms: float
    surface: str
    discovery_score: Optional[float] = None


class EventRequest(BaseModel):
    """User interaction event."""

    user_id: str
    event_type: str  # view, click, purchase, add_to_cart, return
    item_id: str
    category_id: Optional[str] = None
    price: Optional[float] = None
    timestamp: Optional[float] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchEventRequest(BaseModel):
    """Batch of user interaction events."""

    events: List[EventRequest]


class ExplainRequest(BaseModel):
    """Request for recommendation explanation."""

    user_id: str
    item_id: str
    features: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float


# ── Recommendation Engine ─────────────────────────────────────────


class RecommendEngine:
    """The core recommendation engine orchestrator.

    Coordinates all subsystems:
    - DSA layer (FAISS, cache, bloom filter)
    - AI models (two-tower, GraphSAGE, LSTM)
    - Innovations (preference volatility, anti-echo, etc.)
    - Serving (bandit for cold-start, explainability)
    """

    def __init__(self) -> None:
        self._initialized = False
        self._start_time = time.time()
        self._mock_store: Optional[MockDataStore] = None

    async def initialize(self) -> None:
        """Initialize the recommendation engine with mock data for the demo."""
        if self._initialized:
            return

        logger.info("Initializing recommendation engine with mock data...")
        self._mock_store = MockDataStore(num_products=200, num_users=50)
        logger.info(
            f"Mock data loaded: {len(self._mock_store.products)} products, "
            f"{len(self._mock_store.users)} users, "
            f"{len(self._mock_store.categories)} categories"
        )

        self._initialized = True
        logger.info("Recommendation engine initialized (DEMO MODE)")

    async def recommend(
        self,
        request: RecommendRequest,
    ) -> RecommendResponse:
        """Generate personalized recommendations using mock data."""
        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        if self._mock_store is None:
            return RecommendResponse(
                user_id=request.user.user_id,
                request_id=request_id,
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                surface=request.surface,
            )

        user_id = request.user.user_id

        # Build context for volatility engine
        context = {
            "hour_of_day": request.user.time_bucket or 12,
            "day_of_week": request.user.day_bucket or 0,
            "is_payday": request.user.is_payday,
        }

        # Detect emotional context
        if request.user.is_payday:
            emotional_context = "post_payday"
        elif (request.user.time_bucket or 12) < 6 or (request.user.time_bucket or 12) >= 23:
            emotional_context = "late_night"
        elif (request.user.day_bucket or 0) >= 5 and 10 <= (request.user.time_bucket or 12) <= 20:
            emotional_context = "weekend_leisure"
        elif 11 <= (request.user.time_bucket or 12) <= 13 and (request.user.day_bucket or 0) < 5:
            emotional_context = "workday_lunch"
        else:
            emotional_context = "default"

        context["emotional_context"] = emotional_context

        # Get recommendations from mock store
        mock_results = self._mock_store.get_recommendations(
            user_id=user_id,
            top_k=request.top_k,
            surface=request.surface,
            context=context,
        )

        # Format results
        results = []
        for item_id, score, meta in mock_results:
            explanation = None
            if request.include_explanations:
                explanation = self._mock_store.get_explanation(user_id, item_id)

            results.append(RecommendResult(
                item_id=item_id,
                score=score,
                is_discovery=meta.get("is_discovery", False),
                explanation=explanation,
            ))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RecommendResponse(
            user_id=user_id,
            request_id=request_id,
            results=results,
            latency_ms=elapsed_ms,
            surface=request.surface,
            discovery_score=sum(r.is_discovery for r in results) / max(len(results), 1),
        )

    async def explain(
        self,
        request: ExplainRequest,
    ) -> Dict[str, Any]:
        """Generate explanation for a recommendation using mock data."""
        if self._mock_store is None:
            return {
                "user_id": request.user_id,
                "item_id": request.item_id,
                "reason": "Recommended based on your interests",
                "features": [],
            }

        reason = self._mock_store.get_explanation(request.user_id, request.item_id)
        return {
            "user_id": request.user_id,
            "item_id": request.item_id,
            "reason": reason,
            "features": [
                {"name": "category_match", "value": 0.6},
                {"name": "price_compatibility", "value": 0.3},
                {"name": "rating_quality", "value": 0.1},
            ],
        }

    async def log_event(self, event: EventRequest) -> None:
        """Log a user interaction event."""
        logger.info(f"Event: {event.event_type} user={event.user_id} item={event.item_id}")

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time


# ── FastAPI Application ──────────────────────────────────────────


engine = RecommendEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Recommend Engine API...")
    await engine.initialize()
    yield
    logger.info("Shutting down Recommend Engine API...")


app = FastAPI(
    title="Recommend Engine API",
    description="AI-Powered Product Recommendation Engine",
    version=__version__,
    lifespan=lifespan,
)


@app.post("/v1/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Get personalized recommendations."""
    return await engine.recommend(request)


@app.post("/v1/recommend/batch", response_model=List[RecommendResponse])
async def recommend_batch(request: BatchRecommendRequest) -> List[RecommendResponse]:
    """Batch recommendation endpoint."""
    results = []
    for req in request.requests:
        result = await engine.recommend(req)
        results.append(result)
    return results


@app.post("/v1/explain")
async def explain(request: ExplainRequest) -> Dict[str, Any]:
    """Get explanation for a recommendation."""
    return await engine.explain(request)


@app.post("/v1/events")
async def log_events(request: EventRequest) -> Dict[str, str]:
    """Log a user interaction event."""
    await engine.log_event(request)
    return {"status": "ok"}


@app.post("/v1/events/batch")
async def log_events_batch(request: BatchEventRequest) -> Dict[str, str]:
    """Batch log user interaction events."""
    for event in request.events:
        await engine.log_event(event)
    return {"status": "ok"}


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if engine._initialized else "starting",
        version=__version__,
        uptime_seconds=engine.uptime,
    )


@app.get("/v1/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, REGISTRY
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain",
    )


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """Add X-Latency-Ms header to all responses."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
    return response


if __name__ == "__main__":
    import uvicorn
    top_k_default = 10
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )
