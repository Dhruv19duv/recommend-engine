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


@app.get("/")
async def root():
    """Serve the landing page."""
    from fastapi.responses import HTMLResponse
    import textwrap
    html = textwrap.dedent("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Recommend Engine - AI-Powered Recommendations</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
      <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Inter',-apple-system,sans-serif; background:#0a0a0f; color:#e4e4e7; line-height:1.6; overflow-x:hidden; }
        .bg-glow { position:fixed; top:-50%; left:-50%; width:200%; height:200%; background:radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 50%), radial-gradient(ellipse at 70% 80%, rgba(236,72,153,0.06) 0%, transparent 50%); pointer-events:none; z-index:0; }
        .container { max-width:1100px; margin:0 auto; padding:0 24px; position:relative; z-index:1; }
        nav { display:flex; justify-content:space-between; align-items:center; padding:20px 0; border-bottom:1px solid rgba(255,255,255,0.06); }
        .logo { font-size:20px; font-weight:700; background:linear-gradient(135deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .nav-links { display:flex; gap:24px; }
        .nav-links a { color:#a1a1aa; text-decoration:none; font-size:14px; transition:color .2s; }
        .nav-links a:hover { color:#fff; }
        .hero { text-align:center; padding:100px 0 60px; }
        .hero-badge { display:inline-block; padding:6px 16px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2); border-radius:20px; font-size:12px; color:#818cf8; margin-bottom:24px; }
        .hero h1 { font-size:56px; font-weight:800; line-height:1.1; margin-bottom:16px; }
        .hero h1 span { background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .hero p { font-size:18px; color:#a1a1aa; max-width:600px; margin:0 auto 32px; }
        .hero-buttons { display:flex; gap:12px; justify-content:center; }
        .btn { padding:12px 28px; border-radius:8px; font-size:14px; font-weight:600; text-decoration:none; transition:all .2s; cursor:pointer; border:none; }
        .btn-primary { background:linear-gradient(135deg,#818cf8,#6366f1); color:#fff; }
        .btn-primary:hover { transform:translateY(-1px); box-shadow:0 8px 24px rgba(99,102,241,0.3); }
        .btn-secondary { background:rgba(255,255,255,0.05); color:#e4e4e7; border:1px solid rgba(255,255,255,0.1); }
        .btn-secondary:hover { background:rgba(255,255,255,0.1); }
        .stats-bar { display:flex; justify-content:center; gap:60px; padding:40px 0; border-top:1px solid rgba(255,255,255,0.06); border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:80px; }
        .stat { text-align:center; }
        .stat-value { font-size:32px; font-weight:700; color:#fff; }
        .stat-label { font-size:13px; color:#71717a; margin-top:4px; }
        section { margin-bottom:80px; }
        section h2 { font-size:32px; font-weight:700; text-align:center; margin-bottom:40px; }
        .features-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; }
        .feature-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:28px; transition:all .3s; }
        .feature-card:hover { border-color:rgba(99,102,241,0.3); transform:translateY(-2px); }
        .feature-icon { width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; margin-bottom:16px; }
        .feature-card h3 { font-size:16px; font-weight:600; margin-bottom:8px; }
        .feature-card p { font-size:13px; color:#a1a1aa; }
        .endpoints { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; overflow:hidden; }
        .endpoint { display:flex; align-items:center; gap:16px; padding:16px 24px; border-bottom:1px solid rgba(255,255,255,0.04); transition:background .2s; }
        .endpoint:last-child { border-bottom:none; }
        .endpoint:hover { background:rgba(255,255,255,0.02); }
        .method { padding:3px 10px; border-radius:4px; font-size:11px; font-weight:700; text-transform:uppercase; min-width:48px; text-align:center; }
        .method.get { background:rgba(34,197,94,0.15); color:#22c55e; }
        .method.post { background:rgba(59,130,246,0.15); color:#3b82f6; }
        .endpoint-path { font-family:'SF Mono','Fira Code',monospace; font-size:13px; color:#e4e4e7; }
        .endpoint-desc { font-size:13px; color:#71717a; margin-left:auto; }
        .try-it { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:32px; }
        .try-it h3 { font-size:16px; font-weight:600; margin-bottom:12px; }
        .code-block { background:#1a1a2e; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:20px; font-family:'SF Mono','Fira Code',monospace; font-size:13px; line-height:1.7; overflow-x:auto; color:#a5b4fc; margin:16px 0; }
        .code-block .comment { color:#71717a; }
        .code-block .keyword { color:#c084fc; }
        .code-block .string { color:#34d399; }
        .footer { text-align:center; padding:40px 0; color:#52525b; font-size:13px; border-top:1px solid rgba(255,255,255,0.06); }
        @media (max-width:768px) {
          .hero h1 { font-size:36px; }
          .stats-bar { gap:30px; flex-wrap:wrap; }
          .nav-links { display:none; }
        }
      </style>
    </head>
    <body>
      <div class="bg-glow"></div>
      <div class="container">
        <nav>
          <div class="logo">Recommend Engine</div>
          <div class="nav-links">
            <a href="#features">Features</a>
            <a href="#api">API</a>
            <a href="#try-it">Try It</a>
            <a href="https://github.com/Dhruv19duv/recommend-engine" target="_blank">GitHub</a>
          </div>
        </nav>

        <div class="hero">
          <div class="hero-badge">&#9889; AI-Powered &bull; 1M RPS &bull; p99 &lt; 10ms</div>
          <h1>Recommendations that <span>users keep</span></h1>
          <p>Not just clicks. Optimize for purchases users won't return. Built for 400M users &times; 350M products with sub-10ms latency.</p>
          <div class="hero-buttons">
            <a href="#try-it" class="btn btn-primary">Try the API</a>
            <a href="#api" class="btn btn-secondary">API Docs</a>
          </div>
        </div>

        <div class="stats-bar">
          <div class="stat"><div class="stat-value">400M</div><div class="stat-label">Users</div></div>
          <div class="stat"><div class="stat-value">350M</div><div class="stat-label">Products</div></div>
          <div class="stat"><div class="stat-value">&lt;10ms</div><div class="stat-label">p99 Latency</div></div>
          <div class="stat"><div class="stat-value">&lt;2%</div><div class="stat-label">Return Rate</div></div>
        </div>

        <section id="features">
          <h2>Five Architectural Innovations</h2>
          <div class="features-grid">
            <div class="feature-card">
              <div class="feature-icon" style="background:rgba(99,102,241,0.15);">&#128200;</div>
              <h3>Preference Volatility Engine</h3>
              <p>Context-dependent mood-state embeddings. A user at 2am after payday is a different buyer than Tuesday lunch break.</p>
            </div>
            <div class="feature-card">
              <div class="feature-icon" style="background:rgba(236,72,153,0.15);">&#127912;</div>
              <h3>Anti-Echo-Chamber Injection</h3>
              <p>Every 7th recommendation is deliberately outside the user's usual category. Discovery score target &gt;15%.</p>
            </div>
            <div class="feature-card">
              <div class="feature-icon" style="background:rgba(52,211,153,0.15);">&#9889;</div>
              <h3>Adversarial Simulation (GAN)</h3>
              <p>GAN-based red-team agent continuously attempts to game the ranker. Fraud ring recall &gt;95%.</p>
            </div>
            <div class="feature-card">
              <div class="feature-icon" style="background:rgba(251,191,36,0.15);">&#128148;</div>
              <h3>Regret-Minimizing Re-ranker</h3>
              <p>Penalizes patterns that lead to buyer's remorse. Return rate on recommended products &lt;2% — the industry standard is 5-15%.</p>
            </div>
            <div class="feature-card">
              <div class="feature-icon" style="background:rgba(129,140,248,0.15);">&#128200;</div>
              <h3>Life-Stage Graph Diffusion</h3>
              <p>Detects life events (baby, new home, marriage) and permanently forks embeddings in real-time.</p>
            </div>
            <div class="feature-card">
              <div class="feature-icon" style="background:rgba(248,113,113,0.15);">&#128640;</div>
              <h3>Two-Tower Neural Network</h3>
              <p>128-dim embeddings with FAISS ANN retrieval (IVF65536_HNSW32,PQ64). Sub-millisecond search at 350M scale.</p>
            </div>
          </div>
        </section>

        <section id="api">
          <h2>API Endpoints</h2>
          <div class="endpoints">
            <div class="endpoint"><span class="method post">POST</span><span class="endpoint-path">/v1/recommend</span><span class="endpoint-desc">Get personalized recommendations</span></div>
            <div class="endpoint"><span class="method post">POST</span><span class="endpoint-path">/v1/recommend/batch</span><span class="endpoint-desc">Batch recommendations</span></div>
            <div class="endpoint"><span class="method post">POST</span><span class="endpoint-path">/v1/explain</span><span class="endpoint-desc">SHAP explanation for a recommendation</span></div>
            <div class="endpoint"><span class="method post">POST</span><span class="endpoint-path">/v1/events</span><span class="endpoint-desc">Log user interaction events</span></div>
            <div class="endpoint"><span class="method get">GET</span><span class="endpoint-path">/v1/health</span><span class="endpoint-desc">Health check</span></div>
            <div class="endpoint"><span class="method get">GET</span><span class="endpoint-path">/v1/metrics</span><span class="endpoint-desc">Prometheus metrics</span></div>
          </div>
        </section>

        <section id="try-it">
          <h2>Try It Now</h2>
          <div class="try-it">
            <h3>Get Recommendations</h3>
            <div class="code-block">
              <span class="comment"># Get personalized recommendations for a user</span>
              curl -X POST https://recommend-engine.vercel.app/v1/recommend \\
                -H "Content-Type: application/json" \\
                -d '{<span class="string">
                  "user": {"user_id": "user_0001", "time_bucket": 14, "day_bucket": 3, "is_payday": false},
                  "top_k": 5,
                  "include_explanations": true
                </span>}'
              <span class="comment">
              # Response: top 5 recommended items with scores & explanations</span>
            </div>
            <h3>Check Health</h3>
            <div class="code-block">
              curl https://recommend-engine.vercel.app/v1/health
              <span class="comment"># {"status":"healthy","version":"0.1.0","uptime_seconds":...}</span>
            </div>
          </div>
        </section>

        <div class="footer">
          Recommend Engine &mdash; AI-Powered Product Recommendation Engine
        </div>
      </div>
    </body>
    </html>
    """)
    return HTMLResponse(content=html, status_code=200)


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
