#!/usr/bin/env python3
"""
Recommend Engine — Demo Server Entry Point

Starts the FastAPI recommendation server with mock data.
No external infrastructure required — works out of the box.

Usage:
    python run.py

Then test with:
    curl http://localhost:8000/v1/health
    curl -X POST http://localhost:8000/v1/recommend -H "Content-Type: application/json" \
      -d '{"user": {"user_id": "user_0001"}, "top_k": 5, "include_explanations": true}'
"""

import sys
from pathlib import Path

# Ensure src/ is importable (handles relative imports in api.py like `from .. import __version__`)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Welcome Message ─────────────────────────────────────────────


WELCOME_ART = r"""
╔══════════════════════════════════════════════════════════════╗
║              Recommend Engine — Demo Server                 ║
║  AI-Powered Product Recommendation with Preference Volatility ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_welcome():
    """Print a welcome message with available endpoints."""
    print(WELCOME_ART)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Mock data:    200 products × 50 users × 15 categories")
    print(f"  Innovations:  Preference Volatility, Anti-Echo-Chamber,")
    print(f"                Regret Minimizer, Life-Stage Diffusion")
    print()
    print("  ── Available Endpoints ──────────────────────────────")
    print("  Health:      GET  http://localhost:8000/v1/health")
    print("  Recommend:   POST http://localhost:8000/v1/recommend")
    print("  Batch Rec:   POST http://localhost:8000/v1/recommend/batch")
    print("  Explain:     POST http://localhost:8000/v1/explain")
    print("  Events:      POST http://localhost:8000/v1/events")
    print("  Metrics:     GET  http://localhost:8000/v1/metrics")
    print()
    print("  ── Quick Test ───────────────────────────────────────")
    print("  curl -X POST http://localhost:8000/v1/recommend \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"user": {"user_id": "user_0001",')
    print('      "time_bucket": 14, "day_bucket": 3,')
    print('      "is_payday": false},')
    print('      "top_k": 5, "include_explanations": true}\'')
    print()
    print("  ── Try Payday Context ───────────────────────────────")
    print("  curl -X POST http://localhost:8000/v1/recommend \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"user": {"user_id": "user_0001",')
    print('      "time_bucket": 14, "day_bucket": 3,')
    print('      "is_payday": true},')
    print('      "top_k": 5, "include_explanations": true}\'')
    print()
    print("  Starting server at http://localhost:8000 ...")
    print()


# ── Main ───────────────────────────────────────────────────────


def main():
    print_welcome()

    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
