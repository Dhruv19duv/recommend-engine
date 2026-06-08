"""Vercel serverless entry point for the Recommend Engine API."""
import sys
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.api import app  # noqa: E402
