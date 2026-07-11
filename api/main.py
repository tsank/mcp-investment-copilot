"""
api/main.py

FastAPI application entry point for the Portfolio Copilot.

Registers:
    - /api/v1/analyse  POST  — portfolio analysis endpoint
    - /health          GET   — health check
    - /                GET   — API info

Run locally:
    uvicorn api.main:app --reload --port 8080

Run in production (AWS ECS):
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1
    (single worker — LangGraph graph is compiled once at import time)
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from api.routes.aws import router as aws_router

from api.routes.analyse import router as analyse_router

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("api")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Portfolio Copilot",
    description=(
        "Agentic portfolio analysis system using LangGraph orchestration "
        "and 5 MCP servers: Market Data, Risk Engine, Portfolio Optimiser, "
        "Scenario Simulation, and Compliance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow Gradio UI (running on a different port locally) to call the API.
# Tightened in production to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(analyse_router, prefix="/api/v1")
app.include_router(aws_router, prefix="/api/v1/aws")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe for AWS ECS and load balancers."""
    return {"status": "ok", "service": "Portfolio Copilot"}


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["ops"])
async def root() -> dict:
    """API info and available endpoints."""
    return {
        "service":  "Portfolio Copilot",
        "version":  "1.0.0",
        "endpoints": {
            "analyse": "POST /api/v1/analyse",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        },
    }


# ── Lambda entry point ───────────────────────────────────────────────────────
# Mangum adapts FastAPI's ASGI interface to API Gateway's Lambda event
# format. `app` is still used directly for local dev (uvicorn), unaffected.
# `handler` is what the actual AWS Lambda function points to in production.
handler = Mangum(app)