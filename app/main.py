"""
app/main.py
-----------
FastAPI application entrypoint for DocuSense Decoupled Backend Service.
Designed for 512MB RAM free-tier hosting on Render with single-worker Uvicorn.
Features:
  • CORS middleware for cross-origin Streamlit Cloud communication
  • Lifespan startup check for Qdrant Cloud collection initialization
  • OpenAPI /docs documentation for recruiters and peers
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.health import router as health_router
from app.api.query import router as query_router
from app.api.sessions import router as sessions_router
from app.api.upload import router as upload_router
from app.config import config
from app.services.vector_store import init_qdrant_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes vector store collections and singleton warmup on container boot."""
    print("=" * 60)
    print("Starting DocuSense Decoupled Backend Service...")
    print(f"  • Qdrant Cluster : {config.qdrant_url or 'In-Memory Fallback'}")
    print(f"  • Default LLM    : Groq ({config.groq_model})")
    print(f"  • Max Upload     : {config.max_upload_size_mb} MB")
    print(f"  • Demo Quota     : {config.rate_limit_demo_queries} queries/session")

    # Ensure collection exists in Qdrant on boot
    try:
        init_qdrant_collection()
    except Exception as e:
        print(f"  [Warning] Qdrant collection check encountered error: {e}")

    print("=" * 60)
    yield
    print("Shutting down DocuSense API service.")


app = FastAPI(
    title="DocuSense API",
    description="High-performance, decoupled RAG backend for DocuSense Document QA.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for cross-origin access from Streamlit Community Cloud
cors_origins = [orig.strip() for orig in config.cors_origins.split(",") if orig.strip()]
if not cors_origins or "*" in cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
# Root level: /health, /memory
app.include_router(health_router)
# Versioned API routes: /api/v1/...
app.include_router(health_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    # Single-worker mode strictly enforced to stay within 512 MB RAM
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
