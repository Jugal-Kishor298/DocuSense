"""
app/api/health.py
-----------------
Health check and memory diagnostics router for DocuSense backend.
Reports runtime status, Qdrant connectivity, and live RSS memory metrics.
"""

import os
import sys
import psutil
from fastapi import APIRouter

from app.config import config
from app.schemas import HealthResponse, MemoryStatsResponse

router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint used by Render and Streamlit to verify service readiness."""
    return HealthResponse(
        status="ok",
        service="docusense-api",
        version="2.0.0",
        vector_store="qdrant" if config.qdrant_url else "qdrant-memory",
        ingestion_enabled=True,
        ocr_enabled=False,
    )


@router.get("/memory", response_model=MemoryStatsResponse)
async def memory_diagnostics():
    """Live process memory metrics to monitor Render free tier (512 MB) consumption."""
    process = psutil.Process(os.getpid())
    rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    sys_mem = psutil.virtual_memory()

    return MemoryStatsResponse(
        process_rss_mb=rss_mb,
        system_total_mb=round(sys_mem.total / (1024 * 1024), 2),
        system_available_mb=round(sys_mem.available / (1024 * 1024), 2),
        system_percent=sys_mem.percent,
        torch_loaded="torch" in sys.modules,
    )
