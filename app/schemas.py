"""
app/schemas.py
--------------
Pydantic v2 data models for DocuSense decoupled FastAPI backend.
Defines strict request/response schemas for health, sessions, uploads, and queries.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Health & Diagnostics Schemas
# ==============================================================================
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "docusense-api"
    version: str = "2.0.0"
    vector_store: str = "qdrant"
    ingestion_enabled: bool = True
    ocr_enabled: bool = False


class MemoryStatsResponse(BaseModel):
    process_rss_mb: float
    system_total_mb: float
    system_available_mb: float
    system_percent: float
    torch_loaded: bool


# ==============================================================================
# Session Management Schemas
# ==============================================================================
class SessionResponse(BaseModel):
    session_id: str
    remaining_queries: int
    max_queries: int
    created_at: str


class SessionDeleteResponse(BaseModel):
    session_id: str
    status: str
    message: str


# ==============================================================================
# Ingestion & Upload Schemas
# ==============================================================================
class UploadResponse(BaseModel):
    filename: str
    extracted_pages: int
    indexed_chunks: int
    session_id: str
    status: str = "indexed"
    message: str = "Document successfully ingested and indexed into Qdrant."


# ==============================================================================
# Query & Retrieval Schemas
# ==============================================================================
class Citation(BaseModel):
    source_id: int
    filename: str
    page_number: int
    similarity_score: float
    distance: float
    chunk_id: str
    text: str
    is_table: bool = False


class TimingsBreakdown(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language user question.")
    session_id: Optional[str] = Field(None, description="Client session UUID for isolated retrieval.")
    top_k: int = Field(7, ge=1, le=15, description="Final number of top reranked chunks.")
    enable_reranking: bool = Field(True, description="Whether to apply FlashRank cross-encoder reranking.")
    model: Optional[str] = Field(None, description="Optional override for Groq LLM model ID.")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    timings: TimingsBreakdown
    model: str
    session_id: Optional[str] = None
    remaining_queries: Optional[int] = None
