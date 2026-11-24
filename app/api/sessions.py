"""
app/api/sessions.py
-------------------
FastAPI router for session lifecycle and multi-tenant document isolation.
Endpoints:
  • POST /sessions: Creates a unique session UUID and initializes demo query quota
  • DELETE /sessions/{session_id}: Purges all document chunks from Qdrant and resets rate limit
  • GET /sessions/{session_id}/stats: Returns vector count and remaining quota for that session
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from app.api.rate_limit import (
    get_remaining_queries,
    register_session_quota,
    reset_session_quota,
)
from app.config import config
from app.schemas import SessionDeleteResponse, SessionResponse
from app.services.vector_store import delete_session_chunks, get_qdrant_stats

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session():
    """Generates a new anonymous session UUID for isolated recruiter testing."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    max_queries = config.rate_limit_demo_queries
    remaining = register_session_quota(session_id, max_queries=max_queries)

    return SessionResponse(
        session_id=session_id,
        remaining_queries=remaining,
        max_queries=max_queries,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    """Purges all uploaded chunks for this session from Qdrant and resets session quota."""
    if not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty.",
        )

    # Purge vectors from Qdrant Cloud
    delete_session_chunks(session_id)
    # Reset local quota record
    reset_session_quota(session_id)

    return SessionDeleteResponse(
        session_id=session_id,
        status="deleted",
        message=f"All document vectors and quota for session '{session_id}' successfully purged.",
    )


@router.get("/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Returns vector store counts and query quota for the specified session."""
    stats = get_qdrant_stats(session_id=session_id)
    remaining = get_remaining_queries(session_id)

    return {
        "session_id": session_id,
        "indexed_vectors": stats["session_vector_count"],
        "total_cluster_vectors": stats["total_vector_count"],
        "remaining_queries": remaining,
        "max_queries": config.rate_limit_demo_queries,
    }
