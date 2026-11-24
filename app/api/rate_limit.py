"""
app/api/rate_limit.py
---------------------
In-memory rate limiter and anti-abuse quota manager for DocuSense free-tier deployment.
Features:
  • Zero external Redis dependency (optimized for 512MB RAM single-instance containers)
  • Thread-safe in-memory session quota tracking
  • User-override bypass: Requests supplying custom X-Groq-API-Key bypass rate limits completely
  • IP detection via X-Forwarded-For for logging and diagnostics
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from fastapi import Request

from app.config import config

logger = logging.getLogger("docusense.ratelimit")

# Thread-safe in-memory storage for session usage
_lock = threading.Lock()
_session_usage: Dict[str, Dict[str, Any]] = {}


def get_client_ip(request: Request) -> str:
    """Extracts client IP from X-Forwarded-For header or direct socket."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"


def register_session_quota(session_id: str, max_queries: Optional[int] = None) -> int:
    """Initializes a new session quota bucket.

    Args:
        session_id (str): User session identifier.
        max_queries (Optional[int]): Total queries permitted. Defaults to config.rate_limit_demo_queries.

    Returns:
        int: Remaining query count.
    """
    limit = max_queries if max_queries is not None else config.rate_limit_demo_queries
    with _lock:
        if session_id not in _session_usage:
            _session_usage[session_id] = {
                "used": 0,
                "max": limit,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        return _session_usage[session_id]["max"] - _session_usage[session_id]["used"]


def check_and_consume_quota(session_id: str, max_queries: Optional[int] = None) -> Tuple[bool, int]:
    """Evaluates whether the session has remaining query budget and consumes 1 credit if permitted.

    Args:
        session_id (str): User session UUID.
        max_queries (Optional[int]): Maximum query cap.

    Returns:
        Tuple[bool, int]: (allowed, remaining_queries)
    """
    limit = max_queries if max_queries is not None else config.rate_limit_demo_queries
    with _lock:
        if session_id not in _session_usage:
            _session_usage[session_id] = {
                "used": 0,
                "max": limit,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        record = _session_usage[session_id]
        if record["used"] >= record["max"]:
            return False, 0

        record["used"] += 1
        remaining = record["max"] - record["used"]
        return True, remaining


def get_remaining_queries(session_id: str, max_queries: Optional[int] = None) -> int:
    """Returns remaining query balance without consuming a credit."""
    limit = max_queries if max_queries is not None else config.rate_limit_demo_queries
    with _lock:
        if session_id not in _session_usage:
            return limit
        record = _session_usage[session_id]
        return max(0, record["max"] - record["used"])


def reset_session_quota(session_id: str) -> None:
    """Clears quota tracking for a terminated session."""
    with _lock:
        _session_usage.pop(session_id, None)
