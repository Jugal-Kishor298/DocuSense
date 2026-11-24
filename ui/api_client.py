"""
ui/api_client.py
----------------
HTTP and Server-Sent Events (SSE) client for communicating with the
decoupled DocuSense FastAPI backend.
Handles health probes, Render cold-start wake-up polling, session lifecycle,
multipart file uploads, and streaming token generation.
"""

import json
import time
from typing import Any, Dict, Generator, Optional, Tuple
import httpx


class DocuSenseAPIClient:
    """Synchronous HTTP client wrapping DocuSense FastAPI backend endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def check_health(self, timeout: float = 3.0) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Pings /health to verify backend readiness."""
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return True, resp.json()
                return False, None
        except Exception:
            return False, None

    def poll_for_wakeup(self, max_seconds: int = 65, interval: float = 2.5) -> Generator[Dict[str, Any], None, None]:
        """Polls /health until Render instance spins up from sleep.
        Yields progress dictionaries: {"elapsed": int, "online": bool, "attempt": int}.
        """
        start_time = time.time()
        attempt = 1

        while True:
            elapsed = int(time.time() - start_time)
            is_healthy, data = self.check_health(timeout=3.0)
            if is_healthy:
                yield {"elapsed": elapsed, "online": True, "attempt": attempt, "data": data}
                return

            if elapsed >= max_seconds:
                yield {"elapsed": elapsed, "online": False, "attempt": attempt, "error": "Timeout"}
                return

            yield {"elapsed": elapsed, "online": False, "attempt": attempt}
            time.sleep(interval)
            attempt += 1

    def create_session(self) -> Dict[str, Any]:
        """Provisions a new isolated session UUID on backend."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{self.base_url}/api/v1/sessions")
            if resp.status_code != 201:
                detail = resp.text
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(f"Session creation failed: {detail}")
            return resp.json()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Retrieves vector counts and remaining query quota for this session."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/sessions/{session_id}/stats")
            if resp.status_code != 200:
                detail = resp.text
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(f"Failed to fetch session stats: {detail}")
            return resp.json()

    def upload_document(
        self,
        session_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> Dict[str, Any]:
        """Uploads a PDF file to the backend for ingestion, chunking, and Qdrant upsert."""
        files = {"file": (filename, file_bytes, "application/pdf")}
        data = {"session_id": session_id}

        # Generous timeout for remote PDF parsing and embedding generation
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/upload",
                files=files,
                data=data,
            )
            if resp.status_code != 201:
                detail = resp.text
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(detail)
            return resp.json()

    def stream_query(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        enable_reranking: bool = True,
        custom_groq_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams Server-Sent Events (SSE) from POST /api/v1/query/stream.
        Yields event objects:
          • {'type': 'token', 'content': str} for live streaming text chunks
          • {'type': 'done', 'citations': list, 'timings': dict, 'remaining_queries': int} on completion
          • {'type': 'error', 'content': str} on failure
        """
        headers = {}
        if custom_groq_key:
            headers["X-Groq-API-Key"] = custom_groq_key.strip()

        payload = {
            "query": query,
            "session_id": session_id,
            "top_k": top_k,
            "enable_reranking": enable_reranking,
            "model": model or None,
        }

        with httpx.Client(timeout=90.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/v1/query/stream",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code == 429:
                    error_detail = "Quota exceeded."
                    try:
                        error_detail = response.read().decode("utf-8")
                        err_json = json.loads(error_detail)
                        error_detail = err_json.get("detail", error_detail)
                    except Exception:
                        pass
                    yield {"type": "error", "content": error_detail}
                    return

                if response.status_code != 200:
                    err_msg = f"Backend returned status {response.status_code}"
                    try:
                        err_json = json.loads(response.read().decode("utf-8"))
                        err_msg = err_json.get("detail", err_msg)
                    except Exception:
                        pass
                    yield {"type": "error", "content": err_msg}
                    return

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        raw_data = line[len("data: ") :].strip()
                        if raw_data:
                            try:
                                event = json.loads(raw_data)
                                yield event
                            except json.JSONDecodeError:
                                pass

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Purges session document vectors from Qdrant and resets quota."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{self.base_url}/api/v1/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()
