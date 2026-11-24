"""
app/services/embeddings/embedder.py
------------------------------------
Cloud-accelerated vector embedding service using Cloudflare Workers AI.

Why Cloudflare Workers AI instead of local FastEmbed ONNX?
  • Render free tier gives 0.1 vCPU — ONNX inference on 130 chunks takes ~20-30s.
  • Cloudflare runs the same BAAI/bge-small-en-v1.5 model on GPUs server-side.
  • Result: 130 chunks embedded in a single batched API call in ~1-3s.
  • Free tier: 10,000 Neurons/day (resets daily) — well within demo usage.
  • Zero RAM impact on backend: no model loaded locally (~100 MB freed).
  • VECTOR DIMENSION: 384D — identical to our existing Qdrant collection schema.

API: POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/baai/bge-small-en-v1.5
Auth: Bearer {CF_API_TOKEN}
Payload: {"text": ["chunk1", "chunk2", ...]}
Response: {"result": {"data": [[0.1, 0.2, ...], ...]}, "success": true}
"""

import sys
import httpx
from pathlib import Path
from typing import List

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config

# Cloudflare Workers AI endpoint template
_CF_ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}"
    "/ai/run/@cf/baai/bge-small-en-v1.5"
)

# Cloudflare API accepts up to 100 texts per request — we batch at 64 to stay safe
_CF_BATCH_SIZE = 64


def _build_headers() -> dict:
    """Constructs the Authorization header for Cloudflare Workers AI REST API."""
    return {
        "Authorization": f"Bearer {config.cf_api_token}",
        "Content-Type": "application/json",
    }


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Sends a single batched embedding request to Cloudflare Workers AI.

    Args:
        texts: Up to 64 text strings to embed in one API call.

    Returns:
        List of 384D float vectors, one per input text.

    Raises:
        RuntimeError: If CF API returns a non-200 status or success=false.
    """
    url = _CF_ENDPOINT.format(account_id=config.cf_account_id)
    payload = {"text": texts}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=_build_headers())

    if resp.status_code != 200:
        raise RuntimeError(
            f"Cloudflare Workers AI embedding failed [{resp.status_code}]: {resp.text}"
        )

    body = resp.json()
    if not body.get("success", False):
        errors = body.get("errors", [])
        raise RuntimeError(f"Cloudflare Workers AI returned success=false: {errors}")

    vectors = body["result"]["data"]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"CF API returned {len(vectors)} vectors for {len(texts)} texts — mismatch."
        )
    return vectors


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates 384D dense vector embeddings for a list of text strings.

    Batches requests at 64 texts per CF API call to stay within payload limits.
    All batches run sequentially to respect Render's 0.1 vCPU (httpx is I/O-bound,
    so this is effectively free CPU-wise — we're just waiting on CF's GPU).

    Args:
        texts: List of chunk text strings to embed.

    Returns:
        List of 384-dimensional float vectors (one per input text).
    """
    if not texts:
        return []

    all_vectors: List[List[float]] = []
    for i in range(0, len(texts), _CF_BATCH_SIZE):
        batch = texts[i : i + _CF_BATCH_SIZE]
        batch_vectors = _embed_batch(batch)
        all_vectors.extend(batch_vectors)

    return all_vectors


def generate_query_embedding(query: str) -> List[float]:
    """Generates a 384D dense vector embedding for a single user query string.

    Args:
        query: User natural language question.

    Returns:
        384-dimensional query vector as a float list.
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    return generate_embeddings([query])[0]


# Kept for compatibility with existing __init__.py exports — no-op on cloud
def get_embedding_model():
    """No-op compatibility shim. Cloud embeddings have no local model to load."""
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cloudflare Workers AI Embedding Service")
    print(f"  • Account ID : {config.cf_account_id[:8]}...")
    print(f"  • Model      : @cf/baai/bge-small-en-v1.5 (384D)")
    test_vec = generate_query_embedding("What is the attention mechanism in transformers?")
    print(f"  • Vector Dim : {len(test_vec)}D ✓")
    print(f"  • Sample     : {test_vec[:5]}...")
    print("=" * 60)
