import gc
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config

# Global Qdrant client singleton instance
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Returns a singleton instance of the QdrantClient.
    Connects to Qdrant Cloud cluster if QDRANT_URL is configured,
    or falls back to an in-memory instance for testing and local development.
    """
    global _qdrant_client
    if _qdrant_client is None:
        url = config.qdrant_url.strip()
        api_key = config.qdrant_api_key.strip()

        if url:
            print(f"Connecting to Qdrant Cloud at: {url}...")
            _qdrant_client = QdrantClient(
                url=url,
                api_key=api_key or None,
                timeout=30.0,
            )
            print("  • Qdrant Engine: Cloud Managed Cluster Connected")
        else:
            print("QDRANT_URL not configured. Initializing in-memory fallback Qdrant client...")
            _qdrant_client = QdrantClient(":memory:")
            print("  • Qdrant Engine: Local In-Memory Fast Fallback")

    return _qdrant_client


def init_qdrant_collection(collection_name: Optional[str] = None) -> str:
    """Ensures the target Qdrant collection exists with cosine metric and indexed session payload.

    Args:
        collection_name (Optional[str]): Target collection name. Defaults to config.qdrant_collection.

    Returns:
        str: Active collection name.
    """
    target_name = collection_name or config.qdrant_collection
    client = get_qdrant_client()

    collections_response = client.get_collections()
    existing = [col.name for col in collections_response.collections]

    if target_name not in existing:
        print(f"Creating Qdrant collection '{target_name}' with {config.vector_dimension}D Cosine index...")
        client.create_collection(
            collection_name=target_name,
            vectors_config=VectorParams(
                size=config.vector_dimension,
                distance=Distance.COSINE,
            ),
        )
        # Create payload keyword index on session_id for fast filtered multi-tenant lookups
        client.create_payload_index(
            collection_name=target_name,
            field_name="session_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"  • Collection '{target_name}' created with indexed 'session_id' payload.")

    return target_name


def upsert_chunks_to_qdrant(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    session_id: str,
    collection_name: Optional[str] = None,
    batch_size: int = 16,
) -> int:
    """Inserts or updates document chunks and dense embeddings into Qdrant in micro-batches.
    Enforces session isolation by tagging each chunk's payload with `session_id`.
    Applies explicit memory garbage collection after batch processing to stay within free-tier limits.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries containing text, chunk_id, metadata.
        embeddings (List[List[float]]): Corresponding dense float embeddings (e.g. 384D).
        session_id (str): Unique user/browser session UUID for multi-tenant isolation.
        collection_name (Optional[str]): Target collection. Defaults to config.qdrant_collection.
        batch_size (int): Size of micro-batch upsert to prevent memory spikes. Defaults to 16.

    Returns:
        int: Number of points successfully upserted.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

    target_name = init_qdrant_collection(collection_name)
    client = get_qdrant_client()

    total_chunks = len(chunks)
    upserted_count = 0

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        points = []
        for c, emb in zip(batch_chunks, batch_embeddings):
            # Generate deterministic UUID for idempotency per session and chunk
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}_{c['chunk_id']}"))

            payload = {
                "session_id": session_id,
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "page_number": c.get("page_number", 1),
                "filename": c.get("filename", "unknown.pdf"),
                "doc_id": c.get("doc_id", "doc"),
                "chunk_index": c.get("chunk_index", 0),
                "is_table": c.get("is_table", False),
            }

            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        client.upsert(collection_name=target_name, points=points)
        upserted_count += len(points)

        # Free temporary batch references to prevent memory bloat
        del points, batch_chunks, batch_embeddings

    # Explicit garbage collection after full document upsert to reclaim memory
    gc.collect()
    print(f"Successfully upserted {upserted_count} chunks to Qdrant (Session: {session_id}).")
    return upserted_count


def search_qdrant(
    query_vector: List[float],
    session_id: Optional[str] = None,
    top_k: int = 15,
    score_threshold: Optional[float] = None,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Performs cosine vector similarity search in Qdrant with optional session isolation.

    Args:
        query_vector (List[float]): Dense query vector (e.g. 384D).
        session_id (Optional[str]): If provided, limits search strictly to vectors with this session_id.
        top_k (int): Number of candidate points to retrieve.
        score_threshold (Optional[float]): Minimum cosine similarity score required (0.0 to 1.0).
        collection_name (Optional[str]): Target collection. Defaults to config.qdrant_collection.

    Returns:
        List[Dict[str, Any]]: Retrieved chunk dictionaries matching standard DocuSense schema.
    """
    target_name = init_qdrant_collection(collection_name)
    client = get_qdrant_client()

    query_filter = None
    if session_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        )

    response = client.query_points(
        collection_name=target_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for pt in response.points:
        payload = pt.payload or {}
        # Cosine similarity in Qdrant is in [-1.0, 1.0], where 1.0 is exact match
        sim_score = float(pt.score)
        # Cosine distance = 1.0 - similarity
        cos_distance = max(0.0, 1.0 - sim_score)

        results.append({
            "text": payload.get("text", ""),
            "chunk_id": payload.get("chunk_id", str(pt.id)),
            "page_number": payload.get("page_number", 1),
            "filename": payload.get("filename", "unknown.pdf"),
            "doc_id": payload.get("doc_id", "doc"),
            "chunk_index": payload.get("chunk_index", 0),
            "distance": round(cos_distance, 4),
            "similarity_score": round(sim_score, 4),
            "session_id": payload.get("session_id", ""),
        })

    return results


def delete_session_chunks(session_id: str, collection_name: Optional[str] = None) -> int:
    """Purges all stored vectors and payloads belonging to the specified session_id.
    Ensures complete data removal when a user resets their session or leaves.

    Args:
        session_id (str): User/browser session UUID.
        collection_name (Optional[str]): Target collection.

    Returns:
        int: Estimated number of points deleted (or remaining).
    """
    if not session_id:
        return 0

    target_name = init_qdrant_collection(collection_name)
    client = get_qdrant_client()

    query_filter = Filter(
        must=[
            FieldCondition(
                key="session_id",
                match=MatchValue(value=session_id),
            )
        ]
    )

    client.delete(
        collection_name=target_name,
        points_selector=models.FilterSelector(filter=query_filter),
    )
    print(f"Purged all vectors for session '{session_id}' from Qdrant collection '{target_name}'.")
    return 1


def get_qdrant_stats(collection_name: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Returns statistics about stored points in the Qdrant collection.

    Args:
        collection_name (Optional[str]): Target collection.
        session_id (Optional[str]): If provided, counts points belonging to this session only.

    Returns:
        Dict[str, Any]: Statistics dictionary including total and session-specific counts.
    """
    target_name = init_qdrant_collection(collection_name)
    client = get_qdrant_client()

    total_count = client.count(collection_name=target_name).count

    session_count = 0
    if session_id:
        session_count = client.count(
            collection_name=target_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id),
                    )
                ]
            ),
        ).count

    return {
        "collection_name": target_name,
        "total_vector_count": total_count,
        "session_vector_count": session_count,
        "backend": "Qdrant Cloud" if config.qdrant_url else "In-Memory Fallback",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qdrant Vector Store Service (Standalone)")
    print("=" * 60)

    # 1. Test collection initialization
    col = init_qdrant_collection("test_docusense")

    # 2. Mock chunks and embeddings
    test_session_a = "recruiter_session_aaa"
    test_session_b = "recruiter_session_bbb"

    dummy_chunks_a = [
        {"chunk_id": "c1", "text": "Deep Learning with Transformers and Self-Attention.", "page_number": 1, "filename": "attention.pdf"},
        {"chunk_id": "c2", "text": "Residual connections allow training of very deep neural nets.", "page_number": 2, "filename": "resnet.pdf"},
    ]
    dummy_chunks_b = [
        {"chunk_id": "c3", "text": "Quarterly earnings report for financial Q3 2026.", "page_number": 1, "filename": "finance.pdf"},
    ]

    # Dummy 384D normalized vectors
    import random
    random.seed(42)
    def make_dummy_vec():
        v = [random.uniform(-1, 1) for _ in range(384)]
        norm = sum(x**2 for x in v) ** 0.5
        return [x / norm for x in v]

    emb_a = [make_dummy_vec(), make_dummy_vec()]
    emb_b = [make_dummy_vec()]

    # 3. Test batched upsert with session isolation
    print("\n1. Testing Batched Upsert with Session Isolation:")
    upsert_chunks_to_qdrant(dummy_chunks_a, emb_a, session_id=test_session_a, collection_name=col, batch_size=1)
    upsert_chunks_to_qdrant(dummy_chunks_b, emb_b, session_id=test_session_b, collection_name=col, batch_size=1)

    # 4. Verify collection stats
    stats_a = get_qdrant_stats(col, session_id=test_session_a)
    print(f"\n2. Collection Stats for Session A: {stats_a}")
    assert stats_a["session_vector_count"] == 2, "Session A should have exactly 2 vectors"

    # 5. Search with Session A filter using query vector identical to c1
    print("\n3. Testing Isolated Search for Session A:")
    hits_a = search_qdrant(query_vector=emb_a[0], session_id=test_session_a, top_k=5, collection_name=col)
    for h in hits_a:
        print(f"  • [Match {h['similarity_score']:.4f}] {h['filename']} (P.{h['page_number']}): {h['text']}")
    assert len(hits_a) == 2, "Should return only Session A points"
    assert all(h["session_id"] == test_session_a for h in hits_a), "Cross-contamination detected!"

    # 6. Verify Session B cannot see Session A points
    print("\n4. Testing Cross-Tenant Isolation (Session B Querying):")
    hits_b = search_qdrant(query_vector=emb_a[0], session_id=test_session_b, top_k=5, collection_name=col)
    for h in hits_b:
        print(f"  • [Match {h['similarity_score']:.4f}] {h['filename']}: {h['text']}")
    assert all(h["session_id"] == test_session_b for h in hits_b), "Cross-tenant leak detected!"

    # 7. Test session cleanup
    print("\n5. Testing Session Cleanup:")
    delete_session_chunks(session_id=test_session_a, collection_name=col)
    stats_after = get_qdrant_stats(col, session_id=test_session_a)
    print(f"  • Session A vectors after purge: {stats_after['session_vector_count']}")
    assert stats_after["session_vector_count"] == 0, "Session A vectors were not cleaned up!"

    print("\n✅ All Qdrant standalone tests passed successfully!")
