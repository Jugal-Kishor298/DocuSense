from app.services.vector_store.chroma_manager import (
    add_chunks_to_vector_store,
    get_chroma_client,
    get_collection_stats,
    get_or_create_collection,
    reset_collection,
)
from app.services.vector_store.qdrant_manager import (
    delete_session_chunks,
    get_qdrant_client,
    get_qdrant_stats,
    init_qdrant_collection,
    search_qdrant,
    upsert_chunks_to_qdrant,
)

__all__ = [
    # ChromaDB (Local Legacy)
    "add_chunks_to_vector_store",
    "get_chroma_client",
    "get_collection_stats",
    "get_or_create_collection",
    "reset_collection",
    # Qdrant (Cloud & Multi-Tenant)
    "delete_session_chunks",
    "get_qdrant_client",
    "get_qdrant_stats",
    "init_qdrant_collection",
    "search_qdrant",
    "upsert_chunks_to_qdrant",
]
