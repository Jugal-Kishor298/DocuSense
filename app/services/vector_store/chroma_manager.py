import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config

# Global ChromaDB client singleton instance
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a singleton instance of the ChromaDB PersistentClient.
    Initializes disk storage at config.chroma_persist_dir.
    """
    global _chroma_client
    if _chroma_client is None:
        persist_dir = Path(config.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing ChromaDB Persistent Client at: {persist_dir.resolve()}")
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir.resolve()),
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def get_or_create_collection(collection_name: str = "docusense_docs"):
    """Fetches or creates a ChromaDB collection by name.
    Uses cosine distance metric for similarity matching.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity metric
    )


def add_chunks_to_vector_store(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str = "docusense_docs",
) -> int:
    """Inserts or updates document chunks and their dense vector embeddings in ChromaDB.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries containing text, chunk_id, metadata.
        embeddings (List[List[float]]): Corresponding 384D float vector embeddings.
        collection_name (str): Target collection name.

    Returns:
        int: Number of chunks successfully stored/upserted.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

    collection = get_or_create_collection(collection_name)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "page_number": c["page_number"],
            "filename": c["filename"],
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "is_table": c.get("is_table", False),
        }
        for c in chunks
    ]

    # Upsert into ChromaDB (inserts new records or updates existing IDs)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Successfully upserted {len(chunks)} chunks into ChromaDB collection '{collection_name}'.")
    return len(chunks)


def reset_collection(collection_name: str = "docusense_docs") -> bool:
    """Deletes and recreates a collection to clear all indexed documents."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted ChromaDB collection '{collection_name}'.")
    except Exception:
        pass
    get_or_create_collection(collection_name)
    return True


def get_collection_stats(collection_name: str = "docusense_docs") -> Dict[str, Any]:
    """Returns statistics about stored vectors in the specified collection."""
    collection = get_or_create_collection(collection_name)
    count = collection.count()
    return {
        "collection_name": collection_name,
        "total_vector_count": count,
        "persist_directory": config.chroma_persist_dir,
    }


if __name__ == "__main__":
    from app.services.ingestion import load_pdf
    from app.services.chunking import chunk_document
    from app.services.embeddings import generate_embeddings

    sample_pdf = ROOT_DIR / "data" / "raw" / "sample_multi_page.pdf"

    if sample_pdf.exists():
        print("=" * 60)
        print(f"Testing Vector Store Service on: {sample_pdf.name}")
        pages = load_pdf(str(sample_pdf))
        chunks = chunk_document(pages)

        chunk_texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(chunk_texts)

        # Insert into ChromaDB test collection
        added_count = add_chunks_to_vector_store(chunks, embeddings, collection_name="test_collection")
        stats = get_collection_stats("test_collection")

        print("\n--- ChromaDB Collection Stats ---")
        print(f"  • Collection Name : {stats['collection_name']}")
        print(f"  • Total Vectors   : {stats['total_vector_count']}")
        print(f"  • Storage Path    : {stats['persist_directory']}")
    else:
        print("Sample PDF not found. Checking collection stats...")
        stats = get_collection_stats("docusense_docs")
        print(f"Total stored vectors: {stats['total_vector_count']}")
