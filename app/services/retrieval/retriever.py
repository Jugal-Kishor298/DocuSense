import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config
from app.services.embeddings import generate_query_embedding
from app.services.vector_store import get_or_create_collection
from app.services.retrieval.reranker import rerank_chunks


def retrieve_relevant_chunks(
    query: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    collection_name: str = "docusense_docs",
    enable_reranking: bool = True
) -> List[Dict[str, Any]]:
    """Retrieves top-K semantically relevant document chunks for a natural language user query.
    Enforces Two-Stage Retrieval:
      - Stage 1: Vector similarity search (candidate K1 = top_k * 3)
      - Stage 2: FlashRank Cross-Encoder reranking (down to final top_k)

    Args:
        query (str): Natural language user question.
        top_k (int, optional): Final number of top reranked chunks to return. Defaults to config.top_k.
        threshold (float, optional): Maximum allowed cosine distance for Stage 1 candidate filtering.
        collection_name (str): Target ChromaDB collection name.
        enable_reranking (bool): Whether to apply FlashRank cross-encoder reranking. Defaults to True.

    Returns:
        List[Dict[str, Any]]: List of retrieved chunk dictionaries sorted by cross-encoder relevance,
        including text, distance, rerank_score, page_number, filename, and citation metadata.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    final_k = top_k if top_k is not None else config.top_k
    max_distance = threshold if threshold is not None else config.similarity_threshold

    # Stage 1: Retrieve candidate pool (3x final_k for broad recall)
    candidate_k = max(final_k * 3, 15)

    # 1. Generate 384D query vector embedding using FastEmbed
    query_vector = generate_query_embedding(cleaned_query)

    # 2. Query ChromaDB collection HNSW index
    collection = get_or_create_collection(collection_name)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"]
    )

    candidate_chunks = []

    if results and results.get("documents") and results["documents"][0]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for doc_text, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
            if dist <= max_distance:
                candidate_chunks.append({
                    "text": doc_text,
                    "chunk_id": chunk_id,
                    "page_number": meta.get("page_number", 1),
                    "filename": meta.get("filename", "unknown.pdf"),
                    "doc_id": meta.get("doc_id", "doc"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": round(float(dist), 4),
                    "similarity_score": round(1.0 - float(dist), 4),
                })

    if not candidate_chunks:
        return []

    # Stage 2: Cross-Encoder Reranking (FlashRank ONNX)
    if enable_reranking and len(candidate_chunks) > 1:
        reranked = rerank_chunks(query=cleaned_query, chunks=candidate_chunks, top_k=final_k)
        return reranked
    else:
        candidate_chunks.sort(key=lambda x: x["distance"])
        return candidate_chunks[:final_k]


def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clean, citation-annotated context string
    suitable for insertion into the LLM system prompt.
    """
    if not retrieved_chunks:
        return "NO RELEVANT CONTEXT FOUND IN UPLOADED DOCUMENTS."

    formatted_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        source_label = f"[Source {idx}: {chunk['filename']}, Page {chunk['page_number']}]"
        block = f"{source_label}\n{chunk['text']}"
        formatted_blocks.append(block)

    return "\n\n---\n\n".join(formatted_blocks)


if __name__ == "__main__":
    from app.services.ingestion import load_pdf
    from app.services.chunking import chunk_document
    from app.services.embeddings import generate_embeddings
    from app.services.vector_store import add_chunks_to_vector_store

    sample_pdf = ROOT_DIR / "data" / "raw" / "Attention is all you need.pdf"

    if sample_pdf.exists():
        print("=" * 60)
        print(f"Testing Retrieval Service on: {sample_pdf.name}")
        pages = load_pdf(str(sample_pdf))
        chunks = chunk_document(pages)
        embeddings = generate_embeddings([c["text"] for c in chunks])

        # Seed ChromaDB collection with chunks
        add_chunks_to_vector_store(chunks, embeddings, collection_name="retriever_test_coll")

        # Perform semantic retrieval query
        test_query = "What is the RAG pipeline in DocuSense?"
        print(f"\nUser Query: '{test_query}'")
        retrieved = retrieve_relevant_chunks(test_query, top_k=2, threshold=0.6, collection_name="retriever_test_coll")

        print(f"  • Retrieved Chunks : {len(retrieved)}")
        for r in retrieved:
            print(f"  • Match: {r['filename']} (Page {r['page_number']}) | Dist: {r['distance']} | Score: {r['similarity_score']}")
            print(f"    Snippet: {r['text'][:100]}...")

        print("\n--- Formatted Grounded LLM Context ---")
        print(format_retrieved_context(retrieved))
