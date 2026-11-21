import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flashrank import Ranker, RerankRequest

# Singleton instance for lazy loading FlashRank cross-encoder model
_reranker_instance: Optional[Ranker] = None


def get_reranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Ranker:
    """Lazy-loaded singleton provider for FlashRank cross-encoder ONNX model.
    Avoids expensive model re-initialization on every query.

    Args:
        model_name (str): FlashRank model identifier. Defaults to 'ms-marco-TinyBERT-L-2-v2'.

    Returns:
        Ranker: Loaded FlashRank Ranker instance.
    """
    global _reranker_instance
    if _reranker_instance is None:
        # Initialize FlashRank Ranker ONNX C++ engine (~40 MB RAM footprint)
        try:
            _reranker_instance = Ranker(model_name=model_name)
        except Exception:
            _reranker_instance = Ranker()
    return _reranker_instance


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    model_name: str = "ms-marco-TinyBERT-L-2-v2",
    weight_rrf: float = 0.0
) -> List[Dict[str, Any]]:
    """Reranks candidate retrieved chunks using joint query-chunk Cross-Encoder attention
    and optional weighted blending with RRF fusion scores.

    Args:
        query (str): Natural language user question.
        chunks (List[Dict[str, Any]]): Candidate chunks retrieved from vector/hybrid search.
        top_k (int): Number of top reranked chunks to return. Defaults to 5.
        model_name (str): Cross-encoder model identifier.
        weight_rrf (float): Weight factor (0.0 to 1.0) for blending RRF score with Cross-Encoder score. Defaults to 0.5.

    Returns:
        List[Dict[str, Any]]: Top-K reranked chunks with updated 'rerank_score' metadata.
    """
    if not chunks or not query.strip():
        return []

    ranker = get_reranker(model_name=model_name)

    # Format chunks into FlashRank passages list
    passages = []
    for idx, c in enumerate(chunks):
        passages.append({
            "id": idx,
            "text": c["text"],
            "meta": c  # Preserve original chunk metadata dictionary
        })

    # Execute FlashRank ONNX cross-encoder joint scoring
    rerank_req = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_req)

    has_rrf = any("rrf_score" in res.get("meta", {}) for res in reranked_results)

    if weight_rrf > 0.0 and has_rrf and len(reranked_results) > 1:
        ce_scores = [float(res.get("score", 0.0)) for res in reranked_results]
        min_ce, max_ce = min(ce_scores), max(ce_scores)

        rrf_scores = [float(res.get("meta", {}).get("rrf_score", 0.0)) for res in reranked_results]
        min_rrf, max_rrf = min(rrf_scores), max(rrf_scores)

        blended_chunks = []
        for res in reranked_results:
            orig_chunk = dict(res["meta"])
            raw_ce = float(res.get("score", 0.0))
            norm_ce = (raw_ce - min_ce) / (max_ce - min_ce) if max_ce > min_ce else 0.0

            raw_rrf = float(orig_chunk.get("rrf_score", 0.0))
            norm_rrf = (raw_rrf - min_rrf) / (max_rrf - min_rrf) if max_rrf > min_rrf else 0.0

            final_score = (weight_rrf * norm_rrf) + ((1.0 - weight_rrf) * norm_ce)
            orig_chunk["rerank_score"] = round(final_score, 4)
            orig_chunk["ce_score"] = round(raw_ce, 4)
            blended_chunks.append(orig_chunk)

        blended_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return blended_chunks[:top_k]

    # Default pure cross-encoder ranking
    reranked_chunks = []
    for res in reranked_results[:top_k]:
        orig_chunk = dict(res["meta"])
        score_val = float(res.get("score", 0.0))
        orig_chunk["rerank_score"] = round(score_val, 4)
        reranked_chunks.append(orig_chunk)

    return reranked_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FlashRank Cross-Encoder Reranker Service Standalone")
    print("=" * 60)

    test_query = "What is the formula for Scaled Dot-Product Attention?"
    sample_candidate_chunks = [
        {
            "text": "The Transformer uses multi-head attention to attend over information from different representation subspaces.",
            "chunk_id": "c1",
            "page_number": 3,
            "filename": "Attention.pdf"
        },
        {
            "text": "Scaled Dot-Product Attention computes Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V.",
            "chunk_id": "c2",
            "page_number": 4,
            "filename": "Attention.pdf"
        },
        {
            "text": "We trained on 8 NVIDIA P100 GPUs for 100,000 steps taking 12 hours.",
            "chunk_id": "c3",
            "page_number": 7,
            "filename": "Attention.pdf"
        }
    ]

    print(f"\nUser Query: '{test_query}'")
    print(f"Input Candidate Chunks: {len(sample_candidate_chunks)}")

    top_reranked = rerank_chunks(query=test_query, chunks=sample_candidate_chunks, top_k=2)

    print(f"\nReranked Results (Top-{len(top_reranked)}):")
    for r in top_reranked:
        print(f"  • Rerank Score: {r['rerank_score']} | Page {r['page_number']} | Snippet: {r['text'][:80]}...")
