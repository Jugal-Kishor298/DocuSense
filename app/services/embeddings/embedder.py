import sys
from pathlib import Path
from typing import List, Any
from fastembed import TextEmbedding

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config

# Global lazy-loaded FastEmbed singleton instance
_model_instance: Any = None


def get_embedding_model() -> TextEmbedding:
    """Returns a singleton instance of the FastEmbed TextEmbedding model (ONNX Runtime C++).
    Lazy-loads the model into memory on first call to prevent repeated disk/RAM overhead.
    """
    global _model_instance
    if _model_instance is None:
        model_name = config.embed_model
        # Map short name 'all-MiniLM-L6-v2' to FastEmbed canonical identifier
        fe_model_name = (
            "sentence-transformers/all-MiniLM-L6-v2"
            if model_name == "all-MiniLM-L6-v2"
            else model_name
        )
        print(f"Loading FastEmbed (ONNX Runtime C++) model '{fe_model_name}' into memory...")
        _model_instance = TextEmbedding(model_name=fe_model_name)
        print("  • Engine: FastEmbed (ONNX Runtime C++) — High-Performance CPU Inference")
    return _model_instance


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates dense vector embeddings for a list of text strings using FastEmbed.

    Args:
        texts (List[str]): List of chunk text strings to embed.

    Returns:
        List[List[float]]: List of 384-dimensional vector embeddings as native Python float lists.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings_gen = model.embed(texts)
    return [e.tolist() for e in embeddings_gen]


def generate_query_embedding(query: str) -> List[float]:
    """Generates a dense vector embedding for a single user query string.

    Args:
        query (str): User natural language question.

    Returns:
        List[float]: 384-dimensional query vector embedding.
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    embeddings = generate_embeddings([query])
    return embeddings[0]


if __name__ == "__main__":
    from app.services.ingestion import load_pdf
    from app.services.chunking import chunk_document

    sample_pdf = ROOT_DIR / "data" / "raw" / "sample_multi_page.pdf"

    if sample_pdf.exists():
        print("=" * 60)
        print(f"Testing Embedding Service on: {sample_pdf.name}")
        pages = load_pdf(str(sample_pdf))
        chunks = chunk_document(pages)

        chunk_texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(chunk_texts)

        print(f"  • Extracted Chunks  : {len(chunks)}")
        print(f"  • Vectors Generated : {len(embeddings)}")
        if embeddings:
            vector_dim = len(embeddings[0])
            print(f"  • Vector Dimensions : {vector_dim}D")
            print(f"  • Vector 1 Sample   : {embeddings[0][:5]}...")
    else:
        print("Sample PDF not found. Generating dummy test embedding...")
        test_vec = generate_query_embedding("What is DocuSense?")
        print(f"  • Vector Dimensions : {len(test_vec)}D")
