import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator

import ollama

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config
from app.services.retrieval import format_retrieved_context
from prompts.templates import build_grounded_prompt


def _stream_ollama(prompt: str, model_name: str) -> Generator[str, None, None]:
    """Yields tokens from a local Ollama model."""
    try:
        response_stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.0},
        )
        for chunk in response_stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except Exception as e:
        yield f"\n[Ollama error with model '{model_name}': {str(e)}]"


def _stream_groq(prompt: str, model_name: str, groq_api_key: str | None = None) -> Generator[str, None, None]:
    """Yields tokens from a Groq-hosted model via the Groq Python SDK."""
    try:
        from groq import Groq  # lazy import — groq is optional
        import groq as groq_lib
    except ImportError:
        yield "\n[groq package not installed. Run: pip install groq]"
        return

    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        yield "\n[No Groq API key found. Enter your key in the sidebar or set GROQ_API_KEY in .env]"
        return

    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.0,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except groq_lib.AuthenticationError:
        yield f"\n[Invalid Groq API key. Please check the key you entered in the sidebar.]"
    except groq_lib.NotFoundError:
        yield (
            f"\n[Model not found: '{model_name}'.\n"
            f"Check available model IDs at https://console.groq.com/docs/models]"
        )
    except groq_lib.RateLimitError:
        yield "\n[Groq rate limit reached. Please wait a moment and try again.]"
    except groq_lib.APIStatusError as e:
        # Surface just the status code + short message, never the raw JSON body
        yield f"\n[Groq API error {e.status_code}: {e.message}]"
    except Exception as e:
        yield f"\n[Unexpected Groq error: {type(e).__name__}: {str(e)[:120]}]"


def generate_answer_stream(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    provider: str = "ollama",
    groq_api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """Streams grounded response tokens from either a local Ollama or a Groq-hosted LLM.

    Args:
        query (str): Natural language user question.
        retrieved_chunks (List[Dict[str, Any]]): Retrieved & threshold-filtered chunks.
        model_name (str, optional): Target model identifier. Defaults to config.ollama_model.
        provider (str): Backend to use — "ollama" (local) or "groq" (cloud). Defaults to "ollama".
        groq_api_key (str, optional): Groq API key. Falls back to GROQ_API_KEY env var if None.

    Yields:
        str: Streaming text tokens.
    """
    target_model = model_name if model_name is not None else config.ollama_model

    formatted_context = format_retrieved_context(retrieved_chunks)
    prompt = build_grounded_prompt(query, formatted_context)

    if provider == "groq":
        yield from _stream_groq(prompt, target_model, groq_api_key=groq_api_key)
    else:
        yield from _stream_ollama(prompt, target_model)


def generate_answer_sync(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    provider: str = "ollama",
    groq_api_key: Optional[str] = None,
) -> str:
    """Synchronously generates a complete grounded answer string.

    Args:
        query (str): Natural language user question.
        retrieved_chunks (List[Dict[str, Any]]): Retrieved & threshold-filtered chunks.
        model_name (str, optional): Target model name.
        provider (str): "ollama" or "groq".
        groq_api_key (str, optional): Groq API key override.

    Returns:
        str: Complete generated answer string.
    """
    tokens = list(generate_answer_stream(query, retrieved_chunks, model_name=model_name, provider=provider, groq_api_key=groq_api_key))
    return "".join(tokens)


if __name__ == "__main__":
    from app.services.ingestion import load_pdf
    from app.services.chunking import chunk_document
    from app.services.embeddings import generate_embeddings
    from app.services.vector_store import add_chunks_to_vector_store
    from app.services.retrieval import retrieve_relevant_chunks

    ml_pdf = ROOT_DIR / "data" / "raw" / "Machine Learrning PPT.pdf"

    if not ml_pdf.exists():
        print(f"Error: {ml_pdf} not found. Please verify PDF file placement.")
        sys.exit(1)

    print("=" * 60)
    print(f"1. Ingesting Document: {ml_pdf.name}")
    pages = load_pdf(str(ml_pdf), dpi=300)
    print(f"   • Extracted {len(pages)} pages.")

    print("\n2. Chunking Document...")
    chunks = chunk_document(pages)
    print(f"   • Generated {len(chunks)} text chunks.")

    print("\n3. Generating Vector Embeddings & Indexing into ChromaDB...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    collection_name = "ml_ppt_test_collection"
    add_chunks_to_vector_store(chunks, embeddings, collection_name=collection_name)

    query_1 = "What is PCA?"
    print("\n" + "=" * 60)
    print(f"4. User Question: '{query_1}'")
    retrieved_1 = retrieve_relevant_chunks(query_1, top_k=3, threshold=0.5, collection_name=collection_name)
    print(f"   • Retrieved {len(retrieved_1)} relevant chunks.")

    print("\n5. Streaming Ollama Response:\n")
    print("-" * 60)
    for token in generate_answer_stream(query_1, retrieved_1, provider="ollama"):
        print(token, end="", flush=True)
    print("\n" + "-" * 60)
