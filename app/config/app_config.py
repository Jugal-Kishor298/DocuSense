import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file into os.environ
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Centralized strongly-typed configuration class for DocuSense RAG pipeline."""

    # Chunking Hyperparameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval Hyperparameters
    top_k: int = int(os.getenv("TOP_K", "7"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

    # Model & Storage Configurations
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/vector_store/chroma_db")

    # Qdrant Cloud Configurations
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "docusense_docs")
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))

    # LLM Cloud Configurations (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Free Tier Safeguards & Rate Limits
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    rate_limit_demo_queries: int = int(os.getenv("RATE_LIMIT_DEMO_QUERIES", "8"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")


# Instantiate global config singleton for convenient import across application
config = AppConfig()

if __name__ == "__main__":
    print("Loaded DocuSense Configuration:")
    print(f"  • Chunk Size: {config.chunk_size} chars")
    print(f"  • Chunk Overlap: {config.chunk_overlap} chars")
    print(f"  • Top K: {config.top_k}")
    print(f"  • Similarity Threshold: {config.similarity_threshold}")
    print(f"  • Embedding Model: {config.embed_model}")
    print(f"  • Ollama LLM Model: {config.ollama_model}")
    print(f"  • Chroma Store Dir: {config.chroma_persist_dir}")
