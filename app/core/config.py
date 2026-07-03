
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    
    # Application metadata
    APP_NAME: str = "Vector Search Service"

    # Vector database configuration
    CHROMA_PERSIST_DIR: str = "chroma_data"
    
    # Embedding model - This model converts text to 384-dimensional vectors
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Text chunking parameters for splitting documents into overlapping segments
    # Uses token-based chunking (1 token ≈ 4 characters)
    # MAX_TOKENS: Maximum tokens per chunk (recommended 256-512)
    # OVERLAP_TOKENS: Tokens to overlap between chunks (recommended 30-50)
    # 
    # Recommended configurations:
    # - Short documents (< 5 pages): MAX_TOKENS=256, OVERLAP_TOKENS=30
    # - Medium documents (5-50 pages): MAX_TOKENS=300, OVERLAP_TOKENS=50 (default)
    # - Long documents (> 50 pages): MAX_TOKENS=512, OVERLAP_TOKENS=80
    # - High precision search: MAX_TOKENS=200, OVERLAP_TOKENS=40
    MAX_TOKENS: int = 300
    OVERLAP_TOKENS: int = 50

    # Cross-encoder reranking (sentence-transformers CrossEncoder - no extra
    # dependency, ships with sentence-transformers)
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many candidates to over-fetch per requested top_k before reranking
    # narrows back down. Reranking a pool the same size as top_k can only
    # re-order it, not change membership - the overfetch is what lets
    # reranking surface better results that dense/hybrid ranked outside top_k.
    RERANK_OVERFETCH: int = 4

    # RAG generation via Google AI Studio (Gemini API) - has a free tier,
    # unlike most other hosted LLM APIs, so this is usable without billing.
    # Get a free key at https://aistudio.google.com/apikey
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    class Config:
        # Allow loading settings from .env file
        env_file = ".env"


# Create global settings instance
settings = Settings()

# Ensure the Chroma persistence directory exists
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
