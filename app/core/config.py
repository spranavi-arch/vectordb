
from pydantic import BaseSettings
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

    class Config:
        # Allow loading settings from .env file
        env_file = ".env"


# Create global settings instance
settings = Settings()

# Ensure the Chroma persistence directory exists
Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True)
