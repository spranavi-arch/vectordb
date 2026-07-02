"""
Main application entry point for the Vector Search Service.

This module initializes the FastAPI application, sets up all core services
(OCR, embeddings, chunking, vector storage), and registers API routes.

Key responsibilities:
- Initialize FastAPI app
- Set up logging configuration
- Create and configure core services
- Register API routes
- Provide health check endpoint
"""

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.dependencies import vector_service as vector_service_holder

from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ocr_service import OCRService
from app.services.rerank_service import RerankService
from app.services.answer_service import AnswerService
from app.repositories.chroma_repository import ChromaRepository
from app.services.vector_service import VectorService

from app.api.vector_routes import router

# Initialize logging for the application
setup_logging()

# Create FastAPI application instance
app = FastAPI(title=settings.APP_NAME)

# ============================================================================
# SERVICE INITIALIZATION
# ============================================================================
# These services are instantiated once at startup and used for all requests

# ChunkingService: Splits large documents into overlapping chunks for better
# semantic search coverage and improved embedding quality
# Uses token-based chunking (1 token ≈ 4 characters) with configurable limits
chunker = ChunkingService(settings.MAX_TOKENS, settings.OVERLAP_TOKENS)

# EmbeddingService: Converts text into vector embeddings using pre-trained
# sentence transformers model for semantic similarity searches
embedder = EmbeddingService(settings.EMBEDDING_MODEL)

# OCRService: Extracts text from PDF files and images using Tesseract OCR
ocr = OCRService()

# ChromaRepository: Manages vector storage and retrieval using Chroma DB
# with DuckDB+Parquet backend for persistence and efficient queries
chroma_repo = ChromaRepository(
    persist_dir=settings.CHROMA_PERSIST_DIR
)

# RerankService: Cross-encoder second-pass reranking of retrieval candidates
# (used when a search/ask request opts into rerank=True)
reranker = RerankService(settings.RERANK_MODEL)

# VectorService: Orchestrates all services together - handles document indexing,
# searching, filtering, and metadata management
vector_service_holder = VectorService(
    chunker=chunker,
    embedder=embedder,
    ocr=ocr,
    repo=chroma_repo,
    reranker=reranker,
    rerank_overfetch=settings.RERANK_OVERFETCH
)

# AnswerService: Retrieval-augmented generation over VectorService, using
# Google AI Studio's free-tier Gemini API (used by POST /vector/ask)
answer_service_holder = AnswerService(
    vector_service=vector_service_holder,
    api_key=settings.GOOGLE_API_KEY,
    model=settings.GEMINI_MODEL
)

# ============================================================================
# DEPENDENCY INJECTION SETUP
# ============================================================================
# Store the service instances in the dependencies module so they can be
# injected into route handlers via FastAPI's Depends() system
import app.core.dependencies as deps
deps.vector_service = vector_service_holder
deps.answer_service = answer_service_holder

# Include API routes
app.include_router(router)


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================
@app.get("/health")
def health():
    """
    Health check endpoint to verify the application is running and the
    vector database is initialized.
    
    Returns:
        dict: Status information including indexed chunks count
    """
    return {
        "status": "ok",
        "indexed_chunks": chroma_repo.count()
    }

