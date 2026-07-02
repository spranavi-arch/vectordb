"""
FastAPI routes for vector search operations.

This module defines four main endpoints:
1. POST /vector/index - Index documents (convert to embeddings and store)
2. POST /vector/search - Search using semantic similarity (optionally
   hybrid dense+BM25, optionally cross-encoder reranked)
3. GET /vector/stats - Get statistics about indexed documents
4. POST /vector/ask - RAG: retrieve context and generate a cited answer
   via Google AI Studio's Gemini API

All endpoints use dependency injection to access the VectorService /
AnswerService.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from typing import Optional, List
import logging

from app.core.dependencies import get_vector_service, get_answer_service
from app.services.vector_service import VectorService
from app.services.answer_service import AnswerService
from app.schemas.vector_schemas import SearchRequest, AskRequest, AskResponse

logger = logging.getLogger(__name__)

# Create router with shared prefix and tags for API documentation
router = APIRouter(prefix="/vector", tags=["Vector"])


@router.post("/index")
async def index_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    tags: str = Form(...),
    document_name: str = Form(None),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    Index a document by converting it to text chunks and vector embeddings.
    
    The indexing process:
    1. Extract text from PDF or image using OCR
    2. Split text into overlapping chunks
    3. Convert each chunk to vector embedding
    4. Store embeddings with metadata (user_id, tags, source, etc.)
    
    Args:
        file: The document file (PDF or image) to index
        user_id: Owner/creator of the document
        tags: Comma-separated tags for categorization (e.g., "medical,urgent")
        document_name: Optional human-readable name for the document (defaults to filename)
        vector_service: Injected VectorService instance
        
    Returns:
        dict with:
            - document_id: Unique ID for this document
            - total_chunks: Number of chunks created
            
    Raises:
        HTTPException (400): If input validation fails
        HTTPException (500): If indexing process fails
    """
    try:
        # Read the uploaded file into memory
        file_bytes = await file.read()
        
        # Parse comma-separated tags and strip whitespace
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        # Use provided document_name or fall back to filename
        doc_name = document_name or file.filename or "Untitled Document"

        # Index the document (OCR → chunking → embedding → storage).
        # This is CPU/IO-bound synchronous work, so run it in a threadpool
        # to avoid blocking the event loop for other concurrent requests.
        document_id, total_chunks = await run_in_threadpool(
            vector_service.index_document,
            file_bytes=file_bytes,
            content_type=file.content_type,
            user_id=user_id,
            tags=tag_list,
            document_name=doc_name
        )

        return {
            "document_id": document_id,
            "total_chunks": total_chunks
        }

    except ValueError as e:
        # Client-side error (e.g., invalid file format)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Server-side error (e.g., OCR failed, database error)
        error_msg = f"Indexing failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/search")
def search(
    req: SearchRequest,
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    Perform semantic vector search on indexed documents.
    
    The search process:
    1. Convert query text to vector embedding
    2. Find most similar chunk embeddings (using cosine similarity)
    3. Apply filters (user_id, tags, etc.) if provided
    4. Return top_k most relevant results with similarity scores
    
    Args:
        req: SearchRequest containing query text, top_k, and optional filters
        vector_service: Injected VectorService instance
        
    Returns:
        dict with search results and metadata
        
    Raises:
        HTTPException (400): If query is empty or invalid
        HTTPException (404): If no embeddings found
        HTTPException (409): If dimension mismatch
        HTTPException (503): If index not initialized
        HTTPException (500): If search fails
    """
    try:
        # Validate query
        if not req.query or not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Search query cannot be empty or whitespace"
            )
        
        # Validate top_k
        if req.top_k <= 0:
            raise HTTPException(
                status_code=400,
                detail="top_k must be greater than 0"
            )
        
        if req.top_k > 1000:
            raise HTTPException(
                status_code=400,
                detail=f"top_k cannot exceed 1000 (requested: {req.top_k})"
            )
        
        # Validate filters if provided
        if req.filters:
            if hasattr(req.filters, 'top_k') and req.filters.top_k and req.filters.top_k <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Filter top_k must be greater than 0"
                )
        
        # Check if index is initialized (has embeddings)
        total_chunks = vector_service.repo.count()
        if total_chunks == 0:
            raise HTTPException(
                status_code=404,
                detail="No embeddings indexed yet. Please index documents first."
            )
        
        # Perform search - use the dense-only path unchanged unless hybrid
        # or reranking was explicitly requested, so default behavior is
        # untouched by these opt-in flags.
        if req.hybrid or req.rerank:
            results = vector_service.retrieve(
                query=req.query,
                filters=req.filters,
                top_k=req.top_k,
                hybrid=req.hybrid,
                rerank=req.rerank
            )
        else:
            results = vector_service.search(
                query=req.query,
                filters=req.filters,
                top_k=req.top_k
            )

        # Check if results are empty
        if not results.get("ids") or not results["ids"][0]:
            return {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "message": "No matching documents found for the query and filters"
            }
        
        return results
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        error_msg = f"Search failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.get("/stats")
def stats(
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    Get statistics about the vector database.

    Args:
        vector_service: Injected VectorService instance

    Returns:
        dict with statistics like total_chunks indexed
    """
    return {
        "total_chunks": vector_service.repo.count()
    }


@router.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    answer_service: AnswerService = Depends(get_answer_service)
):
    """
    Retrieval-augmented generation: retrieve context chunks (optionally
    hybrid + reranked) and ask Gemini to answer strictly from that context,
    with inline citations back to the source chunks.

    Args:
        req: AskRequest containing the question, top_k, filters, and
             hybrid/rerank retrieval flags
        answer_service: Injected AnswerService instance

    Returns:
        AskResponse with the generated answer and its citations

    Raises:
        HTTPException (400): If query is empty or invalid
        HTTPException (503): If GOOGLE_API_KEY is not configured
        HTTPException (500): If generation fails
    """
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty or whitespace"
            )

        if req.top_k <= 0:
            raise HTTPException(
                status_code=400,
                detail="top_k must be greater than 0"
            )

        # Retrieval + the Gemini API call are both blocking work, so run
        # them in a threadpool to avoid blocking the event loop (same
        # concern already addressed for /vector/index).
        result = await run_in_threadpool(
            answer_service.ask,
            query=req.query,
            filters=req.filters,
            top_k=req.top_k,
            hybrid=req.hybrid,
            rerank=req.rerank
        )

        return result

    except HTTPException:
        raise
    except RuntimeError as e:
        # Missing/misconfigured GOOGLE_API_KEY
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        error_msg = f"Answer generation failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
