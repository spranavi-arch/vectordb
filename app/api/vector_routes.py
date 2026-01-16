"""
FastAPI routes for vector search operations.

This module defines three main endpoints:
1. POST /vector/index - Index documents (convert to embeddings and store)
2. POST /vector/search - Search using semantic similarity
3. GET /vector/stats - Get statistics about indexed documents

All endpoints use dependency injection to access the VectorService.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List

from app.core.dependencies import get_vector_service
from app.services.vector_service import VectorService
from app.schemas.vector_schemas import SearchRequest

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

        # Index the document (OCR → chunking → embedding → storage)
        document_id, total_chunks = vector_service.index_document(
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
        print(f"ERROR: {error_msg}")  # Log to console for debugging
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
        
        # Perform search
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
        print(f"ERROR: {error_msg}")
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
